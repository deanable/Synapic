"""
Hugging Face Hub Integration Utilities
======================================

This module provides utilities for discovering, downloading, and loading AI models
from the Hugging Face Model Hub. It serves as the bridge between Synapic and the
Hugging Face ecosystem, handling both local inference and remote API calls.

Key Features:
- Model Discovery: Search for models by task type with filtering
- Model Management: Download, cache, and track locally available models
- Progress Tracking: Custom tqdm integration for download/load progress
- Device Detection: Auto-detect CUDA/CPU capabilities for optimal performance
- Pipeline Loading: Unified interface for loading transformers pipelines
- API Inference: Support for serverless Hugging Face Inference API

Main Components:
- LiveByteProgressTracker: Aggregates streamed download bytes to the UI
- Model Discovery: find_models_by_task(), find_local_models()
- Model Downloads: download_model_worker(), with accurate progress reporting
- Model Loading: load_model(), load_model_with_progress()
- Size Utilities: get_remote_model_size(), format_size()
- Device Info: get_device_info() for hardware capabilities

Common Tasks:
    # Find and download a model
    >>> models, downloaded = find_models_by_task('image-classification')
    >>> download_model_worker('google/vit-base', progress_queue)

    # Load a model for inference
    >>> pipe = load_model('google/vit-base', 'image-classification', device=0)
    >>> results = pipe(image)

Architecture:
- All I/O operations run in worker threads to keep UI responsive
- Progress is communicated via Queue messages (type, data)
- Model caching uses Hugging Face's standard cache (~/.cache/huggingface)
- Supports both local inference and remote API calls

Author: Synapic Project
"""

import logging
import time
import os
import shutil
import base64
from contextlib import nullcontext
from pathlib import Path
from functools import partial
from tqdm import tqdm
from huggingface_hub import (
    list_models,
    hf_hub_download,
    snapshot_download,
    HfApi,
    InferenceClient,
)
from huggingface_hub import file_download as hf_file_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
import requests
from requests.exceptions import HTTPError
from transformers import (
    pipeline,
    AutoConfig,
    AutoTokenizer,
    AutoProcessor,
    AutoImageProcessor,
)
from threading import RLock
from src.core import config
import json
import sys
import platform
import torch
from typing import Optional, Dict, Any, List, Tuple

# Windows compatibility: Disable symlinks to avoid permission errors
# On Windows without Developer Mode, symlink creation fails with WinError 1314
_USE_SYMLINKS = "auto" if os.name != "nt" else False
_LOCAL_INFERENCE_COMPAT_CACHE: Dict[Tuple[str, str], Optional[str]] = {}


def get_device_info() -> Dict[str, Any]:
    """
    Get detailed information about available compute devices (CPU, CUDA, MPS).

    This function probes the system using PyTorch to determine the best available
    hardware accelerator. It returns a dictionary containing a list of devices,
     the recommended default, and detailed debugging info.

    Returns:
        A dictionary with "devices" (list), "default" (string), and "debug_info" (dict).
    """
    info = {
        "devices": ["CPU"],
        "default": "CPU",
        "debug_info": {
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "cuda_available": torch.cuda.is_available(),
            "mps_available": hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available(),
        },
    }

    # Check CUDA
    if torch.cuda.is_available():
        info["devices"].append("CUDA")
        info["default"] = "CUDA"
        info["debug_info"]["cuda_version"] = torch.version.cuda
        info["debug_info"]["cuda_device_count"] = torch.cuda.device_count()
        info["debug_info"]["cuda_current_device"] = torch.cuda.current_device()
        info["debug_info"]["cuda_device_name"] = torch.cuda.get_device_name(0)
    else:
        info["debug_info"]["cuda_check_error"] = "False (available=False)"

    # Check MPS (Mac)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["devices"].append("MPS")
        if info["default"] == "CPU":  # Prefer MPS over CPU if no CUDA
            info["default"] = "MPS"

    return info


def is_model_compatible(model_id: str) -> bool:
    """
    Check if a model is compatible with standard transformers pipelines.

    This function filters out models that require special quantization libraries
    (auto-gptq, autoawq, llama-cpp-python, exllamav2) which are not bundled
    with Synapic. These models cannot be loaded with the standard pipeline() call.

    Args:
        model_id: The Hugging Face model repository ID (e.g., 'google/vit-base')

    Returns:
        True if the model uses standard formats, False if it requires special libraries
    """
    model_id_lower = model_id.lower()

    for pattern in config.INCOMPATIBLE_MODEL_PATTERNS:
        if pattern.lower() in model_id_lower:
            logging.debug(
                f"Model {model_id} is incompatible (matched pattern: {pattern})"
            )
            return False

    return True


def get_incompatibility_reason(model_id: str) -> Optional[str]:
    """
    Get a human-readable reason why a model is incompatible.

    Args:
        model_id: The Hugging Face model repository ID

    Returns:
        A string explaining why the model is incompatible, or None if compatible
    """
    model_id_lower = model_id.lower()

    reasons = {
        "-gptq": "GPTQ quantized (requires auto-gptq library)",
        "int4": "GPTQ/Int4 quantized (requires auto-gptq library)",
        "int8": "Int8 quantized (requires bitsandbytes library)",
        "-awq": "AWQ quantized (requires autoawq library)",
        "-gguf": "GGUF format (requires llama-cpp-python)",
        "-ggml": "GGML format (requires llama-cpp-python)",
        "-exl2": "EXL2 format (requires exllamav2)",
        "-bnb": "BitsAndBytes quantized",
        "-4bit": "4-bit quantized (requires special library)",
        "-8bit": "8-bit quantized (requires special library)",
    }

    for pattern, reason in reasons.items():
        if pattern.lower() in model_id_lower:
            return reason

    return None


def _get_metadata_incompatibility_reason(
    model_id: str, token: Optional[str] = None
) -> Optional[str]:
    """Check repo metadata for formats/backends we do not support locally."""
    try:
        info = HfApi(token=token).model_info(repo_id=model_id)
    except Exception:
        return None

    library_name = (getattr(info, "library_name", None) or "").lower()
    if library_name and library_name != "transformers":
        return f"Uses '{library_name}' backend instead of the standard transformers runtime"

    tag_reasons = {
        "mlx": "MLX checkpoint/runtime (not loadable by the PyTorch transformers pipeline)",
        "gguf": "GGUF format (requires llama.cpp-compatible runtime)",
        "ggml": "GGML format (requires llama.cpp-compatible runtime)",
        "gptq": "GPTQ quantized model (requires auto-gptq)",
        "awq": "AWQ quantized model (requires autoawq)",
        "bitsandbytes": "BitsAndBytes quantized model",
        "bnb": "BitsAndBytes quantized model",
        "onnx": "ONNX export, not a standard transformers checkpoint",
        "openvino": "OpenVINO export, not a standard transformers checkpoint",
        "ncnn": "NCNN export, not a standard transformers checkpoint",
    }

    for tag in getattr(info, "tags", None) or []:
        lower_tag = tag.lower()
        for marker, reason in tag_reasons.items():
            if marker in lower_tag:
                return reason

    return None


def get_local_inference_incompatibility_reason(
    model_id: str,
    task: Optional[str] = None,
    token: Optional[str] = None,
) -> Optional[str]:
    """
    Return a reason why a model is unsuitable for Synapic local inference.

    This is stricter than name-pattern filtering: it checks the model backend,
    config support in the installed transformers version, and whether we can
    load the processor/tokenizer stack needed for the selected vision task.
    """
    cache_key = (model_id, task or "")
    if token is None and cache_key in _LOCAL_INFERENCE_COMPAT_CACHE:
        return _LOCAL_INFERENCE_COMPAT_CACHE[cache_key]

    reason = get_incompatibility_reason(model_id)
    if reason is not None:
        if token is None:
            _LOCAL_INFERENCE_COMPAT_CACHE[cache_key] = reason
        return reason

    reason = _get_metadata_incompatibility_reason(model_id, token=token)
    if reason is not None:
        if token is None:
            _LOCAL_INFERENCE_COMPAT_CACHE[cache_key] = reason
        return reason

    source = _get_latest_snapshot_path(model_id) or model_id

    try:
        AutoConfig.from_pretrained(source, trust_remote_code=False, token=token)
    except Exception as e:
        reason = (
            f"Unsupported model config for installed Transformers ({type(e).__name__})"
        )
        if token is None:
            _LOCAL_INFERENCE_COMPAT_CACHE[cache_key] = reason
        return reason

    processor_error = None
    try:
        AutoProcessor.from_pretrained(source, trust_remote_code=False, token=token)
        processor_ok = True
    except Exception as e:
        processor_error = e
        processor_ok = False

    image_processor_ok = False
    try:
        AutoImageProcessor.from_pretrained(source, trust_remote_code=False, token=token)
        image_processor_ok = True
    except Exception:
        pass

    if task in (config.MODEL_TASK_IMAGE_CLASSIFICATION, config.MODEL_TASK_ZERO_SHOT):
        if not processor_ok and not image_processor_ok:
            reason = (
                f"Missing a compatible image processor in the installed Transformers version "
                f"({type(processor_error).__name__ if processor_error else 'unknown error'})"
            )
            if token is None:
                _LOCAL_INFERENCE_COMPAT_CACHE[cache_key] = reason
            return reason
    elif not processor_ok:
        try:
            AutoTokenizer.from_pretrained(source, trust_remote_code=False, token=token)
        except Exception as tokenizer_error:
            reason = (
                "Missing a compatible processor/tokenizer for local inference "
                f"({type(tokenizer_error).__name__})"
            )
            if token is None:
                _LOCAL_INFERENCE_COMPAT_CACHE[cache_key] = reason
            return reason

    if token is None:
        _LOCAL_INFERENCE_COMPAT_CACHE[cache_key] = None
    return None


def is_model_suitable_for_local_inference(
    model_id: str,
    task: Optional[str] = None,
    token: Optional[str] = None,
) -> bool:
    """Return True when the model passes the local inference suitability probe."""
    return (
        get_local_inference_incompatibility_reason(model_id, task=task, token=token)
        is None
    )


def get_cache_dir():
    """Returns the Hugging Face cache directory."""
    return HUGGINGFACE_HUB_CACHE


def clear_cache():
    """Clears the Hugging Face Hub cache directory."""
    cache_path = Path(HUGGINGFACE_HUB_CACHE)
    if cache_path.exists():
        shutil.rmtree(cache_path)
        logging.info("Hugging Face cache cleared.")


def get_model_cache_dir(model_id):
    """Returns the cache directory for a given model."""
    return os.path.join(HUGGINGFACE_HUB_CACHE, f"models--{model_id.replace('/', '--')}")


class SilentTqdm(tqdm):
    """Suppress terminal output for the outer snapshot_download progress bar."""

    def __init__(self, *args, **kwargs):
        self.fp = open(os.devnull, "w") if os.name != "nt" else open("NUL", "w")
        kwargs["file"] = self.fp
        super().__init__(*args, **kwargs)

    def close(self):
        super().close()
        if hasattr(self, "fp") and self.fp and not self.fp.closed:
            try:
                self.fp.close()
            except Exception:
                pass


class _LiveByteProgressBar:
    """Minimal tqdm-like progress object used by huggingface_hub internals."""

    def __init__(self, tracker: "LiveByteProgressTracker", bar_id: int):
        self._tracker = tracker
        self._bar_id = bar_id

    def update(self, n=1):
        self._tracker.advance(self._bar_id, n)

    def close(self):
        return None


class LiveByteProgressTracker:
    """Aggregate byte-stream updates across concurrent Hugging Face file downloads."""

    def __init__(self, q, total_bytes: int = 0):
        self.q = q
        self.total_bytes = max(int(total_bytes), 0)
        self._dynamic_total = self.total_bytes == 0
        self.lock = RLock()
        self.downloaded_bytes = 0
        self._bar_progress: Dict[int, int] = {}
        self._next_bar_id = 0
        self._last_queued_bytes = 0

    def register_bar(
        self, total: Optional[int] = None, initial: int = 0
    ) -> _LiveByteProgressBar:
        with self.lock:
            bar_id = self._next_bar_id
            self._next_bar_id += 1
            self._bar_progress[bar_id] = 0

            if total is not None and self._dynamic_total:
                remaining = max(int(total) - int(initial), 0)
                if remaining > 0:
                    self.total_bytes += remaining

            self._queue_progress(force=True)
            return _LiveByteProgressBar(self, bar_id)

    def advance(self, bar_id: int, delta: float) -> None:
        increment = max(int(delta), 0)
        if increment == 0:
            return

        with self.lock:
            self._bar_progress[bar_id] = self._bar_progress.get(bar_id, 0) + increment
            self.downloaded_bytes += increment
            self._queue_progress()

    def complete(self) -> None:
        with self.lock:
            if self.total_bytes > 0 and self.downloaded_bytes < self.total_bytes:
                self.downloaded_bytes = self.total_bytes
            self._queue_progress(force=True)

    def _queue_progress(self, force: bool = False) -> None:
        total = self.total_bytes
        downloaded = self.downloaded_bytes
        if total <= 0:
            return

        threshold = max(int(total * 0.002), 256 * 1024)
        if (
            force
            or downloaded - self._last_queued_bytes >= threshold
            or downloaded >= total
        ):
            self.q.put(("model_download_progress", (downloaded, total)))
            self._last_queued_bytes = downloaded


def _get_latest_snapshot_path(model_id: str) -> Optional[str]:
    """Return the most recent cached snapshot path for a model, if available."""
    model_cache_dir = get_model_cache_dir(model_id)
    snapshot_dir = os.path.join(model_cache_dir, "snapshots")
    if not os.path.exists(snapshot_dir):
        return None

    snapshots = os.listdir(snapshot_dir)
    if not snapshots:
        return None

    latest_snapshot = sorted(snapshots)[-1]
    return os.path.join(snapshot_dir, latest_snapshot)


def _get_missing_repo_files(
    model_id: str, token=None
) -> Tuple[Any, List[Tuple[str, int]]]:
    """
    Inspect repo metadata and return the files that still need downloading.

    Each tuple is `(relative_filename, effective_size_bytes)`. When the Hub does
    not provide a file size, we use a small fallback so progress can still move
    predictably.
    """
    api = HfApi(token=token)
    model_info = api.model_info(repo_id=model_id, files_metadata=True)

    snapshot_dir = os.path.join(get_model_cache_dir(model_id), "snapshots")
    missing_files: List[Tuple[str, int]] = []

    for sibling in model_info.siblings or []:
        if sibling.rfilename.endswith(config.MODEL_FILE_EXCLUSIONS):
            continue

        already_exists = False
        if os.path.exists(snapshot_dir):
            for snap in os.listdir(snapshot_dir):
                if os.path.exists(os.path.join(snapshot_dir, snap, sibling.rfilename)):
                    already_exists = True
                    break

        if not already_exists:
            missing_files.append((sibling.rfilename, sibling.size or 1024 * 1024))

    return model_info, missing_files


def _download_missing_files_with_progress(model_id: str, q, token=None) -> str:
    """
    Download missing repo files and report true live byte-stream progress.

    We hook huggingface_hub's internal per-file progress bar factory so the UI
    sees aggregate byte updates across concurrent downloads and xet/http code
    paths, while still letting snapshot_download manage caching and resume.
    """
    _, missing_files = _get_missing_repo_files(model_id, token=token)

    total_to_download = sum(size for _, size in missing_files)
    files_to_download = len(missing_files)

    if files_to_download == 0:
        snapshot_path = _get_latest_snapshot_path(model_id)
        if snapshot_path is not None:
            q.put(("model_download_progress", (1, 1)))
            return snapshot_path

        local_model_path = snapshot_download(
            repo_id=model_id,
            token=token,
            local_dir_use_symlinks=_USE_SYMLINKS,
        )
        q.put(("model_download_progress", (1, 1)))
        return local_model_path

    q.put(("total_model_size", total_to_download))
    logging.info(
        f"Need to download {files_to_download} files, total size: "
        f"{format_size(total_to_download)} ({total_to_download:,} bytes)"
    )

    q.put(("status_update", f"Downloading {model_id}..."))
    tracker = LiveByteProgressTracker(q=q, total_bytes=total_to_download)
    original_get_progress_bar_context = hf_file_download._get_progress_bar_context

    def _patched_get_progress_bar_context(
        *, total=None, initial=0, _tqdm_bar=None, **kwargs
    ):
        if _tqdm_bar is not None:
            return nullcontext(_tqdm_bar)
        return nullcontext(tracker.register_bar(total=total, initial=initial))

    try:
        hf_file_download._get_progress_bar_context = _patched_get_progress_bar_context
        local_model_path = snapshot_download(
            repo_id=model_id,
            tqdm_class=SilentTqdm,  # type: ignore[arg-type]
            token=token,
            local_dir_use_symlinks=_USE_SYMLINKS,
        )
    finally:
        hf_file_download._get_progress_bar_context = original_get_progress_bar_context

    tracker.complete()
    return local_model_path


def get_dir_size(path: Path) -> int:
    """Calculate the total size of a directory in bytes."""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(Path(entry.path))
    except (OSError, PermissionError):
        pass
    return total


def format_size(size_bytes: int) -> str:
    """Format bytes as a human-readable string."""
    if size_bytes == 0:
        return "0 B"
    units = ("B", "KB", "MB", "GB", "TB")
    i = 0
    while size_bytes >= 1024 and i < len(units) - 1:
        size_bytes /= 1024
        i += 1
    return f"{size_bytes:.1f} {units[i]}"


def get_remote_model_size(model_id: str, token: Optional[str] = None) -> int:
    """Fetch the total size of a model from the Hub in bytes."""
    try:
        api = HfApi(token=token)
        # CRITICAL: files_metadata=True is required to get sizes of all files in repo
        model_info = api.model_info(repo_id=model_id, files_metadata=True)
        return sum(s.size for s in (model_info.siblings or []) if s.size is not None)
    except Exception as e:
        logging.warning(f"Failed to get size for {model_id}: {e}")
        return 0


def is_model_downloaded(model_id, token=None):
    """Check if a model is fully downloaded."""
    try:
        api = HfApi(token=token)
        model_info = api.model_info(repo_id=model_id)
        model_cache_dir = get_model_cache_dir(model_id)
        # Check for snapshot directory
        snapshot_dir = os.path.join(model_cache_dir, "snapshots")
        if not os.path.exists(snapshot_dir):
            return False
        # Get the latest snapshot
        snapshots = os.listdir(snapshot_dir)
        if not snapshots:
            return False
        latest_snapshot = sorted(snapshots)[-1]

        if model_info.siblings:
            for file_info in model_info.siblings:
                if file_info.rfilename.endswith(config.MODEL_FILE_EXCLUSIONS):
                    continue
                file_path = os.path.join(
                    snapshot_dir, latest_snapshot, file_info.rfilename
                )
                if not os.path.exists(file_path):
                    logging.info(
                        f"Model {model_id} is not fully downloaded. Missing file: {file_info.rfilename}"
                    )
                    return False
            return True
    except HTTPError as e:
        if e.response.status_code == 404:
            logging.warning(f"Model not found on Hub: {model_id}")
        else:
            logging.error(f"HTTPError checking model {model_id}: {e}")
        return False
    except Exception as e:
        logging.error(f"Error checking if model {model_id} is downloaded: {e}")
        return False


def get_downloaded_models(task, token=None):
    """Get a list of downloaded models for a given task."""
    logging.info(f"Searching for downloaded models with task: '{task}'")
    try:
        # Limit results to reduce network load and UI clutter
        models = list_models(
            filter=task,
            library="transformers",
            sort="downloads",
            direction=-1,
            limit=config.MODEL_SEARCH_LIMIT,
            token=token,
        )
        downloaded_models = []
        for model in models or []:
            if is_model_downloaded(model.id, token=token):
                downloaded_models.append(model.id)
        logging.info(f"Found {len(downloaded_models)} downloaded models.")
        return downloaded_models
    except Exception as e:
        logging.exception("Failed to find downloaded models.")
        return []


def find_models_worker(task, q, token=None):
    """Worker thread to fetch model list from Hugging Face Hub."""
    logging.info(
        f"Worker searching for top {config.MODEL_SEARCH_LIMIT} models with task: '{task}'"
    )
    try:
        # Request the top N models by downloads to keep the UI responsive.
        models = list_models(
            filter=task,
            library="transformers",
            sort="downloads",
            direction=-1,
            limit=config.MODEL_SEARCH_LIMIT,
            token=token,
        )
        all_found = [m.id for m in models or []]

        logging.info(f"Hub returned {len(all_found)} raw models: {all_found}")

        model_ids = all_found[: config.MODEL_SEARCH_LIMIT]
        logging.info(f"Filtering to top {len(model_ids)}: {model_ids}")

        downloaded_models = []
        for model_id in model_ids:
            is_down = is_model_downloaded(model_id, token=token)
            logging.info(f"Checking if {model_id} is downloaded: {is_down}")
            if is_down:
                downloaded_models.append(model_id)

        logging.info(
            f"Final list to GUI - Found: {len(model_ids)}, Downloaded: {len(downloaded_models)}"
        )
        q.put(("models_found", (model_ids, downloaded_models)))
    except Exception as e:
        logging.exception("Failed to find models.")
        q.put(("error", f"Failed to find models: {e}"))


def get_suggested_task(model_config: dict) -> str:
    """
    Suggests a pipeline task based on model configuration (architectures, pipeline_tag).
    """
    # 1. Check explicit pipeline_tag (from Hub metadata, might be in config if saved by some tools)
    if "pipeline_tag" in model_config:
        return model_config["pipeline_tag"]

    # 2. Check architectures
    archs = model_config.get("architectures", [])
    for arch in archs:
        arch_lower = arch.lower()
        if "forimageclassification" in arch_lower:
            return config.MODEL_TASK_IMAGE_CLASSIFICATION
        if "forconditionalgeneration" in arch_lower:
            return config.MODEL_TASK_IMAGE_TO_TEXT
        if "visionencoderdecoder" in arch_lower:
            return config.MODEL_TASK_IMAGE_TO_TEXT
        if "clipmodel" in arch_lower or "siglipmodel" in arch_lower:
            return config.MODEL_TASK_ZERO_SHOT

    # 3. Check model_type for known multi-modal models
    mtype = model_config.get("model_type", "").lower()
    if mtype in [
        "blip",
        "blip-2",
        "git",
        "qwen2_vl",
        "qwen2_5_vl",
        "qwen3_vl",
        "llava",
    ]:
        return config.MODEL_TASK_IMAGE_TO_TEXT


def get_model_capability(task: str) -> str:
    """Returns a human-readable capability string for a given task."""
    return config.CAPABILITY_MAP.get(task, "Unknown")


def find_local_models() -> Dict[str, Dict[str, Any]]:
    """
    Scan the local Hugging Face cache for previously downloaded models.

    This function parses the standard Hugging Face cache structure, extracts
    model configuration (task, capability), and calculates on-disk size for
    each identified model. Incompatible models (GPTQ, AWQ, etc.) are filtered out.

    Returns:
        A dictionary mapping model IDs to a metadata dict containing:
        'config', 'path', 'size_bytes', 'size_str', 'suggested_task', and 'capability'.
    """
    local_models = {}
    cache_path = Path(HUGGINGFACE_HUB_CACHE)
    if not cache_path.exists():
        return {}

    for model_dir in cache_path.glob("models--*"):
        if not model_dir.is_dir():
            continue

        model_id = model_dir.name[len("models--") :].replace("--", "/")

        # Skip incompatible models (GPTQ, AWQ, etc.)
        if not is_model_compatible(model_id):
            reason = get_incompatibility_reason(model_id)
            logging.debug(f"Skipping incompatible model {model_id}: {reason}")
            continue

        try:
            snapshot_dirs = [
                d for d in (model_dir / "snapshots").iterdir() if d.is_dir()
            ]
            if not snapshot_dirs:
                continue

            latest_snapshot = max(snapshot_dirs, key=lambda p: p.stat().st_mtime)
            config_path = latest_snapshot / "config.json"

            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    model_config = json.load(f)

                # Calculate size of the entire model directory (snapshots + metadata)
                size_bytes = get_dir_size(model_dir)

                # Infer suggested task
                suggested_task = get_suggested_task(model_config)
                capability = get_model_capability(suggested_task)

                local_models[model_id] = {
                    "config": model_config,
                    "path": latest_snapshot,
                    "size_bytes": size_bytes,
                    "size_str": format_size(size_bytes),
                    "suggested_task": suggested_task,
                    "capability": capability,
                }
        except Exception as e:
            logging.debug(f"Could not inspect model {model_id}: {e}")
            continue

    return local_models


def find_local_models_by_task(task: str) -> list[str]:
    """
    Finds locally cached models compatible with a given task by scanning the cache.

    Args:
        task: The pipeline task to filter by (e.g., 'image-classification').

    Returns:
        A list of model IDs that are cached locally and support the task.
    """
    all_local_models = find_local_models()
    task_specific_models = []
    for model_id, model_info in all_local_models.items():
        if model_info["config"].get("pipeline_tag") == task:
            task_specific_models.append(model_id)

    logging.info(f"Found {len(task_specific_models)} local models for task '{task}'.")
    return task_specific_models


def show_model_info_worker(model_id, q, token=None):
    """Worker thread to download a model's README file."""
    logging.info(f"Fetching README for model: {model_id}")
    try:
        readme_path = hf_hub_download(
            repo_id=model_id, filename="README.md", token=token
        )
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read()
        logging.info(f"Successfully fetched README for model: {model_id}")
        q.put(("model_info_found", readme_content))
    except Exception as e:
        logging.warning(f"Could not retrieve README for {model_id}. Error: {e}")
        q.put(("model_info_found", f"Could not retrieve README for {model_id}.\n\n{e}"))


# Legacy implementations kept temporarily while the live-stream path settles.
def _legacy_download_model_worker(model_id, q, token=None):
    """Worker thread specifically for downloading a model with accurate progress reporting."""
    logging.info(f"🚀 Starting model download worker for: {model_id}")
    try:
        if is_model_downloaded(model_id, token=token):
            logging.info(f"✅ Model {model_id} already fully downloaded.")
            q.put(("model_download_progress", (100, 100)))  # Force full progress
            q.put(("download_complete", model_id))
            return

        q.put(("status_update", f"Checking files for {model_id}..."))
        logging.info(f"🔍 Checking which files need to be downloaded for {model_id}...")

        # Determine missing bytes to make progress bar accurate
        api = HfApi(token=token)
        model_info = api.model_info(repo_id=model_id, files_metadata=True)

        model_cache_dir = get_model_cache_dir(model_id)
        snapshot_dir = os.path.join(model_cache_dir, "snapshots")

        total_to_download = 0
        files_to_download = 0
        for sibling in model_info.siblings or []:
            if sibling.rfilename.endswith(config.MODEL_FILE_EXCLUSIONS):
                continue

            already_exists = False
            if os.path.exists(snapshot_dir):
                for snap in os.listdir(snapshot_dir):
                    if os.path.exists(
                        os.path.join(snapshot_dir, snap, sibling.rfilename)
                    ):
                        already_exists = True
                        break

            if not already_exists:
                total_to_download += sibling.size or 0
                files_to_download += 1

        # Fallback if calculation is zero (e.g. metadata only)
        if total_to_download == 0:
            total_to_download = sum(
                s.size for s in (model_info.siblings or []) if s.size
            )

        # If sizes metadata are missing (common for some Hub entries), provide a sane
        # fallback so the UI progress bar can advance. We approximate per-file size
        # as 1MB when we know how many files would have been downloaded.
        if total_to_download == 0 and files_to_download > 0:
            total_to_download = (
                files_to_download * 1024 * 1024
            )  # 1MB per file as fallback
            logging.warning(
                f"Model size metadata missing; using fallback total_to_download={total_to_download} bytes"
            )

        q.put(("total_model_size", total_to_download))
        logging.info(
            f"📦 Need to download {files_to_download} files, total size: {format_size(total_to_download)} ({total_to_download:,} bytes)"
        )

        q.put(("status_update", f"Downloading {model_id}..."))
        logging.info(f"⬇️  Starting download of {model_id}...")

        snapshot_download(
            repo_id=model_id,
            tqdm_class=SilentTqdm,  # type: ignore[arg-type]
            token=token,
            local_dir_use_symlinks=_USE_SYMLINKS,
        )

        # Final update to ensure it hits 100%
        q.put(("model_download_progress", (total_to_download, total_to_download)))

        logging.info(f"✅ Model download complete for {model_id}!")
        q.put(("download_complete", model_id))
    except Exception as e:
        logging.exception(f"❌ Failed to download model: {model_id}")
        q.put(("error", f"Failed to download model: {e}"))


def _legacy_load_model_with_progress(model_id, task, q, token=None, device=-1):
    """Worker thread to load a model with progress reporting."""
    logging.info(f"Starting model load for: {model_id} on device {device}")

    try:
        if not is_model_downloaded(model_id, token=token):
            q.put(("status_update", f"Checking files for {model_id}..."))

            api = HfApi(token=token)
            model_info = api.model_info(repo_id=model_id)

            # Simple missing bytes calculation for progress accuracy
            total_missing = sum(s.size for s in (model_info.siblings or []) if s.size)

            q.put(("total_model_size", total_missing))
            q.put(("status_update", f"Downloading {model_id}..."))
            local_model_path = snapshot_download(
                repo_id=model_id,
                tqdm_class=SilentTqdm,
                token=token,
                local_dir_use_symlinks=_USE_SYMLINKS,
            )
            q.put(("model_download_progress", (total_missing, total_missing)))
        else:
            logging.info(f"Model {model_id} already downloaded.")
            model_cache_dir = get_model_cache_dir(model_id)
            snapshot_dir = os.path.join(model_cache_dir, "snapshots")
            snapshots = os.listdir(snapshot_dir)
            if snapshots:
                latest_snapshot = sorted(snapshots)[-1]
                local_model_path = os.path.join(snapshot_dir, latest_snapshot)
            else:
                local_model_path = snapshot_download(
                    repo_id=model_id,
                    tqdm_class=SilentTqdm,
                    token=token,
                    local_dir_use_symlinks=_USE_SYMLINKS,
                )

        q.put(("status_update", f"Initializing model {model_id}..."))

        # Auto-Task Detection
        try:
            cfg_path = os.path.join(local_model_path, "config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as cf:
                    cfg = json.load(cf)
                suggested = get_suggested_task(cfg)
                if suggested != task:
                    if (
                        task == config.MODEL_TASK_IMAGE_CLASSIFICATION
                        and suggested == config.MODEL_TASK_IMAGE_TO_TEXT
                    ):
                        task = suggested
                    elif (
                        task == config.MODEL_TASK_IMAGE_TO_TEXT
                        and suggested == config.MODEL_TASK_IMAGE_CLASSIFICATION
                    ):
                        task = suggested
        except Exception:
            pass

        # Load pipeline (transformers handles processor/tokenizer automatically for multi-modal)
        # Note: low_cpu_mem_usage must go in model_kwargs, NOT as a top-level kwarg,
        # because pipeline() forwards unknown kwargs to _sanitize_parameters() which
        # rejects them for task-specific pipelines like ImageClassificationPipeline.
        model = pipeline(
            task,
            model=local_model_path,
            device_map="auto" if device != -1 else None,
            device=device if device == -1 else None,
            torch_dtype="auto",
            model_kwargs={"low_cpu_mem_usage": True},
        )

        logging.info(f"Model pipeline ({task}) loaded successfully for: {model_id}")
        q.put(("model_loaded", {"model": model, "model_name": model_id}))

    except Exception as e:
        logging.exception(f"Failed to load model: {model_id}")
        q.put(("error", f"Failed to load model: {e}"))


def download_model_worker(model_id, q, token=None):
    """Worker thread specifically for downloading a model with accurate progress reporting."""
    logging.info(f"Starting model download worker for: {model_id}")
    try:
        if is_model_downloaded(model_id, token=token):
            logging.info(f"Model {model_id} already fully downloaded.")
            q.put(("model_download_progress", (100, 100)))
            q.put(("download_complete", model_id))
            return

        q.put(("status_update", f"Checking files for {model_id}..."))
        logging.info(f"Checking which files need to be downloaded for {model_id}...")

        _download_missing_files_with_progress(model_id, q, token=token)

        logging.info(f"Model download complete for {model_id}!")
        q.put(("download_complete", model_id))
    except Exception as e:
        logging.exception(f"Failed to download model: {model_id}")
        q.put(("error", f"Failed to download model: {e}"))


def load_model_with_progress(model_id, task, q, token=None, device=-1):
    """Worker thread to load a model with progress reporting."""
    logging.info(f"Starting model load for: {model_id} on device {device}")

    try:
        if not is_model_downloaded(model_id, token=token):
            q.put(("status_update", f"Checking files for {model_id}..."))
            local_model_path = _download_missing_files_with_progress(
                model_id, q, token=token
            )
        else:
            logging.info(f"Model {model_id} already downloaded.")
            local_model_path = _get_latest_snapshot_path(model_id)
            if local_model_path is None:
                local_model_path = snapshot_download(
                    repo_id=model_id,
                    tqdm_class=SilentTqdm,  # type: ignore[arg-type]
                    token=token,
                    local_dir_use_symlinks=_USE_SYMLINKS,
                )

        q.put(("status_update", f"Initializing model {model_id}..."))

        try:
            cfg_path = os.path.join(local_model_path, "config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as cf:
                    cfg = json.load(cf)
                suggested = get_suggested_task(cfg)
                if suggested != task:
                    if (
                        task == config.MODEL_TASK_IMAGE_CLASSIFICATION
                        and suggested == config.MODEL_TASK_IMAGE_TO_TEXT
                    ):
                        task = suggested
                    elif (
                        task == config.MODEL_TASK_IMAGE_TO_TEXT
                        and suggested == config.MODEL_TASK_IMAGE_CLASSIFICATION
                    ):
                        task = suggested
        except Exception:
            pass

        model = pipeline(
            task,
            model=local_model_path,
            device_map="auto" if device != -1 else None,
            device=device if device == -1 else None,
            torch_dtype="auto",
            model_kwargs={"low_cpu_mem_usage": True},
        )

        logging.info(f"Model pipeline ({task}) loaded successfully for: {model_id}")
        q.put(("model_loaded", {"model": model, "model_name": model_id}))

    except Exception as e:
        logging.exception(f"Failed to load model: {model_id}")
        q.put(("error", f"Failed to load model: {e}"))


def find_models_by_task(task: str) -> Tuple[List[str], List[str]]:
    """
    Search the Hugging Face Hub for models supporting a specific task.

    This function retrieves the most popular models for the given task,
    filtered to those compatible with the 'transformers' library. It also
    identifies which of the found models are already available in the local cache.

    Args:
        task: The technical task identifier (e.g., 'image-classification').

    Returns:
        A tuple containing (all_model_ids, downloaded_model_ids).
    """
    logging.info(f"Searching for models (sync) with task: '{task}'")
    try:
        # Limit to the top N models to avoid overwhelming the UI and reduce network usage
        models = list_models(
            filter=task,
            library="transformers",
            sort="downloads",
            direction=-1,
            limit=config.MODEL_SEARCH_LIMIT,
        )
        model_ids = [
            model.id
            for model in models or []
            if is_model_suitable_for_local_inference(model.id, task=task)
        ][: config.MODEL_SEARCH_LIMIT]
        downloaded_models = [mid for mid in model_ids if is_model_downloaded(mid)]
        logging.info(
            f"Found {len(model_ids)} models (sync). {len(downloaded_models)} cached locally."
        )
        return model_ids, downloaded_models
    except Exception as e:
        logging.exception("Failed to find models (sync).")
        return [], []


def get_model_info(model_id):
    """Return the README (or a helpful message) for a model synchronously."""
    logging.info(f"Fetching README (sync) for model: {model_id}")
    try:
        readme_path = hf_hub_download(repo_id=model_id, filename="README.md")
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logging.warning(f"Could not retrieve README for {model_id}. Error: {e}")
        return f"Could not retrieve README for {model_id}.\n\n{e}"


def load_model(
    model_id: str,
    task: str,
    progress_queue: Optional[Any] = None,
    token: Optional[str] = None,
    device: int = -1,
) -> Any:
    """
    Synchronously load a Hugging Face model and initialize a pipeline.

    This function handles the end-to-end model loading process:
    1. Checking the local cache.
    2. Downloading model snapshot if missing.
    3. Auto-detecting the optimal pipeline task (e.g., handles VLMs).
    4. Initializing the transformers pipeline with appropriate processors.

    Args:
        model_id: The Hugging Face repository ID (e.g., 'google/vit-base').
        task: The intended model task (e.g., 'image-classification').
        progress_queue: Optional queue for status and percentage updates.
        token: Optional Hugging Face API token for private/gated models.
        device: Device ID to load onto (-1 for CPU, 0+ for CUDA/MPS).

    Returns:
        The initialized transformers Pipeline object.

    Raises:
        Exception: Various errors if downloading or initialization fails.

    Note:
        Modern Multi-modal models (like Qwen2-VL) will automatically be routed
        to the 'image-text-to-text' pipeline regardless of the input 'task'.
    """
    logging.info(f"Starting synchronous model load for: {model_id} on device {device}")
    try:
        q = progress_queue
        if q:
            q.put(
                (
                    "status_update",
                    f"Downloading/initializing model {model_id} on device {device}...",
                )
            )

        if not is_model_downloaded(model_id, token=token):
            logging.info(f"Downloading model files for {model_id} (sync)...")
            if q:
                local_model_path = _download_missing_files_with_progress(
                    model_id, q, token=token
                )
            else:
                local_model_path = snapshot_download(
                    repo_id=model_id,
                    tqdm_class=SilentTqdm,  # type: ignore[arg-type]
                    token=token,
                    local_dir_use_symlinks=_USE_SYMLINKS,
                )
            logging.info(f"Model download complete for {model_id} (sync).")
        else:
            logging.info(f"Model {model_id} is already downloaded (sync).")
            model_cache_dir = get_model_cache_dir(model_id)
            snapshot_dir = os.path.join(model_cache_dir, "snapshots")
            snapshots = os.listdir(snapshot_dir)
            if snapshots:
                latest_snapshot = sorted(snapshots)[-1]
                local_model_path = os.path.join(snapshot_dir, latest_snapshot)
                logging.info(f"Using latest snapshot: {latest_snapshot}")
            else:
                # Should typically not happen if is_model_downloaded returned True,
                # but good for safety.
                logging.warning(
                    f"Snapshot directory exists but is empty for {model_id}. Re-downloading..."
                )
                local_model_path = snapshot_download(
                    repo_id=model_id,
                    tqdm_class=SilentTqdm,
                    token=token,
                    local_dir_use_symlinks=_USE_SYMLINKS,
                )

        if q:
            q.put(("status_update", f"Initializing model {model_id}..."))

        # Validation and Auto-Task Detection
        try:
            cfg_path = os.path.join(local_model_path, "config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as cf:
                    cfg = json.load(cf)
                suggested = get_suggested_task(cfg)
                if suggested != task:
                    if (
                        task == config.MODEL_TASK_IMAGE_CLASSIFICATION
                        and suggested == config.MODEL_TASK_IMAGE_TO_TEXT
                    ):
                        task = suggested
                    elif (
                        task == config.MODEL_TASK_IMAGE_TO_TEXT
                        and suggested == config.MODEL_TASK_IMAGE_CLASSIFICATION
                    ):
                        task = suggested
        except Exception:
            pass

        # Initialize pipeline with processor for multi-modal stability
        # For modern VLMs (Qwen*-VL, LLaVA), 'image-text-to-text' is preferred
        pipeline_task = task
        try:
            cfg_path = os.path.join(local_model_path, "config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as cf:
                    m_cfg = json.load(cf)
                m_type = m_cfg.get("model_type", "").lower()
                # Match any Qwen VL variant (qwen2_vl, qwen2_5_vl, qwen3_vl, …)
                if "qwen" in m_type and "vl" in m_type:
                    pipeline_task = "image-text-to-text"
                    logging.info(
                        f"Using '{pipeline_task}' pipeline for model type '{m_type}'"
                    )
                elif m_type in ["llava", "idefics", "paligemma"]:  # Other known VLMs
                    pipeline_task = "image-text-to-text"
                    logging.info(
                        f"Using '{pipeline_task}' pipeline for model type '{m_type}'"
                    )
                elif m_type in [
                    "llm",
                    "mistral",
                    "gemma",
                    "liquid",
                    "qwen2_vl",
                    "qwen2_5_vl",
                ]:
                    # LiquidAI/LFM2-VL and similar VLMs use this family
                    pipeline_task = "image-text-to-text"
                    logging.info(
                        f"Using '{pipeline_task}' pipeline for model type '{m_type}'"
                    )

            # Also detect VLMs by model_id pattern (LiquidAI/LFM2-VL-*, LLaVA, etc.)
            model_id_lower = model_id.lower()
            if (
                ("lfm" in model_id_lower and "vl" in model_id_lower)
                or ("llava" in model_id_lower)
                or ("liquidai" in model_id_lower)
            ):
                pipeline_task = "image-text-to-text"
                logging.info(
                    f"Using '{pipeline_task}' pipeline for VLM model_id '{model_id}'"
                )
        except Exception:
            pass

        # Load processor if it exists
        processor = None
        try:
            processor = AutoProcessor.from_pretrained(local_model_path)
            logging.debug("AutoProcessor loaded successfully.")
        except Exception:
            logging.debug(
                "No AutoProcessor found, falling back to default pipeline behavior."
            )

        # Load model using memory optimizations:
        # - low_cpu_mem_usage: reduces peak RAM (passed via model_kwargs to avoid
        #   _sanitize_parameters() rejection in task-specific pipelines)
        # - torch_dtype="auto": uses float16 on GPU if available
        # - device_map="auto": handles complex device placement (requires accelerate)
        model = pipeline(
            pipeline_task,
            model=local_model_path,
            processor=processor,
            device_map="auto" if device != -1 else None,
            device=device if device == -1 else None,
            torch_dtype="auto",
            model_kwargs={"low_cpu_mem_usage": True},
        )

        logging.info(
            f"Model pipeline ({pipeline_task}) loaded successfully for: {model_id} on device {device}"
        )
        return model

    except Exception as e:
        logging.exception(f"Failed to load model (sync): {model_id}")
        if progress_queue:
            progress_queue.put(("error", f"Failed to load model: {e}"))
        raise


# -------------------------------------------------------------------------
# API Inference Support
# -------------------------------------------------------------------------


class RateLimitError(Exception):
    """Raised when Hugging Face API rate limit is exceeded."""

    def __init__(self, retry_after=None):
        self.retry_after = retry_after
        msg = f"Hugging Face API rate limit exceeded."
        if retry_after:
            msg += f" Retry after {retry_after} seconds."
        msg += " Consider downloading the model for unlimited local inference."
        super().__init__(msg)


def rate_limit_handler(max_retries=3, initial_delay=1.0):
    """
    Decorator to handle rate limiting and network errors for API calls.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            delay = initial_delay

            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Check for HTTP 429 (Rate Limit)
                    is_rate_limit = False
                    if hasattr(e, "response") and hasattr(e.response, "status_code"):
                        if e.response.status_code == 429:
                            is_rate_limit = True
                    # Also check for message content if exception type is generic
                    if "429" in str(e) or "Rate limit" in str(e):
                        is_rate_limit = True

                    if is_rate_limit:
                        if retries >= max_retries:
                            logging.error(f"Max retries exceeded for API call: {e}")
                            raise

                        # Check for retry-after header
                        wait_time = delay
                        if hasattr(e, "response") and hasattr(e.response, "headers"):
                            if "retry-after" in e.response.headers:
                                try:
                                    wait_time = (
                                        float(e.response.headers["retry-after"]) + 1.0
                                    )  # Add buffer
                                except:
                                    pass

                        logging.warning(
                            f"⚠️ Rate limited by Hugging Face API. "
                            f"Waiting {wait_time:.0f}s before retry {retries + 1}/{max_retries}... "
                            f"Consider downloading the model for unlimited local inference."
                        )
                        time.sleep(wait_time)
                        retries += 1
                        delay *= 2  # Exponential backoff for subsequent defaults

                    elif (
                        hasattr(e, "response")
                        and hasattr(e.response, "status_code")
                        and e.response.status_code >= 500
                    ):
                        # Server error, retry
                        if retries >= max_retries:
                            logging.error(f"Max retries exceeded for server error: {e}")
                            raise
                        logging.warning(
                            f"Server error {e.response.status_code}. Retrying {retries + 1}/{max_retries}..."
                        )
                        time.sleep(delay)
                        retries += 1
                        delay *= 2

                    else:
                        # Other errors (Auth, BadRequest) - do not retry
                        raise

        return wrapper

    return decorator


@rate_limit_handler(max_retries=3)
def run_inference_api(model_id, image_path, task, token, parameters=None):
    """
    Runs inference using the Hugging Face Inference API.

    Args:
        model_id: The model ID on HF Hub.
        image_path: Path to local image file.
        task: The task type (e.g. 'image-classification').
        token: HF API Token.
        parameters: Optional parameters dict.

    Returns:
        The raw JSON response from the API.
    """
    from src.utils.logger import log_api_request, log_api_response

    logger = logging.getLogger(__name__)

    logger.info(
        f"[HuggingFace API] Starting inference - Model: {model_id}, Task: {task}"
    )
    start_time = time.time()

    # Note: InferenceClient is only needed for zero-shot (other paths use direct HTTP).
    # We create it lazily below to avoid allocating connection pools unnecessarily.
    client = None

    # Map internal task names to API tasks if needed, though they usually match.
    # We mainly need to handle the input type.

    # For image tasks, we pass the file.
    try:
        # Check image validity
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # InferenceClient.predict() or specific task methods can be used.
        # .image_classification() is specific.
        # .image_to_text() is specific.

        logger.debug(f"Image path: {image_path}")
        logger.debug(f"Parameters: {parameters}")

        if task == config.MODEL_TASK_IMAGE_CLASSIFICATION:
            try:
                with open(image_path, "rb") as img_f:
                    b64_image = base64.b64encode(img_f.read()).decode("utf-8")

                payload = {"inputs": b64_image}
                # Free the standalone base64 copy now that it's in the payload
                del b64_image

                # Using the router endpoint
                api_url = (
                    f"https://router.huggingface.co/hf-inference/models/{model_id}"
                )
                headers = {"Authorization": f"Bearer {token}"}

                log_api_request(logger, "POST", api_url, headers=headers, data=payload)

                response = requests.post(api_url, headers=headers, json=payload)
                del payload  # Free the payload immediately after sending
                elapsed = time.time() - start_time

                log_api_response(logger, response.status_code, elapsed_time=elapsed)
                response.raise_for_status()

                result = response.json()
                response.close()  # Release socket buffers
                logger.info(
                    f"[HuggingFace API] Inference successful - Duration: {elapsed:.3f}s"
                )
                return result

            except requests.exceptions.HTTPError as e:
                if e.response.status_code in [404, 410]:
                    raise ValueError(
                        f"Model {model_id} is not available on the free Hugging Face Inference API. Status: {e.response.status_code}"
                    )
                raise e

        elif task == config.MODEL_TASK_ZERO_SHOT:
            # Zero shot requires candidate labels in parameters
            if not parameters or "candidate_labels" not in parameters:
                raise ValueError("candidate_labels required for zero-shot api")

            try:
                # Lazily create InferenceClient only when needed
                client = InferenceClient(token=token)
                return client.zero_shot_image_classification(
                    image_path,
                    model=model_id,
                    candidate_labels=parameters["candidate_labels"],
                )
            except Exception as e:
                # Fallback for StopIteration or other client issues
                logging.warning(
                    f"Native zero-shot client failed ({type(e).__name__}), falling back to raw JSON API..."
                )

                with open(image_path, "rb") as img_f:
                    b64_image = base64.b64encode(img_f.read()).decode("utf-8")

                payload = {
                    "inputs": b64_image,
                    "parameters": {"candidate_labels": parameters["candidate_labels"]},
                }
                # Free the standalone base64 copy
                del b64_image

                # Direct API call to bypass client library issues and deprecated endpoints
                # Using the new router endpoint
                api_url = (
                    f"https://router.huggingface.co/hf-inference/models/{model_id}"
                )
                headers = {"Authorization": f"Bearer {token}"}

                response = requests.post(api_url, headers=headers, json=payload)
                del payload  # Free immediately after sending
                try:
                    response.raise_for_status()
                except requests.exceptions.HTTPError as e:
                    if response.status_code in [404, 410]:
                        raise ValueError(
                            f"Model {model_id} is not available on the free Hugging Face Inference API (Status {response.status_code}). Please use 'Local' mode or try a different model."
                        )
                    raise e

                result = response.json()
                response.close()  # Release socket buffers
                return result

        elif task == config.MODEL_TASK_IMAGE_TO_TEXT:
            try:
                with open(image_path, "rb") as img_f:
                    b64_image = base64.b64encode(img_f.read()).decode("utf-8")

                gen_kwargs = parameters.get("generate_kwargs", {}) or {}

                payload = {"inputs": b64_image, "parameters": gen_kwargs}
                # Free the standalone base64 copy
                del b64_image

                # Using the router endpoint
                api_url = (
                    f"https://router.huggingface.co/hf-inference/models/{model_id}"
                )
                headers = {"Authorization": f"Bearer {token}"}

                response = requests.post(api_url, headers=headers, json=payload)
                del payload  # Free immediately after sending
                response.raise_for_status()
                result = response.json()
                response.close()  # Release socket buffers
                return result

            except requests.exceptions.HTTPError as e:
                if e.response.status_code in [404, 410]:
                    raise ValueError(
                        f"Model {model_id} is not available on the free Hugging Face Inference API. Status: {e.response.status_code}"
                    )
                raise e

        else:
            # Fallback to generic — lazily create client
            client = InferenceClient(token=token)
            return client.post(json={"inputs": image_path}, model=model_id, task=task)

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(
            f"[HuggingFace API] Inference failed after {elapsed:.3f}s: {type(e).__name__}: {str(e)}"
        )
        logger.exception("Full traceback:")
        raise
