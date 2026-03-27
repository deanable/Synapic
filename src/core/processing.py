"""
Processing Pipeline Module
===========================

This module implements the core processing pipeline that orchestrates the entire
image tagging workflow. It manages the multi-threaded execution of AI model inference
and metadata writing operations.

Key Components:
- ProcessingManager: Main orchestrator class that runs in a background thread
- Item fetching: Retrieves images from local filesystem or Daminion
- Model initialization: Loads AI models for local inference
- Processing loop: Iterates through items, runs inference, writes metadata
- Progress tracking: Reports status to UI via callbacks

Threading Model:
- Main thread: UI event loop
- Background thread: Processing pipeline (created by ProcessingManager.start())
- The background thread can be interrupted via stop_event

Workflow Stages:
1. Fetch items (local folder scan or Daminion query)
2. Initialize model (if using local inference)
3. Process each item:
   a. Load image
   b. Run AI inference
   c. Extract tags from results
   d. Write metadata (EXIF/IPTC or Daminion)
   e. Verify metadata (optional)
4. Update statistics and progress

Author: Dean
"""

import gc
import logging
import threading
import time

try:
    import psutil

    _PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    _PSUTIL_AVAILABLE = False
from pathlib import Path
from typing import Callable, Optional
from PIL import Image

# Internal modules
from .session import Session
from . import huggingface_utils
from . import openrouter_utils
from . import image_processing
from . import config

# Optional Groq integration (for Groq SDK-based inference)
try:
    from src.integrations.groq_package_client import GroqPackageClient

    GROQ_AVAILABLE = True
except ImportError:
    GroqPackageClient = None
    GROQ_AVAILABLE = False

# Optional Ollama integration (official client with host config)
try:
    from src.integrations.ollama_client import OllamaClient

    OLLAMA_AVAILABLE = True
except ImportError:
    OllamaClient = None
    OLLAMA_AVAILABLE = False

# Optional Nvidia integration
try:
    from src.integrations.nvidia_client import NvidiaClient

    NVIDIA_AVAILABLE = True
except ImportError:
    NvidiaClient = None
    NVIDIA_AVAILABLE = False

# Optional Google AI Studio integration
try:
    from src.integrations.google_ai_client import GoogleAIClient

    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GoogleAIClient = None
    GOOGLE_AI_AVAILABLE = False

# Optional Cerebras Inference integration
try:
    from src.integrations.cerebras_client import CerebrasClient

    CEREBRAS_AVAILABLE = True
except ImportError:
    CerebrasClient = None
    CEREBRAS_AVAILABLE = False

# Optional metadata verification (for testing/debugging)
# This module may not be available in packaged distributions
try:
    import tests.verify_metadata as verifier
except ImportError:
    # Fallback if tests is not in path (e.g. when packaged)
    verifier = None


# ============================================================================
# PROCESSING MANAGER
# ============================================================================


class ProcessingManager:
    """
    Main processing orchestrator that runs the AI tagging pipeline.

    This class manages the entire processing workflow in a background thread,
    allowing the UI to remain responsive. It coordinates between:
    - Data source (local files or Daminion)
    - AI engine (local models or cloud APIs)
    - Metadata writing (EXIF/IPTC or Daminion API)

    The processing runs asynchronously and can be aborted by the user at any time.
    Progress and log messages are sent to the UI via callback functions.

    Attributes:
        session: Session object containing all configuration and state
        log: Callback function for sending log messages to UI
        progress: Callback function for updating progress bar (percentage, current, total)
        stop_event: Threading event used to signal abortion
        thread: Background thread running the processing job
        logger: Python logger for file-based logging
        model: Loaded AI model (only for local inference)

    Example:
        >>> manager = ProcessingManager(session, log_callback, progress_callback)
        >>> manager.start()  # Starts background thread
        >>> # ... user can abort ...
        >>> manager.abort()  # Signals thread to stop
    """

    def __init__(
        self,
        session: Session,
        log_callback: Callable[[str], None],
        progress_callback: Callable[[float, int, int], None],
        auto_paginate: bool = False,
    ):
        """
        Initialize the processing manager.

        Args:
            session: Session object with datasource and engine configuration
            log_callback: Function to call with log messages for UI display
            progress_callback: Function to call with progress updates (percentage, current, total)
            auto_paginate: When True the Daminion fetch loop repeats in 500-record
                           pages until the server has no more items to return.
        """
        self.session = session
        self.log = log_callback  # UI log callback
        self.progress = progress_callback  # UI progress callback
        self.stop_event = threading.Event()  # Signal for aborting
        self.thread = None  # Background processing thread
        self.logger = logging.getLogger(__name__)  # File logger
        self.auto_paginate = (
            auto_paginate  # Whether to page through all 500-record batches
        )

    def start(self):
        """
        Start the processing job in a background thread.

        This method creates and starts a daemon thread that runs the entire
        processing pipeline. The thread will automatically terminate when the
        main program exits.

        The processing workflow is:
        1. Reset statistics
        2. Fetch items from datasource
        3. Initialize model (if local)
        4. Process each item
        5. Report completion
        """
        self.logger.info("Starting processing job")
        self.logger.info(
            f"Datasource: {self.session.datasource.type}, Engine: {self.session.engine.provider}"
        )
        self.logger.info(
            f"Model: {self.session.engine.model_id}, Task: {self.session.engine.task}"
        )

        # Clear any previous abort signal
        self.stop_event.clear()

        # Create and start background thread
        # daemon=True ensures thread terminates when main program exits
        self.thread = threading.Thread(target=self._run_job, daemon=True)
        self.thread.start()

    def abort(self):
        """
        Request abortion of the current processing job.

        This method sets a flag that the background thread checks between
        each item. The thread will stop processing new items but will
        complete the current item before exiting.

        Note: This is a graceful shutdown - the current item will finish processing.
        """
        if self.stop_event.is_set():
            return

        self.logger.warning("Processing job abort requested")
        self.stop_event.set()  # Signal the background thread to stop
        self.log("Stopping job... please wait.")

    def shutdown(self, timeout=2.0):
        """
        Ensure the processing manager shuts down completely.
        Called during application exit.

        Args:
            timeout: Maximum time to wait for the thread to join
        """
        if self.thread and self.thread.is_alive():
            self.logger.info("ProcessingManager shutdown initiated")
            self.abort()
            self.thread.join(timeout=timeout)
            if self.thread.is_alive():
                self.logger.warning(
                    f"Processing thread did not terminate within {timeout}s - proceeding anyway"
                )

    def _run_job(self):
        """
        Main processing loop (runs in background thread).

        Orchestrates the entire workflow:
        1. Fetch a page of items from the datasource
        2. Initialize the AI model (local only, once)
        3. Process each item in the page
        4. If auto_paginate is enabled and the page was full (500 items),
           fetch the next page and repeat until exhausted
        5. Cleanup on completion
        """
        DAMINION_PAGE_SIZE = 500  # Hard limit imposed by Daminion API

        try:
            self.log("Job started.")
            self.session.reset_stats()  # Clear previous run statistics

            # ================================================================
            # STAGE 1: INITIALIZE MODEL / API CLIENT — done once before loop
            # ================================================================
            self._api_client = None  # Will hold the reusable API client
            engine = self.session.engine

            if engine.provider == "local":
                self._init_local_model()
            elif engine.provider == "groq_package":
                if not GROQ_AVAILABLE:
                    raise RuntimeError(
                        "Groq SDK not available. Please install it with: pip install groq"
                    )
                import os

                groq_api_key = engine.groq_api_key
                if groq_api_key:
                    os.environ["GROQ_API_KEY"] = groq_api_key
                self._api_client = GroqPackageClient(api_key=groq_api_key)
                if not self._api_client.is_available():
                    raise RuntimeError(
                        "Groq SDK is not available or not properly configured."
                    )
                self.logger.info("Groq client initialized (reused for all items)")
            elif engine.provider == "ollama":
                if not OLLAMA_AVAILABLE:
                    raise RuntimeError(
                        "Ollama client not available. Please install 'ollama' package."
                    )
                self._api_client = OllamaClient(
                    host=engine.ollama_host, api_key=engine.ollama_api_key
                )
                if not self._api_client.is_available():
                    raise RuntimeError("Ollama client could not be initialized.")
                self.logger.info("Ollama client initialized (reused for all items)")
            elif engine.provider == "nvidia":
                if not NVIDIA_AVAILABLE:
                    raise RuntimeError("Nvidia client not available.")
                self._api_client = NvidiaClient(api_key=engine.nvidia_api_key)
                if not self._api_client.is_available():
                    raise RuntimeError("Nvidia API key not configured.")
                self.logger.info("Nvidia client initialized (reused for all items)")
            elif engine.provider == "google_ai":
                if not GOOGLE_AI_AVAILABLE:
                    raise RuntimeError("Google AI client not available.")
                self._api_client = GoogleAIClient(api_key=engine.google_ai_api_key)
                if not self._api_client.is_available():
                    raise RuntimeError("Google AI API key not configured.")
                self.logger.info("Google AI client initialized (reused for all items)")
            elif engine.provider == "cerebras":
                if not CEREBRAS_AVAILABLE:
                    raise RuntimeError(
                        "Cerebras SDK not available. "
                        "Please install it with: pip install cerebras_cloud_sdk"
                    )
                self._api_client = CerebrasClient(api_key=engine.cerebras_api_key)
                if not self._api_client.is_available():
                    raise RuntimeError(
                        self._api_client.availability_error()
                        or "Cerebras client is unavailable."
                    )
                self.logger.info("Cerebras client initialized (reused for all items)")

            # ================================================================
            # STAGE 2: PAGINATED FETCH + PROCESS LOOP
            # ================================================================
            # For Daminion sources the API caps every response at 500 records.
            # When auto_paginate=True we use a "reload-search" strategy:
            # after each batch is processed we re-fetch from offset 0 instead
            # of advancing the offset.  Items that were just tagged are
            # excluded by the server's own untagged filter, so each fresh
            # fetch naturally returns the next set of untagged items without
            # any risk of re-processing already-tagged records.
            # For local sources offset is ignored and only one pass is made.
            page_num = 0
            grand_total_processed = 0
            last_page_ids: set = set()  # Guard against infinite loops

            # ================================================================
            # PRE-FLIGHT COUNT — log the server-side total before fetching
            # ================================================================
            # For Daminion sources: ask the server how many records match the
            # current filters so we can confirm every page is retrieved.
            ds = self.session.datasource
            if ds.type == "daminion" and self.session.daminion_client:
                try:
                    untagged_fields = []
                    if ds.daminion_untagged_keywords:
                        untagged_fields.append("Keywords")
                    if ds.daminion_untagged_categories:
                        untagged_fields.append("Category")
                    if ds.daminion_untagged_description:
                        untagged_fields.append("Description")
                    expected_total = (
                        self.session.daminion_client.get_filtered_item_count(
                            scope=ds.daminion_scope,
                            saved_search_id=ds.daminion_saved_search_id
                            or ds.daminion_saved_search,
                            collection_id=ds.daminion_collection_id
                            or ds.daminion_catalog_id,
                            search_term=ds.daminion_search_term,
                            untagged_fields=untagged_fields,
                            status_filter=ds.status_filter,
                            force_refresh=True,
                        )
                    )
                    self.logger.info(
                        f"PRE-FLIGHT COUNT: server reports {expected_total} record(s) "
                        f"matching current filters (scope={ds.daminion_scope})"
                    )
                    self.log(
                        f"Server record count: {expected_total} item(s) "
                        f"matching filters before processing starts."
                    )
                except Exception as e:
                    self.logger.warning(f"Pre-flight count failed (non-fatal): {e}")

            while True:
                page_num += 1
                if page_num == 1:
                    self.log("Fetching items...")
                else:
                    self.log("Reloading search for next batch...")

                # ============================================================
                # FETCH ONE PAGE (always from offset 0 – reload-search strategy)
                # ============================================================
                items = self._fetch_items(offset=0)

                if not items:
                    if page_num == 1:
                        self.log("No items found to process.")
                    else:
                        self.log("No more items — all pages processed.")
                    break

                page_count = len(items)

                # ── Infinite-loop guard ──────────────────────────────────────
                # If the server returns the same IDs as the previous batch
                # (e.g. the untagged filter is not applied server-side), stop
                # rather than re-processing the same records forever.
                current_ids = {
                    item.get("id") if isinstance(item, dict) else str(item)
                    for item in items
                }
                if current_ids and current_ids == last_page_ids:
                    self.logger.warning(
                        "Reload-search returned identical items as the previous batch — "
                        "stopping to avoid infinite loop. "
                        "The server-side filter may not be filtering tagged items."
                    )
                    self.log(
                        "Warning: same items returned after reload — pagination stopped. "
                        "Check that the untagged filter is applied server-side."
                    )
                    del items
                    break
                last_page_ids = current_ids
                # ─────────────────────────────────────────────────────────────

                self.session.total_items += page_count
                self.logger.info(
                    f"Page {page_num}: {page_count} items fetched "
                    f"(reload-search, auto_paginate={self.auto_paginate})"
                )
                self.log(f"Page {page_num}: {page_count} item(s) to process.")

                # Reset per-page counter; overall progress uses session counters
                # more_pages=True: this is a page boundary, job is not done yet
                self.progress(
                    self.session.processed_items / max(self.session.total_items, 1),
                    self.session.processed_items,
                    self.session.total_items,
                    more_pages=True,
                )

                # ============================================================
                # PROCESS EACH ITEM IN THIS PAGE
                # ============================================================
                processed_before_batch = self.session.processed_items
                # ============================================================
                for i, item in enumerate(items):
                    if self.stop_event.is_set():
                        self.logger.info(
                            f"Job aborted by user after processing "
                            f"{grand_total_processed} items total"
                        )
                        self.log("Job aborted by user.")
                        # items will be freed by the outer stop_event guard below;
                        # do NOT del here to avoid UnboundLocalError.
                        break

                    self._process_single_item(item)

                    self.session.processed_items += 1
                    grand_total_processed += 1

                    # Log memory consumption after each image for debugging
                    if _PSUTIL_AVAILABLE:
                        mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                        self.logger.info(
                            f"Memory usage after image "
                            f"{self.session.processed_items}/{self.session.total_items}: "
                            f"{mem_mb:.2f} MB"
                        )

                    pct = self.session.processed_items / max(
                        self.session.total_items, 1
                    )
                    # Determine whether more pages will follow this one.
                    # A page is "definitely the last" if:
                    #   - auto_paginate is off (never fetches more), OR
                    #   - this page is partial (< 500 items, server is exhausted)
                    # Otherwise we conservatively keep more_pages=True even on the
                    # last item of a full page — the empty fetch that follows will
                    # simply exit the loop without emitting a misleading pct=1.0.
                    _is_last_page = (not self.auto_paginate) or (
                        page_count < DAMINION_PAGE_SIZE
                    )
                    _last_item_on_page = i == page_count - 1
                    _job_truly_done = _is_last_page and _last_item_on_page
                    self.progress(
                        pct,
                        self.session.processed_items,
                        self.session.total_items,
                        more_pages=not _job_truly_done,
                    )

                # Stop pagination if abort was requested
                if self.stop_event.is_set():
                    if "items" in dir():
                        del items
                    break

                # Stop if auto-pagination is off OR this was a partial page
                # (partial page = server has no more untagged records)
                if not self.auto_paginate or page_count < DAMINION_PAGE_SIZE:
                    if self.auto_paginate and page_count < DAMINION_PAGE_SIZE:
                        self.log(
                            f"Last batch received ({page_count} items) — "
                            "all items processed."
                        )
                    del items
                    break

                del items  # Free before fetching next batch

            # ================================================================
            # STAGE 3: COMPLETION & CLEANUP
            # ================================================================
            self.logger.debug("Hit end of processing loop")
            self.logger.info(
                f"Processing job completed — Processed: {self.session.processed_items}, "
                f"Failed: {self.session.failed_items}, Pages: {page_num}"
            )
            self.log(
                f"Job finished. "
                f"Processed {self.session.processed_items} item(s) across {page_num} page(s)."
            )

            # Explicitly unload model to free memory/VRAM
            if hasattr(self, "model") and self.model:
                self.logger.info("Unloading local model and performing memory cleanup")
                self.log(
                    "Cleaning up: unloading model from memory (this may take a moment)..."
                )
                self.model = None  # Release reference

                # Clear CUDA cache if GPU was used
                if self.session.engine.device == "cuda":
                    try:
                        import torch

                        if torch.cuda.is_available():
                            self.log("Cleaning up: flushing GPU VRAM cache...")
                            torch.cuda.empty_cache()
                            self.logger.info("CUDA cache cleared")
                    except ImportError:
                        pass

            # Close API client to free connection pools / HTTP sessions
            if self._api_client is not None:
                self.logger.info("Closing API client and freeing connection pools")
                if hasattr(self._api_client, "close"):
                    try:
                        self._api_client.close()
                    except Exception:
                        pass
                self._api_client = None

            # Force garbage collection after all cleanup
            gc.collect()
            self.log("Memory cleanup completed.")

        except Exception as e:
            # Catch any unexpected errors in the processing pipeline
            self.logger.exception("Processing job failed with exception")
            logging.exception("Processing failed")
            self.log(f"Error: {e}")
            self.session.failed_items += 1

            # Ensure cleanup even on failure
            if hasattr(self, "model") and self.model:
                self.model = None
            if hasattr(self, "_api_client") and self._api_client:
                if hasattr(self._api_client, "close"):
                    try:
                        self._api_client.close()
                    except Exception:
                        pass
                self._api_client = None
            gc.collect()

    def _fetch_items(self, offset: int = 0):
        """
        Fetch items to process from the configured datasource.

        For Daminion sources the ``offset`` parameter controls which page of
        500 records is returned.  Pass ``offset=0`` for the first page,
        ``offset=500`` for the second, etc.  For local filesystem sources
        this parameter is ignored — all matching files are returned in one
        call.

        Args:
            offset: Starting index for Daminion pagination (default 0).

        Returns:
            list: Items to process.  For local sources: list of Path objects.
                  For Daminion sources: list of item dicts.

        Raises:
            FileNotFoundError: If local path doesn't exist.
            ValueError: If Daminion client is not connected.
        """
        ds = self.session.datasource

        # ================================================================
        # LOCAL FILESYSTEM SOURCE
        # ================================================================
        if ds.type == "local":
            path = Path(ds.local_path)
            if not path.exists():
                raise FileNotFoundError(f"Folder not found: {path}")

            # Define supported image file extensions
            exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

            # Scan directory (recursive or shallow)
            if ds.local_recursive:
                self.logger.info(f"Performing recursive scan of {path}")
                # rglob("*") recursively finds all files in subdirectories
                files = [p for p in path.rglob("*") if p.suffix.lower() in exts]
            else:
                self.logger.info(f"Performing shallow scan of {path}")
                # iterdir() only scans the immediate directory
                files = [p for p in path.iterdir() if p.suffix.lower() in exts]

            self.logger.info(
                f"Found {len(files)} image files in local folder: {path} (recursive={ds.local_recursive})"
            )
            return files

        # ================================================================
        # DAMINION DAM SOURCE
        # ================================================================
        elif ds.type == "daminion":
            # Ensure Daminion client is connected
            if not self.session.daminion_client:
                raise ValueError("Daminion client not connected")

            self.logger.info(
                f"Fetching items from Daminion — Scope: {ds.daminion_scope}, "
                f"Status: {ds.status_filter}, Offset: {offset}"
            )
            self.log("Fetching items from Daminion...")

            # Build list of fields to filter for untagged items
            untagged_fields = []
            if ds.daminion_untagged_keywords:
                untagged_fields.append("Keywords")
            if ds.daminion_untagged_categories:
                untagged_fields.append("Category")
            if ds.daminion_untagged_description:
                untagged_fields.append("Description")

            # Determine maximum items to fetch (0 = unlimited)
            max_to_fetch = ds.max_items if ds.max_items > 0 else None

            # Query Daminion. When offset > 0 we are in a pagination pass
            # and only want the single next page of 500 records.
            items = self.session.daminion_client.get_items_filtered(
                scope=ds.daminion_scope,
                saved_search_id=ds.daminion_saved_search_id or ds.daminion_saved_search,
                collection_id=ds.daminion_collection_id or ds.daminion_catalog_id,
                search_term=ds.daminion_search_term,
                untagged_fields=untagged_fields,
                status_filter=ds.status_filter,
                max_items=max_to_fetch,
                start_index=offset,
            )

            self.logger.info(
                f"Retrieved {len(items)} items from Daminion (offset={offset})"
            )
            self.log(f"Retrieved {len(items)} items from Daminion.")
            return items

        # Unknown datasource type
        return []

    def _init_local_model(self):
        """
        Initialize and load the AI model for local inference.

        This method is only called when using local inference (not API-based).
        It loads the model from Hugging Face's cache into memory and prepares
        it for inference on the selected device (CPU or GPU).

        The method:
        1. Checks model compatibility (rejects GPTQ, AWQ, etc.)
        2. Converts device string ('cpu'/'cuda') to integer format for pipeline
        3. Loads the model using huggingface_utils
        4. Auto-detects and corrects the task if needed
        5. Stores the model in self.model for reuse across all items

        Device mapping:
        - 'cpu' -> -1 (use CPU for inference)
        - 'cuda' -> 0 (use GPU device 0 for inference)

        Raises:
            RuntimeError: If model loading fails or model is incompatible

        Note:
            The model is loaded once and reused for all items in the batch,
            which is much more efficient than loading per-item.
        """
        engine = self.session.engine

        # Check model compatibility before attempting to load
        reason = huggingface_utils.get_local_inference_incompatibility_reason(
            engine.model_id,
            task=engine.task,
        )
        if reason is not None:
            error_msg = (
                f"Cannot load model '{engine.model_id}': {reason}\n\n"
                "This model is not suitable for Synapic's local inference runtime.\n"
                "Please select a different model from the local cache."
            )
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        self.logger.info(f"Initializing local model: {engine.model_id}")
        self.log(f"Loading local model: {engine.model_id}...")

        # Convert device string to integer format expected by transformers pipeline
        # -1 = CPU, 0 = CUDA device 0 (first GPU)
        device_int = -1 if engine.device == "cpu" else 0
        self.logger.info(f"Using device: {engine.device} (device_int={device_int})")

        try:
            # Load model from Hugging Face cache
            # This may download the model if not already cached
            self.model = huggingface_utils.load_model(
                model_id=engine.model_id,
                task=engine.task,
                progress_queue=None,  # No progress tracking for batch load
                device=device_int,
            )

            # Auto-detect actual task from loaded model
            # Some models may have a different task than configured
            # (e.g., VLMs use 'image-text-to-text' instead of 'image-to-text')
            actual_task = getattr(self.model, "task", None)
            if actual_task and actual_task != engine.task:
                self.logger.info(
                    f"Syncing session task from '{engine.task}' to actual pipeline task '{actual_task}'"
                )
                engine.task = actual_task

            self.logger.info(
                f"Local model loaded successfully: {engine.model_id} (Task: {engine.task}, Device: {engine.device})"
            )
            self.log(
                f"Model loaded successfully (Task: {engine.task}, Device: {engine.device})."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def _process_single_item(self, item):
        """
        Process a single image item through the complete AI tagging pipeline.

        This method orchestrates the four-stage processing workflow:
        1. **Image Loading**: Load from local file or download Daminion thumbnail
        2. **AI Inference**: Run the image through the configured AI model
        3. **Tag Extraction**: Parse model output and filter by confidence threshold
        4. **Metadata Writing**: Write tags to EXIF/IPTC or update Daminion

        Args:
            item: Either a Path object (local file) or dict (Daminion item with 'id', 'fileName')

        Processing Flow:
            - Detects item type (local vs Daminion) and loads image accordingly
            - Routes to appropriate inference method (local model vs API)
            - Handles different model types (VLM, captioning, classification, zero-shot)
            - Applies confidence threshold filtering to extracted tags
            - Writes metadata to destination (file or Daminion)
            - Optionally verifies Daminion metadata updates
            - Updates session statistics and results

        Error Handling:
            - Logs detailed error information for debugging
            - Increments failed_items counter
            - Continues processing remaining items (doesn't abort job)
            - Cleans up temporary files even on failure

        Note:
            For Daminion items, thumbnails are downloaded to temp files and
            cleaned up after processing to avoid disk space issues.
        """
        path = None
        is_daminion = isinstance(
            item, dict
        )  # Daminion items are dicts, local items are Path objects
        daminion_client = self.session.daminion_client
        temp_thumb = None  # Track temporary thumbnail file for cleanup

        try:
            engine = self.session.engine

            # ===============================================================
            # STAGE 1: IMAGE LOADING
            # ===============================================================
            # Load the image from either local filesystem or Daminion server
            if is_daminion:
                item_id = item.get("id")
                filename = item.get("fileName") or f"Item {item_id}"
                self.logger.debug(f"Processing Daminion item {item_id}: {filename}")
                self.log(f"Processing Daminion Item: {filename}...")

                # Download image (server-side resized for faster AI inference)
                # Use original at 100%, proportionally scaled preview at lower scales
                ds = self.controller.session.datasource
                scale = getattr(ds, "resize_scale", 100)
                if scale >= 100:
                    path = daminion_client.download_original(item_id)
                    if not path or not path.exists():
                        raise RuntimeError(
                            f"Could not download original for item {item_id}"
                        )
                else:
                    # Get original dimensions first to calculate proportional target size
                    dims = daminion_client.get_item_dimensions(item_id)
                    if dims:
                        orig_w, orig_h = dims
                        target_w = max(75, int(orig_w * scale / 100))
                    else:
                        # Fallback: use scale of a base 2000px size
                        target_w = max(75, int(2000 * scale / 100))
                    path = daminion_client.download_preview(item_id, width=target_w)
                    if not path or not path.exists():
                        raise RuntimeError(
                            f"Could not download preview for item {item_id}"
                        )
            else:
                path = item
                self.logger.debug(f"Processing local file: {path}")
                self.log(f"Processing: {path.name}...")

            # ===============================================================
            # STAGE 2: AI INFERENCE
            # ===============================================================
            # Run the image through the AI model to generate tags
            # The inference method depends on the configured provider:
            # - 'local': Use locally loaded model (self.model)
            # - 'huggingface'/'openrouter': Call API endpoint
            result = None

            if engine.provider == "local":
                # ---------------------------------------------------------------
                # LOCAL INFERENCE (Model loaded in memory)
                # ---------------------------------------------------------------
                # The model was loaded in _init_local_model() and is reused
                # for all items in the batch for efficiency

                if engine.task in [
                    config.MODEL_TASK_IMAGE_TO_TEXT,
                    "image-text-to-text",
                ]:
                    # Image Captioning / Vision-Language Models (VLMs)
                    # Handles both standard captioning (BLIP, GIT) and modern VLMs (Qwen2-VL)
                    with Image.open(path) as img:
                        if img.mode != "RGB":
                            img = img.convert("RGB")

                        # Check if the pipeline is modern image-text-to-text (e.g. Qwen2-VL)
                        # These models expect chat-style messages with structured prompts
                        if (
                            hasattr(self.model, "task")
                            and self.model.task == "image-text-to-text"
                        ):
                            # Build the text instruction (include system prompt if set)
                            system_instruction = (
                                engine.system_prompt.strip()
                                if engine.system_prompt
                                else ""
                            )
                            user_text = (
                                "Analyze the image and return a JSON object with keys: "
                                "'description' (detailed caption), 'category' (single broad category), "
                                "and 'keywords' (list of 5-10 tags). Return ONLY the raw JSON string."
                            )
                            if system_instruction:
                                messages = [
                                    {"role": "system", "content": system_instruction},
                                    {
                                        "role": "user",
                                        "content": [
                                            {"type": "image", "image": img},
                                            {"type": "text", "text": user_text},
                                        ],
                                    },
                                ]
                            else:
                                messages = [
                                    {
                                        "role": "user",
                                        "content": [
                                            {"type": "image", "image": img},
                                            {"type": "text", "text": user_text},
                                        ],
                                    }
                                ]
                            try:
                                # For image-text-to-text pipelines, pass the formatted messages
                                result = self.model(
                                    text=messages,
                                    generate_kwargs={"max_new_tokens": 512},
                                )
                            except Exception as e:
                                self.logger.error(f"VLM inference failed: {e}")
                                raise
                        else:
                            # Standard image-to-text models (BLIP, GIT, etc.)
                            try:
                                result = self.model(
                                    img,
                                    prompt="Describe the image.",
                                    generate_kwargs={"max_new_tokens": 512},
                                )
                            except Exception as e:
                                self.logger.debug(
                                    f"Prompted inference failed ({e}), falling back to simple call."
                                )
                                result = self.model(img)

                elif engine.task == config.MODEL_TASK_ZERO_SHOT:
                    # Zero-Shot Image Classification
                    # Classifies image into one of the provided candidate labels
                    # without requiring training on those specific categories
                    with Image.open(path) as img:
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        result = self.model(
                            img, candidate_labels=config.DEFAULT_CANDIDATE_LABELS
                        )

                else:
                    # Standard Image Classification
                    # Uses pre-trained categories from the model's training
                    with Image.open(path) as img:
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        result = self.model(img)

            elif engine.provider == "groq_package":
                # ---------------------------------------------------------------
                # GROQ SDK INFERENCE (Cloud-based via Groq Python SDK)
                # ---------------------------------------------------------------
                # Uses the reusable Groq client initialized in _run_job()
                groq_client = self._api_client

                # Update API key if it has rotated
                groq_client.api_key = engine.groq_api_key

                # Default model for vision tasks (Groq's vision model)
                model_id = (
                    engine.model_id or "meta-llama/llama-4-scout-17b-16e-instruct"
                )

                # Create a detailed prompt for image analysis
                prompt = (
                    "Analyze this image and provide a detailed response in JSON format with these keys:\n"
                    "- 'description': A detailed description of the image content\n"
                    "- 'category': A single broad category (e.g., 'Nature', 'Architecture', 'People')\n"
                    "- 'keywords': A list of 5-10 relevant tags/keywords\n\n"
                    "Return ONLY the raw JSON object, no additional text."
                )

                # Call Groq API with the image — uses key rotation on quota/rate-limit errors
                self.logger.info("Using Groq API key rotation")
                response_text = groq_client.chat_with_image_rotating(
                    engine_config=engine,
                    model=model_id,
                    prompt=prompt,
                    image_path=str(path),
                )

                if (
                    isinstance(response_text, str)
                    and "Error: All configured Groq API keys have been exhausted"
                    in response_text
                ):
                    self.logger.error(
                        "Groq API key exhaustion reached. Aborting pipeline."
                    )
                    self.log("Groq API quota exhausted. Aborting job.")
                    # Setting stop_event will halt the main fetch loop
                    if hasattr(self, "stop_event") and not self.stop_event.is_set():
                        self.stop_event.set()
                    # Raising RuntimeError breaks out of this specific item correctly marking it failed,
                    # and the loop checks stop_event on the next iteration.
                    raise RuntimeError(
                        "All Groq API keys have been exhausted for this run cycle."
                    )

                # Format result to match expected structure for tag extraction
                result = [{"generated_text": response_text}]
                del response_text  # Free the original string copy

            elif engine.provider == "ollama":
                # ---------------------------------------------------------------
                # OLLAMA INFERENCE (Local or Remote)
                # ---------------------------------------------------------------
                # Uses the reusable Ollama client initialized in _run_job()
                ollama_client = self._api_client

                # Use configured model
                model_id = engine.model_id or "llama3:latest"

                # Create a detailed prompt for image analysis
                prompt = (
                    "Analyze this image and provide a detailed response in "
                    "JSON format with these keys:\n"
                    "- 'description': A detailed description of the image content\n"
                    "- 'category': A single broad category "
                    "(e.g., 'Nature', 'Architecture', 'People')\n"
                    "- 'keywords': A list of 5-10 relevant tags/keywords\n\n"
                    "Return ONLY the raw JSON object, no additional text."
                )

                # Call Ollama with the image path
                response_text = ollama_client.chat_with_image(
                    model_name=model_id, prompt=prompt, image_path=str(path)
                )

                # Format result to match expected structure for tag extraction
                result = [{"generated_text": response_text}]
                del response_text  # Free the original string copy

            elif engine.provider == "nvidia":
                # ---------------------------------------------------------------
                # NVIDIA NIM INFERENCE (Cloud-based via Nvidia Integrate API)
                # ---------------------------------------------------------------
                # Uses the reusable Nvidia client initialized in _run_job()
                nvidia_client = self._api_client

                # Use configured model
                model_id = (
                    engine.model_id or "mistralai/mistral-large-3-675b-instruct-2512"
                )

                # Create a detailed prompt for image analysis
                prompt = (
                    "Analyze this image and provide a detailed response in "
                    "JSON format with these keys:\n"
                    "- 'description': A detailed description of the image content\n"
                    "- 'category': A single broad category "
                    "(e.g., 'Nature', 'Architecture', 'People')\n"
                    "- 'keywords': A list of 5-10 relevant tags/keywords\n\n"
                    "Return ONLY the raw JSON object, no additional text."
                )

                # Call Nvidia NIM with the image path
                response_text = nvidia_client.chat_with_image(
                    model_name=model_id, prompt=prompt, image_path=str(path)
                )

                # Format result to match expected structure for tag extraction
                result = [{"generated_text": response_text}]
                del response_text  # Free the original string copy

            elif engine.provider == "google_ai":
                # ---------------------------------------------------------------
                # GOOGLE AI STUDIO INFERENCE (Cloud-based via Gemini API)
                # ---------------------------------------------------------------
                # Uses the reusable Google AI client initialized in _run_job()
                google_client = self._api_client

                # Use configured model
                model_id = engine.model_id or "gemini-2.5-flash"

                # Create a detailed prompt for image analysis
                prompt = (
                    "Analyze this image and provide a detailed response in "
                    "JSON format with these keys:\n"
                    "- 'description': A detailed description of the image content\n"
                    "- 'category': A single broad category "
                    "(e.g., 'Nature', 'Architecture', 'People')\n"
                    "- 'keywords': A list of 5-10 relevant tags/keywords\n\n"
                    "Return ONLY the raw JSON object, no additional text."
                )

                # Call Google AI with the image path
                response_text = google_client.chat_with_image(
                    model_name=model_id, prompt=prompt, image_path=str(path)
                )

                # Format result to match expected structure for tag extraction
                result = [{"generated_text": response_text}]
                del response_text  # Free the original string copy

            elif engine.provider == "cerebras":
                # ---------------------------------------------------------------
                # CEREBRAS INFERENCE (Cloud-based via Cerebras SDK — world's fastest LLM)
                # ---------------------------------------------------------------
                # Uses the reusable Cerebras client initialized in _run_job()
                cerebras_client = self._api_client

                # Use configured model (default: fast 8B model)
                model_id = engine.model_id or "llama3.1-8b"

                # Create a detailed prompt for image analysis
                prompt = (
                    "Analyze this image and provide a detailed response in "
                    "JSON format with these keys:\n"
                    "- 'description': A detailed description of the image content\n"
                    "- 'category': A single broad category "
                    "(e.g., 'Nature', 'Architecture', 'People')\n"
                    "- 'keywords': A list of 5-10 relevant tags/keywords\n\n"
                    "Return ONLY the raw JSON object, no additional text."
                )

                # Call Cerebras with the image path
                response_text = cerebras_client.chat_with_image(
                    model_name=model_id, prompt=prompt, image_path=str(path)
                )

                # Format result to match expected structure for tag extraction
                result = [{"generated_text": response_text}]
                del response_text  # Free the original string copy

            elif engine.provider in ["huggingface", "openrouter"]:
                # ---------------------------------------------------------------
                # API INFERENCE (Cloud-based)
                # ---------------------------------------------------------------
                # Send image to API endpoint for processing
                # No local model loading required
                provider_module = (
                    huggingface_utils
                    if engine.provider == "huggingface"
                    else openrouter_utils
                )

                # Configure inference parameters
                params = {"max_new_tokens": 1024}
                if engine.task == config.MODEL_TASK_ZERO_SHOT:
                    params["candidate_labels"] = config.DEFAULT_CANDIDATE_LABELS

                result = provider_module.run_inference_api(
                    model_id=engine.model_id,
                    image_path=str(path),
                    task=engine.task,
                    token=engine.api_key,
                    parameters=params,
                )

            # ===============================================================
            # STAGE 3: TAG EXTRACTION
            # ===============================================================
            # Parse the model's output and extract structured metadata
            # The extraction logic handles different output formats:
            # - JSON objects (from VLMs)
            # - Classification results with scores
            # - Plain text descriptions

            # Convert threshold from UI scale (1-100) to model scale (0.0-1.0)
            # Tags with confidence scores below this threshold are filtered out
            threshold = engine.confidence_threshold / 100.0

            # Extract category, keywords, and description from model result
            # The extract_tags_from_result function handles:
            # - Parsing JSON from VLM responses
            # - Filtering classification results by threshold
            # - Extracting top predictions as keywords
            cat, kws, desc = image_processing.extract_tags_from_result(
                result, engine.task, threshold=threshold
            )
            self.logger.debug(
                f"Extracted tags - Category: {cat}, Keywords: {len(kws)}, Description length: {len(desc) if desc else 0}"
            )

            # Free the (potentially large) model result now that tags are extracted
            del result

            # If extraction returned no useful data, write a placeholder so the item
            # is marked as processed and won't be reprocessed in subsequent runs
            if not cat and not kws and not desc:
                desc = "[AI: No Result]"
                self.logger.info(
                    f"No tags extracted for item, using placeholder: {desc}"
                )
                self.log(f"No results - marking with placeholder")

            # ===============================================================
            # STAGE 4: METADATA WRITING
            # ===============================================================
            # Write the extracted tags to the appropriate destination:
            # - Daminion: Update item metadata via API
            # - Local: Write to EXIF/IPTC metadata in image file

            if is_daminion:
                # Update Daminion item metadata via API
                # This sends the tags to the Daminion server for storage
                success = daminion_client.update_item_metadata(
                    item_id=item_id, category=cat, keywords=kws, description=desc
                )

                # Optional: Verify that the metadata was actually written
                # This is useful for debugging API issues or data corruption
                if success and verifier:
                    self.logger.info(
                        f"Verifying metadata for Daminion item {item_id}..."
                    )
                    verified = verifier.verify_metadata_update(
                        client=daminion_client,
                        item_id=item_id,
                        expected_cat=cat,
                        expected_kws=kws,
                        expected_desc=desc,
                    )
                    if verified:
                        self.logger.info(
                            f"Metadata verification successful for item {item_id}"
                        )
                        self.log(f"Verification: Passed")
                    else:
                        self.logger.warning(
                            f"Metadata verification failed for item {item_id}"
                        )
                        self.log(f"Verification: FAILED (Check details in log file)")
                        # We don't fail the whole item if verification fails,
                        # just log it as a warning for manual review

            else:
                # Write metadata to local image file (EXIF/IPTC)
                # This embeds the tags directly in the image file
                success = image_processing.write_metadata(
                    image_path=path, category=cat, keywords=kws, description=desc
                )

            # ===============================================================
            # RESULT TRACKING
            # ===============================================================
            # Log the processing result and add to session results
            status = "Success" if success else "Write Failed"
            tags_str = f"Cat: {cat}, Kws: {len(kws)}, Desc: {desc[:20]}..."
            self.logger.info(
                f"Item processed successfully - Status: {status}, Tags: {tags_str}"
            )
            self.log(f"Result: {tags_str}")

            # Store result for export/review in Step 4
            self.session.results.append(
                {
                    "filename": filename if is_daminion else path.name,
                    "status": status,
                    "tags": tags_str,
                }
            )

        except Exception as e:
            # ===============================================================
            # ERROR HANDLING
            # ===============================================================
            # Log detailed error information for debugging
            # The job continues processing remaining items even if one fails
            name = (
                item.get("fileName")
                if is_daminion
                else (item.name if isinstance(item, Path) else str(item))
            )
            self.logger.error(
                f"Failed to process item '{name}': {type(e).__name__}: {str(e)}"
            )
            self.logger.exception("Full traceback:")
            logging.error(f"Failed to process {name}: {e}")

            # Update failure statistics
            self.session.failed_items += 1
            self.log(f"Failed: {e}")

        finally:
            # ===============================================================
            # CLEANUP
            # ===============================================================
            # Always clean up temporary files, even if processing failed
            # This prevents disk space issues when processing large batches
            if temp_thumb and temp_thumb.exists():
                try:
                    import os

                    os.remove(temp_thumb)
                    self.logger.debug(f"Cleaned up temporary thumbnail: {temp_thumb}")
                except Exception:
                    # Ignore cleanup errors - not critical
                    pass

            # Periodic garbage collection to free any residual base64 strings,
            # API response objects, and other short-lived allocations.
            # Every 3 items balances GC overhead with memory pressure.
            if hasattr(self, "session") and self.session.processed_items % 3 == 0:
                gc.collect()
