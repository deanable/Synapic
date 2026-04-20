"""
Image upscaling utilities powered by Swin2SR models from Hugging Face.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

from PIL import Image, ImageFilter

if TYPE_CHECKING:
    import torch
    from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

StatusCallback = Optional[Callable[[str], None]]

WORKFLOW_QUALITY = "quality"
WORKFLOW_BALANCED = "balanced"
WORKFLOW_FAST = "fast"

_WORKFLOW_MODEL_IDS: Dict[str, Dict[int, str]] = {
    WORKFLOW_QUALITY: {
        2: "caidas/swin2SR-classical-sr-x2-64",
        4: "caidas/swin2SR-classical-sr-x4-64",
    },
    WORKFLOW_BALANCED: {
        4: "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
    },
}

_FORMAT_TO_EXT = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
    "TIFF": ".tif",
    "BMP": ".bmp",
}


@dataclass
class UpscaleOptions:
    workflow: str = WORKFLOW_QUALITY
    precision: str = "auto"  # auto | fp16 | fp32
    denoise_strength: float = 1.0  # Used by balanced workflow (0.0..1.0)
    sharpen_amount: float = 0.0  # 0.0 = off
    output_format: str = "keep"  # keep | JPEG | PNG | WEBP
    jpeg_quality: int = 95
    overwrite_existing: bool = True


class Swin2SRUpscaler:
    """Thread-safe wrapper around Swin2SR super-resolution models."""

    def __init__(self) -> None:
        import torch

        self._torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = Lock()
        self._loaded: Dict[Tuple[str, int], Tuple[Any, Any]] = {}

    def _emit(self, callback: StatusCallback, message: str) -> None:
        if callback:
            callback(message)

    def _load_model(
        self,
        workflow_or_factor: Any,
        factor: Optional[int] = None,
        status_callback: StatusCallback = None,
    ) -> Tuple[Any, Any]:
        # Backward compatibility: _load_model(factor, status_callback=...)
        if isinstance(workflow_or_factor, int) and factor is None:
            workflow = WORKFLOW_QUALITY
            factor = workflow_or_factor
        else:
            workflow = str(workflow_or_factor)

        if factor is None:
            raise ValueError("Upscale factor must be provided.")

        workflow_models = _WORKFLOW_MODEL_IDS.get(workflow)
        if workflow_models is None:
            raise ValueError(f"Unsupported upscale workflow: {workflow}")
        if factor not in workflow_models:
            raise ValueError(
                f"Unsupported factor {factor}x for workflow '{workflow}'."
            )

        with self._lock:
            cache_key = (workflow, factor)
            cached = self._loaded.get(cache_key)
            if cached is not None:
                return cached

            model_id = workflow_models[factor]
            self._emit(
                status_callback,
                f"Loading {workflow} super-resolution model ({factor}x): {model_id}",
            )
            from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

            processor = AutoImageProcessor.from_pretrained(model_id)
            model = Swin2SRForImageSuperResolution.from_pretrained(model_id)
            model.to(self.device)
            model.eval()

            self._loaded[cache_key] = (processor, model)
            return processor, model

    def upscale(
        self,
        input_path: Path,
        factor: int,
        output_path: Optional[Path] = None,
        status_callback: StatusCallback = None,
        options: Optional[UpscaleOptions] = None,
        workflow: Optional[str] = None,
    ) -> Path:
        """
        Upscale an image using Swin2SR and return the saved output path.
        """
        resolved_options = options or UpscaleOptions()
        if workflow is not None:
            resolved_options.workflow = workflow

        if resolved_options.workflow == WORKFLOW_FAST:
            return self._upscale_fast(
                input_path=input_path,
                factor=factor,
                output_path=output_path,
                options=resolved_options,
                status_callback=status_callback,
            )

        model_factor = factor
        if resolved_options.workflow == WORKFLOW_BALANCED and factor == 2:
            # Real-world model is 4x; run 4x then downsample to requested size.
            model_factor = 4

        processor, model = self._load_model(
            resolved_options.workflow, model_factor, status_callback
        )

        with Image.open(input_path) as source_image:
            source_mode = source_image.mode
            source_format = source_image.format
            rgb_image = source_image.convert("RGB")
            alpha_channel = source_image.getchannel("A") if "A" in source_mode else None

        self._emit(
            status_callback,
            f"Running {resolved_options.workflow} AI upscaling ({factor}x)...",
        )
        model_inputs = processor(images=rgb_image, return_tensors="pt")
        model_inputs = {
            key: value.to(self.device) for key, value in model_inputs.items()
        }

        with self._torch.inference_mode():
            if (
                resolved_options.precision in {"auto", "fp16"}
                and self.device.type == "cuda"
            ):
                with self._torch.autocast(device_type="cuda", dtype=self._torch.float16):
                    output = model(**model_inputs).reconstruction
            else:
                output = model(**model_inputs).reconstruction

        output = output.squeeze(0).clamp(0, 1).cpu()
        output_image = self._tensor_to_pil(output)

        # Balanced workflow can blend AI output with smoother interpolation for denoise control.
        if resolved_options.workflow == WORKFLOW_BALANCED:
            target_size = (
                int(rgb_image.width * factor),
                int(rgb_image.height * factor),
            )
            if output_image.size != target_size:
                output_image = output_image.resize(target_size, Image.Resampling.LANCZOS)
            if resolved_options.denoise_strength < 1.0:
                lanczos_ref = rgb_image.resize(target_size, Image.Resampling.LANCZOS)
                alpha = max(0.0, min(1.0, resolved_options.denoise_strength))
                output_image = Image.blend(lanczos_ref, output_image, alpha=alpha)

        if alpha_channel is not None:
            upscaled_alpha = alpha_channel.resize(
                output_image.size, Image.Resampling.LANCZOS
            )
            output_image = output_image.convert("RGBA")
            output_image.putalpha(upscaled_alpha)

        output_image = self._apply_sharpen(output_image, resolved_options.sharpen_amount)
        output_path, save_kwargs = self._resolve_output_target(
            input_path=input_path,
            output_path=output_path,
            source_format=source_format,
            options=resolved_options,
        )

        output_image.save(output_path, **save_kwargs)
        return output_path

    def _upscale_fast(
        self,
        input_path: Path,
        factor: int,
        output_path: Optional[Path],
        options: UpscaleOptions,
        status_callback: StatusCallback,
    ) -> Path:
        self._emit(status_callback, f"Running fast Lanczos upscale ({factor}x)...")
        with Image.open(input_path) as source_image:
            source_format = source_image.format
            new_size = (
                int(source_image.width * factor),
                int(source_image.height * factor),
            )
            output_image = source_image.resize(new_size, Image.Resampling.LANCZOS)
            output_image = self._apply_sharpen(output_image, options.sharpen_amount)

        output_path, save_kwargs = self._resolve_output_target(
            input_path=input_path,
            output_path=output_path,
            source_format=source_format,
            options=options,
        )
        output_image.save(output_path, **save_kwargs)
        return output_path

    def _apply_sharpen(self, image: Image.Image, amount: float) -> Image.Image:
        if amount <= 0:
            return image
        percent = int(max(20, min(300, amount * 100)))
        return image.filter(
            ImageFilter.UnsharpMask(radius=1.6, percent=percent, threshold=2)
        )

    def _resolve_output_target(
        self,
        input_path: Path,
        output_path: Optional[Path],
        source_format: Optional[str],
        options: UpscaleOptions,
    ) -> Tuple[Path, Dict[str, Any]]:
        if output_path is None:
            if options.output_format.lower() == "keep":
                ext = input_path.suffix.lower() or _FORMAT_TO_EXT.get(source_format or "", ".png")
            else:
                ext = _FORMAT_TO_EXT.get(options.output_format.upper(), ".png")

            output_path = input_path.with_name(f"{input_path.stem}_upscaled{ext}")
            if not options.overwrite_existing and output_path.exists():
                output_path = self._next_available_path(output_path)

        save_kwargs: Dict[str, Any] = {}
        final_fmt = options.output_format.upper()
        if options.output_format.lower() == "keep":
            final_fmt = source_format or ""

        if final_fmt == "JPEG":
            save_kwargs["quality"] = max(70, min(100, int(options.jpeg_quality)))
            save_kwargs["optimize"] = True
        elif final_fmt == "WEBP":
            save_kwargs["quality"] = max(70, min(100, int(options.jpeg_quality)))
            save_kwargs["method"] = 6
        return output_path, save_kwargs

    def _next_available_path(self, output_path: Path) -> Path:
        base = output_path.with_suffix("")
        ext = output_path.suffix
        idx = 1
        candidate = output_path
        while candidate.exists():
            candidate = Path(f"{base}_{idx}{ext}")
            idx += 1
        return candidate

    def _tensor_to_pil(self, tensor: Any) -> Image.Image:
        """
        Convert a CHW float tensor in [0, 1] to a PIL image without torchvision.
        """
        array = (
            tensor.mul(255)
            .round()
            .to(dtype=self._torch.uint8)
            .permute(1, 2, 0)
            .numpy()
        )
        if array.shape[2] == 1:
            return Image.fromarray(array[:, :, 0], mode="L")
        return Image.fromarray(array)
