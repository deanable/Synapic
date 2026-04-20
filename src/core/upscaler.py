"""
Image upscaling utilities powered by Swin2SR models from Hugging Face.
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

from PIL import Image

if TYPE_CHECKING:
    import torch
    from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

StatusCallback = Optional[Callable[[str], None]]

_MODEL_IDS: Dict[int, str] = {
    2: "caidas/swin2SR-classical-sr-x2-64",
    4: "caidas/swin2SR-classical-sr-x4-64",
}

_FORMAT_TO_EXT = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
    "TIFF": ".tif",
    "BMP": ".bmp",
}


class Swin2SRUpscaler:
    """Thread-safe wrapper around Swin2SR super-resolution models."""

    def __init__(self) -> None:
        import torch

        self._torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = Lock()
        self._loaded: Dict[int, Tuple[Any, Any]] = {}

    def _emit(self, callback: StatusCallback, message: str) -> None:
        if callback:
            callback(message)

    def _load_model(
        self, factor: int, status_callback: StatusCallback
    ) -> Tuple[Any, Any]:
        if factor not in _MODEL_IDS:
            raise ValueError(f"Unsupported upscale factor: {factor}. Use 2 or 4.")

        with self._lock:
            cached = self._loaded.get(factor)
            if cached is not None:
                return cached

            model_id = _MODEL_IDS[factor]
            self._emit(
                status_callback,
                f"Loading super-resolution model ({factor}x): {model_id}",
            )
            from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

            processor = AutoImageProcessor.from_pretrained(model_id)
            model = Swin2SRForImageSuperResolution.from_pretrained(model_id)
            model.to(self.device)
            model.eval()

            self._loaded[factor] = (processor, model)
            return processor, model

    def upscale(
        self,
        input_path: Path,
        factor: int,
        output_path: Optional[Path] = None,
        status_callback: StatusCallback = None,
    ) -> Path:
        """
        Upscale an image using Swin2SR and return the saved output path.
        """
        processor, model = self._load_model(factor, status_callback)

        with Image.open(input_path) as source_image:
            source_mode = source_image.mode
            source_format = source_image.format
            rgb_image = source_image.convert("RGB")
            alpha_channel = source_image.getchannel("A") if "A" in source_mode else None

        self._emit(status_callback, f"Running AI upscaling ({factor}x)...")
        model_inputs = processor(images=rgb_image, return_tensors="pt")
        model_inputs = {
            key: value.to(self.device) for key, value in model_inputs.items()
        }

        with self._torch.inference_mode():
            output = model(**model_inputs).reconstruction

        output = output.squeeze(0).clamp(0, 1).cpu()
        output_image = self._tensor_to_pil(output)

        if alpha_channel is not None:
            upscaled_alpha = alpha_channel.resize(
                output_image.size, Image.Resampling.LANCZOS
            )
            output_image = output_image.convert("RGBA")
            output_image.putalpha(upscaled_alpha)

        if output_path is None:
            ext = input_path.suffix.lower()
            if not ext:
                ext = _FORMAT_TO_EXT.get(source_format or "", ".png")
            output_path = input_path.with_name(f"{input_path.stem}_upscaled{ext}")

        output_image.save(output_path)
        return output_path

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
