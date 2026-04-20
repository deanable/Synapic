import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from PIL import Image

from src.core.upscaler import Swin2SRUpscaler


class _FakeProcessor:
    def __call__(self, images, return_tensors="pt"):
        return {"pixel_values": torch.zeros((1, 3, 2, 2), dtype=torch.float32)}


class _FakeModel:
    def __call__(self, **kwargs):
        class _Output:
            reconstruction = torch.ones((1, 3, 4, 4), dtype=torch.float32) * 0.5

        return _Output()


class TestSwin2SRUpscaler(unittest.TestCase):
    def test_upscale_generates_output_for_input_without_extension(self):
        with TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "sample_input"
            with Image.new("RGBA", (2, 2), (255, 0, 0, 128)) as img:
                img.save(source_path, format="PNG")

            upscaler = Swin2SRUpscaler()
            with patch.object(
                upscaler,
                "_load_model",
                return_value=(_FakeProcessor(), _FakeModel()),
            ):
                output_path = upscaler.upscale(source_path, factor=2)

            self.assertTrue(output_path.exists())
            self.assertEqual(output_path.suffix, ".png")

            with Image.open(output_path) as result:
                self.assertEqual(result.size, (4, 4))
                self.assertEqual(result.mode, "RGBA")

    def test_unsupported_factor_raises(self):
        upscaler = Swin2SRUpscaler()
        with self.assertRaises(ValueError):
            upscaler._load_model(3, status_callback=None)


if __name__ == "__main__":
    unittest.main()
