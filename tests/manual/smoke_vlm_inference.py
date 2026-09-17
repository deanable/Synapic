"""One-off end-to-end smoke test on transformers 5.x.

Loads LiquidAI/LFM2.5-VL-450M through Synapic's real code path
(huggingface_utils.load_model_sync) and runs the exact inference call
processing.py uses for image-text-to-text pipelines on a synthetic image.
"""

import json
import logging
import queue
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from PIL import Image, ImageDraw

from src.core import huggingface_utils

MODEL_ID = "LiquidAI/LFM2.5-VL-450M"


def make_test_image(path):
    """Synthetic image with unambiguous visual content: blue sky, green ground, red sun."""
    img = Image.new("RGB", (320, 240), "#4AA3DF")  # sky
    d = ImageDraw.Draw(img)
    d.rectangle([0, 170, 320, 240], fill="#3F7D3A")  # ground
    d.ellipse([240, 20, 300, 80], fill="#E23B2E")  # sun
    d.polygon([(60, 170), (110, 90), (160, 170)], fill="#7A7A7A")  # mountain
    img.save(path)
    return path


def main():
    q = queue.Queue()

    print(f"[1/3] load_model_sync({MODEL_ID!r}) — downloads on first run (860 MB)...", flush=True)
    model = huggingface_utils.load_model(
        MODEL_ID, task="image-to-text", progress_queue=q, device=-1
    )
    # drain queue for the log
    while True:
        try:
            _type, _data = q.get_nowait()
        except queue.Empty:
            break
    print(f"      pipeline task = {getattr(model, 'task', '?')}", flush=True)

    image_path = make_test_image("_smoke_test_image.png")

    print("[2/3] Running production-style inference (chat messages, as processing.py does)...", flush=True)
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {
                        "type": "text",
                        "text": "Describe this image in one sentence.",
                    },
                ],
            }
        ]
        result = model(text=messages, generate_kwargs={"max_new_tokens": 128})

    print("[3/3] Raw pipeline output:")
    print(json.dumps(result, indent=2, default=str))

    text = result[0]["generated_text"] if isinstance(result, list) else result
    if isinstance(text, list):  # chat format: list of message dicts
        text = text[-1]["content"]
    print("\nGenerated text:", str(text)[:300])
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    sys.exit(main())
