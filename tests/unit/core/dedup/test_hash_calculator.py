import pytest
import base64
from io import BytesIO
from PIL import Image
from src.core.dedup.hash_calculator import ImageHashCalculator

@pytest.fixture
def calculator():
    return ImageHashCalculator()

@pytest.fixture
def sample_image_base64():
    # Generate a 10x10 pixel image
    img = Image.new('RGB', (10, 10), color='red')
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def test_load_image_from_base64_pure(calculator, sample_image_base64):
    """Test loading a pure base64 string without data URI prefix."""
    img = calculator.load_image_from_base64(sample_image_base64)
    assert isinstance(img, Image.Image)
    assert img.size == (10, 10)

def test_load_image_from_base64_with_prefix(calculator, sample_image_base64):
    """Test loading a base64 string that includes a data URI prefix."""
    prefixed_b64 = f"data:image/png;base64,{sample_image_base64}"
    img = calculator.load_image_from_base64(prefixed_b64)
    assert isinstance(img, Image.Image)
    assert img.size == (10, 10)
