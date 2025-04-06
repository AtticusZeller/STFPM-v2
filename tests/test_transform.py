import numpy as np
import pytest
import torch
from PIL import Image

from expt.data.transform import base_transform, imagenet_transform


@pytest.fixture
def sample_pil_image() -> Image.Image:
    """Create a sample PIL image for testing transforms."""
    # Create a simple RGB PIL image (100x100)
    array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    return Image.fromarray(array)


def test_base_transform_pil_input(sample_pil_image: Image.Image) -> None:
    """Test that base_transform handles PIL images correctly."""

    # Arrange
    transform = base_transform()
    # Act
    output = transform(sample_pil_image)

    # Check output type and shape
    assert torch.is_tensor(output)
    assert output.dtype == torch.float32

    # Channel first
    assert output.shape == (3, 100, 100)

    # Check scaling
    assert output.min() >= 0.0
    assert output.max() <= 1.0


def test_imagenet_transform_pil_input(sample_pil_image: Image.Image) -> None:
    """Test that imagenet_transform handles PIL images correctly."""
    # Arrange
    transform = imagenet_transform()
    # Act
    output = transform(sample_pil_image)

    # Check output type and shape (should be resized to 256x256)
    assert torch.is_tensor(output)
    assert output.dtype == torch.float32
    assert output.shape == (3, 256, 256)

    # Verify normalization was applied
    assert (output.min() < 0.0) and (output.max() > 1.0)


@pytest.mark.parametrize("size", [(50, 75), (200, 150), (300, 300)])
def test_imagenet_transform_different_input_sizes(size) -> None:
    """Test that imagenet_transform handles different input sizes correctly."""
    # Arrange
    transform = imagenet_transform()

    # Create random image with specified size
    array = np.random.randint(0, 256, (size[0], size[1], 3), dtype=np.uint8)
    test_img = Image.fromarray(array)  # Convert to PIL image

    # Act
    output = transform(test_img)

    # Output should always be 256x256
    assert output.shape == torch.Size([3, 256, 256])
