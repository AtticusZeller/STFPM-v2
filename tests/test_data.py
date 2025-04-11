import math
from pathlib import Path
from typing import get_args

import pytest
import torch
from PIL import Image
from torchvision.transforms import v2

from expt.config import AssetType
from expt.data.dataset import DataModule, InsPLADDataset


@pytest.fixture
def dataset_dir() -> Path:
    """Path to the dataset directory."""
    return Path(__file__).parents[1] / "datasets"


@pytest.mark.slow
@pytest.mark.parametrize("asset_type", list(get_args(AssetType)))
def test_insplad_dataset_train(dataset_dir: Path, asset_type: AssetType) -> None:
    """Test the InsPLAD dataset with training data for all asset types."""
    # Arrange
    dataset = InsPLADDataset(root=dataset_dir, asset_type=asset_type, train=True)

    # Act & Assert
    assert len(dataset) > 0

    assert dataset.asset_type == asset_type

    assert dataset.classes == ["good"]
    assert dataset.class_to_idx == {"good": 0}

    # __getitem__
    for img, label in dataset:
        assert isinstance(img, Image.Image)
        assert label == 0  # All train samples should be normal

    # Check that all sample paths are from 'good' directory
    for path in dataset._image_paths:
        assert "good" in str(path)


@pytest.mark.slow
@pytest.mark.parametrize("asset_type", list(get_args(AssetType)))
def test_insplad_dataset_test(dataset_dir: Path, asset_type: AssetType) -> None:
    """Test the InsPLAD dataset with test data for all asset types."""
    # Arrange
    dataset = InsPLADDataset(root=dataset_dir, asset_type=asset_type, train=False)

    # Act & Assert
    assert len(dataset.classes) > 0

    assert "good" in dataset.classes
    assert dataset.class_to_idx["good"] == 0

    # Verify any anomaly classes have index 1
    for _cls, idx in dataset.class_to_idx.items():
        if _cls != "good":
            assert idx == 1

    # __getitem__
    for i, (img, label) in enumerate(dataset):
        assert isinstance(img, Image.Image)
        # Check that all sample paths are from 'good' or anomaly directories
        if "good" in str(dataset._image_paths[i]):
            assert label == 0
        else:
            assert label == 1


@pytest.mark.parametrize("asset_type", list(get_args(AssetType)))
def test_transform_correctly_applied(dataset_dir: Path, asset_type: AssetType) -> None:
    """Test that transforms are correctly applied to dataset images."""
    # Arrange
    dataset = InsPLADDataset(root=dataset_dir, asset_type=asset_type, train=False)

    # Create a simple transform to tensor
    to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    dataset = InsPLADDataset(
        root=dataset_dir, asset_type=asset_type, train=True, transform=to_tensor
    )

    # Act
    img, _ = dataset[0]

    # Assert
    assert torch.is_tensor(img)
    assert img.dtype == torch.float32
    assert img.min() >= 0.0
    assert img.max() <= 1.0
    assert len(img.shape) == 3  # [H, W, C] or [C, H, W]


@pytest.mark.parametrize("asset_type", list(get_args(AssetType)))
def test_data_module_setup(dataset_dir: Path, asset_type: AssetType) -> None:
    """Test DataModule correctly splits training data."""
    # Arrange
    data_module = DataModule(
        data_dir=dataset_dir, asset_type=asset_type, batch_size=32, val_split=0.2
    )

    # Act
    data_module.setup("fit")

    # Assert - check correct train/val split
    total_samples = len(data_module.train_dataset) + len(data_module.val_dataset)
    expected_train_size = int(total_samples * 0.8)
    expected_val_size = total_samples - expected_train_size

    assert math.isclose(
        len(data_module.train_dataset), expected_train_size, abs_tol=1, rel_tol=0
    )
    assert math.isclose(
        len(data_module.val_dataset), expected_val_size, abs_tol=1, rel_tol=0
    )


@pytest.mark.parametrize("asset_type", list(get_args(AssetType)))
def test_data_module_loaders(dataset_dir: Path, asset_type: AssetType) -> None:
    """Test DataModule dataloaders return correctly formatted batches."""
    # Arrange
    batch_size = 32
    data_module = DataModule(
        data_dir=dataset_dir,
        asset_type=asset_type,
        batch_size=batch_size,
        val_split=0.2,
    )

    # Setup both fit and test stages
    data_module.setup("fit")
    data_module.setup("test")

    train_loader = data_module.train_dataloader()
    test_loader = data_module.test_dataloader()
    val_loader = data_module.val_dataloader()

    # Act & Assert - verify dataloader formats
    assert len(train_loader) > 0
    assert len(test_loader) > 0
    assert len(val_loader) > 0


def test_invalid_dataset_path() -> None:
    """Test error handling for invalid dataset paths."""
    with pytest.raises(RuntimeError, match="Dataset not found"):
        InsPLADDataset(
            root="/path/does/not/exist", asset_type=get_args(AssetType)[0], train=True
        )
