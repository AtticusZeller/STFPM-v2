from collections.abc import Callable
from pathlib import Path
from typing import Any
import lightning as L
import torch
from PIL import Image
from torchvision.datasets import VisionDataset


class InsPLADDataset(VisionDataset):
    """InsPLAD Unsupervised Anomaly Detection Dataset

    This dataset is designed for anomaly detection on power line assets from drone images.
    Training data contains only normal ("good") samples, while test data contains both
    normal and anomalous samples.
    """

    # These values should be calculated from your dataset or use standard ImageNet values
    mean = (0.485, 0.456, 0.406)  # RGB channels - use actual values if avexptle
    std = (0.229, 0.224, 0.225)  # RGB channels - use actual values if avexptle

    def __init__(
        self,
        root: str | Path,
        asset_type: str,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        """
        Args:
            root: Root directory of the InsPLAD dataset
            asset_type: Type of asset to use (e.g., 'glass-insulator', 'damper-stockbridge')
            train: If True, creates dataset from training folder, otherwise from test folder
            transform: A function/transform to apply to the images
            target_transform: A function/transform to apply to the targets/labels
        """
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.root = Path(self.root)
        self.asset_type = asset_type
        self.train = train

        # Set up paths
        self.asset_dir = self.root / asset_type
        self.split_dir = self.asset_dir / ("train" if train else "test")

        # Load data paths and labels
        self._load_data_paths()

        # Define class mapping: normal vs anomalous
        self.class_to_idx = {"good": 0}
        self.classes = ["good"]

        # If test set, add anomaly classes
        if not train:
            anomaly_classes = [
                d.name
                for d in self.split_dir.iterdir()
                if d.is_dir() and d.name != "good"
            ]
            for i, _cls in enumerate(anomaly_classes, 1):
                self.class_to_idx[_cls] = i
                self.classes.append(_cls)

    def _load_data_paths(self) -> None:
        """Load image paths and corresponding labels"""
        self.image_paths = []
        self.targets = []

        # Load all directories in split_dir (train/good or test/good, test/rust, etc.)
        for class_dir in self.split_dir.iterdir():
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            class_idx = (
                0 if class_name == "good" else 1
            )  # Binary classification: 0=normal, 1=anomaly

            # Load all images in the class directory
            for img_path in class_dir.glob("*.jpg"):
                self.image_paths.append(img_path)
                self.targets.append(class_idx)

            # Also check for png, jpeg files
            for img_path in class_dir.glob("*.png"):
                self.image_paths.append(img_path)
                self.targets.append(class_idx)

            for img_path in class_dir.glob("*.jpeg"):
                self.image_paths.append(img_path)
                self.targets.append(class_idx)

        # Convert targets to tensor
        self.targets = torch.tensor(self.targets)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        """
        Args:
            index: Index of the data point

        Returns:
            tuple: (image, target) where target is index of the target class
                   (0 for normal, 1 for anomaly in binary classification)
        """
        img_path = self.image_paths[index]
        target = int(self.targets[index])

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Apply transforms if specified
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        """Return the number of images in the dataset"""
        return len(self.image_paths)


class InsPLADLightningDataModule(L.LightningDataModule):
    """Lightning DataModule for InsPLAD dataset"""

    def __init__(
        self,
        data_dir: str | Path,
        asset_type: str,
        batch_size: int = 32,
        train_transform: Callable | None = None,
        val_transform: Callable | None = None,
        test_transform: Callable | None = None,
        val_split: float = 0.2,
        num_workers: int = 8,
    ) -> None:
        """
        Args:
            data_dir: Root directory of InsPLAD dataset
            asset_type: Type of asset to use (e.g., 'glass-insulator')
            batch_size: Size of the batches
            train_transform: Transformations to apply to training data
            val_transform: Transformations to apply to validation data
            test_transform: Transformations to apply to test data
            val_split: Fraction of training data to use for validation
            num_workers: Number of workers for data loading
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.asset_type = asset_type
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.val_transform = val_transform or train_transform
        self.test_transform = test_transform or val_transform
        self.val_split = val_split
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        """Setup train, validation and test datasets"""
        if stage == "fit" or stage is None:
            # Create training dataset
            train_dataset = InsPLADDataset(
                root=self.data_dir,
                asset_type=self.asset_type,
                train=True,
                transform=self.train_transform,
            )

            # Split training dataset into train and validation
            dataset_size = len(train_dataset)
            val_size = int(dataset_size * self.val_split)
            train_size = dataset_size - val_size

            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )

            # Apply correct transforms to validation set
            if self.val_transform != self.train_transform:
                self.val_dataset.dataset.transform = self.val_transform

        if stage == "test" or stage is None:
            # Create test dataset
            self.test_dataset = InsPLADDataset(
                root=self.data_dir,
                asset_type=self.asset_type,
                train=False,
                transform=self.test_transform,
            )

    def train_dataloader(self):
        """Return training dataloader"""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        """Return validation dataloader"""
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        """Return test dataloader"""
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
