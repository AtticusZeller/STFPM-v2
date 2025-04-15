from torchvision.transforms import v2 as v2

from expt.config import AssetType, TransformT
from expt.data.transform import (
    base_transform,
    resnet18_256_transform,
    resnet18_transform,
)

from .dataset import DataModule

__all__ = ["create_data_module"]


def create_data_module(
    asset_type: AssetType,
    name: str = "InsPLAD",
    batch_size: int = 32,
    transform: TransformT = "resnet18",
) -> DataModule:
    match transform:
        case "resnet18":
            return DataModule(
                "datasets",
                asset_type,
                batch_size,
                train_transform=resnet18_transform(True),
                test_transform=resnet18_transform(False),
            )
        case "resnet18_256":
            return DataModule(
                "datasets",
                asset_type,
                batch_size,
                train_transform=resnet18_256_transform(True),
                test_transform=resnet18_256_transform(False),
            )
        case "base":
            return DataModule(
                "datasets", asset_type, batch_size, train_transform=base_transform()
            )
        case _:
            raise ValueError(f"Unknown transform type of {name} dataset: {transform}")
