from typing import Literal

from torchvision.transforms import v2 as v2

from .dataset import DataModule

# from .transform import (
#     base_transform,
#     efficientnetv2_pt_transform,
#     resnet_pt_transform,
#     standardize_transform,
# )

__all__ = ["create_data_module"]


def create_data_module(
    name: str = "mnist",
    batch_size: int = 32,
    transform: Literal[
        "standardize", "base", "resnet_pt", "efficientnetv2_pt"
    ] = "standardize",
) -> DataModule:
    pass
