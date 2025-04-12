import torch
import torchvision.transforms.v2 as v2
from torchvision.transforms import InterpolationMode


def base_transform() -> v2.Compose:
    return v2.Compose(
        [
            # Convert input to standard format (CHW)
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )


def imagenet_transform() -> v2.Compose:
    return v2.Compose(
        [
            # Convert input to standard format (CHW)
            v2.ToImage(),
            # Resize to 256x256 with best quality
            # TODO: check shape?
            v2.Resize(
                (256, 256), interpolation=InterpolationMode.BICUBIC, antialias=True
            ),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
