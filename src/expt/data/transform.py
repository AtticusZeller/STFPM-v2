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


def resnet18_transform(train: bool) -> v2.Compose:
    # Base transforms for both training and inference
    base_transforms = [
        v2.ToImage(),
        v2.Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True),
    ]

    if train:
        augmentation_transforms = [
            # Random rotation
            v2.RandomRotation(degrees=60)
            # # Slight color jitter to simulate different lighting conditions
            # v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            # # Random perspective transformation
            # v2.RandomPerspective(distortion_scale=0.2, p=0.5),
            # # Random affine transformations
            # v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ]
        base_transforms.extend(augmentation_transforms)

    # Final normalization transforms
    final_transforms = [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    base_transforms.extend(final_transforms)
    return v2.Compose(base_transforms)


def resnet18_256_transform(train: bool) -> v2.Compose:
    # Base transforms for both training and inference
    base_transforms = [
        v2.ToImage(),
        v2.Resize((256, 256), interpolation=InterpolationMode.BICUBIC, antialias=True),
    ]

    if train:
        augmentation_transforms = [
            # Random rotation
            v2.RandomRotation(degrees=90)
            # # Slight color jitter to simulate different lighting conditions
            # v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            # # Random perspective transformation
            # v2.RandomPerspective(distortion_scale=0.2, p=0.5),
            # # Random affine transformations
            # v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ]
        base_transforms.extend(augmentation_transforms)

    # Final normalization transforms
    final_transforms = [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    base_transforms.extend(final_transforms)
    return v2.Compose(base_transforms)
