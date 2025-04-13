"""Loss function for Student-Teacher Feature Pyramid Matching model.

This module implements the loss function used to train the STFPM model for anomaly
detection as described in `Wang et al. (2021) <https://arxiv.org/abs/2103.04257>`_.

The loss function:
1. Takes feature maps from teacher and student networks as input
2. Normalizes the features using L2 normalization
3. Computes MSE loss between normalized features
4. Scales the loss by spatial dimensions of feature maps
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812


class STFPMLoss(nn.Module):
    """Loss function for Student-Teacher Feature Pyramid Matching model.

    This class implements the feature pyramid loss function proposed in the STFPM
    paper. The loss measures the discrepancy between feature representations from
    a pre-trained teacher network and a student network that learns to match them.

    The loss computation involves:
    1. Normalizing teacher and student features using L2 normalization
    2. Computing MSE loss between normalized features
    3. Scaling the loss by spatial dimensions of feature maps
    4. Summing losses across all feature layers
    """

    def __init__(self) -> None:
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def _compute_layer_loss(
        self, teacher_feats: torch.Tensor, student_feats: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss between teacher and student features for a single layer.

        This implements the loss computation based on Equation (1,2) in Section 3.2

        of the paper. The loss is computed as:
        1. L2 normalize teacher and student features
        2. Compute MSE loss between normalized features
        3. Scale loss by spatial dimensions (height * width * batch)
        NOTE: we add batch size for scale

        Args:
            teacher_feats (torch.Tensor): Features from teacher network with shape
                `(B, C, H, W)`
            student_feats (torch.Tensor): Features from student network with shape
                `(B, C, H, W)`

        Returns:
            torch.Tensor: Scalar loss value for the layer
        """
        batch_size, _, height, width = teacher_feats.shape

        norm_teacher_features = F.normalize(teacher_feats)
        norm_student_features = F.normalize(student_feats)
        return (0.5 / (width * height * batch_size)) * self.mse_loss(
            norm_teacher_features, norm_student_features
        )

    def forward(
        self, teacher_features: list[torch.Tensor], student_features: list[torch.Tensor]
    ) -> torch.Tensor:
        """Compute total loss across all feature layers.

        This implements the loss computation based on Equation (3) in Section 3.2
        The total loss is computed as the sum of individual layer losses. Each
        layer loss measures the discrepancy between teacher and student features
        at that layer.

        Args:
            teacher_feats (list[torch.Tensor]): Features from teacher network with shape
                `(B, C, H, W)`
            student_feats (list[torch.Tensor]): Features from student network with shape
                `(B, C, H, W)`
        Returns:
            torch.Tensor: Total loss summed across all layers
        """
        layer_losses = torch.zeros(1, device=teacher_features[0].device)
        # Iterate through each layer's features
        for layer_t, layer_s in zip(teacher_features, student_features, strict=True):
            loss = self._compute_layer_loss(layer_t, layer_s)
            layer_losses += loss
        return layer_losses


def reduce_tensor_elems(tensor: torch.Tensor, m: int = 2**24) -> torch.Tensor:
    """Reduce the number of elements in a tensor by random sampling.

    This function flattens an n-dimensional tensor and randomly samples at most ``m``
    elements from it. This is used to handle the limitation of ``torch.quantile``
    operation which supports a maximum of 2^24 elements.

    Reference:
        https://github.com/pytorch/pytorch/blob/b9f81a483a7879cd3709fd26bcec5f1ee33577e6/aten/src/ATen/native/Sorting.cpp#L291

    Args:
        tensor (torch.Tensor): Input tensor of any shape from which elements will be
            sampled.
        m (int, optional): Maximum number of elements to sample. If the flattened
            tensor has more elements than ``m``, random sampling is performed.
            Defaults to ``2**24``.

    Returns:
        torch.Tensor: A flattened tensor containing at most ``m`` elements randomly
            sampled from the input tensor.

    Example:
        >>> import torch
        >>> tensor = torch.randn(1000, 1000)  # 1M elements
        >>> reduced = reduce_tensor_elems(tensor, m=1000)
        >>> reduced.shape
        torch.Size([1000])
    """
    tensor = torch.flatten(tensor)
    if len(tensor) > m:
        # select a random subset with m elements.
        perm = torch.randperm(len(tensor), device=tensor.device)
        idx = perm[:m]
        tensor = tensor[idx]
    return tensor


class HardFeatureSTFPMLoss(STFPMLoss):
    """STFPM loss with hard feature mining.

    This loss extends the standard STFPM loss by implementing hard feature mining,
    which restricts the student's loss to the most challenging parts of an image
    (where the student mimics the teacher the least).
    This approach helps the student learn important patterns in normal images
    while preventing generalization to anomalies.

    The loss computation involves:
    1. Computing squared differences between teacher and student features
    2. Selecting elements with the highest differences based on a quantile threshold
    3. Computing loss only on the selected "hard" elements

    Args:
        phard (float): Mining factor in range [0,1] that determines the quantile
            threshold for selecting hard examples. Default is 0.999, which corresponds
            to using approximately 10% of the values for backpropagation.
    """

    def __init__(self, phard: float = 0.999) -> None:
        """Initialize the loss with hard feature mining capability.

        Args:
            phard (float): Mining factor in range [0,1] that determines the quantile
                threshold for selecting hard examples. Default is 0.999.
        """
        super().__init__()
        self.phard = phard
        # We don't need MSELoss from parent class as we'll compute it manually
        delattr(self, "mse_loss")

    def _compute_layer_loss(
        self, teacher_feats: torch.Tensor, student_feats: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss between teacher and student features with hard mining.

        Instead of using all elements for loss computation, this method:
        1. L2 normalizes teacher and student features
        2. Computes squared differences between normalized features
        3. Selects elements with the highest differences based on phard quantile
        4. Computes mean loss on selected "hard" elements only

        Args:
            teacher_feats (torch.Tensor): Features from teacher network with shape
                `(B, C, H, W)`
            student_feats (torch.Tensor): Features from student network with shape
                `(B, C, H, W)`

        Returns:
            torch.Tensor: Scalar loss value for the layer using hard mining
        """
        batch_size, _, height, width = teacher_feats.shape

        # L2 normalize features
        norm_teacher_features = F.normalize(teacher_feats)
        norm_student_features = F.normalize(student_feats)

        # Compute squared differences for each element
        squared_diff = (norm_teacher_features - norm_student_features) ** 2

        # Radom sampling for quantile computation
        squared_diff_flat = reduce_tensor_elems(squared_diff)

        # Compute the quantile threshold using torch.quantile
        dhard = torch.quantile(squared_diff_flat, self.phard)

        # Select hard elements and compute mean
        hard_elements = squared_diff_flat[squared_diff_flat >= dhard]

        return torch.mean(hard_elements)
