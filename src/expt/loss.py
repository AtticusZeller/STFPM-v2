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
