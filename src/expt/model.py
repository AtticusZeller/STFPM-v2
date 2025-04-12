from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import timm
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torchmetrics.classification import BinaryAUROC

from expt.config import BackBoneT, Config
from expt.eval.logger import LoggerManager
from expt.geometry import compute_anomaly_map
from expt.loss import STFPMLoss


class FeatureExtractor(nn.Module):
    """
    BackBone for feature extraction\n
    Ref:
        1. https://huggingface.co/docs/timm/v1.0.15/en/feature_extraction#flexible-intermediate-feature-map-extraction
        2. https://github.com/open-edge-platform/anomalib/blob/main/src/anomalib/models/image/stfpm/torch_model.py
    """

    def __init__(
        self, name: BackBoneT, pretrained: bool = True, requires_grad: bool = True
    ) -> None:
        super().__init__()
        self._feature_extractor = timm.create_model(
            name, pretrained=pretrained, features_only=True
        )
        self.pretrained = pretrained
        self.requires_grad = requires_grad
        # turn off gradient calculation for all parameters
        if not self.requires_grad:
            self._feature_extractor.eval()
            for parameters in self._feature_extractor.parameters():
                parameters.requires_grad = False

        self.out_dims: list[int] = self._feature_extractor.feature_info.channels()

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            inputs(torch.Tensor): images, `(B, C, H, W)`

        Returns:
            list[torch.Tensor]: Features from layers, with shape `(B, C, H, W)`
        """
        if self.requires_grad:
            return self._feature_extractor(inputs)
        else:
            with torch.no_grad():
                return self._feature_extractor(inputs)


class STFPMModel(nn.Module):
    def __init__(self, backbone: BackBoneT = "resnet18") -> None:
        super().__init__()

        # models
        self.teacher_model = FeatureExtractor(
            name=backbone, pretrained=True, requires_grad=False
        )
        self.student_model = FeatureExtractor(
            name=backbone, pretrained=False, requires_grad=True
        )

    def forward(self, images: torch.Tensor) -> tuple[list[Tensor], list[Tensor]]:
        """
        Args:
            images (torch.Tensor): input images of shape `(B, C, H, W)`

        Returns:
            tuple[list[Tensor], list[Tensor]]: Features from teacher and student models,
                with shape `(B, C, H, W)`
        """
        return self.teacher_model.forward(images), self.student_model.forward(images)


class STFPM(pl.LightningModule):
    """STFPM LightningModule for anomaly detection\n
    Ref:
        1. https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
    """

    def __init__(self, backbone: BackBoneT = "resnet18", lr: float = 1e-3) -> None:
        super().__init__()

        # models
        self.model = STFPMModel(backbone=backbone)
        self.criterion = STFPMLoss()

        # metrics
        self.auroc = BinaryAUROC()

        self.lr = lr
        # save hyperparameters
        self.save_hyperparameters()
        self.test_step_outputs: list[dict[str, Any]] = []

    @property
    def logger(self) -> LoggerManager:
        if self.trainer.logger is None:
            raise ValueError("Logger is not defined. Please set a logger.")
        return self.trainer.logger  # type: ignore

    def training_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        """step for training dataset 80% of all normal images for training
        train_loss for optimizing model
        Args:
            batch (tuple[Tensor, Tensor]): batch of image, batch of labels
            batch_idx (int): batch index
        """
        x, _ = batch
        teacher_features, student_features = self.model.forward(x)
        loss = self.criterion(teacher_features, student_features)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        """step for validation dataset 20% of all normal images for training
        val_loss is watched for saving checkpoint
        Args:
            batch (tuple[Tensor, Tensor]): batch of image, batch of labels
            batch_idx (int): batch index
        """
        x, _ = batch
        teacher_features, student_features = self.model.forward(x)
        loss = self.criterion(teacher_features, student_features)

        # on_epoch for epoch performance
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> STEP_OUTPUT:
        """Compute the anomaly map and image-level anomaly score for each test image.
        And collect the outputs for metrics(AUROC,ROC) calculation.
        Args:
            batch (tuple[Tensor, Tensor]): batch of image, batch of labels
            batch_idx (int): batch index
        """
        x, y = batch
        teacher_features, student_features = self.model.forward(x)

        # Compute loss
        loss = self.criterion(teacher_features, student_features)

        # Get image size for anomaly map calculation
        image_size = x.shape[-2:]  # Height and width of input images

        # Compute anomaly map
        anomaly_maps = compute_anomaly_map(
            teacher_features=teacher_features,
            student_features=student_features,
            image_size=image_size,
        )

        # Calculate image-level anomaly scores (max value in anomaly map)
        anomaly_scores = torch.amax(anomaly_maps, dim=(-2, -1)).squeeze()

        # on_epoch for epoch performance
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "test_anomaly_score",
            anomaly_scores.mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # Return values for epoch_end processing
        test_output = {
            "anomaly_maps": anomaly_maps,  # visualization, (B, 1, H, W)
            "images": x,  # for visualization, (B, C, H, W)
            "anomaly_scores": anomaly_scores,  # for AUROC, (B, )
            "labels": y,  # ground truth labels,  (B, )
        }
        self.test_step_outputs.append(test_output)
        return test_output

    def on_test_epoch_end(self) -> None:
        """Process collected outputs of test_step and compute metrics."""
        # Concatenate all outputs from test steps, ensuring proper dimensions
        all_anomaly_maps = torch.cat(
            [
                output["anomaly_maps"].unsqueeze(0)
                if output["anomaly_maps"].dim() == 3
                else output["anomaly_maps"]
                for output in self.test_step_outputs
            ]
        )

        all_images_normalized = torch.cat(
            [
                output["images"].unsqueeze(0)
                if output["images"].dim() == 3
                else output["images"]
                for output in self.test_step_outputs
            ]
        )

        # Denormalize images (using ImageNet statistics)
        mean = torch.tensor(
            [0.485, 0.456, 0.406], device=all_images_normalized.device
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            [0.229, 0.224, 0.225], device=all_images_normalized.device
        ).view(1, 3, 1, 1)
        all_images = all_images_normalized * std + mean
        all_images = torch.clamp(all_images, 0, 1)  # Ensure values are in [0, 1]

        all_anomaly_scores = torch.cat(
            [
                output["anomaly_scores"].unsqueeze(0)
                if output["anomaly_scores"].dim() == 0
                else output["anomaly_scores"]
                for output in self.test_step_outputs
            ]
        )

        all_labels = torch.cat(
            [
                output["labels"].unsqueeze(0)
                if output["labels"].dim() == 0
                else output["labels"]
                for output in self.test_step_outputs
            ]
        )
        # Calculate image-level AUROC
        self.auroc.update(all_anomaly_scores, all_labels)
        image_auroc = self.auroc.compute()
        self.log("test_image_auroc", image_auroc)

        # Plot image-level ROC
        self.logger.log_roc_curve(
            all_labels.cpu().numpy(),
            all_anomaly_scores.cpu().numpy(),
            title="ROC Curve",
        )
        # Plot sample anomaly maps randomly
        self.logger.log_anomaly_map(
            list(all_anomaly_maps.cpu().numpy()),
            list(all_images.cpu().numpy()),
            list(all_labels.cpu().numpy()),
            "Sample Anomaly Maps",
        )

        # Clear the outputs to free memory
        self.test_step_outputs.clear()
        self.auroc.reset()

    def configure_optimizers(self) -> Optimizer:
        """defines model optimizer"""
        return Adam(self.model.student_model.parameters(), lr=self.lr)


def create_model(config: Config, model_path: Path | None = None) -> pl.LightningModule:
    if config.model.name.lower() == "stpfm":
        return (
            STFPM(backbone=config.model.backbone, lr=config.optimizer.lr)
            if model_path is None
            else STFPM.load_from_checkpoint(model_path)
        )
    else:
        raise ValueError(f"Model name {config.model.name} not supported.")
