from dataclasses import asdict
from pathlib import Path
from typing import Self

import numpy as np
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from numpy.typing import NDArray
from rich import print
from rich.pretty import pprint
from torch import nn

import wandb
from expt.config import Config, DataConfig, ModelConfig, OptimizerConfig, TrainingConfig
from expt.geometry import plot_anomaly_map


class LoggerManager(WandbLogger):
    """
    Initialize the Weights & Biases logging.
    ```bash
    wandb login --relogin
    ```

    Ref:
        1. https://docs.wandb.ai/ref/python/init/
        2. https://docs.wandb.ai/guides/integrations/lightning/
        3. https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb
    """

    def __init__(
        self,
        run_name: str,
        entity: str,
        project: str,
        config: Config,
        id: str | None = None,
        job_type: str = "train",
    ) -> None:
        Path("./logs").mkdir(parents=True, exist_ok=True)
        super().__init__(
            project=project,
            entity=entity,
            name=run_name,
            job_type=job_type,
            id=id,
            resume="must"
            if id is not None
            else "never",  # resume run if id is provided
            config=config,
            save_dir="./logs",
            # offline=True,
        )
        self.entity = entity
        self.job_type = job_type
        self.config = config
        self._watched_models: list[nn.Module] = []

        if self.sweeping:
            # update from sweep
            self._update_config_with_sweep(config)
        # actual runtime config
        pprint(self.experiment.config.as_dict())

        self.artifacts_dir = Path(f"./artifacts/{self.version}")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> Self:
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore
        """Exit the context manager.

        Will automatically unwatch all watched models and finish the wandb run.
        """
        # Unwatch all models that were watched
        for model in self._watched_models:
            self.experiment.unwatch(model)
        # Finish the wandb run
        self.experiment.finish()
        if self.job_type == "train" and not self.sweeping:
            print("\nTraining completed! To test this model, use:")
            print(f"Run ID: [bold cyan]{self.version}[/bold cyan]")

    def load_best_model(self, run_id: str) -> Path:
        """Load the best model from a specific run ID."""
        # Use glob to find model checkpoint files that match the pattern
        model_path_pattern = Path(f"./artifacts/{run_id}").glob("model_*.ckpt")
        model_paths = list(model_path_pattern)
        if not len(model_paths) == 1:
            raise FileNotFoundError(
                f"Multiple or no model files found for run ID {run_id}."
            )
        # Get the first matching model path
        model_path = model_paths[0]

        if not model_path.exists():
            raise FileNotFoundError(f"checkpoint run_id={run_id} not found.")
        return model_path

    def upload_best_model(self) -> None:
        """Upload the best model to wandb"""

        ckpt_path = self.artifacts_dir / "model.ckpt"
        ckpt_name = f"model-{self.version}"
        artifact = wandb.Artifact(name=ckpt_name, type="model")
        artifact.add_file(local_path=ckpt_path.as_posix(), name="model.ckpt")
        self.experiment.log_artifact(artifact, aliases="best")

    def checkpoint_callback(
        self, monitor: str = "val_accuracy", mode: str = "max"
    ) -> ModelCheckpoint:
        """Return the ModelCheckpoint callback

        See Also:
            - [PyTorch Lightning ModelCheckpoint](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html#lightning.pytorch.callbacks.ModelCheckpoint)
        """
        return ModelCheckpoint(
            monitor=monitor,
            mode=mode,
            dirpath=self.artifacts_dir,
            filename=f"model_{{epoch:02d}}_{{{monitor}:.2f}}",
            save_top_k=1,
            auto_insert_metric_name=True,
        )

    def watch(
        self,
        model: nn.Module,
        log: str | None = "all",
        log_freq: int = 100,
        log_graph: bool = False,
    ) -> None:
        """Override watch method to keep track of watched models."""
        super().watch(model, log=log, log_freq=log_freq, log_graph=log_graph)
        self._watched_models.append(model)

    @staticmethod
    def init_sweep(sweep_config_path: Path, project: str, entity: str) -> str:
        """Initialize sweep from config file"""
        with open(sweep_config_path) as f:
            sweep_config = yaml.safe_load(f)

        sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)
        return sweep_id

    @property
    def sweeping(self) -> bool:
        """Check if the current run is part of a sweep"""
        return self.experiment.sweep_id is not None

    def _update_config_with_sweep(self, config: Config) -> Config:
        """Update config passing to experiment and wandb config with sweep values
        if in sweep mode"""
        # derive sweep config from experiment config
        sweep_config = self.experiment.config.as_dict()

        for field in asdict(config):
            sub_config: ModelConfig | OptimizerConfig | DataConfig | TrainingConfig = (
                getattr(config, field)
            )
            for key in asdict(sub_config):
                if key in sweep_config:
                    # update config with sweep values
                    setattr(sub_config, key, sweep_config[key])
                    # update wandb config with sweep values
                    sweep_config[field][key] = sweep_config[key]
        # force update sweep config
        self.experiment.config.update(sweep_config, True)
        return config

    def log_roc_curve(
        self,
        y_true: NDArray[np.int_],
        y_pred: NDArray[np.float_],
        title: str = "ROC Curve",
    ) -> None:
        """Log ROC curve to wandb
        Ref:
            1. https://wandb.ai/wandb/plots/reports/Plot-ROC-Curves-With-Weights-Biases--VmlldzoyNjk3MDE
        """
        # Reshape y_pred to have the format expected by wandb
        # For binary classification, we need probabilities for both classes
        # If y_pred contains scores rather than probabilities for class 1,
        # we'll create the implied probability for class 0 as 1-y_pred
        y_probas = np.zeros((len(y_pred), 2))
        y_probas[:, 1] = y_pred  # Probability for class 1
        y_probas[:, 0] = 1 - y_pred  # Probability for class 0

        self.experiment.log(
            {
                title: wandb.plot.roc_curve(
                    y_true=y_true,  # type: ignore
                    y_probas=y_probas,  # type: ignore
                    labels=[
                        "Normal",
                        "Anomaly",
                    ],  # Optional: provide human-readable labels
                    classes_to_plot=None,
                )
            }
        )

    def log_anomaly_map(
        self,
        images: list[NDArray[np.float_]],
        original_images: list[NDArray[np.float_]],
        labels: list[NDArray[np.float_]],
        title: str = "Anomaly Map",
    ) -> None:
        """Log anomaly maps with corresponding original images to wandb

        Args:
            images (list[NDArray[np.float_]]):
                List of anomaly maps with shape (1, H, W)
            original_images (list[NDArray[np.float_]]):
                List of original images with shape (C, H, W)
            title (str, optional): Title for the logged images.
                Defaults to "Anomaly Map".
        """
        # Create a list to store the final visualization images
        visualizations = plot_anomaly_map(
            images=images, original_images=original_images
        )
        # Create a wandb Table
        table = wandb.Table(columns=["id", "image", "label"])

        # Add all images to the table
        for i, img in enumerate(visualizations):
            lable_str = "good" if labels[i] == 0 else "bad"
            table.add_data(
                i, wandb.Image(img, caption=f"Anomaly Detection {i}"), lable_str
            )

        # Log the table to wandb
        self.experiment.log({title: table})
