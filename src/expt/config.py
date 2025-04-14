# config.py
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from rich import print

BackBoneT = Literal["resnet18"]


@dataclass
class ModelConfig:
    name: str = "STPFM"
    backbone: BackBoneT = "resnet18"


@dataclass
class OptimizerConfig:
    name: str = "adam"
    lr: float = 0.4


# Define asset type literals
AssetType = Literal[
    "damper-preformed",  # test/good, train/good
    "damper-stockbridge",  # test/good or test/rust, train/good
    "glass-insulator",  # test/good or test/missingcap, train/good
    "glass-insulator-big-shackle",  # test/good or test/rust, train/good
    "glass-insulator-small-shackle",  # test/good or test/nest, train/good
    "glass-insulator-tower-shackle",  # test/good or test/rust, train/good
    "lightning-rod-shackle",  # test/good or test/rust, train/good
    "lightning-rod-suspension",  # test/good or test/rust, train/good
    "plate",  # test/good or test/peeling-paint, train/good
    "polymer-insulator",  # test/good or test/torned-up, train/good
    "polymer-insulator-lower-shackle",  # test/good or test/rust, train/good
    "polymer-insulator-tower-shackle",  # test/good or test/rust, train/good
    "polymer-insulator-upper-shackle",  # test/good or test/rust, train/good
    "spacer",  # test/good, train/good
    "vari-grip",  # test/good or test/nest or test/rust, train/good
    "yoke",  # test/good, train/good
    "yoke-suspension",  # test/good or test/rust, train/good
]

TransformT = Literal["resnet18", "base"]


@dataclass
class DataConfig:
    dataset: str = "InsPLAD"
    asset_type: AssetType = "damper-preformed"
    batch_size: int = 4
    augmentation: list[str] | None = None
    transform: TransformT = "resnet18"


@dataclass
class TrainingConfig:
    max_epochs: int = 100


@dataclass
class LoggerConfig:
    run_name: str = "test_run"
    entity: str = "atticux"  # set to name of your wandb team
    project: str = "STPFM-v2"


@dataclass
class Config:
    model: ModelConfig
    logger: LoggerConfig
    data: DataConfig
    training: TrainingConfig
    optimizer: OptimizerConfig


class ConfigManager:
    def __init__(self, config_dir: str | Path = "./config") -> None:
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_map = {
            "model": ModelConfig,
            "optimizer": OptimizerConfig,
            "data": DataConfig,
            "training": TrainingConfig,
            "logger": LoggerConfig,
        }

    def generate_default_configs(self) -> None:
        """generate default configuration files for evaluation and training"""
        print("Generating default configuration files...")
        default_config = {}
        # sub config
        for name, component in self.config_map.items():
            sub_conf = asdict(component())
            default_config[name] = sub_conf
        self._save_config(default_config, self.config_dir / "train.yml")

    def load_config(self, config_path: str | Path) -> Config:
        """load configuration from yml file"""
        config_path = Path(config_path)
        if (
            not config_path.exists()
            or not config_path.is_file()
            or config_path.suffix not in [".yml", ".yaml"]
        ):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        print(f"Loading config from: [bold cyan]{config_path}[/bold cyan]")
        conf = self._load_config(config_path)
        # load all config
        for name in conf:
            if name in self.config_map:
                conf[name] = self.config_map[name](**conf[name])

        return Config(**conf)

    @staticmethod
    def _save_config(config: dict[str, Any], save_path: Path) -> None:
        with open(save_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    @staticmethod
    def _load_config(config_path: Path) -> dict[str, Any]:
        with open(config_path) as f:
            return yaml.safe_load(f)


if __name__ == "__main__":
    config_manager = ConfigManager()
    config_manager.generate_default_configs()
    config = config_manager.load_config("config/train.yml")

    print("\nConfiguration loaded successfully:")
    print(asdict(config))
