import yaml
from pathlib import Path
from pydantic import BaseModel, Field


class PathConfig(BaseModel):
    train_csv: str = Field(default="data/raw/train.csv")
    img_dir: str = Field(default="data/raw/train_images")


class ModelConfig(BaseModel):
    name: str
    num_classes: int
    pretrained: bool


class TrainConfig(BaseModel):
    batch_size: int
    epochs: int
    lr: float
    focal_gamma: float
    seed: int
    img_size: int
    num_workers: int
    val_size: float


class WandbConfig(BaseModel):
    project_name: str
    run_name: str


class AppConfig(BaseModel):
    path: PathConfig
    model: ModelConfig
    train: TrainConfig
    wandb: WandbConfig


def load_config(config_path: str | Path) -> AppConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)
    return AppConfig(**raw_config)


if __name__ == "__main__":
    config = load_config("configs/baseline.yaml")
    print(config)
