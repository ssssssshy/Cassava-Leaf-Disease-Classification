import yaml
from pathlib import Path
from pydantic import (
    BaseModel,
    Field,
    field_validator,
)  # #8: добавлен field_validator для проверки полей конфига


class PathConfig(BaseModel):
    train_csv: str = Field(default="data/raw/train.csv")
    img_dir: str = Field(default="data/raw/train_images")
    save_path: str = Field(default="weights/best_model.pth")  #


class ModelConfig(BaseModel):
    name: str
    num_classes: int
    pretrained: bool


class TrainConfig(BaseModel):
    # добавлены ограничения Field(gt=0, ge=0, lt=1) — Pydantic выбросит ошибку
    # на этапе загрузки конфига вместо криптического падения глубоко в коде
    batch_size: int = Field(gt=0)  # размер батча должен быть > 0
    epochs: int = Field(gt=0)  # количество эпох должно быть > 0
    lr: float = Field(gt=0)  # learning rate должен быть > 0
    focal_gamma: float = Field(
        ge=0
    )  # gamma фокального лосса не может быть отрицательной
    seed: int
    img_size: int = Field(gt=0)  # размер изображения должен быть > 0
    num_workers: int = Field(ge=0)  # число воркеров не может быть отрицательным
    val_size: float = Field(gt=0, lt=1)  # доля валидации строго в (0, 1)
    # save_path вынесен из хардкода в конфиг — можно менять путь сохранения через YAML
    save_path: str = Field(default="weights/best_model.pth")
    weight_decay: float = Field(default=1e-2, ge=0)


class EarlyStoppingConfig(BaseModel):
    patience: int = Field(default=5, gt=0)  # patience должен быть > 0
    min_delta: float = Field(default=0.0, ge=0)  # min_delta не может быть отрицательным
    mode: str = Field(default="max")

    # валидатор поля mode — ловит опечатки в конфиге на этапе загрузки,
    # вместо silent-бага в EarlyStopping (где assert удаляется при python -O)
    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        if v not in ("max", "min"):
            raise ValueError(f"mode должен быть 'max' или 'min', получено: {v}")
        return v


class WandbConfig(BaseModel):
    project_name: str
    run_name: str


class AppConfig(BaseModel):
    path: PathConfig
    model: ModelConfig
    train: TrainConfig
    wandb: WandbConfig
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)


def load_config(config_path: str | Path) -> AppConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)
    return AppConfig(**raw_config)


if __name__ == "__main__":
    config = load_config("configs/baseline.yaml")
    print(config)
