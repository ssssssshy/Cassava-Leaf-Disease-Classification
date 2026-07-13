import torch.nn as nn
import timm
from timm.data.config import resolve_data_config
from src.config import load_config
from ultralytics import YOLO


class YOLOClsWrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        base_name = (
            cfg.model.name.replace("-cls", "").replace(".pt", "").replace(".yaml", "")
        )
        yaml_config = f"{base_name}-cls.yaml"

        yolo_app = YOLO(yaml_config)

        if cfg.model.pretrained:
            print(f"Загрузка весов backbone из {base_name}.pt...")
            yolo_app.load(f"{base_name}.pt")

        self.model: nn.Module = yolo_app.model  # type: ignore

        head = self.model.model[-1]  # type: ignore

        if hasattr(head, "linear"):
            in_features = head.linear.in_features
            head.linear = nn.Linear(in_features, cfg.model.num_classes)
        else:
            raise ValueError(
                f"Не удалось найти слой 'linear' в голове модели {cfg.model.name}"
            )

    def forward(self, x):
        out = self.model(x)

        if isinstance(out, (tuple, list)):
            return out[0]
        return out


def build_yolo_model(cfg):

    model = YOLOClsWrapper(cfg)

    img_size = cfg.train.img_size if hasattr(cfg.train, "img_size") else 224
    data_config = {
        "input_size": (3, img_size, img_size),
        "interpolation": "bicubic",
        "mean": (0.0, 0.0, 0.0),
        "std": (1.0, 1.0, 1.0),
        "crop_pct": 1.0,
    }
    return model, data_config


def build_timm_model(cfg):

    model = timm.create_model(
        cfg.model.name,
        pretrained=cfg.model.pretrained,
        num_classes=cfg.model.num_classes,
    )

    user_data_args = {}
    if hasattr(cfg.train, "img_size"):
        user_data_args["input_size"] = (3, cfg.train.img_size, cfg.train.img_size)

    data_config = resolve_data_config(user_data_args, model=model)
    return model, data_config


def build_model(cfg):

    model_name = cfg.model.name.lower()

    if "yolo" in model_name:
        return build_yolo_model(cfg)
    else:
        return build_timm_model(cfg)


if __name__ == "__main__":
    cfg = load_config("configs/baseline.yaml")
    model, data_config = build_model(cfg)

    print(f"Модель {cfg.model.name} успешно загружена.")
    print("Конфигурация данных:", data_config)
