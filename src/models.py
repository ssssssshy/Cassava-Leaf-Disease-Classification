import torch
import torch.nn as nn
import timm
from timm.data.config import resolve_data_config
from src.config import load_config
from ultralytics import YOLO

try:
    from effdet import create_model_from_config, get_efficientdet_config
except ImportError:
    create_model_from_config, get_efficientdet_config = None, None


class EfficientDetClsWrapper(nn.Module):
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True):
        super().__init__()

        if create_model_from_config is None or get_efficientdet_config is None:
            raise ImportError("pip install effdet")

        config = get_efficientdet_config(model_name)  # type: ignore

        detector = create_model_from_config(  # type: ignore
            config, bench_task="", pretrained=pretrained
        )

        self.backbone: nn.Module = detector.backbone  # type: ignore

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        dummy_input = torch.randn(1, 3, 224, 224)

        features = self.backbone(dummy_input)

        if isinstance(features, (list, tuple)):
            features = features[-1]
        in_features = features.shape[1]

        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]

        out = self.global_pool(feats)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


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
            in_features = head.linear.in_features  # type: ignore
            head.linear = nn.Linear(in_features, cfg.model.num_classes)  # type: ignore
        else:
            raise ValueError(
                f"Не удалось найти слой 'linear' в голове модели {cfg.model.name}"
            )

    def forward(self, x):
        out = self.model(x)

        if isinstance(out, (tuple, list)):
            return out[0]
        return out


def build_effdet_model(cfg):
    model_name = cfg.model.name.lower()

    if "b4" in model_name or "d4" in model_name:
        effdet_name = "tf_efficientdet_d4"
    else:
        effdet_name = "tf_efficientdet_d0"  #

    model = EfficientDetClsWrapper(
        model_name=effdet_name,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
    )

    img_size = cfg.train.img_size if hasattr(cfg.train, "img_size") else 512
    data_config = {
        "input_size": (3, img_size, img_size),
        "interpolation": "bicubic",
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "crop_pct": 1.0,
    }
    return model, data_config


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
    elif "efficientdet" in model_name or "effdet" in model_name:
        return build_effdet_model(cfg)
    else:
        return build_timm_model(cfg)


if __name__ == "__main__":
    cfg = load_config("configs/baseline.yaml")
    model, data_config = build_model(cfg)

    print(f"Модель {cfg.model.name} успешно загружена.")
    print("Конфигурация данных:", data_config)
