import torch.nn as nn
import timm
from timm.data.config import resolve_data_config
from src.config import load_config

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class YOLOClsWrapper(nn.Module):
    """Обертка для адаптации YOLO моделей классификации."""

    def __init__(self, model_name="yolo12n", num_classes=1000, pretrained=True):
        super().__init__()
        if YOLO is None:
            raise ImportError(
                "Необходимо установить ultralytics: pip install ultralytics"
            )

        base_name = (
            model_name.replace("-cls", "").replace(".pt", "").replace(".yaml", "")
        )
        yaml_config = f"{base_name}-cls.yaml"

        yolo_app = YOLO(yaml_config)

        if pretrained:
            print(f"Загрузка весов backbone из {base_name}.pt...")
            yolo_app.load(f"{base_name}.pt")

        self.model = yolo_app.model

        head = self.model.model[-1]
        if hasattr(head, "linear"):
            in_features = head.linear.in_features
            head.linear = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(
                f"Не удалось найти слой 'linear' в голове модели {model_name}"
            )

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


### --- Специфичные билдеры --- ###


def build_timm_model(cfg):
    """
    Билдер для стандартных моделей из библиотеки timm.
    Сюда относятся inception_v3, resnet, efficientnet и т.д.
    """
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


def build_yolo_model(cfg):
    """
    Билдер для моделей семейства YOLO.
    """
    model = YOLOClsWrapper(
        model_name=cfg.model.name,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
    )

    # Специфичный конфиг данных для YOLO (без ImageNet нормализации)
    img_size = cfg.train.img_size if hasattr(cfg.train, "img_size") else 224
    data_config = {
        "input_size": (3, img_size, img_size),
        "interpolation": "bicubic",
        "mean": (0.0, 0.0, 0.0),
        "std": (1.0, 1.0, 1.0),
        "crop_pct": 1.0,
    }
    return model, data_config


### --- Основной диспетчер --- ###


def build_model(cfg):
    """
    Основная функция-фабрика.
    Маршрутизирует создание модели в зависимости от её имени в конфигурации.
    """
    model_name = cfg.model.name.lower()

    if "yolo" in model_name:
        return build_yolo_model(cfg)
    else:
        # По умолчанию все остальные модели (включая inception_v3) уходят в timm
        return build_timm_model(cfg)


if __name__ == "__main__":
    # Тест загрузки
    cfg = load_config("configs/baseline.yaml")
    model, data_config = build_model(cfg)

    print(f"Модель {cfg.model.name} успешно загружена.")
    print("Конфигурация данных:", data_config)
