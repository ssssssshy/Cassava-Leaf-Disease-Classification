import timm
from src.config import load_config


def build_model(cfg):

    model = timm.create_model(
        cfg.model.name,
        pretrained=cfg.model.pretrained,
        num_classes=cfg.model.num_classes,
    )

    return model


if __name__ == "__main__":
    cfg = load_config("configs/baseline.yaml")
    model = build_model(cfg)
    print(model)
