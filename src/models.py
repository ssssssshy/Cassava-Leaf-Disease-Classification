import timm
from timm.data.config import resolve_data_config
from src.config import load_config

# avail_pretrained_models = timm.list_models(pretrained = True)
# avail_pretrained_models


def build_model(cfg):
    model = timm.create_model(
        cfg.model.name,
        pretrained=cfg.model.pretrained,
        num_classes=cfg.model.num_classes,
    )

    data_config = resolve_data_config({}, model=model)

    return model, data_config


if __name__ == "__main__":
    cfg = load_config("configs/baseline.yaml")
    model, data_config = build_model(cfg)

    print("Backbone Data Config:", data_config)
