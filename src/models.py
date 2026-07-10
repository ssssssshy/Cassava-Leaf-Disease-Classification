from xml.parsers.expat import model

import timm
from timm.data.config import resolve_data_config
from src.config import load_config

# avail_pretrained_models = timm.list_models(pretrained = True)
# avail_pretrained_models

cfg = load_config("configs/baseline.yaml")


def build_model(cfg):
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


if __name__ == "__main__":
    cfg = load_config("configs/baseline.yaml")
    model, data_config = build_model(cfg)

    print("Backbone Data Config:", data_config)
