import pytest
import yaml
from src.config import load_config, AppConfig


def test_load_config_valid(tmp_path):
    config_data = {
        "path": {"train_csv": "dummy.csv", "img_dir": "dummy_dir"},
        "model": {"name": "inception_v3", "num_classes": 5, "pretrained": True},
        "train": {
            "batch_size": 32,
            "epochs": 10,
            "lr": 0.001,
            "focal_gamma": 2.0,
            "seed": 42,
            "img_size": 256,
            "num_workers": 2,
            "val_size": 0.2,
        },
        "wandb": {"project_name": "test", "run_name": "test_run"},
    }

    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    cfg = load_config(config_file)

    assert isinstance(cfg, AppConfig)
    assert cfg.train.batch_size == 32
    assert cfg.model.name == "inception_v3"


def test_load_config_missing_required(tmp_path):

    config_data = {"path": {"train_csv": "dummy.csv", "img_dir": "dummy_dir"}}

    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    with pytest.raises(Exception):
        load_config(config_file)
