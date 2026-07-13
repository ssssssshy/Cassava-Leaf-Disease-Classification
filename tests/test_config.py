import pytest
import yaml
from pydantic import ValidationError
from src.config import load_config, AppConfig


@pytest.fixture
def valid_config_data():
    """Фикстура, предоставляющая валидный словарь конфигурации для тестов."""
    return {
        "path": {
            "train_csv": "dummy.csv",
            "img_dir": "dummy_dir",
            "save_path": "weights/test_model.pth",  # <--- Перенесли сюда
        },
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
        "early_stopping": {"patience": 5, "min_delta": 0.0, "mode": "max"},
        "wandb": {"project_name": "test", "run_name": "test_run"},
    }


def test_load_config_valid(tmp_path, valid_config_data):
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(valid_config_data, f)

    cfg = load_config(config_file)

    assert isinstance(cfg, AppConfig)
    assert cfg.train.batch_size == 32
    assert cfg.model.name == "inception_v3"
    assert cfg.path.save_path == "weights/test_model.pth"  # <--- Изменили проверку
    assert cfg.early_stopping.mode == "max"


def test_load_config_missing_required(tmp_path):
    config_data = {"path": {"train_csv": "dummy.csv", "img_dir": "dummy_dir"}}

    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    with pytest.raises(ValidationError):
        load_config(config_file)


def test_invalid_train_parameters(tmp_path, valid_config_data):
    valid_config_data["train"]["val_size"] = 1.5

    config_file = tmp_path / "test_config_invalid_val.yaml"
    with open(config_file, "w") as f:
        yaml.dump(valid_config_data, f)

    with pytest.raises(ValidationError) as exc_info:
        load_config(config_file)

    assert "val_size" in str(exc_info.value)


def test_invalid_early_stopping_mode(tmp_path, valid_config_data):
    valid_config_data["early_stopping"]["mode"] = "average"

    config_file = tmp_path / "test_config_invalid_mode.yaml"
    with open(config_file, "w") as f:
        yaml.dump(valid_config_data, f)

    with pytest.raises(ValidationError) as exc_info:
        load_config(config_file)

    assert "mode должен быть 'max' или 'min'" in str(exc_info.value)
