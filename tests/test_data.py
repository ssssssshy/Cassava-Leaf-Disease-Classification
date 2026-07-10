import os
import numpy as np
import pandas as pd
import torch
import pytest
from PIL import Image  # Заменили cv2 на PIL, чтобы не тянуть лишние зависимости
from src.data import CustomDataset, get_transforms


@pytest.fixture
def dummy_data(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    img_name = "test_img.jpg"
    img_path = os.path.join(img_dir, img_name)

    # Создаем красную картинку 300x300 (RGB) через Pillow
    dummy_img = np.zeros((300, 300, 3), dtype=np.uint8)
    dummy_img[:] = (255, 0, 0)  # RGB
    Image.fromarray(dummy_img).save(img_path)

    df = pd.DataFrame({"image_id": [img_name], "label": [3]})

    return df, str(img_dir)


def test_custom_dataset(dummy_data):
    df, img_dir = dummy_data

    dataset = CustomDataset(df, img_dir, transforms=None)

    assert len(dataset) == 1

    image, label = dataset[0]

    assert isinstance(image, torch.Tensor)
    # read_image возвращает тензор в формате (C, H, W)
    assert image.shape == (3, 300, 300)
    # Сырая картинка должна быть uint8 (от 0 до 255)
    assert image.dtype == torch.uint8

    assert isinstance(label, torch.Tensor)
    assert label.item() == 3


def test_get_transforms_output_shape(dummy_data):
    df, img_dir = dummy_data

    # Размер картинки (256) теперь зашит прямо внутри input_size, как мы и договаривались
    dummy_data_config = {
        "input_size": [3, 256, 256],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "interpolation": "bilinear",
    }

    # Вызываем с чистой сигнатурой (только конфиг)
    train_trans, val_trans = get_transforms(dummy_data_config)

    dataset = CustomDataset(df, img_dir, transforms=val_trans)
    image, label = dataset[0]

    assert isinstance(image, torch.Tensor)
    # Проверяем, что ресайз отработал верно
    assert image.shape == (3, 256, 256)
    # v2.ToDtype(torch.float32, scale=True) должен перевести тензор в float32
    assert image.dtype == torch.float32
