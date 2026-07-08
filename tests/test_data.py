import os
import cv2
import numpy as np
import pandas as pd
import torch
import pytest
from src.data import CustomDataset, get_transforms


@pytest.fixture
def dummy_data(tmp_path):

    img_dir = tmp_path / "images"
    img_dir.mkdir()

    img_name = "test_img.jpg"
    img_path = os.path.join(img_dir, img_name)
    dummy_img = np.zeros((300, 300, 3), dtype=np.uint8)
    dummy_img[:] = (0, 0, 255)  # BGR
    cv2.imwrite(img_path, dummy_img)

    df = pd.DataFrame({"image_id": [img_name], "label": [3]})

    return df, str(img_dir)


def test_custom_dataset(dummy_data):
    df, img_dir = dummy_data

    dataset = CustomDataset(df, img_dir, transforms=None)

    assert len(dataset) == 1

    image, label = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 300, 300)
    assert isinstance(label, torch.Tensor)
    assert label.item() == 3


def test_get_transforms_output_shape(dummy_data):
    df, img_dir = dummy_data

    dummy_data_config = {
        "input_size": [3, 256, 256],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "interpolation": "bilinear",
    }

    train_trans, val_trans = get_transforms(dummy_data_config)

    dataset = CustomDataset(df, img_dir, transforms=val_trans)
    image, label = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 256, 256)
