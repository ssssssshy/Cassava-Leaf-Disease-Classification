import torch
import os
import albumentations as A
import cv2
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


class CustomDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None):
        self.df = df
        self.image_dir = img_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]["image_id"]
        label = self.df.iloc[idx]["label"]

        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)

        if image is None:
            raise FileNotFoundError(f"Картинка не найдена по пути: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        return image, torch.tensor(label, dtype=torch.long)


def get_transforms(img_size):

    train_transforms = A.Compose(
        [
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.15, rotate_limit=60, p=0.5
            ),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(1, max(1, img_size // 8)),
                hole_width_range=(1, max(1, img_size // 8)),
                p=0.3,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    return train_transforms, val_transforms


def get_dataloaders(cfg):

    df = pd.read_csv(cfg.path.train_csv)

    train_df, val_df = train_test_split(
        df,
        test_size=cfg.train.val_size,
        stratify=df["label"],
        random_state=cfg.train.seed,
    )

    train_trans, val_trans = get_transforms(cfg.train.img_size)

    train_dataset = CustomDataset(train_df, cfg.path.img_dir, transforms=train_trans)
    val_dataset = CustomDataset(val_df, cfg.path.img_dir, transforms=val_trans)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader
