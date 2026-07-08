import pathlib
import torch
import albumentations as A
import cv2
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


class CustomDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None):
        self.image_dir = img_dir
        self.transforms = transforms
        self.image_names = df["image_id"].values
        self.labels = df["label"].values

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        label = self.labels[idx]
        img_path = pathlib.Path(self.image_dir) / img_name

        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Картинка не найдена по пути: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        return image, torch.tensor(label, dtype=torch.long)


def get_transforms(data_config):

    img_size = data_config["input_size"][1]
    mean = data_config["mean"]
    std = data_config["std"]

    interpolation = (
        cv2.INTER_CUBIC
        if data_config["interpolation"] == "bicubic"
        else cv2.INTER_LINEAR
    )

    train_transforms = A.Compose(
        [
            A.RandomResizedCrop(
                size=(img_size, img_size), scale=(0.8, 1.0), interpolation=interpolation
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=0.4),
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(1, max(1, img_size // 8)),
                hole_width_range=(1, max(1, img_size // 8)),
                p=0.3,
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=img_size, width=img_size, interpolation=interpolation),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    return train_transforms, val_transforms


def get_dataloaders(
    cfg,
    data_config,
    rank: int = 0,
    world_size: int = 1,
    use_ddp: bool = False,
):
    df = pd.read_csv(cfg.path.train_csv)
    train_df, val_df = train_test_split(
        df,
        test_size=cfg.train.val_size,
        stratify=df["label"],
        random_state=cfg.train.seed,
    )

    train_trans, val_trans = get_transforms(data_config)
    train_dataset = CustomDataset(train_df, cfg.path.img_dir, transforms=train_trans)
    val_dataset = CustomDataset(val_df, cfg.path.img_dir, transforms=val_trans)

    train_sampler = None
    val_sampler = None
    train_shuffle = True

    if use_ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=cfg.train.seed,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )

        train_shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader
