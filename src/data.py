import pathlib
import warnings  # добавлен для предупреждения при singleton-классах

import torch
import pandas as pd

from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
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

        # #13: удалён закомментированный блок старого варианта (OpenCV + Albumentations)

        try:
            image = read_image(str(img_path), mode=ImageReadMode.RGB)
        except Exception as e:
            raise FileNotFoundError(f"Ошибка загрузки картинки: {img_path}") from e

        if self.transforms:
            image = self.transforms(image)

        return image, torch.tensor(label, dtype=torch.long)


# добавлен параметр img_size — теперь размер изображения берётся из конфига, а не молча игнорируется;
def get_transforms(data_config, img_size: int):
    mean = data_config["mean"]
    std = data_config["std"]

    interpolation = (
        InterpolationMode.BICUBIC
        if data_config["interpolation"] == "bicubic"
        else InterpolationMode.BILINEAR
    )

    # img_size теперь явный параметр вместо data_config["input_size"][1],
    # что позволяет управлять размером через конфиг
    train_transforms = v2.Compose(
        [
            v2.RandomResizedCrop(
                size=(img_size, img_size), scale=(0.8, 1.0), interpolation=interpolation
            ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomApply(
                [v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0)],
                p=0.4,
            ),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
            v2.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
        ]
    )

    val_transforms = v2.Compose(
        [
            v2.Resize(size=(img_size, img_size), interpolation=interpolation),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
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

    # обработка singleton-классов — train_test_split(stratify=) падает,
    # если в каком-либо классе всего 1 сэмпл; добавлена проверка и fallback
    class_counts = df["label"].value_counts()
    min_class_count = class_counts.min()
    can_stratify = min_class_count >= 2  # #7: stratify требует ≥2 сэмплов на класс

    if not can_stratify:
        # warning вместо silent fallback — пользователь должен знать о проблеме
        warnings.warn(
            f"Обнаружен класс с {min_class_count} сэмплом(ами): "
            f"stratified split невозможен, используется обычный split",
            stacklevel=2,
        )

    train_df, val_df = train_test_split(
        df,
        test_size=cfg.train.val_size,
        # stratify только если все классы имеют ≥2 сэмплов
        stratify=df["label"] if can_stratify else None,
        random_state=cfg.train.seed,
    )

    # передаём img_size из конфига вместо data_config["input_size"][1]
    train_trans, val_trans = get_transforms(data_config, img_size=cfg.train.img_size)
    train_dataset = CustomDataset(train_df, cfg.path.img_dir, transforms=train_trans)
    val_dataset = CustomDataset(val_df, cfg.path.img_dir, transforms=val_trans)

    train_sampler = None
    val_sampler = None
    train_shuffle = True

    # сохраняем реальный размер val набора ДО дублирования сэмплов в DistributedSampler
    val_len = len(val_dataset)

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

    # drop_last изменён с True на False — при highly-imbalanced данных
    # последний батч может содержать диспропорционально много миноритарных классов,
    # drop_last=True безвозвратно теряет эти сэмплы каждую эпоху
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=False,  # не теряем последний батч с потенциально редкими классами
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

    # возвращаем реальный размер val набора для корректного усреднения loss в DDP
    return train_loader, val_loader, val_len
