import argparse
import random

import numpy as np
import torch
import wandb

from src.config import load_config
from src.data import get_dataloaders
from src.losses import FocalLoss
from src.metrics import CassavaMetrics
from src.models import build_model
from src.trainer import ModelTrainer
from src.utils import EarlyStopping


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение модели Cassava Leaf Disease")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Путь к yaml-конфигу",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(cfg.train.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    wandb.init(
        project=cfg.wandb.project_name,
        name=cfg.wandb.run_name,
        config=cfg.model_dump(),
    )

    train_loader, val_loader = get_dataloaders(cfg)

    model = build_model(cfg)

    criterion = FocalLoss(gamma=cfg.train.focal_gamma)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.epochs
    )

    metrics = CassavaMetrics(cfg, device=device)

    early_stopper = EarlyStopping(
        patience=cfg.early_stopping.patience,
        min_delta=cfg.early_stopping.min_delta,
        mode=cfg.early_stopping.mode,
    )

    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metrics,
        device=device,
        save_path="weights/best_model.pth",
    )

    history = trainer.fit(epochs=cfg.train.epochs, early_stopper=early_stopper)

    print("\nОбучение завершено")
    print(f"Лучший Val F1: {max(history['val_f1']):.4f}")

    wandb.finish()


if __name__ == "__main__":
    main()
