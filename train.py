import argparse
import os
import random

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

from src.config import load_config
from src.data import get_dataloaders
from src.losses import FocalLoss
from src.metrics import CassavaMetrics
from src.models import build_model
from src.trainer import ModelTrainer
from src.utils import EarlyStopping
from src.distributed import setup_ddp, cleanup_ddp, is_main_process


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

    use_ddp = (
        "RANK" in os.environ
        and "WORLD_SIZE" in os.environ
        and int(os.environ.get("WORLD_SIZE", "1")) > 1
    )

    if use_ddp:
        rank, local_rank, world_size = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(cfg.train.seed)

    if is_main_process(rank):
        print(f"Устройство: {device} | DDP: {use_ddp} | World size: {world_size}")
        wandb.init(
            project=cfg.wandb.project_name,
            name=cfg.wandb.run_name,
            config=cfg.model_dump(),
        )

    train_loader, val_loader = get_dataloaders(
        cfg, rank=rank, world_size=world_size, use_ddp=use_ddp
    )

    model = build_model(cfg).to(device)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    criterion = FocalLoss(gamma=cfg.train.focal_gamma)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)

    """scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.epochs
    )"""

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
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
        rank=rank,
        world_size=world_size,
    )

    history = trainer.fit(epochs=cfg.train.epochs, early_stopper=early_stopper)

    if is_main_process(rank):
        print("\nОбучение завершено")
        print(f"Лучший Val F1: {max(history['val_f1']):.4f}")
        wandb.finish()

    if use_ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()
