import os
from typing import Optional, Dict, Any

import torch
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
from tqdm import tqdm

from src.utils import EarlyStopping
from src.distributed import is_main_process, reduce_mean


class ModelTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        metrics,
        device: torch.device,
        save_path: str = "weights/best_model.pth",
        use_amp: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics
        self.device = device
        self.save_path = save_path
        self.best_val_f1 = -float("inf")

        self.rank = rank
        self.world_size = world_size
        self.is_main = is_main_process(rank)

        self.amp_device_type = device.type
        self.use_amp = use_amp and device.type == "cuda"
        self.scaler = GradScaler(self.amp_device_type, enabled=self.use_amp)

        if self.is_main:
            save_dir = os.path.dirname(self.save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)

    def _unwrap_model(self) -> torch.nn.Module:
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _train_epoch(self, epoch: int):
        self.model.train()
        self.metrics.reset()
        running_loss = 0.0

        if isinstance(self.train_loader.sampler, DistributedSampler):
            self.train_loader.sampler.set_epoch(epoch)

        pbar = tqdm(self.train_loader, desc="Training", disable=not self.is_main)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(self.amp_device_type, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_loss = loss.item()
            running_loss += batch_loss

            self.metrics.update(outputs.detach(), labels)

            pbar.set_postfix({"Loss": f"{batch_loss:.4f}"})

        acc, f1, _ = self.metrics.compute()
        avg_loss = running_loss / len(self.train_loader)
        avg_loss = reduce_mean(avg_loss, self.world_size, self.device)
        return avg_loss, acc, f1

    @torch.no_grad()
    def _val_epoch(self):
        self.model.eval()
        self.metrics.reset()
        running_loss = 0.0

        pbar = tqdm(self.val_loader, desc="Validation", disable=not self.is_main)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(self.amp_device_type, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            running_loss += loss.item()
            self.metrics.update(outputs, labels)

        acc, f1, _ = self.metrics.compute()
        avg_loss = running_loss / len(self.val_loader)
        avg_loss = reduce_mean(avg_loss, self.world_size, self.device)
        return avg_loss, acc, f1

    def _save_checkpoint(self, epoch: int, v_f1: float):

        if not self.is_main:
            return
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self._unwrap_model().state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_f1": v_f1,
            },
            self.save_path,
        )

    def _step_scheduler(self, v_loss: float, v_f1: float):
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(v_f1)
        else:
            self.scheduler.step()

    def fit(
        self, epochs: int, early_stopper: Optional[EarlyStopping] = None
    ) -> Dict[str, Any]:
        history = {"train_loss": [], "train_f1": [], "val_loss": [], "val_f1": []}

        for epoch in range(1, epochs + 1):
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.is_main:
                print(f"\nЭпоха {epoch} | LR: {current_lr:.6f}")

            t_loss, t_acc, t_f1 = self._train_epoch(epoch)

            v_loss, v_acc, v_f1 = self._val_epoch()

            self._step_scheduler(v_loss, v_f1)

            history["train_loss"].append(t_loss)
            history["train_f1"].append(t_f1)
            history["val_loss"].append(v_loss)
            history["val_f1"].append(v_f1)

            if self.is_main:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss": t_loss,
                        "train/acc": t_acc,
                        "train/f1": t_f1,
                        "val/loss": v_loss,
                        "val/acc": v_acc,
                        "val/f1": v_f1,
                        "lr": current_lr,
                    }
                )
                print(f"Train F1: {t_f1:.4f} | Val F1: {v_f1:.4f}")

            if v_f1 > self.best_val_f1:
                self.best_val_f1 = v_f1
                self._save_checkpoint(epoch, v_f1)
                if self.is_main:
                    print(
                        f"Новый лучший чекпоинт сохранён: {self.save_path} (F1={v_f1:.4f})"
                    )

            if early_stopper:
                early_stopper(v_f1)
                if early_stopper.early_stop:
                    if self.is_main:
                        print("Ранняя остановка!")
                    break

        return history
