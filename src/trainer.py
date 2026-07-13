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
        val_len: int,
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
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_len = val_len
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
        total_samples = 0

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

            # ИЗМЕНЕНИЕ: Отбираем только параметры с requires_grad=True
            # Это позволяет без ошибок замораживать backbone модели
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            if trainable_params:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_loss = loss.item()
            running_loss += batch_loss
            total_samples += labels.size(0)

            self.metrics.update(outputs.detach(), labels)
            pbar.set_postfix({"Loss": f"{batch_loss:.4f}"})

        acc, f1, prec, rec, _ = self.metrics.compute()
        avg_loss = running_loss / max(total_samples / self.train_loader.batch_size, 1)
        avg_loss = reduce_mean(avg_loss, self.world_size, self.device)

        return avg_loss, acc.item(), f1.item(), prec.item(), rec.item()

    @torch.inference_mode()
    def _val_epoch(self):
        self.model.eval()
        self.metrics.reset()
        running_loss = 0.0
        total_samples = 0

        pbar = tqdm(self.val_loader, desc="Validation", disable=not self.is_main)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(self.amp_device_type, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            batch_samples = labels.size(0)
            running_loss += loss.item() * batch_samples
            total_samples += batch_samples
            self.metrics.update(outputs, labels)

        acc, f1, prec, rec, _ = self.metrics.compute()
        avg_loss = running_loss / max(total_samples, 1)
        avg_loss = reduce_mean(avg_loss, self.world_size, self.device)

        return avg_loss, acc.item(), f1.item(), prec.item(), rec.item()

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
        history = {
            "train_loss": [],
            "train_f1": [],
            "train_prec": [],
            "train_rec": [],
            "val_loss": [],
            "val_f1": [],
            "val_prec": [],
            "val_rec": [],
        }

        for epoch in range(1, epochs + 1):
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.is_main:
                print(f"\nЭпоха {epoch} | LR: {current_lr:.6f}")

            t_loss, t_acc, t_f1, t_prec, t_rec = self._train_epoch(epoch)
            v_loss, v_acc, v_f1, v_prec, v_rec = self._val_epoch()

            self._step_scheduler(v_loss, v_f1)

            history["train_loss"].append(t_loss)
            history["train_f1"].append(t_f1)
            history["train_prec"].append(t_prec)
            history["train_rec"].append(t_rec)

            history["val_loss"].append(v_loss)
            history["val_f1"].append(v_f1)
            history["val_prec"].append(v_prec)
            history["val_rec"].append(v_rec)

            if self.is_main:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss": t_loss,
                        "train/acc": t_acc,
                        "train/f1": t_f1,
                        "train/precision": t_prec,
                        "train/recall": t_rec,
                        "val/loss": v_loss,
                        "val/acc": v_acc,
                        "val/f1": v_f1,
                        "val/precision": v_prec,
                        "val/recall": v_rec,
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
