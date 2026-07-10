import torch
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
)

from src.config import load_config


class CassavaMetrics:
    def __init__(self, cfg, device: torch.device):

        # sync_on_compute=True явно указан — при DDP torchmetrics синхронизирует
        # внутреннее состояние (confusion matrix) между ранками ПЕРЕД вычислением метрики,
        # что даёт корректные глобальные accuracy/F1 без ручного reduce_mean
        self.accuracy = MulticlassAccuracy(
            num_classes=cfg.model.num_classes,
            average="micro",
            sync_on_compute=True,  # явная синхронизация при DDP
        ).to(device)

        # sync_on_compute=True для macro F1 — критично: macro F1 НЕ линейна,
        # простое усреднение per-rank macro F1 даёт неверный результат;
        # синхронизация confusion matrix между ранками обеспечивает корректный глобальный F1
        self.f1_macro = MulticlassF1Score(
            num_classes=cfg.model.num_classes,
            average="macro",
            sync_on_compute=True,  # явная синхронизация для корректного macro F1
        ).to(device)

        self.conf_matrix = MulticlassConfusionMatrix(
            num_classes=cfg.model.num_classes,
            sync_on_compute=True,  # синхронизация confusion matrix для DDP
        ).to(device)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):

        self.accuracy.update(preds, targets)
        self.f1_macro.update(preds, targets)
        self.conf_matrix.update(preds, targets)

    def compute(self):
        # убран .item() — возвращаем тензоры, чтобы DDP sync работал корректно;
        # .item() вызывает immediate materialization, что может нарушить синхронизацию
        acc = self.accuracy.compute()
        f1 = self.f1_macro.compute()
        cm = self.conf_matrix.compute().cpu().numpy()

        return acc, f1, cm

    def reset(self):

        self.accuracy.reset()
        self.f1_macro.reset()
        self.conf_matrix.reset()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config("configs/baseline.yaml")
    metrics = CassavaMetrics(cfg, device=device)

    batch1_preds = torch.randn(32, 5).to(device)
    batch1_targets = torch.randint(0, 5, (32,)).to(device)

    metrics.update(batch1_preds, batch1_targets)
    acc, f1, cm = metrics.compute()

    print(f"Accuracy: {acc * 100:.2f}% | Macro F1: {f1:.4f}")
    print("Confusion Matrix:\n", cm)
