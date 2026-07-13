import torch
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassConfusionMatrix,
)

from src.config import load_config


class CassavaMetrics:
    def __init__(self, cfg, device: torch.device):
        self.accuracy = MulticlassAccuracy(
            num_classes=cfg.model.num_classes,
            average="micro",
            sync_on_compute=True,
        ).to(device)

        self.f1_macro = MulticlassF1Score(
            num_classes=cfg.model.num_classes,
            average="macro",
            sync_on_compute=True,
        ).to(device)

        self.precision_macro = MulticlassPrecision(
            num_classes=cfg.model.num_classes,
            average="macro",
            sync_on_compute=True,
        ).to(device)

        self.recall_macro = MulticlassRecall(
            num_classes=cfg.model.num_classes,
            average="macro",
            sync_on_compute=True,
        ).to(device)

        self.conf_matrix = MulticlassConfusionMatrix(
            num_classes=cfg.model.num_classes,
            sync_on_compute=True,
        ).to(device)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        self.accuracy.update(preds, targets)
        self.f1_macro.update(preds, targets)
        self.precision_macro.update(preds, targets)
        self.recall_macro.update(preds, targets)
        self.conf_matrix.update(preds, targets)

    def compute(self):
        acc = self.accuracy.compute()
        f1 = self.f1_macro.compute()
        precision = self.precision_macro.compute()
        recall = self.recall_macro.compute()
        cm = self.conf_matrix.compute().cpu().numpy()

        return acc, f1, precision, recall, cm

    def reset(self):
        self.accuracy.reset()
        self.f1_macro.reset()
        self.precision_macro.reset()
        self.recall_macro.reset()
        self.conf_matrix.reset()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config("configs/baseline.yaml")
    metrics = CassavaMetrics(cfg, device=device)

    batch1_preds = torch.randn(32, 5).to(device)
    batch1_targets = torch.randint(0, 5, (32,)).to(device)

    metrics.update(batch1_preds, batch1_targets)

    acc, f1, prec, rec, cm = metrics.compute()

    print(f"Accuracy: {acc * 100:.2f}% | Macro F1: {f1:.4f}")
    print(f"Macro Precision: {prec:.4f} | Macro Recall: {rec:.4f}")
    print("Confusion Matrix:\n", cm)
