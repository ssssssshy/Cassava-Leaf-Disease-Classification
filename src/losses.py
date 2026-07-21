import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        ce_loss_clean = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss_clean)

        ce_loss_smoothed = F.cross_entropy(
            inputs, targets, label_smoothing=self.label_smoothing, reduction="none"
        )

        focal_loss = ((1 - pt) ** self.gamma) * ce_loss_smoothed

        if self.alpha is not None:
            focal_loss = focal_loss * self.alpha[targets]

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


if __name__ == "__main__":
    dummy_logits = torch.randn(2, 5)
    dummy_targets = torch.tensor([1, 4])

    loss_fn = FocalLoss(gamma=2.0, label_smoothing=0.1)
    loss_val = loss_fn(dummy_logits, dummy_targets)
    print(f"Focal Loss (with Label Smoothing): {loss_val.item():.4f}")
