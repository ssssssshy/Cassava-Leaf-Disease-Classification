import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        pt = torch.exp(-ce_loss)

        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            self.alpha = self.alpha.to(focal_loss.device)
            focal_loss = focal_loss * self.alpha[targets]

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


if __name__ == "__main__":
    dummy_logits = torch.randn(2, 5)
    dummy_targets = torch.tensor([1, 4])

    loss_fn = FocalLoss(gamma=2.0)
    loss_val = loss_fn(dummy_logits, dummy_targets)

    print(f"Focal Loss: {loss_val.item():.4f}")
