"""
Loss functions for imbalanced ECG classification.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification. Down-weights easy examples."""

    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        fl = (1 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return fl.mean()
        return fl.sum()


class MixupLoss:
    """Wrapper to compute mixed CE when target is (y_a, y_b, lam)."""

    @staticmethod
    def forward(
        criterion: nn.Module,
        logits: torch.Tensor,
        target: tuple[torch.Tensor, torch.Tensor, float],
    ) -> torch.Tensor:
        y_a, y_b, lam = target
        return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


def get_criterion(
    loss_type: str = "weighted_ce",
    class_weights: Optional[torch.Tensor] = None,
    focal_gamma: float = 2.0,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Factory for loss criterion.
    Args:
        loss_type: 'weighted_ce' or 'focal' or 'ce'
        class_weights: (num_classes,) for weighted CE
        focal_gamma: gamma for Focal loss
        device: device for weights
    """
    if loss_type == "weighted_ce" and class_weights is not None:
        if device is not None:
            class_weights = class_weights.to(device)
        return nn.CrossEntropyLoss(weight=class_weights)
    if loss_type == "focal":
        return FocalLoss(gamma=focal_gamma)
    return nn.CrossEntropyLoss()
