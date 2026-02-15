"""
1D CNN baseline for ECG beat classification (5 AAMI classes).
Three conv blocks (BatchNorm + ReLU), global average pooling, FC classifier.
"""
import torch
import torch.nn as nn


def _conv_block(in_c: int, out_c: int, k: int = 5) -> nn.Module:
    return nn.Sequential(
        nn.Conv1d(in_c, out_c, kernel_size=k, padding=k // 2),
        nn.BatchNorm1d(out_c),
        nn.ReLU(inplace=True),
    )


class ECG_CNN1D(nn.Module):
    """1D CNN: 3 conv blocks, global average pooling, FC."""

    def __init__(
        self,
        in_channels: int = 1,
        base_filters: int = 32,
        num_classes: int = 5,
    ):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(in_channels, base_filters, 5),
            nn.MaxPool1d(2),
            _conv_block(base_filters, base_filters * 2, 5),
            nn.MaxPool1d(2),
            _conv_block(base_filters * 2, base_filters * 4, 5),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(base_filters * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> conv expects (B, C, T)
        x = x.transpose(1, 2)
        x = self.features(x)
        x = x.squeeze(-1)
        return self.fc(x)
