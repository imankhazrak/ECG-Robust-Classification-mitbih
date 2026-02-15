"""
2-layer LSTM for ECG beat classification (5 AAMI classes).
Input shape: (batch, sequence_length, channels).
"""
import torch
import torch.nn as nn


class ECG_LSTM(nn.Module):
    """Two-layer LSTM with dropout and FC classifier."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 5,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.drop(out)
        logits = self.fc(out)
        return logits
