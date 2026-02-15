"""
CNN + Transformer Hybrid for ECG beat classification.
Input: (B, 1, 187) -> CNN feature extractor -> Transformer encoder -> Classification head.
"""
import math
from typing import Optional

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv1d block with BatchNorm, ReLU, optional residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        residual: bool = False,
    ):
        super().__init__()
        self.residual = residual and in_channels == out_channels
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn(out)
        if self.residual:
            out = out + x
        return self.relu(out)


class CNNFeatureExtractor(nn.Module):
    """CNN feature extractor: several Conv1D blocks with BatchNorm, ReLU, pooling."""

    def __init__(
        self,
        in_channels: int = 1,
        base_filters: int = 32,
        kernel_size: int = 5,
        num_blocks: int = 3,
    ):
        super().__init__()
        blocks = []
        ch = in_channels
        for i in range(num_blocks):
            out_ch = base_filters * (2**i)
            blocks.append(ConvBlock(ch, out_ch, kernel_size, residual=(i > 0)))
            blocks.append(nn.MaxPool1d(2))
            ch = out_ch
        self.features = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class CNNTransformer(nn.Module):
    """
    CNN + Transformer Hybrid for 5-class ECG beat classification.
    Input shape: (B, 1, 187).
    """

    def __init__(
        self,
        seq_len: int = 187,
        in_channels: int = 1,
        num_classes: int = 5,
        base_filters: int = 32,
        kernel_size: int = 5,
        num_conv_blocks: int = 3,
        d_model: Optional[int] = None,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.cnn = CNNFeatureExtractor(in_channels, base_filters, kernel_size, num_conv_blocks)

        # Infer d_model from CNN output: 187 -> /2/2/2 = 23 (approx) with 3 pools
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, seq_len)
            cnn_out = self.cnn(dummy)
            _, cnn_channels, cnn_seq_len = cnn_out.shape

        d_model = d_model or cnn_channels
        dim_feedforward = dim_feedforward or (4 * d_model)

        self.d_model = d_model
        self.cnn_seq_len = cnn_seq_len
        self.cnn_channels = cnn_channels

        # Project CNN channels to d_model if needed
        self.proj = (
            nn.Linear(cnn_channels, d_model)
            if cnn_channels != d_model
            else nn.Identity()
        )

        # Learnable positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, cnn_seq_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 187) input ECG beats.
        Returns:
            logits: (B, num_classes)
        """
        # CNN: (B, 1, 187) -> (B, C, L)
        x = self.cnn(x)

        # Transpose to (B, L, C) for Transformer
        x = x.transpose(1, 2)

        # Project to d_model
        x = self.proj(x)

        # Add positional encoding
        x = x + self.pos_embed[:, : x.size(1), :]

        # Transformer
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Classifier
        x = self.dropout(x)
        return self.classifier(x)
