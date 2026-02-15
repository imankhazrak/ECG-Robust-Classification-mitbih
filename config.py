"""
Configuration for CNN + Transformer ECG training.
All hyperparameters in one place; no hardcoded values in training code.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Training and model configuration."""

    data_path: Path = Path("data/mitbih.csv")
    train_frac: float = 0.8
    val_frac: float = 0.15
    seq_len: int = 187
    num_classes: int = 5

    base_filters: int = 32
    kernel_size: int = 5
    num_conv_blocks: int = 3

    d_model: int = 0
    nhead: int = 4
    num_encoder_layers: int = 2
    dim_feedforward: int = 0
    dropout: float = 0.3

    batch_size: int = 64
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    patience: int = 15

    loss_type: str = "weighted_ce"
    focal_gamma: float = 2.0

    mixup_alpha: float = 0.0

    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("logs")
    results_dir: Path = Path("results")

    seed: int = 42


def get_config() -> Config:
    return Config()
