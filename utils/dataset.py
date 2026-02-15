"""
ECG Dataset for MIT-BIH CSV format (187 samples per beat).
Supports stratified train/val/test splits and optional Mixup augmentation.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


# MIT-BIH class labels (5 classes)
ID_TO_LABEL = {
    0: "Normal",
    1: "Atrial Premature",
    2: "Premature ventricular contraction",
    3: "Fusion of ventricular and normal",
    4: "Fusion of paced and normal",
}
CLASS_NAMES = [ID_TO_LABEL[i] for i in range(5)]
NUM_CLASSES = 5

# CSV: columns 0..186 = signal, column 187 (or 'label') = class index
SIGNAL_COLS = 187


class ECGDataset(Dataset):
    """
    Dataset for MIT-BIH ECG beats in CSV format.
    Each row: 187 signal values + 1 label.
    Output: (1, 187) tensor for Conv1d (channels, seq_len).
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        phase: str = "train",
        mixup_alpha: float = 0.0,
        mixup_rng: Optional[np.random.Generator] = None,
    ):
        """
        Args:
            data: (N, 187) float32 signal array.
            labels: (N,) int64 class indices 0-4.
            phase: 'train', 'val', or 'test'.
            mixup_alpha: Mixup alpha; 0 disables mixup.
            mixup_rng: Random generator for mixup (reproducibility).
        """
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()
        self.phase = phase
        self.mixup_alpha = mixup_alpha if phase == "train" else 0.0
        self.mixup_rng = mixup_rng or np.random.default_rng()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx].unsqueeze(0)  # (1, 187)
        y = self.labels[idx]

        if self.mixup_alpha > 0:
            lam = self.mixup_rng.beta(self.mixup_alpha, self.mixup_alpha)
            mix_idx = self.mixup_rng.integers(0, len(self.data))
            x_mix = self.data[mix_idx].unsqueeze(0)
            x = lam * x + (1 - lam) * x_mix
            y_mix = self.labels[mix_idx]
            # Return soft labels for mixup: (lam, 1-lam) over (y, y_mix)
            return x, (y, y_mix, lam)

        return x, y


def load_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load MIT-BIH CSV. Expects columns 0..186 = signal, 187 or 'label' = class.
    Returns (data, labels) as numpy arrays.
    """
    df = pd.read_csv(csv_path)

    # Handle column naming: 0..186 signal, 187 or 'label' = class
    if "label" in df.columns:
        label_col = "label"
        signal_cols = [c for c in df.columns if c != "label"]
    elif 187 in df.columns or "187" in df.columns:
        label_col = 187 if 187 in df.columns else "187"
        signal_cols = [c for c in df.columns if c != label_col]
    else:
        label_col = df.columns[-1]
        signal_cols = list(df.columns[:-1])

    signal_cols = signal_cols[:187]

    data = df[signal_cols].values.astype(np.float32)
    labels = df[label_col].values.astype(np.int64)

    # Ensure classes 0-4
    unique = np.unique(labels)
    if not np.all((unique >= 0) & (unique <= 4)):
        raise ValueError(f"Labels must be 0-4, got {unique}")

    return data, labels


def get_dataloaders(
    csv_path: Path,
    batch_size: int = 64,
    train_frac: float = 0.8,
    val_frac: float = 0.15,
    seed: int = 42,
    mixup_alpha: float = 0.0,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, test dataloaders with stratified splits.
    train_frac of full data -> train; of remainder, val_frac -> val, rest -> test.
    E.g. train_frac=0.8, val_frac=0.15: train=80%, val=12%, test=8%.
    """
    data, labels = load_csv(csv_path)
    rng = np.random.default_rng(seed)

    # First split: train+val vs test
    idx = np.arange(len(data))
    trainval_idx, test_idx = train_test_split(
        idx, test_size=1 - train_frac, stratify=labels, random_state=seed
    )
    trainval_data = data[trainval_idx]
    trainval_labels = labels[trainval_idx]
    test_data = data[test_idx]
    test_labels = labels[test_idx]

    # Second split: train vs val
    idx2 = np.arange(len(trainval_data))
    train_idx, val_idx = train_test_split(
        idx2, test_size=val_frac, stratify=trainval_labels, random_state=seed
    )
    train_data = trainval_data[train_idx]
    train_labels = trainval_labels[train_idx]
    val_data = trainval_data[val_idx]
    val_labels = trainval_labels[val_idx]

    train_ds = ECGDataset(train_data, train_labels, "train", mixup_alpha, rng)
    val_ds = ECGDataset(val_data, val_labels, "val")
    test_ds = ECGDataset(test_data, test_labels, "test")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


def get_class_weights(labels: np.ndarray, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Compute inverse frequency weights for weighted CE."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = 1.0 / (counts + 1e-5)
    weights = weights / weights.sum() * num_classes
    return torch.from_numpy(weights).float()
