"""
PyTorch Dataset classes for beat-level ECG data.
Load from serialized tensors and apply split indices from JSON.
"""
import json
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from torch.utils.data import Dataset

from utils import DATA_DIR, SPLITS_DIR


class ECGBeatDataset(Dataset):
    """Beat-level ECG dataset with X (N, seq_len, C) and y (N,) class indices."""

    def __init__(
        self,
        data_path: Path,
        split: str,
        split_name: str = "inter_patient",
        splits_dir: Path | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        """
        Args:
            data_path: Path to processed .pt file (dict with 'X', 'y', 'record_ids').
            split: 'train', 'val', or 'test'.
            split_name: 'inter_patient' or 'intra_patient'.
            splits_dir: Directory containing intra_patient.json and inter_patient.json.
            transform: Optional augmentation applied to X (train only).
        """
        self.splits_dir = Path(splits_dir or SPLITS_DIR)
        data = torch.load(data_path, map_location="cpu", weights_only=True)
        self.X = data["X"]
        self.y = data["y"]
        with open(self.splits_dir / f"{split_name}.json") as f:
            splits = json.load(f)
        self.indices = splits[split]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        i = self.indices[idx]
        x = self.X[i].clone()
        if self.transform is not None:
            x = self.transform(x)
        return x, int(self.y[i].item())
