"""
Main training script: argparse --config and --split; early stopping on validation Macro-F1.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from utils import (
    set_seed,
    PROJECT_ROOT,
    DATA_DIR,
    SPLITS_DIR,
    RESULTS_DIR,
    RESULTS_LOGS,
    AAMI_CLASSES,
    NUM_CLASSES,
    ensure_dirs,
    setup_logging,
)
from datasets import ECGBeatDataset
from models import ECG_LSTM, ECG_CNN1D
from augmentation import build_augment


def get_model(config: dict, seq_len: int, in_channels: int, device: torch.device) -> nn.Module:
    name = config["model"].lower()
    if name == "lstm":
        model = ECG_LSTM(
            input_size=in_channels,
            hidden_size=config.get("hidden_size", 128),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.3),
            num_classes=NUM_CLASSES,
        )
    elif name == "cnn":
        model = ECG_CNN1D(in_channels=in_channels, num_classes=NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model: {name}")
    return model.to(device)


class FocalLoss(nn.Module):
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


def main():
    parser = argparse.ArgumentParser(description="Train ECG classifier")
    parser.add_argument("--config", type=Path, required=True, help="Config YAML")
    parser.add_argument("--split", type=str, default="inter_patient", choices=["inter_patient", "intra_patient"])
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    parser.add_argument("--data-fraction", type=float, default=1.0, help="Use fraction of training data (for small-data experiments)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    set_seed(seed)
    ensure_dirs()
    log_path = RESULTS_LOGS / f"seed_{seed}_{config['model']}_{args.split}.log"
    logger = setup_logging(log_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = config.get("data_path", str(DATA_DIR / "processed_mlii.pt"))
    data_path = Path(data_path) if os.path.isabs(data_path) else (PROJECT_ROOT / data_path)
    augment = build_augment(config)
    train_ds = ECGBeatDataset(data_path, "train", split_name=args.split, splits_dir=SPLITS_DIR, transform=augment)
    val_ds = ECGBeatDataset(data_path, "val", split_name=args.split, splits_dir=SPLITS_DIR)

    if args.data_fraction < 1.0:
        n = len(train_ds)
        keep = max(1, int(n * args.data_fraction))
        train_ds = torch.utils.data.Subset(train_ds, torch.randperm(n, generator=torch.Generator().manual_seed(seed))[:keep].tolist())

    train_loader = DataLoader(
        train_ds,
        batch_size=config.get("batch_size", 128),
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(val_ds, batch_size=config.get("batch_size", 128), shuffle=False)

    x0, _ = train_ds[0]
    seq_len, in_channels = x0.shape[0], x0.shape[1]
    model = get_model(config, seq_len, in_channels, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 1e-3))
    loss_type = config.get("loss", "ce").lower()
    if loss_type == "weighted_ce":
        from torch.utils.data import DataLoader as DL
        from collections import Counter
        labels = [train_ds[i][1] for i in range(len(train_ds))]
        counts = Counter(labels)
        weights = torch.tensor([1.0 / (counts.get(c, 1) + 1e-5) for c in range(NUM_CLASSES)], dtype=torch.float32, device=device)
        weights = weights / weights.sum() * NUM_CLASSES
        criterion = nn.CrossEntropyLoss(weight=weights)
    elif loss_type == "focal":
        criterion = FocalLoss(gamma=config.get("focal_gamma", 2.0))
    else:
        criterion = nn.CrossEntropyLoss()

    epochs = config.get("epochs", 100)
    patience = config.get("patience", 15)
    best_f1 = -1.0
    wait = 0
    ckpt_dir = RESULTS_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    frac_suffix = f"_frac{args.data_fraction}" if args.data_fraction < 1.0 else ""
    ckpt_path = ckpt_dir / f"{config['model']}_{args.split}_seed{seed}{frac_suffix}.pt"

    # NOTE: val_macro_f1 fluctuates because (1) inter-patient: val = unseen patients,
    # (2) class imbalance differs per patient, (3) weighted_ce optimizes minority classes
    # so train loss can decrease while val F1 oscillates. Train F1 tracks memorization.
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_preds_list, train_labels_list = [], []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_preds_list.append(logits.argmax(1).detach().cpu().numpy())
            train_labels_list.append(y.cpu().numpy())
        train_loss /= len(train_loader)
        train_preds = np.concatenate(train_preds_list)
        train_labels = np.concatenate(train_labels_list)
        train_f1 = f1_score(train_labels, train_preds, average="macro", zero_division=0)

        model.eval()
        preds_list, labels_list = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)
                preds_list.append(logits.argmax(1).cpu().numpy())
                labels_list.append(y.numpy())
        preds = np.concatenate(preds_list)
        labels = np.concatenate(labels_list)
        val_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        logger.info(f"Epoch {epoch+1} train_loss={train_loss:.4f} train_macro_f1={train_f1:.4f} val_macro_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            wait = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_macro_f1": val_f1, "config": config}, ckpt_path)
        else:
            wait += 1
        # Early stopping disabled: run full epochs (config epochs=100)
        # if wait >= patience:
        #     logger.info(f"Early stopping at epoch {epoch+1}")
        #     break
    logger.info(f"Best val Macro-F1: {best_f1:.4f}, checkpoint: {ckpt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
