"""
Corruption generation and robustness evaluation.
Corruptions: Gaussian noise (SNR 20/10/5 dB), missing segments (5/10/20%), temporal jitter (±10/±20/±30 ms).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

from utils import (
    set_seed,
    PROJECT_ROOT,
    DATA_DIR,
    SPLITS_DIR,
    RESULTS_DIR,
    RESULTS_FIGURES,
    NUM_CLASSES,
    SAMPLE_RATE,
    ensure_dirs,
)
from datasets import ECGBeatDataset
from models import ECG_LSTM, ECG_CNN1D


def add_gaussian_noise(x: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Add Gaussian noise to achieve target SNR in dB. x: (B, T, C)."""
    signal_power = (x ** 2).mean()
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(x, device=x.device) * (noise_power ** 0.5)
    return x + noise


def add_missing_segments(x: torch.Tensor, fraction: float) -> torch.Tensor:
    """Randomly zero out fraction of the signal (mask segments). x: (B, T, C)."""
    out = x.clone()
    B, T, C = x.shape
    n_missing = max(1, int(T * fraction))
    for b in range(B):
        start = torch.randint(0, T - n_missing + 1, (1,)).item() if T > n_missing else 0
        out[b, start : start + n_missing, :] = 0
    return out


def temporal_jitter(x: torch.Tensor, max_shift_ms: float) -> torch.Tensor:
    """Shift beat window by random amount in [-max_shift_ms, +max_shift_ms] (circular)."""
    shift_samples = int(round(SAMPLE_RATE * max_shift_ms / 1000.0))
    if shift_samples == 0:
        return x
    B, T, C = x.shape
    shifts = np.random.randint(-shift_samples, shift_samples + 1, size=B)
    out = x.clone()
    for b in range(B):
        out[b] = torch.roll(x[b], int(shifts[b]), dims=0)
    return out


def evaluate_corrupted(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    corruption: str,
    level: float,
    apply_fn,
) -> float:
    model.eval()
    preds_list, labels_list = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            x = apply_fn(x)
            logits = model(x)
            preds_list.append(logits.argmax(1).cpu().numpy())
            labels_list.append(y.numpy())
    preds = np.concatenate(preds_list)
    labels = np.concatenate(labels_list)
    return f1_score(labels, preds, average="macro", zero_division=0)


def main():
    parser = argparse.ArgumentParser(description="Robustness evaluation under corruptions")
    parser.add_argument("--model", type=str, required=True, choices=["lstm", "cnn"])
    parser.add_argument("--corruption", type=str, required=True, choices=["gaussian", "missing", "jitter"])
    parser.add_argument("--config", type=Path, default=None, help="Config YAML (default: configs/<model>_baseline.yaml)")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="inter_patient")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seeds", type=str, default="42,1,2,3,4", help="Comma-separated seeds for checkpoint choice")
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dirs()
    config_path = args.config or (PROJECT_ROOT / "configs" / f"{args.model}_baseline.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    data_path = PROJECT_ROOT / config.get("data_path", "data/processed_mlii.pt")
    ckpt_dir = args.checkpoint_dir or (RESULTS_DIR / "checkpoints")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = ECGBeatDataset(data_path, "test", split_name=args.split, splits_dir=SPLITS_DIR)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    x0, _ = test_ds[0]
    seq_len, in_channels = x0.shape[0], x0.shape[1]

    if config["model"].lower() == "lstm":
        model = ECG_LSTM(input_size=in_channels, hidden_size=128, num_layers=2, dropout=0.3, num_classes=NUM_CLASSES)
    else:
        model = ECG_CNN1D(in_channels=in_channels, num_classes=NUM_CLASSES)
    model = model.to(device)

    seeds = [int(s) for s in args.seeds.split(",")]
    rows = []

    if args.corruption == "gaussian":
        levels = [20.0, 10.0, 5.0]
        def apply(x, level):
            return add_gaussian_noise(x, level)
    elif args.corruption == "missing":
        levels = [0.05, 0.10, 0.20]
        def apply(x, level):
            return add_missing_segments(x, level)
    else:
        levels = [10.0, 20.0, 30.0]
        def apply(x, level):
            return temporal_jitter(x, level)

    for seed in seeds:
        ckpt_path = ckpt_dir / f"{args.model}_{args.split}_seed{seed}.pt"
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        for level in levels:
            def apply_fn(x):
                return apply(x, level)
            macro_f1 = evaluate_corrupted(model, loader, device, args.corruption, level, apply_fn)
            rows.append({"model": args.model, "corruption": args.corruption, "level": level, "seed": seed, "macro_f1": macro_f1})

    if not rows:
        print("No checkpoints found; run training first.")
        return 1
    out_csv = RESULTS_DIR / "robustness_curves.csv"
    import csv
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "corruption", "level", "seed", "macro_f1"])
        if not out_csv.exists() or out_csv.stat().st_size == 0:
            w.writeheader()
        w.writerows(rows)
    print(f"Appended {len(rows)} rows to {out_csv}")

    # Plot: average over seeds per level
    level_f1 = {}
    for r in rows:
        l = r["level"]
        level_f1.setdefault(l, []).append(r["macro_f1"])
    levels_sorted = sorted(level_f1.keys())
    means = [np.mean(level_f1[l]) for l in levels_sorted]
    stds = [np.std(level_f1[l]) for l in levels_sorted]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.errorbar(range(len(levels_sorted)), means, yerr=stds, capsize=4, marker="o")
        plt.xticks(range(len(levels_sorted)), [str(l) for l in levels_sorted])
        plt.xlabel(args.corruption + " level")
        plt.ylabel("Macro-F1")
        plt.title(f"Robustness: {args.model} ({args.corruption})")
        plt.tight_layout()
        fig_path = RESULTS_FIGURES / "robustness_plot.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path)
        plt.close()
        print(f"Saved {fig_path}")
    except Exception as e:
        print("Plot failed:", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
