#!/usr/bin/env python3
"""
Generate confusion matrix plots from CNN+Transformer metrics.
Usage: python scripts/plot_confusion_matrices.py [--results-dir results]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from utils.dataset import CLASS_NAMES


def plot_cm(cm: np.ndarray, save_path: Path, title: str = "Confusion Matrix") -> None:
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    args = parser.parse_args()

    results_dir = args.results_dir
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    seeds = [42, 1, 2, 3, 4]
    all_cms = []

    for seed in seeds:
        p = results_dir / f"metrics_seed{seed}.json"
        if not p.exists():
            print(f"Warning: {p} not found, skipping")
            continue
        with open(p) as f:
            data = json.load(f)
        cm = np.array(data["confusion_matrix"])
        all_cms.append(cm)
        plot_cm(cm, figures_dir / f"confusion_cnn_transformer_seed{seed}.png", f"CNN+Transformer Seed {seed}")
        print(f"Saved {figures_dir}/confusion_cnn_transformer_seed{seed}.png")

    if all_cms:
        agg_cm = np.sum(all_cms, axis=0)
        plot_cm(agg_cm, figures_dir / "confusion_cnn_transformer_aggregate.png", "CNN+Transformer Aggregate (5 seeds)")
        print(f"Saved {figures_dir}/confusion_cnn_transformer_aggregate.png")

        # Save aggregate as CSV for easy viewing
        csv_path = results_dir / "confusion_matrix_cnn_transformer_aggregate.csv"
        np.savetxt(csv_path, agg_cm, fmt="%d", delimiter=",", header=",".join(CLASS_NAMES), comments="")
        print(f"Saved {csv_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
