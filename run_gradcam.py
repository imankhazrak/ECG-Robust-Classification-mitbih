#!/usr/bin/env python3
"""
Generate Grad-CAM visualizations for CNN + Transformer ECG model.
Usage: python run_gradcam.py --checkpoint checkpoints/best_model.pt --data data/mitbih.csv --output results/gradcam
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import CNNTransformer
from utils.dataset import load_csv, CLASS_NAMES
from utils.gradcam import grad_cam, plot_gradcam
from config import get_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/mitbih.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/gradcam"))
    parser.add_argument("-n", type=int, default=5, help="Number of samples to visualize")
    args = parser.parse_args()

    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data, labels = load_csv(args.data)
    model = CNNTransformer(
        seq_len=187,
        in_channels=1,
        num_classes=5,
        base_filters=config.base_filters,
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    indices = np.random.choice(len(data), min(args.n, len(data)), replace=False)
    args.output.mkdir(parents=True, exist_ok=True)

    for i, idx in enumerate(indices):
        x = torch.from_numpy(data[idx : idx + 1]).float().unsqueeze(1)
        heatmap, pred_class = grad_cam(model, x.to(device))
        plot_gradcam(
            data[idx : idx + 1],
            heatmap,
            pred_class,
            CLASS_NAMES,
            args.output / f"gradcam_{i}.png",
        )
        print(f"Saved {args.output}/gradcam_{i}.png (pred={CLASS_NAMES[pred_class]})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
