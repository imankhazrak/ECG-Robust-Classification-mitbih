#!/usr/bin/env python3
"""
Prepare MIT-BIH ECG data in CSV format (187 samples per beat).

Options:
1. Convert existing processed_mlii.pt (360 samples) to 187-sample CSV via resampling.
2. Merge mitbih_train.csv + mitbih_test.csv if placed in data/ (from Kaggle heartbeat dataset).

Output: data/mitbih.csv with columns 0..186 (signal) and 187 (label).

Usage:
    python scripts/prepare_csv.py [--from-pt | --from-kaggle-dir DATA_DIR]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_CSV = DATA_DIR / "mitbih.csv"
PROCESSED_PT = DATA_DIR / "processed_mlii.pt"


def resample_360_to_187(x: np.ndarray) -> np.ndarray:
    """Resample (N, 360) to (N, 187) via linear interpolation."""
    n, orig_len = x.shape
    new_len = 187
    indices = np.linspace(0, orig_len - 1, new_len)
    result = np.zeros((n, new_len), dtype=np.float32)
    for i in range(n):
        result[i] = np.interp(indices, np.arange(orig_len), x[i])
    return result


def convert_from_pt(pt_path: Path, output_path: Path) -> None:
    """Convert processed_mlii.pt (360 samples) to 187-sample CSV."""
    print(f"Loading {pt_path}...")
    data = torch.load(pt_path, map_location="cpu", weights_only=True)
    X = data["X"].numpy()  # (N, 360, 1)
    y = data["y"].numpy()  # (N,)

    # Squeeze channel and resample
    X = X.squeeze(-1)  # (N, 360)
    X = resample_360_to_187(X)  # (N, 187)

    cols = {i: X[:, i] for i in range(187)}
    cols[187] = y
    df = pd.DataFrame(cols)
    df.to_csv(output_path, index=False)
    print(f"Saved {output_path} with {len(df)} samples.")


def convert_from_kaggle(train_csv: Path, test_csv: Path, output_path: Path) -> None:
    """Merge Kaggle mitbih_train.csv and mitbih_test.csv into one CSV."""
    print(f"Loading {train_csv} and {test_csv}...")
    df_train = pd.read_csv(train_csv, header=None)
    df_test = pd.read_csv(test_csv, header=None)
    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    # Columns 0..186 = signal, 187 = class (keep as 187)
    df.to_csv(output_path, index=False)
    print(f"Saved {output_path} with {len(df)} samples.")
    print(f"Class distribution:\n{df[187].value_counts().sort_index()}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare MIT-BIH CSV (187 samples/beat)")
    parser.add_argument(
        "--from-pt",
        action="store_true",
        help=f"Convert from {PROCESSED_PT} (360 samples)",
    )
    parser.add_argument(
        "--from-kaggle-dir",
        type=Path,
        default=None,
        help="Directory containing mitbih_train.csv and mitbih_test.csv",
    )
    parser.add_argument("-o", "--output", type=Path, default=OUTPUT_CSV)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.from_pt:
        if not PROCESSED_PT.exists():
            print(f"Error: {PROCESSED_PT} not found. Run src/download_data.py and src/preprocess.py first.")
            return 1
        convert_from_pt(PROCESSED_PT, args.output)
        return 0

    if args.from_kaggle_dir is not None:
        d = Path(args.from_kaggle_dir)
        train_path = d / "mitbih_train.csv"
        test_path = d / "mitbih_test.csv"
        if not train_path.exists() or not test_path.exists():
            print(f"Error: Need {train_path} and {test_path}")
            return 1
        convert_from_kaggle(train_path, test_path, args.output)
        return 0

    # Default: try from-pt if processed exists
    if PROCESSED_PT.exists():
        print("Found processed_mlii.pt. Converting to 187-sample CSV...")
        convert_from_pt(PROCESSED_PT, args.output)
        return 0

    print(
        "No data source specified. Use:\n"
        "  --from-pt        Convert from data/processed_mlii.pt (run preprocess first)\n"
        "  --from-kaggle-dir DIR  Merge mitbih_train.csv and mitbih_test.csv from Kaggle"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
