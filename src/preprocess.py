"""
Beat extraction, normalization, AAMI mapping, and train/test split generation.
Loads MIT-BIH from data dir, produces serialized tensors and split JSONs.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import wfdb
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

from utils import (
    DATA_DIR,
    SPLITS_DIR,
    AAMI_CLASSES,
    SAMPLE_RATE,
    HALF_WINDOW,
    BEAT_WINDOW_SAMPLES,
    set_seed,
)

# AAMI mapping: MIT-BIH annotation symbol -> AAMI class index (N=0, S=1, V=2, F=3, Q=4)
# Based on AAMI EC57 / common literature
SYMBOL_TO_AAMI = {
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0, "B": 0, "n": 0, "r": 0,  # Normal
    "A": 1, "a": 1, "J": 1, "S": 1, "T": 1,  # Supraventricular
    "V": 2, "E": 2,  # Ventricular
    "F": 3,  # Fusion
    "/": 4, "f": 4, "Q": 4, "?": 4, "|": 4, "x": 4,  # Unknown
}
# Only use beat types we map; skip non-beat (e.g. [, ], etc.)
VALID_SYMBOLS = set(SYMBOL_TO_AAMI.keys())

# MIT-BIH record list (same as download_data)
MITBIH_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]


def normalize_zscore(signal: np.ndarray) -> np.ndarray:
    """Z-score normalization per channel (last axis)."""
    mean = np.mean(signal, axis=0, keepdims=True)
    std = np.std(signal, axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (signal - mean) / std


def extract_beats(
    data_dir: Path,
    channels: str = "mlii",
    test_fraction: float = 0.2,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, int]]]:
    """
    Load records from data_dir, extract R-peak-centered windows, normalize, map to AAMI.
    Returns:
        X: (N, seq_len, C) float32
        y: (N,) int64 class indices
        record_ids: (N,) record name per sample (for inter-patient split)
        index_to_record_beat: list of (record_name, beat_index_in_record) for each global index
    """
    set_seed(seed)
    data_dir = Path(data_dir)
    use_mlii_only = channels.strip().lower() == "mlii"
    if use_mlii_only:
        n_channels = 1
    else:
        n_channels = 2  # MLII + V1

    all_beats = []
    all_labels = []
    all_record_ids = []
    all_record_beat_idx = []

    for rec_name in MITBIH_RECORDS:
        dat_path = data_dir / f"{rec_name}.dat"
        atr_path = data_dir / f"{rec_name}.atr"
        if not dat_path.exists() or not atr_path.exists():
            continue
        rec_path = str(data_dir / rec_name)
        record = wfdb.rdrecord(rec_path)
        ann = wfdb.rdann(rec_path, "atr")
        sig = record.p_signal.astype(np.float64)
        fs = record.fs
        if fs != SAMPLE_RATE:
            # resample not required for MIT-BIH (360 Hz)
            pass
        # Select leads: MLII is often first; V1 if present
        sig_names = [s.upper() for s in (record.sig_name or [])]
        if use_mlii_only:
            if "MLII" in sig_names:
                idx = sig_names.index("MLII")
            else:
                idx = 0
            sig = sig[:, idx : idx + 1]
        else:
            # MLII + V1
            idx_mlii = sig_names.index("MLII") if "MLII" in sig_names else 0
            idx_v1 = sig_names.index("V1") if "V1" in sig_names else min(1, sig.shape[1] - 1)
            sig = sig[:, [idx_mlii, idx_v1]]
        sig = normalize_zscore(sig)
        samples = ann.sample
        symbols = ann.symbol
        for i, (samp, sym) in enumerate(zip(samples, symbols)):
            if sym not in VALID_SYMBOLS:
                continue
            left = samp - HALF_WINDOW
            right = samp + HALF_WINDOW
            if left < 0 or right > sig.shape[0]:
                continue
            beat = sig[left:right, :]
            all_beats.append(beat)
            all_labels.append(SYMBOL_TO_AAMI[sym])
            all_record_ids.append(rec_name)
            all_record_beat_idx.append((rec_name, i))

    if not all_beats:
        raise FileNotFoundError(f"No beats extracted. Ensure data is in {data_dir} (run download_data.py).")
    X = np.stack(all_beats, axis=0).astype(np.float32)
    y = np.array(all_labels, dtype=np.int64)
    record_ids = np.array(all_record_ids)
    return X, y, record_ids, all_record_beat_idx


def build_splits(
    X: np.ndarray,
    y: np.ndarray,
    record_ids: np.ndarray,
    splits_dir: Path,
    test_fraction: float = 0.2,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> None:
    """Build intra-patient (stratified) and inter-patient (by record) splits; save JSONs."""
    set_seed(seed)
    n = len(y)
    indices = np.arange(n)

    # Intra-patient: stratified split at beat level
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train_idx, test_idx = next(sss.split(indices, y))
    # validation from train
    train_y = y[train_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction / (1 - test_fraction), random_state=seed)
    sub_train, val_idx = next(sss2.split(train_idx, train_y))
    intra = {
        "train": train_idx[sub_train].tolist(),
        "val": train_idx[val_idx].tolist(),
        "test": test_idx.tolist(),
    }
    splits_dir.mkdir(parents=True, exist_ok=True)
    with open(splits_dir / "intra_patient.json", "w") as f:
        json.dump(intra, f, indent=2)
    for name, idx_list in [("train", intra["train"]), ("val", intra["val"]), ("test", intra["test"])]:
        subset_y = y[np.array(idx_list)]
        n_per_class = [np.sum(subset_y == c) for c in range(len(AAMI_CLASSES))]
        print(f"  Intra {name}: n={len(idx_list)} classes={dict(zip(AAMI_CLASSES, n_per_class))}")

    # Inter-patient: split by record ID
    unique_records = np.unique(record_ids)
    rec_train, rec_test = train_test_split(unique_records.tolist(), test_size=test_fraction, random_state=seed)
    rec_train, rec_val = train_test_split(rec_train, test_size=val_fraction / (1 - test_fraction), random_state=seed)
    train_mask = np.isin(record_ids, rec_train)
    val_mask = np.isin(record_ids, rec_val)
    test_mask = np.isin(record_ids, rec_test)
    inter = {
        "train": indices[train_mask].tolist(),
        "val": indices[val_mask].tolist(),
        "test": indices[test_mask].tolist(),
        "train_records": rec_train,
        "val_records": rec_val,
        "test_records": rec_test,
    }
    with open(splits_dir / "inter_patient.json", "w") as f:
        json.dump(inter, f, indent=2)

    # Verify splits: no index overlap, class distribution per split
    train_idx_set = set(inter["train"])
    val_idx_set = set(inter["val"])
    test_idx_set = set(inter["test"])
    assert len(train_idx_set & val_idx_set) == 0, "Train/val overlap"
    assert len(train_idx_set & test_idx_set) == 0, "Train/test overlap"
    assert len(val_idx_set & test_idx_set) == 0, "Val/test overlap"
    for name, idx_list in [("train", inter["train"]), ("val", inter["val"]), ("test", inter["test"])]:
        subset_y = y[np.array(idx_list)]
        n_per_class = [np.sum(subset_y == c) for c in range(len(AAMI_CLASSES))]
        print(f"  Inter {name}: n={len(idx_list)} classes={dict(zip(AAMI_CLASSES, n_per_class))}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess MIT-BIH and generate splits")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Directory with WFDB records")
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR, help="Where to save split JSONs")
    parser.add_argument("--channels", type=str, default="mlii", choices=["mlii", "mlii_v1"], help="MLII only or MLII+V1")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None, help="Output tensor path (default: data/processed_<channels>.pt)")
    args = parser.parse_args()
    if args.out is None:
        args.out = args.data_dir / f"processed_{args.channels}.pt"

    X, y, record_ids, _ = extract_beats(
        args.data_dir,
        channels=args.channels,
        test_fraction=args.test_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print(f"Extracted {X.shape[0]} beats, shape {X.shape}, classes {AAMI_CLASSES}")
    for c in range(len(AAMI_CLASSES)):
        print(f"  {AAMI_CLASSES[c]}: {(y == c).sum()}")

    build_splits(X, y, record_ids, args.splits_dir, args.test_fraction, args.val_fraction, args.seed)
    print(f"Splits saved to {args.splits_dir}")

    data = {"X": torch.from_numpy(X), "y": torch.from_numpy(y), "record_ids": record_ids.tolist()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.out)
    print(f"Processed tensors saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
