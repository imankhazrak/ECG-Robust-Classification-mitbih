"""
Statistical summaries over 5 seeds: mean ± std, 95% CI.
Read result CSVs and write structured summary CSVs.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from utils import RESULTS_DIR


def ci_95(values: np.ndarray) -> tuple[float, float]:
    """Approximate 95% CI using 1.96 * std."""
    m, s = values.mean(), values.std()
    n = len(values)
    if n <= 1:
        return m, m
    se = s / (n ** 0.5)
    return m - 1.96 * se, m + 1.96 * se


def main():
    parser = argparse.ArgumentParser(description="Compute mean ± std and 95% CI over seeds")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--raw-results", type=Path, default=None, help="CSV with columns seed, macro_f1, ...")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate training/eval results: expect files like inter_patient_results.csv with seed, model, split, macro_f1, balanced_accuracy
    for name, out_name in [
        ("inter_patient_results.csv", "inter_patient_results_summary.csv"),
        ("intra_patient_results.csv", "intra_patient_results_summary.csv"),
    ]:
        path = results_dir / name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "macro_f1" not in df.columns:
            continue
        grouped = df.groupby(["model", "split"], dropna=False) if "split" in df.columns else df.groupby(["model"], dropna=False)
        rows = []
        for key, g in grouped:
            vals = g["macro_f1"].values
            low, high = ci_95(vals)
            row = {"model": key[0] if isinstance(key, tuple) else key, "macro_f1_mean": vals.mean(), "macro_f1_std": vals.std(), "macro_f1_ci95_low": low, "macro_f1_ci95_high": high, "n_seeds": len(vals)}
            if isinstance(key, tuple) and len(key) > 1:
                row["split"] = key[1]
            rows.append(row)
        out = pd.DataFrame(rows)
        summary_path = results_dir / name.replace(".csv", "_summary.csv")
        out.to_csv(summary_path, index=False)
        print(f"Wrote {out_name}")

    # Small-data curve: expect small_data_curve.csv with data_fraction, seed, macro_f1
    small_path = results_dir / "small_data_curve.csv"
    if small_path.exists():
        df = pd.read_csv(small_path)
        if "macro_f1" in df.columns and "data_fraction" in df.columns:
            grouped = df.groupby("data_fraction")["macro_f1"]
            summary = grouped.agg(["mean", "std", "count"])
            summary["ci95_low"] = grouped.apply(lambda x: ci_95(x.values)[0])
            summary["ci95_high"] = grouped.apply(lambda x: ci_95(x.values)[1])
            out_path = results_dir / "small_data_curve_summary.csv"
            summary.to_csv(out_path)
            print(f"Wrote {out_path}")

    # Robustness: robustness_curves.csv with model, corruption, level, seed, macro_f1
    rob_path = results_dir / "robustness_curves.csv"
    if rob_path.exists():
        df = pd.read_csv(rob_path)
        if "macro_f1" in df.columns:
            grouped = df.groupby(["model", "corruption", "level"])["macro_f1"]
            summary = grouped.agg(["mean", "std", "count"]).reset_index()
            summary["ci95_low"] = grouped.apply(lambda x: ci_95(x.values)[0]).values
            summary["ci95_high"] = grouped.apply(lambda x: ci_95(x.values)[1]).values
            summary.to_csv(results_dir / "robustness_curves_summary.csv", index=False)
            print("Wrote robustness_curves_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
