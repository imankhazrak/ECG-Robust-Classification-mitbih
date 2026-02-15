"""
Run small-data experiments: train with 100%, 50%, 25%, 10% of training data over 5 seeds;
record Macro-F1 to results/small_data_curve.csv and optionally plot learning curve.
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import yaml

from utils import PROJECT_ROOT, RESULTS_DIR, RESULTS_FIGURES, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "small_data.yaml")
    parser.add_argument("--split", type=str, default="inter_patient")
    parser.add_argument("--plot", action="store_true", help="Plot learning curve to results/figures/learning_curve.png")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    seeds = config.get("seeds", [42, 1, 2, 3, 4])
    fractions = config.get("data_fractions", [1.0, 0.5, 0.25, 0.1])
    model_name = config.get("model", "lstm")
    ckpt_dir = RESULTS_DIR / "checkpoints"
    out_csv = RESULTS_DIR / "small_data_curve.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = ["model", "split", "data_fraction", "seed", "macro_f1"]
    write_header = not out_csv.exists() or out_csv.stat().st_size == 0

    for frac in fractions:
        for seed in seeds:
            # Train
            cmd_train = [
                sys.executable,
                str(PROJECT_ROOT / "src" / "train.py"),
                "--config", str(args.config),
                "--split", args.split,
                "--seed", str(seed),
                "--data-fraction", str(frac),
            ]
            subprocess.run(cmd_train, cwd=PROJECT_ROOT, check=True)
            ckpt = ckpt_dir / f"{model_name}_{args.split}_seed{seed}_frac{frac}.pt"
            if not ckpt.exists():
                ckpt = ckpt_dir / f"{model_name}_{args.split}_seed{seed}.pt" if frac >= 1.0 else None
            if ckpt is None or not ckpt.exists():
                continue
            # Eval and get macro_f1
            cmd_eval = [
                sys.executable,
                str(PROJECT_ROOT / "src" / "eval.py"),
                "--checkpoint", str(ckpt),
                "--config", str(args.config),
                "--split", args.split,
                "--seed", str(seed),
            ]
            result = subprocess.run(cmd_eval, cwd=PROJECT_ROOT, capture_output=True, text=True)
            macro_f1 = None
            for line in result.stdout.splitlines():
                if line.startswith("Macro-F1:"):
                    macro_f1 = float(line.split(":", 1)[1].strip())
                    break
            if macro_f1 is not None:
                with open(out_csv, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=header)
                    if write_header:
                        w.writeheader()
                        write_header = False
                    w.writerow({"model": model_name, "split": args.split, "data_fraction": frac, "seed": seed, "macro_f1": macro_f1})
                print(f"data_fraction={frac} seed={seed} macro_f1={macro_f1:.4f}")

    if args.plot and out_csv.exists():
        import pandas as pd
        import numpy as np
        df = pd.read_csv(out_csv)
        if "macro_f1" in df.columns and "data_fraction" in df.columns:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            g = df.groupby("data_fraction")["macro_f1"]
            x = sorted(df["data_fraction"].unique())
            means = [g.get_group(f).mean() for f in x]
            stds = [g.get_group(f).std() for f in x]
            plt.figure(figsize=(6, 4))
            plt.errorbar(x, means, yerr=stds, capsize=4, marker="o")
            plt.xlabel("Training data fraction")
            plt.ylabel("Macro-F1")
            plt.title("Learning curve (small-data)")
            plt.tight_layout()
            RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
            plt.savefig(RESULTS_FIGURES / "learning_curve.png")
            plt.close()
            print(f"Saved {RESULTS_FIGURES / 'learning_curve.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
