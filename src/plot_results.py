"""Generate figures from results CSVs (robustness plot, learning curve, confusion matrix placeholder)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from utils import RESULTS_DIR, RESULTS_FIGURES

def main():
    RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
    # Robustness plot from robustness_curves.csv
    rob = RESULTS_DIR / "robustness_curves.csv"
    if rob.exists():
        df = pd.read_csv(rob)
        if "macro_f1" in df.columns:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            for corruption in df["corruption"].unique():
                sub = df[df["corruption"] == corruption]
                g = sub.groupby("level")["macro_f1"]
                levels = sorted(sub["level"].unique())
                means = [g.get_group(l).mean() for l in levels]
                stds = [g.get_group(l).std() for l in levels]
                plt.figure(figsize=(5, 3))
                plt.errorbar(range(len(levels)), means, yerr=stds, capsize=4, marker="o")
                plt.xticks(range(len(levels)), [str(x) for x in levels])
                plt.xlabel(f"{corruption} level")
                plt.ylabel("Macro-F1")
                plt.title(f"Robustness: {corruption}")
                plt.tight_layout()
                plt.savefig(RESULTS_FIGURES / f"robustness_{corruption}.png")
                plt.close()
            # Combined robustness (first corruption)
            g = df.groupby(["corruption", "level"])["macro_f1"].agg(["mean", "std"]).reset_index()
            plt.figure(figsize=(6, 4))
            for corr in g["corruption"].unique():
                sub = g[g["corruption"] == corr]
                plt.errorbar(sub["level"], sub["mean"], yerr=sub["std"], label=corr, capsize=4, marker="o")
            plt.xlabel("Level")
            plt.ylabel("Macro-F1")
            plt.legend()
            plt.title("Robustness curves")
            plt.tight_layout()
            plt.savefig(RESULTS_FIGURES / "robustness_plot.png")
            plt.close()
            print("Saved robustness_plot.png")
    # Learning curve from small_data_curve.csv
    small = RESULTS_DIR / "small_data_curve.csv"
    if small.exists():
        df = pd.read_csv(small)
        if "macro_f1" in df.columns and "data_fraction" in df.columns:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            g = df.groupby("data_fraction")["macro_f1"]
            x = sorted(df["data_fraction"].unique())
            means = [g.get_group(f).mean() for f in x]
            stds = [g.get_group(f).std() for f in x] if len(x) > 1 else [0.0] * len(x)
            plt.figure(figsize=(6, 4))
            plt.errorbar(x, means, yerr=stds, capsize=4, marker="o")
            plt.xlabel("Training data fraction")
            plt.ylabel("Macro-F1")
            plt.title("Learning curve (small-data)")
            plt.tight_layout()
            plt.savefig(RESULTS_FIGURES / "learning_curve.png")
            plt.close()
            print("Saved learning_curve.png")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
