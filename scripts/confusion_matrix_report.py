#!/usr/bin/env python3
"""
Generate confusion matrix report (CSV + Markdown) from CNN+Transformer metrics.
No matplotlib required. For plots, run: python scripts/plot_confusion_matrices.py (on cluster)
"""
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from utils.dataset import CLASS_NAMES


def main() -> int:
    results_dir = PROJECT_ROOT / "results"
    seeds = [42, 1, 2, 3, 4]
    all_cms = []

    for seed in seeds:
        p = results_dir / f"metrics_seed{seed}.json"
        if not p.exists():
            continue
        with open(p) as f:
            data = json.load(f)
        all_cms.append(np.array(data["confusion_matrix"]))

    if not all_cms:
        print("No metrics files found.")
        return 1

    agg_cm = np.sum(all_cms, axis=0).astype(int)

    # Save CSV (rows=true, cols=predicted)
    csv_path = results_dir / "confusion_matrix_cnn_transformer_aggregate.csv"
    with open(csv_path, "w") as f:
        f.write("," + ",".join(CLASS_NAMES) + "\n")
        for i, name in enumerate(CLASS_NAMES):
            f.write(name + "," + ",".join(str(int(agg_cm[i, j])) for j in range(5)) + "\n")
    print(f"Saved {csv_path}")

    # Save Markdown report
    md_path = results_dir / "confusion_matrix_cnn_transformer_report.md"
    lines = [
        "# CNN+Transformer Confusion Matrix (Aggregate over 5 seeds)",
        "",
        "Rows = True label, Cols = Predicted",
        "",
        "|  | " + " | ".join(CLASS_NAMES) + " |",
        "|---|" + "|".join(["---"] * 5) + "|",
    ]
    for i, name in enumerate(CLASS_NAMES):
        row = " | ".join(str(agg_cm[i, j]) for j in range(5))
        lines.append(f"| **{name}** | {row} |")
    lines.append("")
    lines.append("## Per-seed summary")
    for seed, cm in zip(seeds, all_cms):
        with open(results_dir / f"metrics_seed{seed}.json") as f:
            m = json.load(f)
        lines.append(f"- **Seed {seed}**: Acc={m['accuracy']:.4f}, Macro F1={m['macro_f1']:.4f}")
    lines.append("")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {md_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
