"""
Evaluation metrics: Macro-F1, balanced accuracy, per-class recall, optional AUROC.
Confusion matrix; load checkpoint and run on test set.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from utils import (
    set_seed,
    PROJECT_ROOT,
    DATA_DIR,
    SPLITS_DIR,
    RESULTS_DIR,
    RESULTS_FIGURES,
    AAMI_CLASSES,
    NUM_CLASSES,
)
from datasets import ECGBeatDataset
from models import ECG_LSTM, ECG_CNN1D


def get_model(model_name: str, seq_len: int, in_channels: int, device: torch.device):
    if model_name.lower() == "lstm":
        model = ECG_LSTM(input_size=in_channels, hidden_size=128, num_layers=2, dropout=0.3, num_classes=NUM_CLASSES)
    elif model_name.lower() == "cnn":
        model = ECG_CNN1D(in_channels=in_channels, num_classes=NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = NUM_CLASSES,
) -> dict:
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    probs = np.concatenate(all_probs)

    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    balanced_acc = balanced_accuracy_score(labels, preds)
    per_class_recall = recall_score(labels, preds, average=None, zero_division=0)
    cm = confusion_matrix(labels, preds)
    results = {
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(balanced_acc),
        "per_class_recall": per_class_recall.tolist(),
        "confusion_matrix": cm.tolist(),
    }
    try:
        if num_classes == 5 and np.unique(labels).size >= 2:
            auroc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
            results["auroc"] = float(auroc)
    except Exception:
        pass
    return results, preds, labels, probs, cm


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on test set")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint .pt")
    parser.add_argument("--config", type=Path, required=True, help="Config YAML (for data path, model type)")
    parser.add_argument("--split", type=str, default="inter_patient", choices=["inter_patient", "intra_patient"])
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-csv", type=Path, default=None, help="Append row to results CSV")
    parser.add_argument("--save-confusion", type=Path, default=None, help="Save confusion matrix figure path")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.data_path is not None:
        data_path = args.data_path
    else:
        dp = config.get("data_path", "data/processed_mlii.pt")
        data_path = PROJECT_ROOT / dp if not str(dp).startswith("/") else Path(dp)
    dataset = ECGBeatDataset(data_path, "test", split_name=args.split, splits_dir=SPLITS_DIR)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    # Infer shape from dataset
    x0, _ = dataset[0]
    seq_len, in_channels = x0.shape[0], x0.shape[1]
    model = get_model(config["model"], seq_len, in_channels, device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    results, preds, labels, probs, cm = evaluate(model, loader, device)
    print("Macro-F1:", results["macro_f1"])
    print("Balanced Accuracy:", results["balanced_accuracy"])
    print("Per-class recall:", dict(zip(AAMI_CLASSES, results["per_class_recall"])))
    if args.save_csv:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "model": config.get("model", "lstm"),
            "split": args.split,
            "seed": getattr(args, "seed", None),
            "checkpoint": str(args.checkpoint),
            "macro_f1": results["macro_f1"],
            "balanced_accuracy": results["balanced_accuracy"],
        }
        import csv
        exists = args.save_csv.exists()
        with open(args.save_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                w.writeheader()
            w.writerow(row)
    if args.save_confusion:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xticks(range(NUM_CLASSES), AAMI_CLASSES)
        plt.yticks(range(NUM_CLASSES), AAMI_CLASSES)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")
        args.save_confusion.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.save_confusion, bbox_inches="tight")
        plt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
