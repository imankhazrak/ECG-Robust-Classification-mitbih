"""
Evaluation metrics for ECG classification.
"""
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute accuracy, macro/weighted F1, precision, recall, per-class metrics, confusion matrix.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(macro_precision),
        "weighted_precision": float(weighted_precision),
        "macro_recall": float(macro_recall),
        "weighted_recall": float(weighted_recall),
        "per_class_precision": per_class_precision.tolist(),
        "per_class_recall": per_class_recall.tolist(),
        "confusion_matrix": cm.tolist(),
    }
    return metrics


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> str:
    """Print sklearn classification report."""
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    print(report)
    return report


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: Path,
) -> None:
    """Save confusion matrix as figure."""
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
