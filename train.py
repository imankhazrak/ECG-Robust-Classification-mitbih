#!/usr/bin/env python3
"""
Train CNN + Transformer hybrid for MIT-BIH ECG classification.
Usage: python train.py [--config config]
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config, get_config
from models import CNNTransformer
from utils import (
    CLASS_NAMES,
    get_criterion,
    get_dataloaders,
    get_class_weights,
    print_classification_report,
    save_confusion_matrix,
    compute_metrics,
)
from utils.dataset import load_csv
from utils.trainer import Trainer


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main() -> int:
    parser = argparse.ArgumentParser(description="Train CNN+Transformer ECG classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="config",
        help="Config module name (default: config)",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    config = get_config()

    if args.seed is not None:
        config.seed = args.seed

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.log_dir.mkdir(parents=True, exist_ok=True)
    config.results_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(config.log_dir / "train.log"),
        ],
    )
    log = logging.getLogger("ecg_cnn_transformer")
    log.info(f"Device: {device}")

    if not config.data_path.exists():
        log.error(
            f"Data not found: {config.data_path}. "
            "Run: python scripts/prepare_csv.py --from-pt"
        )
        return 1

    train_loader, val_loader, test_loader = get_dataloaders(
        config.data_path,
        batch_size=config.batch_size,
        train_frac=config.train_frac,
        val_frac=config.val_frac,
        seed=config.seed,
        mixup_alpha=config.mixup_alpha,
    )

    data, labels = load_csv(config.data_path)
    class_weights = get_class_weights(labels)
    criterion = get_criterion(
        config.loss_type,
        class_weights,
        config.focal_gamma,
        device,
    )

    model = CNNTransformer(
        seq_len=config.seq_len,
        in_channels=1,
        num_classes=config.num_classes,
        base_filters=config.base_filters,
        kernel_size=config.kernel_size,
        num_conv_blocks=config.num_conv_blocks,
        d_model=config.d_model or None,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        dim_feedforward=config.dim_feedforward or None,
        dropout=config.dropout,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    if config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=5e-6
        )
    elif config.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )
    else:
        scheduler = None

    config_dict = {
        "checkpoint_dir": str(config.checkpoint_dir),
        "patience": config.patience,
    }

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config_dict,
    )

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        log.info(f"Resumed from {args.resume}")

    train_result = trainer.train(train_loader, val_loader, config.epochs)

    # Evaluate on test set
    ckpt_path = config.checkpoint_dir / "best_model.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        log.info(f"Loaded best checkpoint (val_macro_f1={ckpt['val_macro_f1']:.4f})")

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y.numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    metrics = compute_metrics(y_true, y_pred)
    metrics["last_train_accuracy"] = train_result["last_train_accuracy"]
    metrics["last_val_accuracy"] = train_result["last_val_accuracy"]
    log.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
    log.info(f"Last Train Accuracy: {metrics['last_train_accuracy']:.4f}")
    log.info(f"Last Val Accuracy: {metrics['last_val_accuracy']:.4f}")
    log.info(f"Test Macro F1: {metrics['macro_f1']:.4f}")
    log.info(f"Test Weighted F1: {metrics['weighted_f1']:.4f}")
    log.info(f"Test Macro Precision: {metrics['macro_precision']:.4f}")
    log.info(f"Test Macro Recall: {metrics['macro_recall']:.4f}")
    log.info(f"Test Weighted Precision: {metrics['weighted_precision']:.4f}")
    log.info(f"Test Weighted Recall: {metrics['weighted_recall']:.4f}")

    print_classification_report(y_true, y_pred, CLASS_NAMES)
    save_confusion_matrix(
        y_true, y_pred, CLASS_NAMES, config.results_dir / "confusion_matrix.png"
    )

    # Save metrics
    import json
    with open(config.results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save per-class precision and recall
    import csv
    pr_path = config.results_dir / "precision_recall_per_class.csv"
    with open(pr_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall"])
        for i, name in enumerate(CLASS_NAMES):
            w.writerow([
                name,
                f"{metrics['per_class_precision'][i]:.4f}",
                f"{metrics['per_class_recall'][i]:.4f}",
            ])
        w.writerow(["macro_avg", f"{metrics['macro_precision']:.4f}", f"{metrics['macro_recall']:.4f}"])
        w.writerow(["weighted_avg", f"{metrics['weighted_precision']:.4f}", f"{metrics['weighted_recall']:.4f}"])
        w.writerow([])
        w.writerow(["last_train_accuracy", f"{metrics.get('last_train_accuracy', 0):.4f}", ""])
        w.writerow(["last_val_accuracy", f"{metrics.get('last_val_accuracy', 0):.4f}", ""])
    log.info(f"Saved {pr_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
