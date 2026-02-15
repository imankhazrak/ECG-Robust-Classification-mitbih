"""
Trainer for CNN + Transformer ECG classification.
"""
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import CLASS_NAMES
from .metrics import compute_metrics


logger = logging.getLogger("ecg_cnn_transformer")


class Trainer:
    """Training loop with early stopping and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        device: torch.device,
        config: dict,
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config

        self.best_macro_f1 = -1.0
        self.patience = config.get("patience", 15)
        self.wait = 0
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _is_mixup_batch(self, target: Any) -> bool:
        return isinstance(target, (tuple, list)) and len(target) == 3

    def train_epoch(self, train_loader: DataLoader) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        all_preds: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        for x, y in train_loader:
            x = x.to(self.device)
            if self._is_mixup_batch(y):
                y_a, y_b, lam = y
                y_a = y_a.to(self.device)
                y_b = y_b.to(self.device)
                logits = self.model(x)
                loss = lam * self.criterion(logits, y_a) + (1 - lam) * self.criterion(
                    logits, y_b
                )
                # For metrics, use y_a as primary
                y_np = y_a.cpu().numpy()
            else:
                y = y.to(self.device)
                logits = self.model(x)
                loss = self.criterion(logits, y)
                y_np = y.cpu().numpy()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y_np)

        if self.scheduler is not None:
            self.scheduler.step()

        avg_loss = total_loss / len(train_loader)
        preds_cat = np.concatenate(all_preds)
        labels_cat = np.concatenate(all_labels)
        metrics = compute_metrics(labels_cat, preds_cat)

        return {
            "loss": avg_loss,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
        }

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        for x, y in val_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            if self._is_mixup_batch(y):
                y = y[0]
            logits = self.model(x)
            loss = self.criterion(logits, y)

            total_loss += loss.item()
            preds = logits.argmax(1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        preds_cat = np.concatenate(all_preds)
        labels_cat = np.concatenate(all_labels)
        metrics = compute_metrics(labels_cat, preds_cat)

        return {
            "loss": avg_loss,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
    ) -> dict[str, float]:
        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            logger.info(
                f"Epoch {epoch} | "
                f"train loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
                f"train_f1={train_metrics['macro_f1']:.4f} train_prec={train_metrics['macro_precision']:.4f} "
                f"train_rec={train_metrics['macro_recall']:.4f} | "
                f"val loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
                f"val_f1={val_metrics['macro_f1']:.4f} val_prec={val_metrics['macro_precision']:.4f} "
                f"val_rec={val_metrics['macro_recall']:.4f}"
            )

            if val_metrics["macro_f1"] > self.best_macro_f1:
                self.best_macro_f1 = val_metrics["macro_f1"]
                self.wait = 0
                ckpt_path = self.checkpoint_dir / "best_model.pt"
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "epoch": epoch,
                        "val_macro_f1": self.best_macro_f1,
                        "config": self.config,
                    },
                    ckpt_path,
                )
            else:
                self.wait += 1

            # if self.wait >= self.patience:
            #     logger.info(f"Early stopping at epoch {epoch}")
            #     break

        return {
            "best_val_macro_f1": self.best_macro_f1,
            "last_train_accuracy": train_metrics["accuracy"],
            "last_val_accuracy": val_metrics["accuracy"],
        }
