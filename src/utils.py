"""
Seed setting, logging helpers, and path constants for reproducible experiments.
"""
import os
import random
import logging
from pathlib import Path

import numpy as np
import torch

# Default paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = PROJECT_ROOT / "splits"
CONFIGS_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_FIGURES = RESULTS_DIR / "figures"
RESULTS_LOGS = RESULTS_DIR / "logs"

# AAMI class names (5 classes)
AAMI_CLASSES = ["N", "S", "V", "F", "Q"]
NUM_CLASSES = 5

# Sampling and window
SAMPLE_RATE = 360
BEAT_WINDOW_SAMPLES = 360  # 1.0 s: 180 before + 180 after R-peak
HALF_WINDOW = BEAT_WINDOW_SAMPLES // 2


def set_seed(seed: int) -> None:
    """Fix random seeds for PyTorch, NumPy, and Python for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = True


def setup_logging(log_path: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure logging to file and console."""
    logger = logging.getLogger("ecg")
    logger.setLevel(level)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def ensure_dirs() -> None:
    """Create results/splits/configs dirs if they do not exist."""
    for d in (DATA_DIR, SPLITS_DIR, RESULTS_DIR, RESULTS_FIGURES, RESULTS_LOGS):
        d.mkdir(parents=True, exist_ok=True)
