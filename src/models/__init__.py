"""Model definitions for ECG beat classification."""
from .lstm import ECG_LSTM
from .cnn1d import ECG_CNN1D

__all__ = ["ECG_LSTM", "ECG_CNN1D"]
