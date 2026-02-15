# Technical Report: Robust ECG Beat Classification on MIT-BIH

## Summary

This project implements a reproducible pipeline for beat-level ECG classification on the MIT-BIH Arrhythmia Database using an LSTM as the primary model and a 1D-CNN baseline. It evaluates robustness under additive Gaussian noise, missing segments, and temporal jitter, and reports learning curves under small-data regimes (100%, 50%, 25%, 10% of training data). Results are reported with mean ± standard deviation over 5 random seeds and 95% confidence intervals.

## Setup

- **Dataset**: MIT-BIH Arrhythmia Database (48 records, 360 Hz), AAMI classes N, S, V, F, Q.
- **Preprocessing**: Z-score normalization per record; R-peak-centered 1 s windows (180+180 samples); MLII lead.
- **Splits**: Intra-patient (stratified beat-level) and inter-patient (by record ID; test patients unseen).

## Models

- **LSTM**: 2-layer, hidden size 128, dropout 0.3, FC to 5 classes.
- **1D-CNN**: 3 conv blocks (BatchNorm + ReLU), global average pooling, FC classifier.

## Training

- Optimizer: Adam, lr 1e-3, batch size 128, up to 100 epochs with early stopping on validation Macro-F1 (patience 15).
- Loss: CrossEntropy (optional class-weighted CE or Focal Loss via config).

## Metrics

- Primary: Macro-F1. Also: Balanced Accuracy, per-class recall, optional AUROC.
- All experiments run with 5 seeds; reported as mean ± std and 95% CI.

## Robustness

- **Gaussian noise**: SNR 20, 10, 5 dB.
- **Missing segments**: 5%, 10%, 20% of signal masked.
- **Temporal jitter**: ±10, ±20, ±30 ms circular shift.

## Small-Data

- Training with 100%, 50%, 25%, 10% of training data; Macro-F1 vs data fraction plotted as learning curve.

## Reproducibility

- Fixed seeds in PyTorch, NumPy, Python.
- Split definitions and configs in `splits/` and `configs/`.
- Run order: download → preprocess → train (per seed) → eval → robustness → run_small_data → stats.

## Results

See `results/` for CSVs and `results/figures/` for confusion matrix, robustness plot, and learning curve. Summary tables are produced by `python src/stats.py`.
