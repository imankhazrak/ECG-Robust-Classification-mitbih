#!/bin/bash
# Full pipeline: download -> preprocess -> train (5 seeds) -> eval -> robustness -> small-data -> stats -> figures
# Run from project root. Training may take a long time (CPU).
set -e
cd "$(dirname "$0")"
echo "1. Installing dependencies..."
pip install -r requirements.txt -q
echo "2. Downloading MIT-BIH..."
python3 src/download_data.py
echo "3. Preprocessing and generating splits..."
python3 src/preprocess.py
echo "4. Training LSTM (5 seeds)..."
for seed in 42 1 2 3 4; do
  python3 src/train.py --config configs/lstm_baseline.yaml --split inter_patient --seed $seed
done
echo "5. Training CNN (5 seeds)..."
for seed in 42 1 2 3 4; do
  python3 src/train.py --config configs/cnn_baseline.yaml --split inter_patient --seed $seed
done
echo "6. Evaluating (append to results CSVs)..."
for seed in 42 1 2 3 4; do
  python3 src/eval.py --checkpoint results/checkpoints/lstm_inter_patient_seed${seed}.pt --config configs/lstm_baseline.yaml --split inter_patient --save-csv results/inter_patient_results.csv --save-confusion results/figures/confusion_matrix.png
done
echo "7. Robustness..."
python3 src/robustness.py --model lstm --corruption gaussian
python3 src/robustness.py --model lstm --corruption missing
python3 src/robustness.py --model lstm --corruption jitter
echo "8. Small-data experiments..."
python3 src/run_small_data.py --plot
echo "9. Statistical summaries..."
python3 src/stats.py
echo "10. Figures from CSVs..."
python3 src/plot_results.py
echo "Done. See results/ and results/figures/"
