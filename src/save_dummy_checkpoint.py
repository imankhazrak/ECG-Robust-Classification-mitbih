"""Save an untrained LSTM checkpoint so eval/robustness can run without full training."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import yaml
import torch

from utils import PROJECT_ROOT, RESULTS_DIR, NUM_CLASSES
from models import ECG_LSTM

def main():
    with open(PROJECT_ROOT / "configs" / "lstm_baseline.yaml") as f:
        config = yaml.safe_load(f)
    model = ECG_LSTM(input_size=1, hidden_size=128, num_layers=2, dropout=0.3, num_classes=NUM_CLASSES)
    ckpt_dir = RESULTS_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for seed in [42, 1, 2, 3, 4]:
        path = ckpt_dir / f"lstm_inter_patient_seed{seed}.pt"
        torch.save({"model_state_dict": model.state_dict(), "config": config}, path)
        print("Saved", path)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
