# CNN+Transformer Confusion Matrix (Aggregate over 5 seeds)

Rows = True label, Cols = Predicted

|  | Normal | Atrial Premature | Premature ventricular contraction | Fusion of ventricular and normal | Fusion of paced and normal |
|---|---|---|---|---|---|
| **Normal** | 89876 | 436 | 137 | 78 | 58 |
| **Atrial Premature** | 240 | 2518 | 21 | 0 | 1 |
| **Premature ventricular contraction** | 85 | 17 | 7073 | 55 | 5 |
| **Fusion of ventricular and normal** | 88 | 2 | 50 | 660 | 0 |
| **Fusion of paced and normal** | 52 | 4 | 23 | 2 | 8284 |

## Per-seed summary
- **Seed 42**: Acc=0.9882, Macro F1=0.9309
- **Seed 1**: Acc=0.9856, Macro F1=0.9254
- **Seed 2**: Acc=0.9887, Macro F1=0.9395
- **Seed 3**: Acc=0.9876, Macro F1=0.9313
- **Seed 4**: Acc=0.9883, Macro F1=0.9338
