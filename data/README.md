# Data Directory

This directory is populated by the download script. Run from the project root:

```bash
python src/download_data.py
```

This downloads the MIT-BIH Arrhythmia Database using WFDB into this folder. The preprocess script then reads from here and writes processed tensors (e.g. `processed_mlii.pt`) for faster training.
