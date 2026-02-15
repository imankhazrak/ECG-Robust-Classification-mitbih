"""
Download MIT-BIH Arrhythmia Database using WFDB into the project data directory.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wfdb

from utils import DATA_DIR

# Standard 48 records in MIT-BIH Arrhythmia Database (47 patients, 48 recordings)
MITBIH_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]


def download_all(destination: Path, overwrite: bool = False) -> list[str]:
    """
    Download all MIT-BIH records to destination. Uses WFDB to fetch from PhysioNet.
    Returns list of record names that were downloaded or already present.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for rec in MITBIH_RECORDS:
        dat_file = destination / f"{rec}.dat"
        if dat_file.exists() and not overwrite:
            downloaded.append(rec)
            continue
        try:
            # rdrecord with pn_dir downloads from PhysioNet and we then write locally
            record = wfdb.rdrecord(rec, pn_dir="mitdb")
            wfdb.wrsamp(
                rec,
                fs=record.fs,
                units=record.units,
                sig_name=record.sig_name,
                p_signal=record.p_signal,
                write_dir=str(destination),
            )
            ann = wfdb.rdann(rec, "atr", pn_dir="mitdb")
            ann.wrann(write_dir=str(destination))
            downloaded.append(rec)
        except Exception as e:
            print(f"Warning: failed to download {rec}: {e}")
    return downloaded


def main():
    parser = argparse.ArgumentParser(description="Download MIT-BIH Arrhythmia Database")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR, help="Directory to save records")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()
    out = download_all(args.output_dir, overwrite=args.overwrite)
    print(f"Done. Records in {args.output_dir}: {len(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
