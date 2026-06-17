"""
split_dataset.py — Split medical_dialogues_15k.csv into train/val/test sets.

Splits:  80% train | 10% val | 10% test  (stratified shuffle)
Output:  data/splits/train.csv, val.csv, test.csv
"""
import csv
import random
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────
INPUT_FILE = "data/raw/medical_dialogues_15k.csv"
OUTPUT_DIR = "data/splits"

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

RANDOM_SEED = 42

# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    src  = Path(INPUT_FILE)
    dest = Path(OUTPUT_DIR)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Reading {src} ...")
    with src.open("r", encoding="utf-8", newline="") as fh:
        reader   = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        rows     = list(reader)

    total = len(rows)
    print(f"Total rows: {total:,}")

    random.seed(RANDOM_SEED)
    random.shuffle(rows)

    n_train = int(total * TRAIN_RATIO)
    n_val   = int(total * VAL_RATIO)
    n_test  = total - n_train - n_val

    splits = {
        "train.csv": rows[:n_train],
        "val.csv":   rows[n_train : n_train + n_val],
        "test.csv":  rows[n_train + n_val :],
    }

    for filename, split_rows in splits.items():
        out_path = dest / filename
        with out_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_rows)
        print(f"  {filename:<12} {len(split_rows):>6,} rows  ->  {out_path}")

    print(f"\nDone. {n_train:,} train | {n_val:,} val | {n_test:,} test")


if __name__ == "__main__":
    main()
