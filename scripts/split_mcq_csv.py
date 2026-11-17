"""Split an MCQ CSV into train/val CSVs (the released model used 80/20, seed 42)."""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    split = int(args.train_frac * len(df))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.iloc[:split].to_csv(out / "train.csv", index=False)
    df.iloc[split:].to_csv(out / "val.csv", index=False)
    print(f"train: {split} rows, val: {len(df) - split} rows -> {out}")


if __name__ == "__main__":
    main()
