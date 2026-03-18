#!/usr/bin/env python3
"""
Create a random 100-question subset of the skysports2025 split and save it
as skysports_100 so it can be used directly in simulation configs via:

    split: "skysports_100"

Usage:
  python mpi_scripts/data/create_skysports_100.py
  python mpi_scripts/data/create_skysports_100.py --data_dir /fast/nchandak/datasets/OpenForesight --seed 42
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

DEFAULT_DATA_DIR = "/fast/nchandak/datasets/OpenForesight"
SOURCE_SPLIT = "skysports2025"
TARGET_SPLIT = "skysports_100"
N = 100


def main() -> int:
    parser = argparse.ArgumentParser(description="Create skysports_100 subset parquet")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR,
                        help=f"Directory containing split parquets (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--n", type=int, default=N,
                        help=f"Number of questions to sample (default: {N})")
    parser.add_argument("--source", default=SOURCE_SPLIT,
                        help=f"Source split name (default: {SOURCE_SPLIT})")
    parser.add_argument("--target", default=TARGET_SPLIT,
                        help=f"Target split name (default: {TARGET_SPLIT})")
    args = parser.parse_args()

    src_path = os.path.join(args.data_dir, f"{args.source}-00000-of-00001.parquet")
    dst_path = os.path.join(args.data_dir, f"{args.target}-00000-of-00001.parquet")

    print(f"Loading  : {src_path}")
    df = pd.read_parquet(src_path)
    print(f"  Total rows: {len(df)}")

    subset = df.sample(n=args.n, random_state=args.seed).reset_index(drop=True)
    print(f"  Sampled  : {len(subset)} rows (seed={args.seed})")

    subset.to_parquet(dst_path, index=False)
    print(f"Saved to : {dst_path}")
    print(f"\nUse in config:  split: \"{args.target}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
