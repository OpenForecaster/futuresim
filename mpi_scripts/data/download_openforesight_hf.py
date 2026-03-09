#!/usr/bin/env python3
"""
Prefetch OpenForesight-style datasets from Hugging Face and save as parquet.

The MPI cluster's /fast filesystem doesn't support flock, so the HF datasets
cache mechanism fails. This script downloads via a /home-based cache (which
supports flock) and writes parquet files to a /fast output directory that
the simulation can load directly via dataset_path.

Examples:
  python mpi_scripts/data/download_openforesight_hf.py \
    --dataset nikhilchandak/OpenForesight

  python mpi_scripts/data/download_openforesight_hf.py \
    --dataset nikhilchandak/OpenForesight \
    --splits train test validation skysports2025 \
    --output /fast/nchandak/datasets/OpenForesight
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Iterable

from datasets import load_dataset


DEFAULT_HF_CACHE = os.path.expanduser("~/.cache/huggingface/datasets")
DEFAULT_OUTPUT = "/fast/nchandak/datasets/OpenForesight"


def normalize_dataset_ref(ref: str) -> str:
    """Accept either repo_id or full HF datasets URL and return repo_id."""
    s = ref.strip().rstrip("/")
    m = re.match(r"^https?://huggingface\.co/datasets/([^/]+/[^/]+)$", s)
    if m:
        return m.group(1)
    return s


def download_and_save(dataset: str, splits: Iterable[str], cache_dir: str, output_dir: str) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    for split in splits:
        try:
            print(f"Downloading split={split} ...")
            ds = load_dataset(dataset, split=split, cache_dir=cache_dir)
            print(f"  Loaded {len(ds)} rows")

            out_path = os.path.join(output_dir, f"{split}-00000-of-00001.parquet")
            ds.to_parquet(out_path)
            print(f"  Saved to {out_path}")
        except Exception as e:
            print(f"[SKIP] split={split:<10} error={e}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download HF dataset and save as parquet for MPI cluster"
    )
    parser.add_argument(
        "--dataset", required=True,
        help="HF dataset repo_id (e.g. nikhilchandak/OpenForesight)",
    )
    parser.add_argument(
        "--splits", nargs="+",
        default=["train", "validation", "test", "skysports2025"],
        help="Splits to download (default: train validation test skysports2025)",
    )
    parser.add_argument(
        "--cache_dir", default=DEFAULT_HF_CACHE,
        help=f"HF download cache (must be on flock-capable FS, default: {DEFAULT_HF_CACHE})",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output directory for parquet files (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    dataset = normalize_dataset_ref(args.dataset)

    os.environ["HF_DATASETS_CACHE"] = args.cache_dir
    os.environ["HF_HUB_CACHE"] = os.path.join(os.path.dirname(args.cache_dir), "hub")

    print(f"Dataset:    {dataset}")
    print(f"Cache:      {args.cache_dir}")
    print(f"Output dir: {args.output}")
    print()

    download_and_save(dataset, args.splits, args.cache_dir, args.output)

    print(f"\nDone. Use in config: dataset_path: \"{args.output}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
