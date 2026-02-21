#!/usr/bin/env python3
"""
Prefetch OpenForesight-style datasets from Hugging Face into shared cache.

Examples:
  python aisa_scripts/data/download_openforesight_hf_aisa.py \
    --dataset "org/openforesight"

  python aisa_scripts/data/download_openforesight_hf_aisa.py \
    --dataset "https://huggingface.co/datasets/org/openforesight" \
    --splits train test validation
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Iterable

from datasets import load_dataset


DEFAULT_CACHE_ROOT = "/mnt/nfs/datasets_ac/cache/huggingface"
DEFAULT_DATASETS_CACHE = f"{DEFAULT_CACHE_ROOT}/datasets"
DEFAULT_HUB_CACHE = f"{DEFAULT_CACHE_ROOT}/hub"


def normalize_dataset_ref(ref: str) -> str:
    """Accept either repo_id or full HF datasets URL and return repo_id."""
    s = ref.strip()
    s = s.rstrip("/")
    m = re.match(r"^https?://huggingface\.co/datasets/([^/]+/[^/]+)$", s)
    if m:
        return m.group(1)
    return s


def prefetch(dataset: str, splits: Iterable[str], cache_dir: str) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    for split in splits:
        try:
            ds = load_dataset(dataset, split=split, cache_dir=cache_dir)
            print(f"[OK] split={split:<10} rows={len(ds)}")
        except Exception as e:
            print(f"[SKIP] split={split:<10} error={e}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prefetch Hugging Face dataset splits to shared cache")
    parser.add_argument(
        "--dataset",
        required=True,
        help="HF dataset repo_id (org/name) or full URL (https://huggingface.co/datasets/org/name)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        help="Splits to prefetch (default: train validation test)",
    )
    parser.add_argument(
        "--datasets_cache",
        default=DEFAULT_DATASETS_CACHE,
        help=f"HF datasets cache path (default: {DEFAULT_DATASETS_CACHE})",
    )
    parser.add_argument(
        "--hub_cache",
        default=DEFAULT_HUB_CACHE,
        help=f"HF hub cache path (default: {DEFAULT_HUB_CACHE})",
    )
    args = parser.parse_args()

    dataset = normalize_dataset_ref(args.dataset)

    # Keep caches in shared reusable storage.
    os.environ["HF_DATASETS_CACHE"] = args.datasets_cache
    os.environ["HF_HUB_CACHE"] = args.hub_cache
    os.makedirs(args.datasets_cache, exist_ok=True)
    os.makedirs(args.hub_cache, exist_ok=True)

    print(f"Dataset: {dataset}")
    print(f"HF_DATASETS_CACHE={args.datasets_cache}")
    print(f"HF_HUB_CACHE={args.hub_cache}")

    prefetch(dataset, args.splits, args.datasets_cache)
    print("\nDone.")
    print(f"Use this in config or override: dataset_path: \"{dataset}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
