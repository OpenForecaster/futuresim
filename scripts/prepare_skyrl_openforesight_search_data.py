#!/usr/bin/env python3
"""Prepare SkyRL parquet data for OpenForesight warmup search training."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from skyrl_integration.data import prepare_openforesight_search_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_path",
        default=os.getenv("FSIM_DATASET_PATH", "/is/cluster/fast/sgoel/forecasting/qs/OpenForesight/data"),
        help="Path containing OpenForesight split parquet files (train-*.parquet, validation-*.parquet)",
    )
    parser.add_argument(
        "--prepared-data-dir",
        dest="prepared_data_dir",
        default=os.getenv("FSIM_SKYRL_PREPARED_DATA_DIR", ""),
        help="Shared directory for SkyRL train.parquet and validation.parquet (default: FSIM_SKYRL_PREPARED_DATA_DIR)",
    )
    parser.add_argument(
        "--search_db",
        required=True,
        help="Path to LanceDB index used by the environment",
    )
    parser.add_argument(
        "--embedding_model",
        default=os.getenv("FSIM_EMBEDDING_MODEL", ""),
        help="Local path to the embedding model used for LanceDB semantic or hybrid search.",
    )
    parser.add_argument(
        "--embedding_gpu_mem",
        type=float,
        default=0.3,
        help="GPU memory fraction for the embedding vLLM server.",
    )
    parser.add_argument(
        "--aux_cuda_visible_devices",
        default="",
        help="Optional CUDA_VISIBLE_DEVICES spec for the embedding vLLM server.",
    )

    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="validation")
    parser.add_argument("--lookback_days", type=int, default=7)
    parser.add_argument(
        "--global_sim_date",
        default=None,
        help="Optional YYYY-MM-DD simulation date to use for every question instead of resolution_date - lookback_days.",
    )
    parser.add_argument(
        "--search_max_date",
        default=None,
        help="Optional YYYY-MM-DD ceiling on latest searchable article date (min with sim_date - search_cutoff_days).",
    )

    parser.add_argument("--search_type", default="hybrid", choices=["keyword", "semantic", "hybrid"])
    parser.add_argument("--max_search_results", type=int, default=5)
    parser.add_argument("--search_cutoff_days", type=int, default=0)
    parser.add_argument("--search_min_days", type=int, default=0)
    parser.add_argument("--max_snippet_chars", type=int, default=1200)
    parser.add_argument("--max_outcomes_per_question", type=int, default=5)
    parser.add_argument("--warmup_max_actions", type=int, default=None)
    parser.add_argument("--warmup_max_total_tokens", type=int, default=None)
    parser.add_argument("--warmup_submit_reserve_tokens", type=int, default=8192)
    parser.add_argument("--warmup_force_submit_threshold_tokens", type=int, default=16384)
    parser.add_argument("--budget_model_path", default="")
    parser.add_argument("--allow_substring_match", action="store_true", default=True)
    parser.add_argument("--no_allow_substring_match", action="store_true")
    parser.add_argument("--matching", default="exact", choices=["exact", "openrouter"])
    parser.add_argument("--matcher", default="")

    parser.add_argument("--resolution_start", default=None, help="Optional YYYY-MM-DD filter")
    parser.add_argument("--resolution_end", default=None, help="Optional YYYY-MM-DD filter")
    parser.add_argument("--max_train_questions", type=int, default=None)
    parser.add_argument("--max_val_questions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if not str(args.prepared_data_dir).strip():
        parser.error("Pass --prepared-data-dir or set FSIM_SKYRL_PREPARED_DATA_DIR to a shared cache directory.")

    allow_substring_match = args.allow_substring_match and not args.no_allow_substring_match

    result = prepare_openforesight_search_dataset(
        dataset_path=args.dataset_path,
        prepared_data_dir=args.prepared_data_dir,
        search_db=args.search_db,
        embedding_model=args.embedding_model,
        embedding_gpu_mem=args.embedding_gpu_mem,
        aux_cuda_visible_devices=args.aux_cuda_visible_devices,
        train_split=args.train_split,
        val_split=args.val_split,
        lookback_days=args.lookback_days,
        global_sim_date=args.global_sim_date,
        search_max_date=args.search_max_date,
        search_type=args.search_type,
        search_topk=args.max_search_results,
        search_cutoff_days=args.search_cutoff_days,
        search_min_days=args.search_min_days,
        resolution_start=args.resolution_start,
        resolution_end=args.resolution_end,
        max_train_questions=args.max_train_questions,
        max_val_questions=args.max_val_questions,
        seed=args.seed,
        max_snippet_chars=args.max_snippet_chars,
        allow_substring_match=allow_substring_match,
        matching=args.matching,
        matcher=args.matcher,
        warmup_max_actions=args.warmup_max_actions,
        warmup_max_total_tokens=args.warmup_max_total_tokens,
        warmup_submit_reserve_tokens=args.warmup_submit_reserve_tokens,
        warmup_force_submit_threshold_tokens=args.warmup_force_submit_threshold_tokens,
        budget_model_path=args.budget_model_path,
        max_outcomes_per_question=args.max_outcomes_per_question,
    )

    print(f"train rows: {result.train.rows} -> {result.train.path}")
    print(f"validation rows: {result.validation.rows} -> {result.validation.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
