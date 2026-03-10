#!/usr/bin/env python3
"""
Prepare distillation train/val/test splits from eval_with_answer JSONL outputs.

Reads all .jsonl files from an eval output folder, reformats each record into
the verl training parquet format, performs a random train/val/test split, and
saves as parquet files inside a training/ subdirectory of the input folder.

Usage:
    python syntheticQA/prepare_distillation.py --input /path/to/eval_with_answer_output/
    python syntheticQA/prepare_distillation.py --input /path/to/folder/ --train_frac 0.90 --val_frac 0.05 --test_frac 0.05
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List

import datasets


# ---------------------------------------------------------------------------
# Prompt construction (same as eval_qa.py build_eval_prompt)
# ---------------------------------------------------------------------------

def build_eval_prompt(question: str, background: str) -> str:
    """Build the evaluation prompt for a single QA pair."""
    parts = ["You will be asked a factual questions. Please provide your best answer to it."]

    parts.append(f"\nQuestion:\n{question.strip()}")

    if background and background.strip():
        parts.append(f"\nBackground:\n{background.strip()}")

    parts.append(
        "\nThink step by step, then provide your final answer (keep it concise) inside <answer>...</answer> tags."
        "\nFor example: <answer>The answer is X.</answer> /no_think"
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_records(input_dir: str) -> List[dict]:
    """Load all JSONL records from a directory, pooling across sources."""
    root = Path(input_dir)
    all_records = []

    jsonl_files = sorted(f for f in root.iterdir() if f.is_file() and f.suffix == ".jsonl")
    if not jsonl_files:
        print(f"No .jsonl files found in {input_dir}")
        return all_records

    min_records = 5
    for fpath in jsonl_files:
        records = []
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    record["_source_file"] = fpath.name
                    records.append(record)
                except json.JSONDecodeError:
                    continue
        if len(records) < min_records:
            print(f"  Skipping {fpath.name}: only {len(records)} records (need >= {min_records})")
            continue
        all_records.extend(records)
        print(f"  Loaded {len(records)} records from {fpath.name}")

    print(f"Total: {len(all_records)} records from {len(jsonl_files)} files")

    def _get_output(r):
        return r.get("raw_model_output") or r.get("model_answer_raw") or ""

    before = len(all_records)
    all_records = [
        r for r in all_records
        if "<answer>" in _get_output(r) and "</answer>" in _get_output(r)
    ]
    no_tag = before - len(all_records)
    if no_tag:
        print(f"Filtered out {no_tag} records missing <answer> tags ({len(all_records)} remaining)")

    before = len(all_records)
    all_records = [
        r for r in all_records
        if _get_output(r).count("<answer>") == 1 and _get_output(r).count("</answer>") == 1
    ]
    multi_tag = before - len(all_records)
    if multi_tag:
        print(f"Filtered out {multi_tag} records with multiple <answer> tags ({len(all_records)} remaining)")

    before = len(all_records)
    filtered = []
    for r in all_records:
        gt = str(r.get("ground_truth", "")).strip().lower()
        ma = str(r.get("model_answer", "")).strip().lower()
        if gt and ma and gt == ma:
            filtered.append(r)
    mismatch = before - len(filtered)
    all_records = filtered
    if mismatch:
        print(f"Filtered out {mismatch} records where model answer != ground truth ({len(all_records)} remaining)")

    return all_records


# ---------------------------------------------------------------------------
# Record transformation
# ---------------------------------------------------------------------------

def transform_record(record: dict, idx: int, split: str) -> dict:
    """Transform a single eval output record into the verl training format."""
    question = record.get("question", "")
    background = record.get("background", "")
    ground_truth = record.get("ground_truth", "")
    model_answer = record.get("model_answer", "")
    raw_model_output = record.get("raw_model_output", "")
    metadata = record.get("metadata", {})

    prompt_text = record.get("original_prompt") or record.get("prompt_text") or ""
    if not prompt_text.strip():
        prompt_text = build_eval_prompt(question, background)

    # Build the full response (model output as-is)
    response = raw_model_output

    return {
        "data_source": "syntheticqa/distillation",
        "prompt": [
            {
                "role": "user",
                "content": prompt_text,
            }
        ],
        "ability": "qa",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "split": split,
            "index": idx,
            "question": question,
            "background": background,
            "ground_truth": ground_truth,
            "model_answer": model_answer,
            "response": response,
            "prompt": prompt_text,
            "source_file": record.get("_source_file", ""),
            "eval_type": metadata.get("eval_type", ""),
            "eval_model": metadata.get("eval_model", ""),
            "gen_model": metadata.get("gen_model", ""),
            "gen_effort": metadata.get("gen_effort", ""),
            "article_title": metadata.get("article_title", ""),
            "article_url": metadata.get("article_url", ""),
            "source_domain": metadata.get("source_domain", ""),
            "date_publish": metadata.get("date_publish", ""),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare distillation train/val/test splits from eval output JSONL files."
    )
    parser.add_argument(
        "--input", type=str, default="/fast/nchandak/forecast-sim/news/syntheticqa/gpt-oss-20b_q20_a1000_fd2025-04-01_medium_20260213_000154/Qwen3-8B_with_answer_no_think_20260213_022031/",
        help="Path to the eval output folder containing .jsonl files",
    )
    parser.add_argument(
        "--train_frac", type=float, default=0.95,
        help="Fraction of data for training (default: 0.95)",
    )
    parser.add_argument(
        "--val_frac", type=float, default=0.03,
        help="Fraction of data for validation (default: 0.03)",
    )
    parser.add_argument(
        "--test_frac", type=float, default=0.02,
        help="Fraction of data for testing (default: 0.02)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shuffling (default: 42)",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Save as plain JSONL files (verl format) instead of HF parquet.",
    )
    parser.add_argument(
        "--raw_jsonl", action="store_true",
        help="Save raw JSONL (original records, no verl transformation). "
             "Useful for inspection or non-training purposes.",
    )
    args = parser.parse_args()

    # Validate fractions
    total_frac = args.train_frac + args.val_frac + args.test_frac
    if abs(total_frac - 1.0) > 0.01:
        print(f"Warning: fractions sum to {total_frac:.3f}, not 1.0. Proceeding anyway.")

    # Load all records
    print(f"Loading records from: {args.input}")
    all_records = load_all_records(args.input)
    if not all_records:
        print("No records found. Exiting.")
        sys.exit(1)

    # Shuffle
    random.seed(args.seed)
    random.shuffle(all_records)

    # Compute split boundaries
    n = len(all_records)
    n_train = int(args.train_frac * n)
    n_val = int(args.val_frac * n)
    # test gets the remainder
    n_test = n - n_train - n_val

    train_records = all_records[:n_train]
    val_records = all_records[n_train:n_train + n_val]
    test_records = all_records[n_train + n_val:]

    print(f"\nSplit sizes: train={len(train_records)}, val={len(val_records)}, test={len(test_records)}")

    # Save to training/ (or raw/) subdirectory
    if args.raw_jsonl:
        output_dir = Path(args.input) / "raw"
        output_dir.mkdir(parents=True, exist_ok=True)

        splits = [("train", train_records), ("val", val_records), ("test", test_records)]
        for split_name, split_data in splits:
            out_path = output_dir / f"{split_name}_{len(split_data)}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for record in split_data:
                    rec_copy = {k: v for k, v in record.items() if k != "_source_file"}
                    f.write(json.dumps(rec_copy, ensure_ascii=False) + "\n")
            print(f"Saved {split_name} ({len(split_data)} records) -> {out_path}")
    else:
        # Transform records into verl format
        train_data = [transform_record(r, i, "train") for i, r in enumerate(train_records)]
        val_data = [transform_record(r, i, "val") for i, r in enumerate(val_records)]
        test_data = [transform_record(r, i, "test") for i, r in enumerate(test_records)]

        output_dir = Path(args.input) / "training"
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.dry_run:
            # Save as plain JSONL (verl format)
            for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
                out_path = output_dir / f"{split_name}_{len(split_data)}.jsonl"
                with open(out_path, "w", encoding="utf-8") as f:
                    for record in split_data:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"Saved {split_name} ({len(split_data)} records) -> {out_path}")
        else:
            # Save as HF parquet
            train_dataset = datasets.Dataset.from_list(train_data)
            val_dataset = datasets.Dataset.from_list(val_data)
            test_dataset = datasets.Dataset.from_list(test_data)

            train_path = output_dir / f"train_{len(train_dataset)}.parquet"
            val_path = output_dir / f"val_{len(val_dataset)}.parquet"
            test_path = output_dir / f"test_{len(test_dataset)}.parquet"

            train_dataset.to_parquet(str(train_path))
            print(f"Saved train ({len(train_dataset)} records) -> {train_path}")

            val_dataset.to_parquet(str(val_path))
            print(f"Saved val ({len(val_dataset)} records) -> {val_path}")

            test_dataset.to_parquet(str(test_path))
            print(f"Saved test ({len(test_dataset)} records) -> {test_path}")

    print(f"\nDone. Output at: {output_dir}")


if __name__ == "__main__":
    main()
