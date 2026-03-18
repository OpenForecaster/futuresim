"""
Preprocess the Hendrycks MATH benchmark dataset to parquet format for OPD training.

Train split: Level 5 problems only (hardest).
Test split: All levels (for broader eval coverage).

Usage:
    python examples/train/on_policy_distillation/prepare_math_level5.py \
        --output_dir /fast/nchandak/forecast-sim/data/math_level5
"""

import argparse
import os
import sys

import datasets

# Add the skyrl-gym package so we can reuse the AIME answer extraction utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "skyrl-gym"))
from skyrl_gym.envs.aime.utils import last_boxed_only_string, remove_boxed, normalize_final_answer


def extract_ground_truth(solution: str) -> str | None:
    """Extract the ground truth answer from a MATH solution string.

    Finds the last \\boxed{...} expression and returns the normalized content.
    """
    boxed = last_boxed_only_string(solution)
    if boxed is None:
        return None
    return normalize_final_answer(remove_boxed(boxed))


def make_map_fn(split: str):
    def process_fn(example, idx):
        problem = example["problem"]
        solution = example["solution"]
        level = example["level"]
        subject = example["subject"]

        # Use the pre-extracted answer if available, fall back to boxed extraction
        ground_truth = example.get("answer") or extract_ground_truth(solution)
        if ground_truth is None:
            return None

        question = problem + '\nPlease reason step by step, and put your final answer on its own line after "Answer:".'

        return {
            "data_source": "nlile/hendrycks-MATH-benchmark",
            "prompt": [{"role": "user", "content": question}],
            "reward_model": {"ground_truth": ground_truth, "solution": solution, "style": "rule"},
            "extra_info": {
                "split": split,
                "index": idx,
                "level": level,
                "type": subject,
                "solution": solution,
            },
        }

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Hendrycks MATH Level 5 dataset for OPD")
    parser.add_argument("--output_dir", default="/fast/nchandak/forecast-sim/data/math_level5")
    args = parser.parse_args()

    args.output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    data_source = "nlile/hendrycks-MATH-benchmark"

    # Load dataset
    print(f"Loading dataset from {data_source}...")
    dataset = datasets.load_dataset(data_source)

    # Train split: Level 5 only
    train_dataset = dataset["train"]
    print(f"Train split total: {len(train_dataset)} examples")
    train_dataset = train_dataset.filter(lambda x: x["level"] == 5)
    print(f"Train split after Level 5 filter: {len(train_dataset)} examples")

    # Test split: all levels
    test_dataset = dataset["test"]
    print(f"Test split total: {len(test_dataset)} examples")

    # Process
    train_dataset = train_dataset.map(make_map_fn("train"), with_indices=True, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(make_map_fn("test"), with_indices=True, remove_columns=test_dataset.column_names)

    # Filter out any rows where boxed extraction failed (ground_truth was None)
    initial_train = len(train_dataset)
    train_dataset = train_dataset.filter(lambda x: x["reward_model"] is not None)
    initial_test = len(test_dataset)
    test_dataset = test_dataset.filter(lambda x: x["reward_model"] is not None)

    print(f"Train: {len(train_dataset)}/{initial_train} examples after filtering failed extractions")
    print(f"Test: {len(test_dataset)}/{initial_test} examples after filtering failed extractions")

    # Save
    train_path = os.path.join(args.output_dir, "train.parquet")
    test_path = os.path.join(args.output_dir, "test.parquet")
    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)

    print(f"\nSaved train to {train_path}")
    print(f"Saved test to {test_path}")
