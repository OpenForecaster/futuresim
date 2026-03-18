"""
Convert SDPO SciKnowEval chemistry dataset to SkyRL parquet format.

Usage:
    python -m examples.train.mcq.chemistry_dataset \
        --train_json /home/nchandak/SDPO/datasets/sciknoweval/chemistry/train.json \
        --test_json /home/nchandak/SDPO/datasets/sciknoweval/chemistry/test.json \
        --output_dir /fast/nchandak/forecast-sim/data/chemistry
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def convert_sdpo_to_skyrl(json_path: str, output_path: str):
    """Convert SDPO JSON to SkyRL parquet format."""
    with open(json_path) as f:
        rows = [json.loads(line) for line in f]

    records = []
    for i, row in enumerate(rows):
        records.append(
            {
                "data_source": "sciknoweval",
                "prompt": [
                    {"role": "system", "content": row["system"].strip()},
                    {"role": "user", "content": row["prompt"]},
                ],
                "env_class": "mcq",
                "reward_spec": {
                    "method": "rule",
                    "ground_truth": row["answer"],
                },
                "extra_info": {
                    "split": "train" if "train" in str(json_path) else "test",
                    "index": i,
                    "description": row.get("description", ""),
                    "kind": row.get("kind", "mcq"),
                    "elo": row.get("elo", 1500),
                },
            }
        )

    df = pd.DataFrame(records)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow")
    print(f"Wrote {len(df)} samples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--test_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    convert_sdpo_to_skyrl(args.train_json, f"{args.output_dir}/train.parquet")
    convert_sdpo_to_skyrl(args.test_json, f"{args.output_dir}/test.parquet")
