"""Convert verl-format forecasting parquet to SkyRL parquet format.

verl format:
  data_source, prompt, ability, reward_model, extra_info

SkyRL format:
  data_source, prompt, env_class, reward_spec, extra_info
"""

import argparse
import pyarrow as pa
import pyarrow.parquet as pq


def convert_verl_to_skyrl(input_path: str, output_path: str) -> None:
    table = pq.read_table(input_path)
    data = table.to_pydict()
    n = len(data["prompt"])
    print(f"Loaded {n} rows from {input_path}")

    out = {
        "data_source": data["data_source"],
        "prompt": data["prompt"],
        "env_class": ["forecast"] * n,
        "reward_spec": [],
        "extra_info": data["extra_info"],
    }

    for i in range(n):
        rm = data["reward_model"][i]
        ei = data["extra_info"][i]

        # Build reward_spec from reward_model + extra_info
        reward_spec = {
            "method": rm.get("style", "rule"),
            "ground_truth": rm.get("ground_truth", ""),
            # OPSD needs 'solution' — use ground_truth as the solution
            "solution": rm.get("ground_truth", ""),
        }
        out["reward_spec"].append(reward_spec)

    out_table = pa.table(out)
    pq.write_table(out_table, output_path)
    print(f"Wrote {n} rows to {output_path}")

    # Print distribution stats
    from collections import Counter
    source_counts = Counter(out["data_source"])
    print("Data source distribution:")
    for src, cnt in source_counts.most_common():
        print(f"  {src}: {cnt}")


def main():
    parser = argparse.ArgumentParser(description="Convert verl forecasting data to SkyRL format")
    parser.add_argument("--input", required=True, help="Input verl-format parquet file")
    parser.add_argument("--output", required=True, help="Output SkyRL-format parquet file")
    args = parser.parse_args()
    convert_verl_to_skyrl(args.input, args.output)


if __name__ == "__main__":
    main()
