#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
import statistics
from collections import Counter
from typing import List

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib is not installed. Please install it using 'pip install matplotlib'.")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-question update counts (prediction events) for forecast runs."
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="Path to a specific run directory containing actions.jsonl (repeatable).",
    )
    parser.add_argument(
        "--run-root",
        action="append",
        default=[],
        help=(
            "Path to a run root containing timestamp subdirectories. "
            "When provided, the latest timestamp folder is used (repeatable)."
        ),
    )
    parser.add_argument(
        "--include-zero-from-market",
        action="store_true",
        help=(
            "Include qids from market.csv with zero update count. "
            "If omitted, only qids with >=1 update are plotted."
        ),
    )
    parser.add_argument(
        "--output-name",
        default="update_distrib.png",
        help="Output filename under each run's plots/ directory (default: update_distrib.png).",
    )
    parser.add_argument(
        "--style",
        choices=["hist", "qid-bar"],
        default="hist",
        help="Plot style: histogram over update counts (hist) or per-qid bars (qid-bar).",
    )
    parser.add_argument(
        "--count-mode",
        choices=["updates", "predictions"],
        default="updates",
        help=(
            "Count mode: 'updates' uses max(predictions-1, 0), "
            "'predictions' uses raw prediction event count."
        ),
    )
    return parser.parse_args()


def find_latest_run_dir(run_root: str) -> str:
    if not os.path.isdir(run_root):
        raise FileNotFoundError(f"Run root does not exist: {run_root}")

    subdirs = [
        name
        for name in os.listdir(run_root)
        if os.path.isdir(os.path.join(run_root, name))
    ]
    if not subdirs:
        raise FileNotFoundError(f"No run subdirectories found in: {run_root}")

    return os.path.join(run_root, sorted(subdirs)[-1])


def sortable_qid(qid: str):
    try:
        return (0, int(qid))
    except ValueError:
        return (1, qid)


def load_qids_from_market(market_csv_path: str) -> List[str]:
    qids = []
    with open(market_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qids.append(str(row["qid"]))
    return qids


def compute_update_counts(actions_jsonl_path: str) -> Counter:
    counts = Counter()
    with open(actions_jsonl_path) as f:
        for line in f:
            record = json.loads(line)
            if record.get("type") == "prediction":
                counts[str(record["question_id"])] += 1
    return counts


def plot_run(
    run_dir: str,
    include_zeros: bool,
    output_name: str,
    style: str,
    count_mode: str,
) -> str:
    actions_path = os.path.join(run_dir, "actions.jsonl")
    market_path = os.path.join(run_dir, "market.csv")

    if not os.path.isfile(actions_path):
        raise FileNotFoundError(f"Missing actions.jsonl in run dir: {run_dir}")

    update_counts = compute_update_counts(actions_path)

    if include_zeros:
        if not os.path.isfile(market_path):
            raise FileNotFoundError(
                f"--include-zero-from-market was set but market.csv is missing in: {run_dir}"
            )
        qids = sorted(load_qids_from_market(market_path), key=sortable_qid)
        y_vals = [update_counts.get(qid, 0) for qid in qids]
    else:
        qids = sorted(update_counts.keys(), key=sortable_qid)
        y_vals = [update_counts[qid] for qid in qids]

    if count_mode == "updates":
        y_vals = [max(v - 1, 0) for v in y_vals]

    if not qids:
        raise ValueError(f"No qids available to plot for run: {run_dir}")

    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    output_path = os.path.join(plots_dir, output_name)

    # Keep old qid-bar mode available, but default to histogram for distribution shape.
    if style == "qid-bar":
        fig_width = max(14, min(60, len(qids) * 0.18))
        plt.figure(figsize=(fig_width, 8))
        plt.bar(range(len(qids)), y_vals, width=0.9)
        plt.xlabel("Question ID (qid)")
        plt.ylabel("Update count")
        plt.title(f"Question Update Distribution\\n{run_dir}")

        step = max(1, len(qids) // 40)
        tick_positions = list(range(0, len(qids), step))
        tick_labels = [qids[i] for i in tick_positions]
        plt.xticks(tick_positions, tick_labels, rotation=90, fontsize=8)
        plt.grid(axis="y", linestyle="--", alpha=0.4)
    else:
        min_count = min(y_vals)
        max_count = max(y_vals)
        bin_edges = list(range(min_count, max_count + 2))

        plt.figure(figsize=(12, 7))
        plt.hist(y_vals, bins=bin_edges, align="left", rwidth=0.9, edgecolor="black")
        x_label = (
            "Update count per question"
            if count_mode == "updates"
            else "Prediction count per question"
        )
        plt.xlabel(x_label)
        plt.ylabel("Number of questions")
        mean_val = statistics.mean(y_vals)
        median_val = statistics.median(y_vals)
        title_prefix = (
            "Update Count Distribution"
            if count_mode == "updates"
            else "Prediction Count Distribution"
        )
        plt.title(
            f"{title_prefix}\\n{run_dir}\\n"
            f"n={len(y_vals)}, mean={mean_val:.2f}, median={median_val:.2f}"
        )

        if max_count - min_count <= 60:
            plt.xticks(range(min_count, max_count + 1))
        plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def main() -> None:
    args = parse_args()

    run_dirs = list(args.run_dir)
    for run_root in args.run_root:
        run_dirs.append(find_latest_run_dir(run_root))

    if not run_dirs:
        raise ValueError("Provide at least one --run-dir or --run-root")

    for run_dir in run_dirs:
        out = plot_run(
            run_dir=run_dir,
            include_zeros=args.include_zero_from_market,
            output_name=args.output_name,
            style=args.style,
            count_mode=args.count_mode,
        )
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
