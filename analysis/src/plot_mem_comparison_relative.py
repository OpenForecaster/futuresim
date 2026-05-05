import argparse
import os
import sys

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib is not installed. Please install it using 'pip install matplotlib'.")
    sys.exit(1)

import plot_config  # noqa: F401  (applies project-wide science+serif style)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two simulation runs using relative day index instead of calendar date."
    )
    parser.add_argument(
        "--run_a_dir",
        type=str,
        required=True,
        help="Path to the first run directory (containing daily_metrics.csv)",
    )
    parser.add_argument(
        "--run_b_dir",
        type=str,
        required=True,
        help="Path to the second run directory (containing daily_metrics.csv)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save plots",
    )
    parser.add_argument(
        "--run_a_label",
        type=str,
        default="Run A",
        help="Label for the first run",
    )
    parser.add_argument(
        "--run_b_label",
        type=str,
        default="Run B",
        help="Label for the second run",
    )
    return parser.parse_args()


def load_metrics(directory):
    csv_path = os.path.join(directory, "daily_metrics.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by="date").reset_index(drop=True)
    df["relative_day"] = range(len(df))
    return df


def add_day_zero_deltas(df, metrics):
    df = df.copy()
    for col, _, _ in metrics:
        baseline = df[col].iloc[0]
        df[f"{col}_delta"] = df[col] - baseline
    return df


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df_a = load_metrics(args.run_a_dir)
    df_b = load_metrics(args.run_b_dir)

    metrics = [
        ("avg_brier", "Average Brier Skill Score", "higher is better"),
        ("accuracy", "Accuracy (%)", "higher is better"),
        ("exp_acc", "Expected Accuracy", "higher is better"),
    ]

    df_a = add_day_zero_deltas(df_a, metrics)
    df_b = add_day_zero_deltas(df_b, metrics)

    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 4 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    colors = {"a": "#2196F3", "b": "#FF5722"}

    for ax, (col, title, direction) in zip(axes, metrics):
        delta_col = f"{col}_delta"
        ax.plot(
            df_a["relative_day"],
            df_a[delta_col],
            label=args.run_a_label,
            color=colors["a"],
            linewidth=1.5,
            alpha=0.9,
        )
        ax.plot(
            df_b["relative_day"],
            df_b[delta_col],
            label=args.run_b_label,
            color=colors["b"],
            linewidth=1.5,
            alpha=0.9,
        )

        subtitle = f"  ({direction})" if direction else ""
        ax.set_title(f"{title} Change vs Day 0{subtitle}", fontsize=12, fontweight="bold", loc="left")
        ax.set_ylabel(f"Delta {col.replace('_', ' ').title()}", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.axhline(0, color="black", linewidth=1.0, alpha=0.5)
        ax.legend(loc="lower right", fontsize=9)

    axes[-1].set_xlabel("Relative Day Index", fontsize=11)
    axes[-1].set_xlim(left=0)

    plt.tight_layout()
    output_path = os.path.join(args.output_dir, "relative_day_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved to {output_path}")


if __name__ == "__main__":
    main()
