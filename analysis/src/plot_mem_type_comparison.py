import pandas as pd
import argparse
import os
import sys
import glob

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except ImportError:
    print("Error: matplotlib is not installed. Please install it using 'pip install matplotlib'.")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare multiple memory-type runs (multi-run avg + stddev)."
    )
    parser.add_argument(
        "--runs", nargs="+", required=True,
        help="Run specs as label:parent_dir. "
             "E.g., 'Active:/path/to/deepseek_active_mem_r00_restart'",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save plots")
    parser.add_argument(
        "--expected_runs", type=int, default=None,
        help="Optional expected number of runs per type.",
    )
    return parser.parse_args()


def _discover_run_csvs(parent_dir):
    """
    Discover daily_metrics.csv files under parent_dir.

    Each timestamped subdirectory (e.g. 26-03-16-02-05-17/) is treated as
    a separate run.  Layout expected:
        parent_dir/
            <timestamp_1>/daily_metrics.csv
            <timestamp_2>/daily_metrics.csv
            ...
    """
    parent_dir = os.path.abspath(parent_dir)
    if not os.path.isdir(parent_dir):
        print(f"Error: {parent_dir} is not a directory.")
        sys.exit(1)

    discovered = []
    for child in sorted(os.listdir(parent_dir)):
        child_path = os.path.join(parent_dir, child)
        if not os.path.isdir(child_path):
            continue
        csv_path = os.path.join(child_path, "daily_metrics.csv")
        if os.path.isfile(csv_path):
            discovered.append((child, csv_path))

    # Fallback: parent itself has a daily_metrics.csv (single-run case).
    if not discovered:
        csv_path = os.path.join(parent_dir, "daily_metrics.csv")
        if os.path.isfile(csv_path):
            discovered.append((os.path.basename(parent_dir), csv_path))

    return discovered


def load_all_runs(parent_dir, expected_runs=None):
    """Load per-run daily_metrics.csv discovered under parent_dir."""
    dfs = []
    discovered = _discover_run_csvs(parent_dir)
    for run_name, csv_path in discovered:
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        df['run'] = run_name
        dfs.append(df)

    if not dfs:
        print(f"Error: No daily_metrics.csv found under {parent_dir}")
        sys.exit(1)

    print(f"  Loaded {len(dfs)} runs from {parent_dir}")
    for df in dfs:
        print(f"    {df['run'].iloc[0]}: {len(df)} days "
              f"({df['date'].min().date()} to {df['date'].max().date()})")
    if expected_runs is not None and len(dfs) != expected_runs:
        print(f"  Warning: expected {expected_runs} runs, found {len(dfs)} in {parent_dir}")

    return dfs


def aggregate_runs(dfs, metrics_cols):
    """Compute per-date mean and std across runs."""
    combined = pd.concat(dfs, ignore_index=True)
    grouped = combined.groupby('date')[metrics_cols]
    mean_df = grouped.mean().reset_index().sort_values('date')
    std_df = grouped.std().fillna(0.0).reset_index().sort_values('date')
    count_df = grouped.size().reset_index(name='n_runs').sort_values('date')
    return mean_df, std_df, count_df


def parse_run_spec(spec):
    """Parse 'label:parent_dir' into (label, parent_dir)."""
    parts = spec.split(":", 1)
    if len(parts) < 2:
        print(f"Error: invalid run spec '{spec}'. Expected label:parent_dir")
        sys.exit(1)
    return parts[0], parts[1]


# Color palette for up to 6 run types.
COLORS = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800', '#00BCD4']


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    metrics = [
        ('avg_brier', 'Average Brier Skill Score', 'higher is better'),
        ('accuracy', 'Accuracy (%)', 'higher is better'),
        ('exp_acc', 'Avg Probability of Correct Outcome', 'higher is better'),
    ]
    metric_cols = [m[0] for m in metrics]

    # Load and aggregate each run type.
    run_data = []  # list of (label, mean_df, std_df, n_runs, color)
    for i, spec in enumerate(args.runs):
        label, parent_dir = parse_run_spec(spec)
        print(f"Loading '{label}' runs...")
        dfs = load_all_runs(parent_dir, expected_runs=args.expected_runs)
        mean_df, std_df, count_df = aggregate_runs(dfs, metric_cols)
        n = len(dfs)
        run_data.append((label, mean_df, std_df, n, COLORS[i % len(COLORS)]))

    # Align on common date range.
    common_start = max(rd[1]['date'].min() for rd in run_data)
    common_end = min(rd[1]['date'].max() for rd in run_data)
    aligned = []
    for label, mean_df, std_df, n, color in run_data:
        mask = (mean_df['date'] >= common_start) & (mean_df['date'] <= common_end)
        mean_df = mean_df[mask].copy()
        std_df = std_df[(std_df['date'] >= common_start) & (std_df['date'] <= common_end)].copy()
        aligned.append((label, mean_df, std_df, n, color))
    run_data = aligned

    print(f"\nCommon date range: {common_start.date()} to {common_end.date()}")

    # --- Mean + stddev comparison plot ---
    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 4 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, (col, title, direction) in zip(axes, metrics):
        for label, mean_df, std_df, n, color in run_data:
            ax.plot(mean_df['date'], mean_df[col],
                    label=f"{label} (n={n})", color=color, linewidth=1.5, alpha=0.9)
            ax.fill_between(mean_df['date'],
                            mean_df[col] - std_df[col],
                            mean_df[col] + std_df[col],
                            color=color, alpha=0.12)

        subtitle = f"  ({direction})" if direction else ""
        ax.set_title(f"{title}{subtitle}", fontsize=12, fontweight='bold', loc='left')
        ax.set_ylabel(col.replace('_', ' ').title(), fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='lower right', fontsize=9)

        # Final-value annotations (stagger vertically to avoid overlap).
        offsets = [(-12, 10), (-12, -10), (-12, 20), (-12, -20), (-12, 30), (-12, -30)]
        for idx, (label, mean_df, std_df, n, color) in enumerate(run_data):
            last_mean = mean_df.iloc[-1]
            last_std = std_df.iloc[-1]
            val = last_mean[col]
            sd = last_std[col]
            text = f"{val:.4f} ± {sd:.4f}" if pd.notna(sd) and sd > 0 else f"{val:.4f}"
            ox, oy = offsets[idx % len(offsets)]
            ax.annotate(text, xy=(last_mean['date'], val),
                        fontsize=8, color=color, fontweight='bold',
                        xytext=(ox, oy), textcoords='offset points', va='center',
                        arrowprops=dict(arrowstyle='-', color=color, alpha=0.5))

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(axes[-1].get_xticklabels(), rotation=45, ha='right')
    axes[-1].set_xlabel('Date', fontsize=11)

    fig.suptitle('Memory Type Comparison (Multi-Run Average)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    output_path = os.path.join(args.output_dir, 'mem_type_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nMean+std comparison plot saved to {output_path}")

    # --- Std-only comparison plot ---
    fig_std, axes_std = plt.subplots(len(metrics), 1, figsize=(14, 4 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes_std = [axes_std]

    for ax, (col, title, _) in zip(axes_std, metrics):
        for label, mean_df, std_df, n, color in run_data:
            ax.plot(std_df['date'], std_df[col],
                    label=f"{label} std", color=color, linewidth=1.8, alpha=0.95)
        ax.set_title(f"{title} Std Dev", fontsize=12, fontweight='bold', loc='left')
        ax.set_ylabel(f"std({col})", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='lower right', fontsize=9)

        for idx, (label, mean_df, std_df, n, color) in enumerate(run_data):
            last = std_df.iloc[-1]
            ax.annotate(f"{last[col]:.4f}", xy=(last['date'], last[col]),
                        fontsize=8, color=color, fontweight='bold',
                        xytext=(5, 0), textcoords='offset points', va='center')

    axes_std[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(axes_std[-1].get_xticklabels(), rotation=45, ha='right')
    axes_std[-1].set_xlabel('Date', fontsize=11)
    plt.tight_layout()
    std_output_path = os.path.join(args.output_dir, 'mem_type_std_comparison.png')
    plt.savefig(std_output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Std-only comparison plot saved to {std_output_path}")


if __name__ == "__main__":
    main()
