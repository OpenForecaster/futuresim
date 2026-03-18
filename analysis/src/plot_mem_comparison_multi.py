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
    parser = argparse.ArgumentParser(description="Compare memory vs no-memory runs (multi-run avg + stddev).")
    parser.add_argument("--mem_parent", type=str, default=None, help="Parent directory containing memory run subdirectories")
    parser.add_argument("--nomem_parent", type=str, default=None, help="Parent directory containing no-memory run subdirectories")
    parser.add_argument("--mem_child_glob", type=str, default="*",
                        help="Glob (under --mem_parent) selecting memory run dirs (e.g., '*_med_r0[0-2]_restart').")
    parser.add_argument("--nomem_child_glob", type=str, default="*",
                        help="Glob (under --nomem_parent) selecting no-memory run dirs.")
    parser.add_argument("--mem_dirs", type=str, nargs="+", default=None,
                        help="Explicit list of timestamped memory-run directories containing daily_metrics.csv")
    parser.add_argument("--nomem_dirs", type=str, nargs="+", default=None,
                        help="Explicit list of timestamped no-memory-run directories containing daily_metrics.csv")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save plots")
    parser.add_argument("--mem_label", type=str, default="With Memory", help="Label for memory runs")
    parser.add_argument("--nomem_label", type=str, default="Without Memory", help="Label for no-memory runs")
    parser.add_argument("--expected_runs", type=int, default=None,
                        help="Optional expected number of runs per side (e.g., 3 for r00/r01/r02).")
    return parser.parse_args()


def _latest_csv_in_dir(run_dir):
    """Return the most recent daily_metrics.csv in run_dir or None."""
    direct_csv = os.path.join(run_dir, "daily_metrics.csv")
    if os.path.isfile(direct_csv):
        return direct_csv

    # Typical layout:
    # <run_dir>/<timestamp>/daily_metrics.csv
    ts_csvs = glob.glob(os.path.join(run_dir, "*", "daily_metrics.csv"))
    if not ts_csvs:
        return None
    ts_csvs.sort(key=lambda p: os.path.getmtime(p))
    return ts_csvs[-1]


def _discover_explicit_run_csvs(run_dirs):
    """Resolve explicitly provided timestamped run directories to daily_metrics.csv files."""
    discovered = []
    for run_dir in run_dirs:
        run_dir = os.path.abspath(run_dir)
        if not os.path.isdir(run_dir):
            print(f"Error: {run_dir} is not a directory.")
            sys.exit(1)
        csv_path = os.path.join(run_dir, "daily_metrics.csv")
        if not os.path.isfile(csv_path):
            print(f"Error: {run_dir} does not contain daily_metrics.csv")
            sys.exit(1)
        parent = os.path.basename(os.path.dirname(run_dir))
        run_name = f"{parent}/{os.path.basename(run_dir)}"
        discovered.append((run_name, csv_path))
    return discovered


def _discover_run_csvs(parent_dir, child_glob="*"):
    """
    Discover one daily_metrics.csv per run under parent_dir.

    If parent_dir itself is a run directory, use it directly.
    Otherwise, for each immediate child (e.g., r00/r01/r02), select the latest
    timestamped daily_metrics.csv.
    """
    parent_dir = os.path.abspath(parent_dir)
    if not os.path.isdir(parent_dir):
        print(f"Error: {parent_dir} is not a directory.")
        sys.exit(1)

    discovered = []
    children = sorted([p for p in glob.glob(os.path.join(parent_dir, child_glob)) if os.path.isdir(p)])

    # Parent may itself be a run dir.
    parent_csv = _latest_csv_in_dir(parent_dir)
    if parent_csv is not None and not children:
        return [(os.path.basename(parent_dir), parent_csv)]

    for child in children:
        csv_path = _latest_csv_in_dir(child)
        if csv_path is None:
            continue
        discovered.append((os.path.basename(child), csv_path))

    # Fallback: if no child run dirs found but parent has direct/timestamped csv.
    if not discovered and parent_csv is not None:
        discovered.append((os.path.basename(parent_dir), parent_csv))

    return discovered


def load_all_runs(parent_dir=None, expected_runs=None, child_glob="*", run_dirs=None):
    """Load per-run daily_metrics.csv from either discovered run dirs or explicit timestamp dirs."""
    dfs = []
    if run_dirs is not None:
        discovered = _discover_explicit_run_csvs(run_dirs)
        source_label = "explicit run directories"
    else:
        discovered = _discover_run_csvs(parent_dir, child_glob=child_glob)
        source_label = parent_dir
    for run_name, csv_path in discovered:
        try:
            df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            print(f"  Warning: skipping unreadable empty daily_metrics.csv for {run_name}: {csv_path}")
            continue
        if df.empty:
            print(f"  Warning: skipping empty daily_metrics.csv for {run_name}: {csv_path}")
            continue
        df['date'] = pd.to_datetime(df['date'])
        df['run'] = run_name
        dfs.append(df)

    if not dfs:
        print(f"Error: No usable daily_metrics.csv found under {source_label}")
        sys.exit(1)

    print(f"  Loaded {len(dfs)} runs from {source_label}")
    for df in dfs:
        print(f"    {df['run'].iloc[0]}: {len(df)} days ({df['date'].min().date()} to {df['date'].max().date()})")
    if expected_runs is not None and len(dfs) != expected_runs:
        print(f"  Warning: expected {expected_runs} runs, found {len(dfs)} in {source_label}")

    return dfs


def aggregate_runs(dfs, metrics_cols):
    """Compute per-date mean and std across runs."""
    combined = pd.concat(dfs, ignore_index=True)
    grouped = combined.groupby('date')[metrics_cols]
    mean_df = grouped.mean().reset_index()
    # Use sample std; fill NaN (single run on a date) with 0.
    std_df = grouped.std().fillna(0.0).reset_index()
    count_df = grouped.size().reset_index(name='n_runs')
    mean_df = mean_df.sort_values('date')
    std_df = std_df.sort_values('date')
    count_df = count_df.sort_values('date')
    return mean_df, std_df, count_df


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    using_explicit = args.mem_dirs is not None or args.nomem_dirs is not None
    if using_explicit:
        if args.mem_dirs is None or args.nomem_dirs is None:
            print("Error: --mem_dirs and --nomem_dirs must be provided together.")
            sys.exit(1)
    else:
        if args.mem_parent is None or args.nomem_parent is None:
            print("Error: either provide both --mem_parent/--nomem_parent or both --mem_dirs/--nomem_dirs.")
            sys.exit(1)

    print("Loading memory runs...")
    mem_dfs = load_all_runs(
        args.mem_parent,
        expected_runs=args.expected_runs,
        child_glob=args.mem_child_glob,
        run_dirs=args.mem_dirs,
    )
    print("Loading no-memory runs...")
    nomem_dfs = load_all_runs(
        args.nomem_parent,
        expected_runs=args.expected_runs,
        child_glob=args.nomem_child_glob,
        run_dirs=args.nomem_dirs,
    )

    metrics = [
        ('avg_brier', 'Average Brier Skill Score', 'higher is better'),
        # ('tw_peer_score', 'Time-Weighted Peer Score', 'higher is better'),
        ('accuracy', 'Accuracy (%)', 'higher is better'),
        ('exp_acc', 'Avg Probability of Correct Outcome', 'higher is better'),
        # ('total_predictions', 'Total Predictions', ''),
    ]

    metric_cols = [m[0] for m in metrics]

    mem_mean, mem_std, mem_count = aggregate_runs(mem_dfs, metric_cols)
    nomem_mean, nomem_std, nomem_count = aggregate_runs(nomem_dfs, metric_cols)

    # Align on common date range
    common_start = max(mem_mean['date'].min(), nomem_mean['date'].min())
    common_end = min(mem_mean['date'].max(), nomem_mean['date'].max())
    mem_mean = mem_mean[(mem_mean['date'] >= common_start) & (mem_mean['date'] <= common_end)]
    mem_std = mem_std[(mem_std['date'] >= common_start) & (mem_std['date'] <= common_end)]
    mem_count = mem_count[(mem_count['date'] >= common_start) & (mem_count['date'] <= common_end)]
    nomem_mean = nomem_mean[(nomem_mean['date'] >= common_start) & (nomem_mean['date'] <= common_end)]
    nomem_std = nomem_std[(nomem_std['date'] >= common_start) & (nomem_std['date'] <= common_end)]
    nomem_count = nomem_count[(nomem_count['date'] >= common_start) & (nomem_count['date'] <= common_end)]

    n_mem = mem_count['n_runs'].iloc[-1] if len(mem_count) > 0 else 0
    n_nomem = nomem_count['n_runs'].iloc[-1] if len(nomem_count) > 0 else 0

    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 4 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    colors = {'mem': '#2196F3', 'nomem': '#FF5722'}

    for ax, (col, title, direction) in zip(axes, metrics):
        # Plot mean lines
        ax.plot(mem_mean['date'], mem_mean[col],
                label=f"{args.mem_label} (n={n_mem})", color=colors['mem'], linewidth=1.5, alpha=0.9)
        ax.plot(nomem_mean['date'], nomem_mean[col],
                label=f"{args.nomem_label} (n={n_nomem})", color=colors['nomem'], linewidth=1.5, alpha=0.9)

        # Plot std dev bands
        ax.fill_between(mem_mean['date'],
                        mem_mean[col] - mem_std[col],
                        mem_mean[col] + mem_std[col],
                        color=colors['mem'], alpha=0.15)
        ax.fill_between(nomem_mean['date'],
                        nomem_mean[col] - nomem_std[col],
                        nomem_mean[col] + nomem_std[col],
                        color=colors['nomem'], alpha=0.15)

        subtitle = f"  ({direction})" if direction else ""
        ax.set_title(f"{title}{subtitle}", fontsize=12, fontweight='bold', loc='left')
        ax.set_ylabel(col.replace('_', ' ').title(), fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='upper right', fontsize=9)

        # Final-value annotations
        for mean_df, std_df, label, color in [
            (mem_mean, mem_std, args.mem_label, colors['mem']),
            (nomem_mean, nomem_std, args.nomem_label, colors['nomem']),
        ]:
            last_mean = mean_df.iloc[-1]
            last_std = std_df.iloc[-1]
            val = last_mean[col]
            sd = last_std[col]
            if pd.notna(sd) and sd > 0:
                text = f"{val:.4f} ± {sd:.4f}"
            else:
                text = f"{val:.4f}"
            ax.annotate(text, xy=(last_mean['date'], val),
                        fontsize=8, color=color, fontweight='bold',
                        xytext=(5, 0), textcoords='offset points', va='center')

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(axes[-1].get_xticklabels(), rotation=45, ha='right')
    axes[-1].set_xlabel('Date', fontsize=11)

    # fig.suptitle('Memory vs No-Memory Run Comparison (Multi-Run Average)',
    #              fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    output_path = os.path.join(args.output_dir, 'mem_vs_nomem_multi_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nMean+std comparison plot saved to {output_path}")

    # Also save std-only comparison figure.
    fig_std, axes_std = plt.subplots(len(metrics), 1, figsize=(14, 4 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes_std = [axes_std]

    for ax, (col, title, _) in zip(axes_std, metrics):
        ax.plot(mem_std['date'], mem_std[col],
                label=f"{args.mem_label} std", color=colors['mem'], linewidth=1.8, alpha=0.95)
        ax.plot(nomem_std['date'], nomem_std[col],
                label=f"{args.nomem_label} std", color=colors['nomem'], linewidth=1.8, alpha=0.95)
        ax.set_title(f"{title} Std Dev", fontsize=12, fontweight='bold', loc='left')
        ax.set_ylabel(f"std({col})", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='upper right', fontsize=9)

        # Final std annotations.
        for std_df, color in [(mem_std, colors['mem']), (nomem_std, colors['nomem'])]:
            last = std_df.iloc[-1]
            ax.annotate(f"{last[col]:.4f}", xy=(last['date'], last[col]),
                        fontsize=8, color=color, fontweight='bold',
                        xytext=(5, 0), textcoords='offset points', va='center')

    axes_std[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(axes_std[-1].get_xticklabels(), rotation=45, ha='right')
    axes_std[-1].set_xlabel('Date', fontsize=11)
    plt.tight_layout()
    std_output_path = os.path.join(args.output_dir, 'mem_vs_nomem_std_comparison.png')
    plt.savefig(std_output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Std-only comparison plot saved to {std_output_path}")


if __name__ == "__main__":
    main()
