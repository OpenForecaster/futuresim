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
    parser.add_argument("--mem_parent", type=str, required=True, help="Parent directory containing memory run subdirectories")
    parser.add_argument("--nomem_parent", type=str, required=True, help="Parent directory containing no-memory run subdirectories")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save plots")
    parser.add_argument("--mem_label", type=str, default="With Memory", help="Label for memory runs")
    parser.add_argument("--nomem_label", type=str, default="Without Memory", help="Label for no-memory runs")
    return parser.parse_args()


def load_all_runs(parent_dir):
    """Load daily_metrics.csv from all subdirectories under parent_dir."""
    dfs = []
    subdirs = sorted(glob.glob(os.path.join(parent_dir, "*")))
    for subdir in subdirs:
        csv_path = os.path.join(subdir, "daily_metrics.csv")
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)
            df['date'] = pd.to_datetime(df['date'])
            df['run'] = os.path.basename(subdir)
            dfs.append(df)
    if not dfs:
        print(f"Error: No daily_metrics.csv found under {parent_dir}")
        sys.exit(1)
    print(f"  Loaded {len(dfs)} runs from {parent_dir}")
    for df in dfs:
        print(f"    {df['run'].iloc[0]}: {len(df)} days ({df['date'].min().date()} to {df['date'].max().date()})")
    return dfs


def aggregate_runs(dfs, metrics_cols):
    """Compute per-date mean and std across runs."""
    combined = pd.concat(dfs, ignore_index=True)
    grouped = combined.groupby('date')[metrics_cols]
    mean_df = grouped.mean().reset_index()
    std_df = grouped.std().reset_index()
    count_df = grouped.size().reset_index(name='n_runs')
    mean_df = mean_df.sort_values('date')
    std_df = std_df.sort_values('date')
    count_df = count_df.sort_values('date')
    return mean_df, std_df, count_df


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading memory runs...")
    mem_dfs = load_all_runs(args.mem_parent)
    print("Loading no-memory runs...")
    nomem_dfs = load_all_runs(args.nomem_parent)

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
    print(f"\nComparison plot saved to {output_path}")


if __name__ == "__main__":
    main()
