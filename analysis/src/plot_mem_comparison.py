import pandas as pd
import argparse
import os
import sys

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except ImportError:
    print("Error: matplotlib is not installed. Please install it using 'pip install matplotlib'.")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare memory vs no-memory simulation runs.")
    parser.add_argument("--mem_dir", type=str, required=True, help="Path to memory run directory (containing daily_metrics.csv)")
    parser.add_argument("--nomem_dir", type=str, required=True, help="Path to no-memory run directory (containing daily_metrics.csv)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save plots")
    parser.add_argument("--mem_label", type=str, default="With Memory", help="Label for memory run")
    parser.add_argument("--nomem_label", type=str, default="Without Memory", help="Label for no-memory run")
    return parser.parse_args()


def load_metrics(directory):
    csv_path = os.path.join(directory, "daily_metrics.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')
    return df


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df_mem = load_metrics(args.mem_dir)
    df_nomem = load_metrics(args.nomem_dir)

    # Align on common date range
    common_start = max(df_mem['date'].min(), df_nomem['date'].min())
    common_end = min(df_mem['date'].max(), df_nomem['date'].max())
    df_mem = df_mem[(df_mem['date'] >= common_start) & (df_mem['date'] <= common_end)]
    df_nomem = df_nomem[(df_nomem['date'] >= common_start) & (df_nomem['date'] <= common_end)]

    metrics = [
        ('avg_brier', 'Average Brier Skill Score', 'higher is better'),
        # ('tw_peer_score', 'Time-Weighted Peer Score', 'higher is better'),
        ('accuracy', 'Accuracy (%)', 'higher is better'),
        ('exp_acc', 'Expected Accuracy', 'higher is better'),
        # ('total_predictions', 'Total Predictions', ''),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 4 * len(metrics)), sharex=True)

    colors = {'mem': '#2196F3', 'nomem': '#FF5722'}

    for ax, (col, title, direction) in zip(axes, metrics):
        ax.plot(df_mem['date'], df_mem[col], label=args.mem_label, color=colors['mem'], linewidth=1.5, alpha=0.9)
        ax.plot(df_nomem['date'], df_nomem[col], label=args.nomem_label, color=colors['nomem'], linewidth=1.5, alpha=0.9)

        subtitle = f"  ({direction})" if direction else ""
        ax.set_title(f"{title}{subtitle}", fontsize=12, fontweight='bold', loc='left')
        ax.set_ylabel(col.replace('_', ' ').title(), fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='upper right', fontsize=9)

        # Add final-value annotations
        for df, label, color in [(df_mem, args.mem_label, colors['mem']),
                                  (df_nomem, args.nomem_label, colors['nomem'])]:
            last = df.iloc[-1]
            ax.annotate(f"{last[col]:.4f}" if isinstance(last[col], float) else f"{last[col]}",
                        xy=(last['date'], last[col]),
                        fontsize=8, color=color, fontweight='bold',
                        xytext=(5, 0), textcoords='offset points', va='center')

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(axes[-1].get_xticklabels(), rotation=45, ha='right')
    axes[-1].set_xlabel('Date', fontsize=11)

    # fig.suptitle('Memory vs No-Memory Run Comparison', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    output_path = os.path.join(args.output_dir, 'mem_vs_nomem_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {output_path}")


if __name__ == "__main__":
    main()
