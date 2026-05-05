import pandas as pd
import argparse
import os
import sys

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except ImportError:
    print("Error: matplotlib is not installed. Please install it using 'pip install matplotlib'.")
    sys.exit(1)

import plot_config  # noqa: F401  (applies project-wide science+serif style)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot all daily metrics in a single combined vertical figure.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to daily_metrics.csv")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save plot")
    parser.add_argument("--title", type=str, default="", help="Optional suptitle for the figure")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['date', 'agent_id'])

    tw_col = 'tw_score' if 'tw_score' in df.columns else 'tw_peer_score'
    tw_label = 'TW Score' if tw_col == 'tw_score' else 'TW Peer Score'
    metrics = [
        ('avg_brier', 'Avg Brier Score'),
        (tw_col, tw_label),
        ('accuracy', 'Accuracy (%)'),
    ]

    # Style
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
    })

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True,
                             gridspec_kw={'height_ratios': [3, 3, 3, 1.2],
                                          'hspace': 0.12})

    agents = df['agent_id'].unique()
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800',
              '#00BCD4', '#E91E63', '#8BC34A', '#673AB7', '#FFC107']

    for idx, (metric_col, ylabel) in enumerate(metrics):
        ax = axes[idx]
        for i, agent in enumerate(agents):
            agent_data = df[df['agent_id'] == agent]
            color = colors[i % len(colors)]
            x = agent_data['date'].dt.to_pydatetime()
            ax.plot(x, agent_data[metric_col].values, marker='o', markersize=3,
                    linewidth=1.8, color=color, alpha=0.9)
        ax.set_ylabel(ylabel, fontweight='bold', labelpad=15)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Bottom subplot: total predictions
    ax_pred = axes[3]
    for i, agent in enumerate(agents):
        agent_data = df[df['agent_id'] == agent]
        color = colors[i % len(colors)]
        x = agent_data['date'].dt.to_pydatetime()
        ax_pred.fill_between(x, agent_data['total_predictions'].values,
                             alpha=0.3, color=color)
        ax_pred.plot(x, agent_data['total_predictions'].values,
                     linewidth=1.5, color=color, alpha=0.9)
    ax_pred.set_ylabel('Total Preds', fontweight='bold', labelpad=15)
    ax_pred.set_xlabel('Date', fontweight='bold')
    ax_pred.grid(True, linestyle='--', alpha=0.3)
    ax_pred.spines['top'].set_visible(False)
    ax_pred.spines['right'].set_visible(False)
    ax_pred.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax_pred.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
    plt.setp(ax_pred.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    if args.title:
        fig.suptitle(args.title, fontsize=18, fontweight='bold', y=0.98)
        fig.subplots_adjust(top=0.96)

    # Align all y-labels
    fig.align_ylabels(axes)
    fig.savefig(os.path.join(args.output_dir, 'combined_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to {args.output_dir}/combined_metrics.png")


if __name__ == "__main__":
    main()
