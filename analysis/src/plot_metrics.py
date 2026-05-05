import pandas as pd
import argparse
import os
import sys

# Check for matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except ImportError:
    print("Error: matplotlib is not installed. Please install it using 'pip install matplotlib'.")
    sys.exit(1)

import plot_config  # noqa: F401  (applies project-wide science+serif style)

def parse_args():
    parser = argparse.ArgumentParser(description="Plot daily metrics from simulation.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to daily_metrics.csv")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save plots")
    return parser.parse_args()

def plot_metric(df, metric_col, title, output_path):
    # Setup the figure and subplots (2 rows, 1 column, height ratio 3:1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # Get unique agents
    agents = df['agent_id'].unique()
    
    # Colors
    cmap = plt.get_cmap('tab10')
    
    for i, agent in enumerate(agents):
        agent_data = df[df['agent_id'] == agent]
        color = cmap(i % 10)
        
        # Plot metric
        ax1.plot(agent_data['date'], agent_data[metric_col], marker='o', label=agent, color=color)
        
        # Plot predictions
        ax2.plot(agent_data['date'], agent_data['total_predictions'], marker='x', linestyle='--', label=agent, color=color)

    ax1.set_title(title)
    ax1.set_ylabel(metric_col.replace('_', ' ').title())
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend to top plot
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)

    ax2.set_ylabel('Total Predictions')
    ax2.set_xlabel('Date')
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Format Date
    # ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1)) # Force every day if needed
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    # Manually rotate
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    args = parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    df = pd.read_csv(args.input_csv)
    
    # Ensure date is sorted and datetime object
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['date', 'agent_id'])
    
    # 1. Avg Brier
    plot_metric(df, 'avg_brier', 'Average Brier Score over Time', os.path.join(args.output_dir, 'avg_brier.png'))
    
    # 2. TW Score (column name differs between single- and multi-agent runs)
    tw_col = 'tw_score' if 'tw_score' in df.columns else 'tw_peer_score'
    plot_metric(df, tw_col, 'Time-Weighted Score over Time', os.path.join(args.output_dir, f'{tw_col}.png'))
    
    # 3. Accuracy
    plot_metric(df, 'accuracy', 'Accuracy over Time', os.path.join(args.output_dir, 'accuracy.png'))

    print(f"Plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()
