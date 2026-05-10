import argparse
import csv
import glob
import json
import os
import sys

import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib is not installed. Please install it using 'pip install matplotlib'.")
    sys.exit(1)


EFFORTS = ["none", "low", "medium", "high", "xhigh"]
TOOL_ITEM_TYPES = {"mcp_tool_call", "command_execution"}
BLUE_SCALE = {
    "none": "#9ecae1",
    "low": "#6baed6",
    "medium": "#3182bd",
    "high": "#08519c",
    "xhigh": "#08306b",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot accuracy and Brier over time across reasoning-effort runs."
    )
    parser.add_argument("--none_dir", required=True, help="Run dir for reasoning_effort=none")
    parser.add_argument("--low_dir", required=True, help="Run dir for reasoning_effort=low")
    parser.add_argument("--medium_dir", required=True, help="Run dir for reasoning_effort=medium")
    parser.add_argument("--high_dir", required=True, help="Run dir for reasoning_effort=high")
    parser.add_argument("--xhigh_dir", required=True, help="Run dir for reasoning_effort=xhigh")
    parser.add_argument("--output_dir", required=True, help="Directory to save the plot")
    parser.add_argument("--csv_name", default="daily_metrics.csv", help="Metrics CSV name to load")
    parser.add_argument(
        "--accuracy_only",
        action="store_true",
        help="Write a square paper-style accuracy-only plot instead of the default two-panel plot.",
    )
    parser.add_argument(
        "--brier_only",
        action="store_true",
        help="Write a square paper-style Brier-only plot instead of the default two-panel plot.",
    )
    parser.add_argument(
        "--include_tool_calls",
        action="store_true",
        help="Add Codex tool-call counts to the legend and write a tool-call-counts CSV.",
    )
    parser.add_argument("--start_date", default=None, help="Optional inclusive plot start date, e.g. 2026-01-01")
    parser.add_argument("--end_date", default=None, help="Optional inclusive plot end date, e.g. 2026-03-28")
    parser.add_argument("--output_name", default=None, help="Output filename stem or .png path.")
    parser.add_argument(
        "--title",
        default="GPT-5.5 Reasoning Effort Comparison",
        help="Optional figure title",
    )
    return parser.parse_args()


def apply_paper_style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.28,
            "lines.linewidth": 2.0,
        }
    )


def latest_metrics_csv(run_dir, csv_name):
    direct_csv = os.path.join(run_dir, csv_name)
    if os.path.isfile(direct_csv):
        return direct_csv

    timestamped_csvs = glob.glob(os.path.join(run_dir, "*", csv_name))
    if not timestamped_csvs:
        print(f"Error: no {csv_name} found in {run_dir} or one timestamped child.")
        sys.exit(1)

    timestamped_csvs.sort(key=os.path.getmtime)
    return timestamped_csvs[-1]


def load_effort_csv(effort, run_dir, csv_name):
    csv_path = latest_metrics_csv(run_dir, csv_name)
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"Error: {csv_path} is empty.")
        sys.exit(1)

    required_cols = {"date", "avg_brier", "accuracy"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Error: {csv_path} is missing columns: {', '.join(sorted(missing))}")
        sys.exit(1)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["effort"] = effort
    df["source_csv"] = csv_path
    return df.sort_values("date")


def latest_codex_stdout(run_dir):
    direct = glob.glob(os.path.join(run_dir, "agents", "*", "codex_stdout.jsonl"))
    if direct:
        direct.sort(key=os.path.getmtime)
        return direct[-1]

    timestamped = glob.glob(os.path.join(run_dir, "*", "agents", "*", "codex_stdout.jsonl"))
    if timestamped:
        timestamped.sort(key=os.path.getmtime)
        return timestamped[-1]
    return None


def count_codex_tool_calls(run_dir):
    stdout_path = latest_codex_stdout(run_dir)
    counts = {
        "tool_calls": 0,
        "mcp_tool_call": 0,
        "command_execution": 0,
        "search_news": 0,
        "submit_forecasts": 0,
        "next_day": 0,
        "codex_stdout": stdout_path or "",
    }
    if not stdout_path:
        return counts

    with open(stdout_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") != "item.started":
                continue
            item = rec.get("item") or {}
            item_type = item.get("type")
            if item_type not in TOOL_ITEM_TYPES:
                continue
            counts["tool_calls"] += 1
            counts[item_type] += 1
            if item_type == "mcp_tool_call":
                tool = item.get("tool")
                if tool in {"search_news", "submit_forecasts", "next_day"}:
                    counts[tool] += 1
    return counts


def filter_date_range(dfs_by_effort, start_date, end_date):
    if start_date is None and end_date is None:
        return dfs_by_effort

    start = pd.Timestamp(start_date) if start_date else None
    end = pd.Timestamp(end_date) if end_date else None
    filtered = {}
    for effort, df in dfs_by_effort.items():
        part = df.copy()
        if start is not None:
            part = part[part["date"] >= start]
        if end is not None:
            part = part[part["date"] <= end]
        if part.empty:
            print(f"Error: no rows left for {effort} after date filtering.")
            sys.exit(1)
        filtered[effort] = part
    return filtered


def output_paths(output_dir, output_name, default_stem):
    if output_name:
        name = output_name
        if name.endswith(".png"):
            stem = name[:-4]
        else:
            stem = name
    else:
        stem = default_stem
    png_path = os.path.join(output_dir, f"{stem}.png")
    pdf_path = os.path.join(output_dir, f"{stem}.pdf")
    return png_path, pdf_path


def plot_metric(ax, dfs_by_effort, metric_col, ylabel):
    for effort in EFFORTS:
        df = dfs_by_effort[effort]
        ax.plot(
            df["date"],
            df[metric_col],
            color=BLUE_SCALE[effort],
            linewidth=2.1,
            marker="o",
            markersize=2.6,
            label=effort,
            alpha=0.95,
        )

        last = df.iloc[-1]
        ax.annotate(
            f"{last[metric_col]:.3f}",
            xy=(last["date"], last[metric_col]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=BLUE_SCALE[effort],
            fontweight="bold",
        )

    ax.set_ylabel(ylabel, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_single_metric_paper(
    dfs_by_effort,
    output_dir,
    output_name,
    metric_col,
    ylabel,
    default_stem,
    plot_name,
    figsize=(5.2, 5.2),
    tool_counts=None,
):
    apply_paper_style()
    fig, ax = plt.subplots(figsize=figsize)

    for effort in EFFORTS:
        df = dfs_by_effort[effort]
        label = effort
        if tool_counts:
            label = f"{effort} / {tool_counts[effort]['tool_calls']:,}"
        ax.plot(
            df["date"],
            df[metric_col],
            color=BLUE_SCALE[effort],
            linewidth=2.9,
            label=label,
            alpha=0.98,
        )

    ax.set_ylabel(ylabel, fontsize=17, labelpad=14)
    ax.set_xlabel("")
    ax.tick_params(axis="both", labelsize=14)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    ymin = min(float(df[metric_col].min()) for df in dfs_by_effort.values())
    ymax = max(float(df[metric_col].max()) for df in dfs_by_effort.values())
    yspan = ymax - ymin
    if yspan > 0:
        ax.set_ylim(ymin - 0.04 * yspan, ymax + 0.30 * yspan)

    start = min(df["date"].min() for df in dfs_by_effort.values())
    end = max(df["date"].max() for df in dfs_by_effort.values())
    tick_dates = [
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-02-01"),
        pd.Timestamp("2026-03-01"),
        pd.Timestamp("2026-03-28"),
    ]
    tick_dates = [date for date in tick_dates if start <= date <= end]
    ax.set_xlim(start, end)
    ax.set_xticks(tick_dates)
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment("right")

    ax.legend(
        title="Reasoning Effort / Tool calls",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.005),
        ncol=2,
        fontsize=12.0,
        title_fontsize=12.4,
        handlelength=2.4,
        columnspacing=1.15,
        borderaxespad=0.0,
        labelspacing=0.42,
    )

    fig.tight_layout()
    png_path, pdf_path = output_paths(output_dir, output_name, default_stem)
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"{plot_name} plot saved to {png_path}")
    print(f"{plot_name} plot PDF saved to {pdf_path}")


def plot_accuracy_only_paper(dfs_by_effort, output_dir, output_name, tool_counts=None):
    plot_single_metric_paper(
        dfs_by_effort,
        output_dir,
        output_name,
        "accuracy",
        "Accuracy (%)",
        "reasoning_effort_accuracy_paper",
        "Accuracy",
        figsize=(7.8, 4.6),
        tool_counts=tool_counts,
    )


def plot_brier_only_paper(dfs_by_effort, output_dir, output_name, tool_counts=None):
    plot_single_metric_paper(
        dfs_by_effort,
        output_dir,
        output_name,
        "avg_brier",
        "Brier Skill",
        "reasoning_effort_brier_paper",
        "Brier",
        tool_counts=tool_counts,
    )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    run_dirs = {
        "none": args.none_dir,
        "low": args.low_dir,
        "medium": args.medium_dir,
        "high": args.high_dir,
        "xhigh": args.xhigh_dir,
    }
    dfs_by_effort = {
        effort: load_effort_csv(effort, run_dirs[effort], args.csv_name)
        for effort in EFFORTS
    }
    dfs_by_effort = filter_date_range(dfs_by_effort, args.start_date, args.end_date)

    tool_counts = None
    if args.include_tool_calls:
        tool_counts = {effort: count_codex_tool_calls(run_dirs[effort]) for effort in EFFORTS}
        tool_csv = os.path.join(args.output_dir, "reasoning_effort_tool_call_counts.csv")
        with open(tool_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "effort",
                    "tool_calls",
                    "mcp_tool_call",
                    "command_execution",
                    "search_news",
                    "submit_forecasts",
                    "next_day",
                    "codex_stdout",
                ],
            )
            writer.writeheader()
            for effort in EFFORTS:
                row = {"effort": effort}
                row.update(tool_counts[effort])
                writer.writerow(row)
        print(f"Tool-call counts saved to {tool_csv}")

    print("Loaded runs:")
    for effort in EFFORTS:
        df = dfs_by_effort[effort]
        print(
            f"  {effort:>6}: {len(df)} days "
            f"({df['date'].min().date()} to {df['date'].max().date()}) "
            f"from {df['source_csv'].iloc[0]}"
        )

    if args.accuracy_only:
        plot_accuracy_only_paper(dfs_by_effort, args.output_dir, args.output_name, tool_counts=tool_counts)
        return
    if args.brier_only:
        plot_brier_only_paper(dfs_by_effort, args.output_dir, args.output_name, tool_counts=tool_counts)
        return

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig, axes = plt.subplots(2, 1, figsize=(14, 8.5), sharex=True)
    plot_metric(axes[0], dfs_by_effort, "accuracy", "Accuracy (%)")
    axes[0].set_title("Accuracy Over Time", loc="left", fontweight="bold")
    axes[0].legend(title="Reasoning Effort", loc="upper left", frameon=False)

    plot_metric(axes[1], dfs_by_effort, "avg_brier", "Avg Brier Skill")
    axes[1].set_title("Brier Over Time", loc="left", fontweight="bold")
    axes[1].set_xlabel("Date", fontweight="bold")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    axes[1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")

    if args.title:
        fig.suptitle(args.title, fontsize=16, fontweight="bold", y=0.995)

    fig.tight_layout()
    output_path = os.path.join(args.output_dir, "reasoning_effort_acc_brier.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    main()
