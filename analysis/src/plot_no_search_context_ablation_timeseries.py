#!/usr/bin/env python3
"""Time-step plots for GPT-5.5 search-context ablations."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_NORMAL = Path("/fast/sgoel/forecasting/current_sim/neurips_runs/main_fig/gpt-5.5/26-05-04-02-00-26")
DEFAULT_NO_SEARCH_UPDATE = Path(
    "/fast/sgoel/forecasting/current_sim/final_runs_v37/"
    "codex_aljazeeraQ12026v37_gpt55_resume_no_search_update_xhigh_r00/26-05-06-17-06-40"
)
DEFAULT_WARMUP = Path(
    "/fast/sgoel/forecasting/current_sim/final_runs_v37/"
    "gpt55_xhigh_rg1_warmup_r00/26-05-05-01-22-10"
)
DEFAULT_STATIC = Path(
    "/fast/sgoel/forecasting/current_sim/final_runs_v37/"
    "codex_aljazeeraQ12026v37_gpt55_rg1_static_search_xhigh_r00/26-05-05-14-17-38"
)
DEFAULT_OUT_DIR = Path("analysis/plots/no_search_context_ablation_timeseries")


SETTINGS = [
    {
        "label": "Daily context updates",
        "color": "#08519c",
        "run_attr": "normal_dir",
        "reference": False,
    },
    {
        "label": "No context updates",
        "color": "#D55E00",
        "run_attr": "no_search_update_dir",
        "reference": False,
    },
    {
        "label": "Agentic search",
        "color": "#009E73",
        "run_attr": "warmup_dir",
        "reference": True,
    },
    {
        "label": "Single search query",
        "color": "#D62728",
        "run_attr": "static_search_dir",
        "reference": True,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal_dir", type=Path, default=DEFAULT_NORMAL)
    parser.add_argument("--no_search_update_dir", type=Path, default=DEFAULT_NO_SEARCH_UPDATE)
    parser.add_argument("--warmup_dir", type=Path, default=DEFAULT_WARMUP)
    parser.add_argument("--static_search_dir", type=Path, default=DEFAULT_STATIC)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "font.size": 16,
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "cm",
            "axes.labelsize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "lines.linewidth": 4.2,
        }
    )


def load_metrics(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "daily_metrics.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


def plot_metric(
    dfs: dict[str, pd.DataFrame],
    metric: str,
    ylabel: str,
    output_base: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.set_box_aspect(1)
    all_y = []

    for setting in SETTINGS:
        df = dfs[setting["label"]]
        if setting["reference"]:
            y = float(df.iloc[-1][metric])
            ax.plot(
                [df.iloc[-1]["date"]],
                [y],
                color=setting["color"],
                linestyle="None",
                marker="x",
                markersize=20,
                markeredgewidth=4.0,
                label=setting["label"],
                zorder=5,
            )
            all_y.append(y)
            continue

        all_y.extend(df[metric].tolist())
        ax.plot(
            df["date"],
            df[metric],
            color=setting["color"],
            linewidth=4.2,
            alpha=0.98,
            label=setting["label"],
        )

    ax.set_xlabel("")
    ax.set_ylabel(ylabel, labelpad=12)
    ax.tick_params(axis="both", labelsize=14)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    tick_dates = [
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-02-01"),
        pd.Timestamp("2026-03-01"),
        pd.Timestamp("2026-04-01"),
    ]
    start = min(df["date"].min() for df in dfs.values())
    end = pd.Timestamp("2026-04-01")
    ax.set_xlim(start, end)
    ax.set_xticks([date for date in tick_dates if start <= date <= end])
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment("right")
    ymin, ymax = min(all_y), max(all_y)
    yspan = ymax - ymin
    if metric == "accuracy":
        lower = 10.0
        upper = ymax + 0.28 * (ymax - lower)
        ax.set_ylim(lower, upper)
        ax.set_yticks([tick for tick in range(15, int(upper) + 1, 5)])
        for y0 in (0.018, 0.045):
            ax.plot(
                (-0.012, 0.012),
                (y0 - 0.012, y0 + 0.012),
                transform=ax.transAxes,
                color="black",
                linewidth=1.8,
                clip_on=False,
            )
    else:
        ax.set_ylim(ymin - 0.06 * yspan, ymax + 0.70 * yspan)
        ax.set_yticks([-0.20, -0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="upper left",
        fontsize=14.0,
        ncol=1,
        handlelength=1.5,
        handletextpad=0.4,
        borderaxespad=0.35,
        labelspacing=0.55,
        markerscale=0.85,
    )

    ax.margins(x=0.02)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.98, bottom=0.18)
    fig.savefig(output_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    dfs = {}
    for setting in SETTINGS:
        dfs[setting["label"]] = load_metrics(getattr(args, setting["run_attr"]))

    plot_metric(
        dfs,
        "accuracy",
        "Accuracy (%)",
        args.output_dir / "accuracy_timeseries",
    )
    plot_metric(
        dfs,
        "avg_brier",
        "Brier score",
        args.output_dir / "brier_timeseries",
    )

    print(args.output_dir / "accuracy_timeseries.png")
    print(args.output_dir / "accuracy_timeseries.pdf")
    print(args.output_dir / "brier_timeseries.png")
    print(args.output_dir / "brier_timeseries.pdf")


if __name__ == "__main__":
    main()
