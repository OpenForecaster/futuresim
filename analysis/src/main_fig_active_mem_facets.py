"""Faceted active-memory comparison plots.

Creates one row of model-specific subplots for the native-harness vs
our-harness comparison. Each subplot contains exactly two lines:

  solid  = native harness
  dashed = our harness

The run pairing and metric selection intentionally match
`main_fig_active_mem_plot.py`: for each model and metric, use the native
main-fig run closest to the median Day0 value, and the latest corresponding
our-harness run.
"""

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd

from main_fig_active_mem_plot import (
    DEFAULT_RUNS_DIR,
    GROUPS,
    _first_metric,
    collect_runs_for_group,
    pick_median_day0_pair,
)
from plot_config import color_for_label, style_axes  # noqa: F401

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "plots" / "main_fig_active_mem_facets"

METRIC_LABELS = {
    "accuracy": "Accuracy (%)",
    "avg_brier": "Brier Skill Score (↑)",
}

LINEWIDTH = 10.5
TITLE_FONTSIZE = 50
LABEL_FONTSIZE = 62
TICK_FONTSIZE = 44
LEGEND_FONTSIZE = 54
SPINE_WIDTH = 4.0
TICK_WIDTH = 4.0
TICK_LENGTH = 16
BASELINE_COLOR = "#9A9A9A"
BASELINE_LINESTYLE = (0, (42, 24))
BASELINE_LINEWIDTH = 2.0

FIG_WIDTH_PER_MODEL = 10.8
MIN_FIG_WIDTH = 22.0
FIG_HEIGHT = 14.2
TITLE_PAD = 22
Y_LABEL_PAD = 34
X_MARGIN = 0.01
X_TICK_DATES = ["2026-01-01", "2026-02-01", "2026-03-01"]

LEGEND_Y = 0.97
LEGEND_HANDLE_LENGTH = 3.5
LEGEND_COLUMN_SPACING = 1.8

SUBPLOT_LEFT = 0.09
SUBPLOT_RIGHT = 0.99
SUBPLOT_BOTTOM = 0.25
SUBPLOT_TOP = 0.76
SUBPLOT_WSPACE = 0.16

SAVE_BBOX_INCHES = "tight"
SAVE_PAD_INCHES = 0.18


def collect_paired_metric_runs(runs_dir: Path, metric: str) -> list[tuple[str, dict, dict]]:
    per_spec_runs = [(spec, collect_runs_for_group(runs_dir, spec)) for spec in GROUPS]

    have_native = {s.label for s, rs in per_spec_runs if rs and not s.is_active_mem}
    have_our = {s.label for s, rs in per_spec_runs if rs and s.is_active_mem}
    paired_labels = have_native & have_our

    dropped = (have_native | have_our) - paired_labels
    if dropped:
        print(f"Dropping unpaired models: {sorted(dropped)}")

    native_by_label: dict[str, list[dict]] = {}
    our_by_label: dict[str, list[dict]] = {}
    for spec, rs in per_spec_runs:
        if spec.label not in paired_labels:
            continue
        bucket = our_by_label if spec.is_active_mem else native_by_label
        bucket.setdefault(spec.label, []).extend(rs)

    pairs: list[tuple[str, dict, dict]] = []
    seen_labels: set[str] = set()
    for spec in GROUPS:
        label = spec.label
        if label in seen_labels or label not in paired_labels:
            continue
        seen_labels.add(label)
        native_run, our_run = pick_median_day0_pair(
            native_by_label.get(label, []),
            our_by_label.get(label, []),
            metric,
        )
        if native_run is None or our_run is None:
            print(f"  WARN: no Day0 {metric} pair for {label}")
            continue
        print(f"{label}:")
        print(
            f"  native      {native_run['run_dir']}/{native_run['ts_dir']} "
            f"day0_{metric}={_first_metric(native_run['df'], metric):.4f}"
        )
        print(
            f"  our         {our_run['run_dir']}/{our_run['ts_dir']} "
            f"day0_{metric}={_first_metric(our_run['df'], metric):.4f}"
        )
        pairs.append((label, native_run, our_run))
    return pairs


def prep_metric_df(run: dict, metric: str, start_date_filter: str | None) -> pd.DataFrame:
    df = run["df"].copy()
    df["date"] = pd.to_datetime(df["date"])
    if start_date_filter is not None:
        df = df[df["date"] >= pd.to_datetime(start_date_filter)]
    return df.groupby("date", as_index=False)[metric].mean().sort_values("date")


def plot_facets(
    pairs: list[tuple[str, dict, dict]],
    metric: str,
    out_path: Path,
    start_date_filter: str | None,
) -> None:
    if not pairs:
        raise SystemExit(f"No paired runs to plot for {metric}.")

    n_models = len(pairs)
    fig_width = max(FIG_WIDTH_PER_MODEL * n_models, MIN_FIG_WIDTH)
    fig, axes = plt.subplots(
        1,
        n_models,
        figsize=(fig_width, FIG_HEIGHT),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    for ax, (label, native_run, our_run) in zip(axes, pairs):
        style_axes(ax)
        color = color_for_label(label)
        native_df = prep_metric_df(native_run, metric, start_date_filter)
        our_df = prep_metric_df(our_run, metric, start_date_filter)
        if metric == "avg_brier":
            ax.axhline(
                0.0,
                color=BASELINE_COLOR,
                linestyle=BASELINE_LINESTYLE,
                linewidth=BASELINE_LINEWIDTH,
                zorder=1,
            )

        ax.plot(
            native_df["date"],
            native_df[metric],
            color=color,
            linestyle="-",
            linewidth=LINEWIDTH,
            zorder=2,
            solid_capstyle="round",
        )
        ax.plot(
            our_df["date"],
            our_df[metric],
            color=color,
            linestyle="--",
            linewidth=LINEWIDTH,
            zorder=2,
            dash_capstyle="round",
        )

        ax.set_title(label, color=color, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=TITLE_PAD)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.set_xticks(pd.to_datetime(X_TICK_DATES))
        ax.tick_params(
            axis="both",
            which="major",
            direction="out",
            length=TICK_LENGTH,
            width=TICK_WIDTH,
        )
        ax.tick_params(axis="y", right=False)
        ax.tick_params(axis="x", top=False)
        ax.minorticks_off()
        ax.margins(x=X_MARGIN)
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=TICK_FONTSIZE)
        plt.setp(ax.get_yticklabels(), fontsize=TICK_FONTSIZE)
        ax.spines["left"].set_linewidth(SPINE_WIDTH)
        ax.spines["bottom"].set_linewidth(SPINE_WIDTH)

    axes[0].set_ylabel(
        METRIC_LABELS.get(metric, metric),
        fontsize=LABEL_FONTSIZE,
        labelpad=Y_LABEL_PAD,
    )

    handles = [
        mlines.Line2D([], [], color="#333333", linestyle="-", linewidth=LINEWIDTH, label="native harness"),
        mlines.Line2D([], [], color="#333333", linestyle="--", linewidth=LINEWIDTH, label="our harness"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, LEGEND_Y),
        ncol=2,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        handlelength=LEGEND_HANDLE_LENGTH,
        columnspacing=LEGEND_COLUMN_SPACING,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(
        left=SUBPLOT_LEFT,
        right=SUBPLOT_RIGHT,
        bottom=SUBPLOT_BOTTOM,
        top=SUBPLOT_TOP,
        wspace=SUBPLOT_WSPACE,
    )
    fig.savefig(out_path, bbox_inches=SAVE_BBOX_INCHES, pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=Path, default=DEFAULT_RUNS_DIR)
    ap.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--metrics", nargs="+", default=["accuracy", "avg_brier"])
    ap.add_argument(
        "--from_start_date",
        action="store_true",
        help="Filter each run's metrics to start at the latest config.start_date "
        "across the selected runs.",
    )
    args = ap.parse_args()

    for metric in args.metrics:
        print(f"\nSelecting runs for {metric}:")
        pairs = collect_paired_metric_runs(args.runs_dir, metric)
        selected_runs = [run for _label, native, our in pairs for run in (native, our)]
        start_filter = None
        if args.from_start_date:
            starts = [r["start_date"] for r in selected_runs if r["start_date"]]
            start_filter = max(starts) if starts else None
            print(f"Filtering metrics from start_date >= {start_filter}")
        out_path = args.output_dir / f"{metric}.png"
        plot_facets(pairs, metric, out_path, start_filter)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
