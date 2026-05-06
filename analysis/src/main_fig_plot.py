"""Main-figure plot from the curated NeurIPS main_fig run folders.

Each group reads every direct timestamped run in
`/fast/sgoel/forecasting/current_sim/neurips_runs/main_fig/<model>/`, averages
across those runs, and shades the across-run standard deviation. `extra/`
subfolders are intentionally ignored.

Models plotted (all at their max reasoning effort, no active-memory variants):
  - gpt-5.5 codex resume (xhigh = default), handholding_version < v3
  - deepseek-v4-pro (Claude Code), handholding_version >= v3
  - qwen-3.6-plus (OpenCode),    handholding_version >= v3
  - glm-5.1 (Claude Code),       handholding_version >= v3
  - claude-opus-4.6 (Claude Code), handholding_version < v3
"""

import argparse
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_config import (  # noqa: F401 -- import side-effects style
    _tangent_offsets,
    color_for_label,
    format_metric_value,
    place_endcap,
    report_missing_logos,
    style_axes,
)

DEFAULT_RUNS_DIR = Path("/fast/sgoel/forecasting/current_sim/neurips_runs/main_fig")
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "plots" / "main_fig"
END_DATE = "2026-03-28"


def hh_lt_v3(v: str | None) -> bool:
    return v in ("v1", "v2")


def hh_ge_v3(v: str | None) -> bool:
    return v == "v3"


def hh_any(v: str | None) -> bool:
    return True


@dataclass
class GroupSpec:
    label: str                          # legend label
    scaffold: str                       # for diagnostics
    model_id: str                       # for diagnostics
    run_dirs: list[str]                 # exact run-folder names under runs_dir
    hh_predicate: Callable[[str | None], bool] = field(default=lambda v: True)


# Curated allowlist. Each run_dirs entry is the exact model folder name in
# DEFAULT_RUNS_DIR. Direct timestamped children are candidates; `extra/`
# subfolders are ignored by collect_runs_for_group().
GROUPS: list[GroupSpec] = [
    GroupSpec(
        label="gpt-5.5 (Codex)",
        scaffold="Codex",
        model_id="gpt-5.5",
        run_dirs=["gpt-5.5"],
        hh_predicate=hh_any,
    ),
    # gpt-5.4 hidden per-user (kept here so it's easy to re-enable later).
    # GroupSpec(
    #     label="gpt-5.4 (Codex)",
    #     scaffold="Codex",
    #     model_id="gpt-5.4",
    #     run_dirs=[
    #         "codex_aljazeeraQ12026v37_gpt54_resume",
    #         "codex_aljazeeraQ12026v37_gpt54_resume_r00",
    #     ],
    #     hh_predicate=hh_lt_v3,
    # ),
    GroupSpec(
        label="deepseek-v4-pro (Claude Code)",
        scaffold="Claude Code",
        model_id="deepseek-v4-pro",
        run_dirs=["ds-v4-pro"],
        hh_predicate=hh_any,
    ),
    GroupSpec(
        label="qwen-3.6-plus (OpenCode)",
        scaffold="OpenCode",
        model_id="qwen-3.6-plus",
        run_dirs=["qwen-3.6-plus"],
        hh_predicate=hh_any,
    ),
    GroupSpec(
        label="glm-5.1 (Claude Code)",
        scaffold="Claude Code",
        model_id="glm-5.1",
        run_dirs=["glm-5.1"],
        hh_predicate=hh_any,
    ),
    GroupSpec(
        label="claude-opus-4.6 (Claude Code)",
        scaffold="Claude Code",
        model_id="claude-opus-4-6",
        run_dirs=["opus-4.6"],
        hh_predicate=hh_any,
    ),
]


def pick_min_variance_seeds(seeds: list[dict], k: int, metric: str) -> list[dict]:
    """Pick the k seeds whose ``metric`` trajectories cluster tightest together.

    For every C(n, k) subset we compute the cross-seed std at each shared date
    and average across time; the subset with the smallest mean std wins. If
    fewer than k seeds are available, all are returned. If ties occur (e.g.
    only one valid combination), the first wins.
    """
    if len(seeds) <= k:
        return seeds
    series: list[pd.Series] = []
    for r in seeds:
        df = r["df"].copy()
        df["date"] = pd.to_datetime(df["date"])
        s = df.groupby("date")[metric].mean()
        series.append(s)
    best_score = float("inf")
    best_combo: tuple[int, ...] = tuple(range(k))
    for combo in itertools.combinations(range(len(seeds)), k):
        merged = pd.concat([series[i] for i in combo], axis=1, join="inner")
        if merged.empty:
            continue
        std = merged.std(axis=1, ddof=0).mean()
        if std < best_score:
            best_score = std
            best_combo = combo
    return [seeds[i] for i in best_combo]


def collect_runs_for_group(runs_dir: Path, spec: GroupSpec) -> list[dict]:
    """Find every (run_dir, ts_dir) under spec.run_dirs that matches the filter."""
    out = []
    for run_name in spec.run_dirs:
        run_path = runs_dir / run_name
        if not run_path.is_dir():
            print(f"  WARN: missing run dir {run_path}")
            continue
        for ts_dir in sorted(run_path.iterdir()):
            if not ts_dir.is_dir() or ts_dir.name == "extra":
                continue
            metrics_path = ts_dir / "daily_metrics.csv"
            if not metrics_path.is_file() or metrics_path.stat().st_size == 0:
                continue
            df = pd.read_csv(metrics_path)
            if df.empty or str(df["date"].iloc[-1]) != END_DATE:
                continue
            # Drop runs that never produced any prediction.
            if "total_predictions" in df.columns and df["total_predictions"].max() <= 0:
                print(f"  skip (0 predictions): {run_name}/{ts_dir.name}")
                continue
            cfg_path = ts_dir / "config.json"
            cfg = {}
            if cfg_path.is_file():
                with cfg_path.open() as f:
                    cfg = json.load(f)
            hh = cfg.get("handholding_version")
            if not spec.hh_predicate(hh):
                continue
            out.append(
                {
                    "run_dir": run_name,
                    "ts_dir": ts_dir.name,
                    "label": spec.label,
                    "hh": hh,
                    "start_date": cfg.get("start_date"),
                    "df": df,
                }
            )
    return out


def aggregate_by_group(
    runs: list[dict], metric: str, start_date_filter: str | None
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    """Return dict[label] -> (dates, mean, std, n_seeds)."""
    by_group: dict[str, list[pd.DataFrame]] = {}
    for r in runs:
        df = r["df"].copy()
        df["date"] = pd.to_datetime(df["date"])
        if start_date_filter is not None:
            df = df[df["date"] >= pd.to_datetime(start_date_filter)]
        # Multi-agent runs would have multiple agent_ids; defensively average.
        daily = df.groupby("date")[metric].mean().reset_index()
        by_group.setdefault(r["label"], []).append(daily)

    out = {}
    for label, frames in by_group.items():
        merged = frames[0][["date"]].copy()
        for i, fr in enumerate(frames):
            merged = merged.merge(
                fr.rename(columns={metric: f"v_{i}"}), on="date", how="outer"
            )
        merged = merged.sort_values("date").reset_index(drop=True)
        value_cols = [c for c in merged.columns if c.startswith("v_")]
        values = merged[value_cols].to_numpy(dtype=float)
        mean = np.nanmean(values, axis=1)
        std = (
            np.nanstd(values, axis=1, ddof=0)
            if values.shape[1] > 1
            else np.zeros(len(merged))
        )
        out[label] = (merged["date"].to_numpy(), mean, std, values.shape[1])
    return out


METRIC_LABELS = {
    "accuracy": "Accuracy (%)",
    "avg_brier": "Brier Skill Score (↑)",
    "tw_score": "Time-Weighted Score",
}

BASELINE_COLOR = "#9A9A9A"
BASELINE_LINESTYLE = (0, (18, 10))
BASELINE_LINEWIDTH = 1.2

def plot_metric(grouped, metric: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    style_axes(ax)

    # Plot in the order GROUPS was declared, skipping any that produced no data.
    keys = [g.label for g in GROUPS if g.label in grouped]
    endpoints: list[tuple[str, object, float, str]] = []  # (key, x, y, value)
    for key in keys:
        dates, mean, std, n = grouped[key]
        color = color_for_label(key)
        ax.plot(dates, mean, color=color, label=key, linewidth=1.8,
                solid_capstyle="round")
        if n > 1:
            ax.fill_between(
                dates, mean - std, mean + std, color=color, alpha=0.18, linewidth=0
            )
        valid = np.where(~np.isnan(mean))[0]
        if len(valid):
            i = valid[-1]
            endpoints.append(
                (key, dates[i], float(mean[i]), format_metric_value(metric, mean[i]))
            )

    # Brier values can dip well below 0 in the warmup phase (e.g. deepseek
    # starts near -0.20). Clip the visible range so the interesting late-run
    # cluster around 0 stays readable. We then annotate any line whose first
    # value falls below the floor with a downward triangle + numeric label so
    # the reader knows the line continues off-axis. Also draw a Baseline
    # reference at y=0.
    if metric == "avg_brier":
        ax.set_ylim(-0.13, 0.10)
        ax.axhline(
            0.0,
            color=BASELINE_COLOR,
            linestyle=BASELINE_LINESTYLE,
            linewidth=BASELINE_LINEWIDTH,
            zorder=1,
        )

    # Per-user: flip these models' name to *above* the badge so it doesn't
    # crash into the badge of the model just below. (claude-opus and glm
    # both want their name *below* now; gpt-5.5 wants its name above.)
    NAME_ABOVE = {"gpt-5.5 (Codex)"}
    # Tangent offsets: when adjacent endpoints would visually overlap (pixel
    # gap < badge diameter), the lower badge slides down by one radius and
    # the upper badge slides up by one radius. This makes the line endpoint
    # touch the badge boundary (top edge or bottom edge) instead of going
    # through its center, so the user can still tell which line each badge
    # belongs to.
    BADGE_RADIUS_PTS = 12.0
    y_offsets = _tangent_offsets(ax, endpoints, radius_pts=BADGE_RADIUS_PTS)
    for (key, x, y, value_str), y_off in zip(endpoints, y_offsets):
        place_endcap(ax, x, y, key, value_str,
                     y_offset_pts=y_off,
                     name_above=key in NAME_ABOVE)

    # Off-bottom marker for any line whose first valid value is below the
    # y-axis floor (the rest of the line is still visible after it climbs up).
    if metric == "avg_brier":
        ymin, _ = ax.get_ylim()
        for key in keys:
            dates, mean, std, n = grouped[key]
            valid = np.where(~np.isnan(mean))[0]
            if not len(valid):
                continue
            i0 = valid[0]
            if mean[i0] < ymin:
                color = color_for_label(key)
                ax.plot(dates[i0], ymin, marker="v", color=color,
                        markersize=12, clip_on=False, zorder=8)
                ax.annotate(
                    f"{mean[i0]:.2f}",
                    xy=(dates[i0], ymin),
                    xytext=(0, -12),
                    textcoords="offset points",
                    ha="center", va="top",
                    color=color, fontsize=11, fontweight="bold",
                    annotation_clip=False,
                )

    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    tick_dates = pd.to_datetime(
        ["2026-01-01", "2026-01-15", "2026-02-01", "2026-02-15",
         "2026-03-01", "2026-03-15", "2026-03-28"]
    )
    ax.set_xticks(tick_dates)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    ax.margins(x=0.01)
    # Reserve room on the right for the badge (centred at line endpoint), its
    # name annotation above, and the value text immediately to its right.
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax + (xmax - xmin) * 0.04)
    ax.tick_params(axis="both", which="major", direction="out", length=4)
    ax.tick_params(axis="y", right=False)
    ax.tick_params(axis="x", top=False)
    ax.minorticks_off()

    # Two-row color-square legend (KellyBench-style markers). Brier clusters
    # near 0 with empty space at the top, so put its legend in the upper-left
    # corner; accuracy lines rise into the upper area, so its legend stays in
    # the lower-right.
    legend_handles = [
        mlines.Line2D([], [], color=color_for_label(k), marker="s",
                      markersize=7, linestyle="None", label=k)
        for k in keys
    ]
    ncol = max(1, -(-len(legend_handles) // 2))  # ceil(n/2) -> two rows
    if metric == "avg_brier":
        legend_kw = dict(loc="upper left",
                         bbox_to_anchor=(0.02, 0.98),
                         bbox_transform=ax.transAxes)
    else:
        legend_kw = dict(loc="lower right",
                         bbox_to_anchor=(0.94, 0.02),
                         bbox_transform=ax.transAxes)
    ax.legend(
        handles=legend_handles,
        ncol=ncol,
        frameon=False,
        fontsize=12,
        handletextpad=0.4,
        columnspacing=0.5,
        borderaxespad=0.0,
        **legend_kw,
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=Path, default=DEFAULT_RUNS_DIR)
    ap.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument(
        "--from_start_date",
        action="store_true",
        help="Filter each run's metrics to start at the latest config.start_date "
             "across selected runs (skip warmup days).",
    )
    args = ap.parse_args()

    all_runs: list[dict] = []
    for spec in GROUPS:
        rs = collect_runs_for_group(args.runs_dir, spec)
        print(f"{spec.label}: {len(rs)} candidate seed(s)")
        for r in rs:
            print(f"  hh={r['hh']}  {r['run_dir']}/{r['ts_dir']}")
        if not rs:
            print(f"  WARN: no runs matched for {spec.label}")
        all_runs.extend(rs)

    if not all_runs:
        raise SystemExit("No runs matched any group spec.")

    start_filter = None
    if args.from_start_date:
        starts = [r["start_date"] for r in all_runs if r["start_date"]]
        start_filter = max(starts) if starts else None
        print(f"Filtering metrics from start_date >= {start_filter}")

    report_missing_logos({r["label"] for r in all_runs})

    for metric in ("accuracy", "avg_brier"):
        grouped = aggregate_by_group(all_runs, metric, start_filter)
        out_path = args.output_dir / f"{metric}.png"
        plot_metric(grouped, metric, out_path)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
