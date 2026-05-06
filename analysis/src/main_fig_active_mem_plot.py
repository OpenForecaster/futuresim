"""Main-figure plot comparing native harness vs our harness.

Uses native-harness candidates under `neurips_runs/main_fig` and the
corresponding runs under `neurips_runs/our_harness`. For each metric, the
native candidate is chosen as the median direct timestamped main-fig run by
that metric's Day0 value.

Solid lines = native harness, dashed lines = our harness.
Each (model, scaffold) pair shares a color so the comparison reads by line-style.
A single combined legend at the top labels every line.
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory

from plot_config import (  # noqa: F401 -- import side-effects style
    _resolve_endpoint_offsets,
    _tangent_offsets,
    color_for_label,
    format_metric_value,
    place_endcap,
    report_missing_logos,
    style_axes,
)

DEFAULT_RUNS_DIR = Path("/fast/sgoel/forecasting/current_sim/neurips_runs")
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "plots" / "main_fig_active_mem"
END_DATE = "2026-03-28"


def hh_lt_v3(v: str | None) -> bool:
    return v in ("v1", "v2")


def hh_ge_v3(v: str | None) -> bool:
    return v == "v3"


def hh_any(v: str | None) -> bool:
    return True


@dataclass
class GroupSpec:
    label: str                          # model label shared by native/our harness
    is_active_mem: bool                 # True -> our harness dashed, False -> native solid
    run_dirs: list[str]
    hh_predicate: Callable[[str | None], bool] = field(default=lambda v: True)


# Each model has two specs: native harness and our harness. Models without both
# sides are pruned at runtime so only matched pairs are plotted.
GROUPS: list[GroupSpec] = [
    GroupSpec(
        label="gpt-5.5",
        is_active_mem=False,
        run_dirs=["main_fig/gpt-5.5"],
    ),
    GroupSpec(
        label="gpt-5.5",
        is_active_mem=True,
        run_dirs=["our_harness/gpt-5.5"],
    ),
    GroupSpec(
        label="deepseek-v4-pro",
        is_active_mem=False,
        run_dirs=["main_fig/ds-v4-pro"],
    ),
    GroupSpec(
        label="deepseek-v4-pro",
        is_active_mem=True,
        run_dirs=["our_harness/ds-v4-pro"],
    ),
    GroupSpec(
        label="qwen-3.6-plus",
        is_active_mem=False,
        run_dirs=["main_fig/qwen-3.6-plus"],
    ),
    GroupSpec(
        label="qwen-3.6-plus",
        is_active_mem=True,
        run_dirs=["our_harness/qwen-3.6-plus"],
    ),
    GroupSpec(
        label="glm-5.1",
        is_active_mem=False,
        run_dirs=["main_fig/glm-5.1"],
    ),
    GroupSpec(
        label="glm-5.1",
        is_active_mem=True,
        run_dirs=["our_harness/glm-5.1"],
    ),
]


def _read_config(ts_dir: Path) -> dict:
    # Some copied run directories only carry source_config.json.
    for name in ("config.json", "source_config.json"):
        p = ts_dir / name
        if p.is_file():
            with p.open() as f:
                return json.load(f)
    return {}


def collect_runs_for_group(runs_dir: Path, spec: GroupSpec) -> list[dict]:
    out = []
    for run_name in spec.run_dirs:
        run_path = runs_dir / run_name
        if not run_path.is_dir():
            print(f"  WARN: missing run dir {run_path}")
            continue
        for metrics_path in sorted(run_path.glob("*/daily_metrics.csv")):
            ts_dir = metrics_path.parent
            if not metrics_path.is_file() or metrics_path.stat().st_size == 0:
                continue
            df = pd.read_csv(metrics_path)
            if df.empty:
                continue
            # Drop runs that never produced any prediction.
            if "total_predictions" in df.columns and df["total_predictions"].max() <= 0:
                print(f"  skip (0 predictions): {run_name}/{ts_dir.name}")
                continue
            cfg = _read_config(ts_dir)
            hh = cfg.get("handholding_version")
            if not spec.hh_predicate(hh):
                continue
            out.append(
                {
                    "run_dir": run_name,
                    "ts_dir": str(ts_dir.relative_to(run_path)),
                    "label": spec.label,
                    "is_active_mem": spec.is_active_mem,
                    "hh": hh,
                    "start_date": cfg.get("start_date"),
                    "df": df,
                }
            )
    return out


def _first_metric(df: pd.DataFrame, metric: str) -> float | None:
    """Return a metric value on the first day of this seed's daily_metrics.csv."""
    if df.empty or metric not in df.columns:
        return None
    val = df.iloc[0][metric]
    if pd.isna(val):
        return None
    return float(val)


def pick_median_day0_pair(
    native_runs: list[dict], our_runs: list[dict], metric: str
) -> tuple[dict | None, dict | None]:
    """Pick median native seed by Day0 metric and the corresponding our run."""
    native_with_value = [(r, _first_metric(r["df"], metric)) for r in native_runs]
    our_with_value = [(r, _first_metric(r["df"], metric)) for r in our_runs]
    native_with_value = [(r, a) for r, a in native_with_value if a is not None]
    our_with_value = [(r, a) for r, a in our_with_value if a is not None]
    if not native_with_value or not our_with_value:
        return (None, None)
    median_value = float(np.median([v for _r, v in native_with_value]))
    best_dist = min(abs(v - median_value) for _r, v in native_with_value)
    # If there are an even number of candidates and two runs are equally close
    # to the numeric median, prefer the later timestamp.
    median_native = sorted(
        [(r, v) for r, v in native_with_value if abs(v - median_value) == best_dist],
        key=lambda rv: rv[0]["ts_dir"],
    )[-1][0]
    our_run = sorted(our_with_value, key=lambda rv: rv[0]["ts_dir"])[-1][0]
    return (median_native, our_run)


def aggregate_by_series(
    runs: list[dict], metric: str, start_date_filter: str | None
) -> dict[tuple[str, bool], tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    """Return dict[(label, is_active_mem)] -> (dates, mean, std, n_seeds)."""
    by_series: dict[tuple[str, bool], list[pd.DataFrame]] = {}
    for r in runs:
        df = r["df"].copy()
        df["date"] = pd.to_datetime(df["date"])
        if start_date_filter is not None:
            df = df[df["date"] >= pd.to_datetime(start_date_filter)]
        daily = df.groupby("date")[metric].mean().reset_index()
        key = (r["label"], r["is_active_mem"])
        by_series.setdefault(key, []).append(daily)

    out = {}
    for key, frames in by_series.items():
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
        out[key] = (merged["date"].to_numpy(), mean, std, values.shape[1])
    return out


METRIC_LABELS = {
    "accuracy": "Accuracy (%)",
    "avg_brier": "Brier Skill Score (↑)",
    "tw_score": "Time-Weighted Score",
}

DATA_LINEWIDTH = 3.2
REFERENCE_LINEWIDTH = 2.0
LEGEND_LINEWIDTH = 3.2
AXIS_LABEL_FONTSIZE = 24
TICK_FONTSIZE = 20
END_VALUE_FONTSIZE = 17
MODEL_LEGEND_FONTSIZE = 17
STYLE_LEGEND_FONTSIZE = 16
OFF_AXIS_FONTSIZE = 15
MODEL_MARKERSIZE = 9
BASELINE_COLOR = "#9A9A9A"
BASELINE_LINESTYLE = (0, (18, 10))
BASELINE_LINEWIDTH = REFERENCE_LINEWIDTH


def endpoint_metric_value(metric: str, value: float) -> str:
    if metric == "avg_brier":
        return f"{value:.3f}"
    return format_metric_value(metric, value)


def plot_metric(grouped, metric: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    style_axes(ax)
    ax.spines["left"].set_linewidth(1.4)
    ax.spines["bottom"].set_linewidth(1.4)

    # Stable order: follow GROUPS declaration so each model's native+our harness
    # appear consecutively in the legend.
    seen = set()
    ordered_keys = []
    for g in GROUPS:
        k = (g.label, g.is_active_mem)
        if k in grouped and k not in seen:
            seen.add(k)
            ordered_keys.append(k)

    # Cache both line endpoints per label so final values can be labeled.
    native_endpoints: dict[str, tuple] = {}
    our_endpoints: dict[str, tuple] = {}
    for key in ordered_keys:
        dates, mean, std, n = grouped[key]
        label, is_am = key
        valid = np.where(~np.isnan(mean))[0]
        if not len(valid):
            continue
        i = valid[-1]
        (our_endpoints if is_am else native_endpoints)[label] = (dates[i], mean[i])

    for key in ordered_keys:
        dates, mean, std, n = grouped[key]
        label, is_am = key
        color = color_for_label(label)
        linestyle = "--" if is_am else "-"
        ax.plot(dates, mean, color=color, linestyle=linestyle, linewidth=DATA_LINEWIDTH,
                solid_capstyle="round")
        if n > 1:
            ax.fill_between(
                dates, mean - std, mean + std, color=color, alpha=0.12, linewidth=0
            )

    if metric == "accuracy":
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + 0.16 * (ymax - ymin))

    # Brier values can dip well below 0 in the warmup phase (e.g. deepseek
    # starts near -0.30). Clip the visible range BEFORE placing badges so the
    # tangent offset computation runs against the final y-axis. We then
    # annotate any line whose first value falls below the floor with a
    # downward triangle + numeric label so the reader knows the line continues
    # off-axis. Also draw a baseline reference at y=0.
    if metric == "avg_brier":
        ax.set_ylim(-0.12, 0.09)
        ax.axhline(
            0.0,
            color=BASELINE_COLOR,
            linestyle=BASELINE_LINESTYLE,
            linewidth=BASELINE_LINEWIDTH,
            zorder=1,
        )

    # Aligned end-of-line value labels. Badges/logos are hidden (the legend
    # labels colors; the line style legend disambiguates native vs our harness),
    # so each line just gets a numeric value placed in a single fixed column
    # just past the right edge of the axis. We use a blended transform
    # (axes-fraction x, data y) so values stay aligned regardless of where
    # individual lines end, and right-align (`ha="right"`) so the last digit
    # / `%` lines up across rows even when some values carry a leading minus
    # sign. Native and our-harness values share the same column; we then run
    # `_resolve_endpoint_offsets` (which handles arbitrary cluster sizes) so
    # values that would land at the same y are pushed apart vertically.
    VALUE_X_FRAC = 1.04
    MIN_GAP_PTS = 28.0
    items: list[tuple[str, object, float, str]] = []
    for label, (x, y) in native_endpoints.items():
        if metric == "avg_brier" and (y < -0.10 or y > 0.10):
            continue
        items.append((label, x, float(y), endpoint_metric_value(metric, y)))
    for label, (x, y) in our_endpoints.items():
        if metric == "avg_brier" and (y < -0.10 or y > 0.10):
            continue
        items.append((label, x, float(y), endpoint_metric_value(metric, y)))
    y_offsets = _resolve_endpoint_offsets(ax, items, min_gap_pts=MIN_GAP_PTS)
    blended = blended_transform_factory(ax.transAxes, ax.transData)
    brier_value_x = None
    if metric == "avg_brier" and items:
        endpoint_xs = [pd.to_datetime(x) for _label, x, _y, _value_str in items]
        max_endpoint_x = max(endpoint_xs)
        brier_value_x = max_endpoint_x + pd.Timedelta(days=5)
        left, _right = ax.get_xlim()
        ax.set_xlim(left=left, right=mdates.date2num(max_endpoint_x + pd.Timedelta(days=16)))
    for (label, _x, y, value_str), y_off in zip(items, y_offsets):
        xy = (brier_value_x, y) if brier_value_x is not None else (VALUE_X_FRAC, y)
        xycoords = "data" if brier_value_x is not None else blended
        ax.annotate(
            value_str,
            xy=xy,
            xycoords=xycoords,
            xytext=(0, y_off),
            textcoords="offset points",
            ha="left" if brier_value_x is not None else "right", va="center",
            color=color_for_label(label),
            fontsize=END_VALUE_FONTSIZE, fontweight="bold",
            annotation_clip=False,
        )

    # Off-bottom marker for any line (native or our harness) whose first valid
    # value is below the y-axis floor.
    if metric == "avg_brier":
        ymin, _ = ax.get_ylim()
        for key in ordered_keys:
            dates, mean, std, n = grouped[key]
            label, is_am = key
            valid = np.where(~np.isnan(mean))[0]
            if not len(valid):
                continue
            i0 = valid[0]
            if mean[i0] < ymin:
                color = color_for_label(label)
                ax.plot(dates[i0], ymin, marker="v", color=color,
                        markersize=12, clip_on=False, zorder=8)
                ax.annotate(
                    f"{mean[i0]:.2f}",
                    xy=(dates[i0], ymin),
                    xytext=(0, -12),
                    textcoords="offset points",
                    ha="center", va="top",
                    color=color, fontsize=OFF_AXIS_FONTSIZE, fontweight="bold",
                    annotation_clip=False,
                )

    ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=AXIS_LABEL_FONTSIZE)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    tick_dates = pd.to_datetime(
        ["2026-01-01", "2026-02-01", "2026-03-01", "2026-03-28"]
    )
    ax.set_xticks(tick_dates)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=TICK_FONTSIZE)
    plt.setp(ax.get_yticklabels(), fontsize=TICK_FONTSIZE)
    ax.margins(x=0.01)
    ax.tick_params(axis="both", which="major", direction="out", length=5, width=1.4)
    ax.tick_params(axis="y", right=False)
    ax.tick_params(axis="x", top=False)
    ax.minorticks_off()

    # KellyBench-style compact color-square legend (one entry per model).
    seen_labels = set()
    color_handles = []
    for g in GROUPS:
        if g.label in seen_labels:
            continue
        if (g.label, True) not in grouped and (g.label, False) not in grouped:
            continue
        seen_labels.add(g.label)
        short_label = g.label.split(" (")[0]
        color_handles.append(
            mlines.Line2D([], [], color=color_for_label(g.label),
                          marker="s", markersize=MODEL_MARKERSIZE, linestyle="None",
                          label=short_label)
        )
    legend1 = ax.legend(
        handles=color_handles,
        loc="upper center",
        bbox_to_anchor=(0.58, -0.13),
        bbox_transform=ax.transAxes,
        ncol=max(1, min(2, len(color_handles))),
        frameon=False,
        fontsize=MODEL_LEGEND_FONTSIZE,
        handletextpad=0.4,
        columnspacing=1.4,
        borderaxespad=0.0,
    )
    ax.add_artist(legend1)

    # Separate inset legend for line styles (native vs our harness), pinned
    # to the top-left corner of the plot.
    style_handles = [
        mlines.Line2D([], [], color="#444444", linestyle="-",  linewidth=LEGEND_LINEWIDTH, label="native harness"),
        mlines.Line2D([], [], color="#444444", linestyle="--", linewidth=LEGEND_LINEWIDTH, label="our harness"),
    ]
    ax.legend(
        handles=style_handles,
        loc="upper left",
        ncol=1,
        frameon=False,
        fontsize=STYLE_LEGEND_FONTSIZE,
        borderaxespad=0.8,
        handlelength=2.4,
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", bbox_extra_artists=(legend1,))
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

    # Collect everything, then prune labels lacking a native/our-harness pair.
    per_spec_runs: list[tuple[GroupSpec, list[dict]]] = []
    for spec in GROUPS:
        rs = collect_runs_for_group(args.runs_dir, spec)
        per_spec_runs.append((spec, rs))

    have_native = {s.label for s, rs in per_spec_runs if rs and not s.is_active_mem}
    have_our = {s.label for s, rs in per_spec_runs if rs and s.is_active_mem}
    paired_labels = have_native & have_our

    dropped = (have_native | have_our) - paired_labels
    if dropped:
        print(f"Dropping unpaired models: {sorted(dropped)}")

    # Group by label, then for each metric pick the median native Day0 run.
    runs_by_label_native: dict[str, list[dict]] = {}
    runs_by_label_our: dict[str, list[dict]] = {}
    for spec, rs in per_spec_runs:
        if spec.label not in paired_labels:
            continue
        bucket = runs_by_label_our if spec.is_active_mem else runs_by_label_native
        bucket.setdefault(spec.label, []).extend(rs)

    for metric in ("accuracy", "avg_brier"):
        all_runs: list[dict] = []
        print(f"\nSelecting median native runs by Day0 {metric}:")
        for label in [g.label for g in GROUPS if g.label in paired_labels]:
            if label in {r["label"] for r in all_runs}:
                continue
            nr, ar = pick_median_day0_pair(
                runs_by_label_native.get(label, []),
                runs_by_label_our.get(label, []),
                metric,
            )
            if nr is None or ar is None:
                print(f"  WARN: no day-0 {metric} available for pairing in {label}")
                continue
            nv = _first_metric(nr["df"], metric)
            av = _first_metric(ar["df"], metric)
            n_candidates = len([
                r for r in runs_by_label_native.get(label, [])
                if _first_metric(r["df"], metric) is not None
            ])
            print(f"{label}: median Day0 {metric} native run ({n_candidates} candidate(s))")
            print(f"  native      {nr['run_dir']}/{nr['ts_dir']}  day0_{metric}={nv:.4f}")
            print(f"  our         {ar['run_dir']}/{ar['ts_dir']}  day0_{metric}={av:.4f}")
            all_runs.extend([nr, ar])

        if not all_runs:
            raise SystemExit(f"No paired runs to plot for {metric}.")

        start_filter = None
        if args.from_start_date:
            starts = [r["start_date"] for r in all_runs if r["start_date"]]
            start_filter = max(starts) if starts else None
            print(f"Filtering metrics from start_date >= {start_filter}")

        report_missing_logos({r["label"] for r in all_runs})
        grouped = aggregate_by_series(all_runs, metric, start_filter)
        out_path = args.output_dir / f"{metric}.png"
        plot_metric(grouped, metric, out_path)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
