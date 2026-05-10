#!/usr/bin/env python3
"""Plot DeepSeek multi-agent vs single-agent crowd analyses.

Outputs:
- TV distance to each run's own crowd aggregate over time, over the fixed
  intersection of questions predicted by all three agents/runs.
- Brier skill over time.
- Accuracy over time.

Single-agent baselines are treated as a 3-member crowd across r00/r01/r02.
Multi-agent lines are the three agents inside one multi-agent run.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


AGENT_LABELS = ["Agent 1", "Agent 2", "Agent 3"]
BLUE_SHADES = ["#9ecae1", "#3182bd", "#08519c"]
RED_SHADES = ["#fcae91", "#fb6a4a", "#cb181d"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multi_run", required=True, help="Timestamped 3-agent run directory")
    parser.add_argument(
        "--single_parent",
        default="/fast/sgoel/forecasting/current_sim/final_runs_v37",
        help="Parent containing ds_v32_active_mem_aljazeera2026Q1_1agent_r00/r01/r02",
    )
    parser.add_argument(
        "--single_run_dirs",
        nargs=3,
        default=None,
        help="Explicit timestamped single-agent run dirs for Agent 1/2/3",
    )
    parser.add_argument(
        "--single_run_prefix",
        default="ds_v32_active_mem_aljazeera2026Q1_1agent",
        help="Prefix for single-agent run parents under --single_parent",
    )
    parser.add_argument(
        "--output_dir",
        default="analysis/output/deepseek_multi_vs_singleagent",
        help="Directory for plots and derived CSVs",
    )
    parser.add_argument("--title_prefix", default="DeepSeek v3.2", help="Prefix for plot titles")
    parser.add_argument(
        "--only_tv",
        action="store_true",
        help="Only write the TV-distance plot and CSV, skipping brier/accuracy/panel plots",
    )
    return parser.parse_args()


def apply_paper_style() -> None:
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


def latest_timestamp_dir(parent: Path) -> Path:
    candidates = [
        child
        for child in parent.iterdir()
        if child.is_dir() and (child / "actions.jsonl").exists() and (child / "daily_metrics.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No timestamped run directories with actions/daily metrics under {parent}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def discover_single_runs(args: argparse.Namespace) -> List[Path]:
    if args.single_run_dirs:
        return [Path(p).expanduser().resolve() for p in args.single_run_dirs]
    parent = Path(args.single_parent).expanduser().resolve()
    runs = []
    for idx in range(3):
        run_parent = parent / f"{args.single_run_prefix}_r{idx:02d}"
        runs.append(latest_timestamp_dir(run_parent))
    return runs


def load_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def normalize_dist(dist: Mapping[str, object]) -> Dict[str, float]:
    out = {}
    total = 0.0
    for key, value in dist.items():
        try:
            prob = float(value)
        except (TypeError, ValueError):
            continue
        if prob <= 0:
            continue
        out[str(key)] = prob
        total += prob
    if total <= 0:
        return {}
    return {key: val / total for key, val in out.items()}


def tv_distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(float(left.get(k, 0.0)) - float(right.get(k, 0.0))) for k in keys)


def average_distribution(dists: Iterable[Mapping[str, float]]) -> Dict[str, float]:
    dists = list(dists)
    if not dists:
        return {}
    accum: Dict[str, float] = defaultdict(float)
    for dist in dists:
        for outcome, prob in dist.items():
            accum[outcome] += float(prob) / len(dists)
    return dict(accum)


def tv_timeseries_from_group(
    run_paths: List[Path],
    labels: List[str],
    source_kind: str,
    agent_filters: List[str | None] | None = None,
) -> pd.DataFrame:
    """Compute mean per-agent TV-to-crowd over all carried-forward predictions.

    Resolutions are intentionally ignored here. For this analysis, the
    question set should remain fixed over time; each agent's latest prediction
    for each question is carried forward after the question resolves.
    """
    current: Dict[Tuple[str, str], Dict[str, float]] = {}
    records_by_date: Dict[str, List[dict]] = defaultdict(list)

    if agent_filters is None:
        agent_filters = [None] * len(run_paths)

    for run_path, label, agent_filter in zip(run_paths, labels, agent_filters):
        for rec in load_jsonl(run_path / "actions.jsonl"):
            if rec.get("type") == "prediction" and agent_filter is not None:
                if str(rec.get("agent_id")) != str(agent_filter):
                    continue
            rec = dict(rec)
            rec["_series_label"] = label
            records_by_date[str(rec.get("sim_date"))].append(rec)

    rows = []
    for sim_date in sorted(records_by_date):
        if sim_date in {"None", ""}:
            continue
        for rec in records_by_date[sim_date]:
            rtype = rec.get("type")
            qid = str(rec.get("question_id")) if rec.get("question_id") is not None else ""
            if not qid:
                continue
            if rtype == "prediction":
                dist = normalize_dist(rec.get("outcomes") or {})
                if dist:
                    current[(rec["_series_label"], qid)] = dist

        predicted_qids = sorted(
            {
                qid
                for qid in {qid for _, qid in current}
                if all((label, qid) in current for label in labels)
            }
        )
        if not predicted_qids:
            continue

        per_agent_values = {label: [] for label in labels}
        for qid in predicted_qids:
            dists = [current[(label, qid)] for label in labels]
            crowd = average_distribution(dists)
            for label, dist in zip(labels, dists):
                per_agent_values[label].append(tv_distance(dist, crowd))

        for label in labels:
            vals = per_agent_values[label]
            rows.append(
                {
                    "date": sim_date,
                    "agent": label,
                    "source": source_kind,
                    "avg_tv_to_crowd": sum(vals) / len(vals),
                    "n_questions": len(vals),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(f"No TV rows computed for {source_kind}: {run_paths}")
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["date", "source", "agent"])


def brier_timeseries_from_metrics(run_paths: List[Path], labels: List[str], source_kind: str) -> pd.DataFrame:
    frames = []
    for run_path, label in zip(run_paths, labels):
        df = pd.read_csv(run_path / "daily_metrics.csv")
        if df.empty:
            continue
        # Single-agent runs reuse the same internal agent id; multi-agent rows
        # already appear in deterministic 001/002/003 order.
        agent_ids = sorted(df["agent_id"].astype(str).unique())
        if len(agent_ids) == 1:
            part = df.copy()
        else:
            idx = labels.index(label)
            part = df[df["agent_id"].astype(str) == agent_ids[idx]].copy()
        part["date"] = pd.to_datetime(part["date"])
        part["agent"] = label
        part["source"] = source_kind
        part["brier_skill"] = pd.to_numeric(part["avg_brier"], errors="coerce")
        frames.append(part[["date", "agent", "source", "brier_skill"]])
    if not frames:
        raise ValueError(f"No brier rows loaded for {source_kind}: {run_paths}")
    return pd.concat(frames, ignore_index=True).sort_values(["date", "source", "agent"])


def accuracy_timeseries_from_metrics(run_paths: List[Path], labels: List[str], source_kind: str) -> pd.DataFrame:
    frames = []
    for run_path, label in zip(run_paths, labels):
        df = pd.read_csv(run_path / "daily_metrics.csv")
        if df.empty:
            continue
        agent_ids = sorted(df["agent_id"].astype(str).unique())
        if len(agent_ids) == 1:
            part = df.copy()
        else:
            idx = labels.index(label)
            part = df[df["agent_id"].astype(str) == agent_ids[idx]].copy()
        part["date"] = pd.to_datetime(part["date"])
        part["agent"] = label
        part["source"] = source_kind
        part["accuracy"] = pd.to_numeric(part["accuracy"], errors="coerce")
        frames.append(part[["date", "agent", "source", "accuracy"]])
    if not frames:
        raise ValueError(f"No accuracy rows loaded for {source_kind}: {run_paths}")
    return pd.concat(frames, ignore_index=True).sort_values(["date", "source", "agent"])


def common_date_filter(*dfs: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    starts = [df["date"].min() for df in dfs if not df.empty]
    ends = [df["date"].max() for df in dfs if not df.empty]
    start = max(starts)
    end = min(ends)
    return tuple(df[(df["date"] >= start) & (df["date"] <= end)].copy() for df in dfs)


def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, title: str, out_base: Path) -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    _plot_metric_on_axis(ax, df, metric, ylabel, title)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_grouped_metric_square(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    out_base: Path,
    reserve_legend_space: bool = False,
) -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(5.2, 5.2))

    color_map = {
        ("single", AGENT_LABELS[0]): BLUE_SHADES[0],
        ("single", AGENT_LABELS[1]): BLUE_SHADES[1],
        ("single", AGENT_LABELS[2]): BLUE_SHADES[2],
        ("multi", AGENT_LABELS[0]): RED_SHADES[0],
        ("multi", AGENT_LABELS[1]): RED_SHADES[1],
        ("multi", AGENT_LABELS[2]): RED_SHADES[2],
    }
    linestyle_map = {"single": "--", "multi": "-"}

    for source in ["single", "multi"]:
        for agent in AGENT_LABELS:
            part = df[(df["source"] == source) & (df["agent"] == agent)].sort_values("date")
            if part.empty:
                continue
            ax.plot(
                part["date"],
                part[metric],
                color=color_map[(source, agent)],
                linestyle=linestyle_map[source],
                alpha=0.96,
            )

    ax.set_ylabel(ylabel, fontsize=17, labelpad=14)
    ax.set_xlabel("")
    ax.tick_params(axis="both", labelsize=14)
    if reserve_legend_space:
        ymin = float(df[metric].min())
        ymax = float(df[metric].max())
        yspan = ymax - ymin
        if yspan > 0:
            ax.set_ylim(ymin - 0.04 * yspan, ymax + 0.28 * yspan)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    start = pd.Timestamp(df["date"].min())
    end = pd.Timestamp(df["date"].max())
    tick_dates = [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01"), pd.Timestamp("2026-03-01"), end]
    tick_dates = [d for d in tick_dates if start <= d <= end]
    ax.set_xticks(tick_dates)
    ax.set_xlim(start, end)
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment("right")

    single_handles = [
        Line2D([0], [0], color=BLUE_SHADES[i], linestyle="--", linewidth=2.4, label=AGENT_LABELS[i])
        for i in range(3)
    ]
    multi_handles = [
        Line2D([0], [0], color=RED_SHADES[i], linestyle="-", linewidth=2.4, label=AGENT_LABELS[i])
        for i in range(3)
    ]
    single_legend = ax.legend(
        handles=single_handles,
        title="Single Agent Runs",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        fontsize=9.5,
        title_fontsize=9.8,
        handlelength=2.4,
        borderaxespad=0.0,
        labelspacing=0.35,
    )
    ax.add_artist(single_legend)
    ax.legend(
        handles=multi_handles,
        title="Multi Agent Runs",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.385, 0.98),
        fontsize=9.5,
        title_fontsize=9.8,
        handlelength=2.4,
        borderaxespad=0.0,
        labelspacing=0.35,
    )

    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_tv_distance(df: pd.DataFrame, out_base: Path) -> None:
    plot_grouped_metric_square(
        df,
        "avg_tv_to_crowd",
        "TV Distance to Aggregate Prediction",
        out_base,
    )


def _plot_metric_on_axis(ax, df: pd.DataFrame, metric: str, ylabel: str, title: str) -> None:
    color_map = {
        ("single", AGENT_LABELS[0]): BLUE_SHADES[0],
        ("single", AGENT_LABELS[1]): BLUE_SHADES[1],
        ("single", AGENT_LABELS[2]): BLUE_SHADES[2],
        ("multi", AGENT_LABELS[0]): RED_SHADES[0],
        ("multi", AGENT_LABELS[1]): RED_SHADES[1],
        ("multi", AGENT_LABELS[2]): RED_SHADES[2],
    }
    linestyle_map = {"single": "--", "multi": "-"}
    label_prefix = {"single": "Single", "multi": "Multi"}

    for source in ["single", "multi"]:
        for agent in AGENT_LABELS:
            part = df[(df["source"] == source) & (df["agent"] == agent)].sort_values("date")
            if part.empty:
                continue
            ax.plot(
                part["date"],
                part[metric],
                color=color_map[(source, agent)],
                linestyle=linestyle_map[source],
                label=f"{label_prefix[source]} {agent}",
                alpha=0.96,
            )

    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=8))
    ax.legend(ncol=2, frameon=False, loc="best")


def plot_panel(tv_df: pd.DataFrame, brier_df: pd.DataFrame, out_base: Path, title_prefix: str) -> None:
    apply_paper_style()
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)
    _plot_metric_on_axis(
        axes[0],
        tv_df,
        "avg_tv_to_crowd",
        "Avg TV Distance",
        f"{title_prefix}: Distance to Own Crowd Aggregate (All Predictions)",
    )
    _plot_metric_on_axis(
        axes[1],
        brier_df,
        "brier_skill",
        "Brier Skill",
        f"{title_prefix}: Brier Skill Over Time",
    )
    axes[1].set_xlabel("Simulation Date")
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    multi_run = Path(args.multi_run).expanduser().resolve()
    single_runs = discover_single_runs(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    multi_metrics = pd.read_csv(multi_run / "daily_metrics.csv")
    multi_agent_ids = sorted(multi_metrics["agent_id"].astype(str).unique())
    if len(multi_agent_ids) != 3:
        raise ValueError(f"Expected 3 multi-agent IDs in {multi_run}, found {multi_agent_ids}")

    multi_paths = [multi_run, multi_run, multi_run]
    tv_single = tv_timeseries_from_group(single_runs, AGENT_LABELS, "single")
    tv_multi = tv_timeseries_from_group(
        multi_paths,
        AGENT_LABELS,
        "multi",
        agent_filters=multi_agent_ids,
    )
    tv_single, tv_multi = common_date_filter(tv_single, tv_multi)
    tv_df = pd.concat([tv_single, tv_multi], ignore_index=True).sort_values(["date", "source", "agent"])

    tv_csv = out_dir / "deepseek_multi_vs_single_tv_to_crowd_timeseries.csv"
    tv_df.to_csv(tv_csv, index=False)

    plot_tv_distance(tv_df, out_dir / "deepseek_multi_vs_single_tv_to_crowd")

    if args.only_tv:
        manifest = {
            "multi_run": str(multi_run),
            "single_runs": {label: str(path) for label, path in zip(AGENT_LABELS, single_runs)},
            "common_tv_start": str(tv_df["date"].min().date()),
            "common_tv_end": str(tv_df["date"].max().date()),
            "note": (
                "Single-agent baseline crowd aggregate is computed across latest r00/r01/r02 runs; "
                "multi-agent crowd aggregate is computed across the three agents inside the multi-agent run. "
                "TV distance carries each question's latest prediction forward after resolution so the "
                "question universe remains fixed over time."
            ),
            "outputs": {
                "tv_png": str(out_dir / "deepseek_multi_vs_single_tv_to_crowd.png"),
                "tv_pdf": str(out_dir / "deepseek_multi_vs_single_tv_to_crowd.pdf"),
                "tv_csv": str(tv_csv),
            },
        }
        (out_dir / "deepseek_multi_vs_single_tv_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        print("Wrote:")
        for value in manifest["outputs"].values():
            print(f"  {value}")
        return

    brier_single = brier_timeseries_from_metrics(single_runs, AGENT_LABELS, "single")
    brier_multi = brier_timeseries_from_metrics(multi_paths, AGENT_LABELS, "multi")
    accuracy_single = accuracy_timeseries_from_metrics(single_runs, AGENT_LABELS, "single")
    accuracy_multi = accuracy_timeseries_from_metrics(multi_paths, AGENT_LABELS, "multi")

    brier_single, brier_multi = common_date_filter(brier_single, brier_multi)
    accuracy_single, accuracy_multi = common_date_filter(accuracy_single, accuracy_multi)

    brier_df = pd.concat([brier_single, brier_multi], ignore_index=True).sort_values(["date", "source", "agent"])
    accuracy_df = pd.concat([accuracy_single, accuracy_multi], ignore_index=True).sort_values(
        ["date", "source", "agent"]
    )

    brier_csv = out_dir / "deepseek_multi_vs_single_brier_skill_timeseries.csv"
    accuracy_csv = out_dir / "deepseek_multi_vs_single_accuracy_timeseries.csv"
    brier_df.to_csv(brier_csv, index=False)
    accuracy_df.to_csv(accuracy_csv, index=False)

    plot_grouped_metric_square(
        brier_df,
        "brier_skill",
        "Brier Skill Score",
        out_dir / "deepseek_multi_vs_single_brier_skill",
        reserve_legend_space=True,
    )
    plot_grouped_metric_square(
        accuracy_df,
        "accuracy",
        "Accuracy (%)",
        out_dir / "deepseek_multi_vs_single_accuracy",
        reserve_legend_space=True,
    )
    plot_panel(
        tv_df,
        brier_df,
        out_dir / "deepseek_multi_vs_single_tv_and_brier_panel",
        args.title_prefix,
    )

    manifest = {
        "multi_run": str(multi_run),
        "single_runs": {label: str(path) for label, path in zip(AGENT_LABELS, single_runs)},
        "common_tv_start": str(tv_df["date"].min().date()),
        "common_tv_end": str(tv_df["date"].max().date()),
        "common_brier_start": str(brier_df["date"].min().date()),
        "common_brier_end": str(brier_df["date"].max().date()),
        "common_accuracy_start": str(accuracy_df["date"].min().date()),
        "common_accuracy_end": str(accuracy_df["date"].max().date()),
        "note": (
            "Single-agent baseline crowd aggregate is computed across latest r00/r01/r02 runs; "
            "multi-agent crowd aggregate is computed across the three agents inside the multi-agent run. "
            "TV distance carries each question's latest prediction forward after resolution so the "
            "question universe remains fixed over time. "
            "Dashed blue lines are single-agent baselines; solid red lines are multi-agent lines."
        ),
        "outputs": {
            "tv_png": str(out_dir / "deepseek_multi_vs_single_tv_to_crowd.png"),
            "tv_pdf": str(out_dir / "deepseek_multi_vs_single_tv_to_crowd.pdf"),
            "brier_png": str(out_dir / "deepseek_multi_vs_single_brier_skill.png"),
            "brier_pdf": str(out_dir / "deepseek_multi_vs_single_brier_skill.pdf"),
            "accuracy_png": str(out_dir / "deepseek_multi_vs_single_accuracy.png"),
            "accuracy_pdf": str(out_dir / "deepseek_multi_vs_single_accuracy.pdf"),
            "panel_png": str(out_dir / "deepseek_multi_vs_single_tv_and_brier_panel.png"),
            "panel_pdf": str(out_dir / "deepseek_multi_vs_single_tv_and_brier_panel.pdf"),
            "tv_csv": str(tv_csv),
            "brier_csv": str(brier_csv),
            "accuracy_csv": str(accuracy_csv),
        },
    }
    (out_dir / "deepseek_multi_vs_single_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("Wrote:")
    for value in manifest["outputs"].values():
        print(f"  {value}")


if __name__ == "__main__":
    main()
