"""Plot accuracy / brier / tw_score across all completed final_runs.

For each (scaffold, model) group, average across seeds (different run-folders or
different timestamps under the same run-folder) and shade the across-seed std.
Only runs whose daily_metrics.csv ends on END_DATE are included.
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_config  # noqa: F401  (applies project-wide science+serif style)

# Per-script overrides on top of the canonical style: this figure wants the
# scienceplots "grid" sheet and slightly smaller fonts than the default.
plt.style.use(plot_config.SCIENCE_STYLES + ["grid"])
plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "lines.linewidth": 2.4,
    }
)

DEFAULT_RUNS_DIR = Path("/fast/sgoel/forecasting/current_sim/final_runs")
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "plots" / "final_runs"
END_DATE = "2026-03-30"

# Run-folder name fragments to exclude (e.g. models that don't update predictions).
EXCLUDE_RUN_PREFIXES = ("opencode_aljazeeraQ12026_glm47",)

# Run folders to include even if their daily_metrics.csv doesn't reach END_DATE.
INCLUDE_INCOMPLETE_PREFIXES = (
    "opencode_aljazeeraQ12026_gemini3_pro",
)

# Mapping from run-folder name fragments to clean labels.
SCAFFOLD_LABELS = {
    "claude_code": "Claude Code",
    "opencode": "OpenCode",
    "ds_active_mem": "AllQ",
    "codex": "Codex",
}

# Maps the agents[0].model string from config.json to a clean label.
MODEL_LABELS = {
    "claude-opus-4-6": "claude-opus-4.6",
    "deepseek/deepseek-v3.2": "deepseek-v3.2",
    "z-ai/glm-4.7": "glm-4.7",
    "google/gemini-3-flash-preview": "gemini-3-flash",
    "google/gemini-3.1-pro-preview": "gemini-3-pro",
    "qwen/qwen3.6-plus": "qwen3.6-plus",
    "gpt-5.5": "gpt-5.5",
}

SEED_RE = re.compile(r"_r\d+$")


def scaffold_for_run(run_dir_name: str) -> str:
    name = SEED_RE.sub("", run_dir_name)
    scaffold_key = next(
        (k for k in sorted(SCAFFOLD_LABELS, key=len, reverse=True) if name.startswith(k)),
        None,
    )
    if scaffold_key is None:
        raise ValueError(f"Unknown scaffold prefix in run name: {run_dir_name}")
    return SCAFFOLD_LABELS[scaffold_key]


def model_label_from_config(cfg: dict, run_dir_name: str) -> str:
    agents = cfg.get("agents") or []
    raw = agents[0].get("model") if agents else None
    if not raw:
        return run_dir_name
    return MODEL_LABELS.get(raw, raw)


def collect_completed_runs(runs_dir: Path) -> list[dict]:
    """Find every timestamp dir whose daily_metrics.csv ends on END_DATE."""
    runs = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if any(run_dir.name.startswith(p) for p in EXCLUDE_RUN_PREFIXES):
            continue
        for ts_dir in sorted(run_dir.iterdir()):
            metrics_path = ts_dir / "daily_metrics.csv"
            if not metrics_path.is_file() or metrics_path.stat().st_size == 0:
                continue
            df = pd.read_csv(metrics_path)
            if df.empty:
                continue
            last_date = str(df["date"].iloc[-1])
            allow_partial = any(
                run_dir.name.startswith(p) for p in INCLUDE_INCOMPLETE_PREFIXES
            )
            if last_date != END_DATE and not allow_partial:
                continue
            cfg_path = ts_dir / "config.json"
            cfg = {}
            if cfg_path.is_file():
                with cfg_path.open() as f:
                    cfg = json.load(f)
            start_date = cfg.get("start_date")
            scaffold_label = scaffold_for_run(run_dir.name)
            model_label = model_label_from_config(cfg, run_dir.name)
            runs.append(
                {
                    "run_dir": run_dir.name,
                    "ts_dir": ts_dir.name,
                    "scaffold": scaffold_label,
                    "model": model_label,
                    "group_key": f"{model_label} ({scaffold_label})",
                    "start_date": start_date,
                    "df": df,
                }
            )
    return runs


def aggregate_by_group(runs: list[dict], metric: str, start_date_filter: str | None):
    """Return dict[group_key] -> (dates, mean, std, n_seeds)."""
    by_group: dict[str, list[pd.DataFrame]] = {}
    for r in runs:
        df = r["df"].copy()
        df["date"] = pd.to_datetime(df["date"])
        if start_date_filter is not None:
            df = df[df["date"] >= pd.to_datetime(start_date_filter)]
        # Multi-agent runs would have multiple agent_ids; final_runs are all single-agent
        # but average across agent_ids defensively.
        daily = df.groupby("date")[metric].mean().reset_index()
        by_group.setdefault(r["group_key"], []).append(daily)

    out = {}
    for group_key, frames in by_group.items():
        merged = frames[0][["date"]].copy()
        for i, fr in enumerate(frames):
            merged = merged.merge(fr.rename(columns={metric: f"v_{i}"}), on="date", how="outer")
        merged = merged.sort_values("date").reset_index(drop=True)
        value_cols = [c for c in merged.columns if c.startswith("v_")]
        values = merged[value_cols].to_numpy(dtype=float)
        mean = np.nanmean(values, axis=1)
        std = np.nanstd(values, axis=1, ddof=0) if values.shape[1] > 1 else np.zeros(len(merged))
        out[group_key] = (merged["date"].to_numpy(), mean, std, values.shape[1])
    return out


METRIC_LABELS = {
    "accuracy": "Accuracy (%)",
    "avg_brier": "Average Brier Score",
    "tw_score": "Time-Weighted Score",
}

# Distinct, perceptually-balanced palette (Tableau 10-style).
PALETTE = [
    "#4C78A8",  # blue
    "#F58518",  # orange
    "#54A24B",  # green
    "#E45756",  # red
    "#72B7B2",  # teal
    "#B279A2",  # purple
    "#FF9DA6",  # pink
    "#9D755D",  # brown
]


def plot_metric(grouped, metric: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))

    keys = sorted(grouped.keys())
    for i, key in enumerate(keys):
        dates, mean, std, n = grouped[key]
        color = PALETTE[i % len(PALETTE)]
        ax.plot(dates, mean, color=color, label=key, solid_capstyle="round")
        if n > 1:
            ax.fill_between(
                dates,
                mean - std,
                mean + std,
                color=color,
                alpha=0.18,
                linewidth=0,
            )

    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    tick_dates = pd.to_datetime(
        ["2026-01-01", "2026-01-15", "2026-02-01", "2026-02-15",
         "2026-03-01", "2026-03-15", "2026-03-30"]
    )
    ax.set_xticks(tick_dates)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    ax.margins(x=0.01)
    ax.tick_params(axis="both", which="both", direction="out", length=4)

    ncol = min(len(keys), 3)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=ncol,
        borderaxespad=0,
        handlelength=2.0,
        columnspacing=1.6,
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
        help="Filter each run's metrics to start at config.start_date (skip warmup days).",
    )
    args = ap.parse_args()

    runs = collect_completed_runs(args.runs_dir)
    if not runs:
        raise SystemExit(f"No completed runs ending on {END_DATE} found under {args.runs_dir}")

    print(f"Found {len(runs)} completed run timestamps:")
    for r in runs:
        print(f"  {r['group_key']:<40s}  {r['run_dir']}/{r['ts_dir']}")

    # Use the latest start_date among included runs as the warmup cutoff if requested.
    start_filter = None
    if args.from_start_date:
        starts = [r["start_date"] for r in runs if r["start_date"]]
        start_filter = max(starts) if starts else None
        print(f"Filtering metrics from start_date >= {start_filter}")

    for metric in ("accuracy", "avg_brier", "tw_score"):
        grouped = aggregate_by_group(runs, metric, start_filter)
        out_path = args.output_dir / f"{metric}.png"
        plot_metric(grouped, metric, out_path)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
