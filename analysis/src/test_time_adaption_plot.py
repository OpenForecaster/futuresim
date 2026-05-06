"""Test-time adaption plots for active-memory2 bootstrap runs.

The selected runs all use the same fixed warmup bootstrap. This script chooses
the latest timestamp directory inside each run folder, validates that its
config points at ``fixedWarmup``, saves the metrics used for plotting, and
writes presentation-style metric curves similar to ``mem_vs_nomem_plot.py``.
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from plot_config import color_for_label, style_axes  # noqa: F401


DEFAULT_FINAL_RUNS_DIR = Path(
    os.getenv(
        "FSIM_FINAL_RUNS_V37",
        str(Path(os.getenv("FSIM_OUTPUT_BASE", "/fast/sgoel/forecasting/current_sim")) / "final_runs_v37"),
    )
)
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "plots" / "test_time_adaption"
TIMESTAMP_RE = re.compile(r"^\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class RunSpec:
    label: str
    run_dir: str
    timestamp_dir: str | None = None


RUNS: tuple[RunSpec, ...] = (
    RunSpec(
        label="deepseek-v4-pro (Claude Code)",
        run_dir="claude_code_aljazeeraQ12026v37_deepseek_v4_pro_activemem2_bootstrap_r00",
    ),
    RunSpec(
        label="glm-5.1 (Claude Code)",
        run_dir="claude_code_aljazeeraQ12026v37_glm51_activemem2_bootstrap_r00_restart",
        timestamp_dir="26-05-05-15-03-28",
    ),
    RunSpec(
        label="claude-opus-4.6 (Claude Code)",
        run_dir="claude_code_aljazeeraQ12026v37_opus_activemem2_bootstrap_retry_feb14_r00_restart",
    ),
    RunSpec(
        label="gpt-5.5 (Codex)",
        run_dir="codex_aljazeeraQ12026v37_gpt55_activemem2_bootstrap_r00",
    ),
)

NEURIPS_MODEL_RUNS: tuple[RunSpec, ...] = (
    RunSpec(
        label="deepseek-v4-pro (Claude Code)",
        run_dir="ds-v4-pro",
    ),
    RunSpec(
        label="glm-5.1 (Claude Code)",
        run_dir="glm-5.1",
    ),
    RunSpec(
        label="claude-opus-4.6 (Claude Code)",
        run_dir="opus-4.6",
    ),
    RunSpec(
        label="gpt-5.5 (Codex)",
        run_dir="gpt-5.5",
    ),
    RunSpec(
        label="qwen-3.6-plus (OpenCode)",
        run_dir="qwen-3.6-plus",
    ),
)

WARMUP_COMPONENT_RUN_DIRS = (
    "qwen36plus_allq_warmup_only_aljazeeraQ12026v37_r00",
    "qwen36plus_allq_warmup_only_aljazeeraQ12026v37_q307616_r00",
)

METRIC_LABELS = {
    "accuracy": "Accuracy (%)",
    "avg_brier": "Brier Skill Score (↑)",
    "tw_score": "Time-Weighted Score",
    "exp_acc": "Expected Accuracy",
    "total_predictions": "Total Predictions",
    "daily_submissions": "Daily Submissions",
    "avg_submission_tv_to_prev": "Avg Submission TV to Previous",
}

PRIMARY_METRICS = ("accuracy", "avg_brier")
BASELINE_COLOR = "#9A9A9A"
BASELINE_LINESTYLE = (0, (18, 10))
BASELINE_LINEWIDTH = 1.2


def _canonical_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _latest_timestamp_dir(run_dir: Path) -> Path:
    candidates = sorted(
        p for p in run_dir.iterdir() if p.is_dir() and TIMESTAMP_RE.match(p.name)
    )
    if not candidates:
        raise SystemExit(f"No timestamp dirs found under {run_dir}")
    return candidates[-1]


def _latest_nonempty_metrics_dir(run_dir: Path, metric_file: str) -> Path:
    candidates = sorted(
        (p for p in run_dir.iterdir() if p.is_dir() and TIMESTAMP_RE.match(p.name)),
        reverse=True,
    )
    for ts_dir in candidates:
        metrics_path = ts_dir / metric_file
        if not metrics_path.is_file() or metrics_path.stat().st_size == 0:
            continue
        try:
            df = pd.read_csv(metrics_path)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        if "total_predictions" in df.columns and df["total_predictions"].max() <= 0:
            continue
        return ts_dir
    raise SystemExit(f"No non-empty {metric_file} found under {run_dir}")


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _bootstrap_dirs(config: dict) -> list[str]:
    dirs: list[str] = []
    defaults = config.get("defaults")
    if isinstance(defaults, dict) and defaults.get("bootstrap_dir"):
        dirs.append(str(defaults["bootstrap_dir"]))
    for agent in config.get("agents") or []:
        if isinstance(agent, dict) and agent.get("bootstrap_dir"):
            dirs.append(str(agent["bootstrap_dir"]))
    return dirs


def _validate_bootstrap(ts_dir: Path, expected_bootstrap: Path) -> tuple[str, list[str]]:
    expected = _canonical_path(expected_bootstrap)
    sources: list[str] = []
    raw_dirs: list[str] = []

    for name in ("config.json", "source_config.json"):
        path = ts_dir / name
        if not path.is_file():
            continue
        config = _read_json(path)
        for bootstrap_dir in _bootstrap_dirs(config):
            raw_dirs.append(bootstrap_dir)
            actual = _canonical_path(bootstrap_dir)
            if actual != expected:
                raise SystemExit(
                    f"{ts_dir}/{name} has bootstrap_dir={bootstrap_dir}, "
                    f"expected {expected_bootstrap}"
                )
            sources.append(name)

    if not raw_dirs:
        raise SystemExit(f"No bootstrap_dir found in config files under {ts_dir}")

    return ";".join(sorted(set(raw_dirs))), sorted(set(sources))


def _load_metrics(ts_dir: Path, metric_file: str) -> pd.DataFrame:
    metrics_path = ts_dir / metric_file
    if not metrics_path.is_file() or metrics_path.stat().st_size == 0:
        raise SystemExit(f"Missing non-empty {metric_file}: {metrics_path}")
    df = pd.read_csv(metrics_path, parse_dates=["date"])
    if df.empty:
        raise SystemExit(f"Empty metrics file: {metrics_path}")
    return df.sort_values("date").reset_index(drop=True)


def _prediction_map_from_json(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["question_id"]): row["outcomes"] for row in rows}


def _prediction_map_from_actions(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("type") == "prediction":
                out[str(record["question_id"])] = record["outcomes"]
    return out


def build_fixed_warmup_anchor(
    runs_dir: Path, bootstrap_dir: Path, metric_file: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one shared day0 metrics row derived from fixedWarmup's sources."""
    fixed_predictions = _prediction_map_from_json(bootstrap_dir / "prediction.json")
    component_rows: list[pd.Series] = []
    component_summary: list[dict] = []
    combined_predictions: dict[str, dict] = {}

    for run_name in WARMUP_COMPONENT_RUN_DIRS:
        ts_dir = _latest_nonempty_metrics_dir(runs_dir / run_name, metric_file)
        df = _load_metrics(ts_dir, metric_file)
        row = df.iloc[-1].copy()
        component_rows.append(row)

        actions_path = ts_dir / "actions.jsonl"
        predictions = _prediction_map_from_actions(actions_path)
        combined_predictions.update(predictions)
        component_summary.append(
            {
                "component_run_dir": run_name,
                "timestamp_dir": ts_dir.name,
                "metrics_path": str(ts_dir / metric_file),
                "actions_path": str(actions_path),
                "n_predictions": len(predictions),
                "metric_date": row["date"].date().isoformat()
                if hasattr(row["date"], "date")
                else str(row["date"]),
            }
        )

    if combined_predictions != fixed_predictions:
        missing = sorted(set(fixed_predictions) - set(combined_predictions))
        extra = sorted(set(combined_predictions) - set(fixed_predictions))
        changed = sorted(
            qid
            for qid in fixed_predictions.keys() & combined_predictions.keys()
            if fixed_predictions[qid] != combined_predictions[qid]
        )
        raise SystemExit(
            "Warmup component predictions do not match fixedWarmup/prediction.json "
            f"(missing={len(missing)}, extra={len(extra)}, changed={len(changed)})"
        )

    total_predictions = int(sum(float(row["total_predictions"]) for row in component_rows))
    total_daily_submissions = int(
        sum(float(row.get("daily_submissions", 0.0)) for row in component_rows)
    )
    total_tw_score = sum(float(row["tw_score"]) for row in component_rows)
    correct_count = sum(
        round(float(row["accuracy"]) / 100.0 * float(row["total_predictions"]))
        for row in component_rows
    )
    exp_acc = (
        sum(float(row["exp_acc"]) * float(row["total_predictions"]) for row in component_rows)
        / total_predictions
    )

    anchor = {
        "date": pd.to_datetime(min(pd.to_datetime(row["date"]) for row in component_rows)),
        "agent_id": "fixedWarmup",
        "avg_brier": total_tw_score / (100.0 * total_predictions),
        "tw_score": total_tw_score,
        "accuracy": correct_count / total_predictions * 100.0,
        "exp_acc": exp_acc,
        "total_predictions": total_predictions,
        "daily_submissions": total_daily_submissions,
        "avg_submission_tv_to_prev": 0.0,
    }
    return pd.DataFrame([anchor]), pd.DataFrame(component_summary)


def prepend_anchor(df: pd.DataFrame, anchor_df: pd.DataFrame, agent_id: str) -> pd.DataFrame:
    anchor = anchor_df.copy()
    anchor["agent_id"] = agent_id
    if not df.empty and pd.to_datetime(df["date"]).min() <= anchor["date"].iloc[0]:
        return df
    return pd.concat([anchor, df], ignore_index=True).sort_values("date").reset_index(drop=True)


def _slug(label: str) -> str:
    return (
        label.lower()
        .replace("(", "")
        .replace(")", "")
        .replace(".", "")
        .replace("-", "_")
        .replace(" ", "_")
    )


def collect_runs(
    runs_dir: Path,
    run_specs: tuple[RunSpec, ...],
    bootstrap_dir: Path,
    metric_file: str,
    anchor_df: pd.DataFrame | None,
) -> tuple[list[dict], pd.DataFrame]:
    records: list[dict] = []
    summary_rows: list[dict] = []

    for spec in run_specs:
        run_path = runs_dir / spec.run_dir
        if not run_path.is_dir():
            raise SystemExit(f"Missing run dir: {run_path}")

        if spec.timestamp_dir is None:
            ts_dir = _latest_timestamp_dir(run_path)
        else:
            ts_dir = run_path / spec.timestamp_dir
            if not ts_dir.is_dir():
                raise SystemExit(f"Missing timestamp dir: {ts_dir}")
        bootstrap_raw, bootstrap_sources = _validate_bootstrap(ts_dir, bootstrap_dir)
        df = _load_metrics(ts_dir, metric_file)
        raw_first_metric_date = df["date"].min().date().isoformat()

        config = _read_json(ts_dir / "config.json") if (ts_dir / "config.json").is_file() else {}
        agents = config.get("agents") or []
        model = agents[0].get("model") if agents and isinstance(agents[0], dict) else None
        metrics_path = ts_dir / metric_file
        if anchor_df is not None:
            agent_id = str(df["agent_id"].iloc[0]) if "agent_id" in df.columns else spec.label
            df = prepend_anchor(df, anchor_df, agent_id)

        enriched = df.copy()
        enriched.insert(0, "label", spec.label)
        enriched.insert(1, "run_dir", spec.run_dir)
        enriched.insert(2, "timestamp_dir", ts_dir.name)
        enriched.insert(3, "metrics_path", str(metrics_path))

        records.append(
            {
                "spec": spec,
                "ts_dir": ts_dir,
                "df": df,
                "enriched_df": enriched,
            }
        )

        row = {
            "label": spec.label,
            "model": model,
            "run_dir": spec.run_dir,
            "timestamp_dir": ts_dir.name,
            "metrics_path": str(metrics_path),
            "config_path": str(ts_dir / "config.json") if (ts_dir / "config.json").is_file() else "",
            "source_config_path": str(ts_dir / "source_config.json")
            if (ts_dir / "source_config.json").is_file()
            else "",
            "bootstrap_dir": bootstrap_raw,
            "bootstrap_config_sources": ";".join(bootstrap_sources),
            "config_start_date": config.get("start_date"),
            "config_end_date": config.get("end_date"),
            "restart_from": config.get("restart_from"),
            "restart_from_day": config.get("restart_from_day"),
            "raw_first_metric_date": raw_first_metric_date,
            "first_metric_date": df["date"].min().date().isoformat(),
            "last_metric_date": df["date"].max().date().isoformat(),
            "n_metric_days": len(df),
        }
        for metric in [c for c in METRIC_LABELS if c in df.columns]:
            row[f"first_{metric}"] = df[metric].iloc[0]
            row[f"last_{metric}"] = df[metric].iloc[-1]
        summary_rows.append(row)

    return records, pd.DataFrame(summary_rows)


def choose_run_specs(runs_dir: Path) -> tuple[RunSpec, ...]:
    if all((runs_dir / spec.run_dir).is_dir() for spec in RUNS):
        return RUNS
    if all((runs_dir / spec.run_dir).is_dir() for spec in NEURIPS_MODEL_RUNS):
        return NEURIPS_MODEL_RUNS
    return RUNS


def choose_anchor_runs_dir(runs_dir: Path) -> Path:
    if all((runs_dir / run_name).is_dir() for run_name in WARMUP_COMPONENT_RUN_DIRS):
        return runs_dir
    return DEFAULT_FINAL_RUNS_DIR


def save_metrics(
    records: list[dict],
    summary: pd.DataFrame,
    output_dir: Path,
    anchor_df: pd.DataFrame | None,
    anchor_sources: pd.DataFrame | None,
) -> list[str]:
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    combined = pd.concat([r["enriched_df"] for r in records], ignore_index=True)
    combined_path = metrics_dir / "daily_metrics_all_runs.csv"
    combined.to_csv(combined_path, index=False)
    written.append(str(combined_path))

    summary_path = metrics_dir / "run_summary.csv"
    summary.to_csv(summary_path, index=False)
    written.append(str(summary_path))

    if anchor_df is not None:
        anchor_path = metrics_dir / "fixed_warmup_day0_metrics.csv"
        anchor_df.to_csv(anchor_path, index=False)
        written.append(str(anchor_path))
    if anchor_sources is not None:
        anchor_sources_path = metrics_dir / "fixed_warmup_day0_sources.csv"
        anchor_sources.to_csv(anchor_sources_path, index=False)
        written.append(str(anchor_sources_path))

    for record in records:
        spec = record["spec"]
        slug = _slug(spec.label)
        per_run_path = metrics_dir / f"{slug}_daily_metrics.csv"
        record["enriched_df"].to_csv(per_run_path, index=False)
        written.append(str(per_run_path))

        test_path = record["ts_dir"] / "test_daily_metrics.csv"
        if test_path.is_file() and test_path.stat().st_size > 0:
            test_df = pd.read_csv(test_path, parse_dates=["date"])
            test_df.insert(0, "label", spec.label)
            test_df.insert(1, "run_dir", spec.run_dir)
            test_df.insert(2, "timestamp_dir", record["ts_dir"].name)
            test_out = metrics_dir / f"{slug}_test_daily_metrics.csv"
            test_df.to_csv(test_out, index=False)
            written.append(str(test_out))

    return written


def _daily_series(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return df.groupby("date", as_index=False)[metric].mean().sort_values("date")


def plot_metric(records: list[dict], metric: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    style_axes(ax)

    for record in records:
        spec = record["spec"]
        df = _daily_series(record["df"], metric)
        ax.plot(
            df["date"],
            df[metric],
            color=color_for_label(spec.label),
            linewidth=3.0,
            solid_capstyle="round",
            label=spec.label,
        )

    if metric == "avg_brier":
        ax.axhline(
            0.0,
            color=BASELINE_COLOR,
            linestyle=BASELINE_LINESTYLE,
            linewidth=BASELINE_LINEWIDTH,
            zorder=1,
        )

    ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=13, length=5)
    ax.minorticks_off()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    min_date = min(record["df"]["date"].min() for record in records)
    max_date = max(record["df"]["date"].max() for record in records)
    tick_dates = pd.to_datetime(
        ["2026-01-01", "2026-02-01", "2026-03-01", "2026-03-28"]
    )
    tick_dates = [d for d in tick_dates if min_date <= d <= max_date]
    if tick_dates:
        ax.set_xticks(tick_dates)
    ax.set_xlim(min_date, max_date)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=12)

    ax.legend(loc="lower right", fontsize=11, frameon=False, handlelength=2.4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {output_path}")


def plot_combined(records: list[dict], metrics: list[str], output_path: Path) -> None:
    ncols = 2
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.0, 3.3 * nrows), sharex=False)
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, metric in zip(axes_list, metrics):
        style_axes(ax)
        for record in records:
            spec = record["spec"]
            df = _daily_series(record["df"], metric)
            ax.plot(
                df["date"],
                df[metric],
                color=color_for_label(spec.label),
                linewidth=1.8,
                solid_capstyle="round",
                label=spec.label,
            )
        if metric == "avg_brier":
            ax.axhline(
                0.0,
                color=BASELINE_COLOR,
                linestyle=BASELINE_LINESTYLE,
                linewidth=BASELINE_LINEWIDTH,
                zorder=1,
            )
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=11)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.tick_params(axis="both", which="major", labelsize=9, length=3)
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    for ax in axes_list[len(metrics):]:
        ax.axis("off")

    handles, labels = axes_list[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 1.01),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_dir", type=Path, default=DEFAULT_FINAL_RUNS_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--bootstrap_dir", type=Path, default=None)
    parser.add_argument("--metric_file", default="daily_metrics.csv")
    parser.add_argument(
        "--no_prepend_day0",
        action="store_true",
        help="Do not prepend the shared fixedWarmup 2025-12-24 metrics anchor.",
    )
    args = parser.parse_args()

    runs_dir = args.runs_dir
    bootstrap_dir = args.bootstrap_dir or (runs_dir / "fixedWarmup")
    run_specs = choose_run_specs(runs_dir)

    anchor_df = None
    anchor_sources = None
    if not args.no_prepend_day0:
        anchor_runs_dir = choose_anchor_runs_dir(runs_dir)
        anchor_df, anchor_sources = build_fixed_warmup_anchor(
            anchor_runs_dir, bootstrap_dir, args.metric_file
        )

    records, summary = collect_runs(
        runs_dir,
        run_specs,
        bootstrap_dir,
        args.metric_file,
        anchor_df,
    )
    written = save_metrics(records, summary, args.output_dir, anchor_df, anchor_sources)

    print("Selected runs:")
    for row in summary.itertuples(index=False):
        print(
            f"  {row.label}: {row.run_dir}/{row.timestamp_dir} "
            f"({row.first_metric_date} to {row.last_metric_date})"
        )
    print(f"Saved {len(written)} metric file(s) under {args.output_dir / 'metrics'}")

    available_metrics = [m for m in METRIC_LABELS if all(m in r["df"].columns for r in records)]
    for metric in PRIMARY_METRICS:
        if metric in available_metrics:
            plot_metric(records, metric, args.output_dir / f"{metric}.png")

    plot_combined(records, available_metrics, args.output_dir / "combined_metrics.png")


if __name__ == "__main__":
    main()
