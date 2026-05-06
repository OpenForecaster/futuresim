"""Plot accuracy / brier / tw_score across all completed final_runs_v37.

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
        "legend.fontsize": 11,
        "lines.linewidth": 2.4,
    }
)

DEFAULT_RUNS_DIR = Path("/fast/sgoel/forecasting/current_sim/final_runs_v37")
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "plots" / "final_runs_v37"
END_DATE = "2026-03-28"

# Skip these models entirely (across every scaffold).
EXCLUDED_MODELS = {"deepseek-v4-flash", "gpt-5.3-spark", "glm-4.7"}

# When the same model appears under multiple scaffolds, keep only the
# better-performing one. For deepseek-v4-pro, Claude Code wins on mean Brier
# and tw_score (-0.0479 vs -0.0576, -1448 vs -1805), so drop the OpenCode pair.
EXCLUDED_GROUPS = {"deepseek-v4-pro (OpenCode)"}

# Run-folder name fragments to exclude.
EXCLUDE_RUN_PREFIXES = ()

# Active-memory variants are excluded from the main figure — we plot the
# baseline scaffolds only. Any run-folder name containing one of these
# substrings is skipped.
EXCLUDE_RUN_SUBSTRS = ("activemem", "active_mem")

SCAFFOLD_PREFIX_RULES = [
    ("claude_code_", "Claude Code"),
    ("opencode_", "OpenCode"),
    ("codex_", "Codex"),
]

# Maps the agents[0].model string from config.json to a clean label.
MODEL_LABELS = {
    "claude-opus-4-6": "claude-opus-4.6",
    "claude-opus-4-7": "claude-opus-4.7",
    "claude-sonnet-4-6": "claude-sonnet-4.6",
    "deepseek-v4-pro": "deepseek-v4-pro",
    "deepseek/deepseek-v4-pro": "deepseek-v4-pro",
    "deepseek/deepseek-v4-flash": "deepseek-v4-flash",
    "glm-4.7": "glm-4.7",
    "glm-5": "glm-5",
    "glm-5.1": "glm-5.1",
    "gpt-5.3-codex-spark": "gpt-5.3-spark",
    "gpt-5.4": "gpt-5.4",
    "gpt-5.5": "gpt-5.5",
}


def scaffold_for_run(run_dir_name: str) -> str:
    rules = sorted(SCAFFOLD_PREFIX_RULES, key=lambda kv: -len(kv[0]))
    for prefix, label in rules:
        if run_dir_name.startswith(prefix):
            return label
    raise ValueError(f"Unknown scaffold prefix in run name: {run_dir_name}")


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
        if any(s in run_dir.name for s in EXCLUDE_RUN_SUBSTRS):
            continue
        for ts_dir in sorted(run_dir.iterdir()):
            metrics_path = ts_dir / "daily_metrics.csv"
            if not metrics_path.is_file() or metrics_path.stat().st_size == 0:
                continue
            df = pd.read_csv(metrics_path)
            if df.empty:
                continue
            last_date = str(df["date"].iloc[-1])
            if last_date != END_DATE:
                continue
            cfg_path = ts_dir / "config.json"
            cfg = {}
            if cfg_path.is_file():
                with cfg_path.open() as f:
                    cfg = json.load(f)
            start_date = cfg.get("start_date")
            try:
                scaffold_label = scaffold_for_run(run_dir.name)
            except ValueError as e:
                print(f"  skip (unknown scaffold): {run_dir.name}")
                continue
            model_label = model_label_from_config(cfg, run_dir.name)
            if model_label in EXCLUDED_MODELS:
                continue
            group_key = f"{model_label} ({scaffold_label})"
            if group_key in EXCLUDED_GROUPS:
                continue
            runs.append(
                {
                    "run_dir": run_dir.name,
                    "ts_dir": ts_dir.name,
                    "scaffold": scaffold_label,
                    "model": model_label,
                    "group_key": group_key,
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


# -- soft-ceiling computation (any-run-got-it-right per question) -------------
def _normalize(s: str) -> str:
    return (s or "").strip().lower()


def _load_matcher_cache(p: Path) -> dict:
    cache: dict = {}
    if not p.exists():
        return cache
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return cache
    for k, v in raw.items():
        try:
            parts = json.loads(k)
            if isinstance(parts, list) and len(parts) == 4:
                cache[(str(parts[0]), str(parts[1]), str(parts[2]), str(parts[3]))] = bool(v)
        except Exception:
            continue
    return cache


def _load_matcher_log(p: Path) -> dict:
    cache: dict = {}
    if not p.exists():
        return cache
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            inp = e.get("input") or {}
            out = e.get("output") or {}
            if "is_equivalent" not in out:
                continue
            cache[(
                _normalize(inp.get("predicted", "")),
                _normalize(inp.get("ground_truth", "")),
                str(inp.get("question_id") or "None"),
                _normalize(inp.get("question_title", "") or ""),
            )] = bool(out["is_equivalent"])
    return cache


def _is_equiv(pred: str, gt: str, qid: str, title: str, cache: dict):
    pn, gn = _normalize(pred), _normalize(gt)
    if pn == gn:
        return True
    qid_s = str(qid) if qid else "None"
    return cache.get(
        (pn, gn, qid_s, _normalize(title)),
        cache.get((pn, gn, qid_s, ""), None),
    )


def compute_soft_ceilings(runs_dir: Path) -> dict:
    """Soft ceilings across every (run, agent) under runs_dir.

    - accuracy: % of questions where some (run, agent)'s top-1 outcome
      semantically matches ground truth (last pre-resolution prediction).
    - avg_brier: per-question max raw_brier across all (run, agent) pairs,
      averaged across questions.
    """
    global_cache: dict = {}
    titles: dict = {}
    gts: dict = {}
    # Per (qid, run, agent) -> last (sim_date, top_outcome) before resolution.
    last_top: dict = {}
    # Per (qid, run, agent) -> raw_brier from resolution event.
    brier_per_pair: dict = {}

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        for ts in sorted(run_dir.iterdir()):
            actions = ts / "actions.jsonl"
            if not actions.is_file():
                continue
            for k, v in _load_matcher_cache(ts / "matcher_cache.json").items():
                global_cache.setdefault(k, v)
            for k, v in _load_matcher_log(ts / "matcher.jsonl").items():
                global_cache.setdefault(k, v)
            mp = ts / "market.csv"
            if mp.is_file():
                with mp.open(encoding="utf-8") as f:
                    import csv
                    for r in csv.DictReader(f):
                        titles.setdefault(str(r["qid"]), r.get("title", "") or "")
            run_label = f"{run_dir.name}/{ts.name}"
            resolved_on: dict = {}
            with actions.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if e.get("type") == "resolution":
                        qid = str(e.get("question_id", ""))
                        if not qid:
                            continue
                        gts.setdefault(qid, e.get("ground_truth"))
                        resolved_on[qid] = e.get("sim_date")
                        for aid, b in (e.get("raw_brier") or {}).items():
                            brier_per_pair[(qid, run_label, aid)] = float(b)
            with actions.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if e.get("type") != "prediction":
                        continue
                    qid = str(e.get("question_id", ""))
                    aid = e.get("agent_id")
                    sd = e.get("sim_date")
                    if not qid or not aid or not sd:
                        continue
                    rdate = resolved_on.get(qid)
                    if rdate and sd >= rdate:
                        continue
                    outcomes = e.get("outcomes") or {}
                    if not outcomes:
                        continue
                    top = max(outcomes.items(), key=lambda kv: float(kv[1]))[0]
                    key = (qid, run_label, aid)
                    cur = last_top.get(key)
                    if cur is None or sd > cur[0]:
                        last_top[key] = (sd, top)

    correct_qs: set = set()
    for (qid, run_label, aid), (_sd, top) in last_top.items():
        gt = gts.get(qid)
        if gt is None:
            continue
        if _is_equiv(top, gt, qid, titles.get(qid, ""), global_cache):
            correct_qs.add(qid)

    best_brier_per_q: dict = {}
    for (qid, _run, _aid), b in brier_per_pair.items():
        cur = best_brier_per_q.get(qid)
        if cur is None or b > cur:
            best_brier_per_q[qid] = b

    n_q = len(gts)
    acc = 100.0 * len(correct_qs) / n_q if n_q else 0.0
    brier = (sum(best_brier_per_q.values()) / len(best_brier_per_q)) if best_brier_per_q else 0.0
    return {
        "n_questions": n_q,
        "n_correct_any": len(correct_qs),
        "accuracy": acc,
        "avg_brier": brier,
    }

# Distinct, perceptually-balanced palette (extended for v37's larger group set).
PALETTE = [
    "#4C78A8",  # blue
    "#F58518",  # orange
    "#54A24B",  # green
    "#E45756",  # red
    "#72B7B2",  # teal
    "#B279A2",  # purple
    "#FF9DA6",  # pink
    "#9D755D",  # brown
    "#BAB0AC",  # gray
    "#EECA3B",  # yellow
    "#1F77B4",  # darker blue
    "#2CA02C",  # darker green
    "#D62728",  # darker red
    "#9467BD",  # darker purple
    "#17BECF",  # cyan
    "#8C564B",  # darker brown
]


def plot_metric(grouped, metric: str, out_path: Path, ceiling: float | None = None,
                ceiling_label: str | None = None):
    fig, ax = plt.subplots(figsize=(11, 6.5))

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

    if ceiling is not None:
        ax.axhline(
            ceiling,
            color="black",
            linestyle="--",
            linewidth=1.6,
            alpha=0.85,
            label=ceiling_label or "soft ceiling",
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
    ap.add_argument(
        "--soft_ceiling",
        action="store_true",
        help="Compute and overlay 'any-run-got-it' soft ceilings on accuracy and Brier plots.",
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

    ceiling_for_metric: dict = {}
    if args.soft_ceiling:
        print("Computing soft ceilings across every run in", args.runs_dir)
        ceilings = compute_soft_ceilings(args.runs_dir)
        print(
            f"  soft accuracy ceiling: {ceilings['accuracy']:.2f}%  "
            f"({ceilings['n_correct_any']}/{ceilings['n_questions']} questions)"
        )
        print(f"  soft Brier ceiling:    {ceilings['avg_brier']:.4f}")
        ceiling_for_metric = {
            "accuracy": (ceilings["accuracy"], f"soft accuracy ceiling = {ceilings['accuracy']:.1f}%"),
            "avg_brier": (ceilings["avg_brier"], f"soft Brier ceiling = {ceilings['avg_brier']:.3f}"),
        }

    for metric in ("accuracy", "avg_brier"):
        grouped = aggregate_by_group(runs, metric, start_filter)
        out_path = args.output_dir / f"{metric}.png"
        c_val, c_label = ceiling_for_metric.get(metric, (None, None))
        plot_metric(grouped, metric, out_path, ceiling=c_val, ceiling_label=c_label)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
