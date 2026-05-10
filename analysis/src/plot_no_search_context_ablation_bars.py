#!/usr/bin/env python3
"""Bar plot for GPT-5.5 search-context ablations with per-question standard errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
DEFAULT_OUT_DIR = Path("analysis/plots/no_search_context_ablation_bars")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal_dir", type=Path, default=DEFAULT_NORMAL)
    parser.add_argument("--no_search_update_dir", type=Path, default=DEFAULT_NO_SEARCH_UPDATE)
    parser.add_argument("--warmup_dir", type=Path, default=DEFAULT_WARMUP)
    parser.add_argument("--static_search_dir", type=Path, default=DEFAULT_STATIC)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR)
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


def norm(value: object) -> str:
    return str(value or "").strip().lower()


def load_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_matcher_equiv(run_dir: Path) -> Dict[tuple[str, str, str], bool]:
    equiv: Dict[tuple[str, str, str], bool] = {}
    matcher_jsonl = run_dir / "matcher.jsonl"
    if matcher_jsonl.exists():
        for rec in load_jsonl(matcher_jsonl):
            inp = rec.get("input") or {}
            out = rec.get("output") or {}
            if inp.get("type") != "check_guess":
                continue
            key = (
                norm(inp.get("predicted")),
                norm(inp.get("ground_truth")),
                str(inp.get("question_id")),
            )
            equiv[key] = bool(out.get("is_equivalent"))

    matcher_cache = run_dir / "matcher_cache.json"
    if matcher_cache.exists():
        raw = json.loads(matcher_cache.read_text())
        for key_text, value in raw.items():
            try:
                parts = json.loads(key_text)
            except json.JSONDecodeError:
                parts = key_text.split("|||")
            if len(parts) >= 3:
                equiv[(norm(parts[0]), norm(parts[1]), str(parts[2]))] = bool(value)
    return equiv


def top_outcome(outcomes: Mapping[str, object] | None) -> str | None:
    if not outcomes:
        return None
    return max(outcomes.items(), key=lambda kv: float(kv[1]))[0]


def is_equiv(predicted: str | None, truth: str, qid: str, equiv: Mapping[tuple[str, str, str], bool]) -> bool:
    if predicted is None:
        return False
    pred_norm = norm(predicted)
    truth_norm = norm(truth)
    if pred_norm == truth_norm:
        return True
    key = (pred_norm, truth_norm, str(qid))
    if key not in equiv:
        raise KeyError(f"Missing matcher decision for qid={qid!r}: {predicted!r} vs {truth!r}")
    return bool(equiv[key])


def ground_truths(actions: list[dict]) -> Dict[str, str]:
    return {
        str(rec["question_id"]): str(rec["ground_truth"])
        for rec in actions
        if rec.get("type") == "resolution"
    }


def correctness_at_date(run_dir: Path, sim_date: str) -> Dict[str, int]:
    actions = list(load_jsonl(run_dir / "actions.jsonl"))
    equiv = load_matcher_equiv(run_dir)
    truths = ground_truths(actions)
    current: Dict[str, dict] = {}
    for rec in actions:
        if rec.get("type") == "prediction" and str(rec.get("sim_date")) <= sim_date:
            current[str(rec["question_id"])] = rec.get("outcomes") or {}
    return {
        qid: int(is_equiv(top_outcome(current.get(qid)), truth, qid, equiv))
        for qid, truth in truths.items()
    }


def correctness_at_resolution(run_dir: Path) -> Dict[str, int]:
    actions = list(load_jsonl(run_dir / "actions.jsonl"))
    equiv = load_matcher_equiv(run_dir)
    truths = ground_truths(actions)
    current: Dict[str, dict] = {}
    correct: Dict[str, int] = {}
    for rec in actions:
        if rec.get("type") == "prediction":
            current[str(rec["question_id"])] = rec.get("outcomes") or {}
        elif rec.get("type") == "resolution":
            qid = str(rec["question_id"])
            correct[qid] = int(is_equiv(top_outcome(current.get(qid)), str(rec["ground_truth"]), qid, equiv))
    for qid in truths:
        correct.setdefault(qid, 0)
    return correct


def accuracy_and_se(values: np.ndarray) -> tuple[float, float]:
    point = float(values.mean() * 100.0)
    se = float(values.std(ddof=1) / np.sqrt(len(values)) * 100.0)
    return point, se


def assert_matches_csv(run_dir: Path, point: float, row_index: int) -> None:
    df = pd.read_csv(run_dir / "daily_metrics.csv")
    expected = float(df.iloc[row_index]["accuracy"])
    if abs(point - expected) > 0.015:
        raise ValueError(f"{run_dir}: reconstructed accuracy {point:.4f} != CSV {expected:.4f}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        {
            "group": "Simulation - Final Day",
            "label": "Daily context\nupdates",
            "short_label": "final_day_agentic_search",
            "color": "#08519c",
            "run_dir": args.normal_dir,
            "correct": correctness_at_resolution(args.normal_dir),
            "csv_row": -1,
        },
        {
            "group": "Simulation - Final Day",
            "label": "No context\nupdates",
            "short_label": "no_new_context",
            "color": "#D55E00",
            "run_dir": args.no_search_update_dir,
            "correct": correctness_at_resolution(args.no_search_update_dir),
            "csv_row": -1,
        },
        {
            "group": "Direct - One day before resolution",
            "label": "Agentic\nsearch",
            "short_label": "final_day_only_agentic_search",
            "color": "#009E73",
            "run_dir": args.warmup_dir,
            "correct": correctness_at_resolution(args.warmup_dir),
            "csv_row": -1,
        },
        {
            "group": "Direct - One day before resolution",
            "label": "Single search\nquery",
            "short_label": "final_day_only_no_agentic_search",
            "color": "#D62728",
            "run_dir": args.static_search_dir,
            "correct": correctness_at_resolution(args.static_search_dir),
            "csv_row": -1,
        },
    ]

    rows = []
    for spec in specs:
        values = np.array([spec["correct"][qid] for qid in sorted(spec["correct"])], dtype=float)
        if len(values) != 330:
            raise ValueError(f"{spec['short_label']} has {len(values)} questions, expected 330")
        point, se = accuracy_and_se(values)
        assert_matches_csv(spec["run_dir"], point, spec["csv_row"])
        rows.append(
            {
                "label": spec["label"],
                "group": spec["group"],
                "short_label": spec["short_label"],
                "accuracy": point,
                "standard_error": se,
                "n_questions": len(values),
                "n_correct": int(values.sum()),
                "run_dir": str(spec["run_dir"]),
                "color": spec["color"],
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.drop(columns=["color"]).to_csv(args.output_dir / "accuracy_bar_standard_error_summary.csv", index=False)

    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.2), sharey=True)
    y_max = max(out_df["accuracy"] + out_df["standard_error"]) + 6
    for ax, group in zip(axes, ["Simulation - Final Day", "Direct - One day before resolution"]):
        part = out_df[out_df["group"] == group].reset_index(drop=True)
        x = np.arange(len(part))
        y = part["accuracy"].to_numpy()
        ax.bar(
            x,
            y,
            yerr=part["standard_error"].to_numpy(),
            color=part["color"].tolist(),
            edgecolor="black",
            linewidth=0.8,
            error_kw={"elinewidth": 1.2, "capsize": 4, "capthick": 1.2, "ecolor": "#333333"},
        )
        title = "Direct - One day\nbefore resolution" if group.startswith("Direct") else group
        ax.set_title(title, loc="center", fontweight="bold", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(part["label"].tolist(), rotation=0, ha="center", fontsize=13)
        ax.tick_params(axis="y", labelsize=14)
        ax.set_ylim(0, y_max)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
        for idx, row in part.iterrows():
            ax.text(
                idx,
                row["accuracy"] + row["standard_error"] + 1.0,
                f"{row['accuracy']:.1f}",
                ha="center",
                va="bottom",
                fontsize=14,
            )
    axes[0].set_ylabel("Accuracy (%)", fontsize=17, labelpad=10)

    fig.tight_layout()
    out_base = args.output_dir / "accuracy_context_ablation_bars"
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(out_base.with_suffix(".png"))
    print(out_base.with_suffix(".pdf"))
    print(args.output_dir / "accuracy_bar_standard_error_summary.csv")
    print(out_df.drop(columns=["color"]).to_string(index=False))


if __name__ == "__main__":
    main()
