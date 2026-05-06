"""Per-model bar plots: total tool calls and total uncached tokens.

Reuses the same seed selection as `main_fig_plot.py` (curated allowlist +
per-model selection rules: deepseek min-variance triple, claude-opus latest).

For each surviving seed we compute:
  - total tool calls   : sum of per-day counts (parsers from
                         `main_fig_tool_calls_plot`).
  - total tokens       : "unique" tokens excluding cache-read input tokens.
                         Definition is per-harness:
                           codex      : token_usage.total_tokens - cached_input_tokens
                           claude_code: Σ usage.(input + cache_creation + output)
                           opencode   : Σ tokens.(input - cache.read + output + reasoning)

Across-seed aggregation: mean (bar height) ± std (error bar).
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from main_fig_plot import (
    GROUPS,
    DEFAULT_RUNS_DIR,
    collect_runs_for_group,
    pick_min_variance_seeds,
)
from main_fig_tool_calls_plot import count_tool_calls_per_day
from plot_config import color_for_label, style_axes

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "plots" / "main_fig"

DEEPSEEK_LABEL = "deepseek-v4-pro (Claude Code)"
DEEPSEEK_KEEP = 3
CLAUDE_OPUS_LABEL = "claude-opus-4.6 (Claude Code)"


# ---------------------------------------------------------------------------
# Per-harness "unique tokens" (uncached) counters.
# ---------------------------------------------------------------------------

def _codex_tokens(agent_dir: Path) -> int | None:
    p = agent_dir / "token_usage.json"
    if not p.is_file():
        return None
    with p.open() as f:
        u = json.load(f).get("total_token_usage", {})
    total = u.get("total_tokens")
    cached = u.get("cached_input_tokens", 0)
    if total is None:
        return None
    return int(total) - int(cached)


def _claude_code_tokens(agent_dir: Path) -> int | None:
    p = agent_dir / "claude_code_stdout.jsonl"
    if not p.is_file():
        return None
    total = 0
    with p.open() as f:
        for line in f:
            o = json.loads(line)
            if o.get("type") != "assistant":
                continue
            u = o.get("message", {}).get("usage", {}) or {}
            total += int(u.get("input_tokens", 0) or 0)
            total += int(u.get("cache_creation_input_tokens", 0) or 0)
            total += int(u.get("output_tokens", 0) or 0)
    return total


def _opencode_tokens(agent_dir: Path) -> int | None:
    p = agent_dir / "opencode_stdout.jsonl"
    if not p.is_file():
        return None
    total = 0
    with p.open() as f:
        for line in f:
            o = json.loads(line)
            if o.get("type") not in ("step_finish", "step-finish"):
                continue
            tk = o.get("part", {}).get("tokens", {}) or {}
            inp = int(tk.get("input", 0) or 0)
            out = int(tk.get("output", 0) or 0)
            reas = int(tk.get("reasoning", 0) or 0)
            cache_read = int((tk.get("cache") or {}).get("read", 0) or 0)
            total += max(0, inp - cache_read) + out + reas
    return total


def count_unique_tokens(agent_dir: Path) -> int | None:
    if (agent_dir / "token_usage.json").is_file():
        return _codex_tokens(agent_dir)
    if (agent_dir / "claude_code_stdout.jsonl").is_file():
        return _claude_code_tokens(agent_dir)
    if (agent_dir / "opencode_stdout.jsonl").is_file():
        return _opencode_tokens(agent_dir)
    return None


# ---------------------------------------------------------------------------
# Per-seed -> per-label aggregation.
# ---------------------------------------------------------------------------

def per_seed_totals(runs: list[dict], runs_dir: Path):
    """Return dict[label] -> {'tool_calls': [...], 'tokens': [...]}."""
    out: dict[str, dict[str, list[float]]] = {}
    for r in runs:
        ts_path = runs_dir / r["run_dir"] / r["ts_dir"]
        agents = list((ts_path / "agents").iterdir())
        if len(agents) != 1:
            print(f"  skip {r['run_dir']}/{r['ts_dir']}: {len(agents)} agents")
            continue
        agent_dir = agents[0]
        tc_per_day = count_tool_calls_per_day(agent_dir)
        tc_total = sum(tc_per_day) if tc_per_day else None
        tok_total = count_unique_tokens(agent_dir)
        bucket = out.setdefault(r["label"], {"tool_calls": [], "tokens": []})
        if tc_total is not None:
            bucket["tool_calls"].append(float(tc_total))
        if tok_total is not None:
            bucket["tokens"].append(float(tok_total))
    return out


# ---------------------------------------------------------------------------
# Plot.
# ---------------------------------------------------------------------------

def _format_count(v: float) -> str:
    if v >= 1e9:
        return f"{v / 1e9:.1f}B"
    if v >= 1e6:
        return f"{v / 1e6:.1f}M"
    if v >= 1e3:
        return f"{v / 1e3:.1f}k"
    return f"{v:.0f}"


def plot_bar(label_to_values: dict[str, list[float]], ylabel: str,
             out_path: Path, title: str | None = None) -> None:
    # Order bars by GROUPS order, dropping any with no usable data
    # (empty list, or all-zero — local proxies don't surface usage).
    def _has_data(vs: list[float]) -> bool:
        return bool(vs) and any(v > 0 for v in vs)

    ordered = [g.label for g in GROUPS
               if g.label in label_to_values and _has_data(label_to_values[g.label])]
    means = [float(np.mean(label_to_values[k])) for k in ordered]
    stds = [float(np.std(label_to_values[k], ddof=0)) if len(label_to_values[k]) > 1 else 0.0
            for k in ordered]
    colors = [color_for_label(k) for k in ordered]
    short = [k.split(" (")[0] for k in ordered]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    style_axes(ax)
    x = np.arange(len(ordered))
    bars = ax.bar(x, means, yerr=stds, color=colors, edgecolor="white",
                  linewidth=1.0, capsize=4, error_kw={"elinewidth": 1.2,
                                                       "ecolor": "#444444"})

    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=15, ha="right", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.tick_params(axis="y", labelsize=12)
    ax.minorticks_off()
    ax.margins(x=0.04)

    if title:
        ax.set_title(title, fontsize=14)

    ymax = max((m + s) for m, s in zip(means, stds)) if means else 1.0
    pad = 0.04 * ymax
    for xi, m, s in zip(x, means, stds):
        ax.text(xi, m + s + pad, _format_count(m), ha="center", va="bottom",
                fontsize=12, color="#222222")
    ax.set_ylim(0, ymax * 1.18)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=Path, default=DEFAULT_RUNS_DIR)
    ap.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    all_runs: list[dict] = []
    for spec in GROUPS:
        rs = collect_runs_for_group(args.runs_dir, spec)
        if spec.label == DEEPSEEK_LABEL and len(rs) > DEEPSEEK_KEEP:
            rs = pick_min_variance_seeds(rs, DEEPSEEK_KEEP, "avg_brier")
        elif spec.label == CLAUDE_OPUS_LABEL and len(rs) > 1:
            rs = [rs[-1]]
        print(f"{spec.label}: {len(rs)} seed(s)")
        all_runs.extend(rs)

    if not all_runs:
        raise SystemExit("No runs matched.")

    by_label = per_seed_totals(all_runs, args.runs_dir)

    plot_bar({k: v["tool_calls"] for k, v in by_label.items()},
             ylabel="Total tool calls",
             out_path=args.output_dir / "tool_calls_total.png")
    plot_bar({k: v["tokens"] for k, v in by_label.items()},
             ylabel="Total tokens (excluding input cache)",
             out_path=args.output_dir / "tokens_total.png")


if __name__ == "__main__":
    main()
