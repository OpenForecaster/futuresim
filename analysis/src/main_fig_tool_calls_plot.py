"""Tool-calls-per-day curve for every model in the main figure.

Mirrors `main_fig_plot.py` (same GROUPS, same per-group seed pooling) but
plots a different signal: per-day tool-call count parsed from the agent's
harness stdout (Codex / Claude Code / OpenCode have three different log
shapes, all unified to "list[int] per day").

Day boundaries are inferred from `mcp__forecast__next_day` (or the codex
v2 `turn.started`) — every harness emits exactly one `next_day` per
simulated day, so the tool-call counts are aligned 1:1 with the
`daily_metrics.csv` dates of the same run.
"""

import argparse
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from main_fig_plot import (
    GROUPS,
    collect_runs_for_group,
    DEFAULT_RUNS_DIR,
)
from plot_config import (
    color_for_label,
    style_axes,
)

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "plots" / "main_fig"


# ---------------------------------------------------------------------------
# Per-harness tool-call counters. Each returns a list[int] of length = #days,
# i.e. the number of tool calls the agent issued on day k (0-indexed). Day
# boundaries are inferred from the `next_day` MCP call.
# ---------------------------------------------------------------------------

def _split_at_next_day(events: list[tuple[str, bool]]) -> list[int]:
    """Walk a flat list of (tool_name, is_tool_call) and return per-day counts.

    A `next_day` event itself is *excluded* from the day's count and acts as
    the boundary. Any residual tool calls after the final `next_day` are
    attributed to one trailing day.
    """
    counts: list[int] = []
    cur = 0
    for name, is_call in events:
        if name == "next_day":
            counts.append(cur)
            cur = 0
            continue
        if is_call:
            cur += 1
    if cur > 0:
        counts.append(cur)
    return counts


def _codex_per_day(stdout: Path) -> list[int]:
    """Codex has two log formats; we autodetect."""
    has_turn_started = False
    with stdout.open() as f:
        for line in f:
            if '"type":"turn.started"' in line:
                has_turn_started = True
                break

    if has_turn_started:
        # New format: turn.started marks each day; count item.completed events
        # of inner type mcp_tool_call/command_execution between turns.
        counts: list[int] = []
        cur = 0
        started = False
        with stdout.open() as f:
            for line in f:
                o = json.loads(line)
                t = o.get("type")
                if t == "turn.started":
                    if started:
                        counts.append(cur)
                    cur = 0
                    started = True
                elif t == "item.completed" and started:
                    inner = o.get("item", {}).get("type")
                    if inner in ("mcp_tool_call", "command_execution"):
                        cur += 1
        if started:
            counts.append(cur)
        return counts

    # Old format: response_item events with payload.type in
    # {function_call, custom_tool_call, mcp_tool_call}. Use next_day name as
    # day boundary.
    events: list[tuple[str, bool]] = []
    with stdout.open() as f:
        for line in f:
            o = json.loads(line)
            if o.get("type") != "response_item":
                continue
            payload = o.get("payload", {})
            ptype = payload.get("type")
            if ptype not in ("function_call", "custom_tool_call", "mcp_tool_call"):
                continue
            name = (payload.get("name") or "").lower()
            # `next_day` may appear as a custom_tool_call; treat as boundary,
            # not as a counted call.
            events.append((name, True))
    return _split_at_next_day(events)


def _claude_code_per_day(stdout: Path) -> list[int]:
    """Tool uses live as content blocks inside `assistant` events."""
    events: list[tuple[str, bool]] = []
    with stdout.open() as f:
        for line in f:
            o = json.loads(line)
            if o.get("type") != "assistant":
                continue
            for block in o.get("message", {}).get("content", []) or []:
                if block.get("type") != "tool_use":
                    continue
                name = (block.get("name") or "").lower()
                short = name.rsplit("__", 1)[-1]  # mcp__forecast__next_day -> next_day
                events.append((short, True))
    return _split_at_next_day(events)


def _opencode_per_day(stdout: Path) -> list[int]:
    """OpenCode emits flat `tool_use` events with the name in `part.tool`."""
    events: list[tuple[str, bool]] = []
    with stdout.open() as f:
        for line in f:
            o = json.loads(line)
            if o.get("type") != "tool_use":
                continue
            name = (o.get("part", {}).get("tool") or "").lower()
            short = name.rsplit("__", 1)[-1]
            events.append((short, True))
    return _split_at_next_day(events)


HARNESS_PARSERS = {
    "codex_stdout.jsonl": _codex_per_day,
    "claude_code_stdout.jsonl": _claude_code_per_day,
    "opencode_stdout.jsonl": _opencode_per_day,
}


def count_tool_calls_per_day(agent_dir: Path) -> list[int] | None:
    for fname, parser in HARNESS_PARSERS.items():
        p = agent_dir / fname
        if p.exists():
            return parser(p)
    return None


# ---------------------------------------------------------------------------
# Aggregation: per-group mean ± std of per-day tool counts, aligned to dates.
# ---------------------------------------------------------------------------

def aggregate_tool_calls(runs: list[dict], runs_dir: Path):
    """For each group label, return (dates_array, mean, std, n_seeds)."""
    by_label: dict[str, list[pd.DataFrame]] = {}
    for r in runs:
        ts_path = runs_dir / r["run_dir"] / r["ts_dir"]
        agents = list((ts_path / "agents").iterdir())
        if len(agents) != 1:
            print(f"  skip {r['run_dir']}/{r['ts_dir']}: {len(agents)} agents")
            continue
        counts = count_tool_calls_per_day(agents[0])
        if not counts:
            print(f"  skip {r['run_dir']}/{r['ts_dir']}: no stdout / 0 days")
            continue
        df = r["df"].copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        n = min(len(counts), len(df))
        if n < len(counts) or n < len(df):
            print(f"  trim {r['run_dir']}/{r['ts_dir']}: counts={len(counts)} "
                  f"days={len(df)} -> n={n}")
        per_day = pd.DataFrame({"date": df["date"].iloc[:n].values,
                                "tool_calls": counts[:n]})
        by_label.setdefault(r["label"], []).append(per_day)

    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
    for label, frames in by_label.items():
        merged = frames[0][["date"]].copy()
        for i, fr in enumerate(frames):
            merged = merged.merge(
                fr.rename(columns={"tool_calls": f"v_{i}"}), on="date", how="outer"
            )
        merged = merged.sort_values("date").reset_index(drop=True)
        value_cols = [c for c in merged.columns if c.startswith("v_")]
        values = merged[value_cols].to_numpy(dtype=float)
        mean = np.nanmean(values, axis=1)
        std = (np.nanstd(values, axis=1, ddof=0)
               if values.shape[1] > 1 else np.zeros(len(merged)))
        out[label] = (merged["date"].to_numpy(), mean, std, values.shape[1])
    return out


# ---------------------------------------------------------------------------
# Plot.
# ---------------------------------------------------------------------------

def plot(grouped, out_path: Path, log_y: bool = False, skip_day0: bool = True,
         smooth_window: int = 7) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    style_axes(ax)

    keys = [g.label for g in GROUPS if g.label in grouped]
    for key in keys:
        dates, mean, std, n = grouped[key]
        if skip_day0 and len(dates) > 1:
            dates, mean, std = dates[1:], mean[1:], std[1:]
        if smooth_window > 1:
            mean = pd.Series(mean).rolling(smooth_window, min_periods=1,
                                           center=True).mean().to_numpy()
            std = pd.Series(std).rolling(smooth_window, min_periods=1,
                                         center=True).mean().to_numpy()
        color = color_for_label(key)
        ax.plot(dates, mean, color=color, linewidth=1.8, solid_capstyle="round")
        if n > 1:
            ax.fill_between(dates, mean - std, mean + std,
                            color=color, alpha=0.18, linewidth=0)

    if log_y:
        ax.set_yscale("log")
        ax.set_ylabel("Tool calls per day (log)")
    else:
        ax.set_ylabel("Tool calls per day")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    tick_dates = pd.to_datetime(
        ["2026-01-01", "2026-01-15", "2026-02-01", "2026-02-15",
         "2026-03-01", "2026-03-15", "2026-03-28"]
    )
    ax.set_xticks(tick_dates)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    ax.margins(x=0.01)
    ax.tick_params(axis="both", which="major", direction="out", length=4)
    ax.tick_params(axis="y", right=False)
    ax.tick_params(axis="x", top=False)
    ax.minorticks_off()

    legend_handles = [
        mlines.Line2D([], [], color=color_for_label(k), marker="s",
                      markersize=7, linestyle="None", label=k)
        for k in keys
    ]
    ncol = max(1, -(-len(legend_handles) // 2))
    ax.legend(handles=legend_handles, loc="upper right",
              bbox_to_anchor=(0.98, 0.98), bbox_transform=ax.transAxes,
              ncol=ncol, frameon=False, fontsize=12,
              handletextpad=0.4, columnspacing=0.5, borderaxespad=0.0)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=Path, default=DEFAULT_RUNS_DIR)
    ap.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--include_day0", action="store_true",
                    help="Include day 0 (warmup) instead of starting at day 1.")
    ap.add_argument("--smooth_window", type=int, default=7,
                    help="Rolling-mean window in days (1 = no smoothing).")
    args = ap.parse_args()

    DEEPSEEK_LABEL = "deepseek-v4-pro (Claude Code)"
    DEEPSEEK_KEEP = 3
    all_runs: list[dict] = []
    for spec in GROUPS:
        rs = collect_runs_for_group(args.runs_dir, spec)
        if spec.label == DEEPSEEK_LABEL and len(rs) > DEEPSEEK_KEEP:
            rs = rs[:DEEPSEEK_KEEP]
        print(f"{spec.label}: {len(rs)} seed(s)")
        all_runs.extend(rs)

    if not all_runs:
        raise SystemExit("No runs matched.")

    grouped = aggregate_tool_calls(all_runs, args.runs_dir)
    out_path = args.output_dir / "tool_calls.png"
    plot(grouped, out_path,
         skip_day0=not args.include_day0,
         smooth_window=args.smooth_window)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
