"""Overlay tool-call count and accuracy over time for a single agent run.

Tool calls are extracted from `codex_stdout.jsonl`: each `turn.started` event
marks the start of a simulated day; between two turn.started events we count
`item.completed` events whose inner `item.type` is `command_execution` or
`mcp_tool_call`. Days are aligned to the dates in `daily_metrics.csv`
(turn N -> daily_metrics row N), which we also use for accuracy.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_config import style_axes  # noqa: F401 -- side-effects + style helpers

DEFAULT_RUN = Path(
    "/fast/sgoel/forecasting/current_sim/final_runs_v37/"
    "codex_aljazeeraQ12026v37_gpt55_resume/26-04-29-21-43-12"
)
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "plots" / "tool_calls_vs_accuracy"

TOOL_CALL_TYPES = {"command_execution", "mcp_tool_call"}


def count_tool_calls_per_turn(stdout_path: Path) -> list[int]:
    """Return tool-call counts per turn (one entry per turn.started event)."""
    counts: list[int] = []
    current = 0
    started = False
    with stdout_path.open() as f:
        for line in f:
            o = json.loads(line)
            t = o.get("type")
            if t == "turn.started":
                if started:
                    counts.append(current)
                current = 0
                started = True
            elif t == "item.completed" and started:
                inner = o.get("item", {}).get("type")
                if inner in TOOL_CALL_TYPES:
                    current += 1
    if started:
        counts.append(current)
    return counts


def find_agent_dir(run_dir: Path) -> Path:
    agents = list((run_dir / "agents").iterdir())
    if len(agents) != 1:
        raise SystemExit(
            f"Expected exactly 1 agent under {run_dir}/agents, found {len(agents)}"
        )
    return agents[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    agent_dir = find_agent_dir(run_dir)
    stdout_path = agent_dir / "codex_stdout.jsonl"
    metrics_path = run_dir / "daily_metrics.csv"

    metrics = pd.read_csv(metrics_path, parse_dates=["date"]).sort_values("date")
    tool_counts = count_tool_calls_per_turn(stdout_path)

    if len(tool_counts) != len(metrics):
        print(
            f"WARN: turn count ({len(tool_counts)}) != metric rows ({len(metrics)});"
            f" truncating to min."
        )
    n = min(len(tool_counts), len(metrics))
    # Skip day 0 (warmup): Δacc is undefined there and tool-call count is an
    # outlier (full-question sweep). Start from day 1 so both axes are
    # comparable across the rest of the run.
    dates = metrics["date"].iloc[1:n].to_numpy()
    delta_acc = metrics["accuracy"].iloc[:n].diff().iloc[1:].to_numpy()
    tool_counts = tool_counts[1:n]

    fig, ax_acc = plt.subplots(figsize=(11, 5.5))
    style_axes(ax_acc)
    ax_tc = ax_acc.twinx()

    acc_color = "#1f77b4"
    tc_color = "#d62728"

    ax_acc.plot(dates, delta_acc, color=acc_color, linewidth=2.0,
                label="Δ accuracy (% per day)")
    ax_acc.axhline(0.0, color="#9A9A9A", linestyle=":", linewidth=1.0, zorder=1)
    ax_acc.set_ylabel("Daily change in accuracy (%)", color=acc_color)
    ax_acc.tick_params(axis="y", labelcolor=acc_color)

    ax_tc.plot(dates, tool_counts, color=tc_color, linewidth=1.6,
               linestyle="--", label="tool calls / day")
    ax_tc.set_ylabel("Tool calls per day", color=tc_color)
    ax_tc.tick_params(axis="y", labelcolor=tc_color)
    # Right spine should be visible (style_axes hides right spines on the
    # primary axis; we want it back for the second y-axis).
    ax_tc.spines["right"].set_visible(True)
    ax_tc.spines["right"].set_color(tc_color)

    ax_acc.set_xlabel("Date")
    fig.autofmt_xdate()

    # Combined legend.
    lines_acc, labels_acc = ax_acc.get_legend_handles_labels()
    lines_tc, labels_tc = ax_tc.get_legend_handles_labels()
    ax_acc.legend(lines_acc + lines_tc, labels_acc + labels_tc,
                  loc="lower right", frameon=False, fontsize=11)

    title = f"{agent_dir.name} ({run_dir.parent.name}/{run_dir.name})"
    ax_acc.set_title(title, fontsize=11)

    out_dir: Path = args.output_dir / run_dir.parent.name / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tool_calls_vs_accuracy.png"
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
