"""Render the exact string the MinimalHarness agent receives as the return
value of the `next_day` MCP tool on a subsequent day.

Mirrors the assembly in agents/minimalHarnessAgent/mcp_server.py::next_day so
the output is byte-identical to what Claude Code sees in-session.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def build_next_day_response(
    cur_date: date,
    new_date: date,
    new_articles_count: int,
    events: List[Dict[str, Any]],
    agent_predictions: Dict[str, Dict[str, float]],
    agent_id: str,
    total_predictions: int,
    simulation_complete: bool = False,
) -> str:
    # Mirrors mcp_server._build_new_articles_message when a count is known.
    articles_msg = (
        f"{new_articles_count:,} new articles have been published since your last update "
        "and are available via the search_news tool. New date "
        "directories are also now present in articles/."
    )

    response_parts = [
        f"Day advanced to {new_date.isoformat()}.",
        articles_msg,
    ]

    # Cumulative accumulators (mirrors _cumulative in mcp_server).
    brier_sum = 0.0
    tw_peer_sum = 0.0
    accuracy_count = 0
    resolved_count = 0

    if events:
        response_parts.append(
            f"\n## RESULTS SINCE YOUR LAST SESSION ({cur_date.isoformat()} -> {new_date.isoformat()})\n"
        )
        for ev in events:
            qid = ev.get("qid", ev.get("question_id", "?"))
            title = ev.get("title", "")
            gt = ev.get("ground_truth", "?")

            agents_data = ev.get("agents", {})
            my_stats = agents_data.get(agent_id)

            if my_stats:
                brier = float(my_stats.get("brier", 0))
                tw_peer = float(my_stats.get("tw_peer", 0))
                best_outcome = my_stats.get("best_outcome", "?")
                best_prob = float(my_stats.get("best_prob", 0))
                is_accurate = my_stats.get("is_accurate", False)
                brier_sum += brier
                tw_peer_sum += tw_peer
                resolved_count += 1
                if is_accurate:
                    accuracy_count += 1

                pred_dist = agent_predictions.get(str(qid), {})
                if pred_dist:
                    dist_items = sorted(pred_dist.items(), key=lambda kv: -kv[1])
                    dist_str = ", ".join(f"{o}: {p:.2f}" for o, p in dist_items)
                    dist_str = "{" + dist_str + "}"
                else:
                    dist_str = f"{{{best_outcome}: {best_prob:.2f}}}"

                response_parts.append(
                    f"- \"{title}\"\n"
                    f"  Your prediction distribution: {dist_str} | Truth: {gt}\n"
                    f"  Brier: {brier:+.2f} | TW-Score: {tw_peer:+.2f}"
                )
            else:
                response_parts.append(f"- \"{title}\" → {gt}")

        if resolved_count > 0:
            avg_brier = brier_sum / resolved_count
            accuracy = accuracy_count / resolved_count * 100
            response_parts.append(
                f"\n## YOUR CUMULATIVE PERFORMANCE TILL TODAY\n"
                f"- Total Predictions: {total_predictions} ({resolved_count} resolved)\n"
                f"- accuracy: {accuracy:.1f}% | brier skill score: {avg_brier:.3f} | "
                f"time weighted score: {tw_peer_sum:.2f}\n"
                f"  accuracy = fraction of resolved questions where your top outcome "
                f"matched the truth; brier skill score = mean brier skill score across resolved questions; "
                f"time weighted score = sum of brier skill scores across all resolved questions, across all days you held your respective predictions"
            )

    if simulation_complete:
        response_parts.append("\nSimulation is complete. No more days to process.")

    return "\n".join(response_parts)


def main() -> None:
    agent_id = "cc_claude-opus-4-6_001"

    # Mock some realistic resolution events.
    events = [
        {
            "qid": "q101",
            "title": "Will the UN Security Council pass a Gaza ceasefire resolution by Sep 4, 2025?",
            "ground_truth": "No",
            "agents": {
                agent_id: {
                    "brier": 0.58,
                    "tw_peer": 0.12,
                    "best_outcome": "No",
                    "best_prob": 0.70,
                    "is_accurate": True,
                }
            },
        },
        {
            "qid": "q312",
            "title": "Will the S&P 500 close above 5500 on Sep 4, 2025?",
            "ground_truth": "No",
            "agents": {
                agent_id: {
                    "brier": -0.21,
                    "tw_peer": -0.08,
                    "best_outcome": "Yes",
                    "best_prob": 0.55,
                    "is_accurate": False,
                }
            },
        },
    ]

    agent_predictions = {
        "q101": {"No": 0.70, "Yes": 0.30},
        "q312": {"Yes": 0.55, "No": 0.45},
    }

    out = REPO_ROOT / "minimal_harness_next_day_prompt.txt"
    parts = []

    parts.append("=" * 80)
    parts.append("CASE A: Sep 1 -> Sep 2, no resolutions yet (minimal output)")
    parts.append("=" * 80)
    parts.append("")
    parts.append(
        build_next_day_response(
            cur_date=date(2025, 9, 1),
            new_date=date(2025, 9, 2),
            new_articles_count=12345,
            events=[],
            agent_predictions={},
            agent_id=agent_id,
            total_predictions=108,
        )
    )
    parts.append("")
    parts.append("")
    parts.append("=" * 80)
    parts.append("CASE B: Sep 4 -> Sep 5, two resolutions (full feedback output)")
    parts.append("=" * 80)
    parts.append("")
    parts.append(
        build_next_day_response(
            cur_date=date(2025, 9, 4),
            new_date=date(2025, 9, 5),
            new_articles_count=45678,
            events=events,
            agent_predictions=agent_predictions,
            agent_id=agent_id,
            total_predictions=108,
        )
    )
    parts.append("")

    text = "\n".join(parts)
    out.write_text(text)
    print(f"Wrote {len(text):,} chars to {out}")


if __name__ == "__main__":
    main()
