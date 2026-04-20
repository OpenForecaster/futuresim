"""Render the BasicAgent day-start feedback string.

Uses a mocked `feedback_data` dict fed directly into FeedbackHandler.format_feedback.
Renders both single-agent (show_tw_peer=False) and multi-agent (show_tw_peer=True)
variants so the difference is visible.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agents.basicAgent.feedback import FeedbackHandler  # noqa: E402


def main() -> None:
    handler = FeedbackHandler(agent_id="mock")

    feedback_data = {
        "resolved_header": "## RESULTS SINCE YOUR LAST SESSION (2025-04-30 -> 2025-05-01)",
        "resolved_today": [
            {
                "qid": "q101",
                "title": "Will the UN Security Council pass a Gaza ceasefire resolution by April 30, 2025?",
                "my_pred_outcome": "No",
                "my_pred_prob": 0.70,
                "my_pred_distribution": {"No": 0.70, "Yes": 0.30},
                "ground_truth": "No",
                "brier": 0.58,
                "tw_peer": 0.12,
            },
            {
                "qid": "q247",
                "title": "Where will the funeral of Pope Francis be held?",
                "my_pred_outcome": "St. Peter's Square",
                "my_pred_prob": 0.70,
                "my_pred_distribution": {
                    "St. Peter's Square": 0.70,
                    "Basilica of St. John Lateran": 0.10,
                    "Santa Maria Maggiore": 0.10,
                    "St. Peter's Basilica": 0.10,
                },
                "ground_truth": "St. Peter's Square",
                "brier": 0.85,
                "tw_peer": 0.23,
            },
            {
                "qid": "q312",
                "title": "Will the S&P 500 close above 5500 on April 30, 2025?",
                "my_pred_outcome": "Yes",
                "my_pred_prob": 0.55,
                "my_pred_distribution": {"Yes": 0.55, "No": 0.45},
                "ground_truth": "No",
                "brier": -0.21,
                "tw_peer": -0.08,
            },
        ],
        "metrics": {
            "total_predictions": 108,
            "num_resolved": 3,
            "accuracy": 66.7,
            "avg_brier": 0.407,
            "tw_peer_score": 0.27,
        },
    }

    out = REPO_ROOT / "basic_agent_feedback.txt"
    parts = []

    parts.append("=" * 80)
    parts.append("SINGLE-AGENT MODE (show_tw_peer=False)  — used when single_agent_mode=True")
    parts.append("=" * 80)
    parts.append("")
    parts.append(handler.format_feedback(feedback_data, show_tw_peer=False))
    parts.append("")
    parts.append("")
    parts.append("=" * 80)
    parts.append("MULTI-AGENT MODE (show_tw_peer=True)   — used when single_agent_mode=False")
    parts.append("=" * 80)
    parts.append("")
    parts.append(handler.format_feedback(feedback_data, show_tw_peer=True))
    parts.append("")

    text = "\n".join(parts)
    out.write_text(text)
    print(f"Wrote {len(text):,} chars to {out}")


if __name__ == "__main__":
    main()
