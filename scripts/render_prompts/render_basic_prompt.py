"""Render day-0 + memory-update prompts end-to-end for configs/memory/active.yaml.

The active.yaml config uses scaffold="allQ" with active memory and a restart
from shared warmup, so day 0 (= start_date) enters `AllQAgent._build_instructions`,
which calls the BasicAgent builder and appends the post-warmup update reminder.

When the agent calls `next_day()`, the action loop (because this config sets
`memory_update_max_total_tokens=40960`) transitions to the memory-update phase
and injects `_build_active_memory_prompt(current_date, day_qids)` as the next
user message.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agents.allQAgent.agent import AllQAgent  # noqa: E402
from agents.basicAgent.config import AgentConfig  # noqa: E402


def make_agent() -> tuple[AllQAgent, date]:
    config = AgentConfig(
        max_actions=None,
        warmup_max_actions=None,
        max_total_tokens=163840,
        warmup_max_total_tokens=40960,
        submit_reserve_tokens=4096,
        force_submit_threshold_tokens=8192,
        max_outcomes_per_question=5,
        enable_memory=True,
        memory_format="active",
        memory_max_entries=500,
        memory_update_max_total_tokens=40960,
        search_enabled=True,
        max_search_results=5,
        snippet_max_chars=2000,
        article_max_chars=4000,
        search_cutoff_days=0,
        timegap_days=1,
        sampling_params={"temperature": 0.7, "max_tokens": 2048},
    )

    inference = MagicMock()
    inference.model_name = "deepseek/deepseek-v3.2"
    inference.chat_json = MagicMock()
    inference.enable_tools = True

    start_date = date(2025, 5, 1)
    agent = AllQAgent(
        agent_id="allQ_deepseek-v3.2_001",
        inference_provider=inference,
        config=config,
        model_name="deepseek/deepseek-v3.2",
        search_tool=None,
        start_date=start_date,
    )
    agent.warmed_up = True

    active_qids = [f"q{i:04d}" for i in range(312)]
    fi = SimpleNamespace(
        source_name="openforesight",
        source_context="",
        num_agents=1,
        last_active_date=date(2025, 4, 25),
        next_active_date=date(2025, 5, 2),
        resolution_events=[],
        histories={},
        questions={qid: None for qid in active_qids},
    )
    predicted = {qid: {"outcomes": {"Yes": 0.5}} for qid in active_qids[:298]}
    fi.get_agent_predictions = lambda agent_id: predicted
    agent._forecast_interface = fi

    query_handler = MagicMock()
    query_handler.get_info.return_value = {
        "n_rows": 312,
        "n_active": 312,
        "n_resolved": 0,
        "columns": [
            "qid", "title", "background", "resolution_criteria", "answer_type",
            "resolution_date", "is_resolved", "ground_truth", "market_aggregate",
            "num_predictions", "options", "my_prediction", "my_prediction_date",
        ],
        "columns_desc": "\n".join([
            "- qid (str) (Question ID)",
            "- title (str) (Question Content)",
            "- background (object)",
            "- resolution_criteria (object)",
            "- answer_type (object)",
            "- resolution_date (object)",
            "- is_resolved (bool)",
            "- ground_truth (float64)",
            "- market_aggregate (object)",
            "- num_predictions (int64)",
            "- options (object)",
            "- my_prediction (object)",
            "- my_prediction_date (object)",
        ]),
    }
    agent._query_handler = query_handler

    search_handler = MagicMock()
    search_handler.is_available = True
    search_handler.chunk_tokens = 512
    search_handler.count_articles.return_value = 187_432
    agent._search_handler = search_handler

    return agent, start_date


def build_day0_prompt(agent: AllQAgent, current_date: date) -> str:
    return agent._build_instructions(current_date)


def build_next_day_prompt(agent: AllQAgent, current_date: date) -> str:
    # Simulate the state at the moment the agent calls `next_day()`:
    # - feedback has been generated earlier in the turn (no resolutions on day 0)
    # - day_qids tracks the qids the agent interacted with today
    agent._last_feedback_data = {
        "resolved_today": [],
        "resolved_header": "## RESULTS SINCE YOUR LAST SESSION (2025-04-25 -> 2025-05-01)",
        "metrics": {"total_predictions": 298, "num_resolved": 0, "accuracy": 0.0,
                    "avg_brier": 0.0, "tw_peer_score": 0.0},
    }
    day_qids = {"q0300", "q0301"}
    return agent._build_active_memory_prompt(current_date, day_qids=day_qids)


if __name__ == "__main__":
    agent, start_date = make_agent()

    day0 = build_day0_prompt(agent, start_date)
    day0_path = REPO_ROOT / "basic_agent_day0_prompt.txt"
    day0_path.write_text(day0)
    print(f"Wrote {len(day0):,} chars to {day0_path}")

    next_day = build_next_day_prompt(agent, start_date)
    next_day_path = REPO_ROOT / "basic_agent_next_day_prompt.txt"
    next_day_path.write_text(next_day)
    print(f"Wrote {len(next_day):,} chars to {next_day_path}")
