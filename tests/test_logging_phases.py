"""Tests that each action type produces exactly one log entry with the correct phase.

Verifies that the refactored logging (single log at the top of the loop
using parsed.action_type) does not double-log, and assigns the right phase
for query, search, submit, memory, mem_df, invalid, and next_day actions.
"""

import json
from datetime import date
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from agents.basicAgent import AgentConfig
from agents.qwenAgent import QwenBasicAgent
from agents.utils.memory import ActiveMemory, StructuredMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tc_response(name: str, arguments: Dict[str, Any], call_id: str = "call_1",
                 content: str = "", prompt_tokens: int = 100,
                 completion_tokens: int = 50) -> Dict[str, Any]:
    """Build a Chat Completions response with a single tool call."""
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": content,
                "tool_calls": [{
                    "type": "function",
                    "id": call_id,
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments),
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _no_tool_response(content: str = "No tool", prompt_tokens: int = 100,
                      completion_tokens: int = 50) -> Dict[str, Any]:
    return {
        "choices": [{
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class _DummyChatProvider:
    def chat_json(self, messages, sampling_params):
        raise RuntimeError("Should not be called directly")


class DummyForecastInterface:
    def __init__(self):
        self.submitted = []
        self.next_day_calls = 0

    def log_model_output(self, prompt, response, metadata):
        pass

    def next_day(self):
        self.next_day_calls += 1

    def get_market_csv_path(self):
        return None

    def submit_prediction(self, pred):
        self.submitted.append(pred)


class ScriptedAgent(QwenBasicAgent):
    """Scripted agent that captures _log_model_output calls."""

    def __init__(self, responses: List[Any], config: Optional[AgentConfig] = None,
                 memory=None):
        cfg = config or AgentConfig(
            enable_memory=False,
            max_total_tokens=50000,
        )
        super().__init__(
            agent_id="test_log",
            inference_provider=_DummyChatProvider(),
            config=cfg,
        )
        self._scripted_responses = list(responses)
        self._call_index = 0
        self.logged_phases: List[str] = []
        if memory is not None:
            self._memory = memory

    def _setup_day(self, forecast_interface, current_date):
        self._forecast_interface = forecast_interface

    def _build_qwen_instructions(self, current_date: date) -> str:
        return "Test prompt"

    def _build_start_budget_status(self, *, warmup=False, max_actions_override=None):
        return ""

    def _call_chat_json_with_retries(self, *, messages, tools, sampling_params):
        if self._call_index >= len(self._scripted_responses):
            raise RuntimeError("No more scripted responses")
        resp = self._scripted_responses[self._call_index]
        self._call_index += 1
        if isinstance(resp, Exception):
            raise resp
        return resp

    def _log_model_output(self, prompt, response, metadata=None):
        """Capture logged phases instead of writing to disk."""
        phase = metadata.get("phase") if metadata else None
        self.logged_phases.append(phase)


# ===========================================================================
# Tests
# ===========================================================================

class TestLoggingPhases:
    """Each action type should produce exactly one log entry with the correct phase."""

    def test_query_logged_once_as_query(self):
        responses = [
            _tc_response("query_df", {"code": "print(1)"}),
            _tc_response("next_day", {}),
        ]
        agent = ScriptedAgent(responses)
        fi = DummyForecastInterface()
        agent.act(None, fi, date(2025, 6, 1))

        assert agent.logged_phases.count("query") == 1
        assert "llm" not in agent.logged_phases

    def test_search_logged_once_as_search(self):
        responses = [
            _tc_response("search_news", {"query": "test"}),
            _tc_response("next_day", {}),
        ]
        agent = ScriptedAgent(responses)
        fi = DummyForecastInterface()
        agent.act(None, fi, date(2025, 6, 1))

        assert agent.logged_phases.count("search") == 1
        assert "llm" not in agent.logged_phases

    def test_submit_logged_once_as_submit(self):
        responses = [
            _tc_response("submit_forecasts", {
                "forecasts": [{"qid": "Q1", "outcomes": {"Yes": 0.7, "No": 0.3}}]
            }),
            _tc_response("next_day", {}),
        ]
        agent = ScriptedAgent(responses)
        fi = DummyForecastInterface()
        agent.act(None, fi, date(2025, 6, 1))

        assert agent.logged_phases.count("submit") == 1
        assert "llm" not in agent.logged_phases

    def test_no_tool_call_logged_as_llm(self):
        responses = [
            _no_tool_response("I'm thinking..."),
            _tc_response("next_day", {}),
        ]
        agent = ScriptedAgent(responses)
        fi = DummyForecastInterface()
        agent.act(None, fi, date(2025, 6, 1))

        assert agent.logged_phases.count("llm") == 1

    def test_next_day_not_double_logged(self):
        """next_day should only be logged via its dedicated handler, not by the blanket log."""
        responses = [
            _tc_response("next_day", {}),
        ]
        agent = ScriptedAgent(responses)
        fi = DummyForecastInterface()
        agent.act(None, fi, date(2025, 6, 1))

        # "next_day" has dedicated logging as "next_day" phase
        assert agent.logged_phases.count("next_day") == 1
        # Should NOT also appear as "llm" or "next"
        assert "llm" not in agent.logged_phases
        assert "next" not in agent.logged_phases

    def test_memory_retrieve_logged_once(self, tmp_path):
        """memory_retrieve in standalone memory update logs as memory_update with actions metadata."""
        responses = [
            _tc_response("memory_retrieve", {"name": "test-entry"}),
            _tc_response("next_day", {}),
        ]
        mem = StructuredMemory("test_log", memory_dir=str(tmp_path), max_entries=10)
        agent = ScriptedAgent(responses, config=AgentConfig(
            enable_memory=True, memory_format="structured",
            max_total_tokens=50000, memory_dir=str(tmp_path),
        ), memory=mem)
        fi = DummyForecastInterface()
        agent.act(None, fi, date(2025, 6, 1))

        # Routes through _prompt_memory_update → phase is "memory_update",
        # not the raw action type.  Exactly one log for the action turn.
        assert agent.logged_phases.count("memory_update") == 1
        assert "llm" not in agent.logged_phases

    def test_mem_add_logged_once(self, tmp_path):
        """mem_add in standalone memory update logs as memory_update with actions metadata."""
        responses = [
            _tc_response("mem_add", {"qid": "Q1", "memory": "note", "question": "test?"}),
            _tc_response("next_day", {}),
        ]
        mem = ActiveMemory("test_log", memory_dir=str(tmp_path), max_entries=10)
        mem.set_date(date(2025, 6, 1))
        agent = ScriptedAgent(responses, config=AgentConfig(
            enable_memory=True, memory_format="active",
            max_total_tokens=50000, memory_dir=str(tmp_path),
        ), memory=mem)
        fi = DummyForecastInterface()
        agent.act(None, fi, date(2025, 6, 1))

        # Routes through _prompt_memory_update → phase is "memory_update"
        assert agent.logged_phases.count("memory_update") == 1
        assert "llm" not in agent.logged_phases

    def test_invalid_tool_logged_once(self):
        """An unknown tool should be logged once with its action_type (None → 'llm')."""
        responses = [
            _tc_response("nonexistent_tool", {"x": 1}),
            _tc_response("next_day", {}),
        ]
        agent = ScriptedAgent(responses)
        fi = DummyForecastInterface()
        agent.act(None, fi, date(2025, 6, 1))

        # Unknown tool parses to action_type=None, so phase falls through to "llm"
        # The key assertion: it's logged exactly once, not twice
        total_non_next = [p for p in agent.logged_phases if p != "next_day"]
        assert len(total_non_next) == 1

    def test_mixed_actions_each_logged_once(self):
        """A session with query → search → submit → next_day: each logged exactly once."""
        responses = [
            _tc_response("query_df", {"code": "print(1)"}),
            _tc_response("search_news", {"query": "test"}),
            _tc_response("submit_forecasts", {
                "forecasts": [{"qid": "Q1", "outcomes": {"Yes": 0.7}}]
            }),
            _tc_response("next_day", {}),
        ]
        agent = ScriptedAgent(responses)
        fi = DummyForecastInterface()
        agent.act(None, fi, date(2025, 6, 1))

        assert agent.logged_phases.count("query") == 1
        assert agent.logged_phases.count("search") == 1
        assert agent.logged_phases.count("submit") == 1
        assert agent.logged_phases.count("next_day") == 1
        assert "llm" not in agent.logged_phases
        assert len(agent.logged_phases) == 4

    def test_content_filter_still_logged(self):
        """A content_filter response should still produce a log entry."""
        responses = [
            {
                "choices": [{
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "content_filter",
                }],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
            },
            _tc_response("next_day", {}),
        ]
        agent = ScriptedAgent(responses)
        fi = DummyForecastInterface()
        agent.act(None, fi, date(2025, 6, 1))

        # content_filter has no tool call, so parsed=None → phase="llm"
        assert "llm" in agent.logged_phases
