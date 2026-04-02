"""Comprehensive tests for log metadata produced by the agent loop.

Verifies that the single post-dispatch log call in _run_chat_tools_action_loop
and _run_memory_update correctly captures handler-level metadata (submitted_qids,
num_forecasts, dropped_forecasts, error, actions) without double-logging.
"""

import json
from datetime import date
from typing import Any, Dict, List, Optional

import pytest

from agents.basicAgent import AgentConfig
from agents.qwenAgent import QwenBasicAgent
from agents.utils.memory import ActiveMemory, StructuredMemory


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------

def _tc_response(
    name: str,
    arguments: Dict[str, Any],
    call_id: str = "call_1",
    content: str = "",
    finish_reason: str = "tool_calls",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
) -> Dict[str, Any]:
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
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _multi_tc_response(
    calls: List[Dict[str, Any]],
    content: str = "",
    finish_reason: str = "tool_calls",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
) -> Dict[str, Any]:
    """Build a Chat Completions response with multiple tool calls.

    Each entry in *calls* is {"name": ..., "arguments": ..., "id": ...}.
    """
    tool_calls = []
    for i, c in enumerate(calls):
        tool_calls.append({
            "type": "function",
            "id": c.get("id", f"call_{i}"),
            "function": {
                "name": c["name"],
                "arguments": json.dumps(c["arguments"]),
            },
        })
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            },
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _no_tool_response(
    content: str = "No tool",
    finish_reason: str = "stop",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
) -> Dict[str, Any]:
    return {
        "choices": [{
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

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


class MetadataCapturingAgent(QwenBasicAgent):
    """Scripted agent that captures the full metadata dict from every log call."""

    def __init__(
        self,
        responses: List[Any],
        config: Optional[AgentConfig] = None,
        memory=None,
    ):
        cfg = config or AgentConfig(
            enable_memory=False,
            max_total_tokens=50000,
        )
        super().__init__(
            agent_id="test_meta",
            inference_provider=_DummyChatProvider(),
            config=cfg,
        )
        self._scripted_responses = list(responses)
        self._call_index = 0
        self.log_entries: List[Dict[str, Any]] = []
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
        """Capture the full metadata dict (not just the phase)."""
        self.log_entries.append(dict(metadata) if metadata else {})

    # -- Helpers for tests --

    @property
    def phases(self) -> List[str]:
        return [e.get("phase") for e in self.log_entries]

    def entries_with_phase(self, phase: str) -> List[Dict[str, Any]]:
        return [e for e in self.log_entries if e.get("phase") == phase]


# ===========================================================================
# 1. Single-action metadata tests
# ===========================================================================


class TestSingleActionMetadata:
    """Verify metadata fields for each individual action type."""

    def test_submit_metadata_contains_submitted_qids(self):
        responses = [
            _tc_response("submit_forecasts", {
                "forecasts": [{"qid": "Q42", "outcomes": {"Yes": 0.8, "No": 0.2}}],
            }),
            _tc_response("next_day", {}),
        ]
        agent = MetadataCapturingAgent(responses)
        fi = DummyForecastInterface()
        agent.act(None, fi, date(2025, 6, 1))

        submit_entries = agent.entries_with_phase("submit")
        assert len(submit_entries) == 1
        meta = submit_entries[0]
        assert meta["submitted_qids"] == ["Q42"]
        assert meta["num_forecasts"] == 1
        assert meta["dropped_forecasts"] == 0
        assert meta["actions"] == ["submit"]

    def test_submit_metadata_dropped_forecasts(self):
        """When submitting multiple qids (only first kept), dropped_forecasts is recorded."""
        responses = [
            _tc_response("submit_forecasts", {
                "forecasts": [
                    {"qid": "Q1", "outcomes": {"Yes": 0.6, "No": 0.4}},
                    {"qid": "Q2", "outcomes": {"Yes": 0.5, "No": 0.5}},
                    {"qid": "Q3", "outcomes": {"A": 0.3, "B": 0.7}},
                ],
            }),
            _tc_response("next_day", {}),
        ]
        agent = MetadataCapturingAgent(responses)
        fi = DummyForecastInterface()
        agent.act(None, fi, date(2025, 6, 1))

        meta = agent.entries_with_phase("submit")[0]
        assert meta["num_forecasts"] == 1  # only first kept
        assert meta["dropped_forecasts"] == 2

    def test_submit_error_metadata(self):
        """A submit with no forecasts should record the error."""
        responses = [
            _tc_response("submit_forecasts", {"forecasts": []}),
            _tc_response("next_day", {}),
        ]
        agent = MetadataCapturingAgent(responses)
        fi = DummyForecastInterface()
        agent.act(None, fi, date(2025, 6, 1))

        meta = agent.entries_with_phase("submit")[0]
        assert "error" in meta
        assert meta["num_forecasts"] == 0

    def test_query_metadata(self):
        responses = [
            _tc_response("query_df", {"code": "print(len(df))"}),
            _tc_response("next_day", {}),
        ]
        agent = MetadataCapturingAgent(responses)
        fi = DummyForecastInterface()
        agent.act(None, fi, date(2025, 6, 1))

        query_entries = agent.entries_with_phase("query")
        assert len(query_entries) == 1
        assert query_entries[0]["actions"] == ["query"]

    def test_search_metadata(self):
        responses = [
            _tc_response("search_news", {"query": "election results"}),
            _tc_response("next_day", {}),
        ]
        agent = MetadataCapturingAgent(responses)
        fi = DummyForecastInterface()
        agent.act(None, fi, date(2025, 6, 1))

        search_entries = agent.entries_with_phase("search")
        assert len(search_entries) == 1
        assert search_entries[0]["actions"] == ["search"]

    def test_invalid_tool_metadata(self):
        responses = [
            _tc_response("nonexistent_tool", {"x": 1}),
            _tc_response("next_day", {}),
        ]
        agent = MetadataCapturingAgent(responses)
        fi = DummyForecastInterface()
        agent.act(None, fi, date(2025, 6, 1))

        # Unknown tools have no recognized action_type, but _turn_actions
        # still captures whatever single_call_to_parsed_action returned.
        non_next = [e for e in agent.log_entries if e.get("phase") != "next_day"]
        assert len(non_next) == 1

    def test_no_tool_logged_as_llm(self):
        responses = [
            _no_tool_response("thinking..."),
            _tc_response("next_day", {}),
        ]
        agent = MetadataCapturingAgent(responses)
        fi = DummyForecastInterface()
        agent.act(None, fi, date(2025, 6, 1))

        assert agent.phases.count("llm") == 1

    def test_content_filter_logged_as_llm(self):
        responses = [
            _no_tool_response("", finish_reason="content_filter"),
            _tc_response("next_day", {}),
        ]
        agent = MetadataCapturingAgent(responses)
        fi = DummyForecastInterface()
        agent.act(None, fi, date(2025, 6, 1))

        llm_entries = agent.entries_with_phase("llm")
        assert len(llm_entries) == 1
        assert llm_entries[0]["finish_reason"] == "content_filter"


# ===========================================================================
# 2. No double-logging
# ===========================================================================


class TestNoDoubleLogs:
    """Each LLM turn should produce exactly one log entry."""

    def test_query_not_double_logged(self):
        responses = [
            _tc_response("query_df", {"code": "print(1)"}),
            _tc_response("next_day", {}),
        ]
        agent = MetadataCapturingAgent(responses)
        agent.act(None, DummyForecastInterface(), date(2025, 6, 1))
        assert agent.phases.count("query") == 1

    def test_submit_not_double_logged(self):
        responses = [
            _tc_response("submit_forecasts", {
                "forecasts": [{"qid": "Q1", "outcomes": {"Yes": 0.7, "No": 0.3}}],
            }),
            _tc_response("next_day", {}),
        ]
        agent = MetadataCapturingAgent(responses)
        agent.act(None, DummyForecastInterface(), date(2025, 6, 1))
        assert agent.phases.count("submit") == 1

    def test_next_day_not_double_logged(self):
        responses = [_tc_response("next_day", {})]
        agent = MetadataCapturingAgent(responses)
        agent.act(None, DummyForecastInterface(), date(2025, 6, 1))
        assert agent.phases.count("next_day") == 1
        assert "llm" not in agent.phases
        assert "next" not in agent.phases

    def test_mixed_session_one_log_per_turn(self):
        responses = [
            _tc_response("query_df", {"code": "print(1)"}),
            _tc_response("search_news", {"query": "test"}),
            _tc_response("submit_forecasts", {
                "forecasts": [{"qid": "Q1", "outcomes": {"Yes": 0.7}}],
            }),
            _tc_response("next_day", {}),
        ]
        agent = MetadataCapturingAgent(responses)
        agent.act(None, DummyForecastInterface(), date(2025, 6, 1))
        assert len(agent.log_entries) == 4  # query, search, submit, next_day


# ===========================================================================
# 3. Parallel tool calls (multi-tool responses)
# ===========================================================================


class TestParallelToolCallMetadata:
    """Verify metadata when a single LLM response contains multiple tool calls."""

    def test_parallel_query_and_search_phase(self):
        """Two tools in one turn → phase is 'query+search', actions list is recorded."""
        responses = [
            _multi_tc_response([
                {"name": "query_df", "arguments": {"code": "print(1)"}},
                {"name": "search_news", "arguments": {"query": "test"}},
            ]),
            _tc_response("next_day", {}),
        ]
        agent = MetadataCapturingAgent(responses)
        agent.act(None, DummyForecastInterface(), date(2025, 6, 1))

        non_next = [e for e in agent.log_entries if e.get("phase") != "next_day"]
        assert len(non_next) == 1
        meta = non_next[0]
        assert meta["phase"] == "query+search"
        assert meta["actions"] == ["query", "search"]

    def test_parallel_submit_stops_at_first_success(self):
        """Submit in a parallel batch records metadata for the submitted forecast."""
        responses = [
            _multi_tc_response([
                {"name": "query_df", "arguments": {"code": "print(1)"}},
                {"name": "submit_forecasts", "arguments": {
                    "forecasts": [{"qid": "Q5", "outcomes": {"Yes": 0.9, "No": 0.1}}],
                }},
            ]),
            _tc_response("next_day", {}),
        ]
        agent = MetadataCapturingAgent(responses)
        fi = DummyForecastInterface()
        agent.act(None, fi, date(2025, 6, 1))

        # The log entry should contain submit metadata
        non_next = [e for e in agent.log_entries if e.get("phase") != "next_day"]
        assert len(non_next) == 1
        meta = non_next[0]
        assert meta["submitted_qids"] == ["Q5"]
        assert meta["num_forecasts"] == 1

    def test_parallel_memory_actions_in_memory_phase(self, tmp_path):
        """Multiple memory tool calls in the standalone memory update produce one log
        entry per LLM turn.  The standalone _run_memory_update path uses _log_model_output
        with a fixed 'memory_update' phase; the actions list shows what actually ran."""
        # Both responses are consumed by _prompt_memory_update (standalone path).
        responses = [
            _multi_tc_response([
                {"name": "mem_add", "arguments": {
                    "qid": "Q1", "question": "Will X?", "memory": "note1",
                }},
                {"name": "mem_add", "arguments": {
                    "qid": "Q2", "question": "Will Y?", "memory": "note2",
                }},
            ]),
            _tc_response("next_day", {}),
        ]
        mem = ActiveMemory("test_meta", memory_dir=str(tmp_path), max_entries=50)
        mem.set_date(date(2025, 6, 1))
        agent = MetadataCapturingAgent(responses, config=AgentConfig(
            enable_memory=True,
            memory_format="active",
            max_total_tokens=50000,
            memory_dir=str(tmp_path),
        ), memory=mem)
        fi = DummyForecastInterface()
        agent.act(None, fi, date(2025, 6, 1))

        # Standalone memory update uses _log_model_output with fixed "memory_update" phase
        mem_entries = agent.entries_with_phase("memory_update")
        assert len(mem_entries) >= 1

    def test_parallel_next_day_stops_memory_phase(self, tmp_path):
        """next_day mixed with memory actions stops the phase, logs memory_update_done."""
        responses = [
            _tc_response("next_day", {}),  # enter memory phase
            # memory phase: mem_add + next_day in parallel
            _multi_tc_response([
                {"name": "mem_add", "arguments": {
                    "qid": "Q1", "question": "Will X?", "memory": "note",
                }},
                {"name": "next_day", "arguments": {}},
            ]),
        ]
        mem = ActiveMemory("test_meta", memory_dir=str(tmp_path), max_entries=50)
        mem.set_date(date(2025, 6, 1))
        agent = MetadataCapturingAgent(responses, config=AgentConfig(
            enable_memory=True,
            memory_format="active",
            max_total_tokens=50000,
            memory_dir=str(tmp_path),
        ), memory=mem)
        fi = DummyForecastInterface()
        agent.act(None, fi, date(2025, 6, 1))

        assert "memory_update_done" in agent.phases


# ===========================================================================
# 4. Usage and finish_reason propagation
# ===========================================================================


class TestUsageAndFinishReason:
    """Verify that usage stats and finish_reason are passed through to log metadata."""

    def test_usage_tokens_in_log(self):
        responses = [
            _tc_response("query_df", {"code": "print(1)"},
                         prompt_tokens=200, completion_tokens=75),
            _tc_response("next_day", {}),
        ]
        agent = MetadataCapturingAgent(responses)
        agent.act(None, DummyForecastInterface(), date(2025, 6, 1))

        meta = agent.entries_with_phase("query")[0]
        assert meta["usage"]["prompt_tokens"] == 200
        assert meta["usage"]["completion_tokens"] == 75

    def test_finish_reason_in_log(self):
        responses = [
            _tc_response("query_df", {"code": "print(1)"},
                         finish_reason="tool_calls"),
            _tc_response("next_day", {}),
        ]
        agent = MetadataCapturingAgent(responses)
        agent.act(None, DummyForecastInterface(), date(2025, 6, 1))

        meta = agent.entries_with_phase("query")[0]
        assert meta["finish_reason"] == "tool_calls"


# ===========================================================================
# 5. Content filter circuit breaker
# ===========================================================================


class TestContentFilterCircuitBreaker:
    """Verify the circuit breaker ends the loop after N consecutive content_filter responses."""

    def test_circuit_breaker_fires(self):
        """5 consecutive content_filter responses should break the loop.
        The 5th triggers the break before logging, so only 4 are logged."""
        responses = [_no_tool_response("", finish_reason="content_filter") for _ in range(5)]
        agent = MetadataCapturingAgent(responses, config=AgentConfig(
            enable_memory=False,
            max_total_tokens=50000,
            content_filter_circuit_breaker=5,
        ))
        fi = DummyForecastInterface()
        agent.act(None, fi, date(2025, 6, 1))

        # The 5th content_filter hits the circuit breaker and breaks before logging
        llm_entries = agent.entries_with_phase("llm")
        assert len(llm_entries) == 4
        for e in llm_entries:
            assert e["finish_reason"] == "content_filter"

    def test_non_filter_resets_counter(self):
        """A normal response between content_filter responses resets the counter."""
        responses = [
            _no_tool_response("", finish_reason="content_filter"),
            _no_tool_response("", finish_reason="content_filter"),
            _tc_response("query_df", {"code": "print(1)"}),  # resets counter
            _no_tool_response("", finish_reason="content_filter"),
            _tc_response("next_day", {}),
        ]
        agent = MetadataCapturingAgent(responses, config=AgentConfig(
            enable_memory=False,
            max_total_tokens=50000,
            content_filter_circuit_breaker=3,
        ))
        fi = DummyForecastInterface()
        agent.act(None, fi, date(2025, 6, 1))

        # Should complete normally (counter reset after query_df)
        assert "next_day" in agent.phases


# ===========================================================================
# 6. Error metadata propagation
# ===========================================================================


class TestErrorMetadata:
    """Verify that parse/action errors appear in log metadata."""

    def test_query_parse_error_in_metadata(self):
        """query_df with missing code argument → error in metadata."""
        responses = [
            _tc_response("query_df", {}),  # no 'code' key
            _tc_response("next_day", {}),
        ]
        agent = MetadataCapturingAgent(responses)
        agent.act(None, DummyForecastInterface(), date(2025, 6, 1))

        query_entries = agent.entries_with_phase("query")
        assert len(query_entries) == 1
        assert "error" in query_entries[0]

    def test_search_parse_error_in_metadata(self):
        """search_news with missing query → error in metadata."""
        responses = [
            _tc_response("search_news", {}),  # no 'query' key
            _tc_response("next_day", {}),
        ]
        agent = MetadataCapturingAgent(responses)
        agent.act(None, DummyForecastInterface(), date(2025, 6, 1))

        search_entries = agent.entries_with_phase("search")
        assert len(search_entries) == 1
        assert "error" in search_entries[0]
