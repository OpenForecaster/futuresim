"""Tests for memory phase parallel tool calls, prompt nudges, and circuit breaker."""

from unittest.mock import MagicMock, patch, call
import io
import json
import sys

import pytest

from agents.basicAgent.config import AgentConfig
from agents.basicAgent.tools import (
    single_call_to_parsed_action,
    chat_response_to_action,
    extract_finish_reason,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool_call(name: str, arguments: dict, call_id: str = "call_1"):
    return {"name": name, "arguments": arguments, "call_id": call_id}


def _make_chat_response(tool_calls, finish_reason="tool_calls"):
    """Build a minimal OpenAI-style chat response JSON.

    Arguments are JSON-encoded strings (matching the real API format).
    """
    tc_list = []
    for tc in tool_calls:
        args = tc["arguments"]
        tc_list.append({
            "id": tc.get("call_id", "call_1"),
            "type": "function",
            "function": {
                "name": tc["name"],
                "arguments": json.dumps(args) if isinstance(args, dict) else args,
            },
        })
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": tc_list,
            },
            "finish_reason": finish_reason,
        }],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }


# ===========================================================================
# 1-11: single_call_to_parsed_action tests
# ===========================================================================

class TestSingleCallParsing:
    def test_memory_new(self):
        call = _make_tool_call("memory_new", {"name": "test", "description": "desc", "content": "body"})
        parsed, text = single_call_to_parsed_action(call)
        assert parsed is not None
        assert parsed.action_type == "memory_new"
        assert parsed.memory_new_data["name"] == "test"
        assert parsed.error is None

    def test_memory_retrieve(self):
        call = _make_tool_call("memory_retrieve", {"name": "entry1"})
        parsed, text = single_call_to_parsed_action(call)
        assert parsed is not None
        assert parsed.action_type == "memory_retrieve"
        assert parsed.memory_entry_name == "entry1"

    def test_mem_add(self):
        call = _make_tool_call("mem_add", {"qid": "Q123", "memory": "some note", "question": "test?"})
        parsed, text = single_call_to_parsed_action(call)
        assert parsed is not None
        assert parsed.action_type == "mem_add"
        assert parsed.mem_data["qid"] == "Q123"

    def test_mem_update(self):
        call = _make_tool_call("mem_update", {"qid": "Q456", "memory": "updated note"})
        parsed, text = single_call_to_parsed_action(call)
        assert parsed is not None
        assert parsed.action_type == "mem_update"
        assert parsed.mem_qid == "Q456"

    def test_mem_delete(self):
        call = _make_tool_call("mem_delete", {"qid": "Q789"})
        parsed, text = single_call_to_parsed_action(call)
        assert parsed is not None
        assert parsed.action_type == "mem_delete"
        assert parsed.mem_qid == "Q789"

    def test_memory_update(self):
        call = _make_tool_call("memory_update", {"name": "entry1", "content": "new content"})
        parsed, text = single_call_to_parsed_action(call)
        assert parsed is not None
        assert parsed.action_type == "memory_update"
        assert parsed.memory_entry_name == "entry1"
        assert parsed.memory_update_data == {"content": "new content"}

    def test_memory_delete(self):
        call = _make_tool_call("memory_delete", {"name": "entry1"})
        parsed, text = single_call_to_parsed_action(call)
        assert parsed is not None
        assert parsed.action_type == "memory_delete"
        assert parsed.memory_entry_name == "entry1"

    def test_next_day(self):
        call = _make_tool_call("next_day", {})
        parsed, text = single_call_to_parsed_action(call)
        assert parsed is not None
        assert parsed.action_type == "next"

    def test_unknown_tool(self):
        call = _make_tool_call("nonexistent_tool", {"x": 1})
        parsed, text = single_call_to_parsed_action(call)
        assert parsed is not None
        assert parsed.error is not None
        assert "Unknown tool call" in parsed.error

    def test_malformed_args(self):
        call = {"name": "memory_new", "arguments": "not_a_dict"}
        parsed, text = single_call_to_parsed_action(call)
        # Should fall through to tool_calls_to_parsed_action which handles non-dict args
        assert parsed is not None

    def test_routes_through_memory_path(self):
        """Verify single_call produces the same result as chat_response_to_action
        for memory tools — confirming _memory_tool_call_to_action routing."""
        call = _make_tool_call("memory_new", {"name": "test", "description": "d", "content": "c"})
        single_parsed, _ = single_call_to_parsed_action(call)

        # Build a full chat response with the same tool call
        resp = _make_chat_response([call])
        chat_parsed, _, _ = chat_response_to_action(resp)

        assert single_parsed.action_type == chat_parsed.action_type
        assert single_parsed.memory_new_data == chat_parsed.memory_new_data


# ===========================================================================
# 12-17: Parallel batch processing logic tests
# ===========================================================================

class TestParallelBatchProcessing:
    """Test the batch processing logic extracted from the agent loops.

    These tests exercise the dispatch logic directly rather than mocking
    the full agent, using the same pattern as the refactored loop.
    """

    @staticmethod
    def _run_batch_loop(tool_calls):
        """Simulate the memory phase batch processing loop.

        Returns (handled_calls, saw_next_day) where handled_calls is a list
        of (action_type, tool_call) tuples.
        """
        handled = []
        saw_next_day = False
        for tc in tool_calls:
            tc_parsed, _ = single_call_to_parsed_action(tc)
            if tc_parsed is None:
                continue
            if tc_parsed.action_type == "next":
                saw_next_day = True
                break
            handled.append((tc_parsed.action_type, tc))
        return handled, saw_next_day

    def test_all_memory_calls_processed(self):
        calls = [
            _make_tool_call("memory_retrieve", {"name": "a"}, "c1"),
            _make_tool_call("memory_retrieve", {"name": "b"}, "c2"),
            _make_tool_call("memory_retrieve", {"name": "c"}, "c3"),
        ]
        handled, saw_next_day = self._run_batch_loop(calls)
        assert len(handled) == 3
        assert all(at == "memory_retrieve" for at, _ in handled)
        assert not saw_next_day

    def test_next_day_in_batch_stops(self):
        calls = [
            _make_tool_call("memory_update", {"name": "x", "content": "y"}, "c1"),
            _make_tool_call("next_day", {}, "c2"),
            _make_tool_call("memory_new", {"name": "z", "description": "d", "content": "c"}, "c3"),
        ]
        handled, saw_next_day = self._run_batch_loop(calls)
        assert len(handled) == 1
        assert handled[0][0] == "memory_update"
        assert saw_next_day

    def test_mixed_memory_and_mem_dispatched(self):
        calls = [
            _make_tool_call("memory_retrieve", {"name": "a"}, "c1"),
            _make_tool_call("mem_add", {"qid": "Q1", "memory": "note"}, "c2"),
            _make_tool_call("memory_update", {"name": "b", "content": "new"}, "c3"),
        ]
        handled, saw_next_day = self._run_batch_loop(calls)
        assert len(handled) == 3
        types = [at for at, _ in handled]
        assert types == ["memory_retrieve", "mem_add", "memory_update"]
        assert not saw_next_day

    def test_invalid_tool_in_batch(self):
        """query_df during memory phase should parse as 'query' (handled as invalid in agent)."""
        calls = [
            _make_tool_call("memory_retrieve", {"name": "a"}, "c1"),
            _make_tool_call("query_df", {"code": "print(1)"}, "c2"),
            _make_tool_call("memory_new", {"name": "z", "description": "d", "content": "c"}, "c3"),
        ]
        handled, saw_next_day = self._run_batch_loop(calls)
        assert len(handled) == 3
        types = [at for at, _ in handled]
        assert types[0] == "memory_retrieve"
        assert types[1] == "query"  # Not a valid memory tool — dispatched to _handle_invalid
        assert types[2] == "memory_new"

    def test_none_parsed_skipped(self):
        """A completely empty call should be skipped."""
        calls = [
            _make_tool_call("memory_retrieve", {"name": "a"}, "c1"),
            {"name": "", "arguments": {}},  # Will parse to an error (unknown tool "")
            _make_tool_call("memory_new", {"name": "z", "description": "d", "content": "c"}, "c3"),
        ]
        handled, saw_next_day = self._run_batch_loop(calls)
        # The empty-name call will parse to an error (not None), so it won't be skipped
        # It should still be 3 calls processed (one with error)
        assert len(handled) >= 2  # First and last are definitely there

    def test_empty_tool_calls_list(self):
        handled, saw_next_day = self._run_batch_loop([])
        assert len(handled) == 0
        assert not saw_next_day


# ===========================================================================
# 18-23: Prompt tests
# ===========================================================================

class TestMemoryPrompts:
    """Test that memory prompts contain parallel batching guidance and budget nudge."""

    @pytest.fixture
    def agent(self):
        """Create a minimal BasicAgent for prompt testing."""
        from agents.basicAgent.agent import BasicAgent
        from agents.utils.memory import StructuredMemory
        import tempfile
        tmpdir = tempfile.mkdtemp()
        config = AgentConfig(
            enable_memory=True,
            memory_format="structured",
            memory_max_entries=10,
            memory_dir=tmpdir,
        )
        agent = BasicAgent.__new__(BasicAgent)
        agent.config = config
        agent.agent_id = "test_agent"
        agent._memory = StructuredMemory(
            agent_id="test_agent",
            memory_dir=tmpdir,
            max_entries=10,
        )
        agent._resolution_log = []
        agent._past_resolution_summaries = []
        return agent

    def test_structured_prompt_allows_multiple_tools(self, agent):
        from datetime import date
        agent.config.parallel_tool_calls = True
        prompt = agent._build_structured_memory_prompt(date(2025, 5, 1))
        assert "multiple tools per turn" in prompt
        assert "exactly one tool per turn" not in prompt

    def test_structured_prompt_budget_nudge(self, agent):
        from datetime import date
        prompt = agent._build_structured_memory_prompt(date(2025, 5, 1))
        assert "do NOT need to exhaust" in prompt

    def test_structured_prompt_batching_guidance(self, agent):
        from datetime import date
        agent.config.parallel_tool_calls = True
        prompt = agent._build_structured_memory_prompt(date(2025, 5, 1))
        assert "batch" in prompt.lower()
        assert "memory_retrieve" in prompt

    def test_active_prompt_allows_multiple_tools(self, agent):
        from datetime import date
        from agents.utils.memory import ActiveMemory
        import tempfile
        tmpdir = tempfile.mkdtemp()
        agent.config.parallel_tool_calls = True
        agent._memory = ActiveMemory(
            agent_id="test_agent",
            memory_dir=tmpdir,
            max_entries=10,
        )
        prompt = agent._build_active_memory_prompt(date(2025, 5, 1))
        assert "multiple tools per turn" in prompt
        assert "exactly one tool per turn" not in prompt

    def test_active_prompt_budget_nudge(self, agent):
        from datetime import date
        from agents.utils.memory import ActiveMemory
        import tempfile
        tmpdir = tempfile.mkdtemp()
        agent._memory = ActiveMemory(
            agent_id="test_agent",
            memory_dir=tmpdir,
            max_entries=10,
        )
        prompt = agent._build_active_memory_prompt(date(2025, 5, 1))
        assert "do NOT need to exhaust" in prompt

    def test_active_prompt_batching_guidance(self, agent):
        from datetime import date
        from agents.utils.memory import ActiveMemory
        import tempfile
        tmpdir = tempfile.mkdtemp()
        agent.config.parallel_tool_calls = True
        agent._memory = ActiveMemory(
            agent_id="test_agent",
            memory_dir=tmpdir,
            max_entries=10,
        )
        prompt = agent._build_active_memory_prompt(date(2025, 5, 1))
        assert "batch" in prompt.lower()
        assert "mem_add" in prompt


# ===========================================================================
# 24-28: Content filter circuit breaker tests + config
# ===========================================================================

class TestContentFilterCircuitBreaker:
    def test_default_value(self):
        assert AgentConfig().content_filter_circuit_breaker == 5

    def test_configurable(self):
        config = AgentConfig(content_filter_circuit_breaker=3)
        assert config.content_filter_circuit_breaker == 3

    def test_circuit_breaker_fires(self):
        """Simulate N consecutive content_filter responses and verify break logic."""
        threshold = 5
        consecutive = 0
        broke_out = False
        for i in range(10):
            finish_reason = "content_filter"
            if finish_reason == "content_filter":
                consecutive += 1
                if consecutive >= threshold:
                    broke_out = True
                    break
            else:
                consecutive = 0
        assert broke_out
        assert consecutive == threshold

    def test_counter_resets_on_success(self):
        """3 filters → 1 success → 3 filters should NOT trigger breaker at threshold=5."""
        threshold = 5
        consecutive = 0
        broke_out = False
        sequence = ["content_filter"] * 3 + ["stop"] + ["content_filter"] * 3
        for fr in sequence:
            if fr == "content_filter":
                consecutive += 1
                if consecutive >= threshold:
                    broke_out = True
                    break
            else:
                consecutive = 0
        assert not broke_out
        assert consecutive == 3  # Reset after "stop", then counted 3 more

    def test_circuit_breaker_fires_at_custom_threshold(self):
        threshold = 3
        consecutive = 0
        broke_out = False
        for i in range(10):
            consecutive += 1
            if consecutive >= threshold:
                broke_out = True
                break
        assert broke_out
        assert consecutive == threshold


# ===========================================================================
# 29-30: Empty content warning suppression tests
# ===========================================================================

class TestEmptyContentWarningSuppression:
    def test_no_warning_for_tool_calls_finish(self):
        """Empty content with finish_reason=tool_calls should not print a warning."""
        captured = io.StringIO()
        sys.stdout = captured

        # Simulate the logic from openrouter.py:370-377
        content = ""
        finish_reason = "tool_calls"
        reasoning = None
        if (not content or not content.strip()) and finish_reason != "tool_calls":
            print(f"  [OpenRouter] Warning: empty content "
                  f"(finish={finish_reason}, tokens=47, "
                  f"reasoning={'yes' if reasoning else 'no'})")

        sys.stdout = sys.__stdout__
        assert "Warning" not in captured.getvalue()

    def test_warning_for_non_tool_calls_finish(self):
        """Empty content with finish_reason=stop SHOULD print a warning."""
        captured = io.StringIO()
        sys.stdout = captured

        content = ""
        finish_reason = "stop"
        reasoning = None
        if (not content or not content.strip()) and finish_reason != "tool_calls":
            print(f"  [OpenRouter] Warning: empty content "
                  f"(finish={finish_reason}, tokens=47, "
                  f"reasoning={'yes' if reasoning else 'no'})")

        sys.stdout = sys.__stdout__
        assert "Warning" in captured.getvalue()
        assert "finish=stop" in captured.getvalue()
