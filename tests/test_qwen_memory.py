"""Intensive tests for Qwen agent memory support.

Tests cover: tool schema generation, tool call → ParsedAction parsing,
Qwen memory handlers, memory phase transition, end-to-end workflows,
and budget integration.
"""

import json
from datetime import date
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from agents.basicAgent import AgentConfig
from agents.qwenAgent import QwenAllQAgent, QwenBasicAgent
from agents.qwenAgent.tools import (
    build_action_tools,
    build_memory_phase_tools,
    chat_response_to_action,
    _memory_tool_call_to_action,
)
from agents.utils.budget import BudgetSettings, BudgetTracker
from agents.utils.forecast_parser import ParsedAction
from agents.utils.memory import ActiveMemory, StructuredMemory
from inference.vllm import VLLMInference


# ── Helpers ───────────────────────────────────────────────────────────────


def make_tool_call_response(
    name: str,
    arguments: Dict[str, Any],
    call_id: str = "call_1",
    content: str = "",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
) -> Dict[str, Any]:
    """Build a Chat Completions response JSON with a single tool call."""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": call_id,
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def make_no_tool_response(
    *,
    content: str = "No tool call",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
) -> Dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class DummyForecastInterface:
    """Minimal forecast interface for tests."""

    def __init__(self):
        self.logs = []
        self.next_day_calls = 0
        self.submitted = []

    def log_model_output(self, prompt, response, metadata):
        self.logs.append({"prompt": prompt, "response": response, "metadata": metadata})

    def next_day(self):
        self.next_day_calls += 1

    def get_market_csv_path(self):
        return None

    def submit_prediction(self, pred):
        self.submitted.append(pred)


class ScriptedQwenAgent(QwenBasicAgent):
    """Agent that returns scripted Chat Completions responses."""

    def __init__(self, responses: List[Any], config: Optional[AgentConfig] = None, **kwargs):
        cfg = config or AgentConfig(
            enable_memory=True,
            memory_format="active",
            max_total_tokens=100000,
            memory_update_max_total_tokens=40000,
        )
        super().__init__(
            agent_id="test_scripted",
            inference_provider=None,
            config=cfg,
            **kwargs,
        )
        self._scripted_responses = list(responses)
        self._call_index = 0
        self._tools_per_call: List[List[Dict]] = []
        self._messages_per_call: List[List[Dict[str, Any]]] = []
        self._sampling_params_per_call: List[Dict[str, Any]] = []

    def _setup_day(self, forecast_interface, current_date):
        self._forecast_interface = forecast_interface

    def _build_qwen_instructions(self, current_date: date) -> str:
        return "Test prompt"

    def _call_chat_json_with_retries(self, *, messages, tools, sampling_params):
        self._tools_per_call.append(tools)
        self._messages_per_call.append(list(messages))
        self._sampling_params_per_call.append(dict(sampling_params))
        if self._call_index >= len(self._scripted_responses):
            raise RuntimeError("No more scripted responses")
        resp = self._scripted_responses[self._call_index]
        self._call_index += 1
        if isinstance(resp, Exception):
            raise resp
        return resp

    def _build_memory_phase_prompt(self, current_date):
        return "Memory phase: update your memory now. Call next_day when done."

    def _build_start_budget_status(self, *, warmup=False, max_actions_override=None):
        return ""

    def _build_budget_overview(self, *args, **kwargs):
        return ""


class ScriptedQwenAllQAgent(QwenAllQAgent):
    def __init__(self, responses: List[Any], config: Optional[AgentConfig] = None, **kwargs):
        cfg = config or AgentConfig(
            enable_memory=True,
            memory_format="active",
            warmup_max_actions=4,
            warmup_max_total_tokens=40000,
            warmup_submit_reserve_tokens=1024,
            warmup_force_submit_threshold_tokens=2048,
        )
        super().__init__(
            agent_id="test_qwen_warmup",
            inference_provider=None,
            config=cfg,
            start_date=date(2025, 1, 1),
            **kwargs,
        )
        self._scripted_responses = list(responses)
        self._call_index = 0
        self._tools_per_call: List[List[Dict]] = []
        self._messages_per_call: List[List[Dict[str, Any]]] = []
        self._sampling_params_per_call: List[Dict[str, Any]] = []

    def _setup_warmup_day(self, forecast_interface, current_date):
        self._forecast_interface = forecast_interface

    def _call_chat_json_with_retries(self, *, messages, tools, sampling_params):
        self._tools_per_call.append(tools)
        self._messages_per_call.append(list(messages))
        self._sampling_params_per_call.append(dict(sampling_params))
        if self._call_index >= len(self._scripted_responses):
            raise RuntimeError("No more scripted responses")
        resp = self._scripted_responses[self._call_index]
        self._call_index += 1
        if isinstance(resp, Exception):
            raise resp
        return resp

    def _build_start_budget_status(self, *, warmup=False, max_actions_override=None):
        return ""

    def _build_budget_overview(self, *args, **kwargs):
        return ""


class RecordingInference(VLLMInference):
    def __init__(self, response: Optional[Dict[str, Any]] = None):
        self.model_path = "/tmp/fake"
        self.model_name = "fake"
        self.response = response or make_tool_call_response("next_day", {}, call_id="call_default")
        self.calls: List[Dict[str, Any]] = []

    def chat_json(self, messages, sampling_params):
        self.calls.append({"messages": list(messages), "sampling_params": dict(sampling_params)})
        return self.response


class XMLWarmupInference:
    def __init__(self, responses: List[Any]):
        self._responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def chat(self, messages, sampling_params):
        self.calls.append({"messages": list(messages), "sampling_params": dict(sampling_params)})
        if not self._responses:
            raise RuntimeError("No more scripted XML warmup responses")
        resp = self._responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


# ── A. Tool Schema Tests ─────────────────────────────────────────────────


class TestMemoryToolSchemas:
    """Test that build_action_tools generates correct memory tool schemas."""

    def test_no_memory_tools_by_default(self):
        tools = build_action_tools(
            enable_query=True, enable_search=False, max_outcomes_per_question=5,
        )
        names = {t["function"]["name"] for t in tools}
        assert "memory_retrieve" not in names
        assert "mem_add" not in names

    def test_structured_memory_tools_added(self):
        tools = build_action_tools(
            enable_query=True, enable_search=False, max_outcomes_per_question=5,
            enable_memory=True,
        )
        names = {t["function"]["name"] for t in tools}
        assert {"memory_retrieve", "memory_new", "memory_update", "memory_delete"} <= names
        # mem_df tools should not be present without enable_mem_df
        assert "mem_add" not in names

    def test_active_memory_tools_include_mem_df(self):
        tools = build_action_tools(
            enable_query=True, enable_search=False, max_outcomes_per_question=5,
            enable_memory=True, enable_mem_df=True,
        )
        names = {t["function"]["name"] for t in tools}
        assert {"memory_retrieve", "memory_new", "memory_update", "memory_delete"} <= names
        assert {"mem_add", "mem_update", "mem_delete"} <= names

    def test_memory_tool_schemas_are_valid(self):
        """Every memory tool schema has type, function, name, description, parameters."""
        tools = build_action_tools(
            enable_query=False, enable_search=False, max_outcomes_per_question=5,
            enable_memory=True, enable_mem_df=True,
        )
        memory_tools = [t for t in tools if t["function"]["name"].startswith(("memory_", "mem_"))]
        assert len(memory_tools) == 7
        for tool in memory_tools:
            assert tool["type"] == "function"
            fn = tool["function"]
            assert isinstance(fn["name"], str)
            assert isinstance(fn["description"], str)
            assert isinstance(fn["parameters"], dict)
            assert "properties" in fn["parameters"]
            assert "required" in fn["parameters"]

    def test_memory_tools_required_params(self):
        tools = build_action_tools(
            enable_query=False, enable_search=False, max_outcomes_per_question=5,
            enable_memory=True, enable_mem_df=True,
        )
        by_name = {t["function"]["name"]: t for t in tools}

        assert by_name["memory_retrieve"]["function"]["parameters"]["required"] == ["name"]
        assert set(by_name["memory_new"]["function"]["parameters"]["required"]) == {"name", "description", "content"}
        assert by_name["memory_delete"]["function"]["parameters"]["required"] == ["name"]
        assert set(by_name["mem_add"]["function"]["parameters"]["required"]) == {"qid", "memory"}
        assert set(by_name["mem_update"]["function"]["parameters"]["required"]) == {"qid", "memory"}
        assert by_name["mem_delete"]["function"]["parameters"]["required"] == ["qid"]

    def test_next_day_is_last_tool(self):
        """next_day should always be the last tool in the list."""
        tools = build_action_tools(
            enable_query=True, enable_search=True, max_outcomes_per_question=5,
            enable_memory=True, enable_mem_df=True, max_search_results=3,
        )
        assert tools[-1]["function"]["name"] == "next_day"


class TestMemoryPhaseTools:
    """Test build_memory_phase_tools returns only memory tools + next_day."""

    def test_memory_phase_tools_structure(self):
        tools = build_memory_phase_tools(enable_memory=True, enable_mem_df=True)
        names = [t["function"]["name"] for t in tools]
        # Should have memory tools + next_day, no query/search/submit
        assert "next_day" in names
        assert "memory_retrieve" in names
        assert "mem_add" in names
        assert "query_df" not in names
        assert "search_news" not in names
        assert "submit_forecasts" not in names
        # next_day should be last
        assert names[-1] == "next_day"

    def test_memory_phase_tools_no_mem_df(self):
        tools = build_memory_phase_tools(enable_memory=True, enable_mem_df=False)
        names = {t["function"]["name"] for t in tools}
        assert "memory_retrieve" in names
        assert "mem_add" not in names


# ── B. Tool Call Parsing Tests ────────────────────────────────────────────


class TestMemoryToolCallParsing:
    """Test chat_response_to_action for memory tool calls."""

    def test_memory_retrieve(self):
        resp = make_tool_call_response("memory_retrieve", {"name": "q42-lesson"})
        parsed, text, calls = chat_response_to_action(resp)
        assert parsed.action_type == "memory_retrieve"
        assert parsed.memory_entry_name == "q42-lesson"
        assert parsed.error is None

    def test_memory_new(self):
        resp = make_tool_call_response("memory_new", {
            "name": "q99-climate",
            "description": "Climate prediction reasoning",
            "content": "Paris agreement unlikely by 2030.",
        })
        parsed, _, _ = chat_response_to_action(resp)
        assert parsed.action_type == "memory_new"
        assert parsed.error is None
        assert parsed.memory_new_data["name"] == "q99-climate"
        assert parsed.memory_new_data["description"] == "Climate prediction reasoning"
        assert "Paris agreement" in parsed.memory_new_data["content"]

    def test_memory_new_missing_content(self):
        resp = make_tool_call_response("memory_new", {
            "name": "bad-entry",
            "description": "No content",
        })
        parsed, _, _ = chat_response_to_action(resp)
        assert parsed.action_type == "memory_new"
        assert parsed.error is not None
        assert "content" in parsed.error.lower()

    def test_memory_new_default_description(self):
        """If description is missing, it should default to the name."""
        resp = make_tool_call_response("memory_new", {
            "name": "auto-desc",
            "content": "Some content here.",
        })
        parsed, _, _ = chat_response_to_action(resp)
        assert parsed.action_type == "memory_new"
        assert parsed.error is None
        assert parsed.memory_new_data["description"] == "auto-desc"

    def test_memory_update(self):
        resp = make_tool_call_response("memory_update", {
            "name": "bookmaker-cal",
            "content": "Updated calibration to 85%.",
        })
        parsed, _, _ = chat_response_to_action(resp)
        assert parsed.action_type == "memory_update"
        assert parsed.memory_entry_name == "bookmaker-cal"
        assert parsed.memory_update_data == {"content": "Updated calibration to 85%."}
        assert parsed.error is None

    def test_memory_update_description_only(self):
        resp = make_tool_call_response("memory_update", {
            "name": "old-entry",
            "description": "New description only.",
        })
        parsed, _, _ = chat_response_to_action(resp)
        assert parsed.action_type == "memory_update"
        assert parsed.memory_update_data == {"description": "New description only."}

    def test_memory_update_no_fields(self):
        resp = make_tool_call_response("memory_update", {"name": "empty-update"})
        parsed, _, _ = chat_response_to_action(resp)
        assert parsed.action_type == "memory_update"
        assert parsed.error is not None
        assert "no fields" in parsed.error.lower()

    def test_memory_update_missing_name(self):
        resp = make_tool_call_response("memory_update", {"content": "x"})
        parsed, _, _ = chat_response_to_action(resp)
        assert parsed.action_type == "memory_update"
        assert parsed.error is not None

    def test_memory_delete(self):
        resp = make_tool_call_response("memory_delete", {"name": "stale-entry"})
        parsed, _, _ = chat_response_to_action(resp)
        assert parsed.action_type == "memory_delete"
        assert parsed.memory_entry_name == "stale-entry"
        assert parsed.error is None

    def test_memory_delete_missing_name(self):
        resp = make_tool_call_response("memory_delete", {})
        parsed, _, _ = chat_response_to_action(resp)
        assert parsed.action_type == "memory_delete"
        assert parsed.error is not None

    def test_mem_add(self):
        resp = make_tool_call_response("mem_add", {
            "qid": "Q42",
            "question": "Will X happen?",
            "memory": "Key evidence: polls show 55%",
            "category": "politics",
        })
        parsed, _, _ = chat_response_to_action(resp)
        assert parsed.action_type == "mem_add"
        assert parsed.error is None
        assert parsed.mem_data["qid"] == "Q42"
        assert parsed.mem_data["memory"] == "Key evidence: polls show 55%"
        assert parsed.mem_data["category"] == "politics"

    def test_mem_add_minimal(self):
        """mem_add with only required fields (qid, memory)."""
        resp = make_tool_call_response("mem_add", {
            "qid": "Q99",
            "memory": "Minimal evidence",
        })
        parsed, _, _ = chat_response_to_action(resp)
        assert parsed.action_type == "mem_add"
        assert parsed.error is None
        assert parsed.mem_data["qid"] == "Q99"
        assert parsed.mem_data["question"] == ""
        assert parsed.mem_data["category"] == ""

    def test_mem_add_missing_qid(self):
        resp = make_tool_call_response("mem_add", {"memory": "No qid"})
        parsed, _, _ = chat_response_to_action(resp)
        assert parsed.action_type == "mem_add"
        assert parsed.error is not None

    def test_mem_add_missing_memory(self):
        resp = make_tool_call_response("mem_add", {"qid": "Q1"})
        parsed, _, _ = chat_response_to_action(resp)
        assert parsed.action_type == "mem_add"
        assert parsed.error is not None

    def test_mem_update(self):
        resp = make_tool_call_response("mem_update", {
            "qid": "Q42",
            "memory": "Updated evidence: 60%",
            "category": "economics",
        })
        parsed, _, _ = chat_response_to_action(resp)
        assert parsed.action_type == "mem_update"
        assert parsed.mem_qid == "Q42"
        assert parsed.mem_data["memory"] == "Updated evidence: 60%"
        assert parsed.mem_data["category"] == "economics"
        assert parsed.error is None

    def test_mem_update_no_category(self):
        resp = make_tool_call_response("mem_update", {
            "qid": "Q42",
            "memory": "Updated",
        })
        parsed, _, _ = chat_response_to_action(resp)
        assert parsed.action_type == "mem_update"
        assert parsed.mem_data["category"] is None

    def test_mem_update_missing_qid(self):
        resp = make_tool_call_response("mem_update", {"memory": "x"})
        parsed, _, _ = chat_response_to_action(resp)
        assert parsed.action_type == "mem_update"
        assert parsed.error is not None

    def test_mem_delete(self):
        resp = make_tool_call_response("mem_delete", {"qid": "Q42"})
        parsed, _, _ = chat_response_to_action(resp)
        assert parsed.action_type == "mem_delete"
        assert parsed.mem_qid == "Q42"
        assert parsed.error is None

    def test_mem_delete_missing_qid(self):
        resp = make_tool_call_response("mem_delete", {})
        parsed, _, _ = chat_response_to_action(resp)
        assert parsed.action_type == "mem_delete"
        assert parsed.error is not None

    def test_non_memory_tool_falls_through(self):
        """query_df should still be handled by the shared response_to_action."""
        resp = make_tool_call_response("query_df", {"code": "print(df.head())"})
        parsed, _, _ = chat_response_to_action(resp)
        assert parsed.action_type == "query"
        assert parsed.code == "print(df.head())"

    def test_unknown_tool_returns_error(self):
        resp = make_tool_call_response("nonexistent_tool", {"x": 1})
        parsed, _, _ = chat_response_to_action(resp)
        assert parsed.action_type is None
        assert parsed.error is not None
        assert "nonexistent_tool" in parsed.error

    def test_whitespace_stripping(self):
        resp = make_tool_call_response("memory_retrieve", {"name": "  q42-lesson  "})
        parsed, _, _ = chat_response_to_action(resp)
        assert parsed.memory_entry_name == "q42-lesson"


# ── C. Qwen Memory Handler Tests ─────────────────────────────────────────


class TestQwenMemoryHandlers:
    """Test _qwen_handle_memory_action and _qwen_handle_mem_action."""

    @pytest.fixture
    def agent_with_active_memory(self, tmp_path):
        """Create a QwenBasicAgent with real ActiveMemory on a temp dir."""
        cfg = AgentConfig(
            enable_memory=True,
            memory_format="active",
            memory_dir=str(tmp_path),
        )
        agent = QwenBasicAgent(
            agent_id="test_handler",
            inference_provider=None,
            config=cfg,
        )
        agent._memory = ActiveMemory("test_handler", str(tmp_path))
        agent._memory.set_date(date(2025, 6, 1))
        return agent

    @pytest.fixture
    def agent_with_structured_memory(self, tmp_path):
        """Create a QwenBasicAgent with StructuredMemory."""
        cfg = AgentConfig(
            enable_memory=True,
            memory_format="structured",
            memory_dir=str(tmp_path),
        )
        agent = QwenBasicAgent(
            agent_id="test_handler",
            inference_provider=None,
            config=cfg,
        )
        agent._memory = StructuredMemory("test_handler", str(tmp_path))
        agent._memory.set_date(date(2025, 6, 1))
        return agent

    def _make_budget(self):
        return BudgetTracker(BudgetSettings(max_actions=10))

    def _make_tool_call(self, call_id="call_1"):
        return {"call_id": call_id}

    # -- memory_new --

    def test_memory_new_adds_entry(self, agent_with_active_memory):
        agent = agent_with_active_memory
        messages = []
        budget = self._make_budget()
        parsed = ParsedAction(
            action_type="memory_new", code=None, forecasts=None, query=None,
            memory_new_data={"name": "test-insight", "description": "Test desc", "content": "Test content"},
        )
        agent._qwen_handle_memory_action(
            messages=messages, parsed=parsed,
            tool_call=self._make_tool_call(), budget=budget,
        )
        assert agent._memory.entry_count == 1
        assert agent._memory.retrieve("test-insight") is not None
        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        assert "Added" in messages[0]["content"]

    def test_memory_new_duplicate_error(self, agent_with_structured_memory):
        agent = agent_with_structured_memory
        agent._memory.add_entry("existing", "desc", "content")
        messages = []
        budget = self._make_budget()
        parsed = ParsedAction(
            action_type="memory_new", code=None, forecasts=None, query=None,
            memory_new_data={"name": "existing", "description": "dup", "content": "dup"},
        )
        agent._qwen_handle_memory_action(
            messages=messages, parsed=parsed,
            tool_call=self._make_tool_call(), budget=budget,
        )
        assert "ERROR" in messages[0]["content"]

    # -- memory_retrieve --

    def test_memory_retrieve_existing(self, agent_with_active_memory):
        agent = agent_with_active_memory
        agent._memory.add_entry("my-entry", "My desc", "My content details")
        messages = []
        budget = self._make_budget()
        parsed = ParsedAction(
            action_type="memory_retrieve", code=None, forecasts=None, query=None,
            memory_entry_name="my-entry",
        )
        agent._qwen_handle_memory_action(
            messages=messages, parsed=parsed,
            tool_call=self._make_tool_call(), budget=budget,
        )
        assert "My content details" in messages[0]["content"]

    def test_memory_retrieve_nonexistent(self, agent_with_active_memory):
        agent = agent_with_active_memory
        messages = []
        budget = self._make_budget()
        parsed = ParsedAction(
            action_type="memory_retrieve", code=None, forecasts=None, query=None,
            memory_entry_name="nonexistent",
        )
        agent._qwen_handle_memory_action(
            messages=messages, parsed=parsed,
            tool_call=self._make_tool_call(), budget=budget,
        )
        assert "ERROR" in messages[0]["content"]

    # -- memory_update --

    def test_memory_update_content(self, agent_with_active_memory):
        agent = agent_with_active_memory
        agent._memory.add_entry("upd-entry", "desc", "old content")
        messages = []
        budget = self._make_budget()
        parsed = ParsedAction(
            action_type="memory_update", code=None, forecasts=None, query=None,
            memory_entry_name="upd-entry",
            memory_update_data={"content": "new content"},
        )
        agent._qwen_handle_memory_action(
            messages=messages, parsed=parsed,
            tool_call=self._make_tool_call(), budget=budget,
        )
        assert "Updated" in messages[0]["content"]
        assert "new content" in agent._memory.retrieve("upd-entry")

    def test_memory_update_nonexistent(self, agent_with_active_memory):
        agent = agent_with_active_memory
        messages = []
        budget = self._make_budget()
        parsed = ParsedAction(
            action_type="memory_update", code=None, forecasts=None, query=None,
            memory_entry_name="nope",
            memory_update_data={"content": "x"},
        )
        agent._qwen_handle_memory_action(
            messages=messages, parsed=parsed,
            tool_call=self._make_tool_call(), budget=budget,
        )
        assert "ERROR" in messages[0]["content"]

    # -- memory_delete --

    def test_memory_delete_existing(self, agent_with_active_memory):
        agent = agent_with_active_memory
        agent._memory.add_entry("del-entry", "desc", "content")
        messages = []
        budget = self._make_budget()
        parsed = ParsedAction(
            action_type="memory_delete", code=None, forecasts=None, query=None,
            memory_entry_name="del-entry",
        )
        agent._qwen_handle_memory_action(
            messages=messages, parsed=parsed,
            tool_call=self._make_tool_call(), budget=budget,
        )
        assert "Deleted" in messages[0]["content"]
        assert agent._memory.entry_count == 0

    # -- mem_add --

    def test_mem_add(self, agent_with_active_memory):
        agent = agent_with_active_memory
        messages = []
        budget = self._make_budget()
        parsed = ParsedAction(
            action_type="mem_add", code=None, forecasts=None, query=None,
            mem_data={"qid": "Q42", "question": "Will X?", "memory": "Key evidence", "category": "politics"},
        )
        agent._qwen_handle_mem_action(
            messages=messages, parsed=parsed,
            tool_call=self._make_tool_call(), budget=budget,
        )
        assert agent._memory.mem_count == 1
        df = agent._memory.get_mem_df()
        assert df.iloc[0]["qid"] == "Q42"
        assert "Added" in messages[0]["content"]

    # -- mem_update --

    def test_mem_update(self, agent_with_active_memory):
        agent = agent_with_active_memory
        agent._memory.mem_add(qid="Q42", question="Will X?", memory="Old", category="politics")
        messages = []
        budget = self._make_budget()
        parsed = ParsedAction(
            action_type="mem_update", code=None, forecasts=None, query=None,
            mem_qid="Q42",
            mem_data={"qid": "Q42", "memory": "New evidence", "category": None},
        )
        agent._qwen_handle_mem_action(
            messages=messages, parsed=parsed,
            tool_call=self._make_tool_call(), budget=budget,
        )
        assert "Updated" in messages[0]["content"]
        df = agent._memory.get_mem_df()
        assert df.iloc[0]["memory"] == "New evidence"

    # -- mem_delete --

    def test_mem_delete(self, agent_with_active_memory):
        agent = agent_with_active_memory
        agent._memory.mem_add(qid="Q42", question="Will X?", memory="Evidence", category="politics")
        messages = []
        budget = self._make_budget()
        parsed = ParsedAction(
            action_type="mem_delete", code=None, forecasts=None, query=None,
            mem_qid="Q42",
        )
        agent._qwen_handle_mem_action(
            messages=messages, parsed=parsed,
            tool_call=self._make_tool_call(), budget=budget,
        )
        assert "Deleted" in messages[0]["content"]
        assert agent._memory.mem_count == 0

    def test_mem_delete_nonexistent(self, agent_with_active_memory):
        agent = agent_with_active_memory
        messages = []
        budget = self._make_budget()
        parsed = ParsedAction(
            action_type="mem_delete", code=None, forecasts=None, query=None,
            mem_qid="Q999",
        )
        agent._qwen_handle_mem_action(
            messages=messages, parsed=parsed,
            tool_call=self._make_tool_call(), budget=budget,
        )
        assert "ERROR" in messages[0]["content"]

    # -- mem handler without active memory --

    def test_mem_handler_without_active_memory(self, agent_with_structured_memory):
        agent = agent_with_structured_memory
        messages = []
        budget = self._make_budget()
        parsed = ParsedAction(
            action_type="mem_add", code=None, forecasts=None, query=None,
            mem_data={"qid": "Q1", "question": "Q", "memory": "M", "category": "C"},
        )
        agent._qwen_handle_mem_action(
            messages=messages, parsed=parsed,
            tool_call=self._make_tool_call(), budget=budget,
        )
        assert "ERROR" in messages[0]["content"]
        assert "Active memory" in messages[0]["content"]

    # -- handler uses tool role messages --

    def test_handler_uses_tool_role_messages(self, agent_with_active_memory):
        agent = agent_with_active_memory
        messages = []
        budget = self._make_budget()
        parsed = ParsedAction(
            action_type="memory_new", code=None, forecasts=None, query=None,
            memory_new_data={"name": "test", "description": "d", "content": "c"},
        )
        agent._qwen_handle_memory_action(
            messages=messages, parsed=parsed,
            tool_call={"call_id": "call_abc123"}, budget=budget,
        )
        assert messages[0]["role"] == "tool"
        assert messages[0]["tool_call_id"] == "call_abc123"
        assert messages[0]["name"] == "memory_new"

    # -- handler consumes budget action --

    def test_handler_consumes_budget_action(self, agent_with_active_memory):
        agent = agent_with_active_memory
        budget = BudgetTracker(BudgetSettings(max_actions=5))
        assert budget.actions_remaining == 5
        parsed = ParsedAction(
            action_type="mem_add", code=None, forecasts=None, query=None,
            mem_data={"qid": "Q1", "question": "Q", "memory": "M", "category": "C"},
        )
        agent._qwen_handle_mem_action(
            messages=[], parsed=parsed,
            tool_call=self._make_tool_call(), budget=budget,
        )
        assert budget.actions_remaining == 4

    # -- handler with parse error --

    def test_handler_with_parse_error(self, agent_with_active_memory):
        agent = agent_with_active_memory
        messages = []
        budget = self._make_budget()
        parsed = ParsedAction(
            action_type="memory_new", code=None, forecasts=None, query=None,
            error="Missing 'content' parameter",
        )
        agent._qwen_handle_memory_action(
            messages=messages, parsed=parsed,
            tool_call=self._make_tool_call(), budget=budget,
        )
        assert "ERROR" in messages[0]["content"]
        assert "Missing" in messages[0]["content"]


# ── D. Memory Phase Transition Tests ─────────────────────────────────────


class TestQwenMemoryPhaseTransition:
    """Test that the action loop transitions to memory phase correctly."""

    def test_next_triggers_memory_phase_when_eligible(self, tmp_path):
        """When agent calls next_day, loop should transition to memory phase if eligible."""
        responses = [
            # Turn 1: agent calls next_day → triggers memory phase
            make_tool_call_response("next_day", {}, call_id="call_next1"),
            # Turn 2: in memory phase, agent adds a meta-insight
            make_tool_call_response("memory_new", {
                "name": "lesson-1",
                "description": "A lesson",
                "content": "Learned something important",
            }, call_id="call_mem1"),
            # Turn 3: agent ends memory phase
            make_tool_call_response("next_day", {}, call_id="call_next2"),
        ]
        agent = ScriptedQwenAgent(responses)
        agent._memory = ActiveMemory("test", str(tmp_path))
        agent._memory.set_date(date(2025, 6, 1))

        fi = DummyForecastInterface()
        forecasts, ctx_hit = agent._run_qwen_action_loop(
            messages=[{"role": "user", "content": "test"}],
            forecast_interface=fi,
            current_date=date(2025, 6, 1),
        )
        assert not ctx_hit
        assert agent._memory_phase_completed is True
        assert agent._memory.entry_count == 1
        assert agent._memory.retrieve("lesson-1") is not None

    def test_next_ends_day_when_no_memory(self, tmp_path):
        """Without memory, next_day should just break the loop."""
        responses = [
            make_tool_call_response("next_day", {}, call_id="call_next"),
        ]
        cfg = AgentConfig(enable_memory=False)
        agent = ScriptedQwenAgent(responses, config=cfg)
        agent._memory = None

        fi = DummyForecastInterface()
        forecasts, ctx_hit = agent._run_qwen_action_loop(
            messages=[{"role": "user", "content": "test"}],
            forecast_interface=fi,
            current_date=date(2025, 6, 1),
        )
        assert agent._memory_phase_completed is False

    def test_qwen_call_chat_json_preserves_explicit_tool_choice(self):
        inference = RecordingInference()
        agent = QwenBasicAgent(
            agent_id="tool_choice_test",
            inference_provider=inference,
            config=AgentConfig(),
        )

        agent._call_chat_json(
            messages=[{"role": "user", "content": "test"}],
            tools=[{"function": {"name": "mem_add"}}],
            sampling_params={"tool_choice": "required", "max_tokens": 111},
        )

        assert inference.calls[0]["sampling_params"]["tool_choice"] == "required"

    def test_qwen_call_chat_json_defaults_to_auto_tool_choice(self):
        inference = RecordingInference()
        agent = QwenBasicAgent(
            agent_id="tool_choice_test",
            inference_provider=inference,
            config=AgentConfig(),
        )

        agent._call_chat_json(
            messages=[{"role": "user", "content": "test"}],
            tools=[{"function": {"name": "mem_add"}}],
            sampling_params={"max_tokens": 111},
        )

        assert inference.calls[0]["sampling_params"]["tool_choice"] == "auto"

    def test_next_ends_day_in_memory_phase(self, tmp_path):
        """Second next_day in memory phase should actually end the loop."""
        responses = [
            make_tool_call_response("next_day", {}, call_id="call_1"),  # → memory phase
            make_tool_call_response("next_day", {}, call_id="call_2"),  # → end
        ]
        agent = ScriptedQwenAgent(responses)
        agent._memory = ActiveMemory("test", str(tmp_path))
        agent._memory.set_date(date(2025, 6, 1))

        fi = DummyForecastInterface()
        agent._run_qwen_action_loop(
            messages=[{"role": "user", "content": "test"}],
            forecast_interface=fi,
            current_date=date(2025, 6, 1),
        )
        assert agent._memory_phase_completed is True
        assert agent._call_index == 2  # both calls consumed

    def test_memory_tools_only_during_memory_phase(self, tmp_path):
        """During memory phase, tools should be memory-only (no query/search/submit)."""
        responses = [
            make_tool_call_response("next_day", {}, call_id="call_1"),  # → memory phase
            make_tool_call_response("memory_new", {
                "name": "x", "description": "d", "content": "c",
            }, call_id="call_2"),
            make_tool_call_response("next_day", {}, call_id="call_3"),  # → end
        ]
        agent = ScriptedQwenAgent(responses)
        agent._memory = ActiveMemory("test", str(tmp_path))
        agent._memory.set_date(date(2025, 6, 1))

        fi = DummyForecastInterface()
        agent._run_qwen_action_loop(
            messages=[{"role": "user", "content": "test"}],
            forecast_interface=fi,
            current_date=date(2025, 6, 1),
        )

        # First call: full tools (includes query_df, submit_forecasts, etc.)
        first_call_tool_names = {t["function"]["name"] for t in agent._tools_per_call[0]}
        assert "submit_forecasts" in first_call_tool_names
        assert "query_df" in first_call_tool_names

        # Second call (memory phase): memory tools only
        second_call_tool_names = {t["function"]["name"] for t in agent._tools_per_call[1]}
        assert "memory_new" in second_call_tool_names
        assert "mem_add" in second_call_tool_names
        assert "next_day" in second_call_tool_names
        assert "submit_forecasts" not in second_call_tool_names
        assert "query_df" not in second_call_tool_names
        assert "search_news" not in second_call_tool_names

    def test_memory_phase_completed_flag_set(self, tmp_path):
        responses = [
            make_tool_call_response("next_day", {}, call_id="call_1"),
            make_tool_call_response("next_day", {}, call_id="call_2"),
        ]
        agent = ScriptedQwenAgent(responses)
        agent._memory = ActiveMemory("test", str(tmp_path))
        agent._memory.set_date(date(2025, 6, 1))

        fi = DummyForecastInterface()
        agent._run_qwen_action_loop(
            messages=[{"role": "user", "content": "test"}],
            forecast_interface=fi,
            current_date=date(2025, 6, 1),
        )
        assert agent._memory_phase_completed is True

    def test_no_memory_phase_for_warmup(self, tmp_path):
        """Warmup loops still skip the normal daily memory-phase transition."""
        responses = [
            make_tool_call_response("next_day", {}, call_id="call_1"),
        ]
        agent = ScriptedQwenAgent(responses)
        agent._memory = ActiveMemory("test", str(tmp_path))
        agent._memory.set_date(date(2025, 6, 1))

        fi = DummyForecastInterface()
        # current_date=None (default) → no memory phase
        agent._run_qwen_action_loop(
            messages=[{"role": "user", "content": "test"}],
            forecast_interface=fi,
        )
        assert agent._memory_phase_completed is False


# ── E. Warmup Memory Finalization Tests ──────────────────────────────────


class TestQwenWarmupMemoryFinalization:
    def _make_question(self, qid: str = "Q123"):
        return SimpleNamespace(
            qid=qid,
            title="Who wins the election?",
            background="Background",
            resolution_criteria="Criteria",
            answer_type="discrete",
        )

    def test_active_warmup_submit_followed_by_same_context_mem_add(self, tmp_path):
        responses = [
            make_tool_call_response(
                "submit_forecasts",
                {"forecasts": [{"qid": "Q123", "outcomes": {"Alice": 0.7, "Bob": 0.3}}]},
                call_id="submit_call",
            ),
            make_tool_call_response(
                "mem_add",
                {
                    "qid": "Q123",
                    "question": "Who wins the election?",
                    "memory": "Predicted Alice 0.70 after reviewing recent polling and coalition math.",
                    "category": "politics",
                },
                call_id="mem_call",
            ),
        ]
        cfg = AgentConfig(
            enable_memory=True,
            memory_format="active",
            memory_dir=str(tmp_path),
            warmup_max_actions=4,
            warmup_max_total_tokens=40000,
            warmup_submit_reserve_tokens=1024,
            warmup_force_submit_threshold_tokens=2048,
        )
        agent = ScriptedQwenAllQAgent(responses, config=cfg)
        agent._memory = ActiveMemory("warmup", str(tmp_path))
        agent._memory.set_date(date(2025, 1, 1))
        agent._warmup_mem_entries = []

        fi = DummyForecastInterface()
        agent._process_single_question(self._make_question(), date(2025, 1, 1), fi)

        assert len(fi.submitted) == 1
        assert len(agent._warmup_mem_entries) == 1
        entry = agent._warmup_mem_entries[0]
        assert entry["qid"] == "Q123"
        assert entry["category"] == "politics"
        assert "Alice 0.70" in entry["memory"]

        assert len(agent._messages_per_call) == 2
        second_call_messages = agent._messages_per_call[1]
        assert any(msg["role"] == "assistant" for msg in second_call_messages)
        assert any(msg["role"] == "tool" and msg.get("name") == "submit_forecasts" for msg in second_call_messages)
        second_tool_names = {tool["function"]["name"] for tool in agent._tools_per_call[1]}
        assert second_tool_names == {"mem_add"}
        assert agent._sampling_params_per_call[1]["max_tokens"] == agent.WARMUP_MEMORY_MAX_OUTPUT_TOKENS
        assert agent._sampling_params_per_call[1]["tool_choice"] == "required"
        assert "Call exactly one tool now: `mem_add`." in second_call_messages[-1]["content"]

    def test_structured_warmup_submit_followed_by_memory_new(self, tmp_path):
        responses = [
            make_tool_call_response(
                "submit_forecasts",
                {"forecasts": [{"qid": "Q123", "outcomes": {"Alice": 0.8}}]},
                call_id="submit_call",
            ),
            make_tool_call_response(
                "memory_new",
                {
                    "name": "q123-election-read",
                    "description": "Question Q123: polling and coalition evidence",
                    "content": "Alice leads on both polling and coalition formation paths.",
                },
                call_id="memory_call",
            ),
        ]
        cfg = AgentConfig(
            enable_memory=True,
            memory_format="structured",
            memory_dir=str(tmp_path),
            warmup_max_actions=4,
            warmup_max_total_tokens=40000,
        )
        agent = ScriptedQwenAllQAgent(responses, config=cfg)
        agent._memory = StructuredMemory("warmup", str(tmp_path))
        agent._memory.set_date(date(2025, 1, 1))
        agent._warmup_structured_entries = []

        fi = DummyForecastInterface()
        agent._process_single_question(self._make_question(), date(2025, 1, 1), fi)

        assert len(agent._warmup_structured_entries) == 1
        entry = agent._warmup_structured_entries[0]
        assert entry["name"] == "q123-election-read"
        assert "Q123" in entry["description"]
        assert "Alice leads" in entry["content"]
        second_tool_names = {tool["function"]["name"] for tool in agent._tools_per_call[1]}
        assert second_tool_names == {"memory_new"}
        assert agent._sampling_params_per_call[1]["tool_choice"] == "required"
        assert "Call exactly one tool now: `memory_new`." in agent._messages_per_call[1][-1]["content"]

    def test_warmup_without_memory_does_not_add_finalization_call(self):
        responses = [
            make_tool_call_response(
                "submit_forecasts",
                {"forecasts": [{"qid": "Q123", "outcomes": {"Alice": 1.0}}]},
                call_id="submit_call",
            ),
        ]
        cfg = AgentConfig(enable_memory=False, warmup_max_actions=4)
        agent = ScriptedQwenAllQAgent(responses, config=cfg)

        fi = DummyForecastInterface()
        agent._process_single_question(self._make_question(), date(2025, 1, 1), fi)

        assert len(fi.submitted) == 1
        assert agent._call_index == 1

    def test_warmup_memory_invalid_tool_retries_once_then_placeholder(self, tmp_path):
        responses = [
            make_tool_call_response(
                "submit_forecasts",
                {"forecasts": [{"qid": "Q123", "outcomes": {"Alice": 0.6}}]},
                call_id="submit_call",
            ),
            make_no_tool_response(content="I think Alice is ahead."),
            make_no_tool_response(content="Still no tool."),
        ]
        agent = ScriptedQwenAllQAgent(responses)
        agent._memory = ActiveMemory("warmup", str(tmp_path))
        agent._memory.set_date(date(2025, 1, 1))
        agent._warmup_mem_entries = []

        fi = DummyForecastInterface()
        agent._process_single_question(self._make_question(), date(2025, 1, 1), fi)

        assert agent._call_index == 3
        assert len(agent._warmup_mem_entries) == 1
        assert "warmup placeholder" in agent._warmup_mem_entries[0]["memory"].lower()
        assert "Alice=0.60" in agent._warmup_mem_entries[0]["memory"]

    def test_warmup_memory_invalid_tool_falls_back_to_xml_parser(self, tmp_path):
        responses = [
            make_tool_call_response(
                "submit_forecasts",
                {"forecasts": [{"qid": "Q123", "outcomes": {"Alice": 0.6}}]},
                call_id="submit_call",
            ),
            make_no_tool_response(content="I think Alice is ahead."),
            make_no_tool_response(content="Still no tool."),
        ]
        agent = ScriptedQwenAllQAgent(responses)
        agent.inference = XMLWarmupInference([
            (
                "<mem_add>\n"
                "qid: Q123\n"
                "question: Who wins the election?\n"
                "memory: Alice leads after recent polling and coalition math.\n"
                "category: politics\n"
                "</mem_add>",
                {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            )
        ])
        agent._memory = ActiveMemory("warmup", str(tmp_path))
        agent._memory.set_date(date(2025, 1, 1))
        agent._warmup_mem_entries = []

        fi = DummyForecastInterface()
        agent._process_single_question(self._make_question(), date(2025, 1, 1), fi)

        assert agent._call_index == 3
        assert len(agent._warmup_mem_entries) == 1
        entry = agent._warmup_mem_entries[0]
        assert entry["qid"] == "Q123"
        assert entry["category"] == "politics"
        assert "coalition math" in entry["memory"]
        assert len(agent.inference.calls) == 1
        assert "write exactly one question-specific memory entry" in agent.inference.calls[0]["messages"][-1]["content"].lower()

    def test_warmup_memory_transport_timeout_retries_once_then_placeholder(self, tmp_path):
        responses = [
            make_tool_call_response(
                "submit_forecasts",
                {"forecasts": [{"qid": "Q123", "outcomes": {"Alice": 0.6}}]},
                call_id="submit_call",
            ),
            RuntimeError("vLLM chat/completions timeout after 300s"),
            RuntimeError("vLLM chat/completions timeout after 300s"),
        ]
        agent = ScriptedQwenAllQAgent(responses)
        agent._memory = ActiveMemory("warmup", str(tmp_path))
        agent._memory.set_date(date(2025, 1, 1))
        agent._warmup_mem_entries = []

        fi = DummyForecastInterface()
        agent._process_single_question(self._make_question(), date(2025, 1, 1), fi)

        assert agent._call_index == 3
        assert len(agent._warmup_mem_entries) == 1
        assert "warmup placeholder" in agent._warmup_mem_entries[0]["memory"].lower()
        assert "Alice=0.60" in agent._warmup_mem_entries[0]["memory"]

    def test_warmup_budget_reserve_includes_memory_headroom(self, tmp_path):
        cfg = AgentConfig(
            enable_memory=True,
            memory_format="active",
            memory_dir=str(tmp_path),
            warmup_max_total_tokens=12000,
            warmup_submit_reserve_tokens=1024,
            warmup_force_submit_threshold_tokens=2048,
        )
        agent = ScriptedQwenAllQAgent([], config=cfg)
        agent._memory = ActiveMemory("warmup", str(tmp_path))
        agent._memory.set_date(date(2025, 1, 1))

        budget = agent._create_warmup_budget_tracker()

        assert budget.settings.submit_reserve_tokens == 1024 + agent.WARMUP_MEMORY_TOKEN_RESERVE
        assert budget.settings.force_submit_threshold_tokens == 2048 + agent.WARMUP_MEMORY_TOKEN_RESERVE


# ── F. End-to-End Memory Workflow Tests ──────────────────────────────────


class TestQwenMemoryEndToEnd:
    """Full act() with scripted responses, testing memory persistence."""

    def test_full_day_with_memory_phase(self, tmp_path):
        """Active memory: act() → action loop with memory phase → memory persisted to disk."""
        responses = [
            make_tool_call_response("next_day", {}, call_id="call_1"),
            make_tool_call_response("mem_add", {
                "qid": "Q42", "question": "Will X?",
                "memory": "Key evidence for Q42", "category": "politics",
            }, call_id="call_2"),
            make_tool_call_response("memory_new", {
                "name": "lesson-calibration",
                "description": "Calibration lesson from today",
                "content": "Sports questions with odds resolve 80% correctly.",
            }, call_id="call_3"),
            make_tool_call_response("next_day", {}, call_id="call_4"),
        ]
        cfg = AgentConfig(
            enable_memory=True,
            memory_format="active",
            memory_dir=str(tmp_path),
            max_total_tokens=100000,
            memory_update_max_total_tokens=40000,
        )
        agent = ScriptedQwenAgent(responses, config=cfg)
        agent._memory = ActiveMemory("test", str(tmp_path))

        fi = DummyForecastInterface()
        forecasts = agent.act(None, fi, date(2025, 6, 1))

        assert agent._memory_phase_completed is True
        assert fi.next_day_calls == 1

        # Verify memory was persisted to disk
        amem2 = ActiveMemory("test", str(tmp_path))
        amem2.set_date(date(2025, 6, 2))
        assert amem2.mem_count == 1
        df = amem2.get_mem_df()
        assert df.iloc[0]["qid"] == "Q42"
        assert amem2.entry_count == 1
        assert amem2.retrieve("lesson-calibration") is not None

    def test_memory_persists_across_days(self, tmp_path):
        """Memory from day 1 should be loadable on day 2."""
        # Day 1: add memory
        day1_responses = [
            make_tool_call_response("next_day", {}, call_id="c1"),
            make_tool_call_response("mem_add", {
                "qid": "Q10", "memory": "Day 1 evidence",
            }, call_id="c2"),
            make_tool_call_response("next_day", {}, call_id="c3"),
        ]
        cfg = AgentConfig(
            enable_memory=True, memory_format="active",
            memory_dir=str(tmp_path),
            max_total_tokens=100000, memory_update_max_total_tokens=40000,
        )
        agent = ScriptedQwenAgent(day1_responses, config=cfg)
        agent._memory = ActiveMemory("test", str(tmp_path))
        fi = DummyForecastInterface()
        agent.act(None, fi, date(2025, 6, 1))

        # Day 2: verify memory loads
        day2_responses = [
            make_tool_call_response("next_day", {}, call_id="c4"),
            make_tool_call_response("next_day", {}, call_id="c5"),
        ]
        agent2 = ScriptedQwenAgent(day2_responses, config=cfg)
        agent2._memory = ActiveMemory("test", str(tmp_path))
        fi2 = DummyForecastInterface()
        agent2.act(None, fi2, date(2025, 6, 2))

        # Memory from day 1 should have been loaded
        assert agent2._memory.mem_count == 1
        df = agent2._memory.get_mem_df()
        assert df.iloc[0]["qid"] == "Q10"

    def test_act_with_active_memory_does_not_call_prompt_memory_update(self, tmp_path):
        """For active memory, _prompt_memory_update should NOT be called (handled in-loop)."""
        responses = [
            make_tool_call_response("next_day", {}, call_id="c1"),
            make_tool_call_response("next_day", {}, call_id="c2"),
        ]
        cfg = AgentConfig(
            enable_memory=True, memory_format="active",
            memory_dir=str(tmp_path),
            max_total_tokens=100000, memory_update_max_total_tokens=40000,
        )

        class TrackingAgent(ScriptedQwenAgent):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.prompt_memory_update_called = False

            def _prompt_memory_update(self, messages, forecast_interface, current_date):
                self.prompt_memory_update_called = True

        agent = TrackingAgent(responses, config=cfg)
        agent._memory = ActiveMemory("test", str(tmp_path))
        fi = DummyForecastInterface()
        agent.act(None, fi, date(2025, 6, 1))

        assert not agent.prompt_memory_update_called

    def test_context_limit_skips_memory(self, tmp_path):
        """If context limit hit, memory phase should not trigger."""
        cfg = AgentConfig(
            enable_memory=True, memory_format="active",
            memory_dir=str(tmp_path),
            max_total_tokens=100000, memory_update_max_total_tokens=40000,
        )

        class ContextLimitAgent(ScriptedQwenAgent):
            def _call_chat_json_with_retries(self, *, messages, tools, sampling_params):
                raise RuntimeError(
                    "vLLM /v1/chat/completions 400 Bad Request: This model's maximum input length is "
                    "122880 tokens. However, you requested 122881 tokens. type=BadRequestError "
                    "parameter=input_tokens code=invalid_value"
                )

        agent = ContextLimitAgent([], config=cfg)
        agent._memory = ActiveMemory("test", str(tmp_path))
        fi = DummyForecastInterface()
        forecasts = agent.act(None, fi, date(2025, 6, 1))

        assert forecasts == []
        assert agent._memory_phase_completed is False


# ── G. Budget Integration Tests ──────────────────────────────────────────


class TestQwenMemoryBudget:
    """Test budget configuration and consumption for memory features."""

    def test_memory_phase_threshold_configured_for_active_memory(self, tmp_path):
        """Budget should have memory_phase_threshold_tokens set for active memory."""
        cfg = AgentConfig(
            enable_memory=True,
            memory_format="active",
            max_total_tokens=100000,
            memory_update_max_total_tokens=40000,
        )
        agent = QwenBasicAgent(
            agent_id="test_budget",
            inference_provider=None,
            config=cfg,
        )
        agent._memory = ActiveMemory("test", str(tmp_path))
        agent._memory.set_date(date(2025, 6, 1))

        settings = agent._get_budget_settings()
        assert settings.memory_phase_threshold_tokens == 40000

    def test_no_memory_threshold_without_structured_memory(self):
        """Budget should not set memory threshold for plain memory."""
        cfg = AgentConfig(
            enable_memory=True,
            memory_format="plain",
            max_total_tokens=100000,
        )
        agent = QwenBasicAgent(
            agent_id="test_budget",
            inference_provider=None,
            config=cfg,
        )
        # BasicMemory (plain) doesn't trigger memory threshold
        settings = agent._get_budget_settings()
        assert settings.memory_phase_threshold_tokens is None

    def test_memory_actions_consume_budget(self, tmp_path):
        """Memory handler methods should consume one action each."""
        cfg = AgentConfig(enable_memory=True, memory_format="active")
        agent = QwenBasicAgent(
            agent_id="test_budget", inference_provider=None, config=cfg,
        )
        agent._memory = ActiveMemory("test", str(tmp_path))
        agent._memory.set_date(date(2025, 6, 1))

        budget = BudgetTracker(BudgetSettings(max_actions=10))
        initial = budget.actions_remaining

        # memory_new
        agent._qwen_handle_memory_action(
            messages=[], budget=budget,
            parsed=ParsedAction(
                action_type="memory_new", code=None, forecasts=None, query=None,
                memory_new_data={"name": "a", "description": "d", "content": "c"},
            ),
            tool_call={"call_id": "c1"},
        )
        assert budget.actions_remaining == initial - 1

        # mem_add
        agent._qwen_handle_mem_action(
            messages=[], budget=budget,
            parsed=ParsedAction(
                action_type="mem_add", code=None, forecasts=None, query=None,
                mem_data={"qid": "Q1", "question": "Q", "memory": "M", "category": "C"},
            ),
            tool_call={"call_id": "c2"},
        )
        assert budget.actions_remaining == initial - 2

    def test_memory_phase_multi_turn_budget(self, tmp_path):
        """Memory phase with multiple tool calls should consume budget correctly."""
        responses = [
            make_tool_call_response("next_day", {}, call_id="c1"),
            make_tool_call_response("mem_add", {
                "qid": "Q1", "memory": "Evidence 1",
            }, call_id="c2"),
            make_tool_call_response("mem_add", {
                "qid": "Q2", "memory": "Evidence 2",
            }, call_id="c3"),
            make_tool_call_response("memory_new", {
                "name": "meta-1", "description": "d", "content": "c",
            }, call_id="c4"),
            make_tool_call_response("next_day", {}, call_id="c5"),
        ]
        agent = ScriptedQwenAgent(responses)
        agent._memory = ActiveMemory("test", str(tmp_path))
        agent._memory.set_date(date(2025, 6, 1))

        fi = DummyForecastInterface()
        agent._run_qwen_action_loop(
            messages=[{"role": "user", "content": "test"}],
            forecast_interface=fi,
            current_date=date(2025, 6, 1),
        )
        # Should have processed all 5 responses
        assert agent._call_index == 5
        assert agent._memory.mem_count == 2
        assert agent._memory.entry_count == 1


# ── H. _memory_tool_call_to_action unit tests ────────────────────────────


class TestMemoryToolCallToAction:
    """Direct unit tests for the _memory_tool_call_to_action helper."""

    def test_returns_none_for_non_memory_tools(self):
        assert _memory_tool_call_to_action("query_df", {"code": "x"}) is None
        assert _memory_tool_call_to_action("search_news", {"query": "q"}) is None
        assert _memory_tool_call_to_action("submit_forecasts", {}) is None
        assert _memory_tool_call_to_action("next_day", {}) is None
        assert _memory_tool_call_to_action("unknown", {}) is None

    def test_memory_retrieve_valid(self):
        result = _memory_tool_call_to_action("memory_retrieve", {"name": "my-entry"})
        assert result.action_type == "memory_retrieve"
        assert result.memory_entry_name == "my-entry"
        assert result.error is None

    def test_memory_retrieve_empty_name(self):
        result = _memory_tool_call_to_action("memory_retrieve", {"name": ""})
        assert result.error is not None

    def test_memory_retrieve_no_name(self):
        result = _memory_tool_call_to_action("memory_retrieve", {})
        assert result.error is not None

    def test_memory_new_valid(self):
        result = _memory_tool_call_to_action("memory_new", {
            "name": "test", "description": "desc", "content": "body",
        })
        assert result.action_type == "memory_new"
        assert result.memory_new_data["name"] == "test"
        assert result.error is None

    def test_memory_new_missing_name(self):
        result = _memory_tool_call_to_action("memory_new", {
            "description": "d", "content": "c",
        })
        assert result.error is not None

    def test_mem_add_valid(self):
        result = _memory_tool_call_to_action("mem_add", {
            "qid": "Q1", "memory": "evidence",
        })
        assert result.action_type == "mem_add"
        assert result.mem_data["qid"] == "Q1"
        assert result.error is None

    def test_mem_update_valid(self):
        result = _memory_tool_call_to_action("mem_update", {
            "qid": "Q1", "memory": "updated",
        })
        assert result.action_type == "mem_update"
        assert result.mem_qid == "Q1"
        assert result.error is None

    def test_mem_delete_valid(self):
        result = _memory_tool_call_to_action("mem_delete", {"qid": "Q1"})
        assert result.action_type == "mem_delete"
        assert result.mem_qid == "Q1"
        assert result.error is None

    def test_mem_update_empty_category_is_none(self):
        result = _memory_tool_call_to_action("mem_update", {
            "qid": "Q1", "memory": "m", "category": "",
        })
        assert result.mem_data["category"] is None

    def test_mem_add_defaults_for_optional_fields(self):
        result = _memory_tool_call_to_action("mem_add", {
            "qid": "Q1", "memory": "m",
        })
        assert result.mem_data["question"] == ""
        assert result.mem_data["category"] == ""
