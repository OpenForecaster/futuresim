"""Tests for AllQ warmup memory finalization using Chat Completions tool calls."""

import json
from datetime import date
from types import SimpleNamespace
from typing import Any, Dict, List

from agents.allQAgent.agent import AllQAgent
from agents.basicAgent import AgentConfig
from agents.utils.memory import ActiveMemory


def make_submit_tool_response(qid: str) -> Dict[str, Any]:
    """Build a Chat Completions response with a submit_forecasts tool call."""
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_submit",
                    "type": "function",
                    "function": {
                        "name": "submit_forecasts",
                        "arguments": json.dumps({
                            "forecasts": [{"qid": qid, "outcomes": {"Alice": 0.7, "Bob": 0.3}}]
                        }),
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130},
    }


def make_mem_add_tool_response(qid: str, question: str) -> Dict[str, Any]:
    """Build a Chat Completions response with a mem_add tool call."""
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_mem",
                    "type": "function",
                    "function": {
                        "name": "mem_add",
                        "arguments": json.dumps({
                            "qid": qid,
                            "question": question,
                            "memory": "Predicted Alice 0.70 after reviewing recent polling and coalition math.",
                            "category": "politics",
                        }),
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 120, "completion_tokens": 40, "total_tokens": 160},
    }


class DummyForecastInterface:
    def __init__(self):
        self.logs = []
        self.submitted = []

    def log_model_output(self, prompt, response, metadata):
        self.logs.append({"prompt": prompt, "response": response, "metadata": metadata})

    def submit_prediction(self, pred):
        self.submitted.append(pred)


class ScriptedChatInference:
    """Inference provider that returns scripted Chat Completions JSON responses."""

    def __init__(self, responses: List[Any]):
        self.responses = list(responses)
        self.call_index = 0
        self.messages_per_call: List[List[Dict]] = []
        self.sampling_params_per_call: List[Dict] = []

    def chat_json(self, messages, sampling_params):
        self.messages_per_call.append(list(messages))
        self.sampling_params_per_call.append(dict(sampling_params))
        if self.call_index >= len(self.responses):
            raise RuntimeError("No more scripted responses")
        response = self.responses[self.call_index]
        self.call_index += 1
        if isinstance(response, Exception):
            raise response
        return response


class ScriptedAllQAgent(AllQAgent):
    def __init__(self, inference, config, **kwargs):
        super().__init__(
            agent_id="test_allq_warmup",
            inference_provider=inference,
            config=config,
            start_date=date(2025, 1, 1),
            **kwargs,
        )

    def _setup_warmup_day(self, forecast_interface, current_date):
        self._forecast_interface = forecast_interface

    def _build_warmup_system_prompt(self, current_date, q, forecast_interface=None):
        return f"Warmup question {q.qid}: {q.title}"

    def _build_start_budget_status(self, *, warmup=False, max_actions_override=None):
        return ""


def test_allq_active_warmup_uses_same_context_memory_finalization(tmp_path):
    inference = ScriptedChatInference([
        make_submit_tool_response("Q123"),
        make_mem_add_tool_response("Q123", "Who wins the election?"),
    ])
    cfg = AgentConfig(
        enable_memory=True,
        memory_format="active",
        memory_dir=str(tmp_path),
        warmup_max_actions=4,
        warmup_max_total_tokens=40000,
        warmup_submit_reserve_tokens=1024,
        warmup_force_submit_threshold_tokens=2048,
    )
    agent = ScriptedAllQAgent(inference, cfg)
    agent._memory = ActiveMemory("warmup", str(tmp_path))
    agent._memory.set_date(date(2025, 1, 1))
    agent._warmup_mem_entries = []

    fi = DummyForecastInterface()
    question = SimpleNamespace(
        qid="Q123",
        title="Who wins the election?",
        background="Background",
        resolution_criteria="Criteria",
        answer_type="discrete",
    )

    agent._process_single_question(question, date(2025, 1, 1), fi)

    assert len(fi.submitted) == 1
    assert len(agent._warmup_mem_entries) == 1
    assert agent._warmup_mem_entries[0]["qid"] == "Q123"
    assert "Alice 0.70" in agent._warmup_mem_entries[0]["memory"]
    assert len(inference.messages_per_call) == 2
    assert len(inference.messages_per_call[1]) > len(inference.messages_per_call[0])


def test_allq_warmup_memory_transport_failure_falls_back_once(tmp_path):
    inference = ScriptedChatInference([
        make_submit_tool_response("Q123"),
        RuntimeError("vLLM chat/completions timeout after 300s"),
        RuntimeError("vLLM chat/completions timeout after 300s"),
    ])
    cfg = AgentConfig(
        enable_memory=True,
        memory_format="active",
        memory_dir=str(tmp_path),
        warmup_max_actions=4,
        warmup_max_total_tokens=40000,
    )
    agent = ScriptedAllQAgent(inference, cfg)
    agent._memory = ActiveMemory("warmup", str(tmp_path))
    agent._memory.set_date(date(2025, 1, 1))
    agent._warmup_mem_entries = []

    fi = DummyForecastInterface()
    question = SimpleNamespace(
        qid="Q123",
        title="Who wins the election?",
        background="Background",
        resolution_criteria="Criteria",
        answer_type="discrete",
    )

    agent._process_single_question(question, date(2025, 1, 1), fi)

    assert inference.call_index == 3
    assert len(agent._warmup_mem_entries) == 1
    assert "warmup placeholder" in agent._warmup_mem_entries[0]["memory"].lower()
    assert "Alice=0.70" in agent._warmup_mem_entries[0]["memory"]
