from datetime import date
from types import SimpleNamespace

import pytest

from agents.basicAgent import AgentConfig
from agents.qwenAgent import QwenAllQAgent, QwenBasicAgent


CONTEXT_LIMIT_ERROR = (
    "vLLM /v1/chat/completions 400 Bad Request: This model's maximum input length is "
    "122880 tokens. However, you requested 122881 tokens. type=BadRequestError "
    "parameter=input_tokens code=invalid_value"
)


class DummyForecastInterface:
    def __init__(self):
        self.logs = []
        self.next_day_calls = 0

    def log_model_output(self, prompt, response, metadata):
        self.logs.append({"prompt": prompt, "response": response, "metadata": metadata})

    def next_day(self):
        self.next_day_calls += 1


class FakeMemory:
    def set_date(self, current_date):
        self.current_date = current_date


class _DummyChatProvider:
    """Minimal inference provider that satisfies _require_chat_tools()."""
    def chat_json(self, messages, sampling_params):
        raise RuntimeError("Should not be called directly in these tests")


class RetryTestAgent(QwenBasicAgent):
    def __init__(self):
        super().__init__(
            agent_id="retry_test",
            inference_provider=_DummyChatProvider(),
            config=AgentConfig(gptoss_responses_max_retries=3),
        )
        self.calls = 0

    def _call_chat_json(self, *, messages, tools, sampling_params):
        self.calls += 1
        raise RuntimeError(CONTEXT_LIMIT_ERROR)


class SessionContextLimitAgent(QwenBasicAgent):
    def __init__(self):
        super().__init__(
            agent_id="session_test",
            inference_provider=_DummyChatProvider(),
            config=AgentConfig(enable_memory=True),
        )
        self._memory = FakeMemory()
        self.memory_update_calls = 0

    def _setup_day(self, forecast_interface, current_date):
        self._forecast_interface = forecast_interface

    def _build_qwen_instructions(self, current_date: date) -> str:
        return "Test session prompt"

    def _call_chat_json_with_retries(self, *, messages, tools, sampling_params):
        raise RuntimeError(CONTEXT_LIMIT_ERROR)

    def _prompt_memory_update(self, messages, forecast_interface, current_date):
        self.memory_update_calls += 1


class WarmupContextLimitAgent(QwenAllQAgent):
    def __init__(self):
        super().__init__(
            agent_id="warmup_test",
            inference_provider=_DummyChatProvider(),
            config=AgentConfig(enable_memory=False, warmup_max_actions=3),
            start_date=date(2025, 1, 1),
        )
        self.mem_calls = 0

    def _call_chat_json_with_retries(self, *, messages, tools, sampling_params):
        raise RuntimeError(CONTEXT_LIMIT_ERROR)

    def _request_warmup_mem(self, messages, qid, question_title):
        self.mem_calls += 1
        return {
            "qid": qid,
            "question": question_title,
            "memory": "should not be used",
        }


def test_qwen_context_limit_errors_are_not_retried():
    agent = RetryTestAgent()

    with pytest.raises(RuntimeError, match="maximum input length"):
        agent._call_chat_json_with_retries(
            messages=[{"role": "user", "content": "prompt"}],
            tools=[{"function": {"name": "submit_forecasts"}}],
            sampling_params={},
        )

    assert agent.calls == 1


def test_qwen_session_context_limit_ends_session_without_memory_update():
    agent = SessionContextLimitAgent()
    forecast_interface = DummyForecastInterface()

    forecasts = agent.act(None, forecast_interface, date(2025, 1, 5))

    assert forecasts == []
    assert agent.memory_update_calls == 0
    assert forecast_interface.next_day_calls == 1
    assert agent._context_limit_hit is True


def test_qwen_warmup_context_limit_creates_placeholder_mem():
    agent = WarmupContextLimitAgent()
    agent._warmup_mem_entries = []
    forecast_interface = DummyForecastInterface()
    question = SimpleNamespace(
        qid="Q123",
        title="Who wins?",
        background="Background",
        resolution_criteria="Criteria",
        answer_type="discrete",
    )

    agent._process_single_question(question, date(2025, 1, 1), forecast_interface)

    assert agent.mem_calls == 0  # LLM not called for mem (context limit)
    assert len(agent._warmup_mem_entries) == 1  # placeholder created
    assert agent._warmup_mem_entries[0]["qid"] == "Q123"
    assert "placeholder" in agent._warmup_mem_entries[0]["memory"].lower() or "no memory" in agent._warmup_mem_entries[0]["memory"].lower()


def test_qwen_warmup_context_limit_creates_structured_placeholder():
    agent = WarmupContextLimitAgent()
    agent._warmup_structured_entries = []
    forecast_interface = DummyForecastInterface()
    question = SimpleNamespace(
        qid="Q123",
        title="Who wins?",
        background="Background",
        resolution_criteria="Criteria",
        answer_type="discrete",
    )

    agent._process_single_question(question, date(2025, 1, 1), forecast_interface)

    assert len(agent._warmup_structured_entries) == 1
    assert agent._warmup_structured_entries[0]["name"] == "qQ123-placeholder"
    assert "warmup placeholder" in agent._warmup_structured_entries[0]["content"].lower()
