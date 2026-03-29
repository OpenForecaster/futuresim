from datetime import date
from types import SimpleNamespace

from agents.allQAgent.agent import AllQAgent
from agents.basicAgent import AgentConfig
from agents.utils.memory import ActiveMemory


def make_submit_response(qid: str) -> str:
    return f"""<reasoning>Ready to submit.</reasoning>
<action type="submit">
<forecast qid="{qid}">
  <outcome name="Alice" prob="0.7"/>
  <outcome name="Bob" prob="0.3"/>
</forecast>
</action>"""


def make_mem_add_response(qid: str, question: str) -> str:
    return f"""<mem_add>
qid: {qid}
question: {question}
memory: Predicted Alice 0.70 after reviewing recent polling and coalition math.
category: politics
</mem_add>"""


class DummyForecastInterface:
    def __init__(self):
        self.logs = []
        self.submitted = []

    def log_model_output(self, prompt, response, metadata):
        self.logs.append({"prompt": prompt, "response": response, "metadata": metadata})

    def submit_prediction(self, pred):
        self.submitted.append(pred)


class ScriptedTextInference:
    def __init__(self, responses):
        self.responses = list(responses)
        self.call_index = 0
        self.messages_per_call = []
        self.sampling_params_per_call = []

    def chat(self, messages, sampling_params):
        self.messages_per_call.append(list(messages))
        self.sampling_params_per_call.append(dict(sampling_params))
        if self.call_index >= len(self.responses):
            raise RuntimeError("No more scripted responses")
        response = self.responses[self.call_index]
        self.call_index += 1
        if isinstance(response, Exception):
            raise response
        usage = {
            "prompt_tokens": 100 + self.call_index,
            "completion_tokens": 30,
            "total_tokens": 130 + self.call_index,
        }
        return response, usage


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
    inference = ScriptedTextInference(
        [
            make_submit_response("Q123"),
            make_mem_add_response("Q123", "Who wins the election?"),
        ]
    )
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
    assert inference.sampling_params_per_call[1]["max_tokens"] == agent.WARMUP_MEMORY_MAX_OUTPUT_TOKENS


def test_allq_warmup_memory_transport_failure_falls_back_once(tmp_path):
    inference = ScriptedTextInference(
        [
            make_submit_response("Q123"),
            RuntimeError("vLLM chat/completions timeout after 300s"),
            RuntimeError("vLLM chat/completions timeout after 300s"),
        ]
    )
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
