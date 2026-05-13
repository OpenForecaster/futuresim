"""Coverage for BasicAgent prompt builders that aren't otherwise exercised.

Targets gaps left after the agent.py refactor (extracted prompts.py /
memory_prompts.py): structured/plain memory daily prompts, multi-agent and
metaculus prompt branches, imminent-resolution reminder, memory-phase
dispatcher, resolution recap, and a couple of pure helpers.
"""

from dataclasses import dataclass
from datetime import date
from types import SimpleNamespace

import pytest

from agents.basicAgent import AgentConfig, BasicAgent
from agents.basicAgent.prompts import BasicPromptBuilder
from agents.utils.memory import ActiveMemory, StructuredMemory


@dataclass
class _Question:
    resolution_date: date


class _ForecastStub:
    """Minimal forecast_interface compatible with prompts + feedback handler."""

    def __init__(self, *, source_name="openforesight", num_agents=1, questions=None,
                 last_active=date(2025, 1, 1), next_active=date(2025, 1, 11)):
        self.source_name = source_name
        self.source_context = ""
        self.num_agents = num_agents
        self.histories = {}
        self.questions = questions if questions is not None else {"Q1": _Question(date(2099, 1, 1))}
        self.last_active_date = last_active
        self.next_active_date = next_active
        self.resolution_events = []

    def get_agent_predictions(self, agent_id):
        return {}


def _make_agent(config: AgentConfig, *, forecast=None) -> BasicAgent:
    agent = BasicAgent(
        agent_id="prompt_test_agent",
        inference_provider=object(),
        config=config,
        model_name="test-model",
    )
    agent._forecast_interface = forecast or _ForecastStub()
    return agent


# ---------------------------------------------------------------------------
# _build_instructions branches
# ---------------------------------------------------------------------------

class TestBuildInstructionsMemoryBranches:
    def test_structured_memory_section_rendered(self, tmp_path):
        cfg = AgentConfig(
            enable_memory=True, memory_format="structured",
            memory_dir=str(tmp_path), timegap_days=3,
        )
        agent = _make_agent(cfg)
        agent._memory.set_date(date(2025, 1, 6))

        prompt = agent._build_instructions(date(2025, 1, 6))

        assert "## YOUR MEMORY" in prompt
        assert "memory_retrieve" in prompt
        assert "memory_new" in prompt
        assert "memory_update" in prompt
        assert "memory_delete" in prompt
        assert "mem_df" not in prompt  # active-memory column, not structured
        assert "mem_add" not in prompt
        assert "every 3 days" in prompt

    def test_plain_memory_includes_legacy_memory_block(self, tmp_path):
        cfg = AgentConfig(
            enable_memory=True, memory_format="plain", memory_dir=str(tmp_path),
        )
        agent = _make_agent(cfg)
        agent._memory.set_date(date(2025, 1, 6))
        agent._memory.update("REMEMBERED CONTEXT 12345")

        prompt = agent._build_instructions(date(2025, 1, 6))

        assert "REMEMBERED CONTEXT 12345" in prompt
        assert "Use the reasoning and insights above" in prompt
        # Plain memory should NOT advertise structured tool calls
        assert "memory_retrieve" not in prompt
        assert "mem_add" not in prompt


class TestBuildInstructionsMultiAgent:
    def test_multi_agent_block_and_peer_scoring_rendered(self):
        cfg = AgentConfig(enable_memory=False, single_agent_mode=False, timegap_days=1)
        agent = _make_agent(cfg, forecast=_ForecastStub(num_agents=3))

        prompt = agent._build_instructions(date(2025, 1, 6))

        assert "## MULTI-AGENT SETTING" in prompt
        assert "2 other forecasting agents" in prompt
        assert "Time-Weighted Peer Score" in prompt or "TW-Peer" in prompt
        assert "market_aggregate" in prompt

    def test_single_agent_drops_peer_language(self):
        cfg = AgentConfig(enable_memory=False, single_agent_mode=True)
        agent = _make_agent(cfg)

        prompt = agent._build_instructions(date(2025, 1, 6))

        assert "## MULTI-AGENT SETTING" not in prompt
        assert "market_aggregate" not in prompt
        assert "TW-Peer" not in prompt


class TestBuildInstructionsSourceVariants:
    def test_metaculus_binary_uses_binary_brier_and_yes_no_rules(self):
        cfg = AgentConfig(enable_memory=False)
        agent = _make_agent(cfg, forecast=_ForecastStub(source_name="metaculus_binary"))

        prompt = agent._build_instructions(date(2025, 1, 6))

        assert "## BINARY QUESTION RULES" in prompt
        assert '"Yes"' in prompt and '"No"' in prompt
        # Binary brier section, not the brier-skill section.
        assert "## SCORING (Brier Score, Binary)" in prompt
        assert "Brier Score = (p - y)^2" in prompt

    def test_metaculus_mcq_uses_multiple_choice_rules(self):
        cfg = AgentConfig(enable_memory=False)
        agent = _make_agent(cfg, forecast=_ForecastStub(source_name="metaculus_mcq"))

        prompt = agent._build_instructions(date(2025, 1, 6))

        assert "## MULTIPLE CHOICE RULES" in prompt
        assert "EXACT option text" in prompt
        # Default Brier-Skill scoring, not binary.
        assert "Brier Skill Score" in prompt


class TestBuildInstructionsImminentReminder:
    def test_questions_resolving_tomorrow_trigger_important_reminder(self):
        tomorrow = date(2025, 1, 7)
        questions = {
            "Q-imminent-1": _Question(tomorrow),
            "Q-imminent-2": _Question(tomorrow),
            "Q-future": _Question(date(2025, 12, 31)),
        }
        agent = _make_agent(
            AgentConfig(enable_memory=False),
            forecast=_ForecastStub(questions=questions),
        )

        prompt = agent._build_instructions(date(2025, 1, 6))

        assert "**IMPORTANT**" in prompt
        assert "2 question(s) resolve tomorrow" in prompt
        assert "Q-imminent-1" in prompt
        assert "Q-imminent-2" in prompt
        # Non-imminent qid must not appear in the reminder list.
        assert "Q-future" not in prompt.split("**IMPORTANT**", 1)[1].split("\n", 1)[0]

    def test_no_imminent_qids_renders_default_branch(self):
        questions = {"Q-future": _Question(date(2025, 12, 31))}
        agent = _make_agent(
            AgentConfig(enable_memory=False),
            forecast=_ForecastStub(questions=questions),
        )

        prompt = agent._build_instructions(date(2025, 1, 6))

        assert "No questions resolve tomorrow" in prompt
        assert "**IMPORTANT**" not in prompt


# ---------------------------------------------------------------------------
# memory_prompts.py — dispatcher, plain prompt, resolution recap
# ---------------------------------------------------------------------------

class TestMemoryPhaseDispatcher:
    def test_dispatch_to_structured(self, tmp_path):
        agent = _make_agent(AgentConfig(
            enable_memory=True, memory_format="structured",
            memory_dir=str(tmp_path),
        ))
        agent._memory.set_date(date(2025, 1, 6))

        prompt = agent._build_memory_phase_prompt(date(2025, 1, 6))

        assert prompt == agent._build_structured_memory_prompt(date(2025, 1, 6))
        assert "STEP 1" in prompt  # Structured prompt marker.

    def test_dispatch_to_active(self, tmp_path):
        agent = _make_agent(AgentConfig(
            enable_memory=True, memory_format="active",
            memory_dir=str(tmp_path),
        ))
        agent._memory.set_date(date(2025, 1, 6))

        prompt = agent._build_memory_phase_prompt(date(2025, 1, 6))

        assert "mem_add" in prompt
        assert "QUESTION-SPECIFIC NOTES" in prompt

    def test_dispatch_returns_empty_for_plain_memory(self, tmp_path):
        agent = _make_agent(AgentConfig(
            enable_memory=True, memory_format="plain", memory_dir=str(tmp_path),
        ))
        agent._memory.set_date(date(2025, 1, 6))

        # Plain memory has no in-loop memory phase; dispatcher returns "".
        assert agent._build_memory_phase_prompt(date(2025, 1, 6)) == ""

    def test_dispatch_returns_empty_when_memory_disabled(self):
        agent = _make_agent(AgentConfig(enable_memory=False))
        assert agent._build_memory_phase_prompt(date(2025, 1, 6)) == ""


class TestBuildPlainMemoryPrompt:
    def test_plain_memory_prompt_contents(self, tmp_path):
        cfg = AgentConfig(enable_memory=True, memory_format="plain",
                          memory_dir=str(tmp_path))
        agent = _make_agent(cfg)
        agent._memory.update("EXISTING NOTES")
        agent._memory.set_date(date(2025, 1, 6))

        prompt = agent._build_plain_memory_prompt(date(2025, 1, 6))

        assert "## MEMORY UPDATE" in prompt
        assert "NO_MEMORY_UPDATE" in prompt
        assert "End of session 2025-01-06" in prompt
        assert "Current memory length: " in prompt


class TestResolutionRecap:
    def test_no_feedback_renders_no_resolutions_line(self):
        agent = _make_agent(AgentConfig(enable_memory=False))
        text = agent._build_resolution_recap_for_memory()
        assert "No questions resolved this session" in text

    def test_feedback_with_resolved_questions_lists_each(self):
        agent = _make_agent(AgentConfig(enable_memory=False))
        agent._last_feedback_data = {
            "resolved_today": [
                {
                    "qid": "Q-A", "title": "Alpha?",
                    "ground_truth": "Yes", "brier": -0.10,
                    "my_pred_distribution": {"Yes": 0.7, "No": 0.3},
                },
                {
                    "qid": "Q-B", "title": "Beta?",
                    "ground_truth": "No", "brier": +0.50,
                    "my_pred_distribution": {"Yes": 0.2, "No": 0.8},
                },
            ],
        }
        text = agent._build_resolution_recap_for_memory()

        assert "QUESTIONS RESOLVED THIS SESSION" in text
        assert "Q-A" in text and "Alpha?" in text
        assert "Q-B" in text and "Beta?" in text
        assert "Truth: Yes" in text
        assert "Truth: No" in text
        # Both Brier numbers should appear with sign.
        assert "-0.10" in text or "-0.1" in text
        assert "+0.50" in text


# ---------------------------------------------------------------------------
# Small pure helpers — exercise both branches directly
# ---------------------------------------------------------------------------

class TestPromptHelpers:
    def test_normalize_prompt_heading_spacing_inserts_blank_lines(self):
        raw = "## A\nbody1\n## B\nbody2"
        out = BasicPromptBuilder._normalize_prompt_heading_spacing(raw)
        # Each ## section heading should be preceded by exactly "\n\n\n".
        assert "body1\n\n\n## B" in out
        # First heading at start of string remains untouched.
        assert out.startswith("## A")

    def test_normalize_prompt_heading_spacing_handles_empty_input(self):
        assert BasicPromptBuilder._normalize_prompt_heading_spacing("") == ""

    def test_get_data_notes_single_vs_multi(self):
        single = _make_agent(AgentConfig(enable_memory=False, single_agent_mode=True))
        multi = _make_agent(AgentConfig(enable_memory=False, single_agent_mode=False))

        single_text = single._get_data_notes()
        multi_text = multi._get_data_notes()

        assert "market_aggregate" not in single_text
        assert "market_aggregate" in multi_text
        assert "num_predictions" in multi_text

    def test_get_multiagent_context_only_when_multi(self):
        single = _make_agent(AgentConfig(enable_memory=False, single_agent_mode=True))
        single_alone = _make_agent(
            AgentConfig(enable_memory=False, single_agent_mode=False),
            forecast=_ForecastStub(num_agents=1),
        )
        multi = _make_agent(
            AgentConfig(enable_memory=False, single_agent_mode=False),
            forecast=_ForecastStub(num_agents=4),
        )

        assert single._get_multiagent_context() == ""
        # single_agent_mode=False but only 1 agent: still no preamble.
        assert single_alone._get_multiagent_context() == ""
        ctx = multi._get_multiagent_context()
        assert "## MULTI-AGENT SETTING" in ctx
        assert "3 other forecasting agents" in ctx

    def test_get_source_rules_branches(self):
        of = _make_agent(AgentConfig(enable_memory=False),
                         forecast=_ForecastStub(source_name="openforesight"))
        binary = _make_agent(AgentConfig(enable_memory=False),
                             forecast=_ForecastStub(source_name="metaculus_binary"))
        mcq = _make_agent(AgentConfig(enable_memory=False),
                          forecast=_ForecastStub(source_name="metaculus_mcq"))

        assert of._get_source_rules() == ""
        assert "BINARY QUESTION RULES" in binary._get_source_rules()
        assert "MULTIPLE CHOICE RULES" in mcq._get_source_rules()
