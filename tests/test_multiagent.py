"""Comprehensive tests for multiagent run infrastructure.

Covers:
  A. Config parsing & agent creation (create_agents_from_config)
  B. SimulationEnvironment multi-agent orchestration
  C. SimForecastInterface thread safety & prediction submission
  D. Aggregate computation and peer scoring
"""

import os
import threading
from datetime import date
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch, call

import pytest

from agents.base import BaseAgent
from agents.basicAgent.config import AgentConfig
from environment.interfaces import PredictionSubmission
from environment.scoring import (
    DailyPrediction,
    PredictionHistory,
    compute_aggregate,
    compute_snapshot_peer_scores,
    BinaryBrierScorer,
)
from environment.env import SimForecastInterface
from environment.updater import SimLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(**overrides):
    """Return a MagicMock args namespace with sensible defaults."""
    args = MagicMock()
    defaults = dict(
        rate_limit=32.0,
        max_actions=5,
        max_retries=2,
        temperature=0.7,
        max_tokens=1024,
        warmup_max_actions=None,
        max_total_tokens=None,
        warmup_max_total_tokens=None,
        submit_reserve_tokens=8192,
        warmup_submit_reserve_tokens=None,
        force_submit_threshold_tokens=16384,
        warmup_force_submit_threshold_tokens=None,
        warmup_parallelism=20,
        top_p=None,
        top_k=None,
        repetition_penalty=None,
        search_cutoff_days=0,
        resolution_guard=None,
        timegap_days=1,
        resume=None,
        tool_result_keep_last=-1,
        gptoss_prompt_mode="instructions",
        gptoss_reasoning_effort="medium",
        gptoss_include_reasoning=True,
        gptoss_responses_max_retries=3,
        gptoss_retry_backoff_base_s=1.0,
        gptoss_retry_backoff_max_s=16.0,
        sim_start_date=date(2025, 5, 1),
        max_model_len=8192,
        agent_max_model_len=None,
        vllm_gpu_mem=0.3,
        vllm_request_timeout=120.0,
        vllm_max_num_seqs=8,
        vllm_tensor_parallel_size=1,
        vllm_data_parallel_size=1,
        vllm_pipeline_parallel_size=1,
        vllm_enable_expert_parallel=False,
        vllm_all2all_backend=None,
        vllm_startup_timeout=300.0,
        rope_scaling=None,
        vllm_enable_tools=False,
        vllm_tool_call_parser=None,
        vllm_tool_parser_plugin=None,
        vllm_enable_prefix_caching=True,
        agent_cuda_visible_devices=None,
        language_model_only=False,
    )
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(args, k, v)
    return args


def _make_question(qid="q1", title="Will X happen?", resolution_date=None,
                   ground_truth="Yes", options=None):
    from environment.data_loader import Question
    return Question(
        qid=qid,
        title=title,
        background="bg",
        resolution_criteria="criteria",
        answer_type="multiple_choice",
        resolution_date=resolution_date or date(2025, 5, 10),
        ground_truth_answer=ground_truth,
        options=options or ["Yes", "No"],
    )


class StubAgent(BaseAgent):
    """Minimal agent that records act() calls and optionally submits predictions."""

    def __init__(self, agent_id, predictions=None, should_raise=False):
        super().__init__(agent_id=agent_id)
        self.predictions = predictions or {}  # {qid: {outcome: prob}}
        self.should_raise = should_raise
        self.act_calls: List[date] = []

    def act(self, doc_interface, forecast_interface, current_date):
        self.act_calls.append(current_date)
        if self.should_raise:
            raise RuntimeError(f"Agent {self.agent_id} failed intentionally")
        for qid, outcomes in self.predictions.items():
            forecast_interface.submit_prediction(
                PredictionSubmission(question_id=qid, outcomes=outcomes)
            )
        return []


# ===========================================================================
# A. Config parsing & agent creation
# ===========================================================================

class TestCreateAgentsFromConfig:

    @patch("inference.openrouter.OpenRouterInference.__init__", return_value=None)
    def test_basic_config(self, mock_init, tmp_path):
        """Two-agent config returns 2 agents with correct IDs."""
        from scripts.test_basic_agent import create_agents_from_config

        config = {
            "defaults": {"provider": "openrouter", "scaffold": "basic"},
            "agents": [
                {"model": "vendor/model-a"},
                {"model": "vendor/model-b"},
            ],
        }
        agents = create_agents_from_config(config, _make_args(), str(tmp_path), search_tool=None)
        assert len(agents) == 2
        assert agents[0].agent_id == "basic_model-a_001"
        assert agents[1].agent_id == "basic_model-b_001"

    @patch("inference.openrouter.OpenRouterInference.__init__", return_value=None)
    def test_defaults_merge(self, mock_init, tmp_path):
        """Per-agent overrides take precedence over defaults."""
        from scripts.test_basic_agent import create_agents_from_config

        config = {
            "defaults": {"provider": "openrouter", "scaffold": "basic", "temperature": 0.5, "max_actions": 10},
            "agents": [
                {"model": "vendor/model-a", "temperature": 0.9, "max_actions": 3},
            ],
        }
        agents = create_agents_from_config(config, _make_args(), str(tmp_path), search_tool=None)
        assert agents[0].config.sampling_params["temperature"] == 0.9
        assert agents[0].config.max_actions == 3

    @patch("inference.openrouter.OpenRouterInference.__init__", return_value=None)
    def test_unique_ids_same_model(self, mock_init, tmp_path):
        """Three identical agents get sequential IDs."""
        from scripts.test_basic_agent import create_agents_from_config

        config = {
            "defaults": {"provider": "openrouter", "scaffold": "basic"},
            "agents": [
                {"model": "vendor/model-a"},
                {"model": "vendor/model-a"},
                {"model": "vendor/model-a"},
            ],
        }
        agents = create_agents_from_config(config, _make_args(), str(tmp_path), search_tool=None)
        ids = [a.agent_id for a in agents]
        assert ids == ["basic_model-a_001", "basic_model-a_002", "basic_model-a_003"]

    @patch("inference.openrouter.OpenRouterInference.__init__", return_value=None)
    def test_mixed_scaffolds(self, mock_init, tmp_path):
        """Config with basic + allQ scaffolds returns correct agent classes."""
        from scripts.test_basic_agent import create_agents_from_config
        from agents.basicAgent.agent import BasicAgent
        from agents.allQAgent.agent import AllQAgent

        config = {
            "defaults": {"provider": "openrouter"},
            "agents": [
                {"model": "vendor/model-a", "scaffold": "basic"},
                {"model": "vendor/model-b", "scaffold": "allq"},
            ],
        }
        agents = create_agents_from_config(config, _make_args(), str(tmp_path), search_tool=None)
        assert isinstance(agents[0], BasicAgent)
        assert isinstance(agents[1], AllQAgent)

    def test_no_model_raises(self, tmp_path):
        """Agent entry without 'model' key raises ValueError."""
        from scripts.test_basic_agent import create_agents_from_config

        config = {
            "defaults": {"provider": "openrouter", "scaffold": "basic"},
            "agents": [{"scaffold": "basic"}],
        }
        with pytest.raises(ValueError, match="model"):
            create_agents_from_config(config, _make_args(), str(tmp_path), search_tool=None)

    def test_empty_list_raises(self, tmp_path):
        """Empty agents list raises ValueError."""
        from scripts.test_basic_agent import create_agents_from_config

        config = {"defaults": {}, "agents": []}
        with pytest.raises(ValueError, match="No agents"):
            create_agents_from_config(config, _make_args(), str(tmp_path), search_tool=None)

    @patch("inference.openrouter.OpenRouterInference.__init__", return_value=None)
    def test_unknown_scaffold_raises(self, mock_init, tmp_path):
        """Unknown scaffold name raises ValueError."""
        from scripts.test_basic_agent import create_agents_from_config

        config = {
            "defaults": {"provider": "openrouter"},
            "agents": [{"model": "vendor/model-a", "scaffold": "nonexistent"}],
        }
        with pytest.raises(ValueError, match="Unknown scaffold"):
            create_agents_from_config(config, _make_args(), str(tmp_path), search_tool=None)

    @patch("inference.openrouter.OpenRouterInference.__init__", return_value=None)
    def test_per_agent_directories(self, mock_init, tmp_path):
        """Each agent gets its own directory under output_dir/agents/."""
        from scripts.test_basic_agent import create_agents_from_config

        config = {
            "defaults": {"provider": "openrouter", "scaffold": "basic"},
            "agents": [
                {"model": "vendor/model-a"},
                {"model": "vendor/model-b"},
            ],
        }
        create_agents_from_config(config, _make_args(), str(tmp_path), search_tool=None)
        assert (tmp_path / "agents" / "basic_model-a_001").is_dir()
        assert (tmp_path / "agents" / "basic_model-b_001").is_dir()

    @patch("inference.openrouter.OpenRouterInference.__init__", return_value=None)
    def test_single_agent_mode_flag(self, mock_init, tmp_path):
        """Single-agent config sets single_agent_mode=True; multi sets False."""
        from scripts.test_basic_agent import create_agents_from_config

        single_config = {
            "defaults": {"provider": "openrouter", "scaffold": "basic"},
            "agents": [{"model": "vendor/model-a"}],
        }
        agents = create_agents_from_config(single_config, _make_args(), str(tmp_path), search_tool=None)
        assert agents[0].config.single_agent_mode is True

        multi_config = {
            "defaults": {"provider": "openrouter", "scaffold": "basic"},
            "agents": [{"model": "vendor/model-a"}, {"model": "vendor/model-b"}],
        }
        agents = create_agents_from_config(multi_config, _make_args(), str(tmp_path / "multi"), search_tool=None)
        assert agents[0].config.single_agent_mode is False
        assert agents[1].config.single_agent_mode is False

    @patch("inference.openrouter.OpenRouterInference.__init__", return_value=None)
    def test_sampling_params(self, mock_init, tmp_path):
        """Per-agent temperature/max_tokens/top_p flow into AgentConfig.sampling_params."""
        from scripts.test_basic_agent import create_agents_from_config

        config = {
            "defaults": {"provider": "openrouter", "scaffold": "basic"},
            "agents": [
                {"model": "vendor/model-a", "temperature": 0.3, "max_tokens": 512, "top_p": 0.95},
            ],
        }
        agents = create_agents_from_config(config, _make_args(), str(tmp_path), search_tool=None)
        sp = agents[0].config.sampling_params
        assert sp["temperature"] == 0.3
        assert sp["max_tokens"] == 512
        assert sp["top_p"] == 0.95


# ===========================================================================
# B. SimulationEnvironment multi-agent orchestration
# ===========================================================================

class TestEnvMultiAgent:

    def _make_env_with_agents(self, tmp_path, agents, parallel=True):
        """Create a minimal SimulationEnvironment with stub agents, bypassing dataset loading."""
        from environment.env import SimulationEnvironment
        env = SimulationEnvironment.__new__(SimulationEnvironment)
        # Manually initialize the minimum state needed for testing
        env.output_dir = str(tmp_path)
        env.agents = []
        env.agent_scores = {}
        env.agent_correct = {}
        env.agent_wrong = {}
        env.agent_questions = {}
        env.agent_raw_brier = {}
        env.agent_snapshot_peer = {}
        env.agent_exp_acc_sum = {}
        env.prediction_histories = {}
        env._histories_lock = threading.Lock()
        env.current_aggregates = {}
        env.current_date = date(2025, 5, 1)
        env.end_date = date(2025, 5, 8)
        env.start_date = date(2025, 5, 1)
        env.timegap_days = 1
        env.parallel = parallel
        env.resolved_questions = []
        env.resolution_events = []
        env.resolved_agent_predictions = {}
        env.market_csv_path = str(tmp_path / "market.csv")
        env.logger = MagicMock(spec=SimLogger)
        env.market_writer = MagicMock()
        env.market_writer.write.return_value = str(tmp_path / "market.csv")
        env.matcher = None
        env.scorer = BinaryBrierScorer()
        env.resume_dir = None
        env.q_pool = MagicMock()
        env.source_name = "openforesight"
        env.source_context = ""
        for a in agents:
            env.add_agent(a)
        return env

    def test_add_multiple_agents(self, tmp_path):
        """add_agent() called N times populates agents list and score dicts."""
        a1 = StubAgent("agent_1")
        a2 = StubAgent("agent_2")
        a3 = StubAgent("agent_3")
        env = self._make_env_with_agents(tmp_path, [a1, a2, a3])

        assert len(env.agents) == 3
        for aid in ("agent_1", "agent_2", "agent_3"):
            assert aid in env.agent_scores
            assert aid in env.agent_correct
            assert aid in env.agent_wrong
            assert aid in env.agent_questions
            assert aid in env.agent_raw_brier
            assert aid in env.agent_snapshot_peer

    def test_parallel_dispatch(self, tmp_path):
        """With parallel=True and 2+ agents, _run_agents_parallel is called."""
        a1 = StubAgent("agent_1")
        a2 = StubAgent("agent_2")
        env = self._make_env_with_agents(tmp_path, [a1, a2], parallel=True)

        with patch.object(env, "_run_agents_parallel") as mock_par, \
             patch.object(env, "_run_agents_sequential") as mock_seq:
            questions = [_make_question()]
            env.q_pool.get_active.return_value = questions
            env.q_pool.pop_resolving.return_value = []
            env.step()
            mock_par.assert_called_once()
            mock_seq.assert_not_called()

    def test_sequential_dispatch_single_agent(self, tmp_path):
        """With 1 agent, _run_agents_sequential is used regardless of parallel flag."""
        a1 = StubAgent("agent_1")
        env = self._make_env_with_agents(tmp_path, [a1], parallel=True)

        with patch.object(env, "_run_agents_parallel") as mock_par, \
             patch.object(env, "_run_agents_sequential") as mock_seq:
            questions = [_make_question()]
            env.q_pool.get_active.return_value = questions
            env.q_pool.pop_resolving.return_value = []
            env.step()
            mock_seq.assert_called_once()
            mock_par.assert_not_called()

    def test_sequential_dispatch_no_parallel(self, tmp_path):
        """With parallel=False and 2 agents, _run_agents_sequential is used."""
        a1 = StubAgent("agent_1")
        a2 = StubAgent("agent_2")
        env = self._make_env_with_agents(tmp_path, [a1, a2], parallel=False)

        with patch.object(env, "_run_agents_parallel") as mock_par, \
             patch.object(env, "_run_agents_sequential") as mock_seq:
            questions = [_make_question()]
            env.q_pool.get_active.return_value = questions
            env.q_pool.pop_resolving.return_value = []
            env.step()
            mock_seq.assert_called_once()
            mock_par.assert_not_called()

    def test_parallel_all_agents_called(self, tmp_path):
        """Both agents' act() is called exactly once during parallel execution."""
        q = _make_question()
        a1 = StubAgent("agent_1", predictions={q.qid: {"Yes": 0.7, "No": 0.3}})
        a2 = StubAgent("agent_2", predictions={q.qid: {"Yes": 0.4, "No": 0.6}})
        env = self._make_env_with_agents(tmp_path, [a1, a2], parallel=True)

        # Set up prediction histories for the question
        env.prediction_histories[q.qid] = PredictionHistory(
            question_id=q.qid, start_date=date(2025, 5, 1), resolution_date=q.resolution_date
        )
        env.q_pool.get_active.return_value = [q]
        env.q_pool.pop_resolving.return_value = []
        env.step()

        assert len(a1.act_calls) == 1
        assert len(a2.act_calls) == 1
        assert a1.act_calls[0] == date(2025, 5, 1)

    def test_parallel_agent_error_handling(self, tmp_path, capsys):
        """One agent raising an exception doesn't prevent the other from completing."""
        q = _make_question()
        a1 = StubAgent("agent_ok", predictions={q.qid: {"Yes": 0.6, "No": 0.4}})
        a2 = StubAgent("agent_err", should_raise=True)
        env = self._make_env_with_agents(tmp_path, [a1, a2], parallel=True)

        env.prediction_histories[q.qid] = PredictionHistory(
            question_id=q.qid, start_date=date(2025, 5, 1), resolution_date=q.resolution_date
        )
        env.q_pool.get_active.return_value = [q]
        env.q_pool.pop_resolving.return_value = []
        env.step()

        # The working agent should have completed
        assert len(a1.act_calls) == 1
        captured = capsys.readouterr()
        assert "ERROR" in captured.out
        assert "agent_err" in captured.out


# ===========================================================================
# C. SimForecastInterface thread safety & prediction submission
# ===========================================================================

class TestSimForecastInterface:

    def _make_interface(self, questions=None, histories=None):
        q = questions or [_make_question()]
        h = histories or {
            q[0].qid: PredictionHistory(
                question_id=q[0].qid,
                start_date=date(2025, 5, 1),
                resolution_date=q[0].resolution_date,
            )
        }
        logger = MagicMock(spec=SimLogger)
        iface = SimForecastInterface(
            questions=q,
            aggregates={},
            histories=h,
            sim_date=date(2025, 5, 1),
            logger=logger,
            timegap_days=1,
        )
        return iface, h, logger

    def test_agent_context(self):
        """set_agent_context() sets current_agent_id."""
        iface, _, _ = self._make_interface()
        assert iface.current_agent_id is None
        iface.set_agent_context("agent_1")
        assert iface.current_agent_id == "agent_1"

    def test_submit_prediction(self):
        """Prediction is recorded in shared PredictionHistory with correct agent_id."""
        q = _make_question()
        iface, histories, logger = self._make_interface(questions=[q])
        iface.set_agent_context("agent_1")
        iface.submit_prediction(PredictionSubmission(
            question_id=q.qid, outcomes={"Yes": 0.7, "No": 0.3}
        ))

        history = histories[q.qid]
        pred = history.get_latest_prediction("agent_1")
        assert pred is not None
        assert pred.agent_id == "agent_1"
        assert pred.outcomes == {"Yes": 0.7, "No": 0.3}
        logger.log_prediction.assert_called_once()

    def test_concurrent_submissions(self):
        """Two threads submitting predictions simultaneously both get recorded."""
        q = _make_question()
        iface, histories, _ = self._make_interface(questions=[q])
        errors = []

        def submit_as(agent_id, outcomes):
            try:
                # Create a separate interface per thread (like parallel execution does)
                thread_iface = SimForecastInterface(
                    questions=[q],
                    aggregates={},
                    histories=histories,
                    sim_date=date(2025, 5, 1),
                    logger=MagicMock(spec=SimLogger),
                    histories_lock=iface._histories_lock,
                    timegap_days=1,
                )
                thread_iface.set_agent_context(agent_id)
                thread_iface.submit_prediction(PredictionSubmission(
                    question_id=q.qid, outcomes=outcomes
                ))
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=submit_as, args=("a1", {"Yes": 0.8, "No": 0.2}))
        t2 = threading.Thread(target=submit_as, args=("a2", {"Yes": 0.3, "No": 0.7}))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors
        history = histories[q.qid]
        assert history.get_latest_prediction("a1") is not None
        assert history.get_latest_prediction("a2") is not None
        assert history.get_latest_prediction("a1").outcomes["Yes"] == 0.8
        assert history.get_latest_prediction("a2").outcomes["Yes"] == 0.3

    def test_no_agent_context_raises(self):
        """submit_prediction() without set_agent_context() raises ValueError."""
        q = _make_question()
        iface, _, _ = self._make_interface(questions=[q])
        with pytest.raises(ValueError, match="No agent context"):
            iface.submit_prediction(PredictionSubmission(
                question_id=q.qid, outcomes={"Yes": 0.5, "No": 0.5}
            ))

    def test_submit_invalid_question_raises(self):
        """submit_prediction() for a non-active question raises ValueError."""
        q = _make_question()
        iface, _, _ = self._make_interface(questions=[q])
        iface.set_agent_context("agent_1")
        with pytest.raises(ValueError, match="not active"):
            iface.submit_prediction(PredictionSubmission(
                question_id="nonexistent_q", outcomes={"Yes": 0.5}
            ))

    def test_submit_probabilities_over_one_raises(self):
        """Probabilities summing to >1 raise ValueError."""
        q = _make_question()
        iface, _, _ = self._make_interface(questions=[q])
        iface.set_agent_context("agent_1")
        with pytest.raises(ValueError, match="sum to"):
            iface.submit_prediction(PredictionSubmission(
                question_id=q.qid, outcomes={"Yes": 0.8, "No": 0.5}
            ))

    def test_num_agents_attribute(self):
        """num_agents is stored on the interface."""
        q = _make_question()
        iface = SimForecastInterface(
            questions=[q],
            aggregates={},
            histories={},
            sim_date=date(2025, 5, 1),
            logger=MagicMock(spec=SimLogger),
            num_agents=4,
        )
        assert iface.num_agents == 4

    def test_num_agents_defaults_to_one(self):
        """num_agents defaults to 1 when not provided."""
        q = _make_question()
        iface = SimForecastInterface(
            questions=[q],
            aggregates={},
            histories={},
            sim_date=date(2025, 5, 1),
            logger=MagicMock(spec=SimLogger),
        )
        assert iface.num_agents == 1


# ===========================================================================
# D. Aggregate computation and peer scoring
# ===========================================================================

class TestAggregateAndScoring:

    def test_aggregates_computed_after_all_agents(self, tmp_path):
        """Aggregates updated only after all agents finish, not mid-step."""
        q = _make_question()
        a1 = StubAgent("a1", predictions={q.qid: {"Yes": 0.8, "No": 0.2}})
        a2 = StubAgent("a2", predictions={q.qid: {"Yes": 0.4, "No": 0.6}})

        env_cls = type('', (), {})  # We test via the actual env
        from tests.test_multiagent import TestEnvMultiAgent
        helper = TestEnvMultiAgent()
        env = helper._make_env_with_agents(tmp_path, [a1, a2], parallel=True)

        env.prediction_histories[q.qid] = PredictionHistory(
            question_id=q.qid, start_date=date(2025, 5, 1), resolution_date=q.resolution_date
        )
        env.q_pool.get_active.return_value = [q]
        env.q_pool.pop_resolving.return_value = []

        # Before step, no aggregates
        assert q.qid not in env.current_aggregates
        env.step()

        # After step, aggregate should be the mean of both agents' predictions
        agg = env.current_aggregates.get(q.qid)
        assert agg is not None
        assert abs(agg["Yes"] - 0.6) < 1e-6  # (0.8 + 0.4) / 2
        assert abs(agg["No"] - 0.4) < 1e-6   # (0.2 + 0.6) / 2

    def test_compute_aggregate_averages(self):
        """compute_aggregate returns mean probabilities across agents."""
        preds = [
            DailyPrediction("a1", "q1", date(2025, 5, 1), {"Yes": 1.0, "No": 0.0}),
            DailyPrediction("a2", "q1", date(2025, 5, 1), {"Yes": 0.0, "No": 1.0}),
        ]
        agg = compute_aggregate(preds)
        assert abs(agg["Yes"] - 0.5) < 1e-6
        assert abs(agg["No"] - 0.5) < 1e-6

    def test_peer_scoring_multi_agent(self):
        """Better agent gets positive peer score, worse gets negative."""
        # Agent 1 predicts correctly (high prob for ground truth "Yes")
        pred_good = DailyPrediction("good", "q1", date(2025, 5, 1), {"Yes": 0.9, "No": 0.1})
        # Agent 2 predicts poorly
        pred_bad = DailyPrediction("bad", "q1", date(2025, 5, 1), {"Yes": 0.1, "No": 0.9})

        scorer = BinaryBrierScorer()
        peer_scores = compute_snapshot_peer_scores(
            {"good": pred_good, "bad": pred_bad},
            ground_truth="Yes",
            scorer=scorer,
        )

        assert peer_scores["good"] > 0, "Better agent should have positive peer score"
        assert peer_scores["bad"] < 0, "Worse agent should have negative peer score"
        # Peer scores should sum to zero (zero-sum game)
        assert abs(peer_scores["good"] + peer_scores["bad"]) < 1e-6

    def test_peer_scoring_identical_agents(self):
        """Identical predictions yield zero peer scores for both agents."""
        pred_a = DailyPrediction("a", "q1", date(2025, 5, 1), {"Yes": 0.6, "No": 0.4})
        pred_b = DailyPrediction("b", "q1", date(2025, 5, 1), {"Yes": 0.6, "No": 0.4})

        scorer = BinaryBrierScorer()
        peer_scores = compute_snapshot_peer_scores(
            {"a": pred_a, "b": pred_b},
            ground_truth="Yes",
            scorer=scorer,
        )
        assert abs(peer_scores["a"]) < 1e-6
        assert abs(peer_scores["b"]) < 1e-6


# ===========================================================================
# E. get_model_short_name utility
# ===========================================================================

class TestModelShortName:

    def test_openrouter_id(self):
        from scripts.test_basic_agent import get_model_short_name
        assert get_model_short_name("deepseek/deepseek-v3.2") == "deepseek-v3.2"

    def test_openrouter_id_with_variant(self):
        from scripts.test_basic_agent import get_model_short_name
        assert get_model_short_name("xiaomi/mimo-v2-flash:free") == "mimo-v2-flash"

    def test_local_path(self):
        from scripts.test_basic_agent import get_model_short_name
        assert get_model_short_name("/models/Qwen3.5-27B") == "Qwen3.5-27B"
