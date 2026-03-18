from collections import Counter
from datetime import date

import pandas as pd

from agents.basicAgent import AgentConfig, BasicAgent
from environment.data_loader import Question, QuestionPool
from environment.env import SimulationEnvironment
from environment.scoring import DEFAULT_SCORER, DailyPrediction, PredictionHistory


def _write_openforesight_split(tmp_path, split: str, rows):
    df = pd.DataFrame(rows)
    out_path = tmp_path / f"{split}-00000-of-00001.parquet"
    df.to_parquet(out_path, index=False)


def test_openforesight_prepend_train_window_supports_month_caps(tmp_path):
    train_rows = [
        {
            "qid": "train-jan-a",
            "question_title": "Jan A",
            "background": "",
            "resolution_criteria": "",
            "answer_type": "yes/no",
            "resolution_date": "2025-01-03",
            "answer": "Yes",
            "prompt": "",
        },
        {
            "qid": "train-jan-b",
            "question_title": "Jan B",
            "background": "",
            "resolution_criteria": "",
            "answer_type": "yes/no",
            "resolution_date": "2025-01-10",
            "answer": "Yes",
            "prompt": "",
        },
        {
            "qid": "train-jan-c",
            "question_title": "Jan C",
            "background": "",
            "resolution_criteria": "",
            "answer_type": "yes/no",
            "resolution_date": "2025-01-14",
            "answer": "Yes",
            "prompt": "",
        },
        {
            "qid": "train-feb-a",
            "question_title": "Feb A",
            "background": "",
            "resolution_criteria": "",
            "answer_type": "yes/no",
            "resolution_date": "2025-02-01",
            "answer": "Yes",
            "prompt": "",
        },
        {
            "qid": "train-feb-b",
            "question_title": "Feb B",
            "background": "",
            "resolution_criteria": "",
            "answer_type": "yes/no",
            "resolution_date": "2025-02-04",
            "answer": "Yes",
            "prompt": "",
        },
        {
            "qid": "train-feb-c",
            "question_title": "Feb C",
            "background": "",
            "resolution_criteria": "",
            "answer_type": "yes/no",
            "resolution_date": "2025-02-08",
            "answer": "Yes",
            "prompt": "",
        },
        {
            "qid": "train-apr-a",
            "question_title": "Apr A",
            "background": "",
            "resolution_criteria": "",
            "answer_type": "yes/no",
            "resolution_date": "2025-04-09",
            "answer": "Yes",
            "prompt": "",
        },
    ]
    test_rows = [
        {
            "qid": "test-may-a",
            "question_title": "May A",
            "background": "",
            "resolution_criteria": "",
            "answer_type": "yes/no",
            "resolution_date": "2025-05-02",
            "answer": "Yes",
            "prompt": "",
        },
        {
            "qid": "test-may-b",
            "question_title": "May B",
            "background": "",
            "resolution_criteria": "",
            "answer_type": "yes/no",
            "resolution_date": "2025-05-06",
            "answer": "Yes",
            "prompt": "",
        },
        {
            "qid": "test-may-c",
            "question_title": "May C",
            "background": "",
            "resolution_criteria": "",
            "answer_type": "yes/no",
            "resolution_date": "2025-05-09",
            "answer": "Yes",
            "prompt": "",
        },
    ]
    _write_openforesight_split(tmp_path, "train", train_rows)
    _write_openforesight_split(tmp_path, "test", test_rows)

    pool = QuestionPool(
        dataset="openforesight",
        dataset_path=str(tmp_path),
        split="test",
        prepend_train_resolution_start=date(2025, 1, 1),
        prepend_train_resolution_end=date(2025, 4, 30),
        subsample_per_month=2,
        resolution_start=date(2025, 1, 1),
        resolution_end=date(2025, 5, 31),
    )

    month_counts = Counter(q.resolution_date.strftime("%Y-%m") for q in pool.get_active())
    assert month_counts == {
        "2025-01": 2,
        "2025-02": 2,
        "2025-04": 1,
        "2025-05": 3,
    }


def test_active_scores_use_interval_end_for_timegap_runs():
    class DummyAgent:
        def __init__(self, agent_id: str):
            self.agent_id = agent_id

    env = SimulationEnvironment.__new__(SimulationEnvironment)
    env.current_date = date(2025, 1, 5)
    env.timegap_days = 5
    env.matcher = None
    env.scorer = DEFAULT_SCORER
    env.agents = [DummyAgent("agent_a")]

    question = Question(
        qid="q1",
        title="Will outcome happen?",
        background="",
        resolution_criteria="",
        answer_type="yes/no",
        resolution_date=date(2025, 1, 20),
        ground_truth_answer="Yes",
    )
    history = PredictionHistory(
        question_id=question.qid,
        start_date=date(2025, 1, 1),
        resolution_date=question.resolution_date,
    )
    history.add_prediction(
        DailyPrediction(
            agent_id="agent_a",
            question_id=question.qid,
            day=date(2025, 1, 1),
            outcomes={"Yes": 0.0},
        )
    )
    history.add_prediction(
        DailyPrediction(
            agent_id="agent_a",
            question_id=question.qid,
            day=date(2025, 1, 5),
            outcomes={"Yes": 1.0},
        )
    )
    env.prediction_histories = {question.qid: history}

    day_stats = env._compute_daily_active_scores([question], evaluation_date=date(2025, 1, 5))
    interval_stats = env._compute_daily_active_scores([question], evaluation_date=date(2025, 1, 9))

    assert interval_stats["agent_a"]["tw_peer_sum"] > day_stats["agent_a"]["tw_peer_sum"]


def test_basic_agent_cadence_prompt_includes_exact_update_dates():
    agent = BasicAgent(
        agent_id="a1",
        inference_provider=object(),
        config=AgentConfig(enable_memory=False, timegap_days=5),
    )
    agent._forecast_interface = type(
        "ForecastStub",
        (),
        {
            "last_active_date": date(2025, 1, 1),
            "next_active_date": date(2025, 1, 11),
        },
    )()

    cadence = agent._build_cadence_section(date(2025, 1, 6))

    assert "every 5 days" in cadence
    assert "2025-01-01" in cadence
    assert "2025-01-11" in cadence


def test_metrics_builder_can_filter_to_test_split():
    class DummyAgent:
        def __init__(self, agent_id: str):
            self.agent_id = agent_id

    env = SimulationEnvironment.__new__(SimulationEnvironment)
    env.current_date = date(2025, 1, 6)
    env.end_date = date(2025, 1, 31)
    env.timegap_days = 5
    env.start_date = date(2025, 1, 1)
    env.matcher = None
    env.scorer = DEFAULT_SCORER
    env.agents = [DummyAgent("agent_a")]
    env.prediction_histories = {}
    env.resolution_events = [
        {
            "qid": "train_q",
            "source_split": "train",
            "agents": {
                "agent_a": {
                    "brier": 0.9,
                    "snapshot_peer": 0.4,
                    "tw_peer": 0.2,
                    "truth_prob": 0.1,
                    "is_accurate": False,
                }
            },
        },
        {
            "qid": "test_q",
            "source_split": "test",
            "agents": {
                "agent_a": {
                    "brier": 0.2,
                    "snapshot_peer": 0.7,
                    "tw_peer": 0.6,
                    "truth_prob": 0.8,
                    "is_accurate": True,
                }
            },
        },
    ]

    metrics = env._build_metrics_list(active_questions=[], source_split="test")

    assert metrics == [{
        "agent_id": "agent_a",
        "avg_brier": 0.2,
        "peer_score": 0.7,
        "tw_peer_score": 0.6,
        "accuracy": 100.0,
        "exp_acc": 0.8,
        "total_predictions": 1,
    }]
