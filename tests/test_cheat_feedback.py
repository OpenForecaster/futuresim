"""
Tests for the cheat-feedback feature.

Verifies:
1. _compute_agent_cheat_feedback core logic (direction, scores, filtering)
2. Detail modes ("full" vs "direction")
3. Summary statistics
4. FeedbackHandler.format_cheat_feedback formatting
5. SimForecastInterface.get_cheat_feedback integration
6. No-op when disabled
"""

import pytest
from datetime import date
from threading import Lock

from environment.env import _compute_agent_cheat_feedback, SimForecastInterface
from environment.data_loader import Question
from environment.scoring import BrierScorer, DailyPrediction, PredictionHistory
from agents.basicAgent.feedback import FeedbackHandler
from agents.basicAgent.config import AgentConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_question(qid, title, ground_truth, res_date=None):
    return Question(
        qid=qid,
        title=title,
        background="",
        resolution_criteria="",
        answer_type="multiple_choice",
        resolution_date=res_date or date(2024, 3, 1),
        ground_truth_answer=ground_truth,
    )


def _make_history(qid, agent_id, predictions, res_date=None):
    """Create a PredictionHistory with the given list of (day, outcomes) tuples."""
    h = PredictionHistory(
        question_id=qid,
        start_date=predictions[0][0] if predictions else date(2024, 1, 1),
        resolution_date=res_date or date(2024, 3, 1),
    )
    for day, outcomes in predictions:
        pred = DailyPrediction(agent_id, qid, day, outcomes)
        h.add_prediction(pred)
    return h


# ---------------------------------------------------------------------------
# Test 1: Core logic — direction and scores
# ---------------------------------------------------------------------------

class TestComputeCheatFeedback:
    def test_improved_prediction(self):
        """Agent moves probability toward truth → direction = improved."""
        scorer = BrierScorer()
        q = _make_question("q1", "Will X happen?", "Yes")
        # Day 1: 30% on Yes, Day 2 (today): 80% on Yes
        history = _make_history("q1", "a1", [
            (date(2024, 1, 1), {"Yes": 0.3, "No": 0.7}),
            (date(2024, 1, 2), {"Yes": 0.8, "No": 0.2}),
        ])
        result = _compute_agent_cheat_feedback(
            "a1", [q], {"q1": history}, scorer, None, date(2024, 1, 2), "full"
        )
        assert len(result["items"]) == 1
        item = result["items"][0]
        assert item["direction"] == "improved"
        assert item["current_brier"] > item["previous_brier"]

    def test_worsened_prediction(self):
        """Agent moves probability away from truth → direction = worsened."""
        scorer = BrierScorer()
        q = _make_question("q1", "Will X happen?", "Yes")
        # Day 1: 80% on Yes, Day 2 (today): 30% on Yes
        history = _make_history("q1", "a1", [
            (date(2024, 1, 1), {"Yes": 0.8, "No": 0.2}),
            (date(2024, 1, 2), {"Yes": 0.3, "No": 0.7}),
        ])
        result = _compute_agent_cheat_feedback(
            "a1", [q], {"q1": history}, scorer, None, date(2024, 1, 2), "full"
        )
        assert result["items"][0]["direction"] == "worsened"

    def test_unchanged_prediction(self):
        """Same prediction twice → direction = unchanged."""
        scorer = BrierScorer()
        q = _make_question("q1", "Will X happen?", "Yes")
        history = _make_history("q1", "a1", [
            (date(2024, 1, 1), {"Yes": 0.5, "No": 0.5}),
            (date(2024, 1, 2), {"Yes": 0.5, "No": 0.5}),
        ])
        result = _compute_agent_cheat_feedback(
            "a1", [q], {"q1": history}, scorer, None, date(2024, 1, 2), "full"
        )
        assert result["items"][0]["direction"] == "unchanged"

    def test_first_prediction_above_baseline(self):
        """First prediction with positive Brier → direction = improved vs baseline 0."""
        scorer = BrierScorer()
        q = _make_question("q1", "Will X happen?", "Yes")
        # Single prediction today: 70% on truth → Brier > 0
        history = _make_history("q1", "a1", [
            (date(2024, 1, 1), {"Yes": 0.7, "No": 0.3}),
        ])
        result = _compute_agent_cheat_feedback(
            "a1", [q], {"q1": history}, scorer, None, date(2024, 1, 1), "full"
        )
        item = result["items"][0]
        assert item["direction"] == "improved"
        assert item["previous_brier"] == 0.0  # baseline

    def test_first_prediction_below_baseline(self):
        """First prediction with negative Brier → direction = worsened."""
        scorer = BrierScorer()
        q = _make_question("q1", "Will X happen?", "Yes")
        # Single prediction: 100% on wrong outcome → Brier < 0
        history = _make_history("q1", "a1", [
            (date(2024, 1, 1), {"No": 1.0}),
        ])
        result = _compute_agent_cheat_feedback(
            "a1", [q], {"q1": history}, scorer, None, date(2024, 1, 1), "full"
        )
        assert result["items"][0]["direction"] == "worsened"

    def test_excludes_predictions_not_today(self):
        """Predictions not made on sim_date are excluded."""
        scorer = BrierScorer()
        q = _make_question("q1", "Will X happen?", "Yes")
        history = _make_history("q1", "a1", [
            (date(2024, 1, 1), {"Yes": 0.7, "No": 0.3}),
        ])
        # sim_date is day 2, but only prediction is day 1
        result = _compute_agent_cheat_feedback(
            "a1", [q], {"q1": history}, scorer, None, date(2024, 1, 2), "full"
        )
        assert result == {}

    def test_excludes_questions_without_ground_truth(self):
        """Questions without ground truth are skipped."""
        scorer = BrierScorer()
        q = _make_question("q1", "Unknown?", "")
        history = _make_history("q1", "a1", [
            (date(2024, 1, 1), {"Yes": 0.5}),
        ])
        result = _compute_agent_cheat_feedback(
            "a1", [q], {"q1": history}, scorer, None, date(2024, 1, 1), "full"
        )
        assert result == {}

    def test_excludes_other_agent(self):
        """Agent b1's predictions don't appear in a1's feedback."""
        scorer = BrierScorer()
        q = _make_question("q1", "Will X happen?", "Yes")
        history = _make_history("q1", "b1", [
            (date(2024, 1, 1), {"Yes": 0.7}),
        ])
        result = _compute_agent_cheat_feedback(
            "a1", [q], {"q1": history}, scorer, None, date(2024, 1, 1), "full"
        )
        assert result == {}


# ---------------------------------------------------------------------------
# Test 2: Detail mode = "direction"
# ---------------------------------------------------------------------------

class TestDetailDirection:
    def test_direction_mode_no_scores(self):
        """In 'direction' mode, items should NOT contain brier scores."""
        scorer = BrierScorer()
        q = _make_question("q1", "Will X happen?", "Yes")
        history = _make_history("q1", "a1", [
            (date(2024, 1, 1), {"Yes": 0.3, "No": 0.7}),
            (date(2024, 1, 2), {"Yes": 0.8, "No": 0.2}),
        ])
        result = _compute_agent_cheat_feedback(
            "a1", [q], {"q1": history}, scorer, None, date(2024, 1, 2), "direction"
        )
        item = result["items"][0]
        assert item["direction"] == "improved"
        assert "current_brier" not in item
        assert "previous_brier" not in item


# ---------------------------------------------------------------------------
# Test 3: Summary statistics
# ---------------------------------------------------------------------------

class TestSummaryStats:
    def test_mixed_summary(self):
        """Multiple questions with mixed directions produce correct summary."""
        scorer = BrierScorer()
        today = date(2024, 1, 2)
        q1 = _make_question("q1", "Q1?", "Yes")
        q2 = _make_question("q2", "Q2?", "A")
        q3 = _make_question("q3", "Q3?", "B")

        h1 = _make_history("q1", "a1", [
            (date(2024, 1, 1), {"Yes": 0.3}),
            (today, {"Yes": 0.9}),  # improved
        ])
        h2 = _make_history("q2", "a1", [
            (date(2024, 1, 1), {"A": 0.9}),
            (today, {"A": 0.2}),  # worsened
        ])
        h3 = _make_history("q3", "a1", [
            (date(2024, 1, 1), {"B": 0.5}),
            (today, {"B": 0.5}),  # unchanged
        ])

        result = _compute_agent_cheat_feedback(
            "a1", [q1, q2, q3],
            {"q1": h1, "q2": h2, "q3": h3},
            scorer, None, today, "full"
        )
        s = result["summary"]
        assert s["total"] == 3
        assert s["improved"] == 1
        assert s["worsened"] == 1
        assert s["unchanged"] == 1
        assert "avg_brier" in s

    def test_direction_mode_no_avg_brier(self):
        """In 'direction' mode, avg_brier should be absent from summary."""
        scorer = BrierScorer()
        today = date(2024, 1, 1)
        q = _make_question("q1", "Q?", "Yes")
        h = _make_history("q1", "a1", [(today, {"Yes": 0.7})])
        result = _compute_agent_cheat_feedback(
            "a1", [q], {"q1": h}, scorer, None, today, "direction"
        )
        assert "avg_brier" not in result["summary"]


# ---------------------------------------------------------------------------
# Test 4: format_cheat_feedback
# ---------------------------------------------------------------------------

class TestFormatCheatFeedback:
    def test_full_mode_contains_scores(self):
        cheat_data = {
            "items": [
                {"qid": "q1", "title": "Will X?", "direction": "improved",
                 "current_brier": 0.75, "previous_brier": 0.5},
            ],
            "summary": {"total": 1, "improved": 1, "worsened": 0, "unchanged": 0, "avg_brier": 0.75},
        }
        text = FeedbackHandler.format_cheat_feedback(cheat_data, "full")
        assert "PREDICTION PERFORMANCE FEEDBACK" in text
        assert "+0.750" in text
        assert "+0.500" in text
        assert "IMPROVED" in text

    def test_direction_mode_no_scores_in_text(self):
        cheat_data = {
            "items": [
                {"qid": "q1", "title": "Will X?", "direction": "worsened"},
            ],
            "summary": {"total": 1, "improved": 0, "worsened": 1, "unchanged": 0},
        }
        text = FeedbackHandler.format_cheat_feedback(cheat_data, "direction")
        assert "PREDICTION PERFORMANCE FEEDBACK" in text
        assert "WORSENED" in text
        # Should not contain score numbers
        assert "Brier Skill:" not in text

    def test_empty_items_returns_empty_string(self):
        assert FeedbackHandler.format_cheat_feedback({"items": [], "summary": {}}) == ""
        assert FeedbackHandler.format_cheat_feedback({}) == ""


# ---------------------------------------------------------------------------
# Test 5: SimForecastInterface.get_cheat_feedback integration
# ---------------------------------------------------------------------------

class TestInterfaceGetCheatFeedback:
    def test_returns_empty_when_disabled(self):
        """Interface without cheat context returns empty dict."""
        iface = SimForecastInterface(
            questions=[], aggregates={}, histories={},
            sim_date=date(2024, 1, 1), logger=None,
            cheat_feedback_ctx=None,
        )
        iface.current_agent_id = "a1"
        assert iface.get_cheat_feedback() == {}

    def test_returns_empty_when_no_agent(self):
        """Interface with cheat context but no agent set returns empty."""
        scorer = BrierScorer()
        q = _make_question("q1", "Q?", "Yes")
        iface = SimForecastInterface(
            questions=[], aggregates={}, histories={},
            sim_date=date(2024, 1, 1), logger=None,
            cheat_feedback_ctx=([q], scorer, None),
        )
        # current_agent_id is None
        assert iface.get_cheat_feedback() == {}

    def test_returns_feedback_when_enabled(self):
        """Interface with cheat context and agent computes feedback."""
        scorer = BrierScorer()
        today = date(2024, 1, 1)
        q = _make_question("q1", "Will X?", "Yes")
        h = _make_history("q1", "a1", [(today, {"Yes": 0.8, "No": 0.2})])

        iface = SimForecastInterface(
            questions=[], aggregates={}, histories={"q1": h},
            sim_date=today, logger=None,
            cheat_feedback_ctx=([q], scorer, None),
        )
        iface.current_agent_id = "a1"
        result = iface.get_cheat_feedback("full")
        assert "items" in result
        assert len(result["items"]) == 1
        assert result["items"][0]["direction"] == "improved"


# ---------------------------------------------------------------------------
# Test 6: Config defaults
# ---------------------------------------------------------------------------

class TestConfigDefaults:
    def test_cheat_feedback_off_by_default(self):
        config = AgentConfig()
        assert config.cheat_feedback is False
        assert config.cheat_feedback_detail == "full"

    def test_cheat_feedback_configurable(self):
        config = AgentConfig(cheat_feedback=True, cheat_feedback_detail="direction")
        assert config.cheat_feedback is True
        assert config.cheat_feedback_detail == "direction"
