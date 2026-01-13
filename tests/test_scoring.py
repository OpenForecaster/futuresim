"""
Tests for modular scoring system.
Verifies:
1. Zero-sum property of peer scores (both Brier and Log)
2. Brier Skill Score: higher is better, 0 = baseline
3. Penalty for missing truth
4. Carry-forward semantics
"""

import pytest
from datetime import date
from environment.scoring import (
    BrierScorer, LogScorer,
    DailyPrediction, PredictionHistory,
    compute_snapshot_peer_scores, resolve_question
)


class TestBrierScorer:
    def test_perfect_prediction(self):
        """Predicting 100% for correct outcome gives score = 1."""
        scorer = BrierScorer()
        pred = DailyPrediction("a", "q1", date(2024, 1, 1), {"Yes": 1.0})
        score = scorer.score_prediction(pred, "Yes")
        assert score == pytest.approx(1.0, abs=0.001)
    
    def test_abstainer_baseline(self):
        """Empty prediction (abstainer) gets score = 0."""
        scorer = BrierScorer()
        pred = DailyPrediction("a", "q1", date(2024, 1, 1), {})
        score = scorer.score_prediction(pred, "Yes")
        assert score == pytest.approx(0.0, abs=0.001)
    
    def test_missing_truth_penalty(self):
        """Predicting wrong outcome only gets penalized."""
        scorer = BrierScorer()
        pred = DailyPrediction("a", "q1", date(2024, 1, 1), {"No": 1.0})
        # Brier = (1-0)² for "No" being wrong + 1 for missing truth = 2
        # Score = 1 - 2 = -1
        score = scorer.score_prediction(pred, "Yes")
        assert score == pytest.approx(-1.0, abs=0.001)
    
    def test_partial_prediction(self):
        """50% on correct outcome is better than abstaining."""
        scorer = BrierScorer()
        pred = DailyPrediction("a", "q1", date(2024, 1, 1), {"Yes": 0.5})
        # Brier = (0.5-1)² = 0.25, no penalty (truth named)
        # Score = 1 - 0.25 = 0.75
        score = scorer.score_prediction(pred, "Yes")
        assert score == pytest.approx(0.75, abs=0.001)
        assert score > 0  # Better than abstainer baseline
    
    def test_multi_outcome_correct(self):
        """Multi-outcome with correct prediction."""
        scorer = BrierScorer()
        pred = DailyPrediction("a", "q1", date(2024, 1, 1), {"A": 0.6, "B": 0.3})
        # Truth = A: Brier = (0.6-1)² + (0.3-0)² = 0.16 + 0.09 = 0.25
        score = scorer.score_prediction(pred, "A")
        assert score == pytest.approx(0.75, abs=0.01)
    
    def test_multi_outcome_wrong(self):
        """Multi-outcome where truth is not named."""
        scorer = BrierScorer()
        pred = DailyPrediction("a", "q1", date(2024, 1, 1), {"A": 0.5, "B": 0.4})
        # Truth = C (not named): Brier = 0.5² + 0.4² + 1 (penalty) = 0.25 + 0.16 + 1 = 1.41
        # Score = 1 - 1.41 = -0.41
        score = scorer.score_prediction(pred, "C")
        assert score == pytest.approx(-0.41, abs=0.01)
        assert score < 0  # Worse than abstainer
    
    def test_higher_is_better(self):
        """Verify higher_is_better is True."""
        assert BrierScorer().higher_is_better == True
    
    def test_penalizes_overconfidence(self):
        """Brier should penalize overconfidence on named outcomes."""
        scorer = BrierScorer()
        
        # When truth is "other" (unnamed), hedged prediction is better
        hedged = DailyPrediction("a", "q1", date(2024, 1, 1), {"A": 0.1, "B": 0.1})
        overconf = DailyPrediction("b", "q1", date(2024, 1, 1), {"A": 0.5, "B": 0.4})
        
        # Truth = C (other)
        # Hedged: Brier = 0.01 + 0.01 + 1 = 1.02, Score = -0.02
        # Overconf: Brier = 0.25 + 0.16 + 1 = 1.41, Score = -0.41
        hedged_score = scorer.score_prediction(hedged, "C")
        overconf_score = scorer.score_prediction(overconf, "C")
        
        assert hedged_score > overconf_score  # Hedged is better


class TestLogScorer:
    def test_perfect_prediction(self):
        """Predicting ~100% for correct outcome gives log ≈ 0."""
        scorer = LogScorer()
        pred = DailyPrediction("a", "q1", date(2024, 1, 1), {"Yes": 0.999})
        score = scorer.score_prediction(pred, "Yes")
        assert score == pytest.approx(0.0, abs=0.01)
    
    def test_worst_prediction(self):
        """Predicting 0% for correct outcome gives very negative log."""
        scorer = LogScorer()
        pred = DailyPrediction("a", "q1", date(2024, 1, 1), {"No": 1.0})
        score = scorer.score_prediction(pred, "Yes")
        assert score < -6  # log(0.001) ≈ -6.9


class TestPeerScores:
    def test_zero_sum_brier(self):
        """Peer scores must sum to zero (Brier)."""
        scorer = BrierScorer()
        predictions = {
            "a": DailyPrediction("a", "q1", date(2024, 1, 1), {"Yes": 0.9}),
            "b": DailyPrediction("b", "q1", date(2024, 1, 1), {"Yes": 0.5}),
            "c": DailyPrediction("c", "q1", date(2024, 1, 1), {"Yes": 0.1}),
        }
        
        scores = compute_snapshot_peer_scores(predictions, "Yes", scorer)
        assert sum(scores.values()) == pytest.approx(0.0, abs=1e-10)
    
    def test_zero_sum_log(self):
        """Peer scores must sum to zero (Log)."""
        scorer = LogScorer()
        predictions = {
            "a": DailyPrediction("a", "q1", date(2024, 1, 1), {"Yes": 0.9}),
            "b": DailyPrediction("b", "q1", date(2024, 1, 1), {"Yes": 0.5}),
            "c": DailyPrediction("c", "q1", date(2024, 1, 1), {"Yes": 0.1}),
        }
        
        scores = compute_snapshot_peer_scores(predictions, "Yes", scorer)
        assert sum(scores.values()) == pytest.approx(0.0, abs=1e-10)
    
    def test_better_agent_positive_peer(self):
        """Agent with better prediction gets positive peer score."""
        scorer = BrierScorer()
        predictions = {
            "good": DailyPrediction("good", "q1", date(2024, 1, 1), {"Yes": 0.9}),
            "bad": DailyPrediction("bad", "q1", date(2024, 1, 1), {"Yes": 0.1}),
        }
        
        scores = compute_snapshot_peer_scores(predictions, "Yes", scorer)
        assert scores["good"] > 0
        assert scores["bad"] < 0


class TestCarryForward:
    def test_prediction_carries_forward(self):
        """Agent's prediction should be active until updated."""
        history = PredictionHistory(
            question_id="q1",
            start_date=date(2024, 1, 1),
            resolution_date=date(2024, 1, 10)
        )
        
        pred = DailyPrediction(
            agent_id="agent_a",
            question_id="q1",
            day=date(2024, 1, 1),
            outcomes={"Yes": 0.7, "No": 0.3}
        )
        history.add_prediction(pred)
        
        active_pred = history.get_prediction_as_of("agent_a", date(2024, 1, 5))
        assert active_pred is not None
        assert active_pred.outcomes["Yes"] == 0.7


class TestResolveQuestion:
    def test_final_scores_zero_sum(self):
        """Final time-weighted peer scores must sum to zero."""
        history = PredictionHistory(
            question_id="q1",
            start_date=date(2024, 1, 1),
            resolution_date=date(2024, 1, 10)
        )
        
        history.add_prediction(DailyPrediction(
            "agent_a", "q1", date(2024, 1, 1), {"Yes": 0.8}
        ))
        history.add_prediction(DailyPrediction(
            "agent_b", "q1", date(2024, 1, 1), {"Yes": 0.2}
        ))
        
        result = resolve_question(history, "Yes", scorer=BrierScorer())
        
        total = sum(result.agent_scores.values())
        assert total == pytest.approx(0.0, abs=1e-10)
    
    def test_good_agent_wins(self):
        """Agent with better prediction gets positive final score."""
        history = PredictionHistory(
            question_id="q1",
            start_date=date(2024, 1, 1),
            resolution_date=date(2024, 1, 10)
        )
        
        history.add_prediction(DailyPrediction(
            "good", "q1", date(2024, 1, 1), {"Yes": 0.9}
        ))
        history.add_prediction(DailyPrediction(
            "bad", "q1", date(2024, 1, 1), {"Yes": 0.1}
        ))
        
        result = resolve_question(history, "Yes")
        
        assert result.agent_scores["good"] > 0
        assert result.agent_scores["bad"] < 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
