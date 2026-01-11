"""
Metaculus-style scoring system for free-form forecasting.

Based on:
- Log score: ln(P(true_outcome))
- Peer score: 100 × (my_log - avg(others_log))  
- Time-weighted averaging: predictions weighted by duration
"""

import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import date


# Clamp probabilities to avoid log(0)
MIN_PROB = 0.001
MAX_PROB = 0.999


def log_score(prob: float) -> float:
    """
    Compute log score for a probability.
    Higher is better. Max = 0 (at prob=1), Min ≈ -6.9 (at prob=0.001).
    """
    clamped = max(MIN_PROB, min(MAX_PROB, prob))
    return math.log(clamped)


def peer_score(my_log: float, others_logs: List[float]) -> float:
    """
    Compute peer score: how much better than average.
    Returns 100 × (my_log - mean(others_log)).
    Positive = better than peers. Mean peer score = 0.
    """
    if not others_logs:
        return 0.0
    avg_others = sum(others_logs) / len(others_logs)
    return 100 * (my_log - avg_others)


@dataclass
class DailyPrediction:
    """A single prediction made by an agent on a specific day."""
    agent_id: str
    question_id: str
    day: date
    outcomes: Dict[str, float]  # {outcome_str: probability}
    
    def get_prob(self, outcome: str) -> float:
        """Get probability for a specific outcome, 0 if not predicted."""
        return self.outcomes.get(outcome, 0.0)
    
    @property
    def total_prob(self) -> float:
        return sum(self.outcomes.values())


@dataclass
class PredictionHistory:
    """Track all predictions for a question."""
    question_id: str
    start_date: date
    resolution_date: date
    
    # agent_id -> list of predictions (one per day they predicted)
    predictions: Dict[str, List[DailyPrediction]] = field(default_factory=dict)
    
    def add_prediction(self, pred: DailyPrediction):
        if pred.agent_id not in self.predictions:
            self.predictions[pred.agent_id] = []
        self.predictions[pred.agent_id].append(pred)
    
    def get_latest_prediction(self, agent_id: str) -> Optional[DailyPrediction]:
        """Get agent's most recent prediction."""
        if agent_id not in self.predictions or not self.predictions[agent_id]:
            return None
        return self.predictions[agent_id][-1]
    
    def get_all_current_predictions(self) -> Dict[str, DailyPrediction]:
        """Get each agent's current (latest) prediction."""
        return {
            agent_id: preds[-1] 
            for agent_id, preds in self.predictions.items() 
            if preds
        }
    
    def total_days(self) -> int:
        return (self.resolution_date - self.start_date).days + 1


def compute_aggregate(predictions: List[DailyPrediction]) -> Dict[str, float]:
    """
    Compute aggregate probability distribution from multiple predictions.
    Simple average across agents, per outcome.
    """
    if not predictions:
        return {}
    
    # Collect all outcomes
    all_outcomes = set()
    for pred in predictions:
        all_outcomes.update(pred.outcomes.keys())
    
    # Average probability per outcome
    aggregate = {}
    for outcome in all_outcomes:
        probs = [pred.get_prob(outcome) for pred in predictions]
        aggregate[outcome] = sum(probs) / len(predictions)
    
    return aggregate


def compute_daily_peer_scores(
    predictions: List[DailyPrediction],
    ground_truth: str,
    matcher=None
) -> Dict[str, float]:
    """
    Compute peer scores for all predictions made on the same day.
    
    Args:
        predictions: List of predictions from different agents on same day
        ground_truth: The true outcome (for computing log scores)
        matcher: Optional AnswerMatcher for free-form outcome matching
        
    Returns:
        {agent_id: peer_score}
    """
    if len(predictions) < 2:
        # Can't compute peer score with < 2 agents
        return {pred.agent_id: 0.0 for pred in predictions}
    
    # Compute log score for each agent
    log_scores = {}
    for pred in predictions:
        prob = _get_matched_prob(pred, ground_truth, matcher)
        log_scores[pred.agent_id] = log_score(prob)
    
    # Compute peer scores
    peer_scores = {}
    for agent_id, my_log in log_scores.items():
        others_logs = [ls for aid, ls in log_scores.items() if aid != agent_id]
        peer_scores[agent_id] = peer_score(my_log, others_logs)
    
    return peer_scores


def _get_matched_prob(pred: DailyPrediction, ground_truth: str, matcher) -> float:
    """
    Get the probability the agent assigned to the ground truth outcome.
    Uses answer matching if provided.
    """
    # Direct match first
    if ground_truth in pred.outcomes:
        return pred.outcomes[ground_truth]
    
    # Try answer matching
    if matcher:
        for outcome, prob in pred.outcomes.items():
            if matcher.is_equivalent(outcome, ground_truth):
                return prob
    
    # No match - agent didn't predict this outcome
    return 0.0


def compute_time_weighted_score(
    history: PredictionHistory,
    agent_id: str,
    ground_truth: str,
    all_daily_peer_scores: Dict[date, Dict[str, float]]
) -> float:
    """
    Compute time-weighted average of peer scores for an agent.
    
    Each day's peer score is weighted by how long that prediction was active.
    """
    if agent_id not in history.predictions:
        return 0.0
    
    agent_preds = history.predictions[agent_id]
    if not agent_preds:
        return 0.0
    
    total_days = history.total_days()
    weighted_sum = 0.0
    total_weight = 0.0
    
    for i, pred in enumerate(agent_preds):
        # Duration: from this prediction to next (or resolution)
        if i + 1 < len(agent_preds):
            next_date = agent_preds[i + 1].day
        else:
            next_date = history.resolution_date
        
        duration = (next_date - pred.day).days
        if duration <= 0:
            duration = 1  # Minimum 1 day
        
        # Get peer score for this day
        day_scores = all_daily_peer_scores.get(pred.day, {})
        peer_sc = day_scores.get(agent_id, 0.0)
        
        weighted_sum += peer_sc * duration
        total_weight += duration
    
    if total_weight == 0:
        return 0.0
    
    return weighted_sum / total_weight


@dataclass
class QuestionResult:
    """Final scores for a resolved question."""
    question_id: str
    ground_truth: str
    agent_scores: Dict[str, float]  # agent_id -> final time-weighted peer score
    aggregate_at_resolution: Dict[str, float]


def resolve_question(
    history: PredictionHistory,
    ground_truth: str,
    matcher=None
) -> QuestionResult:
    """
    Resolve a question and compute final scores for all agents.
    """
    # Group predictions by day
    predictions_by_day: Dict[date, List[DailyPrediction]] = {}
    for agent_id, preds in history.predictions.items():
        for pred in preds:
            if pred.day not in predictions_by_day:
                predictions_by_day[pred.day] = []
            predictions_by_day[pred.day].append(pred)
    
    # Compute peer scores for each day
    daily_peer_scores = {}
    for day, day_preds in predictions_by_day.items():
        daily_peer_scores[day] = compute_daily_peer_scores(
            day_preds, ground_truth, matcher
        )
    
    # Compute time-weighted score for each agent
    agent_scores = {}
    for agent_id in history.predictions:
        agent_scores[agent_id] = compute_time_weighted_score(
            history, agent_id, ground_truth, daily_peer_scores
        )
    
    # Final aggregate
    current_preds = list(history.get_all_current_predictions().values())
    aggregate = compute_aggregate(current_preds)
    
    return QuestionResult(
        question_id=history.question_id,
        ground_truth=ground_truth,
        agent_scores=agent_scores,
        aggregate_at_resolution=aggregate
    )
