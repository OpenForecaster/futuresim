"""
Modular scoring system for free-form forecasting.

Default: BrierScorer (proper for free-form, no overconfidence incentive)
Alternative: LogScorer (Metaculus-style, but has overconfidence incentive in free-form)

To add a new scoring function:
1. Subclass `BaseScorer` from `base.py`
2. Implement `score_prediction()` 
3. Add import here

Example usage:
    from environment.scoring import resolve_question, BrierScorer, LogScorer
    
    # Use default (Brier)
    result = resolve_question(history, ground_truth)
    
    # Use specific scorer
    result = resolve_question(history, ground_truth, scorer=LogScorer())
"""

from .base import (
    BaseScorer,
    DailyPrediction,
    PredictionHistory,
    QuestionResult,
    compute_aggregate,
)
from .brier import BrierScorer
from .log_score import LogScorer

from typing import Dict, List, Optional
from datetime import date


# Default scorer for free-form prediction
DEFAULT_SCORER = BrierScorer()


def _get_change_points(history: PredictionHistory) -> List[date]:
    """
    Get all dates where any agent's prediction changes.
    Returns sorted list of dates including start and resolution.
    
    Optimization: Instead of iterating every day, we only need to
    evaluate at points where the snapshot changes.
    """
    change_dates = {history.start_date, history.resolution_date}
    
    for agent_preds in history.predictions.values():
        for pred in agent_preds:
            if history.start_date <= pred.day <= history.resolution_date:
                change_dates.add(pred.day)
    
    return sorted(change_dates)


def _get_snapshot_at(
    history: PredictionHistory, 
    target_date: date
) -> Dict[str, DailyPrediction]:
    """
    Get snapshot of all agents' current predictions as of target_date.
    Only includes agents who have made at least one prediction by target_date.
    """
    snapshot = {}
    for agent_id in history.predictions:
        pred = history.get_prediction_as_of(agent_id, target_date)
        if pred is not None:
            snapshot[agent_id] = pred
    return snapshot


def compute_snapshot_peer_scores(
    predictions: Dict[str, DailyPrediction],
    ground_truth: str,
    scorer: BaseScorer,
    matcher=None
) -> Dict[str, float]:
    """
    Compute peer scores for a snapshot of current predictions.
    
    Args:
        predictions: {agent_id: current_prediction} snapshot
        ground_truth: The true outcome
        scorer: Scoring function to use
        matcher: Optional AnswerMatcher for free-form outcome matching
        
    Returns:
        {agent_id: peer_score}
    """
    if len(predictions) < 2:
        # Single agent: compare against baseline abstainer (score 0)
        # This is equivalent to having a virtual abstainer as a peer
        # peer_score = 100 × (agent_score - 0) = 100 × agent_score
        return {
            agent_id: 100 * scorer.score_prediction(pred, ground_truth, matcher)
            for agent_id, pred in predictions.items()
        }
    
    # Compute raw score for each agent
    raw_scores = {}
    for agent_id, pred in predictions.items():
        raw_scores[agent_id] = scorer.score_prediction(pred, ground_truth, matcher)
    
    # Compute peer scores
    peer_scores = {}
    for agent_id, my_score in raw_scores.items():
        others_scores = [s for aid, s in raw_scores.items() if aid != agent_id]
        peer_scores[agent_id] = scorer.peer_score(my_score, others_scores)
    
    return peer_scores


def resolve_question(
    history: PredictionHistory,
    ground_truth: str,
    matcher=None,
    scorer: Optional[BaseScorer] = None
) -> QuestionResult:
    """
    Resolve a question and compute final scores for all agents.
    
    Uses efficient segment-based computation:
    1. Find all "change points" where any prediction changes
    2. For each segment between change points, compute peer scores once
    3. Weight each segment by its duration
    
    Properties:
    - Sum of all peer scores = 0 (per snapshot and overall)
    - Agents score 0 for days before their first prediction
    
    Args:
        history: Prediction history for the question
        ground_truth: The actual outcome
        matcher: Optional AnswerMatcher for semantic matching
        scorer: Scoring function to use (default: BrierScorer)
        
    Returns:
        QuestionResult with final time-weighted peer scores
    """
    if scorer is None:
        scorer = DEFAULT_SCORER
    
    # Get all agents who participated
    all_agents = set(history.predictions.keys())
    if not all_agents:
        return QuestionResult(
            question_id=history.question_id,
            ground_truth=ground_truth,
            agent_scores={},
            aggregate_at_resolution={}
        )
    
    # Find change points for efficient computation
    change_points = _get_change_points(history)
    
    total_days = history.total_days()
    if total_days <= 0:
        total_days = 1
    
    # Accumulate weighted peer scores
    weighted_scores: Dict[str, float] = {agent: 0.0 for agent in all_agents}
    
    # Process each segment
    for i in range(len(change_points) - 1):
        segment_start = change_points[i]
        segment_end = change_points[i + 1]
        segment_days = (segment_end - segment_start).days
        
        if segment_days <= 0:
            continue
        
        # Get snapshot at start of segment (carries forward through segment)
        snapshot = _get_snapshot_at(history, segment_start)
        
        if snapshot:  # Changed from len(snapshot) >= 2 - single agent now gets scored
            # Compute peer scores for this snapshot
            segment_peer_scores = compute_snapshot_peer_scores(
                snapshot, ground_truth, scorer, matcher
            )
            
            # Weight by segment duration
            for agent_id, score in segment_peer_scores.items():
                weighted_scores[agent_id] += score * segment_days
    
    # Normalize by total days
    agent_scores = {
        agent: weighted_scores[agent] / total_days 
        for agent in all_agents
    }
    
    # Final aggregate
    current_preds = list(history.get_all_current_predictions().values())
    aggregate = compute_aggregate(current_preds)
    
    return QuestionResult(
        question_id=history.question_id,
        ground_truth=ground_truth,
        agent_scores=agent_scores,
        aggregate_at_resolution=aggregate
    )


__all__ = [
    'BaseScorer',
    'BrierScorer',
    'LogScorer',
    'DailyPrediction',
    'PredictionHistory',
    'QuestionResult',
    'compute_aggregate',
    'compute_snapshot_peer_scores',
    'resolve_question',
    'DEFAULT_SCORER',
]
