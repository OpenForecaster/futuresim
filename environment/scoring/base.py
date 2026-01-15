"""
Base classes and dataclasses for scoring.

To implement a new scoring function:
1. Subclass `BaseScorer`
2. Implement `compute_score()` and `score_prediction()`
3. Add to __init__.py exports

See `brier.py` and `log_score.py` for examples.
"""

import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import date


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
    
    def get_prediction_as_of(self, agent_id: str, target_date: date) -> Optional[DailyPrediction]:
        """
        Get agent's active prediction as of target_date (carry-forward).
        Returns None if agent hadn't predicted by target_date.
        """
        if agent_id not in self.predictions:
            return None
        
        # Find latest prediction on or before target_date
        latest = None
        for pred in self.predictions[agent_id]:
            if pred.day <= target_date:
                latest = pred
            else:
                break  # Predictions are chronological
        return latest
    
    def total_days(self) -> int:
        return (self.resolution_date - self.start_date).days + 1


@dataclass
class QuestionResult:
    """Final scores for a resolved question."""
    question_id: str
    ground_truth: str
    agent_scores: Dict[str, float]  # agent_id -> final time-weighted peer score
    aggregate_at_resolution: Dict[str, float]


class BaseScorer(ABC):
    """
    Abstract base class for scoring functions.
    
    Subclass this to implement new scoring rules. Key methods:
    - `score_prediction()`: Score a prediction against ground truth
    - `peer_score()`: Convert raw scores to relative peer scores
    
    Properties a proper scorer should have:
    - Honest reporting maximizes expected score
    - No incentive for overconfidence on named outcomes
    """
    
    # Whether higher scores are better (True) or lower (False)
    higher_is_better: bool = True
    
    @abstractmethod
    def score_prediction(
        self, 
        pred: DailyPrediction, 
        ground_truth: str, 
        matcher=None
    ) -> float:
        """
        Score a prediction against the ground truth.
        
        Args:
            pred: The prediction to score
            ground_truth: The actual outcome
            matcher: Optional AnswerMatcher for semantic matching
            
        Returns:
            Raw score (interpretation depends on scorer)
        """
        pass
    
    def peer_score(self, my_score: float, others_scores: List[float]) -> float:
        """
        Compute peer score: how much better than average.
        
        Returns 100 × (my_score - mean(others_scores)) if higher_is_better,
        or 100 × (mean(others_scores) - my_score) if lower is better.
        
        Positive = better than peers. Mean peer score = 0.
        """
        if not others_scores:
            return 0.0
        avg_others = sum(others_scores) / len(others_scores)
        
        if self.higher_is_better:
            return 100 * (my_score - avg_others)
        else:
            return 100 * (avg_others - my_score)
    
    def _get_matched_prob(
        self, 
        pred: DailyPrediction, 
        ground_truth: str, 
        matcher=None
    ) -> float:
        """
        Get the probability the agent assigned to the ground truth outcome.
        Uses answer matching if provided, otherwise normalized string comparison.
        """
        # Helper for normalized comparison (lowercase, no spaces)
        def normalize(s: str) -> str:
            return s.lower().replace(" ", "").strip()
        
        # Direct match first
        if ground_truth in pred.outcomes:
            return pred.outcomes[ground_truth]
        
        # Try answer matching (LLM-based)
        if matcher:
            for outcome, prob in pred.outcomes.items():
                if matcher.is_equivalent(outcome, ground_truth):
                    return prob
        else:
            # Exact matching with normalization (lowercase + no spaces)
            truth_norm = normalize(ground_truth)
            for outcome, prob in pred.outcomes.items():
                if normalize(outcome) == truth_norm:
                    return prob
        
        # No match - agent didn't predict this outcome
        return 0.0


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
