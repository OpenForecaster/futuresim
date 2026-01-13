"""
Log-based scorer (Metaculus-style).

Log score = ln(P(true_outcome))

Properties:
- Higher is better
- Proper scoring rule for fixed-outcome settings
- In free-form: incentivizes overconfidence on named outcomes
- Harshly penalizes low probabilities on correct outcome

Use BrierScorer instead for free-form prediction to avoid
overconfidence incentive.
"""

import math
from .base import BaseScorer, DailyPrediction


# Clamp probabilities to avoid log(0)
MIN_PROB = 0.001
MAX_PROB = 0.999


class LogScorer(BaseScorer):
    """
    Log-based scorer. Metaculus-style.
    
    WARNING: In free-form prediction, this incentivizes overconfidence
    on named outcomes. Consider using BrierScorer instead.
    
    Peer score = 100 × (my_log - avg_others_log)
    Positive = better than peers.
    """
    
    higher_is_better = True  # Higher log score is better
    
    def __init__(self, min_prob: float = MIN_PROB, max_prob: float = MAX_PROB):
        self.min_prob = min_prob
        self.max_prob = max_prob
    
    def score_prediction(
        self, 
        pred: DailyPrediction, 
        ground_truth: str, 
        matcher=None
    ) -> float:
        """
        Compute log score for a prediction.
        
        log_score = ln(P(ground_truth))
        
        If ground_truth doesn't match any outcome, P = 0 (clamped to min_prob).
        """
        prob = self._get_matched_prob(pred, ground_truth, matcher)
        clamped = max(self.min_prob, min(self.max_prob, prob))
        return math.log(clamped)


# Convenience instance
default_scorer = LogScorer()
