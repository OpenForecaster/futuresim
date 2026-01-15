"""
Brier Skill Score for free-form outcome prediction.

Score = 1 - Σ(p_i - y_i)²

Where:
- Sum is over {named outcomes} ∪ {truth}
- p_i = probability assigned (0 if outcome not named)
- y_i = 1 if outcome is truth, 0 otherwise

Properties:
- Higher is better (range: -1 to +1)
- 0 = no information baseline (abstainer)
- Proper scoring rule
- No overconfidence incentive
"""

from .base import BaseScorer, DailyPrediction


class BrierScorer(BaseScorer):
    """
    Brier Skill Score. Default scorer for free-form prediction.
    
    Standard multi-class Brier over {named outcomes} ∪ {truth},
    converted to skill score (1 - Brier) so higher is better.
    
    Key: if truth not in named outcomes, we include it with p=0.
    """
    
    higher_is_better = True
    
    def score_prediction(
        self, 
        pred: DailyPrediction, 
        ground_truth: str, 
        matcher=None
    ) -> float:
        """
        Compute Brier Skill Score.
        
        Score = 1 - Σ(p_i - y_i)² over {named outcomes} ∪ {truth}
        """
        brier = 0.0
        
        # Find which named outcome matches truth (if any)
        matched_outcome = None
        
        # Helper for normalized comparison (lowercase, no spaces)
        def normalize(s: str) -> str:
            return s.lower().replace(" ", "").strip()
        
        truth_norm = normalize(ground_truth)
        
        if ground_truth in pred.outcomes:
            matched_outcome = ground_truth
        elif matcher:
            # LLM-based semantic matching
            for outcome in pred.outcomes:
                if matcher.is_equivalent(outcome, ground_truth):
                    matched_outcome = outcome
                    break
        else:
            # Exact matching with normalization (lowercase + no spaces)
            for outcome in pred.outcomes:
                if normalize(outcome) == truth_norm:
                    matched_outcome = outcome
                    break
        
        # Sum over named outcomes
        for outcome, prob in pred.outcomes.items():
            y = 1.0 if outcome == matched_outcome else 0.0
            brier += (prob - y) ** 2
        
        # If truth not in named outcomes, include it with p=0
        if matched_outcome is None:
            brier += (0.0 - 1.0) ** 2  # = 1
        
        return 1.0 - brier


default_scorer = BrierScorer()
