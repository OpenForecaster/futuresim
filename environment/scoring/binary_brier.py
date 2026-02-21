"""
Binary Brier score for Yes/No questions.

Score = (p_yes - y)^2

Where:
- p_yes is the forecast probability assigned to "Yes"
- y = 1 if truth is Yes, else 0

Properties:
- Lower is better (0 is perfect, 1 is worst)
- Proper scoring rule for binary outcomes
"""

from .base import BaseScorer, DailyPrediction


class BinaryBrierScorer(BaseScorer):
    """Classic binary Brier scorer for Yes/No markets."""

    # Lower is better for raw Brier.
    higher_is_better = False

    def score_prediction(
        self,
        pred: DailyPrediction,
        ground_truth: str,
        matcher=None,
        **kwargs
    ) -> float:
        del matcher
        del kwargs

        y = 1.0 if self._is_yes_label(ground_truth) else 0.0
        p_yes = self._get_yes_probability(pred)
        if p_yes < 0.0:
            p_yes = 0.0
        elif p_yes > 1.0:
            p_yes = 1.0
        return (p_yes - y) ** 2

    @staticmethod
    def _normalize(text: str) -> str:
        return str(text or "").strip().lower()

    @classmethod
    def _is_yes_label(cls, text: str) -> bool:
        norm = cls._normalize(text)
        return norm in {"yes", "true", "1", "1.0"}

    @classmethod
    def _is_no_label(cls, text: str) -> bool:
        norm = cls._normalize(text)
        return norm in {"no", "false", "0", "0.0"}

    @classmethod
    def _get_yes_probability(cls, pred: DailyPrediction) -> float:
        yes_prob = None
        no_prob = None
        for outcome, prob in (pred.outcomes or {}).items():
            if cls._is_yes_label(outcome):
                yes_prob = float(prob)
            elif cls._is_no_label(outcome):
                no_prob = float(prob)

        if yes_prob is not None:
            return yes_prob
        if no_prob is not None:
            return 1.0 - no_prob
        return 0.0


default_scorer = BinaryBrierScorer()

