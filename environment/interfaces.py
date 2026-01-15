"""
Data classes for the forecasting environment.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class PredictionSubmission:
    """Agent's prediction on a question."""
    question_id: str
    outcomes: Dict[str, float]  # {outcome_str: probability}, should sum to <= 1
