"""Tests for ``environment.forecast_metrics`` helpers."""

from datetime import date

import pytest

from environment.forecast_metrics import accuracy_rank_bonus
from environment.scoring.base import DailyPrediction


def test_accuracy_rank_bonus_top_is_correct():
    pred = DailyPrediction("a", "q1", date(2024, 1, 1), {"A": 0.1, "B": 0.5, "C": 0.4})
    assert accuracy_rank_bonus(pred, "B", matcher=None) == pytest.approx(1.0)


def test_accuracy_rank_bonus_worst_rank():
    pred = DailyPrediction("a", "q1", date(2024, 1, 1), {"A": 0.5, "B": 0.3, "C": 0.2})
    assert accuracy_rank_bonus(pred, "C", matcher=None) == pytest.approx(1.0 / 3.0)


def test_accuracy_rank_bonus_middle_rank():
    pred = DailyPrediction("a", "q1", date(2024, 1, 1), {"A": 0.5, "B": 0.3, "C": 0.2})
    assert accuracy_rank_bonus(pred, "B", matcher=None) == pytest.approx(2.0 / 3.0)


def test_accuracy_rank_bonus_tie_breaker_deterministic():
    pred = DailyPrediction("a", "q1", date(2024, 1, 1), {"A": 0.5, "B": 0.5})
    # Same probability: sort key (-p, name) → A before B.
    assert accuracy_rank_bonus(pred, "A", matcher=None) == pytest.approx(1.0)
    assert accuracy_rank_bonus(pred, "B", matcher=None) == pytest.approx(0.5)


def test_accuracy_rank_bonus_no_match():
    pred = DailyPrediction("a", "q1", date(2024, 1, 1), {"A": 1.0})
    assert accuracy_rank_bonus(pred, "Z", matcher=None) == 0.0


def test_accuracy_rank_bonus_empty():
    pred = DailyPrediction("a", "q1", date(2024, 1, 1), {})
    assert accuracy_rank_bonus(pred, "A", matcher=None) == 0.0
