"""
Forecast quality metrics shared by ``SimulationEnvironment`` (``daily_metrics.csv``)
and SkyRL OpenForesight warmup eval (``rollout_metrics`` / W&B).

Single source of truth for top-1 accuracy, truth probability mass (``exp_acc``), and
Brier skill from ``BaseScorer.score_prediction`` (``BrierScorer`` → ``1 - Σ(p−y)²``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .scoring.base import BaseScorer, DailyPrediction


def normalize_outcome(s: str) -> str:
    """Normalize outcome strings for deterministic exact matching."""
    return s.lower().replace(" ", "").strip()


def is_top_choice_correct(
    pred: Optional[DailyPrediction],
    ground_truth: str,
    matcher,
    question_id: Optional[str] = None,
    question_title: Optional[str] = None,
) -> bool:
    """True when the highest-probability predicted outcome matches ground truth."""
    if not pred or not pred.outcomes:
        return False

    max_prob = -1.0
    best_outcome = None
    for outcome, prob in pred.outcomes.items():
        if prob > max_prob:
            max_prob = prob
            best_outcome = outcome

    if not best_outcome:
        return False

    if matcher:
        return bool(
            matcher.is_equivalent(
                best_outcome,
                ground_truth,
                question_id=question_id,
                question_title=question_title,
                match_type="check_guess",
            )
        )

    return normalize_outcome(best_outcome) == normalize_outcome(ground_truth)


def truth_probability_mass(
    pred: Optional[DailyPrediction],
    ground_truth: str,
    matcher,
    question_id: Optional[str] = None,
    question_title: Optional[str] = None,
) -> float:
    """Total probability on outcomes equivalent to ground truth (``exp_acc`` per question)."""
    if not pred or not pred.outcomes or not ground_truth:
        return 0.0

    truth_norm = normalize_outcome(ground_truth)
    prob_sum = 0.0
    for outcome, prob in pred.outcomes.items():
        if outcome == ground_truth:
            is_match = True
        elif matcher:
            is_match = bool(
                matcher.is_equivalent(
                    outcome,
                    ground_truth,
                    question_id=question_id,
                    question_title=question_title,
                    match_type="check_guess",
                )
            )
        else:
            is_match = normalize_outcome(outcome) == truth_norm

        if is_match:
            prob_sum += prob

    return prob_sum


def _outcome_matches_truth(
    outcome: str,
    ground_truth: str,
    matcher,
    *,
    question_id: Optional[str] = None,
    question_title: Optional[str] = None,
) -> bool:
    if outcome == ground_truth:
        return True
    if matcher:
        return bool(
            matcher.is_equivalent(
                outcome,
                ground_truth,
                question_id=question_id,
                question_title=question_title,
                match_type="check_guess",
            )
        )
    return normalize_outcome(outcome) == normalize_outcome(ground_truth)


def accuracy_rank_bonus(
    pred: DailyPrediction,
    ground_truth: str,
    *,
    matcher,
    question_id: Optional[str] = None,
    question_title: Optional[str] = None,
) -> float:
    """
    Exploration-friendly bonus from the rank of the correct outcome when outcomes are sorted
    by assigned probability (descending). For ``O`` predicted outcomes,

        (O - rank + 1) / O

    with ``rank`` in ``1..O`` (1 = highest probability). If no predicted outcome matches the
    truth (exact or matcher), returns ``0.0``. Ties break deterministically by outcome string.
    """
    if not pred.outcomes or not ground_truth:
        return 0.0

    outcomes = pred.outcomes
    o_count = len(outcomes)
    if o_count == 0:
        return 0.0

    ranked = sorted(outcomes.items(), key=lambda kv: (-kv[1], kv[0]))
    matching_ranks: List[int] = []
    for rank, (outcome, _) in enumerate(ranked, start=1):
        if _outcome_matches_truth(
            outcome,
            ground_truth,
            matcher,
            question_id=question_id,
            question_title=question_title,
        ):
            matching_ranks.append(rank)
    if not matching_ranks:
        return 0.0
    best_rank = min(matching_ranks)
    return float(o_count - best_rank + 1) / float(o_count)


def forecast_scalar_metrics(
    pred: DailyPrediction,
    ground_truth: str,
    *,
    matcher,
    scorer: BaseScorer,
    question_id: Optional[str] = None,
    question_title: Optional[str] = None,
) -> tuple[float, bool, float]:
    """
    Return ``(brier_skill, top1_correct, truth_prob_mass)`` for one submitted prediction.
    """
    brier = float(
        scorer.score_prediction(
            pred,
            ground_truth,
            matcher,
            question_id=question_id,
            question_title=question_title,
        )
    )
    top1 = is_top_choice_correct(pred, ground_truth, matcher, question_id=question_id, question_title=question_title)
    truth = truth_probability_mass(pred, ground_truth, matcher, question_id=question_id, question_title=question_title)
    return brier, top1, float(truth)


@dataclass(frozen=True)
class OpenForesightEvalEpisode:
    """One SkyRL episode after ``get_metrics()`` (internal aggregation input)."""

    valid_submit: bool
    brier_skill: float
    top1_correct: bool
    truth_prob: float


def rollup_openforesight_eval_metrics(episodes: Sequence[OpenForesightEvalEpisode]) -> Dict[str, float]:
    """
    Pool episodes like ``daily_metrics`` active snapshot: ``avg_brier`` over valid submits only;
    ``accuracy`` / ``exp_acc`` over all episodes (invalid ⇒ wrong, 0 truth mass).
    """
    n = len(episodes)
    if n == 0:
        return {}

    valid = [e for e in episodes if e.valid_submit]
    brier_vals = [e.brier_skill for e in valid]
    avg_brier = float(sum(brier_vals) / len(brier_vals)) if brier_vals else 0.0

    accuracy = sum(1.0 if e.top1_correct else 0.0 for e in episodes) / n * 100.0
    exp_acc = sum(e.truth_prob for e in episodes) / n
    format_failures = float(n - len(valid))

    return {
        "avg_brier": avg_brier,
        "accuracy": float(accuracy),
        "exp_acc": float(exp_acc),
        "format_failures": format_failures,
        "valid_submits": float(len(valid)),
        "total_episodes": float(n),
    }


def episodes_from_fs_metric_dicts(metrics: List[Dict[str, Any]]) -> List[OpenForesightEvalEpisode]:
    """Parse SkyRL per-episode ``get_metrics()`` rows (``fs_*`` scratch keys)."""
    out: List[OpenForesightEvalEpisode] = []
    for m in metrics:
        valid = float(m.get("fs_valid_submit", 0.0)) >= 0.5
        out.append(
            OpenForesightEvalEpisode(
                valid_submit=valid,
                brier_skill=float(m.get("fs_brier_skill", 0.0)),
                top1_correct=float(m.get("fs_top1_correct", 0.0)) >= 0.5,
                truth_prob=float(m.get("fs_truth_prob", 0.0)),
            )
        )
    return out
