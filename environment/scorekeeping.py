from __future__ import annotations

import json
import os
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from .scoring import compute_snapshot_peer_scores, resolve_question


def get_metrics_evaluation_date(
    sim_date: date,
    *,
    end_date: Optional[date],
    timegap_days: int,
) -> date:
    if end_date is None:
        return sim_date
    return min(sim_date + timedelta(days=timegap_days - 1), end_date)


def compute_daily_active_scores(
    env,
    active_questions,
    evaluation_date: Optional[date] = None,
    metrics_context_date: Optional[date] = None,
) -> Dict[str, Dict[str, float]]:
    active_stats: Dict[str, Dict[str, float]] = {}
    interval_end = evaluation_date or env.current_date
    context_date = metrics_context_date if metrics_context_date is not None else env.current_date

    for q in active_questions:
        if not q.ground_truth_answer:
            continue

        history = env.prediction_histories.get(q.qid)
        if not history or not history.predictions:
            continue

        effective_eval_date = min(interval_end, q.resolution_date - timedelta(days=1))
        if effective_eval_date < context_date:
            continue

        snapshot = {}
        for aid in history.predictions.keys():
            pred = history.get_prediction_as_of(aid, effective_eval_date)
            if pred:
                snapshot[aid] = pred

        snapshot_peers = {}
        if snapshot:
            snapshot_peers = compute_snapshot_peer_scores(
                snapshot,
                q.ground_truth_answer,
                env.scorer,
                env.matcher,
                question_id=q.qid,
                question_title=q.title,
            )

        result = resolve_question(
            history,
            q.ground_truth_answer,
            matcher=env.matcher,
            scorer=env.scorer,
            evaluation_date=effective_eval_date,
            question_title=q.title,
        )
        tw_peers = result.agent_scores

        for agent in env.agents:
            aid = agent.agent_id
            pred = snapshot.get(aid)
            if not pred:
                continue

            brier = env.scorer.score_prediction(
                pred,
                q.ground_truth_answer,
                env.matcher,
                question_id=q.qid,
                question_title=q.title,
            )
            is_accurate = env._is_top_choice_correct(
                pred,
                q.ground_truth_answer,
                question_id=q.qid,
                question_title=q.title,
            )
            truth_prob = env._get_truth_probability_mass(
                pred,
                q.ground_truth_answer,
                question_id=q.qid,
                question_title=q.title,
            )

            entry = active_stats.setdefault(
                aid,
                {
                    "raw_brier": 0.0,
                    "peer_sum": 0.0,
                    "tw_peer_sum": 0.0,
                    "correct_count": 0.0,
                    "truth_prob_sum": 0.0,
                    "count": 0.0,
                },
            )
            entry["raw_brier"] += brier
            entry["peer_sum"] += snapshot_peers.get(aid, 0.0)
            entry["tw_peer_sum"] += tw_peers.get(aid, 0.0)
            entry["correct_count"] += 1.0 if is_accurate else 0.0
            entry["truth_prob_sum"] += truth_prob
            entry["count"] += 1.0

    return active_stats


def warmup_matcher_cache(env, questions) -> None:
    if not env.matcher or not hasattr(env.matcher, "warmup_cache"):
        return

    items = []
    for q in questions:
        gt = q.ground_truth_answer
        if not gt:
            continue
        history = env.prediction_histories.get(q.qid)
        if not history or not history.predictions:
            continue
        seen_outcomes = set()
        for pred_list in history.predictions.values():
            for pred in pred_list:
                if not pred or not pred.outcomes:
                    continue
                for outcome in pred.outcomes:
                    if outcome in seen_outcomes:
                        continue
                    seen_outcomes.add(outcome)
                    items.append((outcome, gt, q.qid, q.title))

    if items:
        env.matcher.warmup_cache(items, max_concurrency=300)


def get_env_matcher_timing_snapshot(env) -> Dict[str, float]:
    if not env.matcher:
        return {"matcher_count": 0, "matcher_total_seconds": 0.0, "matcher_total_cost": 0.0}

    if hasattr(env.matcher, "get_timing_snapshot"):
        snapshot = env.matcher.get_timing_snapshot() or {}
        return {
            "matcher_count": int(snapshot.get("matcher_count", 0)),
            "matcher_total_seconds": float(snapshot.get("matcher_total_seconds", 0.0)),
            "matcher_total_cost": float(snapshot.get("matcher_total_cost", 0.0)),
        }

    stats = env.matcher.get_stats() if hasattr(env.matcher, "get_stats") else {}
    return {
        "matcher_count": int(stats.get("matcher_count", 0)),
        "matcher_total_seconds": float(stats.get("matcher_total_seconds", 0.0)),
        "matcher_total_cost": float(stats.get("matcher_total_cost", 0.0)),
    }


def compute_env_matcher_delta(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
    count = max(0, int(after.get("matcher_count", 0)) - int(before.get("matcher_count", 0)))
    total = max(
        0.0,
        float(after.get("matcher_total_seconds", 0.0)) - float(before.get("matcher_total_seconds", 0.0)),
    )
    cost = max(
        0.0,
        float(after.get("matcher_total_cost", 0.0)) - float(before.get("matcher_total_cost", 0.0)),
    )
    avg = (total / count) if count > 0 else 0.0
    return {
        "matcher_count": count,
        "matcher_total_seconds": round(total, 3),
        "matcher_avg_seconds": round(avg, 3),
        "matcher_cost": round(cost, 6),
    }


def inject_env_matcher_timing_into_agent_logs(env, sim_date: date, day_matcher: Dict[str, float]) -> None:
    if not env.agents:
        return

    target_date = str(sim_date)
    for agent in env.agents:
        stats_path = os.path.join(env.output_dir, "agents", agent.agent_id, "timing_stats.jsonl")
        if not os.path.exists(stats_path):
            continue

        try:
            with open(stats_path, "r") as f:
                lines = f.readlines()

            target_idx = None
            target_row = None
            for idx in range(len(lines) - 1, -1, -1):
                line = lines[idx].strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("date") == target_date:
                    target_idx = idx
                    target_row = row
                    break

            if target_idx is None or target_row is None:
                continue

            target_row.pop("feedback_matcher_count", None)
            target_row.pop("feedback_matcher_total_seconds", None)
            target_row.pop("feedback_matcher_avg_seconds", None)
            target_row.pop("feedback_matcher_cost", None)
            target_row.update(day_matcher)

            lines[target_idx] = json.dumps(target_row) + "\n"
            with open(stats_path, "w") as f:
                f.writelines(lines)
        except OSError:
            continue


def compute_resolved_metrics_from_events(
    env,
    *,
    source_split: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for event in env.resolution_events:
        if source_split and event.get("source_split") != source_split:
            continue
        for agent_id, per_agent in (event.get("agents") or {}).items():
            if not isinstance(per_agent, dict):
                continue
            entry = stats.setdefault(
                agent_id,
                {
                    "raw_brier": 0.0,
                    "peer_sum": 0.0,
                    "tw_peer_sum": 0.0,
                    "correct_count": 0.0,
                    "truth_prob_sum": 0.0,
                    "count": 0.0,
                },
            )
            brier = per_agent.get("brier")
            if brier is None:
                continue
            entry["raw_brier"] += float(brier)
            entry["peer_sum"] += float(per_agent.get("snapshot_peer", 0.0) or 0.0)
            entry["tw_peer_sum"] += float(per_agent.get("tw_peer", 0.0) or 0.0)
            entry["correct_count"] += 1.0 if per_agent.get("is_accurate") else 0.0
            entry["truth_prob_sum"] += float(per_agent.get("truth_prob", 0.0) or 0.0)
            entry["count"] += 1.0
    return stats


def tv_distance(left: Optional[Dict[str, float]], right: Optional[Dict[str, float]]) -> float:
    outcomes = set((left or {}).keys()) | set((right or {}).keys())
    if not outcomes:
        return 0.0
    return 0.5 * sum(
        abs(float((left or {}).get(outcome, 0.0)) - float((right or {}).get(outcome, 0.0)))
        for outcome in outcomes
    )


def build_metrics_list(
    env,
    *,
    active_questions,
    source_split: Optional[str] = None,
) -> List[Dict[str, Any]]:
    filtered_active_questions = [
        q for q in active_questions if not source_split or q.source_split == source_split
    ]
    active_stats = compute_daily_active_scores(
        env,
        filtered_active_questions,
        evaluation_date=get_metrics_evaluation_date(
            env.current_date,
            end_date=env.end_date,
            timegap_days=env.timegap_days,
        ),
    )
    resolved_stats = compute_resolved_metrics_from_events(env, source_split=source_split)

    daily_submission_stats: Dict[str, Dict[str, float]] = {}
    for q in filtered_active_questions:
        history = env.prediction_histories.get(q.qid)
        if not history:
            continue
        for aid, preds in history.predictions.items():
            for idx, pred in enumerate(preds):
                if pred.day != env.current_date:
                    continue
                entry = daily_submission_stats.setdefault(
                    aid,
                    {"count": 0.0, "tv_sum": 0.0, "tv_count": 0.0},
                )
                entry["count"] += 1.0
                if idx > 0:
                    entry["tv_sum"] += tv_distance(pred.outcomes, preds[idx - 1].outcomes)
                    entry["tv_count"] += 1.0

    metrics_list = []
    for agent in env.agents:
        aid = agent.agent_id
        resolved = resolved_stats.get(
            aid,
            {
                "raw_brier": 0.0,
                "peer_sum": 0.0,
                "tw_peer_sum": 0.0,
                "correct_count": 0.0,
                "truth_prob_sum": 0.0,
                "count": 0.0,
            },
        )
        agent_active = active_stats.get(
            aid,
            {
                "raw_brier": 0.0,
                "peer_sum": 0.0,
                "tw_peer_sum": 0.0,
                "correct_count": 0.0,
                "truth_prob_sum": 0.0,
                "count": 0.0,
            },
        )
        submission_stats = daily_submission_stats.get(
            aid,
            {
                "count": 0.0,
                "tv_sum": 0.0,
                "tv_count": 0.0,
            },
        )

        total_brier_sum = resolved["raw_brier"] + agent_active["raw_brier"]
        total_peer_sum = resolved["peer_sum"] + agent_active["peer_sum"]
        total_tw_peer_sum = resolved["tw_peer_sum"] + agent_active["tw_peer_sum"]
        total_correct = resolved["correct_count"] + agent_active["correct_count"]
        total_truth_prob = resolved["truth_prob_sum"] + agent_active["truth_prob_sum"]
        total_count = resolved["count"] + agent_active["count"]

        avg_brier = total_brier_sum / total_count if total_count > 0 else 0.0
        accuracy = (total_correct / total_count * 100) if total_count > 0 else 0.0
        exp_acc = total_truth_prob / total_count if total_count > 0 else 0.0
        avg_submission_tv = (
            submission_stats["tv_sum"] / submission_stats["tv_count"]
            if submission_stats["tv_count"] > 0
            else 0.0
        )

        metrics_list.append(
            {
                "agent_id": aid,
                "avg_brier": avg_brier,
                "peer_score": total_peer_sum,
                "tw_peer_score": total_tw_peer_sum,
                "accuracy": accuracy,
                "exp_acc": exp_acc,
                "total_predictions": int(total_count),
                "daily_submissions": int(submission_stats["count"]),
                "avg_submission_tv_to_prev": avg_submission_tv,
            }
        )
    return metrics_list


def save_daily_metrics(env) -> None:
    active_questions = env.q_pool.get_active()
    metrics_list = build_metrics_list(env, active_questions=active_questions)
    test_metrics_list = build_metrics_list(
        env,
        active_questions=active_questions,
        source_split="test",
    )
    env.logger.log_daily_metrics(env.current_date, metrics_list)
    env.logger.log_test_daily_metrics(env.current_date, test_metrics_list)


__all__ = [
    "build_metrics_list",
    "compute_daily_active_scores",
    "compute_env_matcher_delta",
    "compute_resolved_metrics_from_events",
    "get_env_matcher_timing_snapshot",
    "get_metrics_evaluation_date",
    "inject_env_matcher_timing_into_agent_logs",
    "save_daily_metrics",
    "tv_distance",
    "warmup_matcher_cache",
]
