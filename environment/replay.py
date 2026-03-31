from __future__ import annotations

import json
import os
from datetime import date, timedelta
from typing import Dict, Iterator, Tuple

from .scoring import DailyPrediction, PredictionHistory


def _iter_action_records(
    actions_path: str,
    *,
    warn_on_error: bool = False,
) -> Iterator[Tuple[int, Dict]]:
    with open(actions_path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError:
                if warn_on_error:
                    print("  Warning: Skipped corrupted line in actions.jsonl")
                continue


def rescore(env) -> None:
    """
    Re-evaluate all simulation days and re-write daily_metrics.csv.
    Rebuilds prediction histories from actions.jsonl and replays all resolutions.
    """
    print(f"\nRescoring simulation history from {env.output_dir}...")

    actions_path = os.path.join(env.output_dir, "actions.jsonl")
    if not os.path.exists(actions_path):
        print(f"  Error: {actions_path} not found")
        return

    env.agent_scores = {agent.agent_id: 0.0 for agent in env.agents}
    env.agent_correct = {agent.agent_id: 0 for agent in env.agents}
    env.agent_wrong = {agent.agent_id: 0 for agent in env.agents}
    env.agent_questions = {agent.agent_id: 0 for agent in env.agents}
    env.agent_raw_brier = {agent.agent_id: 0.0 for agent in env.agents}
    env.agent_snapshot_peer = {agent.agent_id: 0.0 for agent in env.agents}
    env.agent_exp_acc_sum = {agent.agent_id: 0.0 for agent in env.agents}
    env.resolved_questions = []
    env.resolution_events = []
    env.resolved_agent_predictions = {}
    env.prediction_histories = {}
    env.q_pool.reset()

    env.logger.reset_metrics_files()

    resolutions_by_date: Dict[date, list] = {}
    all_dates = set()

    print("  Rebuilding prediction histories...")
    for _, record in _iter_action_records(actions_path, warn_on_error=True):
        sim_date_str = record.get("sim_date")
        if not sim_date_str:
            continue

        sim_date = date.fromisoformat(sim_date_str)
        all_dates.add(sim_date)
        rtype = record.get("type")

        if rtype == "prediction":
            qid = str(record.get("question_id")) if record.get("question_id") is not None else None
            agent_id = record.get("agent_id")
            outcomes = record.get("outcomes")
            if not (qid and agent_id and outcomes):
                continue

            if qid not in env.prediction_histories:
                q = env.q_pool.get_question(qid)
                if q:
                    env.prediction_histories[qid] = PredictionHistory(
                        question_id=qid,
                        start_date=sim_date,
                        resolution_date=q.resolution_date,
                    )
            if qid in env.prediction_histories:
                env.prediction_histories[qid].add_prediction(
                    DailyPrediction(
                        agent_id=agent_id,
                        question_id=qid,
                        day=sim_date,
                        outcomes=outcomes,
                    )
                )

        elif rtype == "resolution":
            qid = str(record.get("question_id")) if record.get("question_id") is not None else None
            if qid:
                resolutions_by_date.setdefault(sim_date, []).append(qid)

    if not all_dates:
        print("  No history found to rescore.")
        return

    start_date = min(all_dates)
    end_date = max(all_dates)

    print(f"  Found {len(env.prediction_histories)} questions with predictions")
    print(f"  Window: {start_date} to {end_date}")

    iter_date = start_date
    while iter_date <= end_date:
        env.current_date = iter_date

        if iter_date in resolutions_by_date:
            for qid in resolutions_by_date[iter_date]:
                q = env.q_pool.get_question(qid)
                if q and qid in env.prediction_histories:
                    env._resolve_question(q)
                    env.resolved_questions.append(q)
                    env.q_pool._resolved.add(qid)

        active_questions = [
            env.q_pool.get_question(qid)
            for qid in env.prediction_histories.keys()
            if env.q_pool.get_question(qid)
        ]
        env._update_aggregates(active_questions)

        if env.matcher and active_questions:
            env._warmup_matcher_cache(active_questions)

        env._save_daily_metrics()
        iter_date += timedelta(days=env.timegap_days)

    env.logger.metrics_file.flush()
    print(f"  Rescoring complete. Processed {len(env.resolved_questions)} resolutions.")


def restore_state(env, resume_dir: str) -> None:
    """
    Restore simulation state from actions.jsonl in the resume directory.
    Rebuilds:
    - prediction_histories (active)
    - agent_scores (resolved)
    - resolved_questions list
    - q_pool (mark questions as processed)
    - current_date (set to last_seen_date + timegap_days)
    """
    actions_path = os.path.join(resume_dir, "actions.jsonl")
    print(f"Restoring state from {actions_path}...")

    if not os.path.exists(actions_path):
        raise FileNotFoundError(f"Cannot resume: {actions_path} not found")

    last_date = env.current_date
    records_processed = 0
    env.resolved_agent_predictions = {}

    for _, record in _iter_action_records(actions_path, warn_on_error=True):
        records_processed += 1

        sim_date_str = record.get("sim_date")
        if sim_date_str:
            sim_date = date.fromisoformat(sim_date_str)
            if sim_date > last_date:
                last_date = sim_date
        else:
            sim_date = env.current_date

        rtype = record.get("type")

        if rtype == "prediction":
            qid = str(record.get("question_id")) if record.get("question_id") is not None else None
            agent_id = record.get("agent_id")
            outcomes = record.get("outcomes")

            if qid and agent_id and outcomes:
                if qid not in env.prediction_histories:
                    q = env.q_pool.get_question(qid)
                    if q:
                        env.prediction_histories[qid] = PredictionHistory(
                            question_id=qid,
                            start_date=sim_date,
                            resolution_date=q.resolution_date,
                        )
                        if sim_date < env.prediction_histories[qid].start_date:
                            env.prediction_histories[qid].start_date = sim_date

                if qid in env.prediction_histories:
                    env.prediction_histories[qid].add_prediction(
                        DailyPrediction(
                            agent_id=agent_id,
                            question_id=qid,
                            day=sim_date,
                            outcomes=outcomes,
                        )
                    )

        elif rtype == "resolution":
            qid = str(record.get("question_id")) if record.get("question_id") is not None else None
            scores = record.get("agent_scores", {})
            ground_truth = record.get("ground_truth", "")
            q = env.q_pool.get_question(qid) if qid else None
            question_title = q.title if q else None

            history = env.prediction_histories.get(qid) if qid else None
            final_snapshot = history.get_all_current_predictions() if history else {}
            if qid:
                env._store_resolved_final_predictions(qid, final_snapshot)

            raw_brier = record.get("raw_brier", {})
            snapshot_peer = record.get("snapshot_peer", {})

            per_agent_event = {}
            event_agent_ids = set(final_snapshot) | set(scores) | set(raw_brier) | set(snapshot_peer)
            for aid in event_agent_ids:
                score = float(scores.get(aid, 0.0))
                env.agent_scores[aid] = env.agent_scores.get(aid, 0.0) + score

                pred = final_snapshot.get(aid)
                best_outcome, best_prob = env._get_top_outcome(pred)
                is_accurate = False
                truth_prob = 0.0
                if pred and ground_truth:
                    env.agent_questions[aid] = env.agent_questions.get(aid, 0) + 1
                    truth_prob = env._get_truth_probability_mass(
                        pred,
                        ground_truth,
                        question_id=qid,
                        question_title=question_title,
                    )
                    env.agent_exp_acc_sum[aid] = env.agent_exp_acc_sum.get(aid, 0.0) + truth_prob
                    is_accurate = env._is_top_choice_correct(
                        pred,
                        ground_truth,
                        question_id=qid,
                        question_title=question_title,
                    )
                    if is_accurate:
                        env.agent_correct[aid] = env.agent_correct.get(aid, 0) + 1
                    else:
                        env.agent_wrong[aid] = env.agent_wrong.get(aid, 0) + 1
                else:
                    if score > 0:
                        env.agent_correct[aid] = env.agent_correct.get(aid, 0) + 1
                        is_accurate = True
                    elif score < 0:
                        env.agent_wrong[aid] = env.agent_wrong.get(aid, 0) + 1

                per_agent_event[aid] = {
                    "brier": raw_brier.get(aid),
                    "snapshot_peer": snapshot_peer.get(aid),
                    "tw_peer": score,
                    "best_outcome": best_outcome,
                    "best_prob": best_prob,
                    "truth_prob": truth_prob,
                    "is_accurate": bool(is_accurate),
                }

                if aid in raw_brier:
                    env.agent_raw_brier[aid] = env.agent_raw_brier.get(aid, 0.0) + raw_brier[aid]

                if aid in snapshot_peer:
                    env.agent_snapshot_peer[aid] = env.agent_snapshot_peer.get(aid, 0.0) + snapshot_peer[aid]

            if qid:
                env.q_pool._resolved.add(qid)

                if qid in env.prediction_histories:
                    del env.prediction_histories[qid]

                q = env.q_pool.get_question(qid)
                if q:
                    env.resolved_questions.append(q)
                    env.resolution_events.append(
                        {
                            "sim_date": sim_date_str,
                            "qid": qid,
                            "title": q.title,
                            "source_split": q.source_split,
                            "ground_truth": ground_truth,
                            "agents": per_agent_event,
                        }
                    )

    if records_processed > 0:
        env.current_date = last_date + timedelta(days=env.timegap_days)
        print(f"  Processed {records_processed} records.")
        print(f"  Fast-forwarded to {env.current_date}.")
        print(
            f"  State restored: {len(env.prediction_histories)} active questions, "
            f"{len(env.resolved_questions)} past resolutions."
        )
    else:
        print("  Warning: actions.jsonl was empty. Starting from beginning.")


__all__ = [
    "rescore",
    "restore_state",
]
