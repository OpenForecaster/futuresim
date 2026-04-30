from __future__ import annotations

from datetime import date
from typing import Dict, List


def print_run_start(current_date: date, end_date: date, *, is_resume: bool) -> None:
    print(f"Simulation: {current_date} to {end_date} (Resume: {is_resume})")


def print_step_banner(current_date: date, *, timegap_days: int, horizon: date) -> None:
    if timegap_days > 1:
        print(f"\n--- Wakeup {current_date} (covers through {horizon}) ---")
        return
    print(f"\n--- Day {current_date} ---")


def print_daily_scores(agent_scores: Dict[str, float]) -> None:
    if not agent_scores:
        return
    scores_str = ", ".join(f"{aid}: {score:+.2f}" for aid, score in sorted(agent_scores.items()))
    print(f"  Scores: {scores_str}")


def print_resolution(question, agent_scores: Dict[str, float]) -> None:
    print(f"  Resolved: {question.title[:40]}... -> '{question.ground_truth_answer[:20]}'")
    for agent_id, score in agent_scores.items():
        print(f"    {agent_id}: {score:+.2f}")


def print_final_summary(env, active_stats: Dict[str, Dict[str, float]]) -> None:
    print("\n" + "=" * 90)
    print("FINAL RESULTS")
    print("=" * 90)

    if not env.agents:
        print("  No agents participated.")
        return

    # Mean metrics are taken over ALL questions (resolved + currently active),
    # not just ones an agent predicted on. Unpredicted questions contribute 0.
    total_resolved_questions = len(env.resolution_events)
    total_active_questions = len(env.q_pool.get_active())
    total_question_count = total_resolved_questions + total_active_questions

    header = f"{'Agent':<25} {'Avg Brier':>10} {'Peer':>10} {'TW-Peer':>10} {'Acc %':>8} {'Pred':>6} {'Qs':>6}"
    print(header)
    print("-" * 90)

    for agent in sorted(env.agents, key=lambda a: a.agent_id):
        aid = agent.agent_id

        resolved_brier_sum = env.agent_raw_brier.get(aid, 0.0)
        resolved_snapshot_peer = env.agent_snapshot_peer.get(aid, 0.0)
        resolved_tw_peer = env.agent_scores.get(aid, 0.0)
        resolved_correct = env.agent_correct.get(aid, 0)
        resolved_count = env.agent_questions.get(aid, 0)

        agent_active = active_stats.get(
            aid,
            {
                "raw_brier": 0.0,
                "peer_sum": 0.0,
                "tw_peer_sum": 0.0,
                "correct_count": 0,
                "count": 0,
            },
        )

        total_brier_sum = resolved_brier_sum + agent_active["raw_brier"]
        total_peer_sum = resolved_snapshot_peer + agent_active["peer_sum"]
        total_tw_peer_sum = resolved_tw_peer + agent_active["tw_peer_sum"]
        total_correct = resolved_correct + agent_active["correct_count"]
        predicted_count = resolved_count + agent_active["count"]

        denom = total_question_count
        avg_brier = total_brier_sum / denom if denom > 0 else 0.0
        accuracy = (total_correct / denom * 100) if denom > 0 else 0.0
        display_name = aid[:24] if len(aid) > 24 else aid

        row = (
            f"{display_name:<25} {avg_brier:>+10.3f} {total_peer_sum:>+10.2f} "
            f"{total_tw_peer_sum:>+10.2f} {accuracy:>7.1f}% {predicted_count:>6} {denom:>6}"
        )
        print(row)

    print("-" * 90)
    print("Legend: Avg Brier / Acc are means over ALL questions (resolved + active);")
    print("        unpredicted questions contribute 0. Peer=Snapshot peer sum,")
    print("        TW-Peer=Time-weighted peer sum. Pred=# questions agent predicted,")
    print("        Qs=total question count (denominator).")
    print("=" * 90)


__all__: List[str] = [
    "print_daily_scores",
    "print_final_summary",
    "print_resolution",
    "print_run_start",
    "print_step_banner",
]
