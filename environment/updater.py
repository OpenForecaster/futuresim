from __future__ import annotations

import json
import os
from datetime import date
from threading import Lock
from typing import Any, Dict, List, Optional

import pandas as pd


def _daily_metrics_header(single_agent: bool) -> str:
    """
    Header for daily_metrics.csv. In single-agent mode, the time-weighted
    column is named ``tw_score`` (raw Brier-skill weighting against a 0
    baseline) rather than ``tw_peer_score`` (peer-relative), and the
    ``peer_score`` column is dropped entirely (it reduces to
    ``100 × Σ Brier_skill``, which is redundant with ``avg_brier``).
    """
    if single_agent:
        return (
            "date,agent_id,avg_brier,tw_score,accuracy,exp_acc,"
            "total_predictions,daily_submissions,avg_submission_tv_to_prev\n"
        )
    return (
        "date,agent_id,avg_brier,peer_score,tw_peer_score,accuracy,exp_acc,"
        "total_predictions,daily_submissions,avg_submission_tv_to_prev\n"
    )


DAILY_METRICS_HEADER = _daily_metrics_header(single_agent=False)


class SimLogger:
    """
    Thread-safe centralized logging for simulation events.

    Shared environment artifacts stay here. Agent transcript logging lives in
    the agent layer.
    """

    def __init__(self, output_dir: str = ".", append: bool = False):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        mode = "a" if append else "w"

        self._actions_lock = Lock()
        self.actions_file = open(os.path.join(output_dir, "actions.jsonl"), mode)

        self._metrics_lock = Lock()
        self.metrics_path = os.path.join(output_dir, "daily_metrics.csv")
        self.metrics_file = open(self.metrics_path, mode)
        self.test_metrics_path = os.path.join(output_dir, "test_daily_metrics.csv")
        self.test_metrics_file = open(self.test_metrics_path, mode)
        # Defer header write until the first log_daily_metrics call so we can
        # pick the column name based on single- vs multi-agent mode.
        self._metrics_header_written = bool(append)
        self._test_metrics_header_written = bool(append)

        self._matcher_lock = Lock()
        self.matcher_file = open(os.path.join(output_dir, "matcher.jsonl"), mode)

    def log_prediction(
        self,
        sim_date: date,
        agent_id: str,
        question_id: str,
        outcomes: Dict[str, float],
    ) -> None:
        record = {
            "sim_date": str(sim_date),
            "type": "prediction",
            "agent_id": agent_id,
            "question_id": question_id,
            "outcomes": outcomes,
        }
        with self._actions_lock:
            self.actions_file.write(json.dumps(record) + "\n")
            self.actions_file.flush()

    def log_aggregate(
        self,
        sim_date: date,
        question_id: str,
        aggregate: Dict[str, float],
    ) -> None:
        # Aggregates are intentionally omitted from actions.jsonl to reduce noise.
        return None

    def log_resolution(
        self,
        sim_date: date,
        question_id: str,
        ground_truth: str,
        agent_scores: Dict[str, float],
        raw_brier: Optional[Dict[str, float]] = None,
        snapshot_peer: Optional[Dict[str, float]] = None,
    ) -> None:
        record = {
            "sim_date": str(sim_date),
            "type": "resolution",
            "question_id": question_id,
            "ground_truth": ground_truth,
            "agent_scores": agent_scores,
        }
        if raw_brier:
            record["raw_brier"] = raw_brier
        if snapshot_peer:
            record["snapshot_peer"] = snapshot_peer
        with self._actions_lock:
            self.actions_file.write(json.dumps(record) + "\n")
            self.actions_file.flush()

    def _write_metrics_rows(
        self,
        file_obj,
        sim_date: date,
        metrics_list: List[Dict[str, Any]],
        single_agent: bool,
    ) -> None:
        for m in metrics_list:
            if single_agent:
                row = (
                    f"{sim_date},{m['agent_id']},{m['avg_brier']:.4f},"
                    f"{m['tw_peer_score']:.4f},{m['accuracy']:.2f},{m['exp_acc']:.4f},"
                    f"{m['total_predictions']},{m['daily_submissions']},"
                    f"{m['avg_submission_tv_to_prev']:.4f}\n"
                )
            else:
                row = (
                    f"{sim_date},{m['agent_id']},{m['avg_brier']:.4f},{m['peer_score']:.4f},"
                    f"{m['tw_peer_score']:.4f},{m['accuracy']:.2f},{m['exp_acc']:.4f},"
                    f"{m['total_predictions']},{m['daily_submissions']},"
                    f"{m['avg_submission_tv_to_prev']:.4f}\n"
                )
            file_obj.write(row)

    def log_daily_metrics(self, sim_date: date, metrics_list: List[Dict[str, Any]]) -> None:
        single_agent = len(metrics_list) == 1
        with self._metrics_lock:
            if not self._metrics_header_written:
                self.metrics_file.write(_daily_metrics_header(single_agent=single_agent))
                self._metrics_header_written = True
            self._write_metrics_rows(self.metrics_file, sim_date, metrics_list, single_agent)
            self.metrics_file.flush()

    def log_test_daily_metrics(self, sim_date: date, metrics_list: List[Dict[str, Any]]) -> None:
        single_agent = len(metrics_list) == 1
        with self._metrics_lock:
            if not self._test_metrics_header_written:
                self.test_metrics_file.write(_daily_metrics_header(single_agent=single_agent))
                self._test_metrics_header_written = True
            self._write_metrics_rows(self.test_metrics_file, sim_date, metrics_list, single_agent)
            self.test_metrics_file.flush()

    def reset_metrics_files(self) -> None:
        with self._metrics_lock:
            self.metrics_file.close()
            self.test_metrics_file.close()
            # Preserve the header style (single- vs multi-agent) on reset by
            # deferring again to the next write.
            open(self.metrics_path, "w").close()
            open(self.test_metrics_path, "w").close()
            self.metrics_file = open(self.metrics_path, "a")
            self.test_metrics_file = open(self.test_metrics_path, "a")
            self._metrics_header_written = False
            self._test_metrics_header_written = False

    def log_matcher(
        self,
        input_data: Any,
        output_data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "input": input_data,
            "output": output_data,
            "metadata": metadata or {},
        }
        with self._matcher_lock:
            self.matcher_file.write(json.dumps(record) + "\n")
            self.matcher_file.flush()

    def close(self) -> None:
        self.actions_file.close()
        self.metrics_file.close()
        self.test_metrics_file.close()
        self.matcher_file.close()


class MarketWriter:
    """
    Writes market state to market.csv for agent consumption.

    The CSV contains question data + aggregates but not agent-specific columns.
    Agents add my_prediction columns when loading via DfInterface.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.csv_path = os.path.join(output_dir, "market.csv")

    def write(
        self,
        questions,
        resolved_questions,
        aggregates,
        histories,
    ) -> str:
        rows = []

        for q in questions:
            agg = aggregates.get(q.qid, {})

            num_preds = 0
            history = histories.get(q.qid)
            if history:
                for agent_preds in history.predictions.values():
                    num_preds += len(agent_preds)

            rows.append(
                {
                    "qid": q.qid,
                    "title": q.title,
                    "background": q.background,
                    "resolution_criteria": q.resolution_criteria,
                    "answer_type": q.answer_type,
                    "resolution_date": str(q.resolution_date),
                    "is_resolved": False,
                    "ground_truth": None,
                    "market_aggregate": json.dumps(agg) if agg else None,
                    "num_predictions": num_preds,
                    "options": json.dumps(q.options) if q.options else None,
                }
            )

        for q in resolved_questions:
            rows.append(
                {
                    "qid": q.qid,
                    "title": q.title,
                    "background": q.background,
                    "resolution_criteria": q.resolution_criteria,
                    "answer_type": q.answer_type,
                    "resolution_date": str(q.resolution_date),
                    "is_resolved": True,
                    "ground_truth": q.ground_truth_answer,
                    "market_aggregate": None,
                    "num_predictions": 0,
                    "options": json.dumps(q.options) if q.options else None,
                }
            )

        # Agent processes may make market.csv read-only while active. Restore
        # owner write permissions before the environment refreshes the snapshot.
        if os.path.exists(self.csv_path):
            os.chmod(self.csv_path, 0o644)
        pd.DataFrame(rows).to_csv(self.csv_path, index=False)
        return self.csv_path


__all__ = [
    "DAILY_METRICS_HEADER",
    "MarketWriter",
    "SimLogger",
]
