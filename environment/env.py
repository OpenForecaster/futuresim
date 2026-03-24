import os
import json
import pandas as pd
from datetime import date, timedelta
from threading import Lock
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from .interfaces import PredictionSubmission
from .data_loader import QuestionPool, Question
from .scoring import (
    DailyPrediction, PredictionHistory, 
    compute_aggregate, resolve_question,
    DEFAULT_SCORER, BinaryBrierScorer
)
from .ansmatching import AnswerMatcher
from . import forecast_metrics as _fm


DAILY_METRICS_HEADER = (
    "date,agent_id,avg_brier,peer_score,tw_peer_score,accuracy,exp_acc,"
    "total_predictions,daily_submissions,avg_submission_tv_to_prev\n"
)


class SimLogger:
    """
    Thread-safe centralized logging for simulation events.
    
    Logs shared events (predictions, resolutions) to central files,
    and model outputs to per-agent directories.
    """
    
    def __init__(self, output_dir: str = ".", append: bool = False):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        mode = "a" if append else "w"
        
        # Shared logs (thread-safe with locks)
        self._actions_lock = Lock()
        self.actions_file = open(os.path.join(output_dir, "actions.jsonl"), mode)
        
        self._metrics_lock = Lock()
        self.metrics_path = os.path.join(output_dir, "daily_metrics.csv")
        self.metrics_file = open(self.metrics_path, mode)
        self.test_metrics_path = os.path.join(output_dir, "test_daily_metrics.csv")
        self.test_metrics_file = open(self.test_metrics_path, mode)
        if not append:
            self.metrics_file.write(DAILY_METRICS_HEADER)
            self.test_metrics_file.write(DAILY_METRICS_HEADER)
        
        self._matcher_lock = Lock()
        self.matcher_file = open(os.path.join(output_dir, "matcher.jsonl"), mode)
        
        # Per-agent output files (lazy initialized)
        self.agents_dir = os.path.join(output_dir, "agents")
        self.append = append
        self._agent_files: Dict[str, Any] = {}
        self._warmup_raw_buffers: Dict[str, List[Tuple[str, int, Dict[str, Any]]]] = {}
        self._warmup_raw_seq = 0
        self._agent_files_lock = Lock()
        
    def _get_agent_output_files(self, agent_id: str):
        """Get or create the output files for an agent (thread-safe)."""
        with self._agent_files_lock:
            if agent_id not in self._agent_files:
                agent_dir = os.path.join(self.agents_dir, agent_id)
                os.makedirs(agent_dir, exist_ok=True)
                out_path = os.path.join(agent_dir, "model_outputs.jsonl")
                raw_daily_path = os.path.join(agent_dir, "model_raw_daily.jsonl")
                raw_warmup_path = os.path.join(agent_dir, "model_raw_warmup.jsonl")
                mode = "a" if self.append else "w"
                self._agent_files[agent_id] = (
                    open(out_path, mode),
                    open(raw_daily_path, mode),
                    open(raw_warmup_path, mode),
                )
            return self._agent_files[agent_id]

    @staticmethod
    def _render_prompt_text(prompt: Any) -> str:
        if prompt is None:
            return ""
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, list):
            parts = [SimLogger._render_prompt_text(item) for item in prompt]
            return "\n\n".join(part for part in parts if part)
        if isinstance(prompt, dict):
            if isinstance(prompt.get("content"), str) and prompt.get("content").strip():
                return prompt["content"]
            if isinstance(prompt.get("output"), str) and prompt.get("output").strip():
                return prompt["output"]
        return json.dumps(prompt, ensure_ascii=False)
        
    def log_prediction(self, sim_date: date, agent_id: str, 
                       question_id: str, outcomes: Dict[str, float]):
        record = {
            "sim_date": str(sim_date),
            "type": "prediction",
            "agent_id": agent_id,
            "question_id": question_id,
            "outcomes": outcomes
        }
        with self._actions_lock:
            self.actions_file.write(json.dumps(record) + "\n")
            self.actions_file.flush()
        
    def log_aggregate(self, sim_date: date, question_id: str, 
                      aggregate: Dict[str, float]):
        # We no longer log aggregates to actions.jsonl to reduce noise
        pass
        
    def log_resolution(self, sim_date: date, question_id: str, 
                       ground_truth: str, agent_scores: Dict[str, float],
                       raw_brier: Dict[str, float] = None,
                       snapshot_peer: Dict[str, float] = None):
        record = {
            "sim_date": str(sim_date),
            "type": "resolution",
            "question_id": question_id,
            "ground_truth": ground_truth,
            "agent_scores": agent_scores
        }
        if raw_brier:
            record["raw_brier"] = raw_brier
        if snapshot_peer:
            record["snapshot_peer"] = snapshot_peer
        with self._actions_lock:
            self.actions_file.write(json.dumps(record) + "\n")
            self.actions_file.flush()

    def _write_metrics_rows(self, file_obj, sim_date: date, metrics_list: List[Dict[str, Any]]) -> None:
        for m in metrics_list:
            row = (
                f"{sim_date},{m['agent_id']},{m['avg_brier']:.4f},{m['peer_score']:.4f},"
                f"{m['tw_peer_score']:.4f},{m['accuracy']:.2f},{m['exp_acc']:.4f},"
                f"{m['total_predictions']},{m['daily_submissions']},"
                f"{m['avg_submission_tv_to_prev']:.4f}\n"
            )
            file_obj.write(row)

    def log_daily_metrics(self, sim_date: date, metrics_list: List[Dict[str, Any]]):
        with self._metrics_lock:
            self._write_metrics_rows(self.metrics_file, sim_date, metrics_list)
            self.metrics_file.flush()

    def log_test_daily_metrics(self, sim_date: date, metrics_list: List[Dict[str, Any]]):
        with self._metrics_lock:
            self._write_metrics_rows(self.test_metrics_file, sim_date, metrics_list)
            self.test_metrics_file.flush()
        
    def log_model_output(self, sim_date: date, agent_id: str, prompt: Any,
                         response: str, metadata: Optional[Dict[str, Any]] = None):
        """Log model output to per-agent files (thread-safe)."""
        metadata = metadata or {}
        raw_stream = str(metadata.get("raw_stream", "daily") or "daily").strip().lower()
        if raw_stream not in {"daily", "warmup"}:
            raw_stream = "daily"
        
        # Extract qid from metadata if present
        qid = metadata.get("qid")
        prompt_text = self._render_prompt_text(prompt)
        
        # Raw record: incremental input delta + raw response.
        raw_record = {
            "sim_date": str(sim_date),
            "agent_id": agent_id,
            "qid": qid,
            "prompt": prompt_text,
            "input_delta": prompt,
            "response": response,
            "metadata": metadata
        }
        
        # Clean record: only response, no prompt.
        # Keep reasoning visible to make debugging/parsing behavior easier.
        clean_metadata = metadata.copy()
        clean_response = response
        reasoning = clean_metadata.get("reasoning")
        if reasoning and "<reasoning>" not in (clean_response or ""):
            clean_response = f"<reasoning>{reasoning}</reasoning>\n{clean_response}"

        clean_record = {
            "sim_date": str(sim_date),
            "agent_id": agent_id,
            "qid": qid,
            "response": clean_response,
            "metadata": clean_metadata
        }
        
        out_file, raw_daily_file, raw_warmup_file = self._get_agent_output_files(agent_id)
        # Each agent file is only written by one thread, so no lock needed per-file
        out_file.write(json.dumps(clean_record) + "\n")
        out_file.flush()

        if raw_stream == "warmup":
            with self._agent_files_lock:
                self._warmup_raw_seq += 1
                qid_key = "" if qid is None else str(qid)
                self._warmup_raw_buffers.setdefault(agent_id, []).append((qid_key, self._warmup_raw_seq, raw_record))
        else:
            raw_daily_file.write(json.dumps(raw_record) + "\n")
            raw_daily_file.flush()

    def flush_warmup_raw(self, agent_id: str) -> None:
        with self._agent_files_lock:
            buffered = self._warmup_raw_buffers.pop(agent_id, [])
            if not buffered or agent_id not in self._agent_files:
                return
            _, _, raw_warmup_file = self._agent_files[agent_id]
            for _, _, raw_record in sorted(buffered, key=lambda item: (item[0], item[1])):
                raw_warmup_file.write(json.dumps(raw_record) + "\n")
            raw_warmup_file.flush()
        
    def log_matcher(self, input_data: Any, output_data: Any, metadata: Optional[Dict] = None):
        """Log matcher decisions."""
        record = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "input": input_data,
            "output": output_data,
            "metadata": metadata or {}
        }
        with self._matcher_lock:
            self.matcher_file.write(json.dumps(record) + "\n")
            self.matcher_file.flush()
        
    def close(self):
        self.actions_file.close()
        self.metrics_file.close()
        self.test_metrics_file.close()
        self.matcher_file.close()
        with self._agent_files_lock:
            for agent_id in list(self._warmup_raw_buffers.keys()):
                buffered = self._warmup_raw_buffers.pop(agent_id, [])
                if agent_id in self._agent_files:
                    _, _, raw_warmup_file = self._agent_files[agent_id]
                    for _, _, raw_record in sorted(buffered, key=lambda item: (item[0], item[1])):
                        raw_warmup_file.write(json.dumps(raw_record) + "\n")
                    raw_warmup_file.flush()
            for f_out, f_raw_daily, f_raw_warmup in self._agent_files.values():
                f_out.close()
                f_raw_daily.close()
                f_raw_warmup.close()
            self._agent_files.clear()


class MarketWriter:
    """
    Writes market state to market.csv for agent consumption.
    
    The CSV contains question data + aggregates but NOT agent-specific columns.
    Agents add my_prediction columns when loading via DfInterface.
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.csv_path = os.path.join(output_dir, "market.csv")
    
    def write(self, 
              questions: List['Question'],
              resolved_questions: List['Question'],
              aggregates: Dict[str, Dict[str, float]],
              histories: Dict[str, 'PredictionHistory']) -> str:
        """
        Write current market state to CSV.
        
        Returns the path to the CSV file.
        """
        rows = []
        
        # Active questions
        for q in questions:
            agg = aggregates.get(q.qid, {})
            
            # Count total predictions
            num_preds = 0
            history = histories.get(q.qid)
            if history:
                for agent_preds in history.predictions.values():
                    num_preds += len(agent_preds)
            
            rows.append({
                'qid': q.qid,
                'title': q.title,
                'background': q.background,
                'resolution_criteria': q.resolution_criteria,
                'answer_type': q.answer_type,
                'resolution_date': str(q.resolution_date),
                'is_resolved': False,
                'ground_truth': None,
                'market_aggregate': json.dumps(agg) if agg else None,
                'num_predictions': num_preds,
                'options': json.dumps(q.options) if q.options else None,
            })
        
        # Resolved questions
        for q in resolved_questions:
            rows.append({
                'qid': q.qid,
                'title': q.title,
                'background': q.background,
                'resolution_criteria': q.resolution_criteria,
                'answer_type': q.answer_type,
                'resolution_date': str(q.resolution_date),
                'is_resolved': True,
                'ground_truth': q.ground_truth_answer,
                'market_aggregate': None,
                'num_predictions': 0,
                'options': json.dumps(q.options) if q.options else None,
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(self.csv_path, index=False)
        return self.csv_path


class SimulationEnvironment:
    """
    Main simulation environment for multi-agent forecasting.
    
    Uses Metaculus-style scoring:
    - Log score for accuracy
    - Peer score relative to other agents
    - Time-weighted averaging across prediction history
    
    Supports parallel agent execution for multi-agent simulations.
    """
    
    def __init__(self, 
                 dataset: str = "openforesight",
                 dataset_path: str = None,
                 dataset_cache: str = None,
                 start_date: date = None, # Make start_date optional or handle in signature carefully
                 end_date: date = None,
                 dataset_name: str = None, # Backward compatibility/argparse noise
                 inference_provider=None,
                 output_dir: str = ".",
                 resolution_start: date = None,
                 resolution_end: date = None,
                 parallel: bool = True,
                 split: str = "train",
                 prepend_train_resolution_start: Optional[date] = None,
                 prepend_train_resolution_end: Optional[date] = None,
                 subsample_per_month: Optional[int] = None,
                 timegap_days: int = 1,
                 resume_dir: str = None,
                 min_forecasters: int = 0,
                 resolved_only: bool = False,
                 cheat_feedback: bool = False):
        
        # Handle positional args shift if someone called with positional args (unlikely in this codebase but safe)
        # We'll rely on kwargs mostly.
        
        if dataset_name and dataset == "openforesight":
             dataset = dataset_name # Support legacy arg if passed

        if start_date is None:
             raise ValueError("start_date required")

        self.start_date = start_date
        self.current_date = start_date
        self.end_date = end_date
        self.output_dir = output_dir
        self.parallel = parallel
        self.split = split
        self.prepend_train_resolution_start = prepend_train_resolution_start
        self.prepend_train_resolution_end = prepend_train_resolution_end
        self.subsample_per_month = subsample_per_month
        self.timegap_days = max(1, int(timegap_days or 1))
        self.cheat_feedback = cheat_feedback
        
        # Logging and market state
        self.resume_dir = resume_dir
        if resume_dir:
            self.output_dir = resume_dir
            append_logs = True
        else:
            self.output_dir = output_dir
            append_logs = False
            
        self.logger = SimLogger(self.output_dir, append=append_logs)
        self.market_writer = MarketWriter(self.output_dir)
        matcher_cache_path = os.path.join(self.output_dir, "matcher_cache.json")
        
        # Components - filter questions to resolution window
        self.q_pool = QuestionPool(
            dataset=dataset,
            dataset_path=dataset_path,
            dataset_cache=dataset_cache,
            split=split,
            prepend_train_resolution_start=self.prepend_train_resolution_start,
            prepend_train_resolution_end=self.prepend_train_resolution_end,
            subsample_per_month=self.subsample_per_month,
            resolution_start=resolution_start,
            resolution_end=resolution_end,
            min_forecasters=min_forecasters,
            resolved_only=resolved_only
        )
        self.matcher = AnswerMatcher(inference_provider, logger=self.logger, cache_path=matcher_cache_path) if inference_provider else None
        
        # Store source context for prompts
        self.source_context = self.q_pool.fetcher.get_prompt_context()
        self.source_name = self.q_pool.fetcher.source_name
        
        source_key = str(self.source_name or "").lower()
        if source_key == "metaculus_binary":
            # Keep scoring consistent with binary Brier prompt wording.
            self.scorer = BinaryBrierScorer()
        else:
            self.scorer = DEFAULT_SCORER
        
        # Track prediction history per question
        self.prediction_histories: Dict[str, PredictionHistory] = {}
        self._histories_lock = Lock()  # For thread-safe history updates
        
        # Current aggregate per question (updated end of each day)
        self.current_aggregates: Dict[str, Dict[str, float]] = {}
        
        self.agents = []
        self.agent_scores: Dict[str, float] = {}  # Cumulative time-weighted peer scores
        
        # Track prediction outcomes for summary
        self.agent_correct: Dict[str, int] = {}  # Count of top-choice matches
        self.agent_wrong: Dict[str, int] = {}    # Count of top-choice misses
        self.agent_questions: Dict[str, int] = {}  # Total questions predicted on
        
        # Additional metrics for final summary
        self.agent_raw_brier: Dict[str, float] = {}  # Sum of raw Brier scores (last pred only)
        self.agent_snapshot_peer: Dict[str, float] = {}  # Sum of peer scores at resolution (not time-weighted)
        self.agent_exp_acc_sum: Dict[str, float] = {}  # Sum of P(true outcome) across resolved questions
        
        # Track resolved questions for agent learning
        self.resolved_questions: List[Question] = []
        # Authoritative per-resolution summaries for agent-facing feedback prompts.
        self.resolution_events: List[Dict[str, Any]] = []
        # Final per-agent predictions for resolved questions.
        # Shape: {qid: {agent_id: {"outcomes": {...}, "date": date}}}
        self.resolved_agent_predictions: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
        self.market_csv_path: Optional[str] = None
        
        if self.resume_dir:
            self._restore_state(self.resume_dir)
        
    def add_agent(self, agent):
        self.agents.append(agent)
        # Preserve restored resume stats if they were reconstructed before agents were added.
        self.agent_scores.setdefault(agent.agent_id, 0.0)
        self.agent_correct.setdefault(agent.agent_id, 0)
        self.agent_wrong.setdefault(agent.agent_id, 0)
        self.agent_questions.setdefault(agent.agent_id, 0)
        self.agent_raw_brier.setdefault(agent.agent_id, 0.0)
        self.agent_snapshot_peer.setdefault(agent.agent_id, 0.0)
        self.agent_exp_acc_sum.setdefault(agent.agent_id, 0.0)
        
    def run(self):
        print(f"Simulation: {self.current_date} to {self.end_date} (Resume: {bool(self.resume_dir)})")
        
        while self.current_date <= self.end_date:
            if self.timegap_days > 1:
                horizon = self._get_metrics_evaluation_date(self.current_date)
                print(f"\n--- Wakeup {self.current_date} (covers through {horizon}) ---")
            else:
                print(f"\n--- Day {self.current_date} ---")
            self.step()
            self._print_daily_scores()
            self.current_date += timedelta(days=self.timegap_days)
        
        self.logger.close()
        self._print_final_summary()
    
    def _print_daily_scores(self):
        """Print current cumulative scores for all agents."""
        if self.agent_scores:
            scores_str = ", ".join(f"{aid}: {sc:+.2f}" for aid, sc in sorted(self.agent_scores.items()))
            print(f"  Scores: {scores_str}")

    def _get_metrics_evaluation_date(self, sim_date: Optional[date] = None) -> date:
        """Return the end-of-interval date used for metrics at a wakeup."""
        sim_date = sim_date or self.current_date
        if self.end_date is None:
            return sim_date
        return min(sim_date + timedelta(days=self.timegap_days - 1), self.end_date)

    def _get_last_active_date(self, sim_date: Optional[date] = None) -> Optional[date]:
        sim_date = sim_date or self.current_date
        if sim_date <= self.start_date and not self.resume_dir:
            return None
        return sim_date - timedelta(days=self.timegap_days)

    def _get_next_active_date(self, sim_date: Optional[date] = None) -> Optional[date]:
        sim_date = sim_date or self.current_date
        next_active = sim_date + timedelta(days=self.timegap_days)
        if self.end_date is not None and next_active > self.end_date:
            return None
        return next_active
    
    def _print_final_summary(self):
        """Print formatted summary table with all scoring metrics."""
        print("\n" + "="*90)
        print("FINAL RESULTS")
        print("="*90)
        
        if not self.agents:
            print("  No agents participated.")
            return
        
        # Compute scores for ACTIVE questions (using hidden truth)
        active_questions = self.q_pool.get_active()
        # After run(), current_date is end_date + timegap; active-score gating must use the last
        # simulated day, not the post-loop cursor, or FINAL RESULTS active columns go to zero.
        last_day = self.end_date if self.end_date is not None else (
            self.current_date - timedelta(days=self.timegap_days)
        )
        active_stats = self._compute_daily_active_scores(
            active_questions,
            evaluation_date=last_day,
            metrics_context_date=last_day,
        )
        
        # Table header
        header = f"{'Agent':<25} {'Avg Brier':>10} {'Peer':>10} {'TW-Peer':>10} {'Acc %':>8} {'Total':>6}"
        print(header)
        print("-"*90)
        
        # Table rows
        for agent in sorted(self.agents, key=lambda a: a.agent_id):
            aid = agent.agent_id
            
            # Resolved stats
            resolved_brier_sum = self.agent_raw_brier.get(aid, 0.0)
            resolved_snapshot_peer = self.agent_snapshot_peer.get(aid, 0.0)
            resolved_tw_peer = self.agent_scores.get(aid, 0.0)
            resolved_correct = self.agent_correct.get(aid, 0)
            resolved_count = self.agent_questions.get(aid, 0)
            
            # Active stats
            agent_active = active_stats.get(aid, {
                'raw_brier': 0.0, 
                'peer_sum': 0.0, 
                'tw_peer_sum': 0.0, 
                'correct_count': 0,
                'count': 0
            })
            
            # Combined stats for Avg Brier
            total_brier_sum = resolved_brier_sum + agent_active['raw_brier']
            total_peer_sum = resolved_snapshot_peer + agent_active['peer_sum']
            total_tw_peer_sum = resolved_tw_peer + agent_active['tw_peer_sum']
            total_correct = resolved_correct + agent_active['correct_count']
            total_count = resolved_count + agent_active['count']
            
            avg_brier = total_brier_sum / total_count if total_count > 0 else 0.0
            accuracy = (total_correct / total_count * 100) if total_count > 0 else 0.0
            
            # Truncate long agent names
            display_name = aid[:24] if len(aid) > 24 else aid
            
            row = f"{display_name:<25} {avg_brier:>+10.3f} {total_peer_sum:>+10.2f} {total_tw_peer_sum:>+10.2f} {accuracy:>7.1f}% {total_count:>6}"
            print(row)
        
        print("-"*90)
        print("Legend: Avg Brier=Mean Brier score per question (Active + Resolved)")
        print("        Peer=Snapshot peer sum, TW-Peer=Time-weighted peer sum")
        print("        Acc=% of questions where agent's top choice matched truth.")
        print("="*90)

    def rescore(self):
        """
        Re-evaluate all simulation days and re-write daily_metrics.csv.
        Rebuilds prediction histories from actions.jsonl and replays all resolutions.
        """
        print(f"\nRescoring simulation history from {self.output_dir}...")
        
        actions_path = os.path.join(self.output_dir, "actions.jsonl")
        if not os.path.exists(actions_path):
            print(f"  Error: {actions_path} not found")
            return
        
        # 1. Reset ALL state
        self.agent_scores = {agent.agent_id: 0.0 for agent in self.agents}
        self.agent_correct = {agent.agent_id: 0 for agent in self.agents}
        self.agent_wrong = {agent.agent_id: 0 for agent in self.agents}
        self.agent_questions = {agent.agent_id: 0 for agent in self.agents}
        self.agent_raw_brier = {agent.agent_id: 0.0 for agent in self.agents}
        self.agent_snapshot_peer = {agent.agent_id: 0.0 for agent in self.agents}
        self.agent_exp_acc_sum = {agent.agent_id: 0.0 for agent in self.agents}
        self.resolved_questions = []
        self.resolution_events = []
        self.resolved_agent_predictions = {}
        self.prediction_histories = {}
        self.q_pool.reset()
        
        # 2. Re-truncate metrics file
        metrics_path = os.path.join(self.output_dir, "daily_metrics.csv")
        test_metrics_path = os.path.join(self.output_dir, "test_daily_metrics.csv")
        self.logger.metrics_file.close()
        self.logger.test_metrics_file.close()
        with open(metrics_path, 'w') as f:
            f.write(DAILY_METRICS_HEADER)
        with open(test_metrics_path, 'w') as f:
            f.write(DAILY_METRICS_HEADER)
        self.logger.metrics_file = open(metrics_path, 'a')
        self.logger.test_metrics_file = open(test_metrics_path, 'a')
        
        # 3. Rebuild prediction histories from actions.jsonl (don't process resolutions yet)
        predictions_by_date = {}  # date -> list of prediction records
        resolutions_by_date = {}  # date -> list of (qid, ground_truth) tuples
        all_dates = set()
        
        print("  Rebuilding prediction histories...")
        with open(actions_path, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line)
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
                        
                        if qid and agent_id and outcomes:
                            # Ensure history exists
                            if qid not in self.prediction_histories:
                                q = self.q_pool.get_question(qid)
                                if q:
                                    self.prediction_histories[qid] = PredictionHistory(
                                        question_id=qid,
                                        start_date=sim_date,
                                        resolution_date=q.resolution_date
                                    )
                            
                            # Add prediction
                            if qid in self.prediction_histories:
                                pred = DailyPrediction(
                                    agent_id=agent_id,
                                    question_id=qid,
                                    day=sim_date,
                                    outcomes=outcomes
                                )
                                self.prediction_histories[qid].add_prediction(pred)
                                
                    elif rtype == "resolution":
                        qid = str(record.get("question_id")) if record.get("question_id") is not None else None
                        if qid:
                            if sim_date not in resolutions_by_date:
                                resolutions_by_date[sim_date] = []
                            resolutions_by_date[sim_date].append(qid)
                            
                except json.JSONDecodeError:
                    continue
        
        if not all_dates:
            print("  No history found to rescore.")
            return
        
        start_date = min(all_dates)
        end_date = max(all_dates)
        
        print(f"  Found {len(self.prediction_histories)} questions with predictions")
        print(f"  Window: {start_date} to {end_date}")
        
        # 4. Replay each day: resolve questions and save metrics
        iter_date = start_date
        while iter_date <= end_date:
            self.current_date = iter_date
            
            # Resolve questions that resolved on this date
            if iter_date in resolutions_by_date:
                for qid in resolutions_by_date[iter_date]:
                    q = self.q_pool.get_question(qid)
                    if q and qid in self.prediction_histories:
                        self._resolve_question(q)
                        self.resolved_questions.append(q)
                        self.q_pool._resolved.add(qid)
            
            # Update aggregates for still-active questions
            active_questions = [
                self.q_pool.get_question(qid) 
                for qid in self.prediction_histories.keys()
                if self.q_pool.get_question(qid)
            ]
            self._update_aggregates(active_questions)

            if self.matcher and active_questions:
                self._warmup_matcher_cache(active_questions)

            # Save daily metrics
            self._save_daily_metrics()
            
            iter_date += timedelta(days=self.timegap_days)
        
        self.logger.metrics_file.flush()
        print(f"  Rescoring complete. Processed {len(self.resolved_questions)} resolutions.")

    def step(self):
        matcher_before = self._get_env_matcher_timing_snapshot()

        # 1. Resolve questions expiring today
        resolving = self.q_pool.pop_resolving(self.current_date)

        if resolving and self.matcher:
            self._warmup_matcher_cache(resolving)

        for q in resolving:
            self._resolve_question(q)
            self.resolved_questions.append(q)  # Track for agent learning
            
        # 2. Get active questions
        active_questions = self.q_pool.get_active()
        
        # Initialize prediction histories for new questions
        for q in active_questions:
            if q.qid not in self.prediction_histories:
                self.prediction_histories[q.qid] = PredictionHistory(
                    question_id=q.qid,
                    start_date=self.current_date,
                    resolution_date=q.resolution_date
                )
        
        # 3. Write market.csv for agents to read
        self.market_csv_path = self.market_writer.write(
            active_questions,
            self.resolved_questions,
            self.current_aggregates,
            self.prediction_histories
        )
        
        # 4. Collect predictions from all agents
        if self.parallel and len(self.agents) > 1:
            self._run_agents_parallel(active_questions)
        else:
            self._run_agents_sequential(active_questions)
        
        # 5. Update aggregates (end of day)
        self._update_aggregates(active_questions)
        
        # 6. Pre-populate matcher cache for active-question scoring
        if self.matcher and active_questions:
            self._warmup_matcher_cache(active_questions)

        # 7. Save daily metrics
        self._save_daily_metrics()
        matcher_after = self._get_env_matcher_timing_snapshot()
        day_matcher = self._compute_env_matcher_delta(matcher_before, matcher_after)
        self._inject_env_matcher_timing_into_agent_logs(self.current_date, day_matcher)
    
    def _normalize_outcome(self, s: str) -> str:
        """Normalize outcome strings for deterministic exact matching."""
        return _fm.normalize_outcome(s)

    @staticmethod
    def _get_top_outcome(pred: Optional[DailyPrediction]) -> Tuple[Optional[str], float]:
        """Return (best_outcome, best_prob) for a prediction snapshot."""
        if not pred or not pred.outcomes:
            return None, 0.0
        outcome, prob = max(pred.outcomes.items(), key=lambda kv: kv[1])
        return outcome, float(prob)

    def _store_resolved_final_predictions(
        self,
        qid: str,
        final_snapshot: Dict[str, DailyPrediction],
    ) -> None:
        """Persist each agent's final snapshot for a resolved question."""
        per_agent: Dict[str, Dict[str, Any]] = {}
        for agent_id, pred in (final_snapshot or {}).items():
            if not pred:
                continue
            per_agent[str(agent_id)] = {
                "outcomes": dict(pred.outcomes or {}),
                "date": pred.day,
            }
        if per_agent:
            self.resolved_agent_predictions[str(qid)] = per_agent

    def _is_top_choice_correct(
        self,
        pred: DailyPrediction,
        ground_truth: str,
        question_id: Optional[str] = None,
        question_title: Optional[str] = None
    ) -> bool:
        """Return True when the highest-probability predicted outcome matches truth."""
        return _fm.is_top_choice_correct(
            pred,
            ground_truth,
            self.matcher,
            question_id=question_id,
            question_title=question_title,
        )

    def _get_truth_probability_mass(
        self,
        pred: DailyPrediction,
        ground_truth: str,
        question_id: Optional[str] = None,
        question_title: Optional[str] = None
    ) -> float:
        """
        Return total probability assigned to outcomes that match the ground truth.
        Returns 0.0 when no matching outcome is present.
        """
        return _fm.truth_probability_mass(
            pred,
            ground_truth,
            self.matcher,
            question_id=question_id,
            question_title=question_title,
        )

    def _compute_daily_active_scores(
        self,
        active_questions: List[Question],
        evaluation_date: Optional[date] = None,
        metrics_context_date: Optional[date] = None,
    ):
        """
        Compute interval-end scores only for the currently active questions.
        Uses FUTURE ground truth to assess current performance.
        Includes Brier score, Peer score, TW-Peer score, Accuracy, and Exp-Accuracy.

        metrics_context_date: simulation "as of" day for gating (defaults to self.current_date).
        Pass the last simulated day when calling after run() has advanced current_date past end_date.
        """
        active_stats = {}
        interval_end = evaluation_date or self.current_date
        context_date = (
            metrics_context_date if metrics_context_date is not None else self.current_date
        )

        from environment.scoring import resolve_question, compute_snapshot_peer_scores
            
        for q in active_questions:
            # We strictly need ground truth here
            if not q.ground_truth_answer:
                continue
                
            history = self.prediction_histories.get(q.qid)
            if not history or not history.predictions:
                continue

            effective_eval_date = min(
                interval_end,
                q.resolution_date - timedelta(days=1),
            )
            if effective_eval_date < context_date:
                continue
                
            # Get the snapshot that remains active through the evaluation horizon.
            snapshot = {}
            for aid in history.predictions.keys():
                pred = history.get_prediction_as_of(aid, effective_eval_date)
                if pred:
                    snapshot[aid] = pred
            
            # 1. Compute Active Snapshot Peer Scores
            snapshot_peers = {}
            if snapshot:
                snapshot_peers = compute_snapshot_peer_scores(
                    snapshot, q.ground_truth_answer, self.scorer, self.matcher,
                    question_id=q.qid, question_title=q.title
                )

            # 2. Compute Active TW-Peer Scores (Partial Resolution)
            res = resolve_question(
                history,
                q.ground_truth_answer,
                matcher=self.matcher,
                scorer=self.scorer,
                evaluation_date=effective_eval_date,
                question_title=q.title
            )
            tw_peers = res.agent_scores

            # Process per-agent stats for this question
            for agent in self.agents:
                aid = agent.agent_id
                
                # Check if agent has a prediction in snapshot
                pred = snapshot.get(aid)
                
                if pred:
                    # Brier Score
                    brier = self.scorer.score_prediction(pred, q.ground_truth_answer, self.matcher,
                                                          question_id=q.qid, question_title=q.title)
                    
                    # Accuracy: Highest probability guess matches truth (ties broken by insertion order)
                    is_accurate = self._is_top_choice_correct(
                        pred,
                        q.ground_truth_answer,
                        question_id=q.qid,
                        question_title=q.title
                    )
                    truth_prob = self._get_truth_probability_mass(
                        pred,
                        q.ground_truth_answer,
                        question_id=q.qid,
                        question_title=q.title
                    )
                    
                    # Aggregate stats
                    if aid not in active_stats:
                        active_stats[aid] = {
                            'raw_brier': 0.0, 
                            'peer_sum': 0.0,
                            'tw_peer_sum': 0.0,
                            'correct_count': 0,
                            'truth_prob_sum': 0.0,
                            'count': 0
                        }
                    
                    active_stats[aid]['raw_brier'] += brier
                    active_stats[aid]['peer_sum'] += snapshot_peers.get(aid, 0.0)
                    active_stats[aid]['tw_peer_sum'] += tw_peers.get(aid, 0.0)
                    active_stats[aid]['correct_count'] += 1 if is_accurate else 0
                    active_stats[aid]['truth_prob_sum'] += truth_prob
                    active_stats[aid]['count'] += 1
                    
        return active_stats

    def _warmup_matcher_cache(self, questions) -> None:
        """Pre-populate matcher cache for a list of questions in parallel.

        Collects every (outcome, ground_truth) pair that will be checked during
        resolution / scoring and fires them concurrently via ``matcher.warmup_cache``.
        Subsequent individual ``is_equivalent`` calls become instant cache hits.
        """
        if not hasattr(self.matcher, 'warmup_cache'):
            return

        items = []
        for q in questions:
            gt = q.ground_truth_answer
            if not gt:
                continue
            history = self.prediction_histories.get(q.qid)
            if not history or not history.predictions:
                continue
            seen_outcomes = set()
            for pred_list in history.predictions.values():
                for pred in pred_list:
                    if pred and pred.outcomes:
                        for outcome in pred.outcomes:
                            if outcome not in seen_outcomes:
                                seen_outcomes.add(outcome)
                                items.append((outcome, gt, q.qid, q.title))

        if items:
            self.matcher.warmup_cache(items, max_concurrency=300)

    def _get_env_matcher_timing_snapshot(self) -> Dict[str, float]:
        """Get cumulative matcher timing counters from the environment scorer."""
        if not self.matcher:
            return {"matcher_count": 0, "matcher_total_seconds": 0.0, "matcher_total_cost": 0.0}

        if hasattr(self.matcher, "get_timing_snapshot"):
            snapshot = self.matcher.get_timing_snapshot() or {}
            return {
                "matcher_count": int(snapshot.get("matcher_count", 0)),
                "matcher_total_seconds": float(snapshot.get("matcher_total_seconds", 0.0)),
                "matcher_total_cost": float(snapshot.get("matcher_total_cost", 0.0)),
            }

        # Backward-compatible fallback if matcher doesn't expose raw snapshot.
        stats = self.matcher.get_stats() if hasattr(self.matcher, "get_stats") else {}
        return {
            "matcher_count": int(stats.get("matcher_count", 0)),
            "matcher_total_seconds": float(stats.get("matcher_total_seconds", 0.0)),
            "matcher_total_cost": float(stats.get("matcher_total_cost", 0.0)),
        }

    @staticmethod
    def _compute_env_matcher_delta(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
        """Compute non-negative per-day matcher timing delta from cumulative snapshots."""
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

    def _inject_env_matcher_timing_into_agent_logs(self, sim_date: date, day_matcher: Dict[str, float]) -> None:
        """
        Overwrite per-agent matcher timing fields with env-scoring matcher timings for this day.
        """
        if not self.agents:
            return

        target_date = str(sim_date)
        for agent in self.agents:
            stats_path = os.path.join(self.output_dir, "agents", agent.agent_id, "timing_stats.jsonl")
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

                # Keep output minimal and canonical: matcher_* only.
                target_row.pop("feedback_matcher_count", None)
                target_row.pop("feedback_matcher_total_seconds", None)
                target_row.pop("feedback_matcher_avg_seconds", None)
                target_row.pop("env_matcher_count", None)
                target_row.pop("env_matcher_total_seconds", None)
                target_row.pop("env_matcher_avg_seconds", None)

                target_row["matcher_count"] = int(day_matcher.get("matcher_count", 0))
                target_row["matcher_total_seconds"] = float(day_matcher.get("matcher_total_seconds", 0.0))
                target_row["matcher_avg_seconds"] = float(day_matcher.get("matcher_avg_seconds", 0.0))
                target_row["matcher_cost"] = float(day_matcher.get("matcher_cost", 0.0))

                lines[target_idx] = json.dumps(target_row) + "\n"
                with open(stats_path, "w") as f:
                    f.writelines(lines)

            except Exception as e:
                print(f"  Warning: failed to inject env matcher timing for {agent.agent_id}: {e}")

    def _compute_resolved_metrics_from_events(
        self,
        *,
        source_split: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {}
        for event in self.resolution_events:
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

    @staticmethod
    def _tv_distance(left: Optional[Dict[str, float]], right: Optional[Dict[str, float]]) -> float:
        outcomes = set((left or {}).keys()) | set((right or {}).keys())
        if not outcomes:
            return 0.0
        return 0.5 * sum(abs(float((left or {}).get(outcome, 0.0)) - float((right or {}).get(outcome, 0.0))) for outcome in outcomes)

    def _build_metrics_list(
        self,
        *,
        active_questions: List[Question],
        source_split: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        filtered_active_questions = [
            q for q in active_questions
            if not source_split or q.source_split == source_split
        ]
        active_stats = self._compute_daily_active_scores(
            filtered_active_questions,
            evaluation_date=self._get_metrics_evaluation_date(),
        )
        resolved_stats = self._compute_resolved_metrics_from_events(source_split=source_split)
        daily_submission_stats: Dict[str, Dict[str, float]] = {}
        for q in filtered_active_questions:
            history = self.prediction_histories.get(q.qid)
            if not history:
                continue
            for aid, preds in history.predictions.items():
                for idx, pred in enumerate(preds):
                    if pred.day != self.current_date:
                        continue
                    entry = daily_submission_stats.setdefault(
                        aid,
                        {"count": 0.0, "tv_sum": 0.0, "tv_count": 0.0},
                    )
                    entry["count"] += 1.0
                    if idx > 0:
                        entry["tv_sum"] += self._tv_distance(pred.outcomes, preds[idx - 1].outcomes)
                        entry["tv_count"] += 1.0

        metrics_list = []
        for agent in self.agents:
            aid = agent.agent_id
            resolved = resolved_stats.get(aid, {
                "raw_brier": 0.0,
                "peer_sum": 0.0,
                "tw_peer_sum": 0.0,
                "correct_count": 0.0,
                "truth_prob_sum": 0.0,
                "count": 0.0,
            })
            agent_active = active_stats.get(aid, {
                "raw_brier": 0.0,
                "peer_sum": 0.0,
                "tw_peer_sum": 0.0,
                "correct_count": 0.0,
                "truth_prob_sum": 0.0,
                "count": 0.0,
            })
            submission_stats = daily_submission_stats.get(aid, {
                "count": 0.0,
                "tv_sum": 0.0,
                "tv_count": 0.0,
            })

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

            metrics_list.append({
                "agent_id": aid,
                "avg_brier": avg_brier,
                "peer_score": total_peer_sum,
                "tw_peer_score": total_tw_peer_sum,
                "accuracy": accuracy,
                "exp_acc": exp_acc,
                "total_predictions": int(total_count),
                "daily_submissions": int(submission_stats["count"]),
                "avg_submission_tv_to_prev": avg_submission_tv,
            })
        return metrics_list

    def _save_daily_metrics(self):
        """Save current cumulative metrics for this wakeup session to CSV."""
        active_questions = self.q_pool.get_active()
        metrics_list = self._build_metrics_list(active_questions=active_questions)
        test_metrics_list = self._build_metrics_list(
            active_questions=active_questions,
            source_split="test",
        )
        self.logger.log_daily_metrics(self.current_date, metrics_list)
        self.logger.log_test_daily_metrics(self.current_date, test_metrics_list)

    def _get_safe_active_questions(self, active_questions: List[Question]) -> List[Question]:
        """Return a copy of active questions with ground truth hidden."""
        from dataclasses import replace
        return [replace(q, ground_truth_answer="") for q in active_questions]

    def _run_agents_sequential(self, active_questions: List[Question]):
        """Run agents one at a time (original behavior)."""
        # Hide ground truth from agents
        safe_questions = self._get_safe_active_questions(active_questions)

        # Cheat-feedback context: pass real questions (with ground truth) lazily
        cheat_ctx = None
        if self.cheat_feedback:
            cheat_ctx = (active_questions, self.scorer, self.matcher)

        forecast_interface = SimForecastInterface(
            safe_questions,
            self.current_aggregates,
            self.prediction_histories,
            self.current_date,
            self.logger,
            resolved_questions=self.resolved_questions,
            resolution_events=self.resolution_events,
            resolved_agent_predictions=self.resolved_agent_predictions,
            histories_lock=self._histories_lock,
            market_csv_path=self.market_csv_path,
            cheat_feedback_ctx=cheat_ctx,
            timegap_days=self.timegap_days,
            last_active_date=self._get_last_active_date(),
            next_active_date=self._get_next_active_date(),
            simulation_end_date=self.end_date,
        )
        forecast_interface.source_name = getattr(self, 'source_name', 'openforesight')
        forecast_interface.source_context = getattr(self, 'source_context', '')
        
        for agent in self.agents:
            forecast_interface.set_agent_context(agent.agent_id)
            agent.act(None, forecast_interface, self.current_date)
    
    def _run_agents_parallel(self, active_questions: List[Question]):
        """Run agents in parallel using thread pool."""

        # Hide ground truth from agents
        safe_questions = self._get_safe_active_questions(active_questions)

        # Cheat-feedback context: pass real questions (with ground truth) lazily
        cheat_ctx = None
        if self.cheat_feedback:
            cheat_ctx = (active_questions, self.scorer, self.matcher)

        def run_agent(agent):
            """Execute a single agent's turn."""
            # Each agent gets its own forecast interface instance
            # (they share the same underlying histories dict, but with thread-safe access)
            forecast_interface = SimForecastInterface(
                safe_questions,
                self.current_aggregates,
                self.prediction_histories,
                self.current_date,
                self.logger,
                resolved_questions=self.resolved_questions,
                resolution_events=self.resolution_events,
                resolved_agent_predictions=self.resolved_agent_predictions,
                histories_lock=self._histories_lock,
                market_csv_path=self.market_csv_path,
                cheat_feedback_ctx=cheat_ctx,
                timegap_days=self.timegap_days,
                last_active_date=self._get_last_active_date(),
                next_active_date=self._get_next_active_date(),
                simulation_end_date=self.end_date,
            )
            forecast_interface.source_name = getattr(self, 'source_name', 'openforesight')
            forecast_interface.source_context = getattr(self, 'source_context', '')
            
            forecast_interface.set_agent_context(agent.agent_id)
            
            try:
                agent.act(None, forecast_interface, self.current_date)
                return agent.agent_id, None
            except Exception as e:
                return agent.agent_id, str(e)
        
        # Run all agents in parallel
        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            futures = {executor.submit(run_agent, agent): agent for agent in self.agents}
            
            for future in as_completed(futures):
                agent_id, error = future.result()
                if error:
                    print(f"  [ERROR] Agent {agent_id} failed: {error}")
            
    def _resolve_question(self, q: Question):
        """Resolve a question and compute final scores."""
        from .scoring import compute_snapshot_peer_scores
        
        history = self.prediction_histories.get(q.qid)
        if not history or not history.predictions:
            return
            
        # Resolve and compute time-weighted scores
        result = resolve_question(
            history,
            q.ground_truth_answer,
            self.matcher,
            scorer=self.scorer,
            question_title=q.title,
        )
        
        # Compute snapshot scores (last prediction only, not time-weighted)
        final_snapshot = history.get_all_current_predictions()
        self._store_resolved_final_predictions(q.qid, final_snapshot)
        scorer = self.scorer
        
        # Raw Brier scores (per agent, not peer)
        raw_brier_scores = {}
        for agent_id, pred in final_snapshot.items():
            raw_brier = scorer.score_prediction(pred, q.ground_truth_answer, self.matcher,
                                                 question_id=q.qid, question_title=q.title)
            raw_brier_scores[agent_id] = raw_brier
            if agent_id in self.agent_raw_brier:
                self.agent_raw_brier[agent_id] += raw_brier
        
        # Snapshot peer scores (not time-weighted)
        snapshot_peer_scores = {}
        if final_snapshot:
            snapshot_peer_scores = compute_snapshot_peer_scores(
                final_snapshot, q.ground_truth_answer, scorer, self.matcher,
                question_id=q.qid, question_title=q.title
            )
            for agent_id, peer_score in snapshot_peer_scores.items():
                if agent_id in self.agent_snapshot_peer:
                    self.agent_snapshot_peer[agent_id] += peer_score
        
        # Count resolved snapshot predictions even if the time-weighted window is empty
        # (e.g. warmup forecasts made after the official resolution date).
        per_agent_event: Dict[str, Dict[str, Any]] = {}
        event_agent_ids = set(final_snapshot) | set(result.agent_scores)
        for agent_id in event_agent_ids:
            pred = final_snapshot.get(agent_id)
            tw_peer = float(result.agent_scores.get(agent_id, 0.0))
            best_outcome, best_prob = self._get_top_outcome(pred)
            is_accurate = False
            truth_prob = 0.0
            if pred is not None:
                is_accurate = self._is_top_choice_correct(
                    pred,
                    q.ground_truth_answer,
                    question_id=q.qid,
                    question_title=q.title
                )
                truth_prob = self._get_truth_probability_mass(
                    pred,
                    q.ground_truth_answer,
                    question_id=q.qid,
                    question_title=q.title
                )

            per_agent_event[agent_id] = {
                "brier": raw_brier_scores.get(agent_id),
                "snapshot_peer": snapshot_peer_scores.get(agent_id),
                "tw_peer": tw_peer,
                "best_outcome": best_outcome,
                "best_prob": best_prob,
                "truth_prob": truth_prob,
                "is_accurate": bool(is_accurate),
            }

            if agent_id in self.agent_scores:
                self.agent_scores[agent_id] += tw_peer

            if pred is None:
                continue

            self.agent_questions[agent_id] = self.agent_questions.get(agent_id, 0) + 1
            self.agent_exp_acc_sum[agent_id] = self.agent_exp_acc_sum.get(agent_id, 0.0) + truth_prob
            if is_accurate:
                self.agent_correct[agent_id] = self.agent_correct.get(agent_id, 0) + 1
            else:
                self.agent_wrong[agent_id] = self.agent_wrong.get(agent_id, 0) + 1
                
        self.logger.log_resolution(
            self.current_date, q.qid, 
            q.ground_truth_answer, result.agent_scores,
            raw_brier=raw_brier_scores,
            snapshot_peer=snapshot_peer_scores
        )
        self.resolution_events.append({
            "sim_date": str(self.current_date),
            "qid": q.qid,
            "title": q.title,
            "source_split": q.source_split,
            "ground_truth": q.ground_truth_answer,
            "agents": per_agent_event,
        })
        print(f"  Resolved: {q.title[:40]}... → '{q.ground_truth_answer[:20]}'")
        for aid, sc in result.agent_scores.items():
            print(f"    {aid}: {sc:+.2f}")
        
        # Clean up
        del self.prediction_histories[q.qid]
        if q.qid in self.current_aggregates:
            del self.current_aggregates[q.qid]
            
    def _update_aggregates(self, active_questions: List[Question]):
        """Update aggregate probabilities for all active questions."""
        for q in active_questions:
            history = self.prediction_histories.get(q.qid)
            if history:
                current_preds = list(history.get_all_current_predictions().values())
                if current_preds:
                    new_aggregate = compute_aggregate(current_preds)
                    self.current_aggregates[q.qid] = new_aggregate
                    self.logger.log_aggregate(self.current_date, q.qid, new_aggregate)

    def _restore_state(self, resume_dir: str):
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
            
        last_date = self.current_date
        records_processed = 0
        self.resolved_agent_predictions = {}
        
        dates_seen = set()
        
        with open(actions_path, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    records_processed += 1
                    
                    sim_date_str = record.get("sim_date")
                    if sim_date_str:
                        sim_date = date.fromisoformat(sim_date_str)
                        if sim_date > last_date:
                            last_date = sim_date
                        dates_seen.add(sim_date)
                        
                    rtype = record.get("type")
                    
                    if rtype == "prediction":
                        # Rebuild history
                        qid = str(record.get("question_id")) if record.get("question_id") is not None else None
                        agent_id = record.get("agent_id")
                        outcomes = record.get("outcomes")
                        
                        if qid and agent_id and outcomes:
                            # Ensure history exists
                            if qid not in self.prediction_histories:
                                # Need resolution date to init history
                                q = self.q_pool.get_question(qid)
                                if q:
                                    self.prediction_histories[qid] = PredictionHistory(
                                        question_id=qid,
                                        start_date=sim_date, # Approximation, will update
                                        resolution_date=q.resolution_date
                                    )
                                    # Fix start date if this pred is earlier
                                    if sim_date < self.prediction_histories[qid].start_date:
                                        self.prediction_histories[qid].start_date = sim_date
                            
                            # Add prediction
                            if qid in self.prediction_histories:
                                pred = DailyPrediction(
                                    agent_id=agent_id,
                                    question_id=qid,
                                    day=sim_date,
                                    outcomes=outcomes
                                )
                                self.prediction_histories[qid].add_prediction(pred)
                                
                    elif rtype == "resolution":
                        # Restore scores and resolved status
                        qid = str(record.get("question_id")) if record.get("question_id") is not None else None
                        scores = record.get("agent_scores", {})
                        ground_truth = record.get("ground_truth", "")
                        q = self.q_pool.get_question(qid) if qid else None
                        question_title = q.title if q else None

                        history = self.prediction_histories.get(qid) if qid else None
                        final_snapshot = history.get_all_current_predictions() if history else {}
                        if qid:
                            self._store_resolved_final_predictions(qid, final_snapshot)
                        
                        # Restore raw_brier and snapshot_peer if available (new format)
                        raw_brier = record.get("raw_brier", {})
                        snapshot_peer = record.get("snapshot_peer", {})
                        
                        # Restore snapshot-based metrics even when time-weighted scores were empty.
                        per_agent_event: Dict[str, Dict[str, Any]] = {}
                        event_agent_ids = set(final_snapshot) | set(scores) | set(raw_brier) | set(snapshot_peer)
                        for aid in event_agent_ids:
                            score = float(scores.get(aid, 0.0))
                            self.agent_scores[aid] = self.agent_scores.get(aid, 0.0) + score

                            pred = final_snapshot.get(aid)
                            best_outcome, best_prob = self._get_top_outcome(pred)
                            is_accurate = False
                            truth_prob = 0.0
                            if pred and ground_truth:
                                self.agent_questions[aid] = self.agent_questions.get(aid, 0) + 1
                                truth_prob = self._get_truth_probability_mass(
                                    pred,
                                    ground_truth,
                                    question_id=qid,
                                    question_title=question_title
                                )
                                self.agent_exp_acc_sum[aid] = self.agent_exp_acc_sum.get(aid, 0.0) + truth_prob
                                is_accurate = self._is_top_choice_correct(
                                    pred,
                                    ground_truth,
                                    question_id=qid,
                                    question_title=question_title
                                )
                                if is_accurate:
                                    self.agent_correct[aid] = self.agent_correct.get(aid, 0) + 1
                                else:
                                    self.agent_wrong[aid] = self.agent_wrong.get(aid, 0) + 1
                            else:
                                # Backward-compatible fallback for older logs without enough data.
                                if score > 0:
                                    self.agent_correct[aid] = self.agent_correct.get(aid, 0) + 1
                                    is_accurate = True
                                elif score < 0:
                                    self.agent_wrong[aid] = self.agent_wrong.get(aid, 0) + 1
                                    is_accurate = False

                            per_agent_event[aid] = {
                                "brier": raw_brier.get(aid),
                                "snapshot_peer": snapshot_peer.get(aid),
                                "tw_peer": score,
                                "best_outcome": best_outcome,
                                "best_prob": best_prob,
                                "truth_prob": truth_prob,
                                "is_accurate": bool(is_accurate),
                            }
                            
                            # Restore raw_brier if logged
                            if aid in raw_brier:
                                self.agent_raw_brier[aid] = self.agent_raw_brier.get(aid, 0.0) + raw_brier[aid]
                            
                            # Restore snapshot_peer if logged
                            if aid in snapshot_peer:
                                self.agent_snapshot_peer[aid] = self.agent_snapshot_peer.get(aid, 0.0) + snapshot_peer[aid]

                        # Mark as resolved in pool
                        if qid:
                            self.q_pool._resolved.add(qid)
                            
                            # Clean up history like _resolve_question does
                            if qid in self.prediction_histories:
                                del self.prediction_histories[qid]
                                
                            q = self.q_pool.get_question(qid)
                            if q:
                                self.resolved_questions.append(q)
                                self.resolution_events.append({
                                    "sim_date": sim_date_str,
                                    "qid": qid,
                                    "title": q.title,
                                    "source_split": q.source_split,
                                    "ground_truth": ground_truth,
                                    "agents": per_agent_event,
                                })
                                
                except json.JSONDecodeError:
                    print(f"  Warning: Skipped corrupted line in actions.jsonl")
                    continue
                    
        # Advance to next scheduled wakeup after the last processed date.
        if records_processed > 0:
            self.current_date = last_date + timedelta(days=self.timegap_days)
            print(f"  Processed {records_processed} records.")
            print(f"  Fast-forwarded to {self.current_date}.")
            
            # Re-sync QPool heap (remove resolved queries we just added to _resolved)
            # This is automatically handled by pop_resolving / get_active checking _resolved set
            # But we should ensure we popped any "resolving" events that happened in the past
            
            print(f"  State restored: {len(self.prediction_histories)} active questions, {len(self.resolved_questions)} past resolutions.")
        else:
            print("  Warning: actions.jsonl was empty. Starting from beginning.")



def _compute_agent_cheat_feedback(agent_id, questions, histories, scorer, matcher, sim_date, detail="full"):
    """Compute privileged cheat-feedback for a single agent on today's predictions.

    Returns ``{'items': [...], 'summary': {...}}`` or empty dict if no
    predictions were updated today.
    """
    items = []
    for q in questions:
        if not q.ground_truth_answer:
            continue
        history = histories.get(q.qid)
        if not history:
            continue
        preds = history.predictions.get(agent_id, [])
        if not preds:
            continue
        current = preds[-1]
        # Only include if prediction was made/updated today
        if current.day != sim_date:
            continue
        current_brier = scorer.score_prediction(
            current, q.ground_truth_answer, matcher,
            question_id=q.qid, question_title=q.title,
        )
        previous_brier = None
        direction = "first_prediction"
        if len(preds) >= 2:
            prev = preds[-2]
            previous_brier = scorer.score_prediction(
                prev, q.ground_truth_answer, matcher,
                question_id=q.qid, question_title=q.title,
            )
            if current_brier > previous_brier:
                direction = "improved"
            elif current_brier < previous_brier:
                direction = "worsened"
            else:
                direction = "unchanged"
        else:
            previous_brier = 0.0  # abstainer baseline
            if current_brier > 0.0:
                direction = "improved"
            elif current_brier < 0.0:
                direction = "worsened"
            else:
                direction = "unchanged"

        item = {"qid": q.qid, "title": q.title, "direction": direction}
        if detail == "full":
            item["current_brier"] = current_brier
            item["previous_brier"] = previous_brier
        items.append(item)

    if not items:
        return {}

    n = len(items)
    improved = sum(1 for i in items if i["direction"] == "improved")
    worsened = sum(1 for i in items if i["direction"] == "worsened")
    summary = {
        "total": n,
        "improved": improved,
        "worsened": worsened,
        "unchanged": n - improved - worsened,
    }
    if detail == "full" and n > 0:
        summary["avg_brier"] = sum(i["current_brier"] for i in items) / n
    return {"items": items, "summary": summary}


class SimForecastInterface:
    """
    Agent's interface to forecasting questions.

    Thread-safe for parallel agent execution.
    """
    
    def __init__(self, 
                 questions: List[Question],
                 aggregates: Dict[str, Dict[str, float]],
                 histories: Dict[str, PredictionHistory],
                 sim_date: date,
                 logger: SimLogger,
                 resolved_questions: List[Question] = None,
                 resolution_events: List[Dict[str, Any]] = None,
                 resolved_agent_predictions: Dict[str, Dict[str, Dict[str, Any]]] = None,
                 histories_lock: Lock = None,
                 market_csv_path: str = None,
                 cheat_feedback_ctx: Optional[Tuple] = None,
                 timegap_days: int = 1,
                 last_active_date: Optional[date] = None,
                 next_active_date: Optional[date] = None,
                 simulation_end_date: Optional[date] = None):
        self.questions = {q.qid: q for q in questions}
        self.aggregates = aggregates
        self.histories = histories
        self.sim_date = sim_date
        self.timegap_days = max(1, int(timegap_days or 1))
        self.last_active_date = last_active_date
        self.next_active_date = next_active_date
        self.simulation_end_date = simulation_end_date
        self.logger = logger
        self.current_agent_id: Optional[str] = None
        self.resolved_questions = resolved_questions or []
        self.resolution_events = resolution_events or []
        self._resolved_agent_predictions = resolved_agent_predictions or {}
        self._histories_lock = histories_lock or Lock()
        self._market_csv_path = market_csv_path
        self._day_complete = False
        # Cheat feedback context: (questions_with_truth, scorer, matcher) or None
        self._cheat_feedback_ctx = cheat_feedback_ctx
        
    def set_agent_context(self, agent_id: str):
        self.current_agent_id = agent_id
        self._day_complete = False  # Reset for new agent
    
    def next_day(self) -> None:
        """Agent signals they are done with their current wakeup session."""
        self._day_complete = True
    
    def is_day_complete(self) -> bool:
        """Check if agent has signaled day completion."""
        return self._day_complete

    def get_cheat_feedback(self, detail: str = "full") -> dict:
        """Compute privileged cheat-feedback for the current agent.

        Returns ``{'items': [...], 'summary': {...}}`` when enabled,
        or ``{}`` when cheat-feedback is disabled.
        """
        if not self._cheat_feedback_ctx or not self.current_agent_id:
            return {}
        questions, scorer, matcher = self._cheat_feedback_ctx
        return _compute_agent_cheat_feedback(
            self.current_agent_id, questions, self.histories,
            scorer, matcher, self.sim_date, detail,
        )

    def submit_prediction(self, prediction: PredictionSubmission) -> None:
        """Record a probabilistic prediction (thread-safe)."""
        if not self.current_agent_id:
            raise ValueError("No agent context set")
        if prediction.question_id not in self.questions:
            raise ValueError(f"Question {prediction.question_id} not active")
        
        # Validate probabilities
        total = sum(prediction.outcomes.values())
        if total > 1.0 + 1e-6:
            raise ValueError(f"Probabilities sum to {total} > 1")
        for outcome, prob in prediction.outcomes.items():
            if prob < 0 or prob > 1:
                raise ValueError(f"Invalid probability {prob} for {outcome}")
        
        # Create prediction record
        daily_pred = DailyPrediction(
            agent_id=self.current_agent_id,
            question_id=prediction.question_id,
            day=self.sim_date,
            outcomes=prediction.outcomes
        )
        
        # Add to history (thread-safe)
        with self._histories_lock:
            history = self.histories.get(prediction.question_id)
            if history:
                history.add_prediction(daily_pred)
            
        # Log (already thread-safe)
        self.logger.log_prediction(
            self.sim_date, self.current_agent_id,
            prediction.question_id, prediction.outcomes
        )
    
    def get_market_csv_path(self) -> str:
        """Get path to market.csv for agent to load."""
        return self._market_csv_path
    
    def get_agent_predictions(self, agent_id: str) -> Dict[str, Dict]:
        """
        Get an agent's latest predictions for active and resolved questions.
        
        Returns dict: {qid: {'outcomes': {...}, 'date': date}}
        Used by DfInterface to populate my_prediction columns.
        """
        result = {}
        with self._histories_lock:
            # Resolved snapshots captured by the environment at resolution time.
            for qid, by_agent in self._resolved_agent_predictions.items():
                record = (by_agent or {}).get(agent_id)
                if isinstance(record, dict):
                    result[qid] = {
                        'outcomes': dict(record.get('outcomes') or {}),
                        'date': record.get('date')
                    }

            # Active histories remain source-of-truth for unresolved questions.
            for qid, history in self.histories.items():
                pred = history.get_latest_prediction(agent_id)
                if pred:
                    result[qid] = {
                        'outcomes': pred.outcomes,
                        'date': pred.day
                    }
        return result

    def log_model_output(self, prompt: Any, response: str,
                         metadata: Optional[Dict[str, Any]] = None):
        if self.current_agent_id:
            self.logger.log_model_output(
                self.sim_date, self.current_agent_id, prompt, response, metadata
            )

    def flush_warmup_raw_logs(self) -> None:
        if self.current_agent_id:
            self.logger.flush_warmup_raw(self.current_agent_id)
