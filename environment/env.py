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
    DEFAULT_SCORER
)
from .ansmatching import AnswerMatcher



class SimLogger:
    """
    Thread-safe centralized logging for simulation events.
    
    Logs shared events (predictions, resolutions) to central files,
    and model outputs to per-agent directories.
    """
    
    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Shared logs (thread-safe with locks)
        self._actions_lock = Lock()
        self.actions_file = open(os.path.join(output_dir, "actions.jsonl"), "w")
        
        self._metrics_lock = Lock()
        self.metrics_file = open(os.path.join(output_dir, "daily_metrics.csv"), "w")
        self.metrics_file.write("date,agent_id,avg_brier,peer_score,tw_peer_score,accuracy,total_predictions\n")
        
        self._matcher_lock = Lock()
        self.matcher_file = open(os.path.join(output_dir, "matcher.jsonl"), "w")
        
        # Per-agent output files (lazy initialized)
        self.agents_dir = os.path.join(output_dir, "agents")
        self._agent_files: Dict[str, Any] = {}
        self._agent_files_lock = Lock()
        
    def _get_agent_output_files(self, agent_id: str):
        """Get or create the output files for an agent (thread-safe). Returns (output_file, raw_file)."""
        with self._agent_files_lock:
            if agent_id not in self._agent_files:
                agent_dir = os.path.join(self.agents_dir, agent_id)
                os.makedirs(agent_dir, exist_ok=True)
                out_path = os.path.join(agent_dir, "model_outputs.jsonl")
                raw_path = os.path.join(agent_dir, "model_raw.jsonl")
                self._agent_files[agent_id] = (open(out_path, "w"), open(raw_path, "w"))
            return self._agent_files[agent_id]
        
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
                       ground_truth: str, agent_scores: Dict[str, float]):
        record = {
            "sim_date": str(sim_date),
            "type": "resolution",
            "question_id": question_id,
            "ground_truth": ground_truth,
            "agent_scores": agent_scores
        }
        with self._actions_lock:
            self.actions_file.write(json.dumps(record) + "\n")
            self.actions_file.flush()

    def log_daily_metrics(self, sim_date: date, metrics_list: List[Dict[str, Any]]):
        with self._metrics_lock:
            for m in metrics_list:
                row = f"{sim_date},{m['agent_id']},{m['avg_brier']:.4f},{m['peer_score']:.4f},{m['tw_peer_score']:.4f},{m['accuracy']:.2f},{m['total_predictions']}\n"
                self.metrics_file.write(row)
            self.metrics_file.flush()
        
    def log_model_output(self, sim_date: date, agent_id: str, prompt: str, 
                         response: str, metadata: Optional[Dict[str, Any]] = None):
        """Log model output to per-agent files (thread-safe)."""
        metadata = metadata or {}
        
        # Raw record: includes input prompt and reasoning
        raw_record = {
            "sim_date": str(sim_date),
            "agent_id": agent_id,
            "prompt": prompt,
            "response": response,
            "metadata": metadata
        }
        
        # Clean record: only final response, no prompt
        # We also filter out bulky reasoning from the clean log
        clean_metadata = metadata.copy()
        if "reasoning" in clean_metadata:
            del clean_metadata["reasoning"]

        clean_record = {
            "sim_date": str(sim_date),
            "agent_id": agent_id,
            "response": response,
            "metadata": clean_metadata
        }
        
        out_file, raw_file = self._get_agent_output_files(agent_id)
        # Each agent file is only written by one thread, so no lock needed per-file
        out_file.write(json.dumps(clean_record) + "\n")
        out_file.flush()
        
        raw_file.write(json.dumps(raw_record) + "\n")
        raw_file.flush()
        
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
        self.matcher_file.close()
        with self._agent_files_lock:
            for f_out, f_raw in self._agent_files.values():
                f_out.close()
                f_raw.close()
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
                 dataset_name: str, 
                 start_date: date,
                 end_date: date,
                 inference_provider=None,
                 output_dir: str = ".",
                 resolution_start: date = None,
                 resolution_end: date = None,
                 parallel: bool = True,
                 split: str = "train"):
        self.current_date = start_date
        self.end_date = end_date
        self.output_dir = output_dir
        self.parallel = parallel
        self.split = split
        
        # Logging and market state
        self.logger = SimLogger(output_dir)
        self.market_writer = MarketWriter(output_dir)
        
        # Components - filter questions to resolution window
        self.q_pool = QuestionPool(
            dataset_name,
            split=split,
            resolution_start=resolution_start,
            resolution_end=resolution_end
        )
        self.matcher = AnswerMatcher(inference_provider, logger=self.logger) if inference_provider else None
        self.scorer = DEFAULT_SCORER
        
        # Track prediction history per question
        self.prediction_histories: Dict[str, PredictionHistory] = {}
        self._histories_lock = Lock()  # For thread-safe history updates
        
        # Current aggregate per question (updated end of each day)
        self.current_aggregates: Dict[str, Dict[str, float]] = {}
        
        self.agents = []
        self.agent_scores: Dict[str, float] = {}  # Cumulative time-weighted peer scores
        
        # Track prediction outcomes for summary
        self.agent_correct: Dict[str, int] = {}  # Count of positive scores
        self.agent_wrong: Dict[str, int] = {}    # Count of negative scores
        self.agent_questions: Dict[str, int] = {}  # Total questions predicted on
        
        # Additional metrics for final summary
        self.agent_raw_brier: Dict[str, float] = {}  # Sum of raw Brier scores (last pred only)
        self.agent_snapshot_peer: Dict[str, float] = {}  # Sum of peer scores at resolution (not time-weighted)
        
        # Track resolved questions for agent learning
        self.resolved_questions: List[Question] = []
        
        self.market_csv_path: Optional[str] = None
        
    def add_agent(self, agent):
        self.agents.append(agent)
        self.agent_scores[agent.agent_id] = 0.0
        self.agent_correct[agent.agent_id] = 0
        self.agent_wrong[agent.agent_id] = 0
        self.agent_questions[agent.agent_id] = 0
        self.agent_raw_brier[agent.agent_id] = 0.0
        self.agent_snapshot_peer[agent.agent_id] = 0.0
        
    def run(self):
        print(f"Simulation: {self.current_date} to {self.end_date} ({self.q_pool.total_count} questions)")
        
        while self.current_date <= self.end_date:
            print(f"\n--- Day {self.current_date} ---")
            self.step()
            self._print_daily_scores()
            self.current_date += timedelta(days=1)
        
        self.logger.close()
        self._print_final_summary()
    
    def _print_daily_scores(self):
        """Print current cumulative scores for all agents."""
        if self.agent_scores:
            scores_str = ", ".join(f"{aid}: {sc:+.2f}" for aid, sc in sorted(self.agent_scores.items()))
            print(f"  Scores: {scores_str}")
    
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
        active_stats = self._compute_daily_active_scores(active_questions)
        
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

    def step(self):
        # 1. Resolve questions expiring today
        resolving = self.q_pool.pop_resolving(self.current_date)
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
        
        # 6. Save daily metrics
        self._save_daily_metrics()
    
    def _compute_daily_active_scores(self, active_questions: List[Question]):
        """
        Compute daily scores only for the currently active questions.
        Uses FUTURE ground truth to assess current performance.
        Includes Brier score, Peer score, TW-Peer score, and Accuracy.
        """
        active_stats = {}
        
        # Helper normalization for accuracy check
        def normalize(s: str) -> str:
            return s.lower().replace(" ", "").strip()
        
        from environment.scoring import resolve_question, compute_snapshot_peer_scores
            
        for q in active_questions:
            # We strictly need ground truth here
            if not q.ground_truth_answer:
                continue
                
            history = self.prediction_histories.get(q.qid)
            if not history or not history.predictions:
                continue
                
            # Get latest predictions as of today (Snapshot)
            # Use explicit loop since get_all_current_predictions doesn't accept date
            snapshot = {}
            for aid in history.predictions.keys():
                pred = history.get_prediction_as_of(aid, self.current_date)
                if pred:
                    snapshot[aid] = pred
            
            # 1. Compute Active Snapshot Peer Scores
            snapshot_peers = {}
            if snapshot:
                snapshot_peers = compute_snapshot_peer_scores(
                    snapshot, q.ground_truth_answer, self.scorer, self.matcher
                )

            # 2. Compute Active TW-Peer Scores (Partial Resolution)
            # Use resolve_question with evaluation_date=today
            partial_result = resolve_question(
                history, 
                q.ground_truth_answer, 
                self.matcher, 
                self.scorer, 
                evaluation_date=self.current_date
            )
            tw_peers = partial_result.agent_scores

            # Process per-agent stats for this question
            for agent in self.agents:
                aid = agent.agent_id
                
                # Check if agent has a prediction in snapshot
                pred = snapshot.get(aid)
                
                if pred:
                    # Brier Score
                    brier = self.scorer.score_prediction(pred, q.ground_truth_answer, self.matcher)
                    
                    # Accuracy: Highest probability guess matches truth (ties broken arbitrarily)
                    # Find outcome with max probability
                    max_prob = -1.0
                    best_outcome = None
                    for outcome, prob in pred.outcomes.items():
                        if prob > max_prob:
                            max_prob = prob
                            best_outcome = outcome
                    
                    # Check match
                    is_accurate = False
                    if best_outcome:
                        if self.matcher:
                            if self.matcher.is_equivalent(best_outcome, q.ground_truth_answer):
                                is_accurate = True
                        else:
                            if normalize(best_outcome) == normalize(q.ground_truth_answer):
                                is_accurate = True
                    
                    # Aggregate stats
                    if aid not in active_stats:
                        active_stats[aid] = {
                            'raw_brier': 0.0, 
                            'peer_sum': 0.0,
                            'tw_peer_sum': 0.0,
                            'correct_count': 0,
                            'count': 0
                        }
                    
                    active_stats[aid]['raw_brier'] += brier
                    active_stats[aid]['peer_sum'] += snapshot_peers.get(aid, 0.0)
                    active_stats[aid]['tw_peer_sum'] += tw_peers.get(aid, 0.0)
                    active_stats[aid]['correct_count'] += 1 if is_accurate else 0
                    active_stats[aid]['count'] += 1
                    
        return active_stats
    
    def _save_daily_metrics(self):
        """Save current cumulative metrics for all agents to CSV."""
        # 1. Compute scores for ACTIVE questions (using hidden truth)
        active_questions = self.q_pool.get_active()
        active_stats = self._compute_daily_active_scores(active_questions)
        
        metrics_list = []
        for agent in self.agents:
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
            
            # Combined stats
            total_brier_sum = resolved_brier_sum + agent_active['raw_brier']
            total_peer_sum = resolved_snapshot_peer + agent_active['peer_sum']
            total_tw_peer_sum = resolved_tw_peer + agent_active['tw_peer_sum']
            total_correct = resolved_correct + agent_active['correct_count']
            total_count = resolved_count + agent_active['count']
            
            avg_brier = total_brier_sum / total_count if total_count > 0 else 0.0
            accuracy = (total_correct / total_count * 100) if total_count > 0 else 0.0
            
            metrics_list.append({
                'agent_id': aid,
                'avg_brier': avg_brier,
                'peer_score': total_peer_sum,
                'tw_peer_score': total_tw_peer_sum,
                'accuracy': accuracy,
                'total_predictions': total_count
            })
            
        self.logger.log_daily_metrics(self.current_date, metrics_list)

    def _get_safe_active_questions(self, active_questions: List[Question]) -> List[Question]:
        """Return a copy of active questions with ground truth hidden."""
        from dataclasses import replace
        return [replace(q, ground_truth_answer="") for q in active_questions]

    def _run_agents_sequential(self, active_questions: List[Question]):
        """Run agents one at a time (original behavior)."""
        # Hide ground truth from agents
        safe_questions = self._get_safe_active_questions(active_questions)
        
        forecast_interface = SimForecastInterface(
            safe_questions, 
            self.current_aggregates,
            self.prediction_histories,
            self.current_date,
            self.logger,
            resolved_questions=self.resolved_questions,
            histories_lock=self._histories_lock,
            market_csv_path=self.market_csv_path,
        )
        
        for agent in self.agents:
            forecast_interface.set_agent_context(agent.agent_id)
            agent.act(None, forecast_interface, self.current_date)
    
    def _run_agents_parallel(self, active_questions: List[Question]):
        """Run agents in parallel using thread pool."""
        
        # Hide ground truth from agents
        safe_questions = self._get_safe_active_questions(active_questions)
        
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
                histories_lock=self._histories_lock,
                market_csv_path=self.market_csv_path,
            )
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
        from .scoring import BrierScorer, compute_snapshot_peer_scores
        
        history = self.prediction_histories.get(q.qid)
        if not history or not history.predictions:
            return
            
        # Resolve and compute time-weighted scores
        result = resolve_question(history, q.ground_truth_answer, self.matcher)
        
        # Compute snapshot scores (last prediction only, not time-weighted)
        final_snapshot = history.get_all_current_predictions()
        scorer = BrierScorer()
        
        # Raw Brier scores (per agent, not peer)
        for agent_id, pred in final_snapshot.items():
            if agent_id in self.agent_raw_brier:
                raw_brier = scorer.score_prediction(pred, q.ground_truth_answer, self.matcher)
                self.agent_raw_brier[agent_id] += raw_brier
        
        # Snapshot peer scores (not time-weighted)
        if final_snapshot:
            snapshot_peer = compute_snapshot_peer_scores(
                final_snapshot, q.ground_truth_answer, scorer, self.matcher
            )
            for agent_id, peer_score in snapshot_peer.items():
                if agent_id in self.agent_snapshot_peer:
                    self.agent_snapshot_peer[agent_id] += peer_score
        
        # Update cumulative time-weighted scores
        for agent_id, score in result.agent_scores.items():
            if agent_id in self.agent_scores:
                self.agent_scores[agent_id] += score
                self.agent_questions[agent_id] = self.agent_questions.get(agent_id, 0) + 1
                if score > 0:
                    self.agent_correct[agent_id] = self.agent_correct.get(agent_id, 0) + 1
                elif score < 0:
                    self.agent_wrong[agent_id] = self.agent_wrong.get(agent_id, 0) + 1
                
        self.logger.log_resolution(
            self.current_date, q.qid, 
            q.ground_truth_answer, result.agent_scores
        )
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
                 histories_lock: Lock = None,
                 market_csv_path: str = None):
        self.questions = {q.qid: q for q in questions}
        self.aggregates = aggregates
        self.histories = histories
        self.sim_date = sim_date
        self.logger = logger
        self.current_agent_id: Optional[str] = None
        self.resolved_questions = resolved_questions or []
        self._histories_lock = histories_lock or Lock()
        self._market_csv_path = market_csv_path
        self._day_complete = False
        
    def set_agent_context(self, agent_id: str):
        self.current_agent_id = agent_id
        self._day_complete = False  # Reset for new agent
    
    def next_day(self) -> None:
        """Agent signals they are done with their actions for today."""
        self._day_complete = True
    
    def is_day_complete(self) -> bool:
        """Check if agent has signaled day completion."""
        return self._day_complete
    
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
        Get an agent's current predictions for all questions.
        
        Returns dict: {qid: {'outcomes': {...}, 'date': date}}
        Used by DfInterface to populate my_prediction columns.
        """
        result = {}
        for qid, history in self.histories.items():
            pred = history.get_latest_prediction(agent_id)
            if pred:
                result[qid] = {
                    'outcomes': pred.outcomes,
                    'date': pred.day
                }
        return result
    
    def log_model_output(self, prompt: str, response: str, 
                         metadata: Optional[Dict[str, Any]] = None):
        if self.current_agent_id:
            self.logger.log_model_output(
                self.sim_date, self.current_agent_id, prompt, response, metadata
            )
