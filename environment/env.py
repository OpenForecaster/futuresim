import os
import json
import pandas as pd
from datetime import date, timedelta
from threading import Lock
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from .interfaces import (
    AgentInterface, ForecastInterface, ArticleMeta, 
    QuestionView, PredictionSubmission
)
from .data_loader import QuestionPool, Question
from .scoring import (
    DailyPrediction, PredictionHistory, 
    compute_aggregate, resolve_question
)
from .ansmatching import AnswerMatcher
from .safe_executor import QueryExecutor



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
        
        # Per-agent output files (lazy initialized)
        self.agents_dir = os.path.join(output_dir, "agents")
        self._agent_files: Dict[str, Any] = {}
        self._agent_files_lock = Lock()
        
    def _get_agent_output_file(self, agent_id: str):
        """Get or create the output file for an agent (thread-safe)."""
        with self._agent_files_lock:
            if agent_id not in self._agent_files:
                agent_dir = os.path.join(self.agents_dir, agent_id)
                os.makedirs(agent_dir, exist_ok=True)
                file_path = os.path.join(agent_dir, "model_outputs.jsonl")
                self._agent_files[agent_id] = open(file_path, "w")
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
        record = {
            "sim_date": str(sim_date),
            "type": "aggregate",
            "question_id": question_id,
            "aggregate": aggregate
        }
        with self._actions_lock:
            self.actions_file.write(json.dumps(record) + "\n")
            self.actions_file.flush()
        
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
        
    def log_model_output(self, sim_date: date, agent_id: str, prompt: str, 
                         response: str, metadata: Optional[Dict[str, Any]] = None):
        """Log model output to per-agent file (thread-safe)."""
        record = {
            "sim_date": str(sim_date),
            "agent_id": agent_id,
            "prompt": prompt,
            "response": response,
            "metadata": metadata or {}
        }
        output_file = self._get_agent_output_file(agent_id)
        # Each agent file is only written by one thread, so no lock needed per-file
        output_file.write(json.dumps(record) + "\n")
        output_file.flush()
        
    def close(self):
        self.actions_file.close()
        with self._agent_files_lock:
            for f in self._agent_files.values():
                f.close()
            self._agent_files.clear()


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
                 context_dir: str,
                 inference_provider=None,
                 output_dir: str = ".",
                 resolution_start: date = None,
                 resolution_end: date = None,
                 parallel: bool = True):
        self.current_date = start_date
        self.end_date = end_date
        self.context_dir = context_dir
        self.output_dir = output_dir
        self.parallel = parallel
        
        # Components - filter questions to resolution window
        self.q_pool = QuestionPool(
            dataset_name,
            resolution_start=resolution_start,
            resolution_end=resolution_end
        )
        self.matcher = AnswerMatcher(inference_provider) if inference_provider else None
        
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
        
        # Logging
        self.logger = SimLogger(output_dir)
        
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
        
        # Table header
        header = f"{'Agent':<25} {'Avg Brier':>10} {'Peer':>10} {'TW-Peer':>10} {'Correct':>8} {'Wrong':>6} {'Total':>6}"
        print(header)
        print("-"*90)
        
        # Table rows
        for agent in sorted(self.agents, key=lambda a: a.agent_id):
            aid = agent.agent_id
            raw_brier_sum = self.agent_raw_brier.get(aid, 0.0)
            snapshot_peer = self.agent_snapshot_peer.get(aid, 0.0)
            tw_peer = self.agent_scores.get(aid, 0.0)
            correct = self.agent_correct.get(aid, 0)
            wrong = self.agent_wrong.get(aid, 0)
            total = self.agent_questions.get(aid, 0)
            
            # Average Brier per question
            avg_brier = raw_brier_sum / total if total > 0 else 0.0
            
            # Truncate long agent names
            display_name = aid[:24] if len(aid) > 24 else aid
            
            row = f"{display_name:<25} {avg_brier:>+10.3f} {snapshot_peer:>+10.2f} {tw_peer:>+10.2f} {correct:>8} {wrong:>6} {total:>6}"
            print(row)
        
        print("-"*90)
        print("Legend: Avg Brier=Mean Brier score per question, Peer=Snapshot peer sum, TW-Peer=Time-weighted peer sum")
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
        
        # 3. Collect predictions from all agents
        if self.parallel and len(self.agents) > 1:
            self._run_agents_parallel(active_questions)
        else:
            self._run_agents_sequential(active_questions)
        
        # 4. Update aggregates (end of day)
        self._update_aggregates(active_questions)
    
    def _run_agents_sequential(self, active_questions: List[Question]):
        """Run agents one at a time (original behavior)."""
        doc_interface = SimDocInterface(self.context_dir, self.current_date)
        forecast_interface = SimForecastInterface(
            active_questions, 
            self.current_aggregates,
            self.prediction_histories,
            self.current_date,
            self.logger,
            resolved_questions=self.resolved_questions,
            histories_lock=self._histories_lock,
        )
        
        for agent in self.agents:
            forecast_interface.set_agent_context(agent.agent_id)
            agent.act(doc_interface, forecast_interface, self.current_date)
    
    def _run_agents_parallel(self, active_questions: List[Question]):
        """Run agents in parallel using thread pool."""
        doc_interface = SimDocInterface(self.context_dir, self.current_date)
        
        def run_agent(agent):
            """Execute a single agent's turn."""
            # Each agent gets its own forecast interface instance
            # (they share the same underlying histories dict, but with thread-safe access)
            forecast_interface = SimForecastInterface(
                active_questions, 
                self.current_aggregates,
                self.prediction_histories,
                self.current_date,
                self.logger,
                resolved_questions=self.resolved_questions,
                histories_lock=self._histories_lock,
            )
            forecast_interface.set_agent_context(agent.agent_id)
            
            try:
                agent.act(doc_interface, forecast_interface, self.current_date)
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


class SimDocInterface(AgentInterface):
    """Agent's view of documents. Stubbed for now."""
    
    def __init__(self, root_dir: str, current_date: date):
        self.root_dir = root_dir
        self.current_date = current_date
        
    def list_sources(self) -> List[str]:
        return []
    
    def list_dates(self, source: Optional[str] = None) -> List[date]:
        return []
        
    def list_articles(self, date_obj: date, source: Optional[str] = None) -> List[ArticleMeta]:
        return []

    def read_article(self, article_id: str) -> str:
        return ""


class SimForecastInterface(ForecastInterface):
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
                 histories_lock: Lock = None):
        self.questions = {q.qid: q for q in questions}
        self.aggregates = aggregates
        self.histories = histories
        self.sim_date = sim_date
        self.logger = logger
        self.current_agent_id: Optional[str] = None
        self.resolved_questions = resolved_questions or []
        self._histories_lock = histories_lock or Lock()
        
        # Query execution
        self.query_executor = QueryExecutor(timeout_seconds=5.0)
        self._df_cache: Optional[pd.DataFrame] = None
        
    def set_agent_context(self, agent_id: str):
        self.current_agent_id = agent_id
        self._df_cache = None  # Rebuild DataFrame for each agent (different my_prediction)

    def list_questions(self) -> List[QuestionView]:
        """Return questions with frozen aggregates."""
        views = []
        for q in self.questions.values():
            views.append(QuestionView(
                id=q.qid,
                title=q.title,
                background=q.background,
                resolution_criteria=q.resolution_criteria,
                answer_type=q.answer_type,
                resolution_date=q.resolution_date,
                aggregate=self.aggregates.get(q.qid, {})
            ))
        return views
    
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
    
    def log_model_output(self, prompt: str, response: str, 
                         metadata: Optional[Dict[str, Any]] = None):
        if self.current_agent_id:
            self.logger.log_model_output(
                self.sim_date, self.current_agent_id, prompt, response, metadata
            )
    
    def execute_query(self, code: str, current_date: date = None) -> Tuple[str, Optional[str]]:
        """
        Execute agent's Python code on the questions DataFrame.
        
        Returns (result_string, error_message).
        """
        df = self._get_dataframe()
        return self.query_executor.execute(
            df, code, 
            current_date=current_date or self.sim_date
        )
    
    def get_dataframe_info(self) -> Dict[str, Any]:
        """Get DataFrame schema info for agent prompting."""
        df = self._get_dataframe()
        
        columns_desc = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            columns_desc.append(f"- {col} ({dtype})")
        
        return {
            'n_rows': len(df),
            'n_active': len([q for q in self.questions.values()]),
            'n_resolved': len(self.resolved_questions),
            'columns': list(df.columns),
            'columns_desc': "\n".join(columns_desc),
        }
    
    def _get_dataframe(self) -> pd.DataFrame:
        """Build or return cached DataFrame for current agent."""
        if self._df_cache is not None:
            return self._df_cache
        
        rows = []
        
        # Active questions
        for q in self.questions.values():
            agg = self.aggregates.get(q.qid, {})
            
            # Get agent's current prediction if any (read-only, no lock needed)
            my_pred = None
            my_pred_date = None
            history = self.histories.get(q.qid)
            if history and self.current_agent_id:
                pred = history.get_latest_prediction(self.current_agent_id)
                if pred:
                    my_pred = pred.outcomes
                    my_pred_date = pred.day
            
            # Count total predictions
            num_preds = 0
            if history:
                for agent_preds in history.predictions.values():
                    num_preds += len(agent_preds)
            
            rows.append({
                'qid': q.qid,
                'title': q.title,
                'background': q.background,
                'resolution_criteria': q.resolution_criteria,
                'answer_type': q.answer_type,
                'resolution_date': q.resolution_date,
                'is_resolved': False,
                'ground_truth': None,
                'market_aggregate': agg if agg else None,  # Store as dict, not JSON string
                'num_predictions': num_preds,
                'my_prediction': my_pred if my_pred else None,  # Store as dict, not JSON string
                'my_prediction_date': my_pred_date,
            })
        
        # Resolved questions
        for q in self.resolved_questions:
            rows.append({
                'qid': q.qid,
                'title': q.title,
                'background': q.background,
                'resolution_criteria': q.resolution_criteria,
                'answer_type': q.answer_type,
                'resolution_date': q.resolution_date,
                'is_resolved': True,
                'ground_truth': q.ground_truth_answer,
                'market_aggregate': None,
                'num_predictions': 0,
                'my_prediction': None,
                'my_prediction_date': None,
            })
        
        self._df_cache = pd.DataFrame(rows)
        return self._df_cache
