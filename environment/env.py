import os
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
from . import logging as env_logging
from . import replay, scorekeeping
from .updater import MarketWriter, SimLogger


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
                 matcher_cache_path: Optional[str] = None,
                 max_outcomes_per_question: int = 5):
        
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
        self.max_outcomes_per_question = max(1, int(max_outcomes_per_question or 1))
        
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
        if matcher_cache_path is None:
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
        env_logging.print_run_start(
            self.current_date,
            self.end_date,
            is_resume=bool(self.resume_dir),
        )

        try:
            while self.current_date <= self.end_date:
                horizon = self._get_metrics_evaluation_date(self.current_date)
                env_logging.print_step_banner(
                    self.current_date,
                    timegap_days=self.timegap_days,
                    horizon=horizon,
                )
                self.step()
                self._print_daily_scores()
                self.current_date += timedelta(days=self.timegap_days)

            self._print_final_summary()
        finally:
            if self.matcher and hasattr(self.matcher, "persist_cache"):
                try:
                    self.matcher.persist_cache()
                except Exception as exc:
                    print(f"Warning: failed to persist matcher cache: {exc}")
            self.logger.close()
    
    def _print_daily_scores(self):
        """Print current cumulative scores for all agents."""
        env_logging.print_daily_scores(self.agent_scores)

    def _get_metrics_evaluation_date(self, sim_date: Optional[date] = None) -> date:
        """Return the end-of-interval date used for metrics at a wakeup."""
        return scorekeeping.get_metrics_evaluation_date(
            sim_date or self.current_date,
            end_date=self.end_date,
            timegap_days=self.timegap_days,
        )

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
        active_questions = self.q_pool.get_active()
        last_day = self.end_date if self.end_date is not None else (
            self.current_date - timedelta(days=self.timegap_days)
        )
        active_stats = self._compute_daily_active_scores(
            active_questions,
            evaluation_date=last_day,
            metrics_context_date=last_day,
        )
        env_logging.print_final_summary(self, active_stats)

    def rescore(self):
        replay.rescore(self)

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
        return scorekeeping.compute_daily_active_scores(
            self,
            active_questions,
            evaluation_date=evaluation_date,
            metrics_context_date=metrics_context_date,
        )

    def _warmup_matcher_cache(self, questions) -> None:
        scorekeeping.warmup_matcher_cache(self, questions)

    def _get_env_matcher_timing_snapshot(self) -> Dict[str, float]:
        return scorekeeping.get_env_matcher_timing_snapshot(self)

    @staticmethod
    def _compute_env_matcher_delta(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
        return scorekeeping.compute_env_matcher_delta(before, after)

    def _inject_env_matcher_timing_into_agent_logs(self, sim_date: date, day_matcher: Dict[str, float]) -> None:
        scorekeeping.inject_env_matcher_timing_into_agent_logs(self, sim_date, day_matcher)

    def _compute_resolved_metrics_from_events(
        self,
        *,
        source_split: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        return scorekeeping.compute_resolved_metrics_from_events(
            self,
            source_split=source_split,
        )

    @staticmethod
    def _tv_distance(left: Optional[Dict[str, float]], right: Optional[Dict[str, float]]) -> float:
        return scorekeeping.tv_distance(left, right)

    def _build_metrics_list(
        self,
        *,
        active_questions: List[Question],
        source_split: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return scorekeeping.build_metrics_list(
            self,
            active_questions=active_questions,
            source_split=source_split,
        )

    def _save_daily_metrics(self):
        """Save current cumulative metrics for this wakeup session to CSV."""
        scorekeeping.save_daily_metrics(self)

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
            resolution_events=self.resolution_events,
            resolved_agent_predictions=self.resolved_agent_predictions,
            histories_lock=self._histories_lock,
            market_csv_path=self.market_csv_path,
            timegap_days=self.timegap_days,
            last_active_date=self._get_last_active_date(),
            next_active_date=self._get_next_active_date(),
            simulation_end_date=self.end_date,
            num_agents=len(self.agents),
            max_outcomes_per_question=self.max_outcomes_per_question,
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
                timegap_days=self.timegap_days,
                last_active_date=self._get_last_active_date(),
                next_active_date=self._get_next_active_date(),
                simulation_end_date=self.end_date,
                num_agents=len(self.agents),
                max_outcomes_per_question=self.max_outcomes_per_question,
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
        env_logging.print_resolution(q, result.agent_scores)
        
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
        replay.restore_state(self, resume_dir)

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
                 timegap_days: int = 1,
                 last_active_date: Optional[date] = None,
                 next_active_date: Optional[date] = None,
                 simulation_end_date: Optional[date] = None,
                 num_agents: int = 1,
                 max_outcomes_per_question: int = 5):
        self.questions = {q.qid: q for q in questions}
        self.aggregates = aggregates
        self.histories = histories
        self.sim_date = sim_date
        self.timegap_days = max(1, int(timegap_days or 1))
        self.last_active_date = last_active_date
        self.next_active_date = next_active_date
        self.simulation_end_date = simulation_end_date
        self.num_agents = num_agents
        self.max_outcomes_per_question = max(1, int(max_outcomes_per_question or 1))
        self.logger = logger
        self.current_agent_id: Optional[str] = None
        self.resolved_questions = resolved_questions or []
        self.resolution_events = resolution_events or []
        self._resolved_agent_predictions = resolved_agent_predictions or {}
        self._histories_lock = histories_lock or Lock()
        self._market_csv_path = market_csv_path
        self._day_complete = False
        
    def set_agent_context(self, agent_id: str):
        self.current_agent_id = agent_id
        self._day_complete = False  # Reset for new agent
    
    def next_day(self) -> None:
        """Agent signals they are done with their current wakeup session."""
        self._day_complete = True
    
    def is_day_complete(self) -> bool:
        """Check if agent has signaled day completion."""
        return self._day_complete

    def list_questions(self) -> List[Question]:
        """Return the active questions visible to the current agent."""
        return list(self.questions.values())

    def submit_prediction(self, prediction: PredictionSubmission) -> None:
        """Record a probabilistic prediction (thread-safe)."""
        if not self.current_agent_id:
            raise ValueError("No agent context set")
        if prediction.question_id not in self.questions:
            raise ValueError(f"Question {prediction.question_id} not active")
        if not prediction.outcomes:
            raise ValueError("Prediction must include at least one outcome")

        prediction.outcomes = dict(
            sorted(
                prediction.outcomes.items(),
                key=lambda item: (-float(item[1]), item[0]),
            )[: self.max_outcomes_per_question]
        )
        
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
