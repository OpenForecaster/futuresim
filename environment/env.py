import os
import json
import pandas as pd
from datetime import date, timedelta
from typing import List, Dict, Any, Optional, Tuple

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
    """Centralized logging for simulation events."""
    
    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.actions_file = open(os.path.join(output_dir, "actions.jsonl"), "w")
        self.outputs_file = open(os.path.join(output_dir, "model_outputs.jsonl"), "w")
        
    def log_prediction(self, sim_date: date, agent_id: str, 
                       question_id: str, outcomes: Dict[str, float]):
        record = {
            "sim_date": str(sim_date),
            "type": "prediction",
            "agent_id": agent_id,
            "question_id": question_id,
            "outcomes": outcomes
        }
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
        self.actions_file.write(json.dumps(record) + "\n")
        self.actions_file.flush()
        
    def log_model_output(self, sim_date: date, agent_id: str, prompt: str, 
                         response: str, metadata: Optional[Dict[str, Any]] = None):
        record = {
            "sim_date": str(sim_date),
            "agent_id": agent_id,
            "prompt": prompt,
            "response": response,
            "metadata": metadata or {}
        }
        self.outputs_file.write(json.dumps(record) + "\n")
        self.outputs_file.flush()
        
    def close(self):
        self.actions_file.close()
        self.outputs_file.close()


class SimulationEnvironment:
    """
    Main simulation environment for multi-agent forecasting.
    
    Uses Metaculus-style scoring:
    - Log score for accuracy
    - Peer score relative to other agents
    - Time-weighted averaging across prediction history
    """
    
    def __init__(self, 
                 dataset_name: str, 
                 start_date: date,
                 end_date: date,
                 context_dir: str,
                 inference_provider=None,
                 output_dir: str = ".",
                 resolution_start: date = None,
                 resolution_end: date = None):
        self.current_date = start_date
        self.end_date = end_date
        self.context_dir = context_dir
        
        # Components - filter questions to resolution window
        self.q_pool = QuestionPool(
            dataset_name,
            resolution_start=resolution_start,
            resolution_end=resolution_end
        )
        self.matcher = AnswerMatcher(inference_provider) if inference_provider else None
        
        # Track prediction history per question
        self.prediction_histories: Dict[str, PredictionHistory] = {}
        
        # Current aggregate per question (updated end of each day)
        self.current_aggregates: Dict[str, Dict[str, float]] = {}
        
        self.agents = []
        self.agent_scores: Dict[str, float] = {}  # Cumulative scores
        
        # Track prediction outcomes for summary
        self.agent_correct: Dict[str, int] = {}  # Count of positive scores
        self.agent_wrong: Dict[str, int] = {}    # Count of negative scores
        self.agent_questions: Dict[str, int] = {}  # Total questions predicted on
        
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
        
    def run(self):
        print(f"Starting simulation from {self.current_date} to {self.end_date}")
        print(f"Total questions: {self.q_pool.total_count}")
        
        while self.current_date <= self.end_date:
            print(f"--- Day {self.current_date} (Active: {self.q_pool.active_count}) ---")
            self.step()
            self.current_date += timedelta(days=1)
        
        self.logger.close()
        print("\nSimulation ended.")
        print("\nFinal Scores:")
        for agent_id, score in sorted(self.agent_scores.items()):
            correct = self.agent_correct.get(agent_id, 0)
            wrong = self.agent_wrong.get(agent_id, 0)
            total = self.agent_questions.get(agent_id, 0)
            print(f"  {agent_id}: {score:+.2f} ({correct} correct, {wrong} wrong, {total} total predictions)")

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
        doc_interface = SimDocInterface(self.context_dir, self.current_date)
        forecast_interface = SimForecastInterface(
            active_questions, 
            self.current_aggregates,
            self.prediction_histories,
            self.current_date,
            self.logger,
            resolved_questions=self.resolved_questions
        )
        
        for agent in self.agents:
            forecast_interface.set_agent_context(agent.agent_id)
            agent.act(doc_interface, forecast_interface, self.current_date)
        
        # 4. Update aggregates (end of day)
        self._update_aggregates(active_questions)
            
    def _resolve_question(self, q: Question):
        """Resolve a question and compute final scores."""
        history = self.prediction_histories.get(q.qid)
        if not history or not history.predictions:
            return
            
        # Resolve and compute scores
        result = resolve_question(history, q.ground_truth_answer, self.matcher)
        
        # Update cumulative scores
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
    """Agent's interface to forecasting questions."""
    
    def __init__(self, 
                 questions: List[Question],
                 aggregates: Dict[str, Dict[str, float]],
                 histories: Dict[str, PredictionHistory],
                 sim_date: date,
                 logger: SimLogger,
                 resolved_questions: List[Question] = None):
        self.questions = {q.qid: q for q in questions}
        self.aggregates = aggregates
        self.histories = histories
        self.sim_date = sim_date
        self.logger = logger
        self.current_agent_id: Optional[str] = None
        self.resolved_questions = resolved_questions or []
        
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
        """Record a probabilistic prediction."""
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
        
        # Add to history
        history = self.histories.get(prediction.question_id)
        if history:
            history.add_prediction(daily_pred)
            
        # Log
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
            sample = str(df[col].iloc[0]) if len(df) > 0 else "N/A"
            if len(sample) > 50:
                sample = sample[:50] + "..."
            columns_desc.append(f"- {col} ({dtype}): e.g. {sample}")
        
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
            
            # Get agent's current prediction if any
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
                'market_aggregate': json.dumps(agg) if agg else None,
                'num_predictions': num_preds,
                'my_prediction': json.dumps(my_pred) if my_pred else None,
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

