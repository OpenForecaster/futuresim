import os
import json
from datetime import date, timedelta
from typing import List, Dict, Any, Optional

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
                 output_dir: str = "."):
        self.current_date = start_date
        self.end_date = end_date
        self.context_dir = context_dir
        
        # Components
        self.q_pool = QuestionPool(dataset_name)
        self.matcher = AnswerMatcher(inference_provider) if inference_provider else None
        
        # Track prediction history per question
        self.prediction_histories: Dict[str, PredictionHistory] = {}
        
        # Current aggregate per question (updated end of each day)
        self.current_aggregates: Dict[str, Dict[str, float]] = {}
        
        self.agents = []
        self.agent_scores: Dict[str, float] = {}  # Cumulative scores
        
        # Logging
        self.logger = SimLogger(output_dir)
        
    def add_agent(self, agent):
        self.agents.append(agent)
        self.agent_scores[agent.agent_id] = 0.0
        
    def run(self):
        print(f"Starting simulation from {self.current_date} to {self.end_date}")
        print(f"Total questions: {self.q_pool.total_count}")
        
        while self.current_date <= self.end_date:
            print(f"--- Day {self.current_date} (Active: {self.q_pool.active_count}) ---")
            self.step()
            self.current_date += timedelta(days=1)
        
        self.logger.close()
        print("\nSimulation ended.")
        print("Final Scores:")
        for agent_id, score in sorted(self.agent_scores.items()):
            print(f"  {agent_id}: {score:.2f}")

    def step(self):
        # 1. Resolve questions expiring today
        resolving = self.q_pool.pop_resolving(self.current_date)
        for q in resolving:
            self._resolve_question(q)
            
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
            self.logger
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
                 logger: SimLogger):
        self.questions = {q.qid: q for q in questions}
        self.aggregates = aggregates
        self.histories = histories
        self.sim_date = sim_date
        self.logger = logger
        self.current_agent_id: Optional[str] = None
        
    def set_agent_context(self, agent_id: str):
        self.current_agent_id = agent_id

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
