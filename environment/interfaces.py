from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import date
from dataclasses import dataclass, field


@dataclass
class ArticleMeta:
    id: str
    title: str
    source: str
    published_date: date
    url: str


@dataclass
class QuestionView:
    """Agent's view of a forecasting question."""
    id: str
    title: str
    background: str
    resolution_criteria: str
    answer_type: str
    resolution_date: date
    # Current aggregate probabilities (from all agents' predictions)
    aggregate: Dict[str, float] = field(default_factory=dict)


@dataclass
class PredictionSubmission:
    """Agent's prediction on a question."""
    question_id: str
    outcomes: Dict[str, float]  # {outcome_str: probability}, should sum to <= 1


class AgentInterface(ABC):
    """
    Interface for agents to interact with the document environment.
    """
    
    @abstractmethod
    def list_sources(self) -> List[str]:
        """List all available news sources."""
        pass
    
    @abstractmethod
    def list_dates(self, source: Optional[str] = None) -> List[date]:
        """List available dates, optionally filtered by source."""
        pass
    
    @abstractmethod
    def list_articles(self, 
                     date_obj: date, 
                     source: Optional[str] = None) -> List[ArticleMeta]:
        """List articles for a specific date and optional source."""
        pass
    
    @abstractmethod
    def read_article(self, article_id: str) -> str:
        """Get the full content of an article."""
        pass


class ForecastInterface(ABC):
    """
    Interface for agents to interact with forecasting questions.
    Predictions are probability distributions over outcomes.
    """
    
    @abstractmethod
    def list_questions(self) -> List[QuestionView]:
        """
        List active forecasting questions with current aggregate probabilities.
        Aggregate is frozen at start of day.
        """
        pass
    
    @abstractmethod
    def submit_prediction(self, prediction: PredictionSubmission) -> None:
        """
        Submit a probabilistic prediction.
        
        Args:
            prediction: Contains question_id and {outcome: probability} dict.
                       Probabilities should sum to <= 1.
                       Remaining mass is implicit "Other".
        """
        pass
    
    def log_model_output(self, prompt: str, response: str, 
                         metadata: Optional[Dict[str, Any]] = None):
        """Log model prompt/response for debugging."""
        pass
