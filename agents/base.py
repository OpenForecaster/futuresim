from abc import ABC, abstractmethod
from datetime import date
from typing import Any, List, Dict, Optional


class BaseAgent(ABC):
    """Base class for forecasting agents."""
    
    def __init__(self, 
                 agent_id: str, 
                 inference_provider: Optional[Any] = None,
                 model_name: str = ""):
        self.agent_id = agent_id
        self.inference = inference_provider
        self.model_name = model_name
        
    @abstractmethod
    def act(self, 
            doc_interface: 'AgentInterface', 
            forecast_interface: 'ForecastInterface',
            current_date: date) -> List[Dict[str, Any]]:
        """
        Execute agent logic for the day.
        
        Agent should:
        1. Read questions via forecast_interface.list_questions()
        2. Optionally read context via doc_interface
        3. Submit predictions via forecast_interface.submit_prediction()
        
        Returns list of actions taken (for debugging).
        """
        pass
