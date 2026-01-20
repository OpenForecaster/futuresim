"""BasicAgent configuration."""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class AgentConfig:
    max_actions: int = 10
    max_submit_retries: int = 3
    max_outcomes_per_question: int = 5
    memory_dir: Optional[str] = None
    sampling_params: Optional[Dict[str, Any]] = None
    
    # Search
    search_enabled: bool = False
    max_search_results: int = 5
    snippet_max_chars: int = 2000
    article_max_chars: int = 4000
    search_cutoff_days: int = 0
    
    def __post_init__(self):
        if self.sampling_params is None:
            self.sampling_params = {'temperature': 0.7, 'max_tokens': 2048}
