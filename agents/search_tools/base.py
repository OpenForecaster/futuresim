"""Search tool abstractions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import List, Optional


@dataclass
class SearchResult:
    article_id: str
    title: str
    source: str
    date: Optional[date]  # Download date
    date_publish: Optional[date] = None  # Article publish date
    snippet: str = ""
    score: float = 0.0
    url: str = ""


@dataclass
class Article:
    id: str
    title: str
    source: str
    date: Optional[date]
    content: str
    url: str = ""
    description: str = ""


class BaseSearchTool(ABC):
    """Abstract base for search tools."""
    
    @abstractmethod
    def search(
        self, 
        query: str, 
        max_results: int = 10, 
        max_date: Optional[date] = None,
        search_type: str = "hybrid",
        min_date: Optional[date] = None,
    ) -> List[SearchResult]:
        pass
    
    @abstractmethod
    def get_article(self, article_id: str) -> Optional[Article]:
        pass
    
    def count_articles(self, min_date: Optional[date] = None, max_date: Optional[date] = None) -> Optional[int]:
        """Count articles in a date range. Returns None if not supported."""
        return None

    @property
    @abstractmethod
    def is_available(self) -> bool:
        pass
