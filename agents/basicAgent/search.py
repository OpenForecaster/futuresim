"""BasicAgent search handling."""

from datetime import date
from typing import List, Optional, Tuple

from agents.search_tools.base import BaseSearchTool, SearchResult


class SearchHandler:
    """Wraps search tool for BasicAgent."""
    
    def __init__(self, search_tool: Optional[BaseSearchTool] = None,
                 snippet_max_chars: int = 2000, article_max_chars: int = 4000):
        self._search_tool = search_tool
        self._current_date: Optional[date] = None
        self._chunk_max_chars = snippet_max_chars
    
    @property
    def is_available(self) -> bool:
        return self._search_tool is not None and self._search_tool.is_available
    
    def set_date(self, current_date: date) -> None:
        self._current_date = current_date
    
    def search(self, query: str, max_results: int = 5, search_type: str = "hybrid",
                min_date: Optional[date] = None) -> Tuple[str, Optional[str]]:
        if not self.is_available:
            return "", "Search not available"
        if not self._current_date:
            return "", "Current date not set"
        
        # Validate min_date doesn't exceed current date (prevent future leakage)
        if min_date and min_date > self._current_date:
            min_date = None  # Ignore invalid min_date
        
        try:
            results = self._search_tool.search(
                query, max_results, self._current_date, search_type, min_date
            )
            if not results:
                return "No articles found matching your query.", None
            return self._format_results(results), None
        except Exception as e:
            return "", f"Search error: {e}"
    
    def _format_results(self, results: List[SearchResult]) -> str:
        lines = [f"Found {len(results)} relevant article chunk(s):\n"]
        for i, r in enumerate(results, 1):
            # Format header with headline and metadata
            lines.append(f"═══ [{i}] ═══════════════════════════════════════")
            lines.append(f"HEADLINE: {r.title}")
            lines.append(f"SOURCE: {r.source}")
            lines.append(f"PUBLISHED: {r.date_publish or 'Unknown'} | DOWNLOADED: {r.date}")
            if r.url:
                lines.append(f"URL: {r.url}")
            lines.append("")
            lines.append(r.snippet)  # Full chunk content (already 512 tokens max)
            lines.append("")
        return "\n".join(lines)
