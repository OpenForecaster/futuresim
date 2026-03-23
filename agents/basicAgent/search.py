"""BasicAgent search handling."""

from datetime import date, timedelta
from typing import List, Optional, Tuple

from agents.search_tools.base import BaseSearchTool, SearchResult


class SearchHandler:
    """Wraps search tool for BasicAgent."""
    
    
    def __init__(
        self,
        search_tool: Optional[BaseSearchTool] = None,
        snippet_max_chars: int = 2000,
        article_max_chars: int = 4000,
        search_cutoff_days: int = 0,
        max_search_date: Optional[date] = None,
    ):
        self._search_tool = search_tool
        self._current_date: Optional[date] = None
        self._chunk_max_chars = snippet_max_chars
        self._search_cutoff_days = search_cutoff_days
        # Optional calendar ceiling on the latest article date (min with sim_date - cutoff). Useful for fixed "as of" evals.
        self._max_search_date = max_search_date
    
    @property
    def is_available(self) -> bool:
        return self._search_tool is not None and self._search_tool.is_available

    @property
    def chunk_tokens(self) -> Optional[int]:
        if self._search_tool is None:
            return None
        return getattr(self._search_tool, "chunk_tokens", None)
    
    def set_date(self, current_date: date) -> None:
        self._current_date = current_date
    
    def search(
        self,
        query: str,
        max_results: int = 5,
        search_type: str = "hybrid",
        min_date: Optional[date] = None,
        max_date: Optional[date] = None,
    ) -> Tuple[str, Optional[str]]:
        if not self.is_available:
            return "", "Search not available"
        if not self._current_date:
            return "", "Current date not set"
        
        # Never search past the simulation day (+ optional cutoff) to prevent leakage.
        allowed_max_date = self._current_date - timedelta(days=self._search_cutoff_days)
        if self._max_search_date is not None:
            allowed_max_date = min(allowed_max_date, self._max_search_date)
        effective_max_date = allowed_max_date
        info_lines = []
        if max_date is not None:
            effective_max_date = min(max_date, allowed_max_date)
            if max_date > allowed_max_date:
                info_lines.append(
                    f"Note: maximum allowed search date is {allowed_max_date.isoformat()}. "
                    f"Your requested to-date {max_date.isoformat()} was capped."
                )
        
        # Validate min_date doesn't exceed max_date (prevent leakage)
        if min_date and min_date > effective_max_date:
            info_lines.append(
                f"Note: your requested from-date {min_date.isoformat()} is after the effective to-date "
                f"{effective_max_date.isoformat()}, so from-date was ignored."
            )
            min_date = None  # Ignore invalid min_date
        
        try:
            results = self._search_tool.search(
                query, max_results, effective_max_date, search_type, min_date
            )
            if not results:
                base = "No articles found matching your query."
                if info_lines:
                    return "\n".join(info_lines + ["", base]), None
                return base, None
            formatted = self._format_results(results)
            if info_lines:
                return "\n".join(info_lines + ["", formatted]), None
            return formatted, None
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
            lines.append(r.snippet)  # Full chunk content; chunk size comes from the search backend config.
            lines.append("")
        return "\n".join(lines)
