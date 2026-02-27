from datetime import date

from agents.basicAgent.search import SearchHandler
from agents.search_tools.base import BaseSearchTool, SearchResult


class DummySearchTool(BaseSearchTool):
    def __init__(self):
        self.calls = []
        self._return_by_query = {}

    @property
    def is_available(self) -> bool:
        return True

    def set_results(self, query: str, results):
        self._return_by_query[query] = results

    def search(self, query, max_results=10, max_date=None, search_type="hybrid", min_date=None):
        self.calls.append(
            {
                "query": query,
                "max_results": max_results,
                "max_date": max_date,
                "search_type": search_type,
                "min_date": min_date,
            }
        )
        return self._return_by_query.get(query, [])

    def get_article(self, article_id: str):
        return None


def _mk_result(title: str) -> SearchResult:
    return SearchResult(
        article_id="a1",
        title=title,
        source="Test",
        date=date(2025, 4, 1),
        date_publish=date(2025, 4, 1),
        snippet="Snippet",
        score=1.0,
        url="https://example.com",
    )


def test_search_clamps_max_date_to_current_day():
    tool = DummySearchTool()
    tool.set_results("query", [_mk_result("Hit")])

    handler = SearchHandler(search_tool=tool, search_cutoff_days=0)
    handler.set_date(date(2025, 4, 24))

    result, error = handler.search("query", max_date=date(2025, 5, 10))
    assert error is None
    assert "maximum allowed search date is 2025-04-24" in result
    assert "requested to-date 2025-05-10 was capped" in result
    assert "Found 1 relevant article chunk(s):" in result
    assert tool.calls[0]["max_date"] == date(2025, 4, 24)


def test_search_uses_user_to_date_when_earlier_than_current_day():
    tool = DummySearchTool()
    tool.set_results("query", [_mk_result("Hit")])

    handler = SearchHandler(search_tool=tool, search_cutoff_days=0)
    handler.set_date(date(2025, 4, 24))

    result, error = handler.search("query", max_date=date(2025, 4, 10))
    assert error is None
    assert "Found 1 relevant article chunk(s):" in result
    assert tool.calls[0]["max_date"] == date(2025, 4, 10)


def test_search_clears_min_date_if_invalid_for_effective_max():
    tool = DummySearchTool()
    tool.set_results("query", [_mk_result("Hit")])

    handler = SearchHandler(search_tool=tool, search_cutoff_days=0)
    handler.set_date(date(2025, 4, 24))

    handler.search(
        "query",
        min_date=date(2025, 4, 20),
        max_date=date(2025, 4, 10),
    )
    assert tool.calls[0]["min_date"] is None
    assert tool.calls[0]["max_date"] == date(2025, 4, 10)


def test_search_returns_no_articles_message_when_empty():
    tool = DummySearchTool()

    handler = SearchHandler(search_tool=tool, search_cutoff_days=0)
    handler.set_date(date(2025, 4, 24))

    result, error = handler.search("no-match query")
    assert error is None
    assert result == "No articles found matching your query."
    assert len(tool.calls) == 1
    assert tool.calls[0]["query"] == "no-match query"


def test_search_returns_future_window_note_even_when_no_hits():
    tool = DummySearchTool()
    handler = SearchHandler(search_tool=tool, search_cutoff_days=0)
    handler.set_date(date(2025, 4, 24))

    result, error = handler.search(
        "no-match query",
        min_date=date(2025, 4, 25),
        max_date=date(2025, 5, 10),
    )

    assert error is None
    assert "maximum allowed search date is 2025-04-24" in result
    assert "requested to-date 2025-05-10 was capped" in result
    assert "requested from-date 2025-04-25 is after the effective to-date 2025-04-24" in result
    assert "No articles found matching your query." in result
