"""Tests for the count_articles feature across all layers:
- BaseSearchTool.count_articles (default)
- LanceDBSearchTool.count_articles (with mock table)
- SearchHandler.count_articles (passthrough)
- BasicAgent._build_cadence_section (integration with count)
"""

from datetime import date, timedelta
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from agents.search_tools.base import BaseSearchTool, SearchResult
from agents.basicAgent.search import SearchHandler
from agents.basicAgent.config import AgentConfig


# ---------------------------------------------------------------------------
# Helpers / Fakes
# ---------------------------------------------------------------------------

class FakeSearchTool(BaseSearchTool):
    """Minimal concrete BaseSearchTool that does NOT override count_articles."""

    @property
    def is_available(self) -> bool:
        return True

    def search(self, query, max_results=10, max_date=None, search_type="hybrid", min_date=None):
        return []

    def get_article(self, article_id):
        return None


class CountableSearchTool(FakeSearchTool):
    """Search tool with count_articles that records calls and returns a preset value."""

    def __init__(self, count_return: Optional[int] = 42):
        self._count_return = count_return
        self.count_calls: list = []

    def count_articles(self, min_date=None, max_date=None):
        self.count_calls.append({"min_date": min_date, "max_date": max_date})
        return self._count_return


class UnavailableSearchTool(BaseSearchTool):
    """Search tool that reports itself as unavailable."""

    @property
    def is_available(self) -> bool:
        return False

    def search(self, query, max_results=10, max_date=None, search_type="hybrid", min_date=None):
        return []

    def get_article(self, article_id):
        return None

    def count_articles(self, min_date=None, max_date=None):
        raise AssertionError("Should never be called when unavailable")


class ExplodingCountSearchTool(FakeSearchTool):
    """count_articles raises an exception."""

    def count_articles(self, min_date=None, max_date=None):
        raise RuntimeError("DB connection lost")


# ---------------------------------------------------------------------------
# 1. BaseSearchTool default implementation
# ---------------------------------------------------------------------------

class TestBaseSearchToolCountDefault:
    def test_default_returns_none(self):
        tool = FakeSearchTool()
        assert tool.count_articles() is None

    def test_default_returns_none_with_dates(self):
        tool = FakeSearchTool()
        assert tool.count_articles(min_date=date(2025, 1, 1), max_date=date(2025, 3, 1)) is None


# ---------------------------------------------------------------------------
# 2. LanceDBSearchTool.count_articles (mocked table)
# ---------------------------------------------------------------------------

try:
    import lancedb as _lancedb  # noqa: F401
    _has_lancedb = True
except ImportError:
    _has_lancedb = False


@pytest.mark.skipif(not _has_lancedb, reason="lancedb not installed")
class TestLanceDBCountArticles:
    def _mk_tool(self, table, available=True):
        from agents.search_tools.lancedb.store import LanceDBSearchTool
        tool = LanceDBSearchTool.__new__(LanceDBSearchTool)
        tool._available = available
        tool._table = table
        tool._embedding_model = None
        tool._model_loaded = False
        tool._model_path = None
        tool._db_path = ""
        tool._db = None
        tool._config = {}
        tool._chunk_tokens = 512
        return tool

    def test_no_date_filters(self):
        table = MagicMock()
        table.count_rows.return_value = 100
        tool = self._mk_tool(table)

        result = tool.count_articles()
        assert result == 100
        table.count_rows.assert_called_once_with()

    def test_min_date_only(self):
        table = MagicMock()
        table.count_rows.return_value = 50
        tool = self._mk_tool(table)

        result = tool.count_articles(min_date=date(2025, 4, 1))
        assert result == 50
        where_arg = table.count_rows.call_args[0][0]
        assert "2025-04-01T00:00:00" in where_arg
        assert "date >=" in where_arg

    def test_max_date_only(self):
        table = MagicMock()
        table.count_rows.return_value = 75
        tool = self._mk_tool(table)

        result = tool.count_articles(max_date=date(2025, 6, 15))
        assert result == 75
        where_arg = table.count_rows.call_args[0][0]
        assert "2025-06-15T23:59:59" in where_arg
        assert "date <=" in where_arg

    def test_both_dates(self):
        table = MagicMock()
        table.count_rows.return_value = 30
        tool = self._mk_tool(table)

        result = tool.count_articles(min_date=date(2025, 3, 1), max_date=date(2025, 3, 31))
        assert result == 30
        where_arg = table.count_rows.call_args[0][0]
        assert "2025-03-01T00:00:00" in where_arg
        assert "2025-03-31T23:59:59" in where_arg
        assert " AND " in where_arg

    def test_returns_zero(self):
        table = MagicMock()
        table.count_rows.return_value = 0
        tool = self._mk_tool(table)

        assert tool.count_articles(min_date=date(2099, 1, 1)) == 0

    def test_unavailable_returns_none(self):
        tool = self._mk_tool(table=None, available=False)
        assert tool.count_articles() is None

    def test_exception_returns_none(self):
        table = MagicMock()
        table.count_rows.side_effect = RuntimeError("corrupt index")
        tool = self._mk_tool(table)

        assert tool.count_articles(min_date=date(2025, 1, 1)) is None

    def test_same_min_max_date(self):
        """Single-day range should still produce a valid where clause."""
        table = MagicMock()
        table.count_rows.return_value = 5
        tool = self._mk_tool(table)

        result = tool.count_articles(min_date=date(2025, 5, 10), max_date=date(2025, 5, 10))
        assert result == 5
        where_arg = table.count_rows.call_args[0][0]
        assert "2025-05-10T00:00:00" in where_arg
        assert "2025-05-10T23:59:59" in where_arg


# ---------------------------------------------------------------------------
# 3. SearchHandler.count_articles
# ---------------------------------------------------------------------------

class TestSearchHandlerCount:
    def test_passes_dates_through(self):
        tool = CountableSearchTool(count_return=123)
        handler = SearchHandler(search_tool=tool)

        result = handler.count_articles(min_date=date(2025, 4, 1), max_date=date(2025, 4, 10))
        assert result == 123
        assert len(tool.count_calls) == 1
        assert tool.count_calls[0] == {"min_date": date(2025, 4, 1), "max_date": date(2025, 4, 10)}

    def test_no_dates(self):
        tool = CountableSearchTool(count_return=999)
        handler = SearchHandler(search_tool=tool)

        result = handler.count_articles()
        assert result == 999
        assert tool.count_calls[0] == {"min_date": None, "max_date": None}

    def test_returns_none_when_tool_unavailable(self):
        tool = UnavailableSearchTool()
        handler = SearchHandler(search_tool=tool)

        assert handler.count_articles(min_date=date(2025, 1, 1)) is None

    def test_returns_none_when_no_search_tool(self):
        handler = SearchHandler(search_tool=None)
        assert handler.count_articles() is None

    def test_returns_none_when_tool_default(self):
        """Tool that doesn't override count_articles returns None."""
        tool = FakeSearchTool()
        handler = SearchHandler(search_tool=tool)

        assert handler.count_articles(min_date=date(2025, 1, 1)) is None

    def test_returns_zero(self):
        tool = CountableSearchTool(count_return=0)
        handler = SearchHandler(search_tool=tool)

        assert handler.count_articles() == 0


# ---------------------------------------------------------------------------
# 4. BasicAgent._build_cadence_section integration
# ---------------------------------------------------------------------------

def _mk_agent(search_tool=None, search_cutoff_days=0, timegap_days=3,
              last_active=None, next_active=None):
    """Build a BasicAgent with mocked internals for cadence section testing."""
    from agents.basicAgent.agent import BasicAgent

    config = AgentConfig(
        search_cutoff_days=search_cutoff_days,
        timegap_days=timegap_days,
        enable_memory=False,
    )
    # Bypass __init__ to avoid needing a real inference provider
    agent = BasicAgent.__new__(BasicAgent)
    agent.config = config
    agent._search_handler = SearchHandler(search_tool=search_tool, search_cutoff_days=search_cutoff_days)

    # Mock forecast interface for last/next active dates
    fi = MagicMock()
    fi.last_active_date = last_active
    fi.next_active_date = next_active
    agent._forecast_interface = fi

    return agent


class TestBuildCadenceSection:
    def test_count_shown_when_available(self):
        tool = CountableSearchTool(count_return=1500)
        agent = _mk_agent(
            search_tool=tool,
            timegap_days=3,
            last_active=date(2025, 4, 7),
            next_active=date(2025, 4, 13),
        )
        result = agent._build_cadence_section(date(2025, 4, 10))

        assert "1,500 new articles" in result
        assert "search tool" in result
        # Verify correct date range was queried
        assert len(tool.count_calls) == 1
        assert tool.count_calls[0]["min_date"] == date(2025, 4, 7)
        assert tool.count_calls[0]["max_date"] == date(2025, 4, 10)

    def test_count_with_search_cutoff_days(self):
        tool = CountableSearchTool(count_return=800)
        agent = _mk_agent(
            search_tool=tool,
            search_cutoff_days=2,
            timegap_days=5,
            last_active=date(2025, 4, 5),
            next_active=date(2025, 4, 15),
        )
        result = agent._build_cadence_section(date(2025, 4, 10))

        assert "800 new articles" in result
        # max_date should be current_date - cutoff = Apr 10 - 2 = Apr 8
        assert tool.count_calls[0]["max_date"] == date(2025, 4, 8)
        assert tool.count_calls[0]["min_date"] == date(2025, 4, 5)

    def test_fallback_text_when_no_last_active(self):
        """First update: no last_active, so count is not attempted."""
        tool = CountableSearchTool(count_return=999)
        agent = _mk_agent(
            search_tool=tool,
            timegap_days=3,
            last_active=None,
            next_active=date(2025, 4, 13),
        )
        result = agent._build_cadence_section(date(2025, 4, 10))

        assert "This is your first update." in result
        assert "New articles have been published" not in result
        assert "999" not in result
        assert len(tool.count_calls) == 0  # count never called

    def test_fallback_text_when_search_unavailable(self):
        tool = UnavailableSearchTool()
        agent = _mk_agent(
            search_tool=tool,
            timegap_days=3,
            last_active=date(2025, 4, 7),
            next_active=date(2025, 4, 13),
        )
        result = agent._build_cadence_section(date(2025, 4, 10))

        assert "New articles have been published" in result
        assert "," not in result.split("articles")[0]  # no number

    def test_fallback_text_when_no_search_tool(self):
        agent = _mk_agent(
            search_tool=None,
            timegap_days=3,
            last_active=date(2025, 4, 7),
            next_active=date(2025, 4, 13),
        )
        result = agent._build_cadence_section(date(2025, 4, 10))

        assert "New articles have been published" in result

    def test_fallback_text_when_count_returns_none(self):
        """Tool doesn't support count_articles (default BaseSearchTool)."""
        tool = FakeSearchTool()
        agent = _mk_agent(
            search_tool=tool,
            timegap_days=3,
            last_active=date(2025, 4, 7),
            next_active=date(2025, 4, 13),
        )
        result = agent._build_cadence_section(date(2025, 4, 10))

        assert "New articles have been published" in result

    def test_zero_count_still_shown(self):
        """Zero articles is a valid count and should be displayed."""
        tool = CountableSearchTool(count_return=0)
        agent = _mk_agent(
            search_tool=tool,
            timegap_days=3,
            last_active=date(2025, 4, 7),
            next_active=date(2025, 4, 13),
        )
        result = agent._build_cadence_section(date(2025, 4, 10))

        assert "0 new articles" in result

    def test_large_count_has_comma_formatting(self):
        tool = CountableSearchTool(count_return=1234567)
        agent = _mk_agent(
            search_tool=tool,
            timegap_days=7,
            last_active=date(2025, 4, 3),
            next_active=date(2025, 4, 17),
        )
        result = agent._build_cadence_section(date(2025, 4, 10))

        assert "1,234,567 new articles" in result

    def test_cadence_section_still_has_standard_fields(self):
        """Ensure the rest of the cadence section is intact."""
        tool = CountableSearchTool(count_return=50)
        agent = _mk_agent(
            search_tool=tool,
            timegap_days=3,
            last_active=date(2025, 4, 7),
            next_active=date(2025, 4, 13),
        )
        result = agent._build_cadence_section(date(2025, 4, 10))

        assert "## UPDATE CADENCE" in result
        assert "every 3 days" in result
        assert "Last update: 2025-04-07" in result
        assert "Current date: 2025-04-10" in result
        assert "Next scheduled update: 2025-04-13" in result

    def test_first_update_shows_first_update_text(self):
        agent = _mk_agent(
            search_tool=None,
            timegap_days=1,
            last_active=None,
            next_active=date(2025, 4, 11),
        )
        result = agent._build_cadence_section(date(2025, 4, 10))

        assert "This is your first update." in result

    def test_search_cutoff_equals_timegap(self):
        """Edge case: cutoff == timegap means max_date could equal last_active."""
        tool = CountableSearchTool(count_return=10)
        agent = _mk_agent(
            search_tool=tool,
            search_cutoff_days=3,
            timegap_days=3,
            last_active=date(2025, 4, 7),
            next_active=date(2025, 4, 13),
        )
        result = agent._build_cadence_section(date(2025, 4, 10))

        assert "10 new articles" in result
        # max_date = Apr 10 - 3 = Apr 7, which equals last_active (min_date)
        assert tool.count_calls[0]["min_date"] == date(2025, 4, 7)
        assert tool.count_calls[0]["max_date"] == date(2025, 4, 7)

    def test_search_cutoff_larger_than_timegap(self):
        """Edge case: cutoff > timegap means max_date < last_active."""
        tool = CountableSearchTool(count_return=0)
        agent = _mk_agent(
            search_tool=tool,
            search_cutoff_days=5,
            timegap_days=3,
            last_active=date(2025, 4, 7),
            next_active=date(2025, 4, 13),
        )
        result = agent._build_cadence_section(date(2025, 4, 10))

        # max_date = Apr 10 - 5 = Apr 5, which is BEFORE last_active Apr 7
        # The count still gets called (we don't guard this in agent code)
        assert tool.count_calls[0]["min_date"] == date(2025, 4, 7)
        assert tool.count_calls[0]["max_date"] == date(2025, 4, 5)
