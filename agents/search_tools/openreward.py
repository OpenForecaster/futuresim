"""OpenReward time-gated news search backend."""

from __future__ import annotations

import asyncio
import os
import re
import urllib.parse
from datetime import date, datetime
from typing import Any, Optional

from .base import Article, BaseSearchTool, SearchResult


DEFAULT_SEARCH_URL = "https://search.openreward.ai/search"
DEFAULT_FETCH_URL = "https://search.openreward.ai/fetch"
DEFAULT_DOMAIN_INFO_URL = "https://search.openreward.ai/policy/domain_info"
SEARCH_RESULT_TEXT_CHARS = 5000


def _parse_date(value: Any) -> Optional[date]:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None


def _clean_text(value: Any) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clip_text(text: str, max_chars: int) -> str:
    text = _clean_text(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    clipped = text[:max_chars].rsplit(" ", 1)[0].rstrip()
    return f"{clipped}\n\n...[truncated]"


def _hostname(url: str) -> str:
    return urllib.parse.urlparse(url).netloc


def _run_async(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    coro.close()
    raise RuntimeError("Use OpenRewardSdkSearchTool inside async code.")


class OpenRewardSdkSearchTool:
    """Async SDK-backed search that keeps Futuresim's combined search_news shape."""

    def __init__(
        self,
        *,
        api_key: str,
        search_url: str = "",
        fetch_url: str = "",
        max_snippet_chars: int = SEARCH_RESULT_TEXT_CHARS,
    ):
        from openreward.web_service import WebServiceConfig

        self._api_key = api_key
        self.max_snippet_chars = max_snippet_chars
        truthy = {"1", "true", "yes", "on"}
        self.config = WebServiceConfig(
            api_key=api_key,
            search_url=search_url
            or os.environ.get("OPENREWARD_WEB_SEARCH_URL")
            or os.environ.get("FSIM_OPENREWARD_SEARCH_URL")
            or DEFAULT_SEARCH_URL,
            fetch_url=fetch_url
            or os.environ.get("OPENREWARD_WEB_FETCH_URL")
            or os.environ.get("FSIM_OPENREWARD_FETCH_URL")
            or DEFAULT_FETCH_URL,
            domain_info_url=os.environ.get(
                "OPENREWARD_WEB_DOMAIN_INFO_URL",
                DEFAULT_DOMAIN_INFO_URL,
            ),
            skip_preflight=os.environ.get("OPENREWARD_WEB_SKIP_PREFLIGHT", "").lower() in truthy,
            preapproved_hosts=frozenset(
                host.strip().lower()
                for host in os.environ.get("OPENREWARD_WEB_PREAPPROVED_HOSTS", "").split(",")
                if host.strip()
            ),
            default_as_of=os.environ.get("OPENREWARD_WEB_AS_OF", "").strip() or None,
        )

    @classmethod
    def from_env(
        cls,
        api_key: str = "",
        search_url: str = "",
        fetch_url: str = "",
    ) -> "OpenRewardSdkSearchTool":
        key = api_key or os.environ.get("OPENREWARD_API_KEY", "") or os.environ.get("OR_TOKEN", "")
        return cls(api_key=key, search_url=search_url, fetch_url=fetch_url)

    @property
    def is_available(self) -> bool:
        return bool(self._api_key)

    async def search(
        self,
        query: str,
        max_results: int = 5,
        max_date: Optional[date] = None,
        search_type: str = "hybrid",
        min_date: Optional[date] = None,
    ) -> list[SearchResult]:
        from openreward.tools import Backfetch, Backsearch

        if not self.is_available:
            return []
        if search_type != "hybrid":
            raise ValueError("OpenReward search is configured for hybrid mode only.")

        as_of = max_date or date.today()
        k = min(max(1, int(max_results or 5)), 5)
        result = await Backsearch(config=self.config, k=k).run(query, as_of=as_of.isoformat())
        if not result.ok:
            raise RuntimeError(result.output)

        hits = (result.data or {}).get("hits") or []
        if not isinstance(hits, list):
            return []

        fetcher = Backfetch(config=self.config)
        results: list[SearchResult] = []
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            results.append(await self._hit_to_result(hit, fetcher, as_of))
            if len(results) >= k:
                break
        return results

    async def get_article(self, article_id: str) -> Optional[Article]:
        if not self.is_available or not article_id.startswith(("http://", "https://")):
            return None

        from openreward.tools import Backfetch

        page = await Backfetch(config=self.config).run(
            article_id,
            "Extract the full article text.",
            as_of=date.today().isoformat(),
        )
        if not page.ok:
            return None

        data = page.data or {}
        content = _clean_text(data.get("content") or page.output)
        return Article(
            id=article_id,
            title=str(data.get("title") or ""),
            source=str(data.get("source") or data.get("host") or _hostname(article_id)),
            date=_parse_date(data.get("crawl_date") or data.get("date")),
            content=content,
            url=article_id,
            description=str(data.get("description") or ""),
        )

    async def _hit_to_result(self, hit: dict[str, Any], fetcher: Any, as_of: date) -> SearchResult:
        url = str(hit.get("url") or hit.get("link") or "")
        fetched: dict[str, Any] = {}
        text = _clean_text(hit.get("snippet") or hit.get("text") or hit.get("content"))
        if url:
            page = await fetcher.run(
                url,
                "Extract the article text relevant to this forecasting question.",
                as_of=as_of.isoformat(),
            )
            if page.ok:
                fetched = page.data or {}
                text = _clean_text(fetched.get("content") or page.output or text)
            elif not text:
                text = page.output

        title = str(fetched.get("title") or hit.get("title") or "")
        crawl_date = _parse_date(
            fetched.get("crawl_date") or fetched.get("date") or hit.get("crawl_date") or hit.get("date")
        )
        publish_date = _parse_date(
            fetched.get("publish_date")
            or fetched.get("date_publish")
            or hit.get("publish_date")
            or hit.get("date_publish")
        )
        source = str(
            fetched.get("source")
            or fetched.get("host")
            or hit.get("source")
            or hit.get("host")
            or _hostname(url)
        )
        return SearchResult(
            article_id=str(hit.get("id") or hit.get("article_id") or url),
            title=title,
            source=source,
            date=crawl_date,
            date_publish=publish_date,
            snippet=_clip_text(text, self.max_snippet_chars),
            score=float(hit.get("score") or hit.get("_score") or 0.0),
            url=url,
        )


class OpenRewardSearchTool(BaseSearchTool):
    """Synchronous BaseSearchTool wrapper around OpenReward's SDK web tools."""

    def __init__(self, *, api_key: str = "", search_url: str = "", fetch_url: str = ""):
        self._async_tool = OpenRewardSdkSearchTool.from_env(
            api_key=api_key,
            search_url=search_url,
            fetch_url=fetch_url,
        )

    @classmethod
    def from_env(
        cls,
        api_key: str = "",
        search_url: str = "",
        fetch_url: str = "",
    ) -> "OpenRewardSearchTool":
        return cls(api_key=api_key, search_url=search_url, fetch_url=fetch_url)

    @property
    def is_available(self) -> bool:
        return self._async_tool.is_available

    def search(
        self,
        query: str,
        max_results: int = 10,
        max_date: Optional[date] = None,
        search_type: str = "hybrid",
        min_date: Optional[date] = None,
    ) -> list[SearchResult]:
        return _run_async(
            self._async_tool.search(
                query,
                max_results=max_results,
                max_date=max_date,
                search_type=search_type,
                min_date=min_date,
            )
        )

    def get_article(self, article_id: str) -> Optional[Article]:
        return _run_async(self._async_tool.get_article(article_id))
