"""OpenReward time-gated news search backend."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Optional

from .base import Article, BaseSearchTool, SearchResult


DEFAULT_SEARCH_URL = "https://search.openreward.ai/search"
DEFAULT_FETCH_URL = "https://search.openreward.ai/fetch"


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


def _is_loopback_url(url: str) -> bool:
    host = urllib.parse.urlparse(url).hostname or ""
    return host in {"127.0.0.1", "::1", "localhost"}


@dataclass
class OpenRewardSearchConfig:
    search_url: str = DEFAULT_SEARCH_URL
    fetch_url: str = DEFAULT_FETCH_URL
    api_key: str = ""
    api_key_env: str = "OPENREWARD_API_KEY"
    max_snippet_chars: int = 2000
    timeout_seconds: float = 90.0


class OpenRewardSearchTool(BaseSearchTool):
    """Use OpenReward's beta CC-NEWS search while preserving Futuresim's tool shape."""

    def __init__(self, config: OpenRewardSearchConfig | None = None):
        self.config = config or OpenRewardSearchConfig()
        self._api_key = (
            self.config.api_key
            or os.environ.get(self.config.api_key_env, "")
            or os.environ.get("OR_TOKEN", "")
        )

    @classmethod
    def from_env(cls, api_key: str = "") -> "OpenRewardSearchTool":
        return cls(OpenRewardSearchConfig(
            search_url=os.environ.get("FSIM_OPENREWARD_SEARCH_URL", os.environ.get("OR_API", DEFAULT_SEARCH_URL)),
            fetch_url=os.environ.get("FSIM_OPENREWARD_FETCH_URL", os.environ.get("OR_FETCH_API", DEFAULT_FETCH_URL)),
            api_key=api_key,
            api_key_env=os.environ.get("FSIM_OPENREWARD_API_KEY_ENV", "OPENREWARD_API_KEY"),
            max_snippet_chars=int(os.environ.get("FSIM_OPENREWARD_SNIPPET_CHARS", "2000")),
            timeout_seconds=float(os.environ.get("FSIM_OPENREWARD_TIMEOUT_SECONDS", "90")),
        ))

    @property
    def is_available(self) -> bool:
        return bool(self._api_key)

    def search(
        self,
        query: str,
        max_results: int = 10,
        max_date: Optional[date] = None,
        search_type: str = "hybrid",
        min_date: Optional[date] = None,
    ) -> list[SearchResult]:
        if not self.is_available:
            return []
        if search_type != "hybrid":
            raise ValueError("OpenReward search is configured for hybrid mode only.")

        as_of = max_date or date.today()
        k = min(max(1, int(max_results or 5)), 5)
        response = self._post(self.config.search_url, {
            "query": query,
            "as_of": as_of.isoformat(),
            "k": k,
            "mode": "hybrid",
        })
        hits = response.get("hits") or response.get("results") or []
        if not isinstance(hits, list):
            return []

        results: list[SearchResult] = []
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            results.append(self._hit_to_result(hit, as_of))
            if len(results) >= k:
                break
        return results

    def get_article(self, article_id: str) -> Optional[Article]:
        if not self.is_available:
            return None
        url = article_id if article_id.startswith(("http://", "https://")) else ""
        if not url:
            return None
        data = self._fetch(url, date.today())
        text = _clean_text(data.get("text") or data.get("content") or "")
        return Article(
            id=url,
            title=str(data.get("title") or ""),
            source=str(data.get("source") or data.get("host") or _hostname(url)),
            date=_parse_date(data.get("crawl_date") or data.get("date")),
            content=text,
            url=url,
            description=str(data.get("description") or ""),
        )

    def _hit_to_result(self, hit: dict[str, Any], as_of: date) -> SearchResult:
        url = str(hit.get("url") or hit.get("link") or "")
        fetched: dict[str, Any] = {}
        text = _clean_text(hit.get("snippet") or hit.get("text") or hit.get("content"))
        if url and len(text) < 400:
            try:
                fetched = self._fetch(url, as_of)
                text = _clean_text(fetched.get("text") or fetched.get("content") or text)
            except RuntimeError:
                pass

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
            snippet=_clip_text(text, self.config.max_snippet_chars),
            score=float(hit.get("score") or hit.get("_score") or 0.0),
            url=url,
        )

    def _fetch(self, url: str, as_of: date) -> dict[str, Any]:
        return self._post(self.config.fetch_url, {
            "url": url,
            "as_of": as_of.isoformat(),
            "summarize": False,
        })

    def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode()
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "Authorization": f"Bearer {self._api_key}",
        }
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            if _is_loopback_url(url):
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                response_cm = opener.open(request, timeout=self.config.timeout_seconds)
            else:
                response_cm = urllib.request.urlopen(request, timeout=self.config.timeout_seconds)
            with response_cm as response:
                data = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise RuntimeError(f"OpenReward search HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"OpenReward search request failed: {exc}") from exc
        if isinstance(data, dict) and data.get("detail"):
            raise RuntimeError(f"OpenReward search error: {data['detail']}")
        if not isinstance(data, dict):
            raise RuntimeError("OpenReward search returned non-object JSON")
        return data
