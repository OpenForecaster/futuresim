"""Static search cache helpers for MinimalHarness warmup ablations."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Optional, Tuple

from futuresim_agents.basicAgent.search import SearchHandler


def safe_qid(qid: Any) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(qid))


def static_search_path(output_dir: Path, qid: Any) -> Path:
    return output_dir / f"{safe_qid(qid)}.md"


def run_title_search(
    *,
    search_tool: Any,
    q: Any,
    search_date: date,
    search_type: str = "hybrid",
    search_cutoff_days: int = 0,
    max_results: int = 5,
) -> str:
    if search_tool is None or not getattr(search_tool, "is_available", False):
        raise RuntimeError("Static search requires an available search tool.")

    handler = SearchHandler(
        search_tool=search_tool,
        search_cutoff_days=search_cutoff_days,
    )
    handler.set_date(search_date)
    text, err = handler.search(
        str(q.title),
        max_results=max_results,
        search_type=search_type,
    )
    if err:
        raise RuntimeError(err)
    return text or "No articles found matching your query."


def render_static_search_file(
    *,
    q: Any,
    search_date: date,
    retrieved_articles: str,
    search_type: str = "hybrid",
    max_results: int = 5,
) -> str:
    metadata = {
        "qid": str(q.qid),
        "title": str(q.title),
        "resolution_date": str(getattr(q, "resolution_date", "")),
        "search_date": search_date.isoformat(),
        "query": str(q.title),
        "search_type": search_type,
        "max_results": max_results,
    }
    return (
        "# Static Search Evidence\n\n"
        f"Question ID: {metadata['qid']}\n"
        f"Question Title: {metadata['title']}\n"
        f"Resolution Date: {metadata['resolution_date']}\n"
        f"Search Date: {metadata['search_date']}\n"
        f"Search Query: {metadata['query']}\n"
        f"Search Type: {metadata['search_type']}\n"
        f"Max Results: {metadata['max_results']}\n\n"
        "## Metadata\n"
        "```json\n"
        f"{json.dumps(metadata, indent=2, sort_keys=True)}\n"
        "```\n\n"
        "## Retrieved Articles\n"
        f"{retrieved_articles.rstrip()}\n"
    )


def ensure_static_search_file(
    *,
    output_dir: Path,
    search_tool: Any,
    q: Any,
    search_date: date,
    search_type: str = "hybrid",
    search_cutoff_days: int = 0,
    max_results: int = 5,
    overwrite: bool = False,
) -> Tuple[Path, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = static_search_path(output_dir, q.qid)
    if path.exists() and not overwrite:
        return path, path.read_text()

    retrieved_articles = run_title_search(
        search_tool=search_tool,
        q=q,
        search_date=search_date,
        search_type=search_type,
        search_cutoff_days=search_cutoff_days,
        max_results=max_results,
    )
    text = render_static_search_file(
        q=q,
        search_date=search_date,
        retrieved_articles=retrieved_articles,
        search_type=search_type,
        max_results=max_results,
    )
    path.write_text(text)
    return path, text
