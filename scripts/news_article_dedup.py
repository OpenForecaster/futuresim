#!/usr/bin/env python3
"""Helpers for deduplicating article records across pipeline stages."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _article_content(article: Mapping[str, Any]) -> str:
    return _as_text(article.get("content") or article.get("maintext"))


def article_identity_key(article: Mapping[str, Any]) -> str | None:
    """Return a stable identity key for an article when possible."""
    article_id = _as_text(article.get("id"))
    if article_id:
        return f"id:{article_id}"

    url = _as_text(article.get("url"))
    if url:
        return f"url:{url}"

    title = _as_text(article.get("title"))
    content = _article_content(article)
    if title or content:
        digest = hashlib.md5(f"{title}\n{content}".encode("utf-8")).hexdigest()
        return f"title_content:{digest}"

    return None


def _is_empty(value: Any) -> bool:
    return value in (None, "", [], ())


def _article_score(article: Mapping[str, Any]) -> tuple[int, ...]:
    content = _article_content(article)
    title = _as_text(article.get("title"))
    description = _as_text(article.get("description"))
    url = _as_text(article.get("url"))
    source = _as_text(article.get("source") or article.get("source_domain"))
    authors = article.get("authors") or []
    author_count = len(authors) if isinstance(authors, list) else 1
    return (
        int(bool(content)),
        len(content),
        int(bool(title)),
        len(title),
        int(bool(description)),
        len(description),
        int(bool(url)),
        len(url),
        int(bool(source)),
        len(source),
        author_count,
    )


def merge_article_records(existing: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the richer article record while backfilling missing fields from the other."""
    take_candidate = _article_score(candidate) > _article_score(existing)
    preferred = dict(candidate) if take_candidate else dict(existing)
    other = existing if take_candidate else candidate

    for key, value in other.items():
        if key not in preferred or _is_empty(preferred[key]):
            preferred[key] = value

    return preferred


def dedupe_articles(articles: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate articles while preserving first-seen order."""
    deduped: list[dict[str, Any]] = []
    index_by_key: dict[str, int] = {}
    anonymous_counter = 0

    for article in articles:
        article_copy = dict(article)
        key = article_identity_key(article_copy)
        if key is None:
            key = f"anonymous:{anonymous_counter}"
            anonymous_counter += 1

        if key not in index_by_key:
            index_by_key[key] = len(deduped)
            deduped.append(article_copy)
            continue

        existing_idx = index_by_key[key]
        deduped[existing_idx] = merge_article_records(deduped[existing_idx], article_copy)

    return deduped
