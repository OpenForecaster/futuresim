from scripts.news_article_dedup import (
    article_identity_key,
    dedupe_articles,
    merge_article_records,
)


def test_article_identity_key_prefers_id_then_url_then_content() -> None:
    assert article_identity_key({"id": "abc", "url": "https://example.com/a"}) == "id:abc"
    assert article_identity_key({"url": "https://example.com/a"}) == "url:https://example.com/a"

    content_key = article_identity_key({"title": "Title", "content": "Body"})
    assert content_key is not None
    assert content_key.startswith("title_content:")


def test_merge_article_records_keeps_richer_record_and_backfills_missing_fields() -> None:
    existing = {
        "id": "a1",
        "title": "Short title",
        "content": "",
        "description": "",
        "url": "https://example.com/a1",
    }
    candidate = {
        "id": "a1",
        "title": "A much longer title",
        "content": "Full article body",
        "description": "Summary",
        "url": "",
        "source": "Example",
    }

    merged = merge_article_records(existing, candidate)

    assert merged["id"] == "a1"
    assert merged["title"] == "A much longer title"
    assert merged["content"] == "Full article body"
    assert merged["description"] == "Summary"
    assert merged["url"] == "https://example.com/a1"
    assert merged["source"] == "Example"


def test_dedupe_articles_preserves_first_seen_order_while_merging_duplicates() -> None:
    articles = [
        {"id": "a1", "title": "First", "content": "", "url": "https://example.com/a1"},
        {"id": "a2", "title": "Second", "content": "Body 2", "url": "https://example.com/a2"},
        {"id": "a1", "title": "First richer", "content": "Body 1", "description": "Summary"},
        {"url": "https://example.com/a3", "title": "Third", "content": "Body 3"},
        {"url": "https://example.com/a3", "title": "Third", "content": "Body 3", "source": "Example"},
    ]

    deduped = dedupe_articles(articles)

    assert [article.get("id") or article.get("url") for article in deduped] == [
        "a1",
        "a2",
        "https://example.com/a3",
    ]
    assert deduped[0]["title"] == "First richer"
    assert deduped[0]["content"] == "Body 1"
    assert deduped[0]["description"] == "Summary"
    assert deduped[2]["source"] == "Example"
