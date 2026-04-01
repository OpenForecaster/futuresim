#!/usr/bin/env python3
"""
Filter Kalshi resolved questions JSONL.

Usage:
    # All filters
    python data/kalshi/filter_kalshi.py \
        --input /fast/nchandak/forecast-sim/data/kalshi/kalshi_resolved.jsonl \
        --min_outcomes 3 --no_numeric_answer --deduplicate

    # Just dedup
    python data/kalshi/filter_kalshi.py \
        --input /fast/nchandak/forecast-sim/data/kalshi/kalshi_resolved.jsonl \
        --deduplicate
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def has_numeric_range(answer: str) -> bool:
    """Check if answer looks like a numeric range (prices, temps, percentages)."""
    patterns = [
        r"\$[\d,]+",
        r"\d+\s*to\s*\d+",
        r"\d+°",
        r"\d+\s*or\s*(above|below)",
        r"\d+%\s*and\s*(above|below)",
    ]
    return any(re.search(p, answer, re.IGNORECASE) for p in patterns)


def get_series(event_ticker: str) -> str:
    """Extract the series prefix from an event ticker.

    E.g. KXMLBF5-26MAR3001 -> KXMLBF5, KXCABOUT-29JAN -> KXCABOUT
    """
    return event_ticker.split("-")[0]


def sanitize_resolution_criteria(q: dict) -> dict:
    """Replace any outcome name leaked in resolution_criteria with a generic placeholder.

    Replaces all outcome names (longest first to avoid partial matches),
    but skips very short outcomes (<=2 chars) that cause false positives.
    """
    criteria = q.get("resolution_criteria", "")
    if not criteria:
        return q
    # Sort outcomes longest-first so "Mike McDaniel / No new..." is replaced before "Mike"
    outcomes = sorted(q.get("outcomes", []), key=len, reverse=True)
    for outcome in outcomes:
        if len(outcome) <= 2:
            continue
        if outcome in criteria:
            criteria = criteria.replace(outcome, "the selected outcome")
    if criteria != q["resolution_criteria"]:
        q = {**q, "resolution_criteria": criteria}
    return q


def filter_min_outcomes(questions: list[dict], min_outcomes: int) -> list[dict]:
    return [q for q in questions if q["num_outcomes"] >= min_outcomes]


def filter_numeric_answer(questions: list[dict]) -> list[dict]:
    return [q for q in questions if not has_numeric_range(q["answer"])]


def filter_categories(questions: list[dict], exclude: set[str]) -> list[dict]:
    return [q for q in questions if q["category"] not in exclude]


def normalize_title(title: str) -> str:
    """Strip dates and numbers to get the question template."""
    t = re.sub(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d+", "", title)
    t = re.sub(r"\b20\d{2}\b", "", t)
    t = re.sub(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "", t)
    t = re.sub(r"\(\d+/\d+.*?\)", "", t)
    return re.sub(r"\s+", " ", t).strip()


def deduplicate_by_series(questions: list[dict]) -> list[dict]:
    """Keep the highest-volume question per event series."""
    series_best: dict[str, dict] = {}
    for q in questions:
        s = get_series(q["event_ticker"])
        if s not in series_best or q["total_volume"] > series_best[s]["total_volume"]:
            series_best[s] = q
    return list(series_best.values())


def deduplicate_by_title(questions: list[dict]) -> list[dict]:
    """Collapse only truly repeated questions (same title after stripping dates).

    Questions that share the same series AND normalized title are duplicates
    (e.g. "Top USA Song on Spotify on Mar 27" vs "...on Mar 24").
    Questions with different normalized titles are kept even if same series
    (e.g. different MLB matchups, different TX primary races).
    """
    best: dict[tuple[str, str], dict] = {}  # (series, normalized_title) -> best question
    for q in questions:
        key = (get_series(q["event_ticker"]), normalize_title(q["question_title"]))
        if key not in best or q["total_volume"] > best[key]["total_volume"]:
            best[key] = q
    return list(best.values())


def main():
    parser = argparse.ArgumentParser(description="Filter Kalshi resolved questions JSONL.")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", default=None,
                        help="Output JSONL file (default: <input_stem>_filtered.jsonl)")
    parser.add_argument("--min_outcomes", type=int, default=0,
                        help="Minimum number of outcomes (e.g. 3 to drop binary-like)")
    parser.add_argument("--no_numeric_answer", action="store_true",
                        help="Exclude questions whose answer is a numeric range")
    parser.add_argument("--exclude_categories", nargs="*", default=[],
                        help="Categories to exclude (e.g. Crypto Financials 'Climate and Weather')")
    parser.add_argument("--deduplicate", choices=["series", "title"], default=None,
                        help="Dedup mode: 'series' = 1 per event series (aggressive), "
                             "'title' = collapse only repeated templates (keeps different matchups/races)")
    args = parser.parse_args()

    with open(args.input) as f:
        questions = [json.loads(line) for line in f]
    print(f"Loaded {len(questions)} questions from {args.input}")

    # Always sanitize resolution criteria (remove leaked outcome names)
    questions = [sanitize_resolution_criteria(q) for q in questions]

    if args.min_outcomes > 0:
        before = len(questions)
        questions = filter_min_outcomes(questions, args.min_outcomes)
        print(f"  min_outcomes >= {args.min_outcomes}: {before} -> {len(questions)}")

    if args.no_numeric_answer:
        before = len(questions)
        questions = filter_numeric_answer(questions)
        print(f"  no_numeric_answer: {before} -> {len(questions)}")

    if args.exclude_categories:
        before = len(questions)
        questions = filter_categories(questions, set(args.exclude_categories))
        print(f"  exclude_categories {args.exclude_categories}: {before} -> {len(questions)}")

    if args.deduplicate == "series":
        before = len(questions)
        questions = deduplicate_by_series(questions)
        print(f"  deduplicate by series: {before} -> {len(questions)}")
    elif args.deduplicate == "title":
        before = len(questions)
        questions = deduplicate_by_title(questions)
        print(f"  deduplicate by title: {before} -> {len(questions)}")

    # Summary
    cats = Counter(q["category"] for q in questions)
    print(f"\nFinal: {len(questions)} questions")
    print(f"Categories: {dict(cats)}")

    # Write output
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + "_filtered.jsonl")
    with open(output_path, "w") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"Wrote to {output_path}")


if __name__ == "__main__":
    main()
