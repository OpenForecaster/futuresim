#!/usr/bin/env python3
"""
Print quantitative statistics about news articles in a directory.

Given an input path (directory or single .jsonl file), this script:
- Scans all .jsonl files in the directory (and optionally subdirectories), or processes a single file
- For each article, uses the max of all available dates (date_download, date_modify, date_publish)
- Prints monthly article counts and other summary statistics

Usage:
    python article_stats.py --path /fast/sgoel/forecasting/news/articles2025
    python article_stats.py --path /path/to/articles --recursive
    python article_stats.py --path /path/to/single_source.jsonl
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

from datetime import timezone

from dateutil.parser import parse as parse_date


DATE_FIELDS = ("date_download", "date_modify", "date_publish")


def parse_date_safe(value):
    """Parse a date string, return None on failure."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        return parse_date(value)
    except (ValueError, TypeError):
        return None


def _ensure_aware(dt):
    """Make datetime timezone-aware (assume UTC if naive) for comparison."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def get_article_date(article: dict):
    """
    Get the canonical date for an article: max of all available date fields.
    Returns (datetime, list_of_used_fields) or (None, []) if no parseable date.
    """
    dates = []
    for field in DATE_FIELDS:
        val = article.get(field)
        parsed = parse_date_safe(val)
        if parsed is not None:
            dates.append((_ensure_aware(parsed), field))
    if not dates:
        return None, []
    best = max(dates, key=lambda x: x[0])
    return best[0], [f for _, f in dates]


def collect_articles(root: Path, recursive: bool) -> list[Path]:
    """Collect .jsonl file paths. If root is a single file, return it directly."""
    if root.is_file():
        if root.suffix == ".jsonl":
            return [root]
        raise ValueError(f"Not a .jsonl file: {root}")
    if recursive:
        return list(root.rglob("*.jsonl"))
    return [f for f in root.iterdir() if f.is_file() and f.suffix == ".jsonl"]


def run_stats(root_path: str, recursive: bool = False):
    root = Path(root_path)
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")

    files = collect_articles(root, recursive)

    monthly = defaultdict(int)
    by_source = defaultdict(int)
    total = 0
    no_date = 0
    parse_errors = 0

    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        article = json.loads(line)
                    except json.JSONDecodeError:
                        parse_errors += 1
                        continue

                    dt, _ = get_article_date(article)
                    total += 1

                    if dt is None:
                        no_date += 1
                        continue

                    month_key = dt.strftime("%Y-%m")
                    monthly[month_key] += 1

                    domain = article.get("source_domain", "unknown")
                    by_source[domain] += 1

        except (IOError, OSError) as e:
            print(f"Warning: could not read {fpath}: {e}", file=__import__("sys").stderr)

    return {
        "total": total,
        "no_date": no_date,
        "parse_errors": parse_errors,
        "monthly": dict(monthly),
        "by_source": dict(by_source),
        "files_scanned": len(files),
    }


def print_report(stats: dict, root_path: str):
    total = stats["total"]
    no_date = stats["no_date"]
    parse_errors = stats["parse_errors"]
    monthly = stats["monthly"]
    by_source = stats["by_source"]
    files_scanned = stats["files_scanned"]

    print("=" * 60)
    print(f"Article statistics for: {root_path}")
    print("=" * 60)
    print()
    print("Summary")
    print("-" * 40)
    print(f"  Total articles:        {total:,}")
    print(f"  Articles with no date: {no_date:,}")
    if parse_errors:
        print(f"  JSON parse errors:      {parse_errors:,}")
    print(f"  JSONL files scanned:   {files_scanned:,}")
    print()

    if monthly:
        print("Articles per month (by max of date_download, date_modify, date_publish)")
        print("-" * 40)
        for month in sorted(monthly.keys()):
            count = monthly[month]
            bar = "█" * min(50, count // max(1, max(monthly.values()) // 50))
            print(f"  {month}: {count:>8,}  {bar}")
        print()

        # Top months
        top_months = sorted(monthly.items(), key=lambda x: -x[1])[:5]
        print("Top 5 months by article count:")
        for month, cnt in top_months:
            print(f"  {month}: {cnt:,}")
        print()

    if by_source:
        print("Top 15 sources by article count:")
        print("-" * 40)
        top_sources = sorted(by_source.items(), key=lambda x: -x[1])[:15]
        for domain, cnt in top_sources:
            print(f"  {domain}: {cnt:,}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Print quantitative statistics about news articles in JSONL format."
    )
    parser.add_argument(
        "--path",
        type=str,
        default="/fast/sgoel/forecasting/news/articles2025/deduped/relevant/",
        help="Path to articles directory or a single .jsonl file",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Include all subdirectories (default: only top-level .jsonl files)",
    )
    args = parser.parse_args()

    stats = run_stats(args.path, recursive=args.recursive)
    print_report(stats, args.path)


if __name__ == "__main__":
    main()
