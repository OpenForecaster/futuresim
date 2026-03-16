#!/usr/bin/env python3
"""
Quick status checker for news search artifacts.

Checks:
1. Whether search artifacts exist and look rebuilt (freshness + month coverage)
2. Monthly article counts from Parquet row metadata
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

import pyarrow.parquet as pq


DEFAULT_NEWS_BASE = Path(os.getenv("FSIM_NEWS_BASE", "/is/cluster/fast/sgoel/forecasting/news"))
DEFAULT_PARQUET_DIR = DEFAULT_NEWS_BASE / "deduped_articles" / "data"
DEFAULT_EMBEDDINGS_DIR = (
    DEFAULT_NEWS_BASE / "deduped_articles" / "embeddings" / "Qwen3-Embedding-8B"
)
DEFAULT_LANCE_DIR = (
    DEFAULT_NEWS_BASE
    / "deduped_articles"
    / "lance"
    / "Qwen3-Embedding-8B"
    / "articles.lance"
)


def iso_ts(ts: Optional[float]) -> str:
    if ts is None:
        return "missing"
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def newest_file_mtime(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    newest: Optional[float] = None
    if path.is_file():
        return path.stat().st_mtime
    for root, _, files in os.walk(path):
        for name in files:
            p = Path(root) / name
            try:
                ts = p.stat().st_mtime
            except OSError:
                continue
            if newest is None or ts > newest:
                newest = ts
    return newest


def list_month_dirs(base_dir: Path) -> Set[str]:
    months: Set[str] = set()
    if not base_dir.exists():
        return months
    for y in base_dir.iterdir():
        if not (y.is_dir() and y.name.isdigit() and len(y.name) == 4):
            continue
        for m in y.iterdir():
            if m.is_dir() and m.name.isdigit() and len(m.name) == 2:
                months.add(f"{y.name}-{m.name}")
    return months


def month_in_range(month: str, month_from: Optional[str], month_to: Optional[str]) -> bool:
    if month_from and month < month_from:
        return False
    if month_to and month > month_to:
        return False
    return True


def parquet_month_counts(
    parquet_data_dir: Path, month_from: Optional[str], month_to: Optional[str]
) -> Tuple[Counter, Counter, int, int]:
    rows_by_month: Counter = Counter()
    files_by_month: Counter = Counter()
    total_rows = 0
    total_files = 0

    if not parquet_data_dir.exists():
        return rows_by_month, files_by_month, total_rows, total_files

    for y in sorted(parquet_data_dir.iterdir()):
        if not (y.is_dir() and y.name.isdigit() and len(y.name) == 4):
            continue
        for m in sorted(y.iterdir()):
            if not (m.is_dir() and m.name.isdigit() and len(m.name) == 2):
                continue
            month_key = f"{y.name}-{m.name}"
            if not month_in_range(month_key, month_from, month_to):
                continue
            for d in sorted(m.iterdir()):
                if not d.is_dir():
                    continue
                for pq_file in d.glob("*.parquet"):
                    try:
                        md = pq.ParquetFile(pq_file).metadata
                        nrows = int(md.num_rows)
                    except Exception:
                        continue
                    rows_by_month[month_key] += nrows
                    files_by_month[month_key] += 1
                    total_rows += nrows
                    total_files += 1
    return rows_by_month, files_by_month, total_rows, total_files


def print_month_table(rows_by_month: Dict[str, int], files_by_month: Dict[str, int]) -> None:
    print("\nMonthly article counts (from Parquet metadata)")
    print("month     articles    parquet_files")
    for month in sorted(rows_by_month):
        print(f"{month}  {rows_by_month[month]:>9,}  {files_by_month[month]:>13,}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check news search readiness + monthly counts")
    parser.add_argument("--news-base", type=Path, default=DEFAULT_NEWS_BASE)
    parser.add_argument("--parquet-dir", type=Path, default=None)
    parser.add_argument("--embeddings-dir", type=Path, default=None)
    parser.add_argument("--lance-dir", type=Path, default=None)
    parser.add_argument(
        "--month-from",
        type=str,
        default=None,
        help="Only show months >= YYYY-MM",
    )
    parser.add_argument(
        "--month-to",
        type=str,
        default=None,
        help="Only show months <= YYYY-MM",
    )
    args = parser.parse_args()

    parquet_dir = args.parquet_dir or (args.news_base / "deduped_articles" / "data")
    embeddings_dir = args.embeddings_dir or (
        args.news_base / "deduped_articles" / "embeddings" / "Qwen3-Embedding-8B"
    )
    lance_dir = args.lance_dir or (
        args.news_base
        / "deduped_articles"
        / "lance"
        / "Qwen3-Embedding-8B"
        / "articles.lance"
    )

    parquet_newest = newest_file_mtime(parquet_dir)
    embeddings_newest = newest_file_mtime(embeddings_dir)
    lance_newest = newest_file_mtime(lance_dir)

    parquet_months = list_month_dirs(parquet_dir)
    embedding_months = list_month_dirs(embeddings_dir)
    missing_embeddings = sorted(parquet_months - embedding_months)

    rows_by_month, files_by_month, total_rows, total_files = parquet_month_counts(
        parquet_dir, args.month_from, args.month_to
    )

    print("News Search Rebuild Check")
    print(f"parquet_dir:    {parquet_dir}")
    print(f"embeddings_dir: {embeddings_dir}")
    print(f"lance_dir:      {lance_dir}")
    print("")
    print(f"parquet newest file:    {iso_ts(parquet_newest)}")
    print(f"embeddings newest file: {iso_ts(embeddings_newest)}")
    print(f"lance newest file:      {iso_ts(lance_newest)}")
    print("")
    print(f"parquet months:    {len(parquet_months)}")
    print(f"embedding months:  {len(embedding_months)}")
    print(f"missing emb months: {len(missing_embeddings)}")
    if missing_embeddings:
        preview = ", ".join(missing_embeddings[:12])
        suffix = " ..." if len(missing_embeddings) > 12 else ""
        print(f"missing list: {preview}{suffix}")

    reasons = []
    if not parquet_dir.exists():
        reasons.append("Parquet data missing")
    if not embeddings_dir.exists():
        reasons.append("Embeddings missing")
    if not lance_dir.exists():
        reasons.append("LanceDB index missing")
    if missing_embeddings:
        reasons.append("Embeddings do not cover all Parquet months")
    if lance_newest is not None:
        latest_input = max([x for x in [parquet_newest, embeddings_newest] if x is not None], default=None)
        if latest_input is not None and lance_newest < latest_input:
            reasons.append("LanceDB appears older than inputs (likely stale)")

    print("")
    if reasons:
        print("search_ready: NO")
        for r in reasons:
            print(f"- {r}")
    else:
        print("search_ready: YES")

    print("")
    print(f"counted parquet files: {total_files:,}")
    print(f"counted articles:      {total_rows:,}")
    print_month_table(rows_by_month, files_by_month)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
