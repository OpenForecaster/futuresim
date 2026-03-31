#!/usr/bin/env python3
"""Convert a date-partitioned Parquet article corpus into a date-partitioned JSONL mirror."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pathing import load_repo_env


load_repo_env(REPO_ROOT)


DEFAULT_NEWS_BASE = os.getenv("FSIM_NEWS_BASE", "/is/cluster/fast/sgoel/forecasting/news")
DEFAULT_INPUT_DIR = os.getenv("FSIM_NEWS_ARTICLES_DIR", f"{DEFAULT_NEWS_BASE}/deduped_articles/data")
DEFAULT_OUTPUT_DIR = os.getenv("FSIM_NEWS_JSONL_DIR", f"{DEFAULT_NEWS_BASE}/deduped_articles/jsonl")

_FIELDS = [
    "id",
    "title",
    "source",
    "date",
    "date_publish",
    "date_modify",
    "url",
    "content",
    "authors",
    "description",
]


def _parse_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def _serialize_value(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    return value


def _date_range(start_date: date, end_date: date) -> Iterable[date]:
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def _iter_parquet_files(
    input_dir: Path,
    *,
    start_date: Optional[date],
    end_date: Optional[date],
) -> Iterable[Path]:
    if start_date is not None and end_date is not None:
        for day_value in _date_range(start_date, end_date):
            day_dir = input_dir / day_value.strftime("%Y/%m/%d")
            if not day_dir.exists():
                continue
            for parquet_path in sorted(day_dir.glob("*.parquet")):
                yield parquet_path
        return

    for parquet_path in sorted(input_dir.glob("*/*/*/*.parquet")):
        yield parquet_path


def _convert_file(parquet_path: Path, input_dir: Path, output_dir: Path, overwrite: bool) -> bool:
    rel_path = parquet_path.relative_to(input_dir)
    jsonl_path = (output_dir / rel_path).with_suffix(".jsonl")
    if jsonl_path.exists() and not overwrite:
        return False

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(parquet_path)
    tmp_path = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for batch in table.to_batches():
            for row in batch.to_pylist():
                payload = {field: _serialize_value(row.get(field)) for field in _FIELDS}
                handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp_path.replace(jsonl_path)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", default=DEFAULT_INPUT_DIR, help="Root Parquet corpus dir (YYYY/MM/DD/*.parquet)")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Root JSONL mirror dir (YYYY/MM/DD/*.jsonl)")
    parser.add_argument("--start_date", default=None, help="Optional YYYY-MM-DD lower bound")
    parser.add_argument("--end_date", default=None, help="Optional YYYY-MM-DD upper bound")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing JSONL files")
    args = parser.parse_args()

    input_dir = Path(os.path.expanduser(os.path.expandvars(args.input_dir))).resolve()
    output_dir = Path(os.path.expanduser(os.path.expandvars(args.output_dir))).resolve()
    if not input_dir.exists():
        raise SystemExit(f"Input Parquet corpus does not exist: {input_dir}")

    start_date = _parse_date(args.start_date) if args.start_date else None
    end_date = _parse_date(args.end_date) if args.end_date else None
    if (start_date is None) != (end_date is None):
        raise SystemExit("Pass both --start_date and --end_date together, or neither.")
    if start_date is not None and end_date is not None and start_date > end_date:
        raise SystemExit("--start_date must be <= --end_date")

    converted = 0
    skipped = 0
    for parquet_path in _iter_parquet_files(input_dir, start_date=start_date, end_date=end_date):
        changed = _convert_file(parquet_path, input_dir, output_dir, overwrite=args.overwrite)
        if changed:
            converted += 1
        else:
            skipped += 1

    print(f"Converted {converted} parquet file(s) into JSONL under {output_dir}")
    if skipped:
        print(f"Skipped {skipped} existing JSONL file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
