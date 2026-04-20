"""Materialize a plain-text `articles.jsonl` alongside each day's parquet shard(s).

Walks `<data-dir>/YYYY/MM/DD/` and, for each directory that contains
`articles_b*.parquet`, writes a sibling `articles.jsonl` with one JSON record
per article (all parquet columns preserved).

Idempotent: skips a day whose existing `articles.jsonl` already matches the
combined parquet row count. Writes atomically via tmp-file + rename.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import pandas as pd


DEFAULT_DATA_DIR = "/fast/sgoel/forecasting/news/deduped_articles/data"


def find_day_dirs(data_dir: Path) -> list[Path]:
    return sorted(p.parent for p in data_dir.glob("*/*/*/articles_b*.parquet"))


def convert_day(day_dir: Path, force: bool = False) -> tuple[Path, str, int]:
    """Convert all parquet shards in `day_dir` to a single articles.jsonl.

    Returns (day_dir, status, rows). status ∈ {"skipped", "written", "error:<msg>"}.
    """
    parquets = sorted(day_dir.glob("articles_b*.parquet"))
    if not parquets:
        return day_dir, "skipped", 0

    out_path = day_dir / "articles.jsonl"

    # Count rows cheaply via parquet metadata to decide whether to skip.
    import pyarrow.parquet as pq
    expected_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in parquets)

    if not force and out_path.exists():
        with open(out_path, "rb") as f:
            have_rows = sum(1 for _ in f)
        if have_rows == expected_rows:
            return day_dir, "skipped", have_rows

    tmp_path = out_path.with_suffix(".jsonl.tmp")
    try:
        written = 0
        with open(tmp_path, "w") as f:
            for p in parquets:
                df = pd.read_parquet(p)
                for rec in df.to_dict(orient="records"):
                    f.write(json.dumps(rec, default=str, ensure_ascii=False))
                    f.write("\n")
                    written += 1
        os.replace(tmp_path, out_path)
        return day_dir, "written", written
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        return day_dir, f"error:{type(e).__name__}: {e}", 0


def _worker(args):
    day_dir, force = args
    return convert_day(Path(day_dir), force=force)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help=f"Root of YYYY/MM/DD parquet tree (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--workers", type=int, default=16,
                        help="Parallel worker processes (default: 16)")
    parser.add_argument("--force", action="store_true",
                        help="Rewrite even if articles.jsonl row count matches")
    parser.add_argument("--dry-run", action="store_true",
                        help="List day dirs that would be processed and exit")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.is_dir():
        print(f"error: {data_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    day_dirs = find_day_dirs(data_dir)
    print(f"Found {len(day_dirs)} day directories under {data_dir}")

    if args.dry_run:
        for d in day_dirs[:10]:
            print(f"  {d}")
        if len(day_dirs) > 10:
            print(f"  ... ({len(day_dirs) - 10} more)")
        return

    t0 = time.time()
    tasks = [(str(d), args.force) for d in day_dirs]
    written = skipped = errors = total_rows = 0

    with mp.Pool(processes=args.workers) as pool:
        for i, (day, status, rows) in enumerate(pool.imap_unordered(_worker, tasks, chunksize=4), 1):
            if status == "written":
                written += 1
                total_rows += rows
            elif status == "skipped":
                skipped += 1
            else:
                errors += 1
                print(f"[err] {day}: {status}", file=sys.stderr)
            if i % 100 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed else 0
                print(f"  [{i}/{len(tasks)}] written={written} skipped={skipped} errors={errors} "
                      f"rows={total_rows:,} ({rate:.1f} day/s, {elapsed:.1f}s elapsed)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s: written={written} skipped={skipped} errors={errors} "
          f"rows={total_rows:,}")


if __name__ == "__main__":
    main()
