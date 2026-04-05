#!/usr/bin/env python3
"""
Convert JSONL news articles to daily-partitioned Parquet format.

Uses streaming/batched processing to minimize memory usage.
Each batch writes temporary day shards, then the touched days are compacted
back into a single deduplicated shard by default so reruns stay idempotent.

Usage:
    python convert_jsonl_to_parquet.py \
        --input-dirs /path/to/articlesuntil2024/deduped /path/to/articles2025/deduped \
        --output-dir /path/to/deduped_articles \
        --workers 32 \
        --batch-size 128
"""

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.news_article_dedup import dedupe_articles


def log(msg: str):
    """Print and flush for Condor batch visibility."""
    print(msg, flush=True)


# Parquet schema
SCHEMA = pa.schema([
    ('id', pa.string()),
    ('title', pa.string()),
    ('source', pa.string()),
    ('date', pa.date32()),
    ('date_publish', pa.date32()),
    ('date_modify', pa.date32()),
    ('url', pa.string()),
    ('content', pa.string()),
    ('authors', pa.list_(pa.string())),
    ('description', pa.string()),
])


def parse_date(date_val: Any) -> Optional[date]:
    """Parse various date formats to date object."""
    if date_val is None:
        return None
    if isinstance(date_val, date):
        return date_val
    if isinstance(date_val, datetime):
        return date_val.date()
    if isinstance(date_val, str):
        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"]:
            try:
                return datetime.strptime(date_val.split('+')[0].split('.')[0], fmt).date()
            except ValueError:
                continue
    return None


def process_jsonl_file(jsonl_path: Path) -> Dict[date, List[dict]]:
    """Read a JSONL file and group articles by date."""
    articles_by_date = defaultdict(list)
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    article = json.loads(line)
                    pub_date = parse_date(article.get('date_download'))
                    if pub_date is None:
                        continue
                    normalized = {
                        'id': article.get('id', ''),
                        'title': article.get('title', ''),
                        'source': article.get('source_domain', ''),
                        'date': pub_date,
                        'date_publish': parse_date(article.get('date_publish')),
                        'date_modify': parse_date(article.get('date_modify')),
                        'url': article.get('url', ''),
                        'content': article.get('maintext', ''),
                        'authors': article.get('authors', []) or [],
                        'description': article.get('description', ''),
                    }
                    if normalized['id']:
                        articles_by_date[pub_date].append(normalized)
                except:
                    pass
    except:
        pass
    return dict(articles_by_date)


def _normalize_parquet_row(row: Dict[str, Any]) -> dict:
    return {
        'id': row.get('id', ''),
        'title': row.get('title', ''),
        'source': row.get('source', ''),
        'date': parse_date(row.get('date')),
        'date_publish': parse_date(row.get('date_publish')),
        'date_modify': parse_date(row.get('date_modify')),
        'url': row.get('url', ''),
        'content': row.get('content', ''),
        'authors': row.get('authors', []) or [],
        'description': row.get('description', ''),
    }


def load_articles_from_day_dir(day_dir: Path) -> List[dict]:
    """Load all articles already written for a specific day."""
    articles: List[dict] = []
    for parquet_file in sorted(day_dir.glob("articles_*.parquet")):
        try:
            table = pq.read_table(parquet_file)
            for row in table.to_pylist():
                articles.append(_normalize_parquet_row(row))
        except Exception as e:
            log(f"Error reading {parquet_file}: {e}")
    return articles


def get_day_dir(output_dir: Path, target_date: date) -> Path:
    """Get the directory path for a specific day."""
    return output_dir / "data" / f"{target_date.year:04d}" / f"{target_date.month:02d}" / f"{target_date.day:02d}"


def write_day_dir_files(day_dir: Path, articles: List[dict], batch_id: str) -> Tuple[int, Set[str]]:
    """
    Write articles and headlines into a specific day directory.
    Returns (article_count, sources set).
    """
    if not articles:
        return 0, set()

    day_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = day_dir / f"articles_{batch_id}.parquet"
    headlines_path = day_dir / f"headlines_{batch_id}.json"

    sources = {a['source'] for a in articles}
    data = {
        'id': [a['id'] for a in articles],
        'title': [a['title'] for a in articles],
        'source': [a['source'] for a in articles],
        'date': [a['date'] for a in articles],
        'date_publish': [a.get('date_publish') for a in articles],
        'date_modify': [a.get('date_modify') for a in articles],
        'url': [a['url'] for a in articles],
        'content': [a['content'] for a in articles],
        'authors': [a['authors'] for a in articles],
        'description': [a['description'] for a in articles],
    }
    
    table = pa.table(data, schema=SCHEMA)
    pq.write_table(table, parquet_path, compression='zstd')
    
    # Write headlines for this batch
    headlines = [{'id': a['id'], 'title': a['title'], 'source': a['source']} for a in articles]
    with open(headlines_path, 'w', encoding='utf-8') as f:
        json.dump(headlines, f, ensure_ascii=False)

    return len(articles), sources


def write_articles_to_day(output_dir: Path, target_date: date, articles: List[dict], batch_id: str) -> Tuple[int, Set[str]]:
    """Write articles to a batch shard for a specific day."""
    day_dir = get_day_dir(output_dir, target_date)
    return write_day_dir_files(day_dir, articles, batch_id)


def replace_directory_atomically(target_dir: Path, replacement_dir: Path) -> None:
    """Swap a staged directory into place and restore on failure."""
    backup_dir = target_dir.parent / f".{target_dir.name}.bak_compact"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)

    try:
        if target_dir.exists():
            os.replace(target_dir, backup_dir)
        os.replace(replacement_dir, target_dir)
    except Exception:
        if replacement_dir.exists():
            shutil.rmtree(replacement_dir, ignore_errors=True)
        if backup_dir.exists() and not target_dir.exists():
            os.replace(backup_dir, target_dir)
        raise
    else:
        if backup_dir.exists():
            shutil.rmtree(backup_dir)


def compact_day(output_dir: Path, target_date: date) -> Tuple[int, int, int, bool]:
    """
    Compact all shards for a day into a single deduplicated shard.
    Returns (raw_count, unique_count, shard_count, changed).
    """
    day_dir = get_day_dir(output_dir, target_date)
    shard_paths = sorted(day_dir.glob("articles_*.parquet"))
    if not shard_paths:
        return 0, 0, 0, False

    raw_articles = load_articles_from_day_dir(day_dir)
    raw_count = len(raw_articles)
    deduped_articles = dedupe_articles(raw_articles)
    unique_count = len(deduped_articles)
    shard_count = len(shard_paths)

    existing_headlines = sorted(day_dir.glob("headlines_*.json"))
    needs_rewrite = (
        shard_count != 1
        or raw_count != unique_count
        or len(existing_headlines) != 1
        or not (day_dir / "articles_b0000.parquet").exists()
        or not (day_dir / "headlines_b0000.json").exists()
    )
    if not needs_rewrite:
        return raw_count, unique_count, shard_count, False

    tmp_day_dir = day_dir.parent / f".{day_dir.name}.tmp_compact"
    if tmp_day_dir.exists():
        shutil.rmtree(tmp_day_dir)

    write_day_dir_files(tmp_day_dir, deduped_articles, "b0000")
    replace_directory_atomically(day_dir, tmp_day_dir)
    return raw_count, unique_count, shard_count, True


def process_batch(batch_files: List[Path], output_dir: Path, workers: int, batch_id: str) -> Tuple[int, Set[str], Set[date]]:
    """
    Process a batch of JSONL files: read, group by date, and write to parquet.
    Returns (article_count, sources set, dates set).
    """
    # Read all files in batch
    all_results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_jsonl_file, f): f for f in batch_files}
        for future in as_completed(futures):
            all_results.append(future.result())
    
    # Merge by date
    articles_by_date = defaultdict(list)
    for result in all_results:
        for d, articles in result.items():
            articles_by_date[d].extend(articles)
    
    # Write each day (sequentially to avoid file conflicts)
    batch_article_count = 0
    batch_sources = set()
    batch_dates = set()
    
    for target_date, articles in articles_by_date.items():
        deduped_articles = dedupe_articles(articles)
        count, sources = write_articles_to_day(output_dir, target_date, deduped_articles, batch_id)
        batch_article_count += count
        batch_sources.update(sources)
        batch_dates.add(target_date)
    
    return batch_article_count, batch_sources, batch_dates


def main():
    parser = argparse.ArgumentParser(description="Convert JSONL articles to Parquet (streaming)")
    parser.add_argument('--input-dirs', nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--workers', type=int, default=32, help="Workers for parallel JSONL reading")
    parser.add_argument('--batch-size', type=int, default=128, help="Number of JSONL files to process per batch")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find JSONL files
    jsonl_files = []
    for input_dir in args.input_dirs:
        jsonl_files.extend(Path(input_dir).glob("*.jsonl"))
    
    total_files = len(jsonl_files)
    log(f"Found {total_files} JSONL files")
    log(f"Processing in batches of {args.batch_size} files with {args.workers} workers")
    
    num_batches = (total_files + args.batch_size - 1) // args.batch_size

    touched_dates = set()
    for batch_idx in range(num_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, total_files)
        batch_files = jsonl_files[start_idx:end_idx]
        
        # Use batch index as ID for unique parquet filenames
        batch_id = f"b{batch_idx:04d}"
        
        log(f"Batch {batch_idx + 1}/{num_batches}: processing files {start_idx + 1}-{end_idx}")
        
        count, _, dates = process_batch(batch_files, output_dir, args.workers, batch_id)
        touched_dates.update(dates)

        log(f"  Batch complete: {count:,} articles, {len(dates)} days touched")

    compacted_days = 0
    duplicate_rows_removed = 0
    log("Compacting touched days...")
    for target_date in sorted(touched_dates):
        raw_count, unique_count, shard_count, changed = compact_day(output_dir, target_date)
        if changed:
            compacted_days += 1
            duplicate_rows_removed += raw_count - unique_count
            log(
                f"  {target_date}: {raw_count:,} rows -> {unique_count:,} unique "
                f"across {shard_count} shard(s)"
            )
    log(
        f"Compaction complete: rewrote {compacted_days} day(s), "
        f"removed {duplicate_rows_removed:,} duplicate rows"
    )

    # Write indices (lightweight - no per-article data)
    log("Writing indices...")
    index_dir = output_dir / "index"
    index_dir.mkdir(exist_ok=True)

    data_dir = output_dir / "data"
    found_dates = []
    total_article_count = 0
    all_sources = set()
    if data_dir.exists():
        for year_dir in data_dir.iterdir():
            if year_dir.is_dir():
                for month_dir in year_dir.iterdir():
                    if month_dir.is_dir():
                        for day_dir in month_dir.iterdir():
                            if day_dir.is_dir():
                                try:
                                    target_date = date(
                                        int(year_dir.name),
                                        int(month_dir.name),
                                        int(day_dir.name),
                                    )
                                except:
                                    pass
                                else:
                                    shard_paths = sorted(day_dir.glob("articles_*.parquet"))
                                    if not shard_paths:
                                        continue
                                    found_dates.append(target_date)
                                    for parquet_path in shard_paths:
                                        parquet_file = pq.ParquetFile(parquet_path)
                                        total_article_count += parquet_file.metadata.num_rows
                                        table = pq.read_table(parquet_path, columns=['source'])
                                        all_sources.update(
                                            value for value in table.column('source').to_pylist() if value
                                        )
    
    if found_dates:
        with open(index_dir / "date_range.json", 'w') as f:
            json.dump({
                'min': min(found_dates).isoformat(),
                'max': max(found_dates).isoformat(),
                'total_days': len(found_dates),
                'total_articles': total_article_count
            }, f, indent=2)
    
    with open(index_dir / "sources.json", 'w') as f:
        json.dump(sorted(all_sources), f, indent=2)
    
    (output_dir / "current_sim").mkdir(exist_ok=True)
    
    total_days = len(found_dates) if found_dates else len(all_dates)
    
    with open(output_dir / "README.md", 'w') as f:
        f.write(f"# Deduped Articles\n\n- Articles: {total_article_count:,}\n- Days: {total_days}\n")
        if found_dates:
            f.write(f"- Range: {min(found_dates)} to {max(found_dates)}\n")
    
    log(f"Done! {total_article_count:,} total articles, {total_days} days, {len(all_sources)} sources")


if __name__ == "__main__":
    main()
