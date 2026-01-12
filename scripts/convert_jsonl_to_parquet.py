#!/usr/bin/env python3
"""
Convert JSONL news articles to daily-partitioned Parquet format.

Uses streaming/batched processing to minimize memory usage.
Each batch writes separate parquet files - no reloading of existing data.
ID mappings are written incrementally per batch to avoid memory accumulation.

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
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


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


def get_day_dir(output_dir: Path, target_date: date) -> Path:
    """Get the directory path for a specific day."""
    return output_dir / "data" / f"{target_date.year:04d}" / f"{target_date.month:02d}" / f"{target_date.day:02d}"


def write_articles_to_day(output_dir: Path, target_date: date, articles: List[dict], batch_id: str) -> Tuple[int, Set[str]]:
    """
    Write articles to a new parquet file for this batch.
    Also writes headlines to a batch-specific JSON file.
    Returns (article_count, sources set).
    """
    if not articles:
        return 0, set()
    
    day_dir = get_day_dir(output_dir, target_date)
    day_dir.mkdir(parents=True, exist_ok=True)
    
    # Write to a unique parquet file for this batch
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
        count, sources = write_articles_to_day(output_dir, target_date, articles, batch_id)
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
    
    # Process in batches - only track aggregate counts, not individual IDs
    total_article_count = 0
    all_sources = set()
    all_dates = set()
    
    num_batches = (total_files + args.batch_size - 1) // args.batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, total_files)
        batch_files = jsonl_files[start_idx:end_idx]
        
        # Use batch index as ID for unique parquet filenames
        batch_id = f"b{batch_idx:04d}"
        
        log(f"Batch {batch_idx + 1}/{num_batches}: processing files {start_idx + 1}-{end_idx}")
        
        count, sources, dates = process_batch(batch_files, output_dir, args.workers, batch_id)
        total_article_count += count
        all_sources.update(sources)
        all_dates.update(dates)
        
        log(f"  Batch complete: {count:,} articles, {len(dates)} days touched")
    
    # Write indices (lightweight - no per-article data)
    log("Writing indices...")
    index_dir = output_dir / "index"
    index_dir.mkdir(exist_ok=True)
    
    # Scan for all dates in output to get accurate range
    data_dir = output_dir / "data"
    found_dates = []
    if data_dir.exists():
        for year_dir in data_dir.iterdir():
            if year_dir.is_dir():
                for month_dir in year_dir.iterdir():
                    if month_dir.is_dir():
                        for day_dir in month_dir.iterdir():
                            if day_dir.is_dir():
                                try:
                                    found_dates.append(date(
                                        int(year_dir.name),
                                        int(month_dir.name),
                                        int(day_dir.name)
                                    ))
                                except:
                                    pass
    
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
