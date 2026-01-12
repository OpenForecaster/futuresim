#!/usr/bin/env python3
"""
Convert JSONL news articles to daily-partitioned Parquet format.

Uses streaming/batched processing to minimize memory usage.

Usage:
    python convert_jsonl_to_parquet.py \
        --input-dirs /path/to/articlesuntil2024/deduped /path/to/articles2025/deduped \
        --output-dir /path/to/deduped_articles \
        --workers 8 \
        --batch-size 50
"""

import argparse
import json
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


def append_articles_to_day(output_dir: Path, target_date: date, articles: List[dict]) -> Tuple[Dict[str, str], Set[str]]:
    """
    Append articles to a day's parquet file. Creates or appends as needed.
    Returns (id_to_date mapping, sources set).
    """
    day_dir = get_day_dir(output_dir, target_date)
    day_dir.mkdir(parents=True, exist_ok=True)
    
    parquet_path = day_dir / "articles.parquet"
    headlines_path = day_dir / "headlines.json"
    
    # Load existing articles if parquet exists
    existing_articles = []
    if parquet_path.exists():
        try:
            existing_table = pq.read_table(parquet_path)
            existing_articles = existing_table.to_pylist()
        except:
            pass
    
    # Load existing headlines if exists
    existing_headlines = []
    if headlines_path.exists():
        try:
            with open(headlines_path, 'r', encoding='utf-8') as f:
                existing_headlines = json.load(f)
        except:
            pass
    
    # Get existing IDs to avoid duplicates
    existing_ids = {a['id'] for a in existing_articles}
    
    # Filter new articles to only those not already present
    new_articles = [a for a in articles if a['id'] not in existing_ids]
    
    if not new_articles:
        # No new articles to add
        return {}, set()
    
    # Combine existing + new
    all_articles = existing_articles + new_articles
    
    # Build table data
    sources = {a['source'] for a in all_articles}
    data = {
        'id': [a['id'] for a in all_articles],
        'title': [a['title'] for a in all_articles],
        'source': [a['source'] for a in all_articles],
        'date': [a['date'] if isinstance(a['date'], date) else parse_date(a['date']) for a in all_articles],
        'date_publish': [a.get('date_publish') if isinstance(a.get('date_publish'), (date, type(None))) else parse_date(a.get('date_publish')) for a in all_articles],
        'date_modify': [a.get('date_modify') if isinstance(a.get('date_modify'), (date, type(None))) else parse_date(a.get('date_modify')) for a in all_articles],
        'url': [a['url'] for a in all_articles],
        'content': [a['content'] for a in all_articles],
        'authors': [a['authors'] for a in all_articles],
        'description': [a['description'] for a in all_articles],
    }
    
    table = pa.table(data, schema=SCHEMA)
    pq.write_table(table, parquet_path, compression='zstd')
    
    # Update headlines
    new_headlines = [{'id': a['id'], 'title': a['title'], 'source': a['source']} for a in new_articles]
    all_headlines = existing_headlines + new_headlines
    with open(headlines_path, 'w', encoding='utf-8') as f:
        json.dump(all_headlines, f, ensure_ascii=False)
    
    # Return mapping only for new articles
    id_mapping = {a['id']: target_date.isoformat() for a in new_articles}
    return id_mapping, sources


def process_batch(batch_files: List[Path], output_dir: Path, workers: int) -> Tuple[Dict[str, str], Set[str], Set[date]]:
    """
    Process a batch of JSONL files: read, group by date, and write to parquet.
    Returns (id_to_date mapping, sources set, dates set).
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
    batch_id_to_date = {}
    batch_sources = set()
    batch_dates = set()
    
    for target_date, articles in articles_by_date.items():
        id_mapping, sources = append_articles_to_day(output_dir, target_date, articles)
        batch_id_to_date.update(id_mapping)
        batch_sources.update(sources)
        batch_dates.add(target_date)
    
    return batch_id_to_date, batch_sources, batch_dates


def main():
    parser = argparse.ArgumentParser(description="Convert JSONL articles to Parquet (streaming)")
    parser.add_argument('--input-dirs', nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--workers', type=int, default=8, help="Workers for parallel JSONL reading")
    parser.add_argument('--batch-size', type=int, default=50, help="Number of JSONL files to process per batch")
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
    
    # Process in batches
    all_id_to_date = {}
    all_sources = set()
    all_dates = set()
    
    num_batches = (total_files + args.batch_size - 1) // args.batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, total_files)
        batch_files = jsonl_files[start_idx:end_idx]
        
        log(f"Batch {batch_idx + 1}/{num_batches}: processing files {start_idx + 1}-{end_idx}")
        
        id_mapping, sources, dates = process_batch(batch_files, output_dir, args.workers)
        all_id_to_date.update(id_mapping)
        all_sources.update(sources)
        all_dates.update(dates)
        
        log(f"  Batch complete: {len(id_mapping):,} new articles, {len(dates)} days touched")
    
    # Write indices
    log("Writing indices...")
    index_dir = output_dir / "index"
    index_dir.mkdir(exist_ok=True)
    
    # Load existing id_to_date if exists and merge
    existing_id_to_date = {}
    id_to_date_path = index_dir / "id_to_date.json"
    if id_to_date_path.exists():
        try:
            with open(id_to_date_path, 'r') as f:
                existing_id_to_date = json.load(f)
        except:
            pass
    existing_id_to_date.update(all_id_to_date)
    
    with open(id_to_date_path, 'w') as f:
        json.dump(existing_id_to_date, f)
    
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
                'total_articles': len(existing_id_to_date)
            }, f, indent=2)
    
    # Load existing sources and merge
    sources_path = index_dir / "sources.json"
    existing_sources = set()
    if sources_path.exists():
        try:
            with open(sources_path, 'r') as f:
                existing_sources = set(json.load(f))
        except:
            pass
    all_sources.update(existing_sources)
    
    with open(sources_path, 'w') as f:
        json.dump(sorted(all_sources), f, indent=2)
    
    (output_dir / "current_sim").mkdir(exist_ok=True)
    
    total_articles = len(existing_id_to_date)
    total_days = len(found_dates) if found_dates else len(all_dates)
    
    with open(output_dir / "README.md", 'w') as f:
        f.write(f"# Deduped Articles\n\n- Articles: {total_articles:,}\n- Days: {total_days}\n")
        if found_dates:
            f.write(f"- Range: {min(found_dates)} to {max(found_dates)}\n")
    
    log(f"Done! {total_articles:,} total articles, {total_days} days, {len(all_sources)} sources")


if __name__ == "__main__":
    main()
