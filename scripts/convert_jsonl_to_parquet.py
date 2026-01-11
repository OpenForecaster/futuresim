#!/usr/bin/env python3
"""
Convert JSONL news articles to daily-partitioned Parquet format.

Usage:
    python convert_jsonl_to_parquet.py \
        --input-dirs /path/to/articlesuntil2024/deduped /path/to/articles2025/deduped \
        --output-dir /path/to/deduped_articles \
        --workers 8
"""

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any

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


def write_day_parquet_worker(args: tuple) -> tuple:
    """Write articles for a single day to Parquet. Returns (id_to_date, sources)."""
    articles, output_dir_str, target_date = args
    output_dir = Path(output_dir_str)
    day_dir = output_dir / "data" / f"{target_date.year:04d}" / f"{target_date.month:02d}" / f"{target_date.day:02d}"
    day_dir.mkdir(parents=True, exist_ok=True)
    
    sources = {a['source'] for a in articles}
    data = {k: [a[k] for a in articles] for k in ['id', 'title', 'source', 'date', 'url', 'content', 'authors', 'description']}
    
    table = pa.table(data, schema=SCHEMA)
    pq.write_table(table, day_dir / "articles.parquet", compression='zstd')
    
    headlines = [{'id': a['id'], 'title': a['title'], 'source': a['source']} for a in articles]
    with open(day_dir / "headlines.json", 'w', encoding='utf-8') as f:
        json.dump(headlines, f, ensure_ascii=False)
    
    return {a['id']: target_date.isoformat() for a in articles}, sources


def main():
    parser = argparse.ArgumentParser(description="Convert JSONL articles to Parquet")
    parser.add_argument('--input-dirs', nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find JSONL files
    jsonl_files = []
    for input_dir in args.input_dirs:
        jsonl_files.extend(Path(input_dir).glob("*.jsonl"))
    log(f"Found {len(jsonl_files)} JSONL files, using {args.workers} workers")
    
    # Phase 1: Read JSONL files
    log("Phase 1: Reading JSONL files...")
    all_results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_jsonl_file, f): f for f in jsonl_files}
        for i, future in enumerate(as_completed(futures), 1):
            all_results.append(future.result())
            if i % 50 == 0 or i == len(jsonl_files):
                log(f"  Read {i}/{len(jsonl_files)} files")
    
    # Merge by date
    articles_by_date = defaultdict(list)
    for result in all_results:
        for d, articles in result.items():
            articles_by_date[d].extend(articles)
    dates = sorted(articles_by_date.keys())
    log(f"Phase 1 done: {sum(len(v) for v in articles_by_date.values()):,} articles across {len(dates)} days")
    
    # Phase 2: Write Parquet files
    log("Phase 2: Writing Parquet files...")
    id_to_date = {}
    all_sources = set()
    write_tasks = [(articles_by_date[d], str(output_dir), d) for d in dates]
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(write_day_parquet_worker, t): t[2] for t in write_tasks}
        for i, future in enumerate(as_completed(futures), 1):
            mapping, sources = future.result()
            id_to_date.update(mapping)
            all_sources.update(sources)
            if i % 100 == 0 or i == len(dates):
                log(f"  Written {i}/{len(dates)} days")
    log(f"Phase 2 done: {len(id_to_date):,} articles written")
    
    # Phase 3: Write indices
    log("Phase 3: Writing indices...")
    index_dir = output_dir / "index"
    index_dir.mkdir(exist_ok=True)
    
    with open(index_dir / "id_to_date.json", 'w') as f:
        json.dump(id_to_date, f)
    with open(index_dir / "sources.json", 'w') as f:
        json.dump(sorted(all_sources), f, indent=2)
    with open(index_dir / "date_range.json", 'w') as f:
        json.dump({'min': min(dates).isoformat(), 'max': max(dates).isoformat(), 
                   'total_days': len(dates), 'total_articles': len(id_to_date)}, f, indent=2)
    
    (output_dir / "current_sim").mkdir(exist_ok=True)
    
    with open(output_dir / "README.md", 'w') as f:
        f.write(f"# Deduped Articles\n\n- Articles: {len(id_to_date):,}\n- Days: {len(dates)}\n- Range: {min(dates)} to {max(dates)}\n")
    
    log(f"Done! {len(id_to_date):,} articles, {len(dates)} days, {len(all_sources)} sources")


if __name__ == "__main__":
    main()
