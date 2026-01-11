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
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Parquet schema
SCHEMA = pa.schema([
    ('id', pa.string()),
    ('title', pa.string()),
    ('source', pa.string()),
    ('date', pa.date32()),
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
        # Try common formats
        for fmt in [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d",
        ]:
            try:
                return datetime.strptime(date_val.split('+')[0].split('.')[0], fmt.replace('%z', '')).date()
            except ValueError:
                continue
    return None


def process_jsonl_file(jsonl_path: Path) -> Dict[date, List[dict]]:
    """
    Read a JSONL file and group articles by date.
    Returns: {date: [article_dicts]}
    """
    articles_by_date = defaultdict(list)
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    article = json.loads(line)
                    
                    # Parse date
                    pub_date = parse_date(article.get('date_publish'))
                    if pub_date is None:
                        continue
                    
                    # Extract and normalize fields
                    normalized = {
                        'id': article.get('id', ''),
                        'title': article.get('title', ''),
                        'source': article.get('source_domain', ''),
                        'date': pub_date,
                        'url': article.get('url', ''),
                        'content': article.get('maintext', ''),
                        'authors': article.get('authors', []) or [],
                        'description': article.get('description', ''),
                    }
                    
                    if normalized['id']:  # Only include if has ID
                        articles_by_date[pub_date].append(normalized)
                        
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON at {jsonl_path}:{line_num}")
                except Exception as e:
                    logger.warning(f"Error processing {jsonl_path}:{line_num}: {e}")
                    
    except Exception as e:
        logger.error(f"Failed to read {jsonl_path}: {e}")
    
    return dict(articles_by_date)


def write_day_parquet(
    articles: List[dict],
    output_dir: Path,
    target_date: date
) -> Dict[str, str]:
    """
    Write articles for a single day to Parquet and create headlines.json.
    Returns: {article_id: date_str} for id_to_date index
    """
    # Create directory: YYYY/MM/DD/
    day_dir = output_dir / "data" / f"{target_date.year:04d}" / f"{target_date.month:02d}" / f"{target_date.day:02d}"
    day_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for Parquet
    data = {
        'id': [a['id'] for a in articles],
        'title': [a['title'] for a in articles],
        'source': [a['source'] for a in articles],
        'date': [a['date'] for a in articles],
        'url': [a['url'] for a in articles],
        'content': [a['content'] for a in articles],
        'authors': [a['authors'] for a in articles],
        'description': [a['description'] for a in articles],
    }
    
    # Write Parquet
    table = pa.table(data, schema=SCHEMA)
    pq.write_table(table, day_dir / "articles.parquet", compression='zstd')
    
    # Write headlines.json
    headlines = [
        {'id': a['id'], 'title': a['title'], 'source': a['source']}
        for a in articles
    ]
    with open(day_dir / "headlines.json", 'w', encoding='utf-8') as f:
        json.dump(headlines, f, ensure_ascii=False)
    
    # Return id -> date mapping
    date_str = target_date.isoformat()
    return {a['id']: date_str for a in articles}


def merge_articles_by_date(
    all_results: List[Dict[date, List[dict]]]
) -> Dict[date, List[dict]]:
    """Merge article lists from multiple files by date."""
    merged = defaultdict(list)
    for result in all_results:
        for d, articles in result.items():
            merged[d].extend(articles)
    return dict(merged)


def main():
    parser = argparse.ArgumentParser(description="Convert JSONL articles to Parquet")
    parser.add_argument(
        '--input-dirs', 
        nargs='+', 
        required=True,
        help='Input directories containing JSONL files'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for Parquet structure'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of parallel workers'
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all JSONL files
    jsonl_files = []
    for input_dir in args.input_dirs:
        input_path = Path(input_dir)
        jsonl_files.extend(input_path.glob("*.jsonl"))
    
    logger.info(f"Found {len(jsonl_files)} JSONL files to process")
    
    # Phase 1: Read all files in parallel and group by date
    logger.info("Phase 1: Reading and parsing JSONL files...")
    all_results = []
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_jsonl_file, f): f for f in jsonl_files}
        
        for i, future in enumerate(as_completed(futures)):
            jsonl_file = futures[future]
            try:
                result = future.result()
                all_results.append(result)
                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i + 1}/{len(jsonl_files)} files")
            except Exception as e:
                logger.error(f"Failed to process {jsonl_file}: {e}")
    
    # Merge results by date
    logger.info("Merging articles by date...")
    articles_by_date = merge_articles_by_date(all_results)
    logger.info(f"Total unique dates: {len(articles_by_date)}")
    
    # Phase 2: Write Parquet files for each day
    logger.info("Phase 2: Writing Parquet files...")
    id_to_date = {}
    all_sources = set()
    dates = sorted(articles_by_date.keys())
    
    for i, target_date in enumerate(dates):
        articles = articles_by_date[target_date]
        
        # Track sources
        for a in articles:
            all_sources.add(a['source'])
        
        # Write day's data
        date_mapping = write_day_parquet(articles, output_dir, target_date)
        id_to_date.update(date_mapping)
        
        if (i + 1) % 100 == 0:
            logger.info(f"Written {i + 1}/{len(dates)} days")
    
    # Phase 3: Write global indices
    logger.info("Phase 3: Writing global indices...")
    index_dir = output_dir / "index"
    index_dir.mkdir(exist_ok=True)
    
    # id_to_date.json
    with open(index_dir / "id_to_date.json", 'w', encoding='utf-8') as f:
        json.dump(id_to_date, f)
    logger.info(f"Written id_to_date.json with {len(id_to_date)} entries")
    
    # sources.json
    with open(index_dir / "sources.json", 'w', encoding='utf-8') as f:
        json.dump(sorted(all_sources), f, indent=2)
    logger.info(f"Written sources.json with {len(all_sources)} sources")
    
    # date_range.json
    with open(index_dir / "date_range.json", 'w', encoding='utf-8') as f:
        json.dump({
            'min': min(dates).isoformat(),
            'max': max(dates).isoformat(),
            'total_days': len(dates),
            'total_articles': len(id_to_date)
        }, f, indent=2)
    
    # Create current_sim directory (empty, managed by env.py)
    (output_dir / "current_sim").mkdir(exist_ok=True)
    
    # Write README
    with open(output_dir / "README.md", 'w') as f:
        f.write(f"""# Deduped Articles Dataset

## Structure
- `data/YYYY/MM/DD/articles.parquet` - Daily article files
- `data/YYYY/MM/DD/headlines.json` - Headlines for quick preview
- `index/id_to_date.json` - Article ID to date mapping
- `index/sources.json` - List of all sources
- `index/date_range.json` - Date range metadata
- `current_sim/` - Symlinks for simulation (managed by env.py)

## Stats
- Date range: {min(dates)} to {max(dates)}
- Total articles: {len(id_to_date):,}
- Total sources: {len(all_sources)}
- Total days: {len(dates)}

## Generated
{datetime.now().isoformat()}
""")
    
    logger.info("Conversion complete!")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Articles: {len(id_to_date):,}")
    logger.info(f"  Sources: {len(all_sources)}")
    logger.info(f"  Date range: {min(dates)} to {max(dates)}")


if __name__ == "__main__":
    main()
