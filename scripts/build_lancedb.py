#!/usr/bin/env python
"""
Build LanceDB index from articles and precomputed embeddings.

Creates a LanceDB table with all article chunks, embeddings, and metadata.
Supports date filtering at query time via the 'date' column.

Output:
  {output_dir}/lance/{model_name}/articles.lance

Usage:
  # Build index from precomputed embeddings:
  python scripts/build_lancedb.py --start_date 2023-01-01 --end_date 2025-12-31

  # Build without embeddings (keyword search only):
  python scripts/build_lancedb.py --start_date 2023-01-01 --end_date 2025-12-31 --no_embeddings
"""

import argparse
import os
import sys
import gc
import queue
import traceback
import multiprocessing as mp
from datetime import date, datetime, timedelta, time
from pathlib import Path
from typing import List, Dict, Optional
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import lancedb

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.search_tools.chunking import chunk_article
from scripts.news_article_dedup import dedupe_articles


# Default paths
DEFAULT_NEWS_BASE = os.getenv("FSIM_NEWS_BASE", "/is/cluster/fast/sgoel/forecasting/news")
DEFAULT_ARTICLES_DIR = os.getenv("FSIM_NEWS_ARTICLES_DIR", f"{DEFAULT_NEWS_BASE}/deduped_articles/data")
DEFAULT_EMBEDDINGS_DIR = os.getenv("FSIM_NEWS_EMBEDDINGS_DIR", f"{DEFAULT_NEWS_BASE}/deduped_articles/embeddings")
DEFAULT_OUTPUT_DIR = os.getenv("FSIM_NEWS_LANCEDB_DIR", f"{DEFAULT_NEWS_BASE}/deduped_articles/lance")
DEFAULT_MODEL = "Qwen3-Embedding-8B"


def list_tables(db) -> List[str]:
    """Compatibility wrapper for LanceDB table listing."""
    if hasattr(db, "list_tables"):
        tables = db.list_tables()
        # LanceDB 0.26.x returns a ListTablesResponse object, older versions may return list[str].
        if isinstance(tables, list):
            return tables
        if hasattr(tables, "tables"):
            return list(tables.tables or [])
        try:
            return list(tables)
        except TypeError:
            pass
    return db.table_names()


def table_exists(db, table_name: str) -> bool:
    return table_name in list_tables(db)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build LanceDB index from articles and embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Date range
    parser.add_argument(
        "--start_date", type=str, required=True,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end_date", type=str, required=True,
        help="End date (YYYY-MM-DD) inclusive"
    )
    
    # Paths
    parser.add_argument(
        "--articles_dir", type=str, default=DEFAULT_ARTICLES_DIR,
        help="Path to articles directory (with YYYY/MM/DD structure)"
    )
    parser.add_argument(
        "--embeddings_dir", type=str, default=DEFAULT_EMBEDDINGS_DIR,
        help="Path to embeddings directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Output directory for LanceDB"
    )
    
    # Model (for embedding directory lookup)
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help="Embedding model name (used to find embedding directory)"
    )
    
    # Options
    parser.add_argument(
        "--no_embeddings", action="store_true",
        help="Build index without embeddings (keyword search only)"
    )
    parser.add_argument(
        "--chunk_tokens", type=int, default=512,
        help="Max tokens per chunk (default: 512)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=10000,
        help="Batch size for inserting into LanceDB"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing LanceDB table"
    )
    parser.add_argument(
        "--skip_scalar_index", action="store_true",
        help="Skip scalar index creation on 'date' after ingest"
    )
    parser.add_argument(
        "--skip_fts_index", action="store_true",
        help="Skip full-text index creation on 'content' after ingest"
    )
    fts_position_group = parser.add_mutually_exclusive_group()
    fts_position_group.add_argument(
        "--fts_with_position",
        dest="fts_with_position",
        action="store_true",
        help="Enable position indexing for FTS (default)",
    )
    fts_position_group.add_argument(
        "--no_fts_with_position",
        dest="fts_with_position",
        action="store_false",
        help="Disable phrase-position indexing for FTS (lower memory usage)",
    )
    parser.add_argument(
        "--scalar_index_timeout_minutes", type=int, default=30,
        help="Timeout in minutes for scalar index build subprocess (0 disables timeout)"
    )
    parser.add_argument(
        "--fts_index_timeout_minutes", type=int, default=240,
        help="Timeout in minutes for FTS index build subprocess (0 disables timeout)"
    )
    
    parser.set_defaults(fts_with_position=True)
    return parser.parse_args()


def get_date_range(start_date: str, end_date: str) -> List[date]:
    """Generate list of dates in range."""
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    
    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=1)
    
    return dates


def get_model_dirname(model_name: str) -> str:
    """Convert model name to valid directory name."""
    return model_name.replace("/", "_").replace(":", "_").replace(" ", "_")


def load_embeddings_for_date(
    embeddings_dir: str, 
    model_name: str, 
    target_date: date
) -> Optional[Dict]:
    """Load precomputed embeddings for a date."""
    model_dirname = get_model_dirname(model_name)
    emb_path = Path(embeddings_dir) / model_dirname / target_date.strftime("%Y/%m/%d") / "embeddings.npz"
    
    if not emb_path.exists():
        return None
    
    data = np.load(emb_path, allow_pickle=True)
    
    return {
        'chunk_ids': data['chunk_ids'].tolist(),
        'embeddings': data['embeddings'],
        'metadata': json.loads(str(data['metadata']))
    }


def load_articles_for_date(articles_dir: str, target_date: date) -> List[Dict]:
    """Load all articles for a specific date from parquet files."""
    date_dir = Path(articles_dir) / target_date.strftime("%Y/%m/%d")
    
    if not date_dir.exists():
        return []
    
    articles = []
    for parquet_file in sorted(date_dir.glob("articles_*.parquet")):
        try:
            table = pq.read_table(parquet_file)
            for row in table.to_pylist():
                articles.append({
                    'id': row.get('id', ''),
                    'title': row.get('title', ''),
                    'source': row.get('source', ''),
                    'date': row.get('date'),  # Download/processing date
                    'date_publish': row.get('date_publish'),  # Article publish date
                    'content': row.get('content', ''),
                    'url': row.get('url', ''),
                    'description': row.get('description', ''),
                })
        except Exception as e:
            print(f"Error reading {parquet_file}: {e}")

    return dedupe_articles(articles)


def create_chunks_with_embeddings(
    articles: List[Dict],
    embeddings_data: Optional[Dict],
    max_tokens: int = 512
) -> List[Dict]:
    """Create chunk records with embeddings if available."""
    records = []
    
    # Build embedding lookup if available
    emb_lookup = {}
    if embeddings_data:
        for i, chunk_id in enumerate(embeddings_data['chunk_ids']):
            emb_lookup[chunk_id] = embeddings_data['embeddings'][i]
    
    for article in articles:
        chunks = chunk_article(
            article_id=article['id'],
            title=article['title'],
            content=article['content'],
            max_tokens=max_tokens
        )
        
        for chunk_idx, (chunk_id, chunk_text) in enumerate(chunks):
            # Convert date to datetime for LanceDB compatibility
            # (LanceDB bug #1636: date type doesn't work with hybrid/FTS search filters)
            article_date = article['date']
            if article_date and isinstance(article_date, date) and not isinstance(article_date, datetime):
                article_date = datetime.combine(article_date, time.min)
            
            article_date_publish = article.get('date_publish')
            if article_date_publish and isinstance(article_date_publish, date) and not isinstance(article_date_publish, datetime):
                article_date_publish = datetime.combine(article_date_publish, time.min)
            
            record = {
                'chunk_id': chunk_id,
                'article_id': article['id'],
                'chunk_index': chunk_idx,
                'title': article['title'],
                'source': article['source'],
                'date': article_date,  # Download date (for filtering) - stored as timestamp
                'date_publish': article_date_publish,  # Publish date - stored as timestamp
                'content': chunk_text,  # Full chunk content for FTS
                'url': article['url'],
            }
            
            # Add embedding if available
            if chunk_id in emb_lookup:
                record['vector'] = emb_lookup[chunk_id].tolist()
            
            records.append(record)
    
    return records


def create_lancedb_table(db, table_name: str, records: List[Dict], has_embeddings: bool):
    """Create or append to LanceDB table."""
    if not records:
        return
    
    # LanceDB will infer schema from first batch
    # For subsequent batches, it will validate against existing schema
    
    if table_exists(db, table_name):
        table = db.open_table(table_name)
        table.add(records)
    else:
        db.create_table(table_name, records)


def _index_worker(db_path: str, table_name: str, step: str, fts_with_position: bool, result_queue) -> None:
    """Run a single index step in an isolated process."""
    try:
        db = lancedb.connect(db_path)
        table = db.open_table(table_name)
        if step == "scalar_date":
            table.create_scalar_index("date", replace=True)
        elif step == "fts":
            table.create_fts_index("content", with_position=fts_with_position, replace=True)
        else:
            raise ValueError(f"Unknown index step: {step}")
        result_queue.put(("ok", ""))
    except Exception:
        result_queue.put(("error", traceback.format_exc()))
        raise


def run_index_step_with_timeout(
    db_path: Path,
    table_name: str,
    step: str,
    timeout_minutes: int,
    fts_with_position: bool = False,
) -> tuple[bool, str]:
    """Run index creation in a subprocess and enforce timeout."""
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_index_worker,
        args=(str(db_path), table_name, step, fts_with_position, result_queue),
        daemon=False,
    )
    proc.start()

    timeout_seconds = None if timeout_minutes <= 0 else timeout_minutes * 60
    proc.join(timeout_seconds)

    if proc.is_alive():
        proc.terminate()
        proc.join(30)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return False, f"timed out after {timeout_minutes} minute(s)"

    try:
        status, details = result_queue.get_nowait()
    except queue.Empty:
        status, details = ("error", f"index subprocess exited with code {proc.exitcode}")

    if proc.exitcode == 0 and status == "ok":
        return True, ""
    if details:
        return False, details.strip()
    return False, f"index subprocess exited with code {proc.exitcode}"


def main():
    args = parse_args()
    
    # Setup paths
    model_dirname = get_model_dirname(args.model)
    db_path = Path(args.output_dir) / model_dirname
    db_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Building LanceDB index")
    print(f"  Date range: {args.start_date} to {args.end_date}")
    print(f"  Model: {args.model}")
    print(f"  Output: {db_path}")
    print(f"  Embeddings: {'disabled' if args.no_embeddings else 'enabled'}")
    
    # Connect to LanceDB
    db = lancedb.connect(str(db_path))
    table_name = "articles"
    
    # Handle overwrite
    if args.overwrite and table_exists(db, table_name):
        print(f"  Dropping existing table '{table_name}'")
        db.drop_table(table_name)
    
    # Get dates to process
    all_dates = get_date_range(args.start_date, args.end_date)
    print(f"  Processing {len(all_dates)} days")
    
    # Process in batches
    batch_records = []
    total_chunks = 0
    has_embeddings = not args.no_embeddings
    
    for i, target_date in enumerate(all_dates):
        if i % 30 == 0:
            print(f"\n  Processing {target_date} ({i+1}/{len(all_dates)})")
        
        # Load articles
        articles = load_articles_for_date(args.articles_dir, target_date)
        if not articles:
            continue
        
        # Load embeddings if enabled
        embeddings_data = None
        if has_embeddings:
            embeddings_data = load_embeddings_for_date(
                args.embeddings_dir, args.model, target_date
            )
            if not embeddings_data:
                # Skip if embeddings expected but not found
                print(f"    Warning: No embeddings for {target_date}")
        
        # Create chunk records
        records = create_chunks_with_embeddings(
            articles, embeddings_data, args.chunk_tokens
        )
        
        batch_records.extend(records)
        total_chunks += len(records)
        
        # Insert batch when large enough
        if len(batch_records) >= args.batch_size:
            print(f"    Inserting batch of {len(batch_records)} records...")
            create_lancedb_table(db, table_name, batch_records, has_embeddings)
            batch_records = []
    
    # Insert remaining records
    if batch_records:
        print(f"  Inserting final batch of {len(batch_records)} records...")
        create_lancedb_table(db, table_name, batch_records, has_embeddings)
        batch_records = []
    
    print(f"\nDone! Created {total_chunks} chunks in {db_path}/{table_name}")
    gc.collect()
    
    # Build indices in isolated subprocesses to avoid deadlocks/hangs in the main ingest process.
    if table_exists(db, table_name):
        if args.skip_scalar_index:
            print("Skipping scalar index on 'date' (--skip_scalar_index)")
        else:
            print(
                f"Creating scalar index on 'date' (timeout: {args.scalar_index_timeout_minutes} min)...",
                flush=True,
            )
            ok, err = run_index_step_with_timeout(
                db_path=db_path,
                table_name=table_name,
                step="scalar_date",
                timeout_minutes=args.scalar_index_timeout_minutes,
            )
            if ok:
                print("Scalar index on 'date' created successfully")
            else:
                print(f"Warning: Scalar index creation failed: {err}")

        if args.skip_fts_index:
            print("Skipping full-text index on 'content' (--skip_fts_index)")
        else:
            print(
                "Creating full-text search index on 'content' "
                f"(with_position={args.fts_with_position}, timeout: {args.fts_index_timeout_minutes} min)...",
                flush=True,
            )
            ok, err = run_index_step_with_timeout(
                db_path=db_path,
                table_name=table_name,
                step="fts",
                timeout_minutes=args.fts_index_timeout_minutes,
                fts_with_position=args.fts_with_position,
            )
            if ok:
                mode = "with positions" if args.fts_with_position else "without positions"
                print(f"FTS index created successfully ({mode})")
            else:
                print(
                    "Warning: FTS index creation failed. "
                    f"Hybrid/keyword search may be unavailable until this is built separately. Details: {err}"
                )
    
    # Save config for LanceDB store
    config = {
        "model": args.model,
        "chunk_tokens": args.chunk_tokens,
        "has_embeddings": has_embeddings,
    }
    config_path = db_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")


if __name__ == "__main__":
    main()
