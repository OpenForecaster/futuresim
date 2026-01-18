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
from datetime import date, datetime, timedelta
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


# Default paths
DEFAULT_ARTICLES_DIR = "/is/cluster/fast/sgoel/forecasting/news/deduped_articles/data"
DEFAULT_EMBEDDINGS_DIR = "/is/cluster/fast/sgoel/forecasting/news/deduped_articles/embeddings"
DEFAULT_OUTPUT_DIR = "/is/cluster/fast/sgoel/forecasting/news/deduped_articles/lance"
DEFAULT_MODEL = "Qwen3-Embedding-8B"


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
    for parquet_file in date_dir.glob("articles_*.parquet"):
        try:
            table = pq.read_table(parquet_file)
            df = table.to_pandas()
            
            for _, row in df.iterrows():
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
    
    return articles


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
            record = {
                'chunk_id': chunk_id,
                'article_id': article['id'],
                'chunk_index': chunk_idx,
                'title': article['title'],
                'source': article['source'],
                'date': article['date'],  # Download date (for filtering)
                'date_publish': article.get('date_publish'),  # Publish date
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
    
    if table_name in db.table_names():
        table = db.open_table(table_name)
        table.add(records)
    else:
        db.create_table(table_name, records)


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
    if args.overwrite and table_name in db.table_names():
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
    
    print(f"\nDone! Created {total_chunks} chunks in {db_path}/{table_name}")
    
    # Create FTS index for keyword search
    if table_name in db.table_names():
        print("Creating full-text search index...")
        table = db.open_table(table_name)
        try:
            table.create_fts_index("content")
            print("FTS index created successfully")
        except Exception as e:
            print(f"Warning: Could not create FTS index: {e}")
    
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
