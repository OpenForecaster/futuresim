#!/usr/bin/env python
"""
Embed articles for semantic search.

Creates embeddings for article chunks and saves them organized by model name.
Designed for multi-GPU parallel processing via HTCondor array jobs.

Output structure:
  {output_dir}/{model_name}/{YYYY}/{MM}/{DD}/embeddings.npz
  
Each .npz file contains:
  - chunk_ids: array of chunk IDs (article_id_chunkN)
  - embeddings: array of embedding vectors
  - metadata: dict with article info per chunk

Usage:
  # Single process (for testing):
  python scripts/embed_articles.py --start_date 2023-01-01 --end_date 2023-01-31

  # Multi-GPU via HTCondor (see embed_job.sub):
  condor_submit_bid 25 scripts/embed_job.sub -queue 8
"""

import argparse
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json



import numpy as np
import pyarrow.parquet as pq

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.search_tools.chunking import chunk_article


# Default paths
DEFAULT_ARTICLES_DIR = "/is/cluster/fast/sgoel/forecasting/news/deduped_articles/data"
DEFAULT_OUTPUT_DIR = "/is/cluster/fast/sgoel/forecasting/news/deduped_articles/embeddings"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Fast default, user can specify Qwen


def parse_args():
    parser = argparse.ArgumentParser(
        description="Embed article chunks for semantic search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Date range
    parser.add_argument(
        "--start_date", type=str, required=True,
        help="Start date (YYYY-MM-DD). Default embeds from 2023-01-01."
    )
    parser.add_argument(
        "--end_date", type=str, required=True,
        help="End date (YYYY-MM-DD) inclusive."
    )
    
    # Model configuration
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Embedding model name/path. Default: {DEFAULT_MODEL}"
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Local path to model weights (for Qwen etc). If not set, uses model name from HF."
    )
    
    # Paths
    parser.add_argument(
        "--articles_dir", type=str, default=DEFAULT_ARTICLES_DIR,
        help="Path to articles directory (with YYYY/MM/DD structure)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Output directory for embeddings"
    )
    
    # Chunking
    parser.add_argument(
        "--chunk_tokens", type=int, default=512,
        help="Max tokens per chunk (default: 512)"
    )
    
    # Processing
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for embedding (default: 32)"
    )
    parser.add_argument(
        "--worker_id", type=int, default=0,
        help="Worker ID for parallel processing (0-indexed)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=1,
        help="Total number of parallel workers"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip days that already have embeddings"
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


def get_worker_dates(all_dates: List[date], worker_id: int, num_workers: int) -> List[date]:
    """Get dates assigned to this worker (round-robin distribution)."""
    return [d for i, d in enumerate(all_dates) if i % num_workers == worker_id]


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
                    'date': row.get('date'),
                    'content': row.get('content', ''),
                    'url': row.get('url', ''),
                    'description': row.get('description', ''),
                })
        except Exception as e:
            print(f"Error reading {parquet_file}: {e}")
    
    return articles


def create_chunks_for_articles(
    articles: List[Dict], 
    max_tokens: int = 512
) -> Tuple[List[str], List[str], List[Dict]]:
    """
    Create chunks for all articles.
    
    Returns:
        (chunk_ids, chunk_texts, chunk_metadata)
    """
    chunk_ids = []
    chunk_texts = []
    chunk_metadata = []
    
    for article in articles:
        chunks = chunk_article(
            article_id=article['id'],
            title=article['title'],
            content=article['content'],
            max_tokens=max_tokens
        )
        
        for chunk_id, chunk_text in chunks:
            chunk_ids.append(chunk_id)
            chunk_texts.append(chunk_text)
            chunk_metadata.append({
                'article_id': article['id'],
                'title': article['title'],
                'source': article['source'],
                'date': str(article['date']) if article['date'] else None,
                'url': article['url'],
            })
    
    return chunk_ids, chunk_texts, chunk_metadata


def load_embedding_model(model_path: str):
    """Load embedding model using vLLM (V0 engine, see notes/embed_aisa_changes.md)."""
    from vllm import LLM
    return LLM(model=model_path, task="embed", trust_remote_code=True)


def embed_texts(texts: List[str], model, batch_size: int = 32) -> np.ndarray:
    """Embed texts using vLLM.

    Per official Qwen3-Embedding docs: NO instruction prefix for documents.
    Only queries need instruction prefix.
    """
    # Truncate very long texts by character count (safety net)
    texts = [t[:30000] for t in texts]

    # vLLM V0 handles batching internally — pass all texts at once
    outputs = model.embed(texts)
    embeddings = np.array(
        [o.outputs.embedding for o in outputs], dtype=np.float32
    )
    return embeddings


def get_model_dirname(model_name: str) -> str:
    """Convert model name to valid directory name."""
    # Replace slashes and special chars
    return model_name.replace("/", "_").replace(":", "_").replace(" ", "_")


def save_embeddings(
    output_dir: str,
    model_name: str,
    target_date: date,
    chunk_ids: List[str],
    embeddings: np.ndarray,
    chunk_metadata: List[Dict]
):
    """Save embeddings to .npz file."""
    model_dirname = get_model_dirname(model_name)
    date_dir = Path(output_dir) / model_dirname / target_date.strftime("%Y/%m/%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = date_dir / "embeddings.npz"
    
    np.savez_compressed(
        output_path,
        chunk_ids=np.array(chunk_ids),
        embeddings=embeddings,
        metadata=json.dumps(chunk_metadata)
    )
    
    print(f"Saved {len(chunk_ids)} embeddings to {output_path}")


def check_existing(output_dir: str, model_name: str, target_date: date) -> bool:
    """Check if embeddings already exist for this date."""
    model_dirname = get_model_dirname(model_name)
    output_path = Path(output_dir) / model_dirname / target_date.strftime("%Y/%m/%d") / "embeddings.npz"
    return output_path.exists()


def save_config(
    output_dir: str,
    model_name: str,
    model_path: str,
    chunk_tokens: int,
    embedding_dim: int
):
    """
    Save embedding config to config.json in the model directory.
    
    This config is read by the search tool to ensure consistent chunking.
    """
    model_dirname = get_model_dirname(model_name)
    config_path = Path(output_dir) / model_dirname / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    config = {
        "model_name": model_name,
        "model_path": model_path,
        "chunk_tokens": chunk_tokens,
        "embedding_dim": embedding_dim,
    }
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Saved config to {config_path}")



def main():
    args = parse_args()
    
    # Get dates to process
    all_dates = get_date_range(args.start_date, args.end_date)
    worker_dates = get_worker_dates(all_dates, args.worker_id, args.num_workers)
    
    print(f"Worker {args.worker_id}/{args.num_workers}: Processing {len(worker_dates)} days")
    print(f"Model: {args.model}")
    print(f"Output: {args.output_dir}/{get_model_dirname(args.model)}/")
    
    # Load model
    print("Loading embedding model...")
    model = load_embedding_model(args.model_path or args.model)
    print(f"Model loaded")
    
    # Save config (worker 0 only to avoid race)
    # Note: embedding_dim will be determined from first batch and saved later
    config_saved = False
    
    # Process each date
    for target_date in worker_dates:
        print(f"\n{'='*60}")
        print(f"Processing {target_date}")
        
        if args.resume and check_existing(args.output_dir, args.model, target_date):
            print(f"  Skipping (already exists)")
            continue
        
        articles = load_articles_for_date(args.articles_dir, target_date)
        if not articles:
            print(f"  No articles found")
            continue
        
        print(f"  Loaded {len(articles)} articles")
        
        chunk_ids, chunk_texts, chunk_metadata = create_chunks_for_articles(articles, max_tokens=args.chunk_tokens)
        print(f"  Created {len(chunk_ids)} chunks")
        
        if not chunk_ids:
            continue
        
        print(f"  Embedding...")
        embeddings = embed_texts(chunk_texts, model, args.batch_size)
        print(f"  Embedding shape: {embeddings.shape}")
        
        # Save config on first successful embedding (worker 0 only)
        if not config_saved and args.worker_id == 0:
            save_config(args.output_dir, args.model, args.model_path or "", args.chunk_tokens, embeddings.shape[1])
            config_saved = True
        
        # Save
        save_embeddings(
            args.output_dir, args.model, target_date,
            chunk_ids, embeddings, chunk_metadata
        )
    
    print(f"\nDone! Worker {args.worker_id} complete.")


if __name__ == "__main__":
    main()
