#!/usr/bin/env python3
"""
News Pipeline - Submit all processing jobs in sequence.

This script submits HTCondor jobs for the full news processing pipeline:
1. JSON → JSONL conversion
2. Deduplication
3. JSONL → Parquet conversion
4. Embedding generation
5. LanceDB index rebuild

Since some steps depend on previous outputs, run this script multiple times
after each step completes (check with condor_q).

Usage:
    # Step 1: Convert JSON to JSONL
    python run_news_pipeline.py --step jsonl
    
    # Step 2: Deduplicate (after step 1 completes)
    python run_news_pipeline.py --step dedup
    
    # Step 3: Convert to Parquet (after step 2 completes)
    python run_news_pipeline.py --step parquet
    
    # Step 4: Generate embeddings (after step 3 completes)
    python run_news_pipeline.py --step embed
    
    # Step 5: Build LanceDB index (after step 4 completes)
    python run_news_pipeline.py --step lancedb
    
    # Show status of all steps
    python run_news_pipeline.py --status
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import date

# Add parent directories to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))  # data/news
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts')))  # data/news/scripts
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))  # forecast-sim root

# Paths - configure these for your setup
NEWS_BASE = "/is/cluster/fast/sgoel/forecasting/news"

# Input: Where news-please extracts articles
RAW_ARTICLES_DIR = f"{NEWS_BASE}/filtered_cc_articles_2025_2026"

# Processing directories
JSONL_DIR = f"{NEWS_BASE}/articles_2025_2026"
DEDUPED_DIR = f"{JSONL_DIR}/deduped"

# Output: Merged with existing data
PARQUET_OUTPUT_DIR = f"{NEWS_BASE}/deduped_articles"
EMBEDDINGS_DIR = f"{NEWS_BASE}/deduped_articles/embeddings"
LANCEDB_DIR = f"{NEWS_BASE}/deduped_articles/lance"

# Date range for new articles
START_DATE = "2025-08-01"
END_DATE = "2026-01-31"

# Legacy deduped dirs to include in parquet merge
LEGACY_DEDUPED_DIRS = [
    f"{NEWS_BASE}/articlesuntil2024/deduped",
    f"{NEWS_BASE}/articles2025/deduped",
]

def check_dir_exists(path: str) -> bool:
    """Check if directory exists and has files."""
    p = Path(path)
    if not p.exists():
        return False
    if p.is_dir():
        return len(list(p.iterdir())) > 0
    return False


def show_status():
    """Show status of each processing step."""
    print("\n=== News Pipeline Status ===\n")
    
    checks = [
        ("1. Raw articles", RAW_ARTICLES_DIR),
        ("2. JSONL files", JSONL_DIR),
        ("3. Deduped JSONL", DEDUPED_DIR),
        ("4. Parquet data", f"{PARQUET_OUTPUT_DIR}/data"),
        ("5. Embeddings", f"{EMBEDDINGS_DIR}/Qwen3-Embedding-8B"),
        ("6. LanceDB index", f"{LANCEDB_DIR}/Qwen3-Embedding-8B"),
    ]
    
    for name, path in checks:
        exists = check_dir_exists(path)
        status = "✓ Ready" if exists else "✗ Missing"
        print(f"  {name}: {status}")
        print(f"    Path: {path}")
    
    print("\n  Run with --step <name> to process each step.")
    print("  Steps: jsonl, dedup, parquet, embed, lancedb\n")


def run_jsonl_step():
    """Step 1: Convert raw JSON articles to JSONL."""
    print("\n=== Step 1: JSON → JSONL Conversion ===")
    
    if not check_dir_exists(RAW_ARTICLES_DIR):
        print(f"ERROR: Raw articles directory does not exist: {RAW_ARTICLES_DIR}")
        print("Run news-please first to extract articles from CCNews.")
        return False
    
    from launch_jsonl_conversion import launch_jsonl_conversion_job
    
    launch_jsonl_conversion_job(
        json_dir=RAW_ARTICLES_DIR,
        output_dir=JSONL_DIR,
        workers=48,
        verify=0.1,
        delete=False,
        job_cpus=48,
    )
    
    print(f"\nJob submitted. Check progress with: condor_q")
    print(f"Output will be in: {JSONL_DIR}")
    return True


def run_dedup_step():
    """Step 2: Deduplicate JSONL files."""
    print("\n=== Step 2: Deduplication ===")
    
    if not check_dir_exists(JSONL_DIR):
        print(f"ERROR: JSONL directory does not exist: {JSONL_DIR}")
        print("Run step 1 (jsonl) first.")
        return False
    
    from launch_dedup import launch_dedup_job
    
    launch_dedup_job(
        jsonl_path=JSONL_DIR,
        num_workers=16,
        job_cpus=48,
    )
    
    print(f"\nJob submitted. Check progress with: condor_q")
    print(f"Output will be in: {DEDUPED_DIR}")
    return True


def run_parquet_step():
    """Step 3: Convert deduped JSONL to Parquet (merge with existing)."""
    print("\n=== Step 3: Parquet Conversion ===")
    
    if not check_dir_exists(DEDUPED_DIR):
        print(f"ERROR: Deduped directory does not exist: {DEDUPED_DIR}")
        print("Run step 2 (dedup) first.")
        return False
    
    # Collect all deduped directories
    input_dirs = [DEDUPED_DIR]
    for legacy_dir in LEGACY_DEDUPED_DIRS:
        if check_dir_exists(legacy_dir):
            input_dirs.append(legacy_dir)
    
    print(f"Input directories: {input_dirs}")
    
    from launch_parquet_conversion import launch_parquet_conversion_job
    
    launch_parquet_conversion_job(
        input_dirs=input_dirs,
        output_dir=PARQUET_OUTPUT_DIR,
        workers=32,
        batch_size=128,
        job_cpus=48,
    )
    
    print(f"\nJob submitted. Check progress with: condor_q")
    print(f"Output will be in: {PARQUET_OUTPUT_DIR}")
    return True


def run_embed_step():
    """Step 4: Generate embeddings."""
    import subprocess
    
    print("\n=== Step 4: Embedding Generation ===")
    
    parquet_data_dir = f"{PARQUET_OUTPUT_DIR}/data"
    if not check_dir_exists(parquet_data_dir):
        print(f"ERROR: Parquet data directory does not exist: {parquet_data_dir}")
        print("Run step 3 (parquet) first.")
        return False
    
    # Use subprocess to run the embed submit script
    cmd = [
        sys.executable, 
        '/home/sgoel/forecast-sim/mpi_scripts/embed/submit_job.py',
        '--gpus', '1',
        '--memory', '64',
        '--bid', '25',
        '--num_workers', '8',
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    
    print(f"\nOutput will be in: {EMBEDDINGS_DIR}")
    return True


def run_lancedb_step():
    """Step 5: Build LanceDB index."""
    import subprocess
    
    print("\n=== Step 5: LanceDB Index Build ===")
    
    emb_dir = f"{EMBEDDINGS_DIR}/Qwen3-Embedding-8B"
    if not check_dir_exists(emb_dir):
        print(f"ERROR: Embeddings directory does not exist: {emb_dir}")
        print("Run step 4 (embed) first.")
        return False
    
    # Use condor_submit_bid to submit the lancedb job
    cmd = [
        'condor_submit_bid', '15',
        '/home/sgoel/forecast-sim/mpi_scripts/build_lancedb/build_lancedb.sub'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    
    print(f"\nOutput will be in: {LANCEDB_DIR}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="News Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--step', type=str, 
                       choices=['jsonl', 'dedup', 'parquet', 'embed', 'lancedb'],
                       help="Which step to run")
    parser.add_argument('--status', action='store_true',
                       help="Show status of all steps")
    
    args = parser.parse_args()
    
    if args.status or (not args.step):
        show_status()
        return
    
    step_funcs = {
        'jsonl': run_jsonl_step,
        'dedup': run_dedup_step,
        'parquet': run_parquet_step,
        'embed': run_embed_step,
        'lancedb': run_lancedb_step,
    }
    
    step_func = step_funcs[args.step]
    success = step_func()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
