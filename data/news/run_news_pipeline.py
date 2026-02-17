#!/usr/bin/env python3
"""
News Pipeline - Submit all processing jobs in sequence.

This script submits HTCondor jobs for the full news processing pipeline:
1. JSON → JSONL conversion
2. Deduplication
3. JSONL → Parquet conversion
4. Embedding generation
5. LanceDB table + scalar date index (stage 1/2)
6. LanceDB FTS + IVF-PQ vector index (stage 2/2)

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
    
    # Step 5: Build LanceDB table + scalar date index only (after step 4 completes)
    python run_news_pipeline.py --step lancedb

    # Step 6: Build/refresh FTS (with positions) + vector index
    #         (after step 5 completes)
    python run_news_pipeline.py --step lancedb_index
    
    # Show status of all steps
    python run_news_pipeline.py --status
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import date, datetime

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


def get_latest_mtime(path: str, recursive: bool = False):
    """
    Return newest mtime (epoch seconds) for path contents.
    If recursive=False, only checks direct children for directories.
    """
    p = Path(path)
    if not p.exists():
        return None
    if p.is_file():
        return p.stat().st_mtime

    latest = None
    if recursive:
        for root, dirs, files in os.walk(p):
            # Ignore hidden/cache dirs (e.g. .cache/huggingface upload metadata).
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for fname in files:
                if fname.startswith("."):
                    continue
                fp = Path(root) / fname
                try:
                    ts = fp.stat().st_mtime
                except OSError:
                    continue
                if latest is None or ts > latest:
                    latest = ts
    else:
        for child in p.iterdir():
            if child.name.startswith("."):
                continue
            try:
                ts = child.stat().st_mtime
            except OSError:
                continue
            if latest is None or ts > latest:
                latest = ts

    # Fallback to dir mtime if empty
    if latest is None:
        return p.stat().st_mtime
    return latest


def fmt_mtime(ts):
    if ts is None:
        return "n/a"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def get_lancedb_index_flags(db_dir: str, table_name: str = "articles"):
    """Return whether LanceDB table has FTS and vector indices."""
    flags = {"ok": False, "has_fts": False, "has_vector": False, "error": None}
    try:
        import lancedb  # Local import so --status still works without lancedb installed.
    except Exception as e:
        flags["error"] = f"lancedb import failed: {e}"
        return flags

    try:
        db = lancedb.connect(db_dir)
        table = db.open_table(table_name)
        indices = table.list_indices()
        for idx in indices:
            if "FTS" in str(idx).upper():
                flags["has_fts"] = True
            else:
                flags["has_vector"] = True
        # Tantivy-backed FTS may not always appear in list_indices() on some versions.
        if not flags["has_fts"]:
            try:
                table.search("__lancedb_fts_probe_token__").limit(1).to_list()
                flags["has_fts"] = True
            except Exception as probe_err:
                if "Cannot perform full text search unless an INVERTED index" not in str(probe_err):
                    flags["error"] = f"fts probe error: {probe_err}"
        flags["ok"] = True
    except Exception as e:
        flags["error"] = str(e)
    return flags


def show_status():
    """Show status of each processing step with freshness checks."""
    print("\n=== News Pipeline Status ===\n")

    stages = [
        {
            "id": "raw",
            "name": "1. Raw articles",
            "path": RAW_ARTICLES_DIR,
            "deps": [],
            "recursive": False,
        },
        {
            "id": "jsonl",
            "name": "2. JSONL files",
            "path": JSONL_DIR,
            "deps": ["raw"],
            "recursive": False,
        },
        {
            "id": "dedup",
            "name": "3. Deduped JSONL",
            "path": DEDUPED_DIR,
            "deps": ["jsonl"],
            "recursive": False,
        },
        {
            "id": "parquet",
            "name": "4. Parquet data",
            "path": f"{PARQUET_OUTPUT_DIR}/data",
            "deps": ["dedup"],
            "recursive": False,
        },
        {
            "id": "embed",
            "name": "5. Embeddings",
            "path": f"{EMBEDDINGS_DIR}/Qwen3-Embedding-8B",
            "deps": ["parquet"],
            "recursive": False,
        },
        {
            "id": "lancedb",
            "name": "6. LanceDB table + scalar index (stage 1/2)",
            "path": f"{LANCEDB_DIR}/Qwen3-Embedding-8B",
            "deps": ["parquet", "embed"],
            "recursive": False,
        },
    ]

    stage_info = {}
    for stage in stages:
        path = stage["path"]
        exists = check_dir_exists(path)
        stage_info[stage["id"]] = {
            "exists": exists,
            "mtime": get_latest_mtime(path, recursive=stage["recursive"]) if exists else None,
            "state": None,
            "reasons": [],
        }

    for stage in stages:
        info = stage_info[stage["id"]]
        path = stage["path"]
        status = "✓ Ready"
        stale_reasons = []

        if not info["exists"]:
            status = "✗ Missing"
            info["state"] = "missing"
        else:
            for dep in stage["deps"]:
                dep_info = stage_info[dep]
                if dep_info["state"] in {"missing", "stale"}:
                    status = "⚠ Stale"
                    stale_reasons.append(f"dependency not ready: {dep}")
                    continue
                if info["mtime"] is not None and dep_info["mtime"] is not None and info["mtime"] < dep_info["mtime"]:
                    status = "⚠ Stale"
                    stale_reasons.append(f"older than {dep} ({fmt_mtime(dep_info['mtime'])})")
            if status == "✓ Ready":
                info["state"] = "ready"
            else:
                info["state"] = "stale"
        info["reasons"] = stale_reasons

        print(f"  {stage['name']}: {status}")
        print(f"    Path: {path}")
        print(f"    Latest update: {fmt_mtime(info['mtime'])}")
        if stale_reasons:
            print(f"    Reason: {', '.join(stale_reasons)}")

    # Stage 2 readiness checks (metadata-based, not mtime-only).
    print("\n  7. LanceDB FTS index (stage 2/2):", end=" ")
    lancedb_path = f"{LANCEDB_DIR}/Qwen3-Embedding-8B"
    lancedb_ready = stage_info["lancedb"]["state"] == "ready"
    if not lancedb_ready:
        print("✗ Missing")
        print("    Reason: Step 6 output is not ready")
    else:
        flags = get_lancedb_index_flags(lancedb_path)
        if not flags["ok"]:
            print("⚠ Unknown")
            print(f"    Reason: could not inspect indices ({flags['error']})")
        elif flags["has_fts"]:
            print("✓ Ready")
        else:
            print("⚠ Missing")
            print("    Reason: no FTS index found; run --step lancedb_index")

    print("\n  8. LanceDB vector index (required IVF-PQ):", end=" ")
    if not lancedb_ready:
        print("✗ Missing")
        print("    Reason: Step 6 output is not ready")
    else:
        flags = get_lancedb_index_flags(lancedb_path)
        if not flags["ok"]:
            print("⚠ Unknown")
            print(f"    Reason: could not inspect indices ({flags['error']})")
        elif flags["has_vector"]:
            print("✓ Ready")
        else:
            print("⚠ Missing")
            print("    Reason: no vector index found; run --step lancedb_index")
        print(f"    FTS present: {flags.get('has_fts', False)}")

    print("\n  NOTE: 'Ready' means present and not older than upstream dependencies.")
    print("  Run with --step <name> to process each step.")
    print("  Steps: jsonl, dedup, parquet, embed, lancedb, lancedb_index\n")


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
    """Step 5: Build LanceDB table + scalar index (stage 1/2)."""
    import subprocess
    
    print("\n=== Step 5: LanceDB Table Build (Stage 1/2, no FTS) ===")
    
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
    print("Next: run --step lancedb_index for stage 2/2 (FTS + vector index).")
    return True


def run_lancedb_index_step():
    """Step 6: Build LanceDB FTS + vector index (stage 2/2)."""
    import subprocess

    print("\n=== Step 6: LanceDB Index Build (Stage 2/2: FTS + vector index) ===")

    lancedb_model_dir = f"{LANCEDB_DIR}/Qwen3-Embedding-8B"
    if not check_dir_exists(lancedb_model_dir):
        print(f"ERROR: LanceDB directory does not exist: {lancedb_model_dir}")
        print("Run step 5 (lancedb) first.")
        return False

    cmd = [
        "condor_submit_bid", "25",
        "/home/sgoel/forecast-sim/mpi_scripts/build_lancedb/build_index.sub",
    ]

    env = os.environ.copy()
    # Force stage-2 defaults for the one-line pipeline command.
    # This avoids shell env leakage (e.g. BUILD_VECTOR_INDEX=0 from prior runs)
    # accidentally disabling vector index creation.
    env["BUILD_FTS"] = "1"
    env["FTS_WITH_POSITION"] = "1"
    env["FTS_USE_TANTIVY"] = "1"
    env["TANTIVY_INDEX_ROOT"] = os.path.join(
        os.path.expanduser("~"), "forecasting", "lancedb_tantivy_indices"
    )
    env["BUILD_VECTOR_INDEX"] = "1"
    env["NUM_PARTITIONS"] = "4096"
    env["NUM_SUB_VECTORS"] = "64"
    env["VECTOR_METRIC"] = "cosine"

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
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
                       choices=['jsonl', 'dedup', 'parquet', 'embed', 'lancedb', 'lancedb_index'],
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
        'lancedb_index': run_lancedb_index_step,
    }
    
    step_func = step_funcs[args.step]
    success = step_func()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
