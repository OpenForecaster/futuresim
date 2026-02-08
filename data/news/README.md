# News Pipeline

Self-contained news download and processing pipeline for forecast-sim.

## Overview

This pipeline downloads news articles from CommonCrawl News (CCNews) and processes them for use in LanceDB semantic search. Current data spans from 2016 to August 2025.

**Pipeline stages:**
1. **CCNews Download** - Extract articles from CommonCrawl WARCs via news-please
2. **JSONL Conversion** - Convert individual JSON files to JSONL per domain
3. **Deduplication** - Remove duplicate articles based on title + content hash
4. **Parquet Conversion** - Convert to daily-partitioned Parquet for efficient storage
5. **Embedding** - Generate Qwen3-Embedding-8B embeddings for semantic search
6. **LanceDB Index** - Build searchable index with vector + FTS capabilities

## Quick Start: Update News (Aug 2025 - Jan 2026)

### Prerequisites

```bash
# Activate environment
source ~/forecast-sim/fsim/bin/activate
cd ~/forecast-sim

# Initialize news pipeline dependencies (applies patches)
./data/news/scripts/setup_news_pipeline.sh
```

### Step 0: Configure news-please

The `setup_news_pipeline.sh` script automatically applies a patch to `news-please/examples/commoncrawl.py` to:
1.  Load domains from `data/news/domains.txt` relative to the script location.
2.  Accept `NEWS_START_DATE` and `NEWS_END_DATE` from environment variables (handled by `launch_news_crawl.py`).

No manual editing of `commoncrawl.py` is required.

### Step 1: Download CCNews (Long-running)

```bash
# Launch download job for Aug 2025 - Jan 2026 (takes days/weeks)
python data/news/scripts/launch_news_crawl.py \
    --start-date 2025-08-01 \
    --end-date 2026-01-31 \
    --bid 15

# Or download just one month:
python data/news/scripts/launch_news_crawl.py --start-date 2025-10-01 --end-date 2025-10-31

# Monitor progress
condor_q
tail -f /fast/sgoel/logs/forecasting-sim/news/crawl/*.out
```

### Step 2-6: Run Processing Pipeline

After download completes, run each step sequentially:

```bash
cd ~/forecast-sim

# Check status
python data/news/run_news_pipeline.py --status

# Step 1: JSON → JSONL
python data/news/run_news_pipeline.py --step jsonl
# Wait for job to complete (check: condor_q)

# Step 2: Deduplicate
python data/news/run_news_pipeline.py --step dedup
# Wait for completion

# Step 3: Convert to Parquet (merges with existing data)
python data/news/run_news_pipeline.py --step parquet
# Wait for completion

# Step 4: Generate embeddings
python data/news/run_news_pipeline.py --step embed
# Wait for completion (GPU job, takes hours)

# Step 5: Build LanceDB index
python data/news/run_news_pipeline.py --step lancedb
# Wait for completion
```

### Alternative: Manual Jobs

Run steps directly via HTCondor:

```bash
# Parquet conversion
python data/news/scripts/launch_parquet_conversion.py \
    --input-dirs /is/cluster/fast/sgoel/forecasting/news/articles_2025_2026/deduped \
    --output-dir /is/cluster/fast/sgoel/forecasting/news/deduped_articles

# Embedding
cd mpi_scripts/embed && python submit_job.py --gpus 1 --bid 25 --num_workers 8

# LanceDB
condor_submit_bid 15 mpi_scripts/build_lancedb/build_lancedb.sub
```

## Directory Structure

```
data/news/
├── README.md                      # This file
├── domains.txt                    # ~130 high-quality news domains
├── run_news_pipeline.py           # Main pipeline orchestrator
├── news-please/                   # Clone of news-please repo
├── scripts/
│   ├── to_jsonl.py               # JSON → JSONL conversion
│   ├── deduplicate_news_jsonl.py # Deduplication
│   ├── convert_jsonl_to_parquet.py
│   ├── launch_*.py               # HTCondor job launchers
└── condor/
    └── run_*.sh                  # Shell wrappers for jobs
```

## Data Locations

| Data                | Path                                                              |
|---------------------|-------------------------------------------------------------------|
| Raw articles (new)  | `/is/cluster/fast/sgoel/forecasting/news/filtered_cc_articles_2025_2026/` |
| JSONL (new)         | `/is/cluster/fast/sgoel/forecasting/news/articles_2025_2026/`     |
| Deduped JSONL       | `.../articles_2025_2026/deduped/`                                 |
| Parquet articles    | `/is/cluster/fast/sgoel/forecasting/news/deduped_articles/data/`  |
| Embeddings          | `.../deduped_articles/embeddings/Qwen3-Embedding-8B/`             |
| LanceDB index       | `.../deduped_articles/lance/Qwen3-Embedding-8B/`                  |

## Troubleshooting

### Download stalls
The CCNews download streams WARCs in semi-random order. Some months may have fewer articles. Check the extracted articles dir for progress.

### Job fails with disk quota
Request more disk in the job submission: add `request_disk = 200GB` to the .sub file.

### Resuming after failure
Most scripts support resuming:
- JSONL conversion tracks `processed_dirs.txt`
- Embedding uses `--resume` flag by default
- Run the same command to continue
