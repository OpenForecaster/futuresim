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
6. **LanceDB Table + Scalar (Stage 1/2)** - Build article table and scalar date index only
7. **LanceDB FTS + IVF-PQ (Stage 2/2)** - Build full-text index (with positions) and vector optimization

## Quick Start: Update News (Aug 2025 - Jan 2026)

### Prerequisites

```bash
# Activate environment
source ~/forecast-sim/fsim/bin/activate
cd ~/forecast-sim

# Initialize news pipeline dependencies (applies patches)
./data/news/scripts/setup_news_pipeline.sh
```

`setup_news_pipeline.sh` also ensures required NLTK tokenizer data is present (e.g. `punkt_tab`),
installs `indic-nlp-library` in `fsim` to avoid newspaper4k extraction failures on Bengali pages,
and installs `tantivy` + `pylance` so Stage 2 can use Tantivy FTS backend for stable phrase indexing.
If either setup step fails, the script exits non-zero.

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

### Step 2-7: Run Processing Pipeline

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

# Step 5: Build LanceDB table + scalar index only (Stage 1/2)
python data/news/run_news_pipeline.py --step lancedb
# Wait for completion

# Step 6: Build/refresh FTS (with positions) + vector index (Stage 2/2 default)
python data/news/run_news_pipeline.py --step lancedb_index
# Wait for completion
```

The pipeline forces Stage 2 defaults (`BUILD_FTS=1`, `FTS_WITH_POSITION=1`,
`FTS_USE_TANTIVY=1`, `BUILD_VECTOR_INDEX=1`, and
`TANTIVY_INDEX_ROOT=~/forecasting/lancedb_tantivy_indices`) so stale shell env vars
cannot accidentally disable vector indexing.

### Alternative: Manual Jobs

Run steps directly via HTCondor:

```bash
# Parquet conversion
python data/news/scripts/launch_parquet_conversion.py \
    --input-dirs /is/cluster/fast/sgoel/forecasting/news/articles_2025_2026/deduped \
    --output-dir /is/cluster/fast/sgoel/forecasting/news/deduped_articles

# Embedding
cd mpi_scripts/embed && python submit_job.py --gpus 1 --bid 25 --num_workers 8

# LanceDB Stage 1/2 (table + scalar date index)
condor_submit_bid 15 mpi_scripts/build_lancedb/build_lancedb.sub

# LanceDB Stage 2/2 (FTS with positions + vector index)
# Use Tantivy backend with external index files on /lustre to avoid /is lock issues.
BUILD_FTS=1 FTS_WITH_POSITION=1 FTS_USE_TANTIVY=1 \
TANTIVY_INDEX_ROOT=/lustre/home/sgoel/forecasting/lancedb_tantivy_indices \
BUILD_VECTOR_INDEX=1 \
condor_submit_bid 25 mpi_scripts/build_lancedb/build_index.sub

# Stage 2 alternative: FTS only (skip vector index)
BUILD_FTS=1 FTS_WITH_POSITION=1 FTS_USE_TANTIVY=1 \
TANTIVY_INDEX_ROOT=/lustre/home/sgoel/forecasting/lancedb_tantivy_indices \
BUILD_VECTOR_INDEX=0 \
condor_submit_bid 25 mpi_scripts/build_lancedb/build_index.sub
```

### Collaborator setup note

If collaborators are not on the MPI cluster, setup can differ by filesystem:
1. If LanceDB table path supports lockfiles (typical local disk, many `/lustre` mounts), Tantivy can write FTS in-table directly.
2. If LanceDB table path does not support lockfiles (for example `/is/cluster/fast` on this cluster), keep `FTS_USE_TANTIVY=1` and set `TANTIVY_INDEX_ROOT` to a lock-capable path (for example under `$HOME` or `/lustre`).
3. The one-line pipeline command (`--step lancedb_index`) already sets a safe default `TANTIVY_INDEX_ROOT` under `~/forecasting/`.

### Important dependency note

`lancedb_index` does **not** run before embedding, and it is not an embedding dependency.
Correct order is:
1. `embed`
2. `lancedb` (table + scalar date index only)
3. `lancedb_index` (FTS with positions + vector optimization)

If `lancedb_index` fails, do **not** rerun ingest by default. Embeddings and table data are
usually already complete after `lancedb`; rerun only Stage 2 (`lancedb_index`).

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
