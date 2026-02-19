# Search Tools Setup

LanceDB-based article search for forecasting agents.

## Prerequisites

1. **News articles** in parquet format at `/is/cluster/fast/sgoel/forecasting/news/deduped_articles/data/YYYY/MM/DD/`
2. **Embedding model**: `Qwen3-Embedding-8B` at `/is/cluster/fast/sgoel/models/Qwen3-Embedding-8B`
3. **High-memory node** for LanceDB indexing (GPU optional; required only for embedding)

## Setup Steps

### 1. Generate Embeddings (one-time, on GPU cluster)

```bash
# Submit embedding job for date range
python mpi_scripts/embed/submit_job.py --start_date 2023-01-01 --end_date 2025-12-31
```

**Output**: `/is/cluster/fast/sgoel/forecasting/news/embeddings/Qwen3-Embedding-8B/`

### 2. Build LanceDB Table (Stage 1/2, CPU-heavy)

This step compiles articles and embeddings into a LanceDB table and builds:
1. **Scalar Index on `date`** column (for filter pre-pruning).
2. **No FTS in this stage** (FTS is built in Stage 2).

```bash
# Build LanceDB table from articles + embeddings
python scripts/build_lancedb.py \
  --start_date 2023-01-01 \
  --end_date 2025-12-31 \
  --skip_fts_index

# Output: /is/cluster/fast/sgoel/forecasting/news/deduped_articles/lance/Qwen3-Embedding-8B/
```

### 3. Build Search Indices (Stage 2/2)

Build FTS first (with phrase positions), then optional IVF-PQ vector index.

```bash
# FTS + vector
python scripts/build_lancedb_index.py --build_fts --fts_with_position --force

# FTS only (retry path when vector is not needed yet)
python scripts/build_lancedb_index.py --build_fts --skip_vector_index --fts_with_position --force
```

## Running Search Agent

```bash
python scripts/test_basic_agent.py \
    --sim_name search_test \
    --provider openrouter \
    --openrouter_model xiaomi/mimo-v2-flash:free \
    --matching vllm \
    --matcher /fast/rolmedo/models/qwen3-4b-it-2507 \
    --search_db /is/cluster/fast/sgoel/forecasting/news/deduped_articles/lance/Qwen3-Embedding-8B \
    --embedding_model /is/cluster/fast/sgoel/models/Qwen3-Embedding-8B \
    --embedding_gpu_mem 0.4 \
    --matcher_gpu_mem 0.3 \
    --start_date 2024-12-25 --end_date 2024-12-27
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--search_db` | - | Path to LanceDB directory |
| `--embedding_model` | - | Path to embedding model for semantic search |
| `--embedding_gpu_mem` | 0.4 | GPU memory fraction for embedding model |
| `--matcher_gpu_mem` | 0.3 | GPU memory fraction for matcher model |

## Search Types

- **hybrid** (default): Vector + keyword search combined via RRF
- **semantic**: Vector-only search
- **keyword**: BM25 full-text search only (fastest, no embedding needed)

## Agent Search Syntax

```xml
<action type="search">
query text here
</action>

<!-- With date range -->
<action type="search" from="2024-12-01" to="2024-12-15">
query text here
</action>
```

## Timing Metrics

Agent timing stats saved to: `<agent_dir>/timing_stats.jsonl`

Fields: `llm_count`, `llm_avg_seconds`, `search_count`, `search_avg_seconds`, `df_query_count`, `df_query_avg_seconds`, `matcher_count`, `matcher_avg_seconds`, `day_total_seconds`
