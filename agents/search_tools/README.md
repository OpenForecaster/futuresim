# Search Tools Setup

LanceDB-based article search for forecasting agents.

## Prerequisites

1. **News articles** in parquet format at `/is/cluster/fast/sgoel/forecasting/news/deduped_articles/data/YYYY/MM/DD/`
2. **Embedding model**: `Qwen3-Embedding-8B` at `/is/cluster/fast/sgoel/models/Qwen3-Embedding-8B`
3. **GPU node** with 80GB+ memory for embedding and index building

## Setup Steps

### 1. Generate Embeddings (one-time, on GPU cluster)

```bash
# Submit embedding job for date range
python mpi_scripts/embed/submit_job.py --start_date 2023-01-01 --end_date 2025-12-31
```

Embeddings saved to: `/is/cluster/fast/sgoel/forecasting/news/embeddings/Qwen3-Embedding-8B/`

### 2. Build LanceDB Index

```bash
# Build LanceDB from articles + embeddings
python scripts/build_lancedb.py --start_date 2023-01-01 --end_date 2025-12-31

# Output: /is/cluster/fast/sgoel/forecasting/news/deduped_articles/lance/Qwen3-Embedding-8B/
```

### 3. Create Vector Index (CRITICAL for performance)

Without IVF index: ~300 sec/query
With IVF index: ~5 sec/query

```bash
# Run on GPU node with 80GB+ RAM
python scripts/build_lancedb_index.py
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

Fields: `llm_count`, `llm_avg_seconds`, `search_count`, `search_avg_seconds`, `df_query_count`, `df_query_avg_seconds`, `day_total_seconds`
