# Search Agent

## Prerequisites

- a built LanceDB directory, usually via `FSIM_SEARCH_DB`
- the embedding model path, usually via `FSIM_EMBEDDING_MODEL`
- Stage 2 built for the DB you plan to query:
  FTS is required for `keyword` and `hybrid`
  IVF-PQ is optional but recommended for fast `semantic` and `hybrid`
- for `hybrid` and `semantic`, the runtime still needs the embedding model to encode queries even if the DB already stores document vectors

If the collaborator does not already have the prebuilt DB locally, download it first:

- `https://huggingface.co/datasets/shash42/forecast-news-embeddings`

```bash
hf download shash42/forecast-news-embeddings \
  --repo-type dataset \
  --local-dir /mnt/nfs/datasets_ac/news/deduped_articles/lance/Qwen3-Embedding-8B \
  --max-workers 8
```

## Search-Enabled Run

```bash
python scripts/test_basic_agent.py \
  --sim_name search_test \
  --provider openrouter \
  --openrouter_model xiaomi/mimo-v2-flash:free \
  --matching vllm \
  --matcher "${FSIM_MATCHER_MODEL}" \
  --search_db "${FSIM_SEARCH_DB}" \
  --embedding_model "${FSIM_EMBEDDING_MODEL}" \
  --embedding_gpu_mem 0.4 \
  --matcher_gpu_mem 0.3 \
  --start_date 2024-12-25 \
  --end_date 2024-12-27
```

## Search Modes

- `hybrid`: vector + keyword search combined
- `semantic`: vector-only
- `keyword`: BM25/FTS only

If collaborators receive a prebuilt LanceDB table, they should usually keep that table and only run the Stage 2 wrapper first so keyword/hybrid search is ready locally:

```bash
condor_submit_bid 25 mpi_scripts/build_lancedb/build_index.sub
```

or on AISA:

```bash
sbatch aisa_scripts/build_lancedb/build_index_aisa.sh
```

## Agent Search Syntax

```xml
<action type="search">
query text here
</action>
```

With a date filter:

```xml
<action type="search" from="2024-12-01" to="2024-12-15">
query text here
</action>
```

## Timing Logs

Search timing stats are written per agent to `agents/<agent_id>/timing_stats.jsonl`.
