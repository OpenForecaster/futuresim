# Search Agent

## Prerequisites

- a built LanceDB directory, usually via `FSIM_SEARCH_DB`
- the embedding model path, usually via `FSIM_EMBEDDING_MODEL`
- Stage 2 FTS built for the DB you plan to query

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
