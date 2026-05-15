# Futuresim

Multi-agent forecasting simulator where LLM agents predict on free-form questions and are scored against each other.

## Quick Start

```bash
git clone https://github.com/OpenForecaster/futuresim.git
cd futuresim

uv sync
source .venv/bin/activate

cp .env.example .env
# Edit .env: set OPENROUTER_API_KEY and, if needed, storage paths.

python scripts/run_forecast_sim.py --config configs/shared/default_sim.yaml
```
- OpenRouter configs require `OPENROUTER_API_KEY`.
- The default search-enabled configs use LanceDB. Use
  `configs/shared/default_nosearch_sim.yaml` to run without retrieval.

## Environment

`scripts/run_forecast_sim.py` loads `.env` from the repo root automatically.
Shell exports override `.env` values. Inside `.env`, `${FSIM_REPO_DIR}` expands
to this checkout. The `FSIM_*` prefix is kept for compatibility with existing
configs. `pathing.py` is the small helper that loads `.env`, expands config
placeholders, and errors if a required placeholder is still unresolved.

Common variables:

| Variable | Required? | Use |
|----------|-----------|-----|
| `OPENROUTER_API_KEY` | yes for OpenRouter configs | Agent and answer-matcher API calls |
| `FSIM_OUTPUT_BASE` | no | Simulation output root |
| `FSIM_DATASET_PATH` | no | Hugging Face dataset id or local dataset path |
| `FSIM_DATASET_CACHE` | no | Hugging Face dataset cache directory |
| `FSIM_ARTIFACT_BASE` | no | Parent directory for downloaded public artifacts |
| `FSIM_SEARCH_DB` | for bundled LanceDB search | LanceDB artifact path |
| `FSIM_ARTICLES_BASE` | for MinimalHarness article browsing | Dated article JSONL tree |
| `FSIM_EMBEDDING_MODEL` | for search | Embedding model used by the LanceDB index |
| `FSIM_MATCHER_MODEL` | no | OpenRouter/vLLM model used for answer matching |
| `FSIM_SIM_MATCHER_CACHE_DIR` | no | Optional shared matcher-cache directory |

For local overrides, edit `.env`; for one-off runs, prefix the command:

```bash
FSIM_OUTPUT_BASE=/scratch/$USER/futuresim-runs \
python scripts/run_forecast_sim.py --config configs/shared/default_sim.yaml
```

## Data And Search

OpenForesight questions load from Hugging Face by default:
`nikhilchandak/OpenForesight`. The default config uses the
`aljazeera2026Q1` split.

The simulator itself does not require a search backend. The bundled
search-enabled configs use LanceDB through `agents/search_tools`; download the
prebuilt artifact for those runs:

```bash
source .venv/bin/activate
export FSIM_SEARCH_DB=${FSIM_SEARCH_DB:-$(pwd)/artifacts/forecast-news-embeddings}

hf download shash42/forecast-news-embeddings \
  --repo-type dataset \
  --local-dir "$FSIM_SEARCH_DB" \
  --max-workers 8

python scripts/check_search_readiness.py --db-path "$FSIM_SEARCH_DB"
```

Set `FSIM_SEARCH_DB` in `.env` to keep this artifact outside the repo.

The browsable article corpus is a separate dated tree:

```bash
export FSIM_ARTICLES_BASE=${FSIM_ARTICLES_BASE:-$(pwd)/artifacts/forecast-news}

hf download shash42/forecast-news \
  --repo-type dataset \
  --local-dir "$FSIM_ARTICLES_BASE" \
  --include '2025/12/**' \
  --include '2026/**' \
  --max-workers 8
```

`FSIM_SEARCH_DB` is read by the default runner to construct the bundled
LanceDB search tool. `articles_base` is only for MinimalHarness runs that expose the existing
`articles/YYYY/MM/DD/articles.jsonl` files inside the agent workspace. The
current Hugging Face corpus covers articles through 2026-03-31.

### Custom Question Sets

The simulator needs smaller schema than the full OpenForesight columns.
Use `--dataset custom --dataset_path <file-or-dir>` with CSV, JSONL, JSON, or
Parquet. A directory may contain `test.jsonl`, `test.parquet`, or
`test-*.parquet` style split files.

Required columns:

| Column | Meaning | Accepted aliases |
|--------|---------|------------------|
| `qid` | Stable question id | `question_id`, `id` |
| `title` | Forecast question shown to agents | `question_title`, `question` |
| `resolution_date` | Date when the question resolves | `close_time`, `resolve_time` |
| `ground_truth_answer` | Resolved answer used for scoring | `ground_truth`, `answer`, `resolution`, `resolved_to` |

Optional columns:

| Column | Default | Use |
|--------|---------|-----|
| `background` | empty | Context shown to agents |
| `resolution_criteria` | empty | Resolution rules shown to agents |
| `answer_type` | `freeform` | Prompt hint such as `binary`, `mcq`, `numeric`, or `freeform` |
| `options` | empty | JSON/list of allowed options for enumerated questions |
| `source_split` | CLI `--split` | Split-specific metrics, especially `test_daily_metrics.csv` |
| `prompt` | empty | Optional upstream prompt text retained for compatible scaffolds |

OpenForesight-specific article columns such as `url`, `article_maintext`,
`article_publish_date`, and `prompt_without_retrieval` are not required by the
simulator. For example, a ForecastBench-style source should first be converted
by joining its questions and resolutions into the columns above, then run with:

```bash
python scripts/run_forecast_sim.py \
  --dataset custom \
  --dataset_path /path/to/questions.jsonl \
  --split test
```

### Custom News Corpora

Question sets and news corpora are independent. The environment advances time,
exposes questions, and scores forecasts; agents decide what retrieval tools to
use. In the public runner, leaving `search_db` empty disables retrieval. When
`search_db`/`FSIM_SEARCH_DB` is set, `scripts/run_forecast_sim.py` constructs
the bundled LanceDB tool and passes it into the agents.

To swap in another corpus while using the bundled LanceDB tool, build a table
named `articles` with these fields:

| Field | Meaning |
|-------|---------|
| `chunk_id` | Unique id for this retrieved chunk |
| `article_id` | Source document id |
| `chunk_index` | Chunk number within the document |
| `title` | Article/document title |
| `source` | Publisher or corpus source |
| `date` | Timestamp used for no-future-leakage filtering |
| `date_publish` | Optional publish timestamp, also leakage-filtered when present |
| `content` | Text searched and returned to agents |
| `url` | Optional source URL |
| `vector` | Embedding vector, required for semantic/hybrid search |

Keyword search only needs `content` plus an FTS index. Semantic and hybrid
search also need vectors built with the same embedding model named by
`FSIM_EMBEDDING_MODEL`.

To use a different retrieval backend, add an implementation of
`agents/search_tools/base.py`'s `BaseSearchTool` contract and wire it into your
agent or runner. The search results consumed by agents are only
`article_id`, `title`, `source`, `date`, optional `date_publish`, `snippet`,
`score`, and optional `url`.

For MinimalHarness article browsing, set `articles_base`/`FSIM_ARTICLES_BASE` to
a dated JSONL tree: `YYYY/MM/DD/articles.jsonl`. Rows should provide `title`,
`source`, `date`, and `content`; `date_publish`, `url`, `id`, and `date_modify`
are optional but useful.

## Directory Structure

| Directory | Description |
|-----------|-------------|
| `agents/` | Agent implementations (BasicAgent, AllQAgent) |
| `environment/` | Simulation environment, scoring, data loading |
| `scripts/` | CLI scripts for running simulations |
| `configs/` | YAML configuration files |

## Key Commands

### Run Simulation
```bash
# Default shared simulation
python scripts/run_forecast_sim.py --config configs/shared/default_sim.yaml

# Shared variant without search
python scripts/run_forecast_sim.py --config configs/shared/default_nosearch_sim.yaml
```

Shared answer-matching cache:
- Sim runs still fall back to a per-run `matcher_cache.json`.
- If `FSIM_SIM_MATCHER_CACHE_DIR` is set, `split: "test"` runs automatically reuse `<cache_dir>/<matcher_slug>.json` and merge new entries back only when the run exits.
- For non-`test` runs, opt in with top-level YAML:
  `matcher_cache: {enabled: true, path: null}`
- Set `matcher_cache.path` to pin a specific JSON file, or `matcher_cache.enabled: false` to force the old per-run cache.
- Point `FSIM_SIM_MATCHER_CACHE_DIR` at a writable shared directory if multiple runs should reuse matcher results.

### Scaffold Names

Scaffold selection is explicit.

- `basic`, `allQ`, and `allqd` are the base chat-tools scaffolds.
- `qwenbasic` and `qwenallq` are thin Qwen-named compatibility wrappers over the shared chat-tools loop.
- `minimalHarness` runs external CLI backends such as Codex, Claude Code, and OpenCode.
- Qwen scaffolds intentionally do not replay historical hidden thinking across turns; only final assistant content and tool calls are fed back into history.
- Model names do not automatically switch scaffolds.

Set the scaffold in the config under `defaults.scaffold`.

### Resume / Restart
```bash
# Resume from last day
python scripts/run_forecast_sim.py --resume /path/to/output_dir

# Restart from specific day (preserves predictions before that day)
python scripts/run_forecast_sim.py \
    --restart_from /path/to/original/run \
    --restart_from_day 2025-04-05
```

## Documentation

- **[agents/search_tools/README.md](agents/search_tools/README.md)** — Search tool contract used by agents
- **[agents/allQAgent/README.md](agents/allQAgent/README.md)** — AllQ scaffold notes and token-budget fields
- **[agents/minimalHarnessAgent/README.md](agents/minimalHarnessAgent/README.md)** — External CLI harness notes

## Output

Simulation results are saved to `FSIM_OUTPUT_BASE/<sim_name>/<timestamp>/`:
- `config.json` — Run configuration
- `actions.jsonl` — All predictions and resolutions
- `daily_metrics.csv` — One cumulative metrics row per wakeup session, including daily submission count and average TV shift vs the previous submission
- `test_daily_metrics.csv` — Same metrics, filtered to questions whose `source_split` is `test`
- `agents/<agent_id>/model_raw_warmup.jsonl` — Warmup raw logs written by the agent scaffold, grouped by question id and logging only per-turn input deltas
- `agents/<agent_id>/model_raw_daily.jsonl` — Post-warmup raw logs written by the agent scaffold, logging only per-turn input deltas
- `agents/<agent_id>/` — Per-agent logs and memory

## OpenForesight Notes

- `timegap_days` changes the simulator from daily wakeups to one session every `N` days. BasicAgent-style prompts mention the last and next wakeup dates during normal sessions, and metrics for active questions are evaluated through the end of that wakeup interval.
- OpenForesight configs can prepend a window from the `train` split ahead of the main `split` with:
  - `prepend_train_resolution_start`
  - `prepend_train_resolution_end`
  - `subsample_per_month`
- Each OpenForesight question carries a `source_split` tag at load time so split-specific metrics can be logged without a separate loader path.
