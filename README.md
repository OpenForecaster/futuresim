# Futuresim

Futuresim is a forecasting simulator for LLM agents. It advances a dated
question market, exposes only the information available at each simulated date,
records agent forecasts, and scores them over time.

For background, see the [Futuresim blogpost](https://openforecaster.github.io/futuresim/)
and [paper](https://arxiv.org/abs/2605.15188).

## Quick Start

```bash
git clone https://github.com/OpenForecaster/futuresim.git
cd futuresim

uv sync
source .venv/bin/activate

cp .env.example .env
# Edit .env if you need OpenRouter keys, custom output paths, or local artifacts.

python scripts/run_forecast_sim.py --config configs/shared/default_sim.yaml
```

The default search-enabled configs use LanceDB and require `FSIM_SEARCH_DB`.
For a no-retrieval smoke run, use:

```bash
python scripts/run_forecast_sim.py --config configs/shared/default_nosearch_sim.yaml
```

## Configuration

`scripts/run_forecast_sim.py` loads `.env` from the repo root. Shell exports
override `.env` values. `${FSIM_REPO_DIR}` expands to this checkout.

Common variables:

| Variable | Use |
| --- | --- |
| `OPENROUTER_API_KEY` | Required for OpenRouter-backed agents or answer matching |
| `FSIM_OUTPUT_BASE` | Simulation output root |
| `FSIM_DATASET_PATH` | Hugging Face dataset id or local dataset path |
| `FSIM_DATASET_CACHE` | Hugging Face dataset cache directory |
| `FSIM_SEARCH_DB` | LanceDB index path for bundled hybrid search |
| `FSIM_ARTICLES_BASE` | Dated article JSONL tree for filesystem article browsing |
| `FSIM_EMBEDDING_MODEL` | Embedding model used by the LanceDB index |
| `FSIM_MATCHER_MODEL` | OpenRouter/vLLM answer-matcher model |
| `FSIM_SIM_MATCHER_CACHE_DIR` | Optional shared answer-matcher cache directory |

One-off override example:

```bash
FSIM_OUTPUT_BASE=/scratch/$USER/futuresim-runs \
python scripts/run_forecast_sim.py --config configs/shared/default_sim.yaml
```

## Data And Search

[OpenForesight](https://huggingface.co/datasets/nikhilchandak/OpenForesight)
questions load from Hugging Face by default. The default config uses the
`aljazeera2026Q1` split.

Futuresim separates the question market from agent retrieval:

- The environment owns dates, visible questions, visible article files, forecast
  ingestion, answer matching, and scoring.
- Agents own their retrieval strategy. They can use the filesystem article
  corpus, the bundled LanceDB hybrid search tool, or a custom tool.

Download the prebuilt
[LanceDB artifact](https://huggingface.co/datasets/shash42/forecast-news-embeddings)
for the bundled hybrid search configs:

```bash
export FSIM_SEARCH_DB=${FSIM_SEARCH_DB:-$(pwd)/artifacts/forecast-news-embeddings}

hf download shash42/forecast-news-embeddings \
  --repo-type dataset \
  --local-dir "$FSIM_SEARCH_DB" \
  --max-workers 8

python scripts/check_search_readiness.py --db-path "$FSIM_SEARCH_DB"
```

The public embedding model used with this index is
[Qwen/Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B).
Set `FSIM_EMBEDDING_MODEL` to a local checkout, a model id, or an embedding
server target supported by your search backend.

Download the browsable
[article corpus](https://huggingface.co/datasets/shash42/forecast-news)
separately:

```bash
export FSIM_ARTICLES_BASE=${FSIM_ARTICLES_BASE:-$(pwd)/artifacts/forecast-news}

hf download shash42/forecast-news \
  --repo-type dataset \
  --local-dir "$FSIM_ARTICLES_BASE" \
  --include '2025/12/**' \
  --include '2026/**' \
  --max-workers 8
```

`FSIM_ARTICLES_BASE` must point to a dated tree:
`YYYY/MM/DD/articles.jsonl`. Rows should include `title`, `source`, `date`, and
`content`; `date_publish`, `url`, `id`, and `date_modify` are optional.

## Custom Data

Use `--dataset custom --dataset_path <file-or-dir>` with CSV, JSONL, JSON, or
Parquet. A directory may contain split files such as `test.jsonl`,
`test.parquet`, or `test-*.parquet`.

Required columns:

| Column | Accepted aliases |
| --- | --- |
| `qid` | `question_id`, `id` |
| `title` | `question_title`, `question` |
| `resolution_date` | `close_time`, `resolve_time` |
| `ground_truth_answer` | `ground_truth`, `answer`, `resolution`, `resolved_to` |

Optional columns: `background`, `resolution_criteria`, `answer_type`,
`options`, `source_split`, and `prompt`.

Example:

```bash
python scripts/run_forecast_sim.py \
  --dataset custom \
  --dataset_path /path/to/questions.jsonl \
  --split test
```

To use a custom search backend, implement the `BaseSearchTool` contract in
[agents/search_tools/base.py](https://github.com/OpenForecaster/futuresim/blob/main/agents/search_tools/base.py).
For LanceDB, semantic/hybrid search needs an `articles` table with chunk ids,
article ids, date fields, content, optional metadata, and vectors built with the
configured embedding model.

## Platform Integrations

Futuresim includes adapters for Prime Intellect Verifiers and OpenReward/ORS.
They use the same `SimulationEnvironment` and run a MinimalHarness-compatible
CLI agent through the packaged MCP server:

```bash
python -m futuresim_agents.minimalHarnessAgent.mcp_server
```

Important defaults:

- The filesystem article corpus is the default information source.
- Hybrid LanceDB search is opt-in via `futuresim.enable_hybrid_search: true`.
- Hosted runs only accept forecasts submitted through MCP
  `submit_forecasts` and finalized with `next_day`.
- Sandboxes block general internet by default to avoid future leakage.
- Codex/Claude CLI reproductions require each user to provide their own private
  CLI/provider credentials through platform secrets or an equivalent private
  setup.

See [integrations/README.md](https://github.com/OpenForecaster/futuresim/blob/main/integrations/README.md)
for sandbox image requirements, credential handling, network/egress guidance,
and publication steps for Verifiers and OpenReward.

## Common Commands

```bash
# Default shared simulation
python scripts/run_forecast_sim.py --config configs/shared/default_sim.yaml

# No-retrieval variant
python scripts/run_forecast_sim.py --config configs/shared/default_nosearch_sim.yaml

# Resume from the last day in a run directory
python scripts/run_forecast_sim.py --resume /path/to/output_dir

# Restart from a specific day while preserving prior forecasts
python scripts/run_forecast_sim.py \
  --restart_from /path/to/original/run \
  --restart_from_day 2025-04-05
```

Scaffold selection is explicit in config under `defaults.scaffold`:

- `basic`, `allQ`, `allqd`: base chat-tools scaffolds.
- `qwenbasic`, `qwenallq`: Qwen-named compatibility wrappers.
- `minimalHarness`: external CLI backends such as Codex, Claude Code, and
  OpenCode.

## Outputs

Runs are saved to `FSIM_OUTPUT_BASE/<sim_name>/<timestamp>/`.

Key files:

| File | Contents |
| --- | --- |
| `config.json` | Fully resolved run configuration |
| `actions.jsonl` | Predictions and resolutions |
| `daily_metrics.csv` | Cumulative metrics per wakeup session |
| `test_daily_metrics.csv` | Same metrics filtered to `source_split == "test"` |
| `matcher_cache.json` | Per-run answer-matcher cache unless shared caching is configured |
| `agents/<agent_id>/` | Per-agent transcripts, logs, and memory |

If `FSIM_SIM_MATCHER_CACHE_DIR` is set, `split: "test"` runs reuse
`<cache_dir>/<matcher_slug>.json` and merge new entries back when the run exits.
Other splits can opt in with top-level YAML:
`matcher_cache: {enabled: true, path: null}`.

## Notes

- `timegap_days` changes the simulator from daily wakeups to one session every
  `N` days. Metrics for active questions are evaluated through the end of that
  wakeup interval.
- OpenForesight configs can prepend train-split questions with
  `prepend_train_resolution_start`, `prepend_train_resolution_end`, and
  `subsample_per_month`.
- Each OpenForesight question carries a `source_split` tag so split-specific
  metrics can be logged without a separate loader path.

## More Documentation

- [agents/search_tools/README.md](https://github.com/OpenForecaster/futuresim/blob/main/agents/search_tools/README.md): search tool contract.
- [agents/allQAgent/README.md](https://github.com/OpenForecaster/futuresim/blob/main/agents/allQAgent/README.md): AllQ scaffold notes.
- [agents/minimalHarnessAgent/README.md](https://github.com/OpenForecaster/futuresim/blob/main/agents/minimalHarnessAgent/README.md): external CLI harness notes.
- [integrations/README.md](https://github.com/OpenForecaster/futuresim/blob/main/integrations/README.md): Verifiers and OpenReward integration details.
