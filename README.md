# Futuresim

Futuresim is a forecasting simulator for LLM agents. It advances a dated
question market, exposes only the information available at each simulated date,
records agent forecasts, and scores them over time.

For background, see the [Futuresim blogpost](https://openforecaster.github.io/futuresim/)
and [paper](https://arxiv.org/abs/2605.15188).

## Contents

- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Data And Search](#data-and-search)
- [Launch Commands](#launch-commands)
- [Outputs](#outputs)
- [Platform Integrations](#platform-integrations)
  - [OpenReward / ORS](#openreward--ors)
  - [Prime Intellect Verifiers](#prime-intellect-verifiers)
- [Custom Data](#custom-data)
- [Notes](#notes)
- [More Documentation](#more-documentation)

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

## Launch Commands

For local CLI-agent Futuresim runs, use `scaffold: minimalHarness` in a YAML
config and set `defaults.harness_backend` to `codex`, `claude_code`, or
`opencode`. The config carries backend-specific paths, resume behavior, sandbox
settings, and search settings.

Local CLI setup is explicit in the config:

| Backend | Setup | Config fields |
| --- | --- | --- |
| `codex` | Install Codex, run `codex login`, and set `CODEX_PATH` or put `codex` on `PATH`. | `defaults.harness_backend: codex`, `defaults.codex_path`, `defaults.codex_resume`, `defaults.reasoning_effort` |
| `claude_code` | Install Claude Code, authenticate it, and set the CLI path. | `defaults.harness_backend: claude_code`, `agents[].claude_code_path` or `defaults.claude_code_path`, `defaults.claude_code_resume` |
| `opencode` | Install OpenCode and provide `OPENROUTER_API_KEY` for provider-qualified models. | `defaults.harness_backend: opencode`, `defaults.opencode_path` |

Futuresim repo runs use configs in `configs/minimalHarness/`; those configs
set `defaults.scaffold: minimalHarness`, select or inherit the CLI backend, and
pin the CLI path/auth expectations for that run.

Codex example using the public OpenForesight split and local LanceDB search:

```bash
export OPENROUTER_API_KEY=...
export CODEX_PATH="$(command -v codex)"
export FSIM_OUTPUT_BASE="$PWD/logs/current_sim"
export FSIM_ARTICLES_BASE=/path/to/forecast-news/articles
export FSIM_SEARCH_DB=/path/to/forecast-news-embeddings/lancedb
export FSIM_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B

python scripts/run_forecast_sim.py \
  --config configs/minimalHarness/aljazeera2026Q1_codex_gpt55_lancedb_search.yaml
```

For a Day 0 local CLI smoke run, use the same config and override only the
simulation end date. With `start_date=2025-12-31` and `lookback_days=7`, Day 0
is `2025-12-24`:

```bash
export OPENROUTER_API_KEY=...
export CODEX_PATH="$(command -v codex)"
export FSIM_OUTPUT_BASE="$PWD/logs/current_sim"
export FSIM_ARTICLES_BASE=/path/to/forecast-news/articles
export FSIM_SEARCH_DB=/path/to/forecast-news-embeddings/lancedb
export FSIM_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B

python scripts/run_forecast_sim.py \
  --config configs/minimalHarness/aljazeera2026Q1_codex_gpt55_lancedb_search.yaml \
  --sim_name codex_aljazeera2026Q1_gpt55_lancedb_day0 \
  --end_date 2025-12-24
```

Claude Code and OpenCode local runs use the same command shape with a
MinimalHarness YAML whose defaults select the backend:

```yaml
defaults:
  scaffold: minimalHarness
  harness_backend: claude_code  # or: opencode
  claude_code_path: /path/to/claude
  opencode_path: /path/to/opencode
```

```bash
python scripts/run_forecast_sim.py --config path/to/minimalHarness-claude-code.yaml
python scripts/run_forecast_sim.py --config path/to/minimalHarness-opencode.yaml
```

Resume and restart remain available for long runs:

```bash
python scripts/run_forecast_sim.py --resume /path/to/output_dir

python scripts/run_forecast_sim.py \
  --restart_from /path/to/original/run \
  --restart_from_day 2025-04-05
```

Scaffold selection is explicit in config under `defaults.scaffold`:

- `basic`, `allQ`, `allqd`: base chat-tools scaffolds.
- `qwenbasic`, `qwenallq`: Qwen-named compatibility wrappers.
- `minimalHarness`: external CLI backends such as Codex, Claude Code, and
  OpenCode.

For a plain API-key run through OpenRouter, set `OPENROUTER_API_KEY` and choose
the chat scaffold with `--scaffold`. Useful values are `basic`, `allQ`,
`allqd`, `qwenbasic`, and `qwenallq`.

```bash
export OPENROUTER_API_KEY=...

python scripts/run_forecast_sim.py \
  --sim_name openrouter_api_allq \
  --provider openrouter \
  --openrouter_model deepseek/deepseek-v4-flash \
  --matching openrouter \
  --matcher deepseek/deepseek-v3.2 \
  --scaffold allQ \
  --dataset openforesight \
  --dataset_path nikhilchandak/OpenForesight \
  --split aljazeera2026Q1 \
  --start_date 2025-12-31 \
  --end_date 2026-03-28 \
  --lookback_days 7
```

For a Day 0 plain API-key smoke run, keep the same question resolution window
but stop the simulation on the first simulated day:

```bash
export OPENROUTER_API_KEY=...

python scripts/run_forecast_sim.py \
  --sim_name openrouter_api_day0 \
  --provider openrouter \
  --openrouter_model deepseek/deepseek-v4-flash \
  --matching openrouter \
  --matcher deepseek/deepseek-v3.2 \
  --scaffold allQ \
  --dataset openforesight \
  --dataset_path nikhilchandak/OpenForesight \
  --split aljazeera2026Q1 \
  --start_date 2025-12-31 \
  --end_date 2025-12-24 \
  --resolution_start 2025-12-31 \
  --resolution_end 2026-03-28 \
  --lookback_days 7
```

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

## Platform Integrations

Futuresim includes adapters for OpenReward/ORS and Prime Intellect Verifiers.
They share the same `SimulationEnvironment`, but differ in how the agent is
launched and how tools are exposed.

### OpenReward / ORS

The OpenReward integration exposes Futuresim through
[OpenReward/ORS](https://docs.openreward.ai/) and
[Firehorse](https://docs.openreward.ai/harnesses/quickstart). The model sees an
`openreward` MCP server with sandbox filesystem/shell tools plus Futuresim task
tools: `search_news`, `submit_forecasts`, and `next_day`. See also
[OpenReward harness toolsets](https://docs.openreward.ai/harnesses/harness-toolsets).

For `codex` and `claude-code` agents, Firehorse launches the user's local CLI
with the user's local auth and connects it to the OpenReward sandbox tools.
The environment itself does not perform model I/O; custom prompt scaffolds can
be supplied through `openreward_agent.prompt_builder` in the task spec.
For API-key models, Firehorse can instead run an API agent such as `react` with
an OpenReward harness toolset such as `claude-code`; this is useful for smoke
tests and model comparisons, but it does not reproduce the original paper runs
because it is not the actual local Codex or Claude Code CLI harness.

OpenReward run commands use Firehorse and choose the harness with `--agent`.
The closest reproduction of Futuresim results uses a CLI agent such as `codex`
or `claude-code`, launched with the user's local CLI auth.

```bash
pip install --no-compile -e ".[openreward]" firehorse-cli

export OPENREWARD_API_KEY=...
export OPENROUTER_API_KEY=...

# Closest reproduction path: Codex CLI, using local Codex auth.
futuresim-openreward-firehorse \
  --env <namespace>/futuresim \
  --agent codex \
  --model openai/gpt-5.5 \
  --effort xhigh \
  --split test \
  --max-tasks 1 \
  --output-dir runs/openreward/codex-gpt55-xhigh \
  --task-spec configs/openreward/aljazeera2026Q1_v1_day0_day1.yaml
```

The bundled OpenReward task specs include day 0 + day 1 and 7-day smoke
windows. With the default `start_date=2025-12-31` and `lookback_days=7`, the
2-day specs run `2025-12-24` and `2025-12-25`; the 7-day specs run through
`2025-12-30`. Use `configs/openreward/aljazeera2026Q1_v1_7day.yaml` or
`configs/openreward/aljazeera2026Q1_v3_7day.yaml` for the 7-day versions.

Claude Code runs use the same wrapper with different `--agent` / `--model`
values:

```bash
# Claude Code CLI, using local Claude Code auth.
futuresim-openreward-firehorse \
  --env <namespace>/futuresim \
  --agent claude-code \
  --model anthropic/claude-sonnet-4-6 \
  --effort high \
  --split test \
  --max-tasks 1 \
  --output-dir runs/openreward/claude-sonnet \
  --task-spec configs/openreward/aljazeera2026Q1_v1_day0_day1.yaml
```

OpenRouter API models can run without local Codex or Claude Code auth by using
the Firehorse React agent with an OpenReward toolset:

```bash
# API-direct model, using the OpenReward Claude Code toolset.
futuresim-openreward-firehorse \
  --env <namespace>/futuresim \
  --agent react \
  --toolset claude-code \
  --model openrouter/deepseek/deepseek-v4-pro \
  --split test \
  --max-tasks 1 \
  --output-dir runs/openreward/deepseek-v4-pro-react \
  --task-spec configs/openreward/aljazeera2026Q1_v3_day0_day1.yaml
```

`--task-spec` accepts a JSON or YAML OpenReward task-spec overlay. Copy one of
the files under `configs/openreward/` to change the dataset, date window,
answer matcher, handholding version, article mounting, or sandbox settings.
When `--output-dir` is provided and the task spec leaves `futuresim.output_base`
blank, the wrapper sets `output_base` to the same directory.

The OpenReward-native path uses the same Futuresim simulation/scoring core, but
differs from original local Futuresim runs in the following ways:

- The default `search_news` is backed by OpenReward hosted search. It calls
  OpenReward `Backsearch` for up to 5 hits and `Backfetch` for those hits, then
  caps each fetched article body at 5,000 characters before formatting.
- The default OpenReward sandbox is CPU-only; search uses OpenReward hosted
  search and answer matching uses OpenRouter by default, so no sandbox GPU is
  needed.
- The sandbox does not include a grep-able `articles/` tree unless
  `openreward_sandbox.mount_articles: true` and `futuresim.articles_base` are
  configured.

### Prime Intellect Verifiers

The [Verifiers](https://docs.primeintellect.ai/verifiers/overview)
integration targets MinimalHarness-compatible CLI reproduction. The CLI agent
runs through the packaged Futuresim MCP server:

```bash
python -m futuresim_agents.minimalHarnessAgent.mcp_server
```

The current Prime/Verifiers hosted path should be treated as experimental for
strict Codex/Claude reproduction. Futuresim needs the agent shell to have no
general internet access while still allowing model-provider endpoints.
[Prime Sandboxes](https://docs.primeintellect.ai/sandboxes/overview) document
network disabling for secure agent runs, but the hosted Verifiers path does not
currently provide the custom URL allowlist/blocklist surface Futuresim needs for
this strict setup.

Important Verifiers/local MinimalHarness defaults:

- The filesystem article corpus is the default information source.
- Hybrid LanceDB search is opt-in via `futuresim.enable_hybrid_search: true`.
- Runs only accept forecasts submitted through MCP
  `submit_forecasts` and finalized with `next_day`.
- Sandboxes block general internet by default to avoid future leakage.
- Codex/Claude CLI reproductions require each user to provide their own private
  CLI/provider credentials through platform secrets or an equivalent private
  setup.

See [integrations/README.md](https://github.com/OpenForecaster/futuresim/blob/main/integrations/README.md)
for sandbox image requirements, credential handling, network/egress guidance,
and publication steps for Verifiers and OpenReward.

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
- [agents/minimalHarnessAgent/README.md](https://github.com/OpenForecaster/futuresim/blob/main/agents/minimalHarnessAgent/README.md): external CLI harness notes.
- [integrations/README.md](https://github.com/OpenForecaster/futuresim/blob/main/integrations/README.md): Verifiers and OpenReward integration details.
