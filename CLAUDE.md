# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Forecast-Sim is a multi-agent LLM forecasting simulator. Agents read news articles, predict on free-form forecasting questions, and are scored against each other using proper scoring rules (Brier, peer-relative). The goal is training forecasting behaviors that transfer to real prediction markets (Metaculus, OpenForesight).

## Setup and Commands

```bash
# Install (requires uv, Python 3.12, Linux x86_64)
uv sync
source .venv/bin/activate

# Run tests
pytest tests/
pytest tests/test_scoring.py          # single test file

# Run a simulation
python scripts/test_basic_agent.py \
    --start_date 2025-04-01 --end_date 2025-04-05 --sim_name test_run

# Run from YAML config
python scripts/test_basic_agent.py --config configs/qwen3.5/allq_nomem_restart_qwen3.5_27b.yaml

# Resume from last checkpoint
python scripts/test_basic_agent.py --resume /path/to/output_dir

# Restart from a specific day
python scripts/test_basic_agent.py \
    --restart_from /path/to/original/run --restart_from_day 2025-04-05

# Submit HTCondor cluster jobs
python mpi_scripts/run_sim/submit_sim.py --config configs/shared/metaculus_sim.yaml --runs 3
```

SkyRL training uses the **same repo `.venv`** as the rest of forecast-sim: run `uv sync` at the repo root, then `cd third_party/SkyRL && uv sync --active --extra fsdp` so the SkyRL submodule’s dependencies install into that env (see `.agents/skills/skyrl-training/references/setup-and-launch.md`). FlashAttention is optional — install manually if needed.

## Architecture

### Core Loop (`environment/env.py`)
`SimulationEnvironment` runs a daily loop: resolve maturing questions → compute scores → call each agent's `act()` → log results. Thread-safe via locks. Agents interact through `SimForecastInterface` which exposes: `list_questions()`, `get_market_csv_path()`, `submit_prediction(qid, {outcome: prob})`, `search(query, from_date, to_date)`, `get_article(id)`, `query(python_code)`.

### Agent Variants (`agents/`)
- **BaseAgent** (`base.py`): Abstract class. `act(doc_interface, forecast_interface, current_date) -> List[actions]`.
- **BasicAgent** (`basicAgent/agent.py`): Standard day-by-day agent with optional memory and search. This is the largest file (~39KB) — handles LLM prompting, action parsing, memory snapshots.
- **AllQAgent** (`allQAgent/agent.py`): Warmup variant — Day 0 iterates through ALL questions (parallelized), then standard BasicAgent behavior on subsequent days.
- **AllQDailyAgent**: Every day predicts on each question sequentially, no DataFrame queries.
- **GPTOSSBasicAgent/GPTOSSAllQAgent** (`gptossAgent/`): OpenAI Responses API variants with extended thinking support.
- **QwenBasicAgent/QwenAllQAgent** (`qwenAgent/`): vLLM Chat Completions with native `tool_calls` — **intended for Qwen3.5** (`qwen3_coder` parser). **Qwen3** should use `BasicAgent`/`AllQAgent` scaffolds with `vllm_enable_tools: false` (see `.agents/skills/agent-scaffolds/references/model-specific-notes.md`).

### Scoring (`environment/scoring/`)
Brier score (`1 - Σ(p_i - y_i)²`), peer score (`100 × (my_score - avg_others)`), time-weighted peer score. Answer matching uses LLM-based semantic matching with Union-Find for transitive closure (`environment/ansmatching.py`).

### Search Infrastructure (`agents/search_tools/`)
LanceDB-powered hybrid search (semantic + keyword via tantivy) over CommonCrawl news articles. Embeddings generated with sentence-transformers (Qwen3-Embedding-8B). Setup docs in `agents/search_tools/README.md`.

### Data Pipeline (`data/`)
- `data/fetchqs/`: Question loaders for OpenForesight and Metaculus datasets
- `data/news/`: Full CommonCrawl pipeline — download, extract, deduplicate, embed, index. Uses `news-please` submodule. Setup docs in `data/news/README.md`.

### Configuration (`configs/`)
YAML files with sections: `restart_from/restart_from_day`, `sim_name`, `start_date/end_date`, `dataset/dataset_path/split`, `search_db/embedding_model`, `matching` (answer matching backend), `defaults` (provider, scaffold, memory, temperature), `agents` (list with model overrides), `resources` (HTCondor GPU/CPU/memory/bid).

### Cluster Submission (`mpi_scripts/`)
HTCondor job wrappers for simulations, embedding generation, LanceDB building, and vLLM serving. Main entry: `mpi_scripts/run_sim/submit_sim.py`.

## Key Design Decisions

- Agents submit predictions as `{outcome_string: probability}` dicts, not indices — outcome strings are semantically matched at resolution time.
- The "Other" outcome absorbs unassigned probability mass (1 - sum of explicit predictions).
- Memory is per-agent, per-day, stored as YAML-structured documents (v2) or plain text. Memory state is snapshotted at end of each day for resume/restart.
- The primary entry point `scripts/test_basic_agent.py` handles both single-agent CLI args and multi-agent YAML configs.
- API keys are loaded from `.env` at project root (never committed).

## Output Structure

Results go to `<base_dir>/<sim_name>/<timestamp>/`:
- `config.json` — run configuration
- `actions.jsonl` — all predictions and resolutions
- `daily_metrics.csv` — per-day per-agent scores
- `matcher.jsonl` — answer matching logs
- `agents/<agent_id>/model_outputs.jsonl` — cleaned model responses
- `agents/<agent_id>/model_raw.jsonl` — full prompt + response
