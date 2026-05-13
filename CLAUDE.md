# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Skills

Before making ANY code changes, invoke ALL matching skills from `.claude/skills/`. This is not optional — read the skill references before writing code. Skills are symlinked from `.agents/skills/` (the git-tracked source of truth).

- Create a new skill only when it captures an important recurring pattern or repo-specific workflow that future coding agents really need to know. If the guidance is minor, one-off, or obvious from the code, do not make a skill for it.
- When an important breaking change makes an existing skill outdated, update that skill in the same coding-agent session so the repo guidance stays current.

## Forecast-Sim Agent Guide

This repo is shared across multiple users and clusters. Optimize for collaboration first.

- Do not reward hack by making dummy files or something. When in doubt, please just ask the user, unless the user explicitly told you to keep going until something is fixed, in which case please take rational decisions and raise anything suspicious you might have done to the user later.
- When a design choice has real tradeoffs, ask the user for a preference with concise evidence from the repo instead of hedging or silently picking a path.
- Explain changes clearly and concisely so collaborators can learn the codebase quickly.

- Prefer minimal, clean, easy-to-understand implementations. We will add complexity only as we need it.
- Avoid unnecessary helper layers, `try/except`, or `if/else` bloat when a simpler structure works.
- When classes inherit from higher up classes, sometimes it might be best to make changes to the highest level of class (when it makes sense) and just propagate it to lower ones. In other words, I always want the most modular change possible instead of code bloat.
- Do not add unnecessary function indirection, preferring to write the logic where its used directly unless its really being reused somewhere else.

- Do not commit user-specific absolute paths when a repo-relative path or `FSIM_*` env var can be used.
- For files inside the repo, derive paths from `__file__` or `$BASH_SOURCE` instead of `/home/<user>/forecast-sim`.
- For external data, models, logs, caches, and cluster storage, use the shared `FSIM_*` variables from `.env`.
- On MPI, `/fast/shash42` and `/fast/nchandak` can often be shared by permissions, so do not over-abstract those paths if a simple shared path already works. Treat AISA as the main cross-cluster compatibility boundary.
- Do not commit concrete `restart_from` run directories in shared configs. Pass them with `--set restart_from=...` or keep them local.
- Preserve multi-cluster behavior when editing `configs/`, `mpi_scripts/`, `aisa_scripts/`, `syntheticQA/`, `data/news/`, or other path-sensitive code.
- Before editing path-sensitive code, use the repo skill at `.agents/skills/collaboration-paths/`.

## Shared Instructions

All shared project instructions live in [AGENTS.md](AGENTS.md). Read and follow it — it is the canonical source of project conventions maintained by all collaborators.

## Setup and Commands

```bash
# Install (requires uv, Python 3.12, Linux x86_64)
uv sync
source .venv/bin/activate

# Run tests
pytest tests/
pytest tests/test_scoring.py # single test file

# Run a simulation
python scripts/test_basic_agent.py \
 --start_date 2025-04-01 --end_date 2025-04-05 --sim_name test_run

# Run from YAML config
python scripts/test_basic_agent.py --config configs/shared/default_sim.yaml

# Resume from last checkpoint
python scripts/test_basic_agent.py --resume /path/to/output_dir

# Restart from a specific day
python scripts/test_basic_agent.py \
 --restart_from /path/to/original/run --restart_from_day 2025-04-05

# Submit HTCondor cluster jobs
python mpi_scripts/run_sim/submit_sim.py --config configs/shared/metaculus_sim.yaml --runs 3
```

## Architecture

### Core Loop (`environment/env.py`)
`SimulationEnvironment` runs a daily loop: resolve maturing questions → compute scores → call each agent's `act()` → log shared simulation results. Thread-safe via locks. Agents interact through `SimForecastInterface`, which primarily exposes market snapshot paths, submission/history access, cadence metadata, and env-computed scoring/resolution context. Query/search prompt logic and per-agent transcript logging live in the scaffolds under `agents/`.

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
