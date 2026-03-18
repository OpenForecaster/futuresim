# Forecast-Sim

Multi-agent forecasting simulator where LLM agents predict on free-form questions and are scored against each other.

## Quick Start

```bash
# 0. Clone with submodules (or run submodule update in an existing clone)
git clone --recurse-submodules <repo-url>
cd forecast-sim
# If already cloned:
git submodule update --init --recursive

# 1. Install dependencies (requires uv)
uv sync
./data/news/scripts/setup_news_pipeline.sh  # Initialize news pipeline

# 2. Activate environment
source .venv/bin/activate

# 3. Run a basic simulation
python scripts/test_basic_agent.py \
    --config configs/metaculus_sim.yaml \
    --start_date 2025-04-01 \
    --end_date 2025-04-05
```

Notes:
- The committed `uv.lock` is aligned to Linux `x86_64` on Python `3.12` (matching the current `fsim` stack).
- PyTorch CUDA wheels are resolved via `tool.uv.sources` in `pyproject.toml`; collaborators can use plain `uv sync`.
- FlashAttention is intentionally not installed by default because support depends on machine/CUDA/toolchain compatibility.
- If you want FlashAttention, follow upstream instructions:
  https://github.com/Dao-AILab/flash-attention#installation-and-features
- Example install after activating `.venv`:
  `MAX_JOBS=4 uv pip install --no-build-isolation flash-attn`

## Directory Structure

| Directory | Description |
|-----------|-------------|
| `agents/` | Agent implementations (BasicAgent, AllQAgent) |
| `.agents/skills/` | Agent-facing repo workflows and architecture guidance |
| `environment/` | Simulation environment, scoring, data loading |
| `scripts/` | CLI scripts for running simulations |
| `configs/` | YAML configuration files |
| `data/` | Data fetchers and news pipeline |
| `third_party/` | External code dependencies (e.g., SkyRL submodule) |
| `mpi_scripts/` | HTCondor cluster job submission |
| `notes/` | Scratch notes and experiment logs; not the source of truth |

## Key Commands

### Run Simulation
```bash
# AllQAgent (warmup on all questions Day 0)
python scripts/test_basic_agent.py --config configs/allq_sim.yaml

# With news search enabled
python scripts/test_basic_agent.py --config configs/search_sim.yaml
```

### Scaffold Names

Scaffold selection is explicit.

- `basic`, `allQ`, `allqd`, and `og` mean the plain base scaffolds.
- `qwenbasic` and `qwenallq` select the Qwen-native tool-calling agents.
- `gptossbasic` and `gptossallq` select the GPT-OSS-specific agents.
- A Qwen or GPT-OSS model will not automatically switch scaffolds anymore just because the model name matches.

Example:

```bash
python mpi_scripts/run_sim/submit_sim.py \
  --config configs/warmup_only_qwen3.5_27b.yaml \
  --runs 1 \
  --set sim_name=qwenallq_warmup_only_qwen3.5-27b \
  --set defaults.scaffold=qwenallq
```

### Resume / Restart
```bash
# Resume from last day
python scripts/test_basic_agent.py --resume /path/to/output_dir

# Restart from specific day (preserves predictions before that day)
python scripts/test_basic_agent.py \
    --restart_from /path/to/original/run \
    --restart_from_day 2025-04-05
```

### Submit Cluster Jobs
```bash
python mpi_scripts/run_sim/submit_sim.py --config configs/metaculus_sim.yaml --runs 3

# SkyRL warmup-style search GRPO training
python mpi_scripts/skyrl_search/submit_skyrl_search_train.py \
  --config configs/skyrl_openforesight_search_warmup_qwen3.5_4b.yaml --runs 1
```

## Documentation

- **[.agents/skills/run-simulation/](.agents/skills/run-simulation/)** — Local runs, HTCondor submission, resume, restart
- **[.agents/skills/forecast-architecture/](.agents/skills/forecast-architecture/)** — Agent/environment interaction, scoring, memory, warmup
- **[.agents/skills/news-pipeline-search/](.agents/skills/news-pipeline-search/)** — News ingestion, embeddings, LanceDB, search-enabled runs
- **[.agents/skills/skyrl-training/](.agents/skills/skyrl-training/)** — SkyRL setup, data prep, launch, submodule maintenance
- **[.agents/skills/agent-scaffolds/](.agents/skills/agent-scaffolds/)** — Explicit scaffold routing and model-specific variants

## Output

Simulation results are saved to `FSIM_OUTPUT_BASE/<sim_name>/<timestamp>/`:
- `config.json` — Run configuration
- `actions.jsonl` — All predictions and resolutions
- `daily_metrics.csv` — One cumulative metrics row per wakeup session, including daily submission count and average TV shift vs the previous submission
- `test_daily_metrics.csv` — Same metrics, filtered to questions whose `source_split` is `test`
- `agents/<agent_id>/model_raw_warmup.jsonl` — Warmup raw logs, grouped by question id and logging only per-turn input deltas
- `agents/<agent_id>/model_raw_daily.jsonl` — Post-warmup raw logs, logging only per-turn input deltas
- `agents/<agent_id>/` — Per-agent logs and memory

## OpenForesight Notes

- `timegap_days` changes the simulator from daily wakeups to one session every `N` days. Prompts mention the last and next wakeup dates during normal sessions, and metrics for active questions are evaluated through the end of that wakeup interval.
- OpenForesight configs can prepend a window from the `train` split ahead of the main `split` with:
  - `prepend_train_resolution_start`
  - `prepend_train_resolution_end`
  - `subsample_per_month`
- Each OpenForesight question carries a `source_split` tag at load time so split-specific metrics can be logged without a separate loader path.
