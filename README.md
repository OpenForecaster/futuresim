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
| `environment/` | Simulation environment, scoring, data loading |
| `scripts/` | CLI scripts for running simulations |
| `configs/` | YAML configuration files |
| `data/` | Data fetchers and news pipeline |
| `third_party/` | External code dependencies (e.g., SkyRL submodule) |
| `mpi_scripts/` | HTCondor cluster job submission |
| `notes/` | Design documentation and decisions |

## Key Commands

### Run Simulation
```bash
# AllQAgent (warmup on all questions Day 0)
python scripts/test_basic_agent.py --config configs/allq_sim.yaml

# With news search enabled
python scripts/test_basic_agent.py --config configs/search_sim.yaml
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
```

## Documentation

- **[notes/memory.md](notes/memory.md)** — Design decisions, scoring approach, architecture
- **[agents/search_tools/README.md](agents/search_tools/README.md)** — Search infrastructure setup
- **[data/news/README.md](data/news/README.md)** — News pipeline (download, embed, index)
- **[mpi_scripts/run_sim/README.md](mpi_scripts/run_sim/README.md)** — Cluster job submission
- **[third_party/SKYRL_MAINTENANCE.md](third_party/SKYRL_MAINTENANCE.md)** — Fork/upstream SkyRL submodule maintenance workflow

## Output

Simulation results are saved to `/fast/sgoel/forecasting/current_sim/<sim_name>/<timestamp>/`:
- `config.json` — Run configuration
- `actions.jsonl` — All predictions and resolutions
- `daily_metrics.csv` — Per-day scores
- `agents/<agent_id>/` — Per-agent logs and memory
