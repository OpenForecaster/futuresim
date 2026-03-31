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
    --config configs/shared/metaculus_sim.yaml \
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
# Default shared simulation
python scripts/test_basic_agent.py --config configs/shared/default_sim.yaml

# Shared variant without search
python scripts/test_basic_agent.py --config configs/shared/default_nosearch_sim.yaml
```

Shared answer-matching cache:
- Sim runs still fall back to a per-run `matcher_cache.json`.
- If `FSIM_SIM_MATCHER_CACHE_DIR` is set, `split: "test"` runs automatically reuse `<cache_dir>/<matcher_slug>.json` and merge new entries back only when the run exits.
- For non-`test` runs, opt in with top-level YAML:
  `matcher_cache: {enabled: true, path: null}`
- Set `matcher_cache.path` to pin a specific JSON file, or `matcher_cache.enabled: false` to force the old per-run cache.
- On MPI, `FSIM_SIM_MATCHER_CACHE_DIR=/fast/sgoel/forecasting/sim_matcher_cache` is a usable shared root for collaborators on the same cluster, including `nchandak` if permissions already allow it.

### Scaffold Names

Scaffold selection is explicit.

- `basic`, `allQ`, `allqd`, and `og` mean the plain base scaffolds.
- `qwenbasic` and `qwenallq` select **Qwen3.5** vLLM native tool-calling (`agents/qwenAgent`). **Do not use them for Qwen3** — use `basic` / `allQ` / `allqd` with `vllm_enable_tools: false` (see `configs/qwen3/`). Details: `.agents/skills/agent-scaffolds/references/model-specific-notes.md`.
- `gptossbasic` and `gptossallq` select the GPT-OSS-specific agents.
- Qwen scaffolds intentionally do not replay historical hidden thinking across turns; only final assistant content and tool calls are fed back into history.
- A Qwen or GPT-OSS model will not automatically switch scaffolds anymore just because the model name matches.

Example:

```bash
python mpi_scripts/run_sim/submit_sim.py \
  --config configs/qwen3.5/warmup_only_qwen3.5_27b.yaml \
  --runs 1 \
  --set sim_name=qwenallq_warmup_only_qwen3.5-27b \
  --set defaults.scaffold=qwenallq
```

### Token Budgets

- `max_total_tokens` tracks current prompt occupancy/headroom, not cumulative token spend.
- `force_submit_threshold_tokens` is the soft landing threshold: once remaining context is at or below it, budget-aware loops switch into final-submit mode.
- `submit_reserve_tokens` is the hard floor for the action loop: once remaining context drops below it, the loop stops taking more forecast actions.
- Keep `force_submit_threshold_tokens >= submit_reserve_tokens` so there is runway for the final submit turn and its transcript growth.
- Reserve exhaustion currently stops the forecast/action loop only. End-of-session memory update is still attempted afterward unless the scaffold hit a real provider context-limit error.

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
python mpi_scripts/run_sim/submit_sim.py --config configs/shared/metaculus_sim.yaml --runs 3

# SkyRL warmup-style search GRPO training
python mpi_scripts/skyrl_search/submit_skyrl_search_train.py \
  --config configs/skyrl/skyrl_openforesight_search_warmup_qwen3.5_4b.yaml --runs 1
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
