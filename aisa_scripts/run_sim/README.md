# Simulation Job Submission

Submit forecasting simulation jobs to HTCondor.

## Setup

1. Create `.env` file in project root with your OpenRouter API key:
```bash
echo 'OPENROUTER_API_KEY=your-key-here' > /home/sgoel/forecast-sim/.env
```

2. Make sure LanceDB Stage 2 indices are built (for search mode):
```bash
python scripts/build_lancedb_index.py --build_fts --fts_with_position --force
```

## Usage

```bash
# Basic run using config:
python mpi_scripts/run_sim/submit_sim.py --config configs/default_sim.yaml

# Override config keys at submit time (repeatable):
python mpi_scripts/run_sim/submit_sim.py --config configs/default_sim.yaml --runs 5 \
  --set sim_name=my_test_run \
  --set resources.cpus=32 \
  --set resources.memory_gb=120

# Dry run to check generated config:
python mpi_scripts/run_sim/submit_sim.py --config configs/default_sim.yaml --dry-run
```

## Resuming / Restarting

### Resume (continue from last day)
Continue a simulation that was interrupted. Restores state from `actions.jsonl` and fast-forwards to the last recorded day:

```bash
python scripts/test_basic_agent.py --resume /path/to/past/output_dir
```

### Restart from Specific Day
Re-run from a specific day, preserving all predictions before that day (e.g., to keep costly Day 0 warmup):

```bash
python scripts/test_basic_agent.py \
    --restart_from /path/to/original/run \
    --restart_from_day 2025-04-05
```

This creates a new directory with truncated logs, then resumes from the restart day.

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | required | Path to YAML configuration file |
| `--runs` | 1 | Number of parallel runs (for variance testing) |
| `--set key=value` | repeatable | Override YAML config values (supports dot paths and list indices like `agents[0].model`) |

## Output

- Logs: `/fast/sgoel/logs/forecasting-sim/sims/<sim_name>/`
- Results: `/is/cluster/fast/sgoel/forecasting/current_sim/<sim_name>_r<N>/`
- Final metrics: `final_results.json` in each run directory
