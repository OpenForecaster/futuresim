# Simulation Job Submission

Submit forecasting simulation jobs to HTCondor.

## Setup

1. Create `.env` file in project root with your OpenRouter API key:
```bash
echo 'OPENROUTER_API_KEY=your-key-here' > /home/sgoel/forecast-sim/.env
```

2. Make sure LanceDB index is built (for search mode):
```bash
python scripts/build_lancedb_index.py
```

## Usage

```bash
# Basic run using config:
python mpi_scripts/run_sim/submit_sim.py --config configs/default_sim.yaml

# Override simulation name and runs:
python mpi_scripts/run_sim/submit_sim.py --config configs/default_sim.yaml --name my_test_run --runs 5

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
| `--name` | optional | Override simulation name prefix |
| `--runs` | 1 | Number of parallel runs (for variance testing) |
| `--gpus` | 1 | GPUs per job |
| `--memory` | 80 | Memory in GB |
| `--bid` | 25 | HTCondor bid |

## Output

- Logs: `mpi_scripts/run_sim/logs/<name>/`
- Results: `/is/cluster/fast/sgoel/forecasting/current_sim/<name>_r<N>/`
- Final metrics: `final_results.json` in each run directory
