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

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
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
