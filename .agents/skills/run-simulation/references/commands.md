# Commands

## Environment-Backed Defaults

These values come from `.env` and the tracked examples in `.env.example`, `.env.mpi.example`, and `.env.aisa.example`.

- `FSIM_DATASET_PATH`
- `FSIM_DATASET_CACHE`
- `FSIM_OUTPUT_BASE`
- `FSIM_MATCHER_MODEL`
- `FSIM_EMBEDDING_MODEL`
- `FSIM_SEARCH_DB`
- `FSIM_SIM_LOG_BASE`

`scripts/test_basic_agent.py` loads `.env` directly. `pathing.py` handles repo env loading and `${FSIM_*}` expansion for config-driven code.

## Local Simulation

```bash
uv sync
source .venv/bin/activate

python scripts/test_basic_agent.py \
  --config configs/metaculus_sim.yaml \
  --start_date 2025-04-01 \
  --end_date 2025-04-05
```

Search-enabled run:

```bash
python scripts/test_basic_agent.py \
  --config configs/search_sim.yaml \
  --search_db "${FSIM_SEARCH_DB}"
```

## HTCondor Submission

Dry-run first:

```bash
python mpi_scripts/run_sim/submit_sim.py \
  --config configs/default_sim.yaml \
  --dry-run
```

Submit one or more runs:

```bash
python mpi_scripts/run_sim/submit_sim.py \
  --config configs/default_sim.yaml \
  --runs 3 \
  --set sim_name=my_test_run \
  --set resources.cpus=32 \
  --set resources.memory_gb=120
```

Explicit scaffold example:

```bash
python mpi_scripts/run_sim/submit_sim.py \
  --config configs/warmup_only_qwen3.5_27b.yaml \
  --runs 1 \
  --set sim_name=qwenallq_warmup_only_qwen3.5-27b \
  --set defaults.scaffold=qwenallq
```

## Outputs

- Direct runs: `FSIM_OUTPUT_BASE/<sim_name>/<timestamp>/`
- Cluster logs: `FSIM_SIM_LOG_BASE/<sim_name>/`
- Common artifacts:
  - `config.json`
  - `actions.jsonl`
  - `daily_metrics.csv` (one cumulative row per wakeup session)
  - `test_daily_metrics.csv` (same metrics, test questions only)
  - `agents/<agent_id>/model_outputs.jsonl`
  - `agents/<agent_id>/memory/`

## OpenForesight Session And Split Knobs

Useful shared config fields for OpenForesight runs:

- `timegap_days`: wake the agent every `N` days instead of daily
- `split`: main split to simulate, usually `train` or `test`
- `prepend_train_resolution_start`
- `prepend_train_resolution_end`
- `subsample_per_month`

The OpenForesight loader uses the main `split` plus an optional prepended `train` window. Prepended questions are tagged with `source_split="train"` and the main split keeps its own `source_split`, which is what enables `test_daily_metrics.csv`.
