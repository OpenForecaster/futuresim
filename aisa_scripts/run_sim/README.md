# Simulation Job Submission

Submit forecasting simulation jobs to HTCondor.

## Setup

1. Create `.env` file in project root with your OpenRouter API key:
```bash
echo 'OPENROUTER_API_KEY=your-key-here' > /home/sgoel/forecast-sim/.env
```

2. Make sure `FSIM_SEARCH_DB` and `FSIM_EMBEDDING_MODEL` point at the shared AISA paths from `.env.aisa.example`.

3. Download the news search artifacts from Hugging Face if those paths are not populated yet.

Prebuilt LanceDB table for search:

- Dataset: `https://huggingface.co/datasets/shash42/forecast-news-embeddings`
- Note: despite the repo name, this currently ships the prebuilt LanceDB table (`config.json` + `articles.lance/...`), not raw per-day `embeddings.npz`

```bash
hf download shash42/forecast-news-embeddings \
  --repo-type dataset \
  --local-dir /mnt/nfs/datasets_ac/news/deduped_articles/lance/Qwen3-Embedding-8B \
  --max-workers 8
```

Canonical parquet corpus for rebuilds or inspection:

- Dataset: `https://huggingface.co/datasets/shash42/forecast-news`

```bash
hf download shash42/forecast-news \
  --repo-type dataset \
  --local-dir /mnt/nfs/datasets_ac/news/deduped_articles/data \
  --max-workers 8
```

4. Make sure the LanceDB search DB is ready for hybrid search.

If you already have a prebuilt LanceDB table at `FSIM_SEARCH_DB`, do not rebuild Stage 1. Run Stage 2 only to create the local search-serving indices needed for keyword/hybrid search:

- rebuild the Tantivy FTS sidecar on AISA
- optionally refresh the IVF-PQ serving index
- keep the shipped LanceDB table itself unchanged

```bash
sbatch aisa_scripts/build_lancedb/build_index_aisa.sh
```

If you only have parquet + embeddings, then you must build Stage 1 and then Stage 2:

```bash
sbatch --cpus-per-task=8 --mem=128G --tmp=50G \
  aisa_scripts/build_lancedb/build_lancedb_aisa.sh

sbatch aisa_scripts/build_lancedb/build_index_aisa.sh
```

Stage 2 defaults are the recommended hybrid-search setup:

- Tantivy-backed FTS with positions
- IVF-PQ vector index enabled
- external/shared Tantivy root under `/mnt/nfs/datasets_ac/lancedb_tantivy_indices`

## Usage

```bash
# Basic run using config:
python mpi_scripts/run_sim/submit_sim.py --config configs/shared/default_sim.yaml

# Override config keys at submit time (repeatable):
python mpi_scripts/run_sim/submit_sim.py --config configs/shared/default_sim.yaml --runs 5 \
  --set sim_name=my_test_run \
  --set resources.cpus=32 \
  --set resources.memory_gb=120

# Dry run to check generated config:
python mpi_scripts/run_sim/submit_sim.py --config configs/shared/default_sim.yaml --dry-run
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
