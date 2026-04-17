#!/bin/bash
# Embed article chunks for semantic search.
# Uses Qwen3-Embedding-8B model.
# Run via: python submit_job.py --gpus 1 --memory 64 --bid 25 --num_workers 8

set -euo pipefail

# Setup PATH for minimal HTCondor environment
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
source ~/.bashrc 2>/dev/null || true

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export FSIM_REPO_DIR="${FSIM_REPO_DIR:-${REPO_DIR}}"

if [ -f "${REPO_DIR}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_DIR}/.env"
  set +a
fi

# Filelock fix for /is/cluster/fast
export SOFT_FILELOCK=1
NEWS_BASE="${FSIM_NEWS_BASE:-/is/cluster/fast/sgoel/forecasting/news}"
export HF_HOME="${FSIM_HF_HOME:-/is/cluster/fast/sgoel/hfcache}"

# Paths
ARTICLES_DIR="${FSIM_NEWS_ARTICLES_DIR:-${NEWS_BASE}/deduped_articles/data}"
OUTPUT_DIR="${FSIM_NEWS_EMBEDDINGS_DIR:-${NEWS_BASE}/deduped_articles/embeddings}"

# Model
MODEL="Qwen3-Embedding-8B"
MODEL_PATH="${FSIM_EMBEDDING_MODEL:-/is/cluster/fast/sgoel/models/Qwen3-Embedding-8B}"

# Date range (2023 onwards)
START_DATE="2023-01-01"
END_DATE="2026-03-31"

# Worker config (passed from submit_job.py)
WORKER_ID=${1:-0}
NUM_WORKERS=${2:-1}

# Load CUDA for flash_attn
module load cuda/12.1

# Activate environment
source "${REPO_DIR}/.venv/bin/activate"
cd "$REPO_DIR"

echo "Starting embedding job..."
echo "Worker: $WORKER_ID / $NUM_WORKERS"
echo "Model: $MODEL_PATH"
echo "Date range: $START_DATE to $END_DATE"

python scripts/embed_articles.py \
    --start_date "$START_DATE" \
    --end_date "$END_DATE" \
    --model "$MODEL" \
    --model_path "$MODEL_PATH" \
    --articles_dir "$ARTICLES_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --chunk_tokens 512 \
    --worker_id "$WORKER_ID" \
    --num_workers "$NUM_WORKERS" \
    --resume

echo "Embedding complete for worker $WORKER_ID!"
