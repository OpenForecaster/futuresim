#!/bin/bash
# Embed article chunks for semantic search.
# Uses Qwen3-Embedding-8B model.
# Run via: python submit_job.py --gpus 1 --memory 64 --bid 25 --num_workers 8

set -euo pipefail

# Setup PATH for minimal HTCondor environment
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
source ~/.bashrc 2>/dev/null || true

# Filelock fix for /is/cluster/fast
export SOFT_FILELOCK=1
export HF_HOME=/is/cluster/fast/sgoel/hfcache

# Paths
REPO_DIR="/home/sgoel/forecast-sim"
ARTICLES_DIR="/is/cluster/fast/sgoel/forecasting/news/deduped_articles/data"
OUTPUT_DIR="/is/cluster/fast/sgoel/forecasting/news/deduped_articles/embeddings"

# Model
MODEL="Qwen3-Embedding-8B"
MODEL_PATH="/is/cluster/fast/sgoel/models/Qwen3-Embedding-8B"

# Date range (2023 onwards)
START_DATE="2023-01-01"
END_DATE="2025-12-31"

# Worker config (passed from submit_job.py)
WORKER_ID=${1:-0}
NUM_WORKERS=${2:-1}

# Load CUDA for flash_attn
module load cuda/12.1

# Activate environment
source ~/forecast-sim/fsim/bin/activate
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
    --batch_size 32 \
    --worker_id "$WORKER_ID" \
    --num_workers "$NUM_WORKERS" \
    --resume

echo "Embedding complete for worker $WORKER_ID!"
