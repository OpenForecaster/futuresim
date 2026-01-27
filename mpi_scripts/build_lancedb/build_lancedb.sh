#!/bin/bash
# Build LanceDB index from articles and precomputed embeddings.
# This is a CPU-only job but needs memory for loading embeddings.
#
# Run via: condor_submit_bid 15 build_lancedb.sub

set -euo pipefail

# Setup PATH
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
source ~/.bashrc 2>/dev/null || true

# Filelock fix for /is/cluster/fast
export SOFT_FILELOCK=1

# Paths
REPO_DIR="/home/sgoel/forecast-sim"
ARTICLES_DIR="/is/cluster/fast/sgoel/forecasting/news/deduped_articles/data"
EMBEDDINGS_DIR="/is/cluster/fast/sgoel/forecasting/news/deduped_articles/embeddings"
OUTPUT_DIR="/is/cluster/fast/sgoel/forecasting/news/deduped_articles/lance"

# Model
MODEL="Qwen3-Embedding-8B"

# Date range
START_DATE="2023-01-01"
END_DATE="2025-12-31"

# Activate environment
source ~/forecast-sim/fsim/bin/activate
cd "$REPO_DIR"

echo "Building LanceDB index..."
echo "Date range: $START_DATE to $END_DATE"
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"

python scripts/build_lancedb.py \
    --start_date "$START_DATE" \
    --end_date "$END_DATE" \
    --model "$MODEL" \
    --articles_dir "$ARTICLES_DIR" \
    --embeddings_dir "$EMBEDDINGS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --overwrite

echo "LanceDB build complete!"
