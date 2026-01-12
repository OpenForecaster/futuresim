#!/bin/bash
# Convert JSONL news articles to Parquet format.
# Uses streaming/batched processing - memory scales with batch size, not total data.
# Run via: python submit_job.py --cpus 32 --memory 100

set -euo pipefail

# Paths
REPO_DIR="/home/sgoel/forecast-sim"
DATA_INPUT_1="/is/cluster/fast/sgoel/forecasting/news/articlesuntil2024/deduped"
DATA_INPUT_2="/is/cluster/fast/sgoel/forecasting/news/articles2025/deduped"
DATA_OUTPUT="/is/cluster/fast/sgoel/forecasting/news/deduped_articles"
WORKERS=32
BATCH_SIZE=128  # 4x workers for efficient parallelism

# Activate environment
source ~/forecast/bin/activate
cd "$REPO_DIR"

# Run conversion
echo "Starting JSONL to Parquet conversion (streaming mode)..."
echo "Input dirs: $DATA_INPUT_1, $DATA_INPUT_2"
echo "Output dir: $DATA_OUTPUT"
echo "Workers: $WORKERS, Batch size: $BATCH_SIZE"

python scripts/convert_jsonl_to_parquet.py \
    --input-dirs "$DATA_INPUT_1" "$DATA_INPUT_2" \
    --output-dir "$DATA_OUTPUT" \
    --workers "$WORKERS" \
    --batch-size "$BATCH_SIZE"

echo "Conversion complete!"
