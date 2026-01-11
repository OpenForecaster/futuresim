#!/bin/bash
# Convert JSONL news articles to Parquet format.
# Run via: python submit_job.py --cpus 32 --memory 256

set -euo pipefail

# Paths
REPO_DIR="/home/sgoel/forecast-sim"
DATA_INPUT_1="/is/cluster/fast/sgoel/forecasting/news/articlesuntil2024/deduped"
DATA_INPUT_2="/is/cluster/fast/sgoel/forecasting/news/articles2025/deduped"
DATA_OUTPUT="/is/cluster/fast/sgoel/forecasting/news/deduped_articles"
WORKERS=32

# Activate environment
source ~/forecast/bin/activate
cd "$REPO_DIR"

# Run conversion
echo "Starting JSONL to Parquet conversion..."
echo "Input dirs: $DATA_INPUT_1, $DATA_INPUT_2"
echo "Output dir: $DATA_OUTPUT"
echo "Workers: $WORKERS"

python scripts/convert_jsonl_to_parquet.py \
    --input-dirs "$DATA_INPUT_1" "$DATA_INPUT_2" \
    --output-dir "$DATA_OUTPUT" \
    --workers "$WORKERS"

echo "Conversion complete!"
