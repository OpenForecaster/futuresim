#!/bin/bash
# Convert JSONL news articles to Parquet format.
# Uses streaming/batched processing - memory scales with batch size, not total data.
# Run via: python submit_job.py --cpus 32 --memory 100

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export FSIM_REPO_DIR="${FSIM_REPO_DIR:-${REPO_DIR}}"
NEWS_BASE="${FSIM_NEWS_BASE:-/is/cluster/fast/sgoel/forecasting/news}"

# Paths
DATA_INPUT_1="${FSIM_LEGACY_DEDUPED_DIR_1:-${NEWS_BASE}/articlesuntil2024/deduped}"
DATA_INPUT_2="${FSIM_LEGACY_DEDUPED_DIR_2:-${NEWS_BASE}/articles2025/deduped}"
DATA_OUTPUT="${FSIM_PARQUET_OUTPUT_DIR:-${NEWS_BASE}/deduped_articles}"
WORKERS=32
BATCH_SIZE=32  # Keep small to limit memory (~60-70GB peak per batch)

# Activate environment
source "${REPO_DIR}/.venv/bin/activate"
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
