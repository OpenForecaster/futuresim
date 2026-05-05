#!/bin/bash
# Build LanceDB table from articles and embeddings (Stage 1/2).
# This stage ingests data + creates scalar date index only.
# Full-text / vector indices are built in build_index.sh (Stage 2/2).
#
# Run via: condor_submit_bid 15 build_lancedb.sub

set -euo pipefail

# Setup PATH
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
export PYTHONUNBUFFERED=1
NEWS_BASE="${FSIM_NEWS_BASE:-/is/cluster/fast/sgoel/forecasting/news}"

# Paths
ARTICLES_DIR="${FSIM_NEWS_ARTICLES_DIR:-${NEWS_BASE}/deduped_articles/data}"
EMBEDDINGS_DIR="${FSIM_NEWS_EMBEDDINGS_DIR:-${NEWS_BASE}/deduped_articles/embeddings}"
OUTPUT_DIR="${FSIM_NEWS_LANCEDB_DIR:-${NEWS_BASE}/deduped_articles/lance}"

# Model
MODEL="Qwen3-Embedding-8B"

# Date range
START_DATE="${FSIM_NEWS_START_DATE:-2023-01-01}"
END_DATE="${FSIM_NEWS_END_DATE:-2026-03-31}"

# Activate environment
source "${REPO_DIR}/.venv/bin/activate"
cd "$REPO_DIR"

SCALAR_INDEX_TIMEOUT_MINUTES="${SCALAR_INDEX_TIMEOUT_MINUTES:-30}"
BUILD_SCALAR_INDEX="${BUILD_SCALAR_INDEX:-1}"

echo "Building LanceDB table (stage 1/2)..."
echo "Date range: $START_DATE to $END_DATE"
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"
echo "Scalar index timeout (minutes): $SCALAR_INDEX_TIMEOUT_MINUTES"

if [[ "$BUILD_SCALAR_INDEX" == "1" ]]; then
  echo "Scalar index on 'date': enabled"
  SCALAR_FLAG=()
else
  echo "Scalar index on 'date': skipped (BUILD_SCALAR_INDEX=0)"
  SCALAR_FLAG=(--skip_scalar_index)
fi

ARGS=(
  --start_date "$START_DATE"
  --end_date "$END_DATE"
  --model "$MODEL"
  --articles_dir "$ARTICLES_DIR"
  --embeddings_dir "$EMBEDDINGS_DIR"
  --output_dir "$OUTPUT_DIR"
  --skip_fts_index
  --scalar_index_timeout_minutes "$SCALAR_INDEX_TIMEOUT_MINUTES"
  --overwrite
)

python -u scripts/build_lancedb.py "${ARGS[@]}" "${SCALAR_FLAG[@]}"

echo "LanceDB stage 1/2 complete (table + scalar index)."
