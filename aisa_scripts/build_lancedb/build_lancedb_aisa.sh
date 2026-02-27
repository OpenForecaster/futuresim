#!/bin/bash
# Build LanceDB table from articles and embeddings (Stage 1/2, AISA paths).
# This stage ingests data + creates scalar date index only.
# Full-text / vector indices are built in build_index_aisa.sh (Stage 2/2).
#
# Example SLURM:
#   sbatch --cpus-per-task=8 --mem=128G --tmp=50G aisa_scripts/build_lancedb/build_lancedb_aisa.sh

set -euo pipefail

# Setup PATH
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
source ~/.bashrc 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SHARED_DATA_ROOT="/mnt/nfs/datasets_ac"

export SOFT_FILELOCK=1
export PYTHONUNBUFFERED=1

# Paths
ARTICLES_DIR="${SHARED_DATA_ROOT}/news/deduped_articles/data"
EMBEDDINGS_DIR="${SHARED_DATA_ROOT}/news/deduped_articles/embeddings"
OUTPUT_DIR="${SHARED_DATA_ROOT}/news/deduped_articles/lance"

# Model
MODEL="Qwen3-Embedding-8B"

# Date range
START_DATE="2023-01-01"
END_DATE="2026-01-31"

# Activate environment
if [ -f "${REPO_DIR}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${REPO_DIR}/.venv/bin/activate"
elif [ -f "${REPO_DIR}/fsim/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${REPO_DIR}/fsim/bin/activate"
fi
cd "${REPO_DIR}"

SCALAR_INDEX_TIMEOUT_MINUTES="${SCALAR_INDEX_TIMEOUT_MINUTES:-30}"
BUILD_SCALAR_INDEX="${BUILD_SCALAR_INDEX:-1}"

echo "Building LanceDB table (stage 1/2)..."
echo "Date range: ${START_DATE} to ${END_DATE}"
echo "Model: ${MODEL}"
echo "Output: ${OUTPUT_DIR}"
echo "Scalar index timeout (minutes): ${SCALAR_INDEX_TIMEOUT_MINUTES}"

if [[ "${BUILD_SCALAR_INDEX}" == "1" ]]; then
  echo "Scalar index on 'date': enabled"
  SCALAR_FLAG=()
else
  echo "Scalar index on 'date': skipped (BUILD_SCALAR_INDEX=0)"
  SCALAR_FLAG=(--skip_scalar_index)
fi

ARGS=(
  --start_date "${START_DATE}"
  --end_date "${END_DATE}"
  --model "${MODEL}"
  --articles_dir "${ARTICLES_DIR}"
  --embeddings_dir "${EMBEDDINGS_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --skip_fts_index
  --scalar_index_timeout_minutes "${SCALAR_INDEX_TIMEOUT_MINUTES}"
  --overwrite
)

python -u scripts/build_lancedb.py "${ARGS[@]}" "${SCALAR_FLAG[@]}"

echo "LanceDB stage 1/2 complete (table + scalar index)."
