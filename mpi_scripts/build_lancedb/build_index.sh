#!/bin/bash
# Build LanceDB search indices (Stage 2/2):
# - FTS on "content" (with_position enabled by default)
# - Optional IVF-PQ vector index
#
# Run via: condor_submit_bid 25 build_index.sub
#
# Optional env overrides:
#   BUILD_FTS=1|0                (default: 1)
#   FTS_WITH_POSITION=1|0        (default: 1)
#   FTS_USE_TANTIVY=1|0          (default: 1)
#   TANTIVY_INDEX_ROOT=<path>    (default: /lustre/... ; used on /is DB paths)
#   BUILD_VECTOR_INDEX=1|0       (default: 1)
#   NUM_PARTITIONS=<int>         (default: 4096)
#   NUM_SUB_VECTORS=<int>        (default: 64)
#   VECTOR_METRIC=cosine|L2|dot  (default: cosine)
#   LOAD_CUDA_MODULE=1|0         (default: 0)
#   DB_PATH=<path>               (default: Qwen3-Embedding-8B db path)
#   TABLE_NAME=<name>            (default: articles)

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

if [[ "${LOAD_CUDA_MODULE:-0}" == "1" ]]; then
  module load cuda/12.1
fi

export SOFT_FILELOCK=1
export PYTHONUNBUFFERED=1
NEWS_BASE="${FSIM_NEWS_BASE:-/is/cluster/fast/sgoel/forecasting/news}"

# Activate environment
source "${REPO_DIR}/.venv/bin/activate"
cd "${REPO_DIR}"

BUILD_FTS="${BUILD_FTS:-1}"
FTS_WITH_POSITION="${FTS_WITH_POSITION:-1}"
FTS_USE_TANTIVY="${FTS_USE_TANTIVY:-1}"
# Tantivy needs POSIX flock — /is/cluster/fast and /lustre/fast do NOT support it.
# /lustre/scratch (BeeGFS) and /home (NFS) do. Prefer scratch to avoid eating home quota.
if [[ -d "/lustre/scratch/${USER}" ]]; then
  TANTIVY_INDEX_ROOT="${TANTIVY_INDEX_ROOT:-/lustre/scratch/${USER}/forecast-sim/lancedb_tantivy_indices}"
else
  TANTIVY_INDEX_ROOT="${TANTIVY_INDEX_ROOT:-${HOME}/forecasting/lancedb_tantivy_indices}"
fi
BUILD_VECTOR_INDEX="${BUILD_VECTOR_INDEX:-1}"
NUM_PARTITIONS="${NUM_PARTITIONS:-4096}"
NUM_SUB_VECTORS="${NUM_SUB_VECTORS:-64}"
VECTOR_METRIC="${VECTOR_METRIC:-cosine}"
DB_PATH="${DB_PATH:-${FSIM_SEARCH_DB:-${NEWS_BASE}/deduped_articles/lance/Qwen3-Embedding-8B}}"
TABLE_NAME="${TABLE_NAME:-articles}"

echo "Building LanceDB indices (stage 2/2)..."
echo "DB_PATH=$DB_PATH"
echo "TABLE_NAME=$TABLE_NAME"
echo "BUILD_FTS=$BUILD_FTS"
echo "FTS_WITH_POSITION=$FTS_WITH_POSITION"
echo "FTS_USE_TANTIVY=$FTS_USE_TANTIVY"
echo "TANTIVY_INDEX_ROOT=$TANTIVY_INDEX_ROOT"
echo "BUILD_VECTOR_INDEX=$BUILD_VECTOR_INDEX"
echo "NUM_PARTITIONS=$NUM_PARTITIONS"
echo "NUM_SUB_VECTORS=$NUM_SUB_VECTORS"
echo "VECTOR_METRIC=$VECTOR_METRIC"

if [[ "$FTS_USE_TANTIVY" == "1" ]]; then
  mkdir -p "$TANTIVY_INDEX_ROOT"
fi

if [[ "$BUILD_FTS" != "1" && "$BUILD_VECTOR_INDEX" != "1" ]]; then
  echo "Nothing to do: both BUILD_FTS=0 and BUILD_VECTOR_INDEX=0"
  exit 2
fi

ARGS=(
  --db_path "$DB_PATH"
  --table_name "$TABLE_NAME"
  --force
  --num_partitions "$NUM_PARTITIONS"
  --num_sub_vectors "$NUM_SUB_VECTORS"
  --metric "$VECTOR_METRIC"
  --tantivy_index_root "$TANTIVY_INDEX_ROOT"
)

if [[ "$BUILD_FTS" == "1" ]]; then
  ARGS+=(--build_fts)
  if [[ "$FTS_WITH_POSITION" == "1" ]]; then
    ARGS+=(--fts_with_position)
  else
    ARGS+=(--no_fts_with_position)
  fi
  if [[ "$FTS_USE_TANTIVY" == "1" ]]; then
    ARGS+=(--fts_use_tantivy)
  else
    ARGS+=(--fts_use_native)
  fi
else
  ARGS+=(--no_fts_with_position)
fi

if [[ "$BUILD_VECTOR_INDEX" != "1" ]]; then
  ARGS+=(--skip_vector_index)
fi

python -u scripts/build_lancedb_index.py "${ARGS[@]}"

echo "LanceDB stage 2/2 complete."
