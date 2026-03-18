#!/bin/bash
#SBATCH --job-name=lancedb-index
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/lancedb/lancedb_index_%j.out
#SBATCH --error=logs/lancedb/lancedb_index_%j.err
#
# Build LanceDB search indices (Stage 2/2, AISA paths):
# - FTS on "content" (with_position enabled by default)
# - Optional IVF-PQ vector index (GPU-accelerated when available)
#
# Submit:
#   sbatch aisa_scripts/build_lancedb/build_index_aisa.sh
#
# Optional env overrides:
#   BUILD_FTS=1|0                (default: 1)
#   FTS_WITH_POSITION=1|0        (default: 1)
#   FTS_USE_TANTIVY=1|0          (default: 1)
#   TANTIVY_INDEX_ROOT=<path>    (default: /mnt/nfs/datasets_ac/lancedb_tantivy_indices)
#   BUILD_VECTOR_INDEX=1|0       (default: 1)
#   NUM_PARTITIONS=<int>         (default: 4096)
#   NUM_SUB_VECTORS=<int>        (default: 64)
#   VECTOR_METRIC=cosine|L2|dot  (default: cosine)
#   ACCELERATOR=cuda|""          (default: cuda)
#   LOAD_CUDA_MODULE=1|0         (default: 0)
#   DB_PATH=<path>               (default: /mnt/nfs/datasets_ac/news/deduped_articles/lance/Qwen3-Embedding-8B)
#   TABLE_NAME=<name>            (default: articles)

# Setup PATH — source bashrc before strict mode (lmod uses unset vars)
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
source ~/.bashrc 2>/dev/null || true

set -euo pipefail

# Use SLURM_SUBMIT_DIR if available (batch jobs), else resolve from script path
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SHARED_DATA_ROOT="/mnt/nfs/datasets_ac"

if [[ "${LOAD_CUDA_MODULE:-0}" == "1" ]]; then
  if command -v module >/dev/null 2>&1; then
    module load cuda/12.1 || true
  fi
fi

export SOFT_FILELOCK=1
export PYTHONUNBUFFERED=1

# Activate environment
if [ -f "${REPO_DIR}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${REPO_DIR}/.venv/bin/activate"
else
  echo "ERROR: Missing virtualenv at ${REPO_DIR}/.venv/bin/activate"
  exit 2
fi
cd "${REPO_DIR}"

BUILD_FTS="${BUILD_FTS:-1}"
FTS_WITH_POSITION="${FTS_WITH_POSITION:-1}"
FTS_USE_TANTIVY="${FTS_USE_TANTIVY:-1}"
TANTIVY_INDEX_ROOT="${TANTIVY_INDEX_ROOT:-${SHARED_DATA_ROOT}/lancedb_tantivy_indices}"
BUILD_VECTOR_INDEX="${BUILD_VECTOR_INDEX:-1}"
NUM_PARTITIONS="${NUM_PARTITIONS:-4096}"
NUM_SUB_VECTORS="${NUM_SUB_VECTORS:-64}"
VECTOR_METRIC="${VECTOR_METRIC:-cosine}"
ACCELERATOR="${ACCELERATOR:-cuda}"
DB_PATH="${DB_PATH:-${SHARED_DATA_ROOT}/news/deduped_articles/lance/Qwen3-Embedding-8B}"
TABLE_NAME="${TABLE_NAME:-articles}"

echo "Building LanceDB indices (stage 2/2)..."
echo "DB_PATH=${DB_PATH}"
echo "TABLE_NAME=${TABLE_NAME}"
echo "BUILD_FTS=${BUILD_FTS}"
echo "FTS_WITH_POSITION=${FTS_WITH_POSITION}"
echo "FTS_USE_TANTIVY=${FTS_USE_TANTIVY}"
echo "TANTIVY_INDEX_ROOT=${TANTIVY_INDEX_ROOT}"
echo "BUILD_VECTOR_INDEX=${BUILD_VECTOR_INDEX}"
echo "NUM_PARTITIONS=${NUM_PARTITIONS}"
echo "NUM_SUB_VECTORS=${NUM_SUB_VECTORS}"
echo "VECTOR_METRIC=${VECTOR_METRIC}"
echo "ACCELERATOR=${ACCELERATOR}"

if [[ "${FTS_USE_TANTIVY}" == "1" ]]; then
  mkdir -p "${TANTIVY_INDEX_ROOT}"
fi

if [[ "${BUILD_FTS}" != "1" && "${BUILD_VECTOR_INDEX}" != "1" ]]; then
  echo "Nothing to do: both BUILD_FTS=0 and BUILD_VECTOR_INDEX=0"
  exit 2
fi

ARGS=(
  --db_path "${DB_PATH}"
  --table_name "${TABLE_NAME}"
  --force
  --num_partitions "${NUM_PARTITIONS}"
  --num_sub_vectors "${NUM_SUB_VECTORS}"
  --metric "${VECTOR_METRIC}"
  --tantivy_index_root "${TANTIVY_INDEX_ROOT}"
)

if [[ "${BUILD_FTS}" == "1" ]]; then
  ARGS+=(--build_fts)
  if [[ "${FTS_WITH_POSITION}" == "1" ]]; then
    ARGS+=(--fts_with_position)
  else
    ARGS+=(--no_fts_with_position)
  fi
  if [[ "${FTS_USE_TANTIVY}" == "1" ]]; then
    ARGS+=(--fts_use_tantivy)
  else
    ARGS+=(--fts_use_native)
  fi
else
  ARGS+=(--no_fts_with_position)
fi

if [[ "${BUILD_VECTOR_INDEX}" != "1" ]]; then
  ARGS+=(--skip_vector_index)
fi

if [[ -n "${ACCELERATOR}" ]]; then
  ARGS+=(--accelerator "${ACCELERATOR}")
fi

python -u scripts/build_lancedb_index.py "${ARGS[@]}"

echo "LanceDB stage 2/2 complete."
