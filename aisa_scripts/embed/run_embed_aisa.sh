#!/bin/bash
# Embed article chunks for semantic search (AISA paths).
# Can run directly or as a SLURM array task.
#
# Direct mode:
#   bash run_embed_aisa.sh <worker_id> <num_workers>
#
# SLURM mode:
#   sbatch --array=0-7 run_embed_aisa.sh 8

set -euo pipefail

# Setup PATH for minimal environments
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
if ! command -v module >/dev/null 2>&1; then
  # Initialize Environment Modules without sourcing user .bashrc.
  if [ -f /etc/profile.d/modules.sh ]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh 2>/dev/null || true
  fi
fi

SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
if [[ -n "${AISA_REPO_DIR:-}" ]]; then
  REPO_DIR="${AISA_REPO_DIR}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/embed_articles.py" ]]; then
  REPO_DIR="${SLURM_SUBMIT_DIR}"
else
  REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
SHARED_DATA_ROOT="/mnt/nfs/datasets_ac"

if [[ ! -f "${REPO_DIR}/scripts/embed_articles.py" ]]; then
  echo "ERROR: Could not locate repo root. REPO_DIR=${REPO_DIR}" >&2
  echo "Set AISA_REPO_DIR or submit from repository root." >&2
  exit 1
fi

export SOFT_FILELOCK=1
export PYTHONUNBUFFERED=1
export HF_HOME="${SHARED_DATA_ROOT}/cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Force vLLM V0 engine — V1 has a fatal hang bug for embed() on this cluster.
export VLLM_USE_V1=0
mkdir -p "${HF_HOME}"

# Paths
ARTICLES_DIR="${SHARED_DATA_ROOT}/news/deduped_articles/data"
OUTPUT_DIR="${SHARED_DATA_ROOT}/news/deduped_articles/embeddings"

# Model
MODEL="Qwen3-Embedding-8B"
MODEL_PATH="${SHARED_DATA_ROOT}/models/Qwen3-Embedding-8B"

# Date range (2023 onwards)
START_DATE="2023-01-01"
END_DATE="2026-01-31"
BATCH_SIZE="${EMBED_BATCH_SIZE:-32}"

# Worker config
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  WORKER_ID="${SLURM_ARRAY_TASK_ID}"
  NUM_WORKERS="${1:-${SLURM_ARRAY_TASK_COUNT:-1}}"
else
  WORKER_ID="${1:-0}"
  NUM_WORKERS="${2:-1}"
fi

# Load CUDA for flash_attn if available
if command -v module >/dev/null 2>&1; then
  CUDA_MODULE="${CUDA_MODULE:-CUDA/12.4}"
  module load "${CUDA_MODULE}" || module load CUDA/12.6 || module load CUDA/12.1.1 || true
fi

# Activate environment
if [ -f "${REPO_DIR}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${REPO_DIR}/.venv/bin/activate"
else
  echo "ERROR: Missing virtualenv at ${REPO_DIR}/.venv/bin/activate"
  exit 2
fi
cd "${REPO_DIR}"

echo "Starting embedding job..."
echo "Worker: ${WORKER_ID} / ${NUM_WORKERS}"
echo "Model: ${MODEL_PATH}"
echo "Date range: ${START_DATE} to ${END_DATE}"

python scripts/embed_articles.py \
    --start_date "${START_DATE}" \
    --end_date "${END_DATE}" \
    --model "${MODEL}" \
    --model_path "${MODEL_PATH}" \
    --articles_dir "${ARTICLES_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --chunk_tokens 512 \
    --batch_size "${BATCH_SIZE}" \
    --worker_id "${WORKER_ID}" \
    --num_workers "${NUM_WORKERS}" \
    --resume

echo "Embedding complete for worker ${WORKER_ID}!"
