#!/bin/bash
# Run simulation job with AISA path conventions.
# Arguments: $1 = config path

# Source bashrc before strict mode (lmod uses unset vars)
source ~/.bashrc 2>/dev/null || true

set -euo pipefail

# Use SLURM_SUBMIT_DIR if available (batch jobs), else resolve from script path
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export FSIM_REPO_DIR="${FSIM_REPO_DIR:-${REPO_DIR}}"
SHARED_DATA_ROOT="${FSIM_HF_HOME:-/mnt/nfs/datasets_ac/cache/huggingface}"

# Optional conda setup (if available on this node).
if [ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
  conda activate base || true
fi

# Activate project virtual environment.
if [ -f "${REPO_DIR}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${REPO_DIR}/.venv/bin/activate"
else
  echo "ERROR: Missing virtualenv at ${REPO_DIR}/.venv/bin/activate"
  exit 2
fi

cd "${REPO_DIR}"

if command -v module >/dev/null 2>&1; then
  module load cuda/12.1 || true
fi

# Keep vLLM on the legacy engine unless explicitly overridden.
export VLLM_USE_V1="${VLLM_USE_V1:-0}"

# Keep Hugging Face caches on shared storage.
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${FSIM_DATASET_CACHE:-${SHARED_DATA_ROOT}/datasets}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${SHARED_DATA_ROOT}/hub}"
mkdir -p "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}"

# Load API keys from repo .env if present.
if [ -f "${REPO_DIR}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_DIR}/.env"
  set +a
fi

# Fallback to shell startup file if key still missing.
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -f "${HOME}/.bashrc" ]; then
  # shellcheck disable=SC1091
  source "${HOME}/.bashrc"
fi

CONFIG_PATH="${1:-}"
if [ -z "${CONFIG_PATH}" ]; then
  echo "Usage: $0 <config_path>"
  exit 1
fi

echo "Running simulation with config: ${CONFIG_PATH}"
python -u scripts/test_basic_agent.py --config "${CONFIG_PATH}"
echo "Simulation complete!"
