#!/bin/bash
# Run simulation job with AISA path conventions.
# Arguments: $1 = config path

# Source bashrc before strict mode (lmod uses unset vars)
source ~/.bashrc 2>/dev/null || true

set -euo pipefail

# Use SLURM_SUBMIT_DIR if available (batch jobs), else resolve from script path
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export FSIM_REPO_DIR="${FSIM_REPO_DIR:-${REPO_DIR}}"

# Load repo .env early so FSIM_VENV_PATH and shared path overrides can affect
# environment activation and cache locations.
if [ -f "${REPO_DIR}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_DIR}/.env"
  set +a
fi

SHARED_DATA_ROOT="${FSIM_HF_HOME:-/mnt/nfs/datasets_ac/cache/huggingface}"
VENV_PATH="${FSIM_VENV_PATH:-${REPO_DIR}/.venv}"

# Optional conda setup (if available on this node).
if [ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
  conda activate base || true
fi

# Activate project virtual environment.
if [ -f "${VENV_PATH}/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${VENV_PATH}/bin/activate"
else
  echo "ERROR: Missing virtualenv at ${VENV_PATH}/bin/activate"
  exit 2
fi

if command -v module >/dev/null 2>&1; then
  CUDA_MODULE="${CUDA_MODULE:-CUDA/12.6.0}"
  module load "${CUDA_MODULE}" || module load CUDA/12.1.1 || module load cuda/12.1 || true
fi

# Keep Hugging Face caches on shared storage.
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${FSIM_DATASET_CACHE:-${SHARED_DATA_ROOT}/datasets}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${SHARED_DATA_ROOT}/hub}"
mkdir -p "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}"

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
CONFIG_PATH="$(readlink -f "${CONFIG_PATH}")"

setup_core_dump_dir() {
  local default_core_dump_dir="${REPO_DIR}/logs/core-dumps"
  if [ -n "${FSIM_SIM_LOG_BASE:-}" ]; then
    default_core_dump_dir="$(dirname "${FSIM_SIM_LOG_BASE}")/core-dumps"
  fi
  export FSIM_CORE_DUMP_DIR="${FSIM_CORE_DUMP_DIR:-${default_core_dump_dir}}"
  mkdir -p "${FSIM_CORE_DUMP_DIR}"
  cd "${FSIM_CORE_DUMP_DIR}"
}

setup_core_dump_dir

echo "Using virtualenv: ${VENV_PATH}"
echo "Running simulation with config: ${CONFIG_PATH}"
echo "Core dumps will land in: ${FSIM_CORE_DUMP_DIR}"
python -u "${REPO_DIR}/scripts/test_basic_agent.py" --config "${CONFIG_PATH}"
echo "Simulation complete!"
