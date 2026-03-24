#!/bin/bash
# Run SkyRL OpenForesight warmup search training on HTCondor GPU nodes.
# Arguments:
#   $1 = config path

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export FSIM_REPO_DIR="${FSIM_REPO_DIR:-${REPO_ROOT}}"
CONFIG_PATH="$(readlink -f "${1:?Usage: $0 <config-path>}")"

module load cuda/12.9

# Some execute nodes start with a minimal PATH.
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"

if [ -z "${HOME:-}" ] || [ "${HOME}" = "/" ] || [ ! -w "${HOME}" ]; then
    export HOME="${REPO_ROOT}"
fi

# Prefer the repo `.venv` (same env as `uv sync` + SkyRL submodule deps), then fall back to `.skyrl-venv`.
ACTIVATED=0
for _env in ".venv" ".skyrl-venv"; do
    if [ ! -f "${REPO_ROOT}/${_env}/bin/activate" ]; then
        continue
    fi
    # shellcheck disable=SC1090
    source "${REPO_ROOT}/${_env}/bin/activate"
    if python - <<'PY' >/dev/null 2>&1
import skyrl
import skyrl_gym
PY
    then
        ACTIVATED=1
        echo "Using Python environment: ${_env}"
        break
    fi
    deactivate || true
done

if [ "${ACTIVATED}" -ne 1 ]; then
    echo "No usable Python environment found (need skyrl + skyrl_gym in .venv or .skyrl-venv)." >&2
    exit 1
fi

# Load optional runtime secrets/settings.
if [ -f "${REPO_ROOT}/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    source "${REPO_ROOT}/.env"
    set +a
fi

# Match mpi_scripts/run_sim/run_sim.sh: cores go to cwd; kernel core_pattern "core" ignores FSIM_* unless we chdir here.
setup_core_dump_dir() {
    local default_core_dump_dir="${REPO_ROOT}/logs/core-dumps"
    if [ -n "${FSIM_SKYRL_LOG_BASE:-}" ]; then
        default_core_dump_dir="$(dirname "${FSIM_SKYRL_LOG_BASE}")/core-dumps"
    fi
    export FSIM_CORE_DUMP_DIR="${FSIM_CORE_DUMP_DIR:-${default_core_dump_dir}}"
    mkdir -p "${FSIM_CORE_DUMP_DIR}"
    cd "${FSIM_CORE_DUMP_DIR}"
}

setup_core_dump_dir

if [ -z "${WANDB_API_KEY:-}" ]; then
    USER_HOME="$(getent passwd "$(id -un)" | cut -d: -f6 || true)"
    WANDB_NETRC_PATH=""
    if [ -n "${USER_HOME}" ] && [ -f "${USER_HOME}/.netrc" ]; then
        WANDB_NETRC_PATH="${USER_HOME}/.netrc"
    elif [ -f "${HOME}/.netrc" ]; then
        WANDB_NETRC_PATH="${HOME}/.netrc"
    fi
    if [ -n "${WANDB_NETRC_PATH}" ]; then
        WANDB_API_KEY="$(
            python - "${WANDB_NETRC_PATH}" <<'PY'
import netrc
import sys

path = sys.argv[1]
try:
    auth = netrc.netrc(path).authenticators("api.wandb.ai")
except Exception:
    auth = None
if auth and auth[2]:
    print(auth[2], end="")
PY
        )"
        export WANDB_API_KEY
    fi
fi

# Keep submodule and repo-local integration modules importable.
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/SkyRL:${REPO_ROOT}/third_party/SkyRL/skyrl-gym:${PYTHONPATH:-}"

# /fast has no flock; use soft file lock where needed.
export SOFTFILELOCK=1

# Keep datasets cache on HOME (locking-safe), and model cache on /fast.
export HF_HOME="${HF_HOME:-/fast/sgoel/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HOME}/.cache/huggingface/datasets}"

# Avoid tiny /tmp quota issues on cluster jobs.
export TMPDIR="${TMPDIR:-/fast/sgoel/tmp}"
mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${TMPDIR}"

# Ray: use node-local Condor scratch when available (avoids raylet/worker socket EOF seen when
# plasma/session lived only on shared lustre under /fast on some MPI execute nodes).
# Pick a base for Ray state. Prefer HTCondor execute scratch (local disk). Avoid long mktemp
# prefixes: Ray's AF_UNIX socket paths must stay under the OS limit (~107 bytes including session suffix).
_RAY_BASE=""
for _cand in "${_CONDOR_SCRATCH_DIR:-}" "${CONDOR_SCRATCH_DIR:-}"; do
    if [ -n "${_cand}" ] && [ -d "${_cand}" ]; then
        _RAY_BASE="${_cand}"
        break
    fi
done
if [ -z "${_RAY_BASE}" ]; then
    _RAY_BASE="${TMPDIR}"
fi
if [ -z "${RAY_TMPDIR:-}" ]; then
    export RAY_TMPDIR="${_RAY_BASE}/r"
    mkdir -p "${RAY_TMPDIR}"
fi
export RAY_USE_MULTIPROCESSING_CPU_COUNT="${RAY_USE_MULTIPROCESSING_CPU_COUNT:-1}"
export RAY_DISABLE_DOCKER_CPU_WARNING="${RAY_DISABLE_DOCKER_CPU_WARNING:-1}"

# vLLM v1 path is required in our stack.
export VLLM_USE_V1=1

# Optional (debugging): SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1 merges Ray/vLLM infra into job .out (noisy).

# Ray opens many fds (object store, workers); low defaults can cause raylet/worker EOF on some nodes.
ulimit -n 65536 2>/dev/null || ulimit -n 8192 2>/dev/null || true

# Keep full GPU visibility and let SkyRL map LOCAL_RANK from ray.get_gpu_ids().
# This avoids duplicate-GPU NCCL init failures on some packed H100 nodes.
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "Running SkyRL OpenForesight training with config: ${CONFIG_PATH}"
echo "Core dumps will land in: ${FSIM_CORE_DUMP_DIR}"
echo "RAY_TMPDIR=${RAY_TMPDIR} (TMPDIR=${TMPDIR}, _CONDOR_SCRATCH_DIR=${_CONDOR_SCRATCH_DIR:-})"

RUN_LOG_PATH="$(sed -n 's/^  log_path: //p' "${CONFIG_PATH}" | head -n1)"
if [ -n "${RUN_LOG_PATH}" ]; then
    mkdir -p "${RUN_LOG_PATH}"
    export SIM_OUTPUT_DIR="${SIM_OUTPUT_DIR:-${RUN_LOG_PATH}}"
    copy_ray_logs() {
        if [ -d "${RAY_TMPDIR:-}" ]; then
            mkdir -p "${RUN_LOG_PATH}/ray"
            cp -a "${RAY_TMPDIR}/." "${RUN_LOG_PATH}/ray/" 2>/dev/null || true
        fi
    }
    trap copy_ray_logs EXIT
fi

python -u "${REPO_ROOT}/scripts/run_skyrl_openforesight_search.py" --config "${CONFIG_PATH}"

echo "SkyRL training script exited cleanly."
