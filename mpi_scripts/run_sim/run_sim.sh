#!/bin/bash
# Run simulation job on HTCondor GPU node.
# Arguments: $1 = config path

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export FSIM_REPO_DIR="${FSIM_REPO_DIR:-${REPO_ROOT}}"
CONFIG_PATH="$(readlink -f "${1:?Usage: $0 <config-path>}")"

source "${REPO_ROOT}/.venv/bin/activate"
module load cuda/12.9

fix_home() {
    # HTCondor can launch with HOME unset, "/", or a non-writable location.
    if [ -z "${HOME:-}" ] || [ "${HOME}" = "/" ] || [ ! -w "${HOME}" ]; then
        local home_fallback
        home_fallback="$(getent passwd "$(id -u)" | cut -d: -f6 || true)"
        export HOME="${home_fallback:-${REPO_ROOT}}"
    fi
}

load_runtime_env() {
    # Load runtime env from .env (API keys + optional overrides).
    if [ -f "${REPO_ROOT}/.env" ]; then
        set -a
        # shellcheck disable=SC1090
        source "${REPO_ROOT}/.env"
        set +a
    fi

    # Fall back to bashrc only when key is still missing.
    if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -f "${HOME}/.bashrc" ]; then
        # shellcheck disable=SC1090
        source "${HOME}/.bashrc"
    fi
}

setup_core_dump_dir() {
    local default_core_dump_dir="${REPO_ROOT}/logs/core-dumps"
    if [ -n "${FSIM_SIM_LOG_BASE:-}" ]; then
        default_core_dump_dir="$(dirname "${FSIM_SIM_LOG_BASE}")/core-dumps"
    fi
    export FSIM_CORE_DUMP_DIR="${FSIM_CORE_DUMP_DIR:-${default_core_dump_dir}}"
    mkdir -p "${FSIM_CORE_DUMP_DIR}"
    cd "${FSIM_CORE_DUMP_DIR}"
}

prepend_unique_path() {
    local new_path="$1"
    [ -n "${new_path}" ] || return 0
    case ":${LD_LIBRARY_PATH:-}:" in
        *":${new_path}:"*) ;;
        *) export LD_LIBRARY_PATH="${new_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
    esac
}

config_uses_local_vllm() {
    python - "$CONFIG_PATH" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1])) or {}
defaults = cfg.get("defaults") or {}
agents = cfg.get("agents") or []

uses_vllm_agent = defaults.get("provider") == "vllm" or any(
    isinstance(agent, dict) and agent.get("provider") == "vllm"
    for agent in agents
)
uses_vllm_matcher = cfg.get("matching") == "vllm"
uses_vllm_embedding = bool(cfg.get("embedding_model"))

print("1" if (uses_vllm_agent or uses_vllm_matcher or uses_vllm_embedding) else "0")
PY
}

enable_vllm_compat() {
    # Keep the active venv's CUDA libs ahead of module-provided ones so local
    # torch/vLLM/FlashInfer all resolve against the same stack.
    while IFS= read -r libdir; do
        prepend_unique_path "${libdir}"
    done < <(
        python - <<'PY'
import importlib
import os

for pkg in ("cublas", "cuda_runtime", "cudnn", "cufft", "curand", "cusolver"):
    try:
        mod = importlib.import_module(f"nvidia.{pkg}")
    except Exception:
        continue
    base = next(iter(mod.__path__), "")
    libdir = os.path.join(base, "lib")
    if os.path.isdir(libdir):
        print(libdir)
PY
    )

    # SkyRL and modern vLLM both expect the V1 engine path.
    export VLLM_USE_V1="${VLLM_USE_V1:-1}"
}

fix_home
load_runtime_env
setup_core_dump_dir

# FlashInfer uses filelock+flock on FLASHINFER_WORKSPACE_BASE; many shared paths (e.g. /fast) lack flock.
# HTCondor sets local scratch here — use it so JIT locks work (overrides repo .env when present).
if [ -n "${_CONDOR_SCRATCH_DIR:-}" ] && [ -d "${_CONDOR_SCRATCH_DIR}" ] && [ -w "${_CONDOR_SCRATCH_DIR}" ]; then
    export FLASHINFER_WORKSPACE_BASE="${_CONDOR_SCRATCH_DIR}"
fi

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HOME}/.cache/huggingface/datasets}"
mkdir -p "${HF_DATASETS_CACHE}"

if [ "$(config_uses_local_vllm)" = "1" ]; then
    enable_vllm_compat
fi

echo "Running simulation with config: ${CONFIG_PATH}"
echo "Core dumps will land in: ${FSIM_CORE_DUMP_DIR}"

python -u "${REPO_ROOT}/scripts/test_basic_agent.py" --config "${CONFIG_PATH}"

echo "Simulation complete!"
