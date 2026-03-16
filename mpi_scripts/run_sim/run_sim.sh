#!/bin/bash
# Run simulation job on HTCondor GPU node.
# Arguments: $1 = config path

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export FSIM_REPO_DIR="${FSIM_REPO_DIR:-${REPO_ROOT}}"
cd "${REPO_ROOT}"

source .venv/bin/activate
module load cuda/12.9

CONFIG_PATH="${1:?Usage: $0 <config-path>}"

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

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HOME}/.cache/huggingface/datasets}"
mkdir -p "${HF_DATASETS_CACHE}"

if [ "$(config_uses_local_vllm)" = "1" ]; then
    enable_vllm_compat
fi

echo "Running simulation with config: ${CONFIG_PATH}"

python -u scripts/test_basic_agent.py --config "${CONFIG_PATH}"

echo "Simulation complete!"
