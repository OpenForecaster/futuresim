#!/bin/bash
# Run fully async SkyRL OpenForesight warmup search training on HTCondor GPU nodes.
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

# Triton/JIT cache writes on shared filesystems can later fail with stale file handles when kernels
# are re-read during long async runs. Keep the whole Triton working set on node-local scratch.
_TRITON_BASE=""
for _cand in "${_CONDOR_SCRATCH_DIR:-}" "${CONDOR_SCRATCH_DIR:-}"; do
    if [ -n "${_cand}" ] && [ -d "${_cand}" ]; then
        _TRITON_BASE="${_cand}/triton"
        break
    fi
done
if [ -z "${_TRITON_BASE}" ]; then
    _TRITON_BASE="${TMPDIR}/triton"
fi
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${_TRITON_BASE}/cache}"
export TRITON_OVERRIDE_DIR="${TRITON_OVERRIDE_DIR:-${_TRITON_BASE}/override}"
export TRITON_DUMP_DIR="${TRITON_DUMP_DIR:-${_TRITON_BASE}/dump}"
mkdir -p "${TRITON_CACHE_DIR}" "${TRITON_OVERRIDE_DIR}" "${TRITON_DUMP_DIR}"

# Reduce CUDA allocator fragmentation when FSDP + vLLM share a GPU (OOM with large free+reserved gaps).
# Using `-` rather than `:-` lets an explicitly empty value disable this for ablations.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF-expandable_segments:True}"

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
# HTCondor: default `=1` matches known-good eval runs (Ray logs "Detected RAY_USE_MULTIPROCESSING_CPU_COUNT=1").
# Without it, some nodes abort during ray.init() right after "Started a local Ray instance" (exit 1, no Python traceback).
# Set RAY_USE_MULTIPROCESSING_CPU_COUNT=0 to opt into Ray's default CPU detection.
export RAY_USE_MULTIPROCESSING_CPU_COUNT="${RAY_USE_MULTIPROCESSING_CPU_COUNT:-1}"
export RAY_DISABLE_DOCKER_CPU_WARNING="${RAY_DISABLE_DOCKER_CPU_WARNING:-1}"
export RAY_USAGE_STATS_ENABLED="${RAY_USAGE_STATS_ENABLED:-0}"
export RAY_USAGE_STATS_PROMPT_ENABLED="${RAY_USAGE_STATS_PROMPT_ENABLED:-0}"
# Some 8-GPU HTCondor nodes intermittently lose the local raylet during bootstrap before any
# SkyRL actors or placement groups exist. Give Ray a gentler startup/reconnect window by default;
# callers can still override these env vars explicitly when debugging.
export RAY_health_check_initial_delay_ms="${RAY_health_check_initial_delay_ms:-240000}"
export RAY_health_check_period_ms="${RAY_health_check_period_ms:-10000}"
export RAY_health_check_timeout_ms="${RAY_health_check_timeout_ms:-120000}"
export RAY_health_check_failure_threshold="${RAY_health_check_failure_threshold:-30}"
export RAY_worker_register_timeout_seconds="${RAY_worker_register_timeout_seconds:-300}"
export RAY_gcs_rpc_server_reconnect_timeout_s="${RAY_gcs_rpc_server_reconnect_timeout_s:-600}"
export RAY_raylet_rpc_server_reconnect_timeout_base_s="${RAY_raylet_rpc_server_reconnect_timeout_base_s:-60}"
export RAY_raylet_rpc_server_reconnect_timeout_max_s="${RAY_raylet_rpc_server_reconnect_timeout_max_s:-300}"
export RAY_core_worker_rpc_server_reconnect_timeout_base_s="${RAY_core_worker_rpc_server_reconnect_timeout_base_s:-60}"
export RAY_core_worker_rpc_server_reconnect_timeout_max_s="${RAY_core_worker_rpc_server_reconnect_timeout_max_s:-300}"
export RAY_py_gcs_connect_timeout_s="${RAY_py_gcs_connect_timeout_s:-300}"
export RAY_nums_py_gcs_reconnect_retry="${RAY_nums_py_gcs_reconnect_retry:-60}"

# Defaults match known-good eval runs (see skyrl_integration/train/main_openforesight_search.py).
# Override with FSIM_RAY_* env if you need larger sync_registries payloads.
export FSIM_RAY_HEAP_MEMORY_BYTES="${FSIM_RAY_HEAP_MEMORY_BYTES:-$((4 * 1024 * 1024 * 1024))}"
export FSIM_RAY_OBJECT_STORE_BYTES="${FSIM_RAY_OBJECT_STORE_BYTES:-$((2 * 1024 * 1024 * 1024))}"

# vLLM v1 path is required in our stack.
export VLLM_USE_V1=1

# Optional (debugging): SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1 merges Ray/vLLM infra into job .out (noisy).

# Ray opens many fds (object store, workers); low defaults can cause raylet/worker EOF on some nodes.
ulimit -n 65536 2>/dev/null || ulimit -n 8192 2>/dev/null || true

# Keep full GPU visibility and let SkyRL map LOCAL_RANK from ray.get_gpu_ids().
# This avoids duplicate-GPU NCCL init failures on some packed H100 nodes.
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

if [ -z "${FSIM_RAY_NODE_IP:-}" ]; then
    FSIM_RAY_NODE_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
    if [ -z "${FSIM_RAY_NODE_IP}" ]; then
        FSIM_RAY_NODE_IP="$(getent ahostsv4 "$(hostname)" 2>/dev/null | awk 'NR==1 {print $1}')"
    fi
    export FSIM_RAY_NODE_IP
fi

maybe_reserve_aux_gpu_for_async_embeddings() {
    local aux_cfg_value
    aux_cfg_value="$(sed -n 's/^  aux_cuda_visible_devices: //p' "${CONFIG_PATH}" | head -n1 | tr -d '[:space:]\"'\''')"
    if [ -z "${aux_cfg_value}" ] || [ "${aux_cfg_value}" = "null" ]; then
        return
    fi

    local ray_gpu_budget="${FSIM_RAY_NUM_GPUS:-}"
    if ! [[ "${ray_gpu_budget}" =~ ^[0-9]+$ ]]; then
        return
    fi

    local -a all_gpu_rows=()
    mapfile -t all_gpu_rows < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits 2>/dev/null | sed '/^$/d')
    local total_visible_gpus="${#all_gpu_rows[@]}"
    if [ "${total_visible_gpus}" -le "${ray_gpu_budget}" ] || [ "${ray_gpu_budget}" -le 0 ]; then
        return
    fi

    local -a ray_gpu_uuids=()
    local -a aux_gpu_uuids=()
    local row index uuid
    local i=0
    for row in "${all_gpu_rows[@]}"; do
        index="${row%%,*}"
        uuid="${row#*,}"
        index="${index//[[:space:]]/}"
        uuid="${uuid#"${uuid%%[![:space:]]*}"}"
        uuid="${uuid%"${uuid##*[![:space:]]}"}"
        if [ "${i}" -lt "${ray_gpu_budget}" ]; then
            ray_gpu_uuids+=("${uuid}")
        else
            aux_gpu_uuids+=("${uuid}")
        fi
        i=$((i + 1))
    done

    local ray_cuda_visible_devices
    local reserved_aux_cuda_visible_devices
    ray_cuda_visible_devices="$(IFS=,; echo "${ray_gpu_uuids[*]}")"
    reserved_aux_cuda_visible_devices="$(IFS=,; echo "${aux_gpu_uuids[*]}")"

    export CUDA_VISIBLE_DEVICES="${ray_cuda_visible_devices}"
    export FSIM_RESERVED_AUX_CUDA_VISIBLE_DEVICES="${reserved_aux_cuda_visible_devices}"
    export FSIM_RAY_NUM_GPUS="${#ray_gpu_uuids[@]}"
    echo "Reserved aux embedding GPU(s): ${FSIM_RESERVED_AUX_CUDA_VISIBLE_DEVICES}"
    echo "Ray-visible CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
}

maybe_reserve_aux_gpu_for_async_embeddings

echo "Running fully async SkyRL OpenForesight training with config: ${CONFIG_PATH}"
echo "Core dumps will land in: ${FSIM_CORE_DUMP_DIR}"
echo "RAY_TMPDIR=${RAY_TMPDIR} (TMPDIR=${TMPDIR}, _CONDOR_SCRATCH_DIR=${_CONDOR_SCRATCH_DIR:-})"
echo "FSIM_RAY_NODE_IP=${FSIM_RAY_NODE_IP:-unset}"

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

run_python_with_ray_bootstrap_retries() {
    local max_attempts="${FSIM_RAY_BOOTSTRAP_ATTEMPTS:-5}"
    local base_sleep_seconds="${FSIM_RAY_BOOTSTRAP_RETRY_SLEEP_SECONDS:-15}"
    local return_timeout_seconds="${FSIM_RAY_BOOTSTRAP_RETURN_TIMEOUT_SECONDS:-180}"
    if ! [[ "${max_attempts}" =~ ^[0-9]+$ ]] || [ "${max_attempts}" -lt 1 ]; then
        max_attempts=1
    fi
    if ! [[ "${base_sleep_seconds}" =~ ^[0-9]+$ ]] || [ "${base_sleep_seconds}" -lt 1 ]; then
        base_sleep_seconds=15
    fi
    if ! [[ "${return_timeout_seconds}" =~ ^[0-9]+$ ]] || [ "${return_timeout_seconds}" -lt 1 ]; then
        return_timeout_seconds=180
    fi

    local attempt exit_code attempt_dir attempt_out attempt_err attempt_artifacts_dir sleep_seconds
    local python_pid bootstrap_started_at now timed_out saw_local_instance saw_returned
    ray stop --force >/dev/null 2>&1 || true
    rm -rf "${RAY_TMPDIR:?}/"*
    mkdir -p "${RAY_TMPDIR}"

    for attempt in $(seq 1 "${max_attempts}"); do
        attempt_dir="${_CONDOR_SCRATCH_DIR:-${TMPDIR}}/skyrl_bootstrap_attempt_${attempt}"
        mkdir -p "${attempt_dir}"
        attempt_out="${attempt_dir}/stdout.log"
        attempt_err="${attempt_dir}/stderr.log"
        : > "${attempt_out}"
        : > "${attempt_err}"

        if [ "${attempt}" -gt 1 ]; then
            sleep_seconds=$((base_sleep_seconds * (attempt - 1)))
            echo "Retrying Ray bootstrap (${attempt}/${max_attempts}) after early startup failure; sleeping ${sleep_seconds}s first..."
            ray stop --force >/dev/null 2>&1 || true
            rm -rf "${RAY_TMPDIR:?}/"*
            mkdir -p "${RAY_TMPDIR}"
            sleep "${sleep_seconds}"
        fi

        saw_local_instance=0
        saw_returned=0
        timed_out=0
        bootstrap_started_at=0

        set +e
        python -u "${REPO_ROOT}/scripts/run_skyrl_openforesight_search_fully_async.py" --config "${CONFIG_PATH}" \
            > >(tee -a "${attempt_out}") \
            2> >(tee -a "${attempt_err}" >&2) &
        python_pid=$!
        set -e

        while kill -0 "${python_pid}" 2>/dev/null; do
            if grep -q 'ray.init: returned' "${attempt_out}" "${attempt_err}" 2>/dev/null; then
                saw_returned=1
                break
            fi
            if grep -q 'Started a local Ray instance' "${attempt_out}" "${attempt_err}" 2>/dev/null; then
                if [ "${saw_local_instance}" -eq 0 ]; then
                    saw_local_instance=1
                    bootstrap_started_at="$(date +%s)"
                fi
                now="$(date +%s)"
                if [ $((now - bootstrap_started_at)) -ge "${return_timeout_seconds}" ]; then
                    echo "Ray bootstrap exceeded ${return_timeout_seconds}s without 'ray.init: returned'; killing attempt ${attempt}/${max_attempts} and retrying..." | tee -a "${attempt_out}" >&2
                    timed_out=1
                    kill "${python_pid}" 2>/dev/null || true
                    sleep 5
                    kill -9 "${python_pid}" 2>/dev/null || true
                    break
                fi
            fi
            sleep 5
        done

        set +e
        wait "${python_pid}"
        exit_code=$?
        set -e

        if [ "${exit_code}" -eq 0 ]; then
            return 0
        fi

        if [ -n "${RUN_LOG_PATH:-}" ]; then
            attempt_artifacts_dir="${RUN_LOG_PATH}/bootstrap_attempt_${attempt}"
            mkdir -p "${attempt_artifacts_dir}"
            cp -f "${attempt_out}" "${attempt_artifacts_dir}/stdout.log" 2>/dev/null || true
            cp -f "${attempt_err}" "${attempt_artifacts_dir}/stderr.log" 2>/dev/null || true
            if [ -d "${RAY_TMPDIR:-}" ]; then
                mkdir -p "${attempt_artifacts_dir}/ray"
                cp -a "${RAY_TMPDIR}/." "${attempt_artifacts_dir}/ray/" 2>/dev/null || true
            fi
        fi

        if [ "${saw_returned}" -eq 1 ]; then
            return "${exit_code}"
        fi

        if [ "${saw_local_instance}" -eq 0 ] && ! grep -q 'Started a local Ray instance' "${attempt_out}" "${attempt_err}" 2>/dev/null; then
            return "${exit_code}"
        fi
    done

    return "${exit_code}"
}

run_python_with_ray_bootstrap_retries

echo "SkyRL training script exited cleanly."
