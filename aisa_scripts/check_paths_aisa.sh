#!/usr/bin/env bash
set -euo pipefail

SHARED_ROOT="/mnt/nfs/datasets_ac"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OPTIONAL_AGENT_MODEL="${FSIM_AGENT_MODEL_PATH:-}"

CREATE_DIRS=0
if [[ "${1:-}" == "--create" ]]; then
  CREATE_DIRS=1
fi

required_paths=(
  "${SHARED_ROOT}/models/Qwen3-Embedding-8B"
  "${SHARED_ROOT}/models/qwen3-4b-it-2507"
  "${SHARED_ROOT}/models/gpt-oss-20b"
  "${SHARED_ROOT}/news/deduped_articles/lance/Qwen3-Embedding-8B"
  "${SHARED_ROOT}/cache/huggingface/datasets"
  "${PROJECT_ROOT}/models/Qwen3-Embedding-8B"
)

shared_dirs=(
  "${SHARED_ROOT}/cache/huggingface/datasets"
  "${SHARED_ROOT}/cache/huggingface/hub"
)

project_dirs=(
  "${PROJECT_ROOT}/logs/current_sim"
  "${PROJECT_ROOT}/logs/sims"
  "${PROJECT_ROOT}/logs/lancedb"
  "${PROJECT_ROOT}/logs/embed"
)

echo "Checking AISA runtime paths"
echo "  Shared root:  ${SHARED_ROOT}"
echo "  Project root: ${PROJECT_ROOT}"
if [[ -n "${OPTIONAL_AGENT_MODEL}" ]]; then
  echo "  Agent model:  ${OPTIONAL_AGENT_MODEL}"
fi
echo

missing=0
for p in "${required_paths[@]}"; do
  if [[ -e "${p}" ]]; then
    echo "[OK] ${p}"
  else
    echo "[MISSING] ${p}"
    missing=$((missing + 1))
  fi
done

if [[ -n "${OPTIONAL_AGENT_MODEL}" ]]; then
  if [[ -e "${OPTIONAL_AGENT_MODEL}" ]]; then
    echo "[OK] ${OPTIONAL_AGENT_MODEL}"
  else
    echo "[MISSING] ${OPTIONAL_AGENT_MODEL}"
    missing=$((missing + 1))
  fi
fi

echo
echo "Shared directories (heavy reusable):"
for d in "${shared_dirs[@]}"; do
  if [[ -d "${d}" ]]; then
    echo "[OK] ${d}"
  else
    if [[ "${CREATE_DIRS}" == "1" ]]; then
      mkdir -p "${d}"
      echo "[CREATED] ${d}"
    else
      echo "[MISSING DIR] ${d}"
      missing=$((missing + 1))
    fi
  fi
done

echo
echo "Project directories (logs/outputs):"
for d in "${project_dirs[@]}"; do
  if [[ -d "${d}" ]]; then
    echo "[OK] ${d}"
  else
    if [[ "${CREATE_DIRS}" == "1" ]]; then
      mkdir -p "${d}"
      echo "[CREATED] ${d}"
    else
      echo "[MISSING DIR] ${d}"
      missing=$((missing + 1))
    fi
  fi
done

echo
if [[ "${missing}" -gt 0 ]]; then
  echo "Found ${missing} missing path(s)."
  echo "Tip: run with --create to create writable cache/output/log directories."
  exit 1
fi

echo "All required AISA paths are present."
