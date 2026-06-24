#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export FSIM_REPO_DIR="${FSIM_REPO_DIR:-$(pwd)}"
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

: "${OPENREWARD_API_KEY:?Set OPENREWARD_API_KEY to your OpenReward key.}"
: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY for the answer matcher.}"

CODEX_PATH="${CODEX_PATH:-$(command -v codex || true)}"
if [[ -z "$CODEX_PATH" ]]; then
  echo "Set CODEX_PATH or put codex on PATH." >&2
  exit 1
fi
export CODEX_PATH

CONFIG="${CONFIG:-configs/minimalHarness/aljazeera2026Q1_codex_gpt55_openreward_search.yaml}"
cmd=(python scripts/run_forecast_sim.py --config "$CONFIG" "$@")

printf 'Launching:'
printf ' %q' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"
