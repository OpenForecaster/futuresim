#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

exec bash "${REPO_DIR}/mpi_scripts/hf_upload/run_hf_upload.sh" "$@"
