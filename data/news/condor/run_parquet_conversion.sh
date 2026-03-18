#!/bin/bash
# Run Parquet conversion
# Usage: ./run_parquet_conversion.sh

set -euo pipefail

# Setup PATH for minimal HTCondor environment
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
source ~/.bashrc 2>/dev/null || true

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

# Activate environment
source "${REPO_DIR}/.venv/bin/activate"

# Run conversion
cd "${REPO_DIR}"
python data/news/scripts/convert_jsonl_to_parquet.py "$@"

echo "Parquet conversion complete!"
