#!/bin/bash
# Run JSON → JSONL conversion
# Usage: ./run_jsonl_conversion.sh

set -euo pipefail

# Setup PATH for minimal HTCondor environment
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
source ~/.bashrc 2>/dev/null || true

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

# Activate environment
source "${REPO_DIR}/.venv/bin/activate"

# Parse arguments
JSON_DIR=$1
OUTPUT_DIR=$2
shift 2
EXTRA_ARGS="$@"

# Create output directory if needed
mkdir -p "$OUTPUT_DIR"

# Run conversion
cd "${REPO_DIR}"
python data/news/scripts/to_jsonl.py "$JSON_DIR" --output_dir "$OUTPUT_DIR" $EXTRA_ARGS

echo "JSONL conversion complete!"
