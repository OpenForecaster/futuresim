#!/bin/bash
# Run JSON → JSONL conversion
# Usage: ./run_jsonl_conversion.sh

set -euo pipefail

# Setup PATH for minimal HTCondor environment
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
source ~/.bashrc 2>/dev/null || true

# Activate environment
source ~/forecast-sim/.venv/bin/activate

# Parse arguments
JSON_DIR=$1
OUTPUT_DIR=$2
shift 2
EXTRA_ARGS="$@"

# Create output directory if needed
mkdir -p "$OUTPUT_DIR"

# Run conversion
cd /home/sgoel/forecast-sim
python data/news/scripts/to_jsonl.py "$JSON_DIR" --output_dir "$OUTPUT_DIR" $EXTRA_ARGS

echo "JSONL conversion complete!"
