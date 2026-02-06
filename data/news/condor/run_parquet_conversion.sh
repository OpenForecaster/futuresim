#!/bin/bash
# Run Parquet conversion
# Usage: ./run_parquet_conversion.sh

set -euo pipefail

# Setup PATH for minimal HTCondor environment
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
source ~/.bashrc 2>/dev/null || true

# Activate environment
source ~/forecast-sim/fsim/bin/activate

# Run conversion
cd /home/sgoel/forecast-sim
python data/news/scripts/convert_jsonl_to_parquet.py "$@"

echo "Parquet conversion complete!"
