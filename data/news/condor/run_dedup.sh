#!/bin/bash
# Run deduplication on JSONL files
# Usage: ./run_dedup.sh

set -euo pipefail

# Setup PATH for minimal HTCondor environment
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
source ~/.bashrc 2>/dev/null || true

# Activate environment
source ~/forecast-sim/.venv/bin/activate

# Run deduplication
cd /home/sgoel/forecast-sim
python data/news/scripts/deduplicate_news_jsonl.py "$@"

echo "Deduplication complete!"
