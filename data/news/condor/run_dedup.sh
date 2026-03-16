#!/bin/bash
# Run deduplication on JSONL files
# Usage: ./run_dedup.sh

set -euo pipefail

# Setup PATH for minimal HTCondor environment
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
source ~/.bashrc 2>/dev/null || true

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

# Activate environment
source "${REPO_DIR}/.venv/bin/activate"

# Run deduplication
cd "${REPO_DIR}"
python data/news/scripts/deduplicate_news_jsonl.py "$@"

echo "Deduplication complete!"
