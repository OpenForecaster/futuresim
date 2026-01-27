#!/bin/bash
# Build IVF-PQ index for LanceDB on GPU
# Run via: condor_submit_bid 25 build_index.sub

set -euo pipefail

# Setup PATH
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
source ~/.bashrc 2>/dev/null || true
module load cuda/12.1

export SOFT_FILELOCK=1

# Activate environment
source ~/forecast-sim/fsim/bin/activate
cd /home/sgoel/forecast-sim

echo "Building LanceDB vector index..."
# Settings for ~14M rows:
# - partitions: 4096 (sqrt(rows))
# - sub_vectors: 64 (standard PQ)
# - metric: cosine (for semantic similarity)

# Bypass the "Rebuild? [y/N]" prompt by piping "y"
echo "y" | python scripts/build_lancedb_index.py \
    --num_partitions 4096 \
    --num_sub_vectors 64 \
    --metric cosine 

echo "Index build complete!"
