#!/bin/bash
# Run simulation job on HTCondor GPU node
# Arguments: $1 = config path

set -e

# Setup environment
source /home/sgoel/miniforge3/etc/profile.d/conda.sh
conda activate base
cd /home/sgoel/forecast-sim
source fsim/bin/activate

module load cuda/12.1

# Force vLLM to use the legacy V0 engine to avoid instability and socket errors
# associated with the experimental V1 engine (EngineCoreRequestType error).
export VLLM_USE_V1=0

# Split HF Cache:
# 1. Datasets go to HOME (supports locking, avoids 'flock' errors)
# 2. Models go to FAST (large files, no locking issues usually)
export HF_DATASETS_CACHE="/home/sgoel/.cache/huggingface/datasets"
export HF_HUB_CACHE="/is/cluster/fast/sgoel/hfcache/hub"

# DO NOT set HF_HOME if we are splitting them like this

# Load API keys from .env if exists
if [ -f /home/sgoel/forecast-sim/.env ]; then
    export $(grep -v '^#' /home/sgoel/forecast-sim/.env | xargs)
fi

# Or load from bashrc if OPENROUTER_API_KEY not set
if [ -z "$OPENROUTER_API_KEY" ]; then
    source ~/.bashrc
fi

CONFIG_PATH="$1"

echo "Running simulation with config: $CONFIG_PATH"

python -u scripts/test_basic_agent.py --config "$CONFIG_PATH"

echo "Simulation complete!"
