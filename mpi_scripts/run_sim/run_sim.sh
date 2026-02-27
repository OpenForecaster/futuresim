#!/bin/bash
# Run simulation job on HTCondor GPU node
# Arguments: $1 = config path

set -e

# Setup environment
# source /home/sgoel/miniforge3/etc/profile.d/conda.sh
cd /home/nchandak/forecast-sim
source .venv/bin/activate
module load cuda/12.1

# Force vLLM to use V1 engine (required for vllm 0.11.0+)
export VLLM_USE_V1=1

# Split HF Cache:
# 1. Datasets go to HOME (supports locking, avoids 'flock' errors)
# 2. Models go to FAST (large files, no locking issues usually)
export HF_DATASETS_CACHE="/home/nchandak/.cache/huggingface/datasets"

# DO NOT set HF_HOME if we are splitting them like this

# Load API keys from .env if exists
if [ -f /home/nchandak/forecast-sim/.env ]; then
    export $(grep -v '^#' /home/nchandak/forecast-sim/.env | xargs)
fi

# Or load from bashrc if OPENROUTER_API_KEY not set
if [ -z "$OPENROUTER_API_KEY" ]; then
    source ~/.bashrc
fi

CONFIG_PATH="$1"

echo "Running simulation with config: $CONFIG_PATH"

python -u scripts/test_basic_agent.py --config "$CONFIG_PATH"

echo "Simulation complete!"
