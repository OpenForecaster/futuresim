#!/bin/bash
# Run simulation job on HTCondor GPU node (CUDA 12.9 variant)
# Needed for Qwen3.5 models: FlashInfer GDN kernel JIT requires nvcc >=12.8,
# and pip-installed CUDA libs must take precedence to avoid cuBLAS mismatches.
# Arguments: $1 = config path

set -e

cd /home/nchandak/forecast-sim
source .venv/bin/activate
module load cuda/12.9

# Prepend pip-installed CUDA libs (matching PyTorch cu128) so they take
# precedence over system CUDA module libs, avoiding cuBLAS mismatches.
for _pkg in cublas cuda_runtime cudnn cufft curand cusolver; do
    _p=$(python -c "import nvidia.${_pkg}; print(nvidia.${_pkg}.__path__[0])" 2>/dev/null)
    [ -n "$_p" ] && [ -d "${_p}/lib" ] && LD_LIBRARY_PATH="${_p}/lib:${LD_LIBRARY_PATH}"
done
export LD_LIBRARY_PATH

# Force vLLM to use V1 engine (required for vllm 0.11.0+)
export VLLM_USE_V1=1

# Split HF Cache:
# 1. Datasets go to HOME (supports locking, avoids 'flock' errors)
# 2. Models go to FAST (large files, no locking issues usually)
export HF_DATASETS_CACHE="/home/nchandak/.cache/huggingface/datasets"

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
