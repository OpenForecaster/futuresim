#!/bin/bash
#
# Generic job runner for SkyRL training HTCondor jobs.
# Usage: run_job.sh <training_script.sh> [extra_overrides...]
#
# Arguments:
#   training_script - Shell script to run (relative to SkyRL root),
#                     e.g. examples/train/on_policy_distillation/run_on_policy_distill_math_qwen3_0.6b_from_4b.sh
#   extra_overrides - Additional config overrides forwarded to the training script via $@
#
set -e
set -u

export HOME="/home/nchandak"
export PATH="$HOME/.local/bin:$PATH"

SKYRL_ROOT="/home/nchandak/forecast-sim/SkyRL"
cd "$SKYRL_ROOT"

source /home/nchandak/forecast-sim/.skyrl-venv/bin/activate
module load cuda/12.1
source "$SKYRL_ROOT/.env"

export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1

echo "=== SkyRL training job runner ==="
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

if [ $# -lt 1 ]; then
    echo "Usage: run_job.sh <training_script.sh> [extra_overrides...]"
    exit 1
fi

TRAINING_SCRIPT="$1"
shift

if [ ! -f "$TRAINING_SCRIPT" ]; then
    echo "Error: Script not found: $TRAINING_SCRIPT"
    exit 1
fi

echo "Running: bash $TRAINING_SCRIPT $@"
echo "================================"

bash "$TRAINING_SCRIPT" "$@"

echo "Job complete!"
