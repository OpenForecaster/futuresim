#!/bin/bash
#
# Generic job runner for syntheticQA HTCondor jobs.
# Usage: run_job.sh <script> [extra_args...]
#
# Arguments:
#   script      - Python script to run (e.g., generate_qa.py, eval_qa.py)
#   extra_args  - Additional arguments to pass to the Python script
#
set -e
set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export FSIM_REPO_DIR="${FSIM_REPO_DIR:-${REPO_DIR}}"

cd "${REPO_DIR}"
source "${REPO_DIR}/.venv/bin/activate"
module load cuda/12.1

export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1

# Verify CUDA is available
echo "=== syntheticQA job runner ==="
echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}')"

# Get the script to run (first argument)
if [ $# -lt 1 ]; then
    echo "Usage: run_job.sh <script> [extra_args...]"
    echo "Example: run_job.sh generate_qa.py --num_q 10 --num_article 100"
    exit 1
fi

PYTHON_SCRIPT="$1"
shift  # Remove first argument, rest are extra args

# Check if script exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FULL_SCRIPT_PATH="$SCRIPT_DIR/$PYTHON_SCRIPT"
if [ ! -f "$FULL_SCRIPT_PATH" ]; then
    echo "Error: Script not found: $FULL_SCRIPT_PATH"
    exit 1
fi

echo "Running: python -u $FULL_SCRIPT_PATH $@"
echo "================================"

python -u "$FULL_SCRIPT_PATH" "$@"

echo "Job complete!"
