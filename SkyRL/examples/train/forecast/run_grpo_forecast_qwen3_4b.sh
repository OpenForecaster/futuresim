set -x

# GRPO for Forecasting — mirrors verl nosft_retrieval/shuffled_8b.sh exactly.
# Model: Qwen3-4B-Instruct-2507
#
# Key settings from verl:
#   - use_kl_loss=true (NOT kl_in_reward — prevents length collapse)
#   - kl_loss_coef=0.005
#   - grpo_norm_by_std=false (verl: norm_adv_by_std_in_grpo=False)
#   - eval_before_train=false (verl: val_before_train=False)
#   - reward_manager=prime equivalent: zero_variance_filter=true
#
# bash examples/train/forecast/run_grpo_forecast_qwen3_4b.sh

# ─── Data ────────────────────────────────────────────────────────────────────
DATA_DIR="/fast/nchandak/forecasting/datasets/skyrl/forecast"
TRAIN_FILE="$DATA_DIR/openforesight_train_part2.parquet"
TEST_FILE="$DATA_DIR/val.parquet"
LOGGER=wandb

# ─── Model ───────────────────────────────────────────────────────────────────
MODEL="/fast/nchandak/models/Qwen3-4B-Instruct-2507"

# ─── Algorithm (matching verl exactly) ──────────────────────────────────────
ADVANTAGE_ESTIMATOR="grpo"
POLICY_LOSS="regular"
# verl: use_kl_loss=True, NOT kl_in_reward
USE_KL_IN_REWARD=false
USE_KL_LOSS=true
KL_LOSS_COEF=0.005

# ─── GPU / placement ────────────────────────────────────────────────────────
NUM_GPUS=8
NUM_GPUS_PER_NODE=$NUM_GPUS
NUM_INFERENCE_ENGINES=$NUM_GPUS
INFERENCE_ENGINE_TP_SIZE=1

# ─── Sampling params (verl lines 66-67, 111, 116-117) ───────────────────────
TEMPERATURE=1.0
TOP_P=1.0
EVAL_TEMPERATURE=0.6
EVAL_TOP_P=0.95
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=4096

APPLY_OVERLONG_FILTERING=true

# ─── Training params (verl lines 81, 94, 109, 89, 128) ─────────────────────
TRAIN_BATCH_SIZE=256
MINI_BATCH_SIZE=256
N_SAMPLES_PER_PROMPT=8
EVAL_N_SAMPLES_PER_PROMPT=1
ENFORCE_EAGER=true
LR=5e-6

# ─── Checkpointing ──────────────────────────────────────────────────────────
CKPT_INTERVAL=36
MAX_CKPTS=5

# ─── Naming ─────────────────────────────────────────────────────────────────
PROJECT_NAME="forecast_grpo_vs_opsd"
RUN_NAME="grpo_Qwen3-4B_v3_kl-loss_lr${LR}"
CKPT_PATH="/lustre/scratch/nchandak/forecast-sim/skyrl/${PROJECT_NAME}/${RUN_NAME}"

# ─── Judge server setup ─────────────────────────────────────────────────────
JUDGE_MODEL="/fast/nchandak/models/Qwen3-4B-Instruct-2507"
JUDGE_PORT=8234
export FORECAST_JUDGE_URL="http://localhost:${JUDGE_PORT}/v1"

IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
JUDGE_GPU="${GPUS[-1]}"

echo "Starting judge vLLM server on physical GPU $JUDGE_GPU (port $JUDGE_PORT)..."
CUDA_VISIBLE_DEVICES=$JUDGE_GPU python -m vllm.entrypoints.openai.api_server \
    --model "$JUDGE_MODEL" \
    --port $JUDGE_PORT \
    --gpu-memory-utilization 0.3 \
    --max-model-len 1024 \
    --dtype bfloat16 \
    --enforce-eager \
    --disable-log-requests \
    &
JUDGE_PID=$!

echo "Waiting for judge server to start..."
for i in $(seq 1 60); do
    if curl -s "http://localhost:${JUDGE_PORT}/v1/models" > /dev/null 2>&1; then
        echo "Judge server ready!"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "WARNING: Judge server not ready after 5 min, continuing anyway (will use string-match fallback)"
    fi
    sleep 5
done

cleanup() {
    echo "Cleaning up judge server (PID: $JUDGE_PID)..."
    kill $JUDGE_PID 2>/dev/null || true
    wait $JUDGE_PID 2>/dev/null || true
}
trap cleanup EXIT

# ─── Training ────────────────────────────────────────────────────────────────
GPU_MEMORY_UTIL=0.5

uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_base \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="['$TEST_FILE']" \
  trainer.algorithm.advantage_estimator=$ADVANTAGE_ESTIMATOR \
  trainer.algorithm.policy_loss_type=$POLICY_LOSS \
  trainer.policy.model.path=$MODEL \
  trainer.ref.model.path=$MODEL \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TP_SIZE \
  trainer.epochs=7 \
  trainer.eval_batch_size=512 \
  trainer.eval_before_train=false \
  trainer.eval_interval=36 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  generator.inference_engine.enforce_eager=$ENFORCE_EAGER \
  generator.apply_overlong_filtering=$APPLY_OVERLONG_FILTERING \
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  generator.sampling_params.temperature=$TEMPERATURE \
  generator.sampling_params.top_p=$TOP_P \
  generator.eval_sampling_params.temperature=$EVAL_TEMPERATURE \
  generator.eval_sampling_params.top_p=$EVAL_TOP_P \
  generator.eval_sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  generator.eval_n_samples_per_prompt=$EVAL_N_SAMPLES_PER_PROMPT \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.policy.optimizer_config.num_warmup_steps=3 \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.kl_loss_coef=$KL_LOSS_COEF \
  trainer.algorithm.use_kl_in_reward=$USE_KL_IN_REWARD \
  trainer.algorithm.grpo_norm_by_std=false \
  trainer.algorithm.zero_variance_filter=true \
  trainer.algorithm.use_entropy_loss=true \
  trainer.algorithm.entropy_loss_coef=0.02 \
  trainer.policy.fsdp_config.fsdp_size=$NUM_GPUS_PER_NODE \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=false \
  generator.inference_engine.engine_init_kwargs.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  generator.batched=true \
  environment.env_class=forecast \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.inference_engine.gpu_memory_utilization=$GPU_MEMORY_UTIL \
  trainer.logger="$LOGGER" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.run_name="$RUN_NAME" \
  trainer.resume_mode=none \
  trainer.export_path="/lustre/scratch/nchandak/forecast-sim/skyrl/${PROJECT_NAME}/exports/$RUN_NAME" \
  trainer.hf_save_interval=36 \
  trainer.max_ckpts_to_keep=$MAX_CKPTS \
  trainer.ckpt_interval=$CKPT_INTERVAL \
  trainer.ckpt_path="$CKPT_PATH" \
  $@
