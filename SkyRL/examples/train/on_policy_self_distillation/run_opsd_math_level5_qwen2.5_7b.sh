set -x

# On-Policy Self-Distillation (OPSD) for Hendrycks MATH Level 5.
# The SAME model (Qwen2.5-7B-Instruct) acts as both teacher and student.
# Teacher sees question + full step-by-step solution (privileged context).
# Student sees only the question.
# Train: MATH train split (Level 5 only), Eval: MATH test split (all levels).
#
# bash examples/train/on_policy_self_distillation/run_opsd_math_level5_qwen2.5_7b.sh

: "${DATA_DIR:="/fast/nchandak/forecast-sim/data/math_level5"}"
TRAIN_FILE="$DATA_DIR/train.parquet"
TEST_FILE="$DATA_DIR/test.parquet"
: "${LOGGER:=wandb}"

# OPSD: same model for both teacher and student
MODEL="/fast/rolmedo/models/qwen2.5-7b-it"
ADVANTAGE_ESTIMATOR="no_op"
POLICY_LOSS="importance_sampling"
USE_KL_IN_REWARD=true
USE_KL_LOSS=false

# Placement args — 7B needs TP=2 for inference engines
NUM_GPUS=8
NUM_GPUS_PER_NODE=$NUM_GPUS
NUM_INFERENCE_ENGINES=4
INFERENCE_ENGINE_TP_SIZE=2

# Sampling params
TEMPERATURE=1.0
TOP_P=1.0
EVAL_TOP_P=0.7
MAX_PROMPT_LENGTH=$((1024 * 2))
MAX_RESPONSE_LENGTH=$((1024 * 8))

APPLY_OVERLONG_FILTERING=false

# Training parameters — smaller batch for 7B memory
TRAIN_BATCH_SIZE=256
MINI_BATCH_SIZE=256
N_SAMPLES_PER_PROMPT=4
EVAL_N_SAMPLES_PER_PROMPT=8
ENFORCE_EAGER=true
LR=1e-6

# Project and run name
PROJECT_NAME="math_level5_opsd"
RUN_NAME="opsd_qwen2.5_7b_self_distill"

uv run --isolated --extra fsdp -m examples.train.on_policy_self_distillation.main_opsd \
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
  trainer.epochs=20 \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  generator.inference_engine.enforce_eager=$ENFORCE_EAGER \
  generator.apply_overlong_filtering=$APPLY_OVERLONG_FILTERING \
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  generator.sampling_params.temperature=$TEMPERATURE \
  generator.sampling_params.top_p=$TOP_P \
  generator.eval_sampling_params.temperature=$TEMPERATURE \
  generator.eval_sampling_params.top_p=$EVAL_TOP_P \
  generator.eval_sampling_params.max_generate_length=8192 \
  generator.eval_n_samples_per_prompt=$EVAL_N_SAMPLES_PER_PROMPT \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.policy.optimizer_config.num_warmup_steps=10 \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.use_kl_in_reward=$USE_KL_IN_REWARD \
  trainer.algorithm.use_entropy_loss=true \
  trainer.algorithm.entropy_loss_coef=0.02 \
  trainer.policy.fsdp_config.fsdp_size=$NUM_GPUS_PER_NODE \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=false \
  generator.batched=true \
  environment.env_class=aime \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.inference_engine.gpu_memory_utilization=0.7 \
  trainer.logger="$LOGGER" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.run_name="$RUN_NAME" \
  trainer.resume_mode=latest \
  trainer.export_path="/lustre/scratch/nchandak/forecast-sim/skyrl/${PROJECT_NAME}/exports/$RUN_NAME" \
  trainer.hf_save_interval=10 \
  trainer.max_ckpts_to_keep=3 \
  trainer.ckpt_interval=10 \
  trainer.ckpt_path="/lustre/scratch/nchandak/forecast-sim/skyrl/${PROJECT_NAME}/${RUN_NAME}" \
  $@
