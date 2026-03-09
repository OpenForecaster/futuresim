set -x

# On-Policy Self-Distillation (OPSD) for Math reasoning.
# The SAME model acts as both teacher and student. The teacher sees the question + reference
# answer (privileged context), while the student sees only the question.
# See: https://arxiv.org/abs/2601.18734
#
# Setup:
#   uv run examples/algorithms/dapo/prepare_dapo_data.sh
#   bash examples/train/on_policy_self_distillation/run_opsd_math.sh

DATA_DIR="$HOME/data/dapo"
TRAIN_FILE="$DATA_DIR/dapo-math-17k-cleaned.parquet"
TEST_FILE="$DATA_DIR/aime-2024-cleaned.parquet"
LOGGER=wandb

# OPSD: same model for both teacher and student
MODEL="Qwen/Qwen3-1.7B-Base"
ADVANTAGE_ESTIMATOR="no_op"
POLICY_LOSS="importance_sampling"
USE_KL_IN_REWARD=true
USE_KL_LOSS=false

# Placement args (override with e.g. NUM_GPUS=4 bash ...)
NUM_GPUS=8
NUM_GPUS_PER_NODE=$NUM_GPUS
NUM_INFERENCE_ENGINES=$NUM_GPUS
INFERENCE_ENGINE_TP_SIZE=1

# Sampling params
TEMPERATURE=1.0
TOP_P=1.0
EVAL_TOP_P=0.7

# Training params
TRAIN_BATCH_SIZE=256
MINI_BATCH_SIZE=256
N_SAMPLES_PER_PROMPT=8
EVAL_N_SAMPLES_PER_PROMPT=32
ENFORCE_EAGER=true
LR=1e-5

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
  trainer.epochs=10 \
  trainer.eval_batch_size=512 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=2048 \
  generator.inference_engine.enforce_eager=$ENFORCE_EAGER \
  generator.sampling_params.max_generate_length=8192 \
  generator.sampling_params.temperature=$TEMPERATURE \
  generator.sampling_params.top_p=$TOP_P \
  generator.eval_sampling_params.temperature=$TEMPERATURE \
  generator.eval_sampling_params.top_p=$EVAL_TOP_P \
  generator.eval_sampling_params.max_generate_length=8192 \
  generator.eval_n_samples_per_prompt=$EVAL_N_SAMPLES_PER_PROMPT \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.policy.optimizer_config.num_warmup_steps=0 \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.use_kl_in_reward=$USE_KL_IN_REWARD \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.async_engine=false \
  generator.batched=true \
  environment.env_class=aime \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="aime_opsd" \
  trainer.run_name="opsd_aime_qwen3_1.7b_base_self_distill" \
  trainer.resume_mode=latest \
  trainer.export_path="$HOME/exports/aime_opsd_1.7b_base_self_distill" \
  trainer.hf_save_interval=10 \
  trainer.max_ckpts_to_keep=3 \
  trainer.ckpt_interval=10 \
  trainer.ckpt_path="$HOME/ckpts/aime_opsd_1.7b_base_self_distill" \
  $@
