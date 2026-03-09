"""
On-Policy Self-Distillation (OPSD) for SkyRL.

Implements the OPSD framework (https://arxiv.org/abs/2601.18734) where the same model
acts as both teacher and student. The teacher is given privileged context (the reference
answer) when scoring student rollouts, while the student sees only the question.

This builds on the existing on-policy distillation example by overriding
`fwd_logprobs_values_reward` to construct separate privileged teacher sequences.
"""

import sys
from typing import List, Optional, Tuple, Dict, Any

import torch
import ray
from loguru import logger

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import initialize_ray
from skyrl.train.generators.base import GeneratorInput, GeneratorOutput
from skyrl.backends.skyrl_train.utils.ppo_utils import register_advantage_estimator
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch

TEACHER_TRANSITION_PROMPT = (
    "\n\nHere is a reference solution to this problem:\n{solution}\n\n"
    "After understanding the reference solution and the rationale behind each step, "
    "now articulate your own step-by-step reasoning that derives the same final answer "
    "to the problem below:\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}."
)


class OPSDTrainer(RayPPOTrainer):
    """
    On-Policy Self-Distillation trainer.

    The same model serves as both student and teacher. The teacher receives privileged
    context (question + reference answer) while scoring the student's on-policy rollouts.
    The student sees only the question.

    Overrides:
      - generate: captures generator_input env_extras for ground-truth extraction
      - convert_to_training_input: carries ground-truth and prompt info through metadata
      - fwd_logprobs_values_reward: builds separate teacher sequences with privileged context
      - apply_reward_kl_penalty: sets rewards to the reverse KL penalty
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_generator_input: Optional[GeneratorInput] = None

    def apply_reward_kl_penalty(
        self,
        data: TrainingInputBatch,
    ) -> TrainingInputBatch:
        """Sets rewards to the reverse KL penalty (teacher - student log probs)."""
        loss_masks_all: torch.Tensor = data["loss_mask"]
        teacher_action_log_probs: torch.Tensor = data["base_action_log_probs"]
        action_log_probs: torch.Tensor = data["action_log_probs"]

        rewards = -(action_log_probs - teacher_action_log_probs) * loss_masks_all
        data["rewards"] = rewards
        return data

    @torch.no_grad()
    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        """Wraps base generate to capture the input batch for ground-truth extraction."""
        self._last_generator_input = input_batch
        return await super().generate(input_batch)

    def _extract_ground_truths(self, env_extras: List[Dict[str, Any]]) -> List[Optional[str]]:
        """Extract ground-truth solutions from env_extras, supporting multiple dataset formats."""
        ground_truths = []
        for extras in env_extras:
            gt = None
            if "reward_model" in extras and isinstance(extras["reward_model"], dict):
                gt = extras["reward_model"].get("ground_truth")
            if gt is None and "reward_spec" in extras and isinstance(extras["reward_spec"], dict):
                gt = extras["reward_spec"].get("ground_truth")
            ground_truths.append(gt)
        return ground_truths

    def convert_to_training_input(
        self, generator_output: GeneratorOutput, uids: List[str]
    ) -> TrainingInputBatch:
        """
        Extends the base conversion to store ground-truth solutions,
        prompt token IDs, and response IDs in metadata for building teacher sequences.
        """
        training_input = super().convert_to_training_input(generator_output, uids)

        training_input.metadata["prompt_token_ids"] = generator_output["prompt_token_ids"]
        training_input.metadata["response_ids"] = generator_output["response_ids"]

        if self._last_generator_input is not None:
            ground_truths = self._extract_ground_truths(
                self._last_generator_input["env_extras"]
            )
            training_input.metadata["ground_truths"] = ground_truths
        else:
            logger.warning("No generator_input captured; ground_truths will be unavailable.")
            training_input.metadata["ground_truths"] = None

        return training_input

    def pad_batch(self, training_input: TrainingInputBatch) -> TrainingInputBatch:
        """Extends base pad_batch to also pad OPSD-specific list metadata."""
        training_input = super().pad_batch(training_input)
        pad_size = training_input.metadata.get("pad_size", 0)
        if pad_size == 0:
            return training_input

        for list_key in ["prompt_token_ids", "response_ids", "ground_truths"]:
            if list_key in training_input.metadata and isinstance(training_input.metadata[list_key], list):
                lst = training_input.metadata[list_key]
                training_input.metadata[list_key] = lst + lst[:pad_size]

        return training_input

    def _build_teacher_prompt_text(self, student_prompt_text: str, ground_truth: str) -> str:
        """Augment the student prompt with the reference solution for the teacher."""
        return student_prompt_text + TEACHER_TRANSITION_PROMPT.format(solution=ground_truth)

    def build_teacher_sequences(
        self,
        training_input: TrainingInputBatch,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build privileged teacher sequences: teacher_prompt + student_response.

        For each sample, replaces the student prompt with a privileged prompt
        (question + ground-truth solution) while keeping the response tokens identical.

        Returns:
            teacher_sequences: (batch_size, teacher_seq_len) - padded
            teacher_attention_mask: (batch_size, teacher_seq_len) - padded
        """
        prompt_token_ids_list = training_input.metadata["prompt_token_ids"]
        response_ids_list = training_input.metadata["response_ids"]
        ground_truths = training_input.metadata["ground_truths"]
        batch_size = len(prompt_token_ids_list)

        teacher_prompt_ids_list = []
        for i in range(batch_size):
            gt = ground_truths[i] if i < len(ground_truths) else None
            if gt is None:
                teacher_prompt_ids_list.append(prompt_token_ids_list[i])
                continue

            student_prompt_text = self.tokenizer.decode(
                prompt_token_ids_list[i], skip_special_tokens=True
            )
            teacher_user_content = self._build_teacher_prompt_text(student_prompt_text, gt)
            teacher_messages = [{"role": "user", "content": teacher_user_content}]
            teacher_prompt_ids = self.tokenizer.apply_chat_template(
                teacher_messages, add_generation_prompt=True, tokenize=True
            )
            teacher_prompt_ids_list.append(teacher_prompt_ids)

        pad_token_id = self.tokenizer.pad_token_id
        max_teacher_prompt_len = max(len(p) for p in teacher_prompt_ids_list)
        max_response_len = max(len(r) for r in response_ids_list)

        teacher_sequences = []
        teacher_attention_masks = []

        for i in range(batch_size):
            t_prompt = teacher_prompt_ids_list[i]
            response = response_ids_list[i]

            t_prompt_len = len(t_prompt)
            resp_len = len(response)

            padded_prompt = [pad_token_id] * (max_teacher_prompt_len - t_prompt_len) + list(t_prompt)
            prompt_attn = [0] * (max_teacher_prompt_len - t_prompt_len) + [1] * t_prompt_len

            padded_response = list(response) + [pad_token_id] * (max_response_len - resp_len)
            response_attn = [1] * resp_len + [0] * (max_response_len - resp_len)

            teacher_sequences.append(padded_prompt + padded_response)
            teacher_attention_masks.append(prompt_attn + response_attn)

        teacher_sequences_tensor = torch.tensor(teacher_sequences, dtype=torch.long)
        teacher_attention_mask_tensor = torch.tensor(teacher_attention_masks, dtype=torch.long)

        return teacher_sequences_tensor, teacher_attention_mask_tensor

    @torch.no_grad()
    def fwd_logprobs_values_reward(
        self,
        training_input: TrainingInputBatch,
    ):
        """
        Compute log probs from teacher (ref) and student (policy) models.

        The key OPSD difference: the ref/teacher model receives privileged sequences
        (prompt + ground-truth + student response), while the policy model receives
        the standard sequences (prompt + student response).
        """
        student_data = training_input.select(
            keys=["sequences", "attention_mask"], metadata_keys=["response_length"]
        )

        values = None
        base_log_probs = None
        action_log_probs = None

        if self.has_critic:
            critic_output = self.dispatch.forward("critic", student_data)
            values = critic_output["output"]

        if self.ref_model is not None:
            has_ground_truths = (
                "ground_truths" in training_input.metadata
                and training_input.metadata["ground_truths"] is not None
                and any(gt is not None for gt in training_input.metadata["ground_truths"])
            )

            if has_ground_truths:
                teacher_sequences, teacher_attention_mask = self.build_teacher_sequences(
                    training_input
                )
                response_length = training_input.metadata["response_length"]

                teacher_data = TrainingInputBatch({
                    "sequences": teacher_sequences,
                    "attention_mask": teacher_attention_mask,
                })
                teacher_data.metadata = {"response_length": response_length}

                ref_output = self.dispatch.forward("ref", teacher_data)
            else:
                logger.warning("No ground_truths available, falling back to standard ref forward.")
                ref_output = self.dispatch.forward("ref", student_data)

            base_log_probs = ref_output["output"]
            self.dispatch.empty_cache("ref")

        policy_output = self.dispatch.forward("policy", student_data)
        action_log_probs = policy_output["output"]

        self.dispatch.empty_cache()

        sequences_all = training_input["sequences"]
        base_log_probs = base_log_probs[: len(sequences_all)] if base_log_probs is not None else None
        action_log_probs = action_log_probs[: len(sequences_all)]
        values = values[: len(sequences_all)] if values is not None else None

        training_input["base_action_log_probs"] = base_log_probs
        training_input["action_log_probs"] = action_log_probs
        training_input["values"] = values

        if training_input.get("rollout_logprobs", None) is not None:
            logprobs_diff = (
                training_input["rollout_logprobs"][training_input["loss_mask"] > 0]
                - action_log_probs[training_input["loss_mask"] > 0]
            ).abs()
            self.all_metrics.update({
                "policy/rollout_train_logprobs_abs_diff_mean": logprobs_diff.mean().item(),
                "policy/rollout_train_logprobs_abs_diff_std": logprobs_diff.std().item(),
            })

        return training_input


@register_advantage_estimator("no_op")
def compute_no_op_advantage(token_level_rewards: torch.Tensor, **kwargs):
    return token_level_rewards, token_level_rewards


class OPSDExp(BasePPOExp):
    def get_trainer(self, *args, **kwargs):
        return OPSDTrainer(*args, **kwargs)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    exp = OPSDExp(cfg)
    exp.run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
