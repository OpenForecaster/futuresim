"""
OPSD for Forecasting — overrides the teacher prompt to match forecasting format.

The generic OPSD teacher prompt uses math-style "Answer:" format, which causes
the student to drift away from the forecasting <answer>/<probability> tags.
This module overrides only the teacher transition prompt.
"""

import sys
import ray

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl.train.utils import initialize_ray

# Import the OPSD trainer and advantage estimator from the generic module
from examples.train.on_policy_self_distillation.main_opsd import (
    OPSDTrainer,
    compute_no_op_advantage,  # noqa: F401 — triggers registration
)


class ForecastOPSDTrainer(OPSDTrainer):
    """OPSD trainer with forecasting-specific teacher prompt."""

    def _build_teacher_prompt_text(self, student_prompt_text: str, ground_truth: str) -> str:
        """Augment the student prompt with the reference answer for the teacher.

        Uses forecasting-specific formatting so the teacher's output distribution
        matches the <answer>/<probability> tags the student is expected to produce.
        """
        teacher_hint = (
            "\n\nThe correct answer to this forecasting question is: "
            f"{ground_truth}\n\n"
            "Given this information, provide your reasoning and then state your "
            "final answer in <answer></answer> tags and your confidence as a number "
            "between 0 and 1 in <probability></probability> tags."
        )
        return student_prompt_text + teacher_hint


class ForecastOPSDExp(BasePPOExp):
    def get_trainer(self, *args, **kwargs):
        return ForecastOPSDTrainer(*args, **kwargs)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    exp = ForecastOPSDExp(cfg)
    exp.run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
