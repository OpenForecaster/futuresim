"""Forecasting environment for SkyRL training.

Supports both binary (YES/NO with probability) and freeform (answer + probability)
forecasting questions. Uses Brier scoring and optionally an LLM judge for
freeform answer verification.
"""

from typing import Any, Dict, List

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.forecast.utils import (
    JudgeClient,
    compute_score_binary,
    compute_score_freeform,
)


class ForecastEnv(BaseTextEnv):
    """Environment for forecasting tasks with Brier-score rewards."""

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec is required in extras"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required"

        reward_spec = extras["reward_spec"]
        extra_info = extras.get("extra_info", {})

        self.ground_truth = reward_spec["ground_truth"]
        self.question = extra_info.get("question", "")
        self.question_source = extra_info.get("question_source", "")
        self.answer_type = extra_info.get("answer_type", "")
        self.resolution = extra_info.get("resolution", -1)

        self.is_binary = (
            "binary" in self.question_source
            or "metaculus" in self.question_source
            or self.answer_type in ("binary", "binary (yes/no)")
        )

        # Whether to add +1 correctness bonus for freeform (matches verl add_correctness)
        self.add_correctness = True

        # Shared singleton judge client (lazy connect to vLLM server)
        self._judge = JudgeClient.get_instance()

        # Episode metrics
        self._step_metadata: Dict[str, Any] = {}

    def step(self, action: str) -> BaseTextEnvStepOutput:
        done = True

        if self.is_binary:
            result = compute_score_binary(action, self.resolution)
        else:
            result = compute_score_freeform(
                action,
                self.ground_truth,
                self.question,
                self._judge,
                add_correctness=self.add_correctness,
            )

        self._step_metadata = result["metadata"]

        return BaseTextEnvStepOutput(
            observations=[],
            reward=result["reward"],
            done=done,
            metadata=result["metadata"],
        )

    def get_metrics(self) -> Dict[str, Any]:
        return self._step_metadata

    @staticmethod
    def aggregate_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across episodes, split by binary/freeform.

        These appear in wandb under ``environment/`` prefix.
        """
        if not metrics:
            return {}

        binary_metrics = [m for m in metrics if m.get("is_binary", False)]
        freeform_metrics = [m for m in metrics if not m.get("is_binary", False)]

        def _avg(lst, key):
            vals = [m[key] for m in lst if m.get(key) is not None]
            return sum(vals) / len(vals) if vals else 0.0

        def _frac_valid_format(lst):
            """Fraction of samples with valid format (format_reward == 0)."""
            vals = [m.get("format_reward") for m in lst if m.get("format_reward") is not None]
            return sum(1 for v in vals if v >= 0) / len(vals) if vals else 0.0

        # -- overall --
        agg: Dict[str, Any] = {
            "total_count": float(len(metrics)),
            "avg_correctness": _avg(metrics, "correctness"),
            "avg_brier": _avg(metrics, "brier"),
            "avg_format_reward": _avg(metrics, "format_reward"),
            "avg_probability": _avg(metrics, "probability"),
            "valid_format_rate": _frac_valid_format(metrics),
        }

        # -- binary --
        if binary_metrics:
            agg["binary/count"] = float(len(binary_metrics))
            agg["binary/avg_correctness"] = _avg(binary_metrics, "correctness")
            agg["binary/avg_brier"] = _avg(binary_metrics, "brier")
            agg["binary/avg_probability"] = _avg(binary_metrics, "probability")
            agg["binary/valid_format_rate"] = _frac_valid_format(binary_metrics)

        # -- freeform --
        if freeform_metrics:
            agg["freeform/count"] = float(len(freeform_metrics))
            agg["freeform/avg_correctness"] = _avg(freeform_metrics, "correctness")
            agg["freeform/avg_brier"] = _avg(freeform_metrics, "brier")
            agg["freeform/avg_probability"] = _avg(freeform_metrics, "probability")
            agg["freeform/valid_format_rate"] = _frac_valid_format(freeform_metrics)

        return agg
