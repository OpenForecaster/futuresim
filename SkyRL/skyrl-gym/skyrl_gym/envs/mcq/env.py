from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.mcq import utils
from typing import Dict, Any


class MCQEnv(BaseTextEnv):
    """Environment for multiple-choice question (MCQ) tasks."""

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self.ground_truth = extras["reward_spec"]["ground_truth"]

    def step(self, action: str) -> BaseTextEnvStepOutput:
        done = True
        score_info = utils.compute_score(action, self.ground_truth)
        reward = score_info["score"]
        return BaseTextEnvStepOutput(
            observations=[],
            reward=reward,
            done=done,
            metadata={"acc": score_info["acc"], "pred": score_info["pred"]},
        )
