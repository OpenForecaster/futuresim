"""Forecasting reward utilities: answer/probability extraction, Brier scoring, and LLM judge."""

import os
import re
import logging
from typing import Optional, Dict, Any

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def extract_answer(completion: str) -> Optional[str]:
    """Extract text from the last <answer>...</answer> tag pair."""
    matches = list(re.finditer(r"<answer>(.*?)</answer>", completion, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()[:100]


def extract_probability(completion: str) -> Optional[float]:
    """Extract probability from the last <probability>...</probability> tag pair."""
    matches = list(re.finditer(r"<probability>(.*?)</probability>", completion, re.DOTALL))
    if not matches:
        return None
    try:
        prob = float(matches[-1].group(1).strip())
        if prob < -0.01 or prob > 1.01:
            logger.warning("Probability %.4f out of range, clipping", prob)
            prob = max(0.0, min(1.0, prob))
        return prob
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Brier scoring
# ---------------------------------------------------------------------------

def calculate_brier_score_freeform(probability: float, is_correct: bool) -> float:
    """Freeform Brier: -(1-p)^2 if correct, -(1+p^2) if incorrect. Range [-2, 0]."""
    if is_correct:
        return -((1 - probability) ** 2)
    else:
        return -(1 + probability ** 2)


def calculate_brier_score_binary(probability: float, resolution: int) -> float:
    """Binary Brier: (1-p)^2 if resolution=1, p^2 if resolution=0. Range [0, 1]."""
    if resolution == 1:
        return (1 - probability) ** 2
    else:
        return probability ** 2


# ---------------------------------------------------------------------------
# Judge prompt (mirrors verl verifier.py)
# ---------------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = """Your task is to judge whether the given response to a question matches a given ground truth answer or not. You are provided with a question, a ground truth response, and the response you need to judge.
For a response to "match", it must have the same information as in the ground-truth (not less nor unnecessary extra).
The response can be more specific than the ground-truth (for example, "Labrador" is more specific than "dog"), or have additional possible correct answers. But it must cover everything mentioned in the ground-truth. It is okay if it covers it in different words, i.e. paraphrased.
For numeric answers, the relative error, defined as |response - ground truth| / mean(response, ground truth), must be <= 1% for the response to be judged as a correct match. Here, if the ground truth is a specific numeric quantity but the response is a range, then they don't match (even if the range contains the ground truth).

Possible judgments:

"0": The response does not match the ground-truth answer.
"1": The response matches the ground-truth.

Question: "{question}"
Ground truth: "{ground_truth}"
Response: "{student_answer}"

Your job is to ONLY check whether the given response matches the ground truth answer or not in the context of the question. You DO NOT NEED to assess the correctness of the response. This is part of an automated evaluation process, therefore you MUST OUTPUT your final answer as "0" or "1" in <answer> </answer> tags.
Think step by step and end your response with <answer>0</answer> OR <answer>1</answer> TAGS."""


def _build_judge_prompt(question: str, ground_truth: str, student_answer: str) -> str:
    return JUDGE_PROMPT_TEMPLATE.format(
        question=question, ground_truth=ground_truth, student_answer=student_answer,
    )


# ---------------------------------------------------------------------------
# Judge client (connects to a vLLM HTTP server)
# ---------------------------------------------------------------------------

class JudgeClient:
    """Stateless client for a vLLM-served judge model."""

    _instance: Optional["JudgeClient"] = None

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or os.environ.get(
            "FORECAST_JUDGE_URL", "http://localhost:8234/v1"
        )
        self._available: Optional[bool] = None
        self._model_name: Optional[str] = None

    @classmethod
    def get_instance(cls) -> "JudgeClient":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def is_available(self) -> bool:
        if self._available is None:
            try:
                resp = requests.get(f"{self.base_url}/models", timeout=5)
                if resp.status_code == 200:
                    models = resp.json().get("data", [])
                    if models:
                        self._model_name = models[0].get("id")
                    self._available = True
                else:
                    self._available = False
            except Exception:
                self._available = False
        return self._available

    def judge(self, question: str, ground_truth: str, student_answer: str) -> int:
        """Returns 1 if student_answer matches ground_truth, 0 otherwise."""
        if not self.is_available():
            return self._fallback_match(student_answer, ground_truth)

        prompt = _build_judge_prompt(question, ground_truth, student_answer)
        try:
            resp = requests.post(
                f"{self.base_url}/completions",
                json={
                    "model": self._model_name or "judge",
                    "prompt": prompt,
                    "max_tokens": 512,
                    "temperature": 0,
                },
                timeout=60,
            )
            text = resp.json()["choices"][0]["text"]
            match = re.search(r"<answer>\s*([01])\s*</answer>", text)
            if match:
                return int(match.group(1))
            # Fallback: last occurrence of 0 or 1
            last0, last1 = text.rfind("0"), text.rfind("1")
            if last0 == -1 and last1 == -1:
                return self._fallback_match(student_answer, ground_truth)
            return 1 if last1 > last0 else 0
        except Exception as exc:
            logger.warning("Judge request failed: %s – falling back to string match", exc)
            return self._fallback_match(student_answer, ground_truth)

    @staticmethod
    def _fallback_match(student_answer: str, ground_truth: str) -> int:
        """Case-insensitive string matching fallback."""
        if student_answer.lower().strip() == ground_truth.lower().strip():
            return 1
        # Numeric fallback: relative error <= 1%
        try:
            s, g = float(student_answer), float(ground_truth)
            mean_val = (abs(s) + abs(g)) / 2
            if mean_val > 0 and abs(s - g) / mean_val <= 0.01:
                return 1
        except (ValueError, TypeError):
            pass
        return 0


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def compute_score_binary(action: str, resolution: int) -> Dict[str, Any]:
    """Compute reward for binary forecasting questions."""
    response = action
    if "</think>" in response:
        response = response.split("</think>", 1)[1]

    probability = extract_probability(response)

    if probability is not None and -0.01 <= probability <= 1.01:
        probability = max(0.0, min(1.0, probability))
        brier = calculate_brier_score_binary(probability, resolution)
        score = -brier          # range [-1, 0]
        format_reward = 0.0
        correctness = 1.0 if (
            (resolution == 1 and probability >= 0.5)
            or (resolution == 0 and probability < 0.5)
        ) else 0.0
    else:
        score = -0.25           # uninformative prior penalty
        format_reward = -1.0
        correctness = 0.0
        probability = None

    reward = score + format_reward
    return {
        "reward": reward,
        "metadata": {
            "brier": score,
            "format_reward": format_reward,
            "correctness": correctness,
            "probability": probability,
            "is_binary": True,
        },
    }


def compute_score_freeform(
    action: str,
    ground_truth: str,
    question: str,
    judge: JudgeClient,
    add_correctness: bool = True,
) -> Dict[str, Any]:
    """Compute reward for freeform forecasting questions."""
    response = action
    if "</think>" in response:
        response = response.split("</think>", 1)[1]

    answer = extract_answer(response)
    probability = extract_probability(response)

    if answer and probability is not None and -0.01 <= probability <= 1.01:
        probability = max(0.0, min(1.0, probability))
        correctness = float(judge.judge(question, ground_truth, answer))
        brier = 1 + calculate_brier_score_freeform(probability, correctness == 1.0)
        format_reward = 0.0
        reward = brier + format_reward
        if add_correctness:
            reward += correctness
    else:
        reward = -1.0           # format penalty
        format_reward = -1.0
        brier = 0.0
        correctness = 0.0

    return {
        "reward": reward,
        "metadata": {
            "brier": brier,
            "format_reward": format_reward,
            "correctness": correctness,
            "probability": probability,
            "answer": answer,
            "is_binary": False,
        },
    }
