import re


def extract_mcq_answer(response: str) -> str | None:
    """Extract an MCQ answer letter (A, B, C, D) from a model response.

    Tries multiple patterns in order of specificity:
    1. <answer>X</answer> tags (SDPO format)
    2. \\boxed{X}
    3. "The answer is X" / "Answer: X"
    4. Standalone letter at end of response
    """
    response = response.strip()

    # Pattern 1: <answer>X</answer> tags
    match = re.search(r"<answer>\s*([A-Da-d])\s*</answer>", response)
    if match:
        return match.group(1).upper()

    # Pattern 2: \boxed{X}
    match = re.search(r"\\boxed\{([A-Da-d])\}", response)
    if match:
        return match.group(1).upper()

    # Pattern 3: "The answer is X" or "Answer: X"
    match = re.search(r"(?:the answer is|answer:\s*)\(?([A-Da-d])\)?", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Pattern 4: last standalone letter (A-D) in the response
    match = re.search(r"\b([A-Da-d])\s*[.)\]]*\s*$", response)
    if match:
        return match.group(1).upper()

    return None


def compute_score(response: str, ground_truth: str) -> dict:
    """Compute MCQ score. Returns dict with score and metadata."""
    predicted = extract_mcq_answer(response)
    if predicted is None:
        return {"score": 0.0, "acc": 0.0, "pred": None}

    correct = predicted.upper() == ground_truth.upper()
    return {"score": 1.0 if correct else 0.0, "acc": 1.0 if correct else 0.0, "pred": predicted}
