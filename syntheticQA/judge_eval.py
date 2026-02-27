#!/usr/bin/env python3
"""
LLM-as-judge scoring for eval_qa.py outputs.

Reads all .jsonl files from an eval output folder, sends each
(question, ground_truth, model_answer) triple to a judge model,
and writes a `score_{judge_model_name}` field (float 0-1) back
into the same files.

Skips records that already have the score field, so the script is
safely re-runnable / resumable.

Usage:
    python syntheticQA/judge_eval.py --input /path/to/eval_output_folder/
    python syntheticQA/judge_eval.py --input /path/to/eval_output_folder/ --judge /fast/nchandak/models/Qwen3-8B
"""

import argparse
import gc
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

def build_judge_prompt(question: str, ground_truth: str, model_answer: str,
                       cot: bool = True) -> str:
    """Build a continuous-score judge prompt (0-1)."""
    prompt = (
        "Your task is to score the given response to a question on a scale "
        "of 0 to 1, where 0 means the response does not match the ground "
        "truth answer and 1 means the response matches the ground truth "
        "answer. You are provided with a question, its ground truth "
        "response, and the response you need to score.\n"
        "For a response to \"match\", it must have at least as much "
        "information as the ground-truth if the ground truth is not numeric "
        "or date related.\n"
        "The response can have more information than the ground-truth. It "
        "can be more specific (for example, \"Labrador\" is more specific "
        "than \"dog\"), or have additional possible correct answers. But it "
        "must cover everything mentioned in the ground-truth. It is okay if "
        "it covers it in different words, i.e. paraphrased.\n"
        "For numeric or date related answers, first compute the relative "
        "error, defined as |response - ground truth| / mean(response, "
        "ground truth). Then, the score of the response is 1 - relative "
        "error. For example, if the ground truth is 6 and the response is "
        "4, then the relative error is |4 - 6| / mean(4, 6) = 2/5 = 0.4. "
        "Hence, the score of the response is 1 - 0.4 = 0.6.\n"
        "\n"
        f"Question: \"{question}\"\n"
        f"Ground truth: \"{ground_truth}\"\n"
        f"Response: \"{model_answer}\"\n"
        "\n"
        "Your job is to SCORE the given response based on how close it is "
        "to the ground truth answer in the context of the question. You "
        "should provide a continuous score only if the ground truth is "
        "numeric or date, otherwise provide only 0 or 1 (binary score). "
        "You DO NOT NEED to assess the correctness of the response. This "
        "is part of an automated evaluation process, therefore you MUST "
        "OUTPUT your final answer in <answer> </answer> tags."
    )
    if cot:
        prompt += (
            "\nThink step by step and end your response with "
            "<answer> XYZ </answer> TAGS where XYZ is the score "
            "(between 0 and 1)."
        )
    else:
        prompt += (
            "\nYOU SHOULD ALWAYS END YOUR RESPONSE WITH "
            "<answer> XYZ </answer> TAGS where XYZ is the score "
            "(between 0 and 1)."
        )
    return prompt


def parse_judge_score(text: str) -> Optional[float]:
    """Extract score from <answer>...</answer> tags. Returns 0-1 or None."""
    matches = list(re.finditer(
        r"<answer>\s*([+-]?\d+(?:\.\d+)?)\s*</answer>", text
    ))
    if matches:
        val = float(matches[-1].group(1))
        return max(0.0, min(1.0, val))
    return None


# ---------------------------------------------------------------------------
# Chat template helper
# ---------------------------------------------------------------------------

def apply_chat_template(tokenizer, prompt: str, model_name: str = "",
                        enable_thinking: bool = True) -> str:
    try:
        chat = [{"role": "user", "content": prompt}]
        if "qwen3" in model_name.lower() and len(model_name) < 10:
            return tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        else:
            return tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True,
            )
    except Exception as e:
        print(f"  Warning: chat template failed ({e}), using raw prompt")
        return prompt


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def save_jsonl(records: List[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_judge(args):
    from vllm import LLM, SamplingParams

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"Error: {args.input} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Derive score field name from judge model
    judge_short = os.path.basename(os.path.normpath(args.judge))
    score_field = f"score_{judge_short.replace('-', '_').replace('.', '_')}"
    print(f"Score field: {score_field}")

    # Discover JSONL files recursively
    jsonl_files = sorted(input_dir.rglob("*.jsonl"))
    if not jsonl_files:
        print("No .jsonl files found (recursive). Exiting.")
        return

    print(f"Found {len(jsonl_files)} .jsonl files")

    # Load all records, remembering file provenance
    file_records: Dict[Path, List[dict]] = {}
    all_to_judge = []  # (file_path, record_index, record)

    for fpath in jsonl_files:
        records = load_jsonl(fpath)
        file_records[fpath] = records
        file_judge_count = 0
        for idx, rec in enumerate(records):
            if score_field in rec:
                continue  # already scored
            question = rec.get("question", "")
            ground_truth = (rec.get("ground_truth", "")
                            or rec.get("answer", "")
                            or rec.get("target", ""))
            model_answer = (rec.get("model_answer", "")
                            or rec.get("extracted_answer", "")
                            or rec.get("response", ""))
            if not question.strip() or not model_answer.strip():
                continue
            all_to_judge.append((fpath, idx, rec))
            file_judge_count += 1
        rel = fpath.relative_to(input_dir)
        already = sum(1 for r in records if score_field in r)
        print(f"  {rel}: {len(records)} records "
              f"({already} already scored, {file_judge_count} to judge)")

    total_records = sum(len(v) for v in file_records.values())
    print(f"\nTotal: {total_records} records, {len(all_to_judge)} need scoring")

    if not all_to_judge:
        print("All records already scored. Nothing to do.")
        _print_summary(file_records, score_field, input_dir)
        return

    # Auto-detect GPU count
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        tp = len(result.stdout.strip().split("\n")) if result.stdout.strip() else 1
    except Exception:
        tp = 1
    print(f"Auto-detected {tp} GPUs")

    # Load judge model
    print(f"\nLoading judge model: {args.judge}")
    print(f"Tensor parallel size: {tp}")

    tokenizer = AutoTokenizer.from_pretrained(args.judge, trust_remote_code=True)
    llm = LLM(
        model=args.judge,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        tensor_parallel_size=tp,
    )
    print("Judge model loaded!")

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.95,
        max_tokens=args.max_tokens,
    )

    # Build judge prompts (always no-think for deterministic judging)
    prompts = []
    for _, _, rec in all_to_judge:
        question = rec.get("question", "")
        ground_truth = (rec.get("ground_truth", "")
                        or rec.get("answer", "")
                        or rec.get("target", ""))
        model_answer = (rec.get("model_answer", "")
                        or rec.get("extracted_answer", "")
                        or rec.get("response", ""))
        raw_prompt = build_judge_prompt(
            question, ground_truth, model_answer, cot=True,
        )
        # raw_prompt += " /no_think"
        formatted = apply_chat_template(
            tokenizer, raw_prompt, judge_short,
            enable_thinking=False,
        )
        prompts.append(formatted)

    print(f"Running batch judging ({len(prompts)} prompts)...")
    outputs = llm.generate(prompts, sampling_params)

    # Parse scores and write back
    scored = 0
    parse_failures = 0
    for i, output in enumerate(outputs):
        text = output.outputs[0].text
        score = parse_judge_score(text)
        if score is None:
            parse_failures += 1
            score = 0.0  # default to 0 if unparseable

        fpath, rec_idx, _ = all_to_judge[i]
        file_records[fpath][rec_idx][score_field] = score
        scored += 1

    print(f"\nScored {scored} records ({parse_failures} parse failures, defaulted to 0.0)")

    # Save all files back
    for fpath, records in file_records.items():
        save_jsonl(records, fpath)
        rel = fpath.relative_to(input_dir)
        print(f"  Saved {rel} ({len(records)} records)")

    _print_summary(file_records, score_field, input_dir)

    # Cleanup
    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def _print_summary(file_records: Dict[Path, List[dict]], score_field: str,
                    base_dir: Path = None):
    """Print per-file and overall accuracy summary."""
    total_scored = 0
    total_score_sum = 0.0
    total_binary_correct = 0

    print(f"\n{'='*60}")
    print(f"Summary ({score_field})")
    print(f"{'='*60}")

    for fpath in sorted(file_records):
        records = file_records[fpath]
        scored = [r for r in records if score_field in r]
        if not scored:
            continue
        scores = [r[score_field] for r in scored]
        avg = sum(scores) / len(scores)
        binary = sum(1 for s in scores if s >= 0.5)
        total_scored += len(scores)
        total_score_sum += sum(scores)
        total_binary_correct += binary
        label = str(fpath.relative_to(base_dir)) if base_dir else fpath.name
        print(f"  {label}: avg={avg:.3f}, "
              f"binary_acc={binary}/{len(scores)} "
              f"({binary / len(scores) * 100:.1f}%)")

    if total_scored > 0:
        overall_avg = total_score_sum / total_scored
        overall_bin = total_binary_correct / total_scored * 100
        print(f"\n  Overall: avg_score={overall_avg:.3f}, "
              f"binary_acc={total_binary_correct}/{total_scored} "
              f"({overall_bin:.1f}%)")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-judge scoring for eval_qa.py outputs. "
                    "Adds a score_{judge_model} field (0-1) to each record."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to eval output folder containing .jsonl files "
             "(e.g. output of eval_qa.py)",
    )
    parser.add_argument(
        "--judge", type=str, default="/fast/nchandak/models/Qwen3-4B",
        help="Path to the judge model (default: /fast/nchandak/models/Qwen3-4B)",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=2048,
        help="Max tokens for judge response (default: 2048)",
    )
    parser.add_argument(
        "--max_model_len", type=int, default=8192,
        help="Max model context length (default: 8192)",
    )
    args = parser.parse_args()

    run_judge(args)


if __name__ == "__main__":
    main()
