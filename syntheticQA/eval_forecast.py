#!/usr/bin/env python3
"""
Evaluate a model on OpenForesight forecasting questions.

Loads questions from a parquet dataset (e.g. /fast/nchandak/datasets/OpenForesight),
uses the `prompt_without_retrieval` column, and runs batch inference via vLLM.

The model must output its final answer inside <answer>...</answer> tags
and a probability inside <probability>...</probability> tags.

With --with_answer, the ground truth answer is injected into the prompt so the
model learns to reproduce it through its own reasoning (for distillation).

Supports gpt-oss models (Harmony format) and non-gpt-oss models (chat template).

Usage:
    python syntheticQA/eval_forecast.py --split validation
    python syntheticQA/eval_forecast.py --model /fast/nchandak/models/gpt-oss-20b --split validation --effort medium
    python syntheticQA/eval_forecast.py --model /fast/nchandak/models/gpt-oss-20b --split validation --with_answer
"""

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Harmony helpers (shared with eval_qa.py / eval_with_answer.py)
# ---------------------------------------------------------------------------

_harmony_encoding = None
_harmony_import_error = None


def _is_gpt_oss(model_path: str) -> bool:
    return "gpt-oss" in (model_path or "").lower()


def _get_harmony_encoding():
    global _harmony_encoding, _harmony_import_error
    if _harmony_encoding is not None:
        return _harmony_encoding
    if _harmony_import_error is not None:
        return None
    try:
        from openai_harmony import HarmonyEncodingName, load_harmony_encoding
        _harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        return _harmony_encoding
    except Exception as exc:
        _harmony_import_error = exc
        print(f"  [Harmony] Unavailable: {exc}", flush=True)
        return None


def _build_harmony_token_ids(messages: List[Dict[str, str]], effort: str = "medium") -> Optional[List[int]]:
    encoding = _get_harmony_encoding()
    if encoding is None:
        return None
    try:
        from openai_harmony import (
            Conversation, Message, ReasoningEffort, Role, SystemContent,
        )
        effort_map = {
            "low": ReasoningEffort.LOW,
            "medium": ReasoningEffort.MEDIUM,
            "high": ReasoningEffort.HIGH,
        }
        reasoning_effort = effort_map.get(effort.lower(), ReasoningEffort.MEDIUM)
        harmony_messages = [
            Message.from_role_and_content(
                Role.SYSTEM,
                SystemContent.new().with_reasoning_effort(reasoning_effort),
            )
        ]
        role_map = {"system": Role.SYSTEM, "user": Role.USER, "assistant": Role.ASSISTANT}
        for msg in messages:
            role = role_map.get(msg.get("role", "user"), Role.USER)
            content = msg.get("content")
            if content is None:
                continue
            harmony_messages.append(Message.from_role_and_content(role, content))
        convo = Conversation.from_messages(harmony_messages)
        return encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
    except Exception as exc:
        print(f"  [Harmony] Failed to build prompt: {exc}", flush=True)
        return None


def _parse_harmony_output(output_text: str, output_token_ids: Optional[List[int]] = None) -> str:
    encoding = _get_harmony_encoding()
    if encoding is None or output_token_ids is None:
        return output_text or ""
    try:
        from openai_harmony import Role
        entries = encoding.parse_messages_from_completion_tokens(
            output_token_ids, Role.ASSISTANT
        )
    except Exception:
        return output_text or ""

    def _content_to_str(content):
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                p if isinstance(p, str)
                else (p.get("text") or p.get("content") or "")
                for p in content if isinstance(p, (str, dict))
            )
        return str(content)

    finals = []
    for entry in entries:
        d = entry.to_dict()
        role_val = d.get("role")
        role_str = role_val.value if hasattr(role_val, "value") else str(role_val)
        if role_str != "assistant":
            continue
        channel = d.get("channel")
        if channel not in ("final", "analysis"):
            continue
        val = _content_to_str(d.get("content"))
        if not val:
            continue
        if channel == "analysis":
            val = f"<think>{val}</think>"
        finals.append(val)

    if finals:
        return "\n".join(finals).strip()
    return output_text or ""


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
# Data loading
# ---------------------------------------------------------------------------

def load_forecast_questions(dataset_path: str, split: str) -> pd.DataFrame:
    """Load forecasting questions from parquet files."""
    root = Path(dataset_path)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    import glob
    parquet_files = sorted(glob.glob(str(root / f"{split}-*.parquet")))
    if not parquet_files:
        raise ValueError(f"No parquet files found for split '{split}' in {dataset_path}")

    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} questions from {split} split ({len(parquet_files)} file(s))")
    return df


# ---------------------------------------------------------------------------
# Answer / probability extraction
# ---------------------------------------------------------------------------

def extract_answer(model_output: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", model_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    lines = [l.strip() for l in model_output.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


def extract_probability(model_output: str) -> Optional[float]:
    match = re.search(r"<probability>(.*?)</probability>", model_output, re.DOTALL)
    if match:
        try:
            return float(match.group(1).strip())
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# With-answer prompt wrapper
# ---------------------------------------------------------------------------

def build_forecast_with_answer_prompt(base_prompt: str, answer: str, target_probability: float) -> str:
    """Prepend answer-aware instructions to an existing forecasting prompt.

    The model sees the correct answer and a target probability, and must
    reproduce both through its own reasoning — useful for generating
    distillation training data with diverse probability calibration.
    """
    preamble = (
        "You are given a forecasting question along with its correct (official) answer "
        "and a target probability. "
        "Your task is to first understand the answer, then arrive at the "
        "same answer (you may paraphrase) using your own reasoning and "
        "knowledge. You must NEVER mention or hint that the answer or "
        "probability was provided to you — respond as if you are answering "
        "the question from scratch.\n\n"
        f"OFFICIAL ANSWER:\n{answer.strip()}\n"
        f"TARGET PROBABILITY: {target_probability:.2f}\n\n"
        "Instructions:\n"
        "1. Study the question and the official answer carefully.\n"
        "2. Think hard about possible ways to arrive at the answer. Consider "
        "multiple angles: historical precedents, expert consensus, "
        "recent trends, geopolitical context, domain-specific knowledge, and "
        "any other relevant sources of information you know from your knowledge.\n"
        "3. Reason step by step as if you are solving the question on your own. "
        "Weigh evidence from different sources — where do they agree, where do "
        "they conflict, and what does the balance of evidence suggest?\n"
        "4. Arrive at the same answer through your own reasoning. Use forecasting "
        "skill: consider reference classes, track records from your parametric knowledge, and update on evidence.\n"
        "5. NEVER reveal that the answer or probability was given to you.\n"
        "6. Your final probability must be approximately {prob:.2f}. Build your "
        "reasoning so that this probability feels natural and justified — "
        "if it is low, emphasize genuine uncertainty, reason why the question may be hard to predict; "
        "if it is high, emphasize strong supporting reasoning why you believe so and why that might be the outcome in YOUR OPINION.\n\n"
        "Now here is the question:\n\n"
    ).format(prob=target_probability)
    return preamble + base_prompt


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_evaluation(args):
    use_harmony = _is_gpt_oss(args.model)
    if use_harmony:
        os.environ["VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS"] = "1"
        os.environ["VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8"] = "0"
        os.environ["VLLM_USE_FLASHINFER_MOE_MXFP4_BF16"] = "0"
        print("GPT-OSS model detected: setting FlashInfer CUTLASS env vars")

    from vllm import LLM, SamplingParams

    df = load_forecast_questions(args.dataset, args.split)

    prompt_col = "prompt_without_retrieval"
    if prompt_col not in df.columns:
        print(f"Warning: column '{prompt_col}' not found, falling back to 'prompt'")
        prompt_col = "prompt"
    print(f"Using prompt column: {prompt_col}")
    if args.with_answer:
        print("With-answer mode: ground truth answer will be injected into prompts")

    eval_model_short = os.path.basename(os.path.normpath(args.model))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    answer_tag = "_withanswer" if args.with_answer else ""
    if use_harmony:
        eval_dir_name = f"{eval_model_short}_{args.split}{answer_tag}_{args.effort}_{timestamp}"
    else:
        think_tag = "nothink" if args.no_think else "think"
        eval_dir_name = f"{eval_model_short}_{args.split}{answer_tag}_{think_tag}_{timestamp}"

    output_dir = Path(args.output_base) / "forecast_eval" / eval_dir_name
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"Skipping: output directory already exists with files: {output_dir}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    print(f"\nLoading vLLM model: {args.model}")
    print(f"Harmony format: {use_harmony}")
    if use_harmony:
        print(f"Reasoning effort: {args.effort}")
    else:
        print(f"Thinking enabled: {not args.no_think}")
    print(f"Tensor parallel size: {args.tp}")

    tokenizer = None
    if not use_harmony:
        print(f"Loading tokenizer from: {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        tensor_parallel_size=args.tp,
    )
    print("Model loaded successfully!")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=args.max_tokens,
    )

    if use_harmony:
        encoding = _get_harmony_encoding()
        if encoding is not None:
            stop_ids = encoding.stop_tokens_for_assistant_actions()
            sampling_params = SamplingParams(
                temperature=args.temperature,
                top_p=0.95,
                max_tokens=args.max_tokens,
                stop_token_ids=stop_ids,
            )
            print(f"  Harmony stop tokens configured ({len(stop_ids)} tokens)")

    all_messages = []
    all_indices = []
    all_original_prompts = []

    for idx, row in df.iterrows():
        original_prompt = row.get(prompt_col, "")
        if not original_prompt or not str(original_prompt).strip():
            continue
        original_prompt = str(original_prompt)
        prompt_text = original_prompt
        if args.with_answer:
            answer = str(row.get("answer", ""))
            if not answer.strip():
                continue
            target_prob = round(random.uniform(0.01, 0.99), 2)
            prompt_text = build_forecast_with_answer_prompt(original_prompt, answer, target_prob)
        all_messages.append([{"role": "user", "content": prompt_text}])
        all_indices.append(idx)
        all_original_prompts.append(original_prompt)

    if not all_messages:
        print("No valid prompts. Exiting.")
        return

    print(f"\nBuilt {len(all_messages)} prompts from {len(df)} questions")

    if use_harmony:
        from vllm.inputs.data import TokensPrompt

        token_prompts = []
        valid_map = []
        for i, msgs in enumerate(all_messages):
            token_ids = _build_harmony_token_ids(msgs, effort=args.effort)
            if token_ids is not None:
                token_prompts.append(TokensPrompt(prompt_token_ids=token_ids))
                valid_map.append(i)
            else:
                print(f"  Warning: Harmony encoding failed for prompt {i}, skipping")

        if not token_prompts:
            print("No valid prompts after Harmony encoding. Exiting.")
            return

        print(f"Running batch evaluation ({len(token_prompts)} prompts)...")
        outputs = llm.generate(token_prompts, sampling_params=sampling_params)

        all_outputs = [None] * len(all_messages)
        for out_idx, output in enumerate(outputs):
            orig_idx = valid_map[out_idx]
            raw_text = output.outputs[0].text
            tok_ids = list(output.outputs[0].token_ids) if hasattr(output.outputs[0], "token_ids") else None
            all_outputs[orig_idx] = _parse_harmony_output(raw_text, tok_ids)
    else:
        model_name_short = os.path.basename(os.path.normpath(args.model))
        enable_thinking = not args.no_think
        plain_prompts = []
        for msgs in all_messages:
            formatted = apply_chat_template(
                tokenizer, msgs[0]["content"], model_name_short,
                enable_thinking=enable_thinking,
            )
            plain_prompts.append(formatted)

        print(f"Running batch evaluation ({len(plain_prompts)} prompts)...")
        outputs = llm.generate(plain_prompts, sampling_params)

        all_outputs = [output.outputs[0].text for output in outputs]

    answer_tag_found = 0
    answer_tag_missing = 0
    prob_found = 0
    results = []

    for i, df_idx in enumerate(all_indices):
        row = df.iloc[df_idx] if isinstance(df_idx, int) else df.loc[df_idx]
        raw_output = all_outputs[i] or ""

        model_answer = extract_answer(raw_output)
        model_prob = extract_probability(raw_output)

        if "<answer>" in raw_output and "</answer>" in raw_output:
            answer_tag_found += 1
        else:
            answer_tag_missing += 1
        if model_prob is not None:
            prob_found += 1

        result = {
            "qid": str(row.get("qid", "")),
            "question_title": row.get("question_title", ""),
            "background": row.get("background", ""),
            "resolution_criteria": row.get("resolution_criteria", ""),
            "answer_type": row.get("answer_type", ""),
            "ground_truth": row.get("answer", ""),
            "original_prompt": all_original_prompts[i],
            "prompt": all_messages[i][0]["content"],
            "model_answer": model_answer,
            "model_probability": model_prob,
            "raw_model_output": raw_output,
            "metadata": {
                "eval_model": eval_model_short,
                "eval_model_path": args.model,
                "eval_effort": args.effort if use_harmony else None,
                "eval_temperature": args.temperature,
                "eval_max_tokens": args.max_tokens,
                "split": args.split,
                "with_answer": args.with_answer,
                "resolution_date": str(row.get("resolution_date", "")),
                "question_start_date": str(row.get("question_start_date", "")),
                "data_source": row.get("data_source", ""),
                "news_source": row.get("news_source", ""),
            }
        }
        results.append(result)

    output_path = output_dir / f"{args.split}.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(results)
    print(f"\nTotal: {total} results saved to {output_path}")
    print(f"Answer tag stats: {answer_tag_found} found, {answer_tag_missing} missing "
          f"({answer_tag_found / max(total, 1) * 100:.1f}% compliance)")
    print(f"Probability tag stats: {prob_found}/{total} "
          f"({prob_found / max(total, 1) * 100:.1f}%)")

    # Save config for reproducibility
    config = {
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "with_answer": args.with_answer,
        "prompt_column": prompt_col,
        "effort": args.effort if use_harmony else None,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "tp": args.tp,
        "total_questions": total,
        "answer_tag_found": answer_tag_found,
        "prob_found": prob_found,
        "timestamp": timestamp,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    del llm
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on OpenForesight forecasting questions."
    )
    parser.add_argument(
        "--model", type=str, default="/fast/nchandak/models/gpt-oss-20b",
        help="Path to vLLM model",
    )
    parser.add_argument(
        "--dataset", type=str, default="/fast/nchandak/datasets/OpenForesight",
        help="Path to OpenForesight parquet dataset directory",
    )
    parser.add_argument(
        "--split", type=str, default="validation",
        help="Dataset split to evaluate (default: validation)",
    )
    parser.add_argument(
        "--output_base", type=str,
        default="/fast/nchandak/forecast-sim/news/syntheticqa",
        help="Base output directory",
    )
    parser.add_argument(
        "--with_answer", action="store_true",
        help="Inject ground truth answer into prompt (for distillation data generation)",
    )
    parser.add_argument(
        "--effort", type=str, default="medium", choices=["low", "medium", "high"],
        help="Reasoning effort for gpt-oss Harmony format (default: medium)",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Sampling temperature (default: 0.6, or 1.0 for gpt-oss)",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=4096,
        help="Max tokens to generate per question (default: 4096)",
    )
    parser.add_argument(
        "--max_model_len", type=int, default=16384,
        help="Max model context length (default: 16384)",
    )
    parser.add_argument(
        "--gpu_mem", type=float, default=0.85,
        help="GPU memory utilization (default: 0.85)",
    )
    parser.add_argument(
        "--tp", type=int, default=None,
        help="Tensor parallel size (number of GPUs). Default: auto-detect.",
    )
    parser.add_argument(
        "--no_think", action="store_true",
        help="Disable thinking for non-gpt-oss models (e.g. Qwen3).",
    )
    args = parser.parse_args()

    if args.temperature is None:
        args.temperature = 1.0 if _is_gpt_oss(args.model) else 0.6
        print(f"Using default temperature: {args.temperature}")

    if args.tp is None:
        import subprocess
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            args.tp = len(result.stdout.strip().split("\n")) if result.stdout.strip() else 1
        except Exception:
            args.tp = 1
        print(f"Auto-detected {args.tp} GPUs")

    run_evaluation(args)


if __name__ == "__main__":
    main()
