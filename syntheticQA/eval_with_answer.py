#!/usr/bin/env python3
"""
Answer-aware evaluation: give the model the golden answer and ask it to
internalise the answer, then reproduce it in its own words through its own
reasoning — without ever revealing that the answer was provided.

The model must output its final (paraphrased) answer inside <answer>...</answer>
tags.

Supports:
  - gpt-oss models (Harmony format)
  - non-gpt-oss models (chat template, e.g. Qwen3)
  - --no_think flag to disable thinking for non-gpt-oss models (e.g. Qwen3)

Usage:
    python syntheticQA/eval_with_answer.py --questions /path/to/syntheticqa/run_folder/
    python syntheticQA/eval_with_answer.py --model /fast/nchandak/models/Qwen3-8B --questions /path/ --no_think
    python syntheticQA/eval_with_answer.py --model /fast/nchandak/models/gpt-oss-20b --questions /path/ --effort medium
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Harmony helpers (shared with eval_qa.py / generate_qa.py)
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
    """Render OpenAI-style messages to Harmony prompt token IDs."""
    encoding = _get_harmony_encoding()
    if encoding is None:
        return None
    try:
        from openai_harmony import (
            Conversation,
            Message,
            ReasoningEffort,
            Role,
            SystemContent,
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

        role_map = {
            "system": Role.SYSTEM,
            "user": Role.USER,
            "assistant": Role.ASSISTANT,
        }

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
    """Extract final assistant text from Harmony completion.

    Mirrors parse_final_text() from eval_gptoss.py: collects both the
    "analysis" channel (wrapped in <think>…</think>) and the "final"
    channel, concatenated in order.
    """
    encoding = _get_harmony_encoding()
    if encoding is None or output_token_ids is None:
        return output_text or ""
    try:
        from openai_harmony import HarmonyError, Role
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
            buf = []
            for part in content:
                if isinstance(part, str):
                    buf.append(part)
                elif isinstance(part, dict):
                    if isinstance(part.get("text"), str):
                        buf.append(part["text"])
                    elif isinstance(part.get("content"), str):
                        buf.append(part["content"])
            return "".join(buf)
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


# ---------------------------------------------------------------------------
# Chat template helper (for non-gpt-oss models)
# ---------------------------------------------------------------------------

def apply_chat_template(tokenizer, prompt: str, model_name: str = "",
                        enable_thinking: bool = True) -> str:
    """Apply the tokenizer's chat template with model-specific handling."""
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
# QA loading
# ---------------------------------------------------------------------------

def load_qa_files(questions_path: str) -> Dict[str, List[dict]]:
    """
    Load QA JSONL files from a syntheticQA output folder.
    Returns {source_filename: [qa_record, ...]}.
    """
    root = Path(questions_path)
    results = {}
    jsonl_files = sorted(f for f in root.iterdir() if f.is_file() and f.suffix == ".jsonl")

    for fpath in jsonl_files:
        records = []
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        records.append(record)
                    except json.JSONDecodeError:
                        continue
        except (IOError, OSError) as e:
            print(f"Warning: could not read {fpath}: {e}", file=sys.stderr)
            continue

        if records:
            results[fpath.name] = records
            print(f"  Loaded {len(records)} QA pairs from {fpath.name}")

    return results


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_eval_with_answer_prompt(question: str, background: str, answer: str) -> str:
    """Build a prompt that gives the model the golden answer and asks it to
    internalise and reproduce it through its own reasoning."""

    parts = []
    parts.append(
        "You are given a factual question along with its correct (official) answer. "
        "Your task is to first understand the answer, then arrive at the "
        "same answer (you may paraphrase) using your own reasoning and "
        "knowledge. You must NEVER mention or hint that the answer was "
        "provided to you — respond as if you are answering the question "
        "from scratch."
    )

    parts.append(f"\nQuestion:\n{question.strip()}")

    if background and background.strip():
        parts.append(f"\nBackground:\n{background.strip()}")

    parts.append(f"\OFFICIAL ANSWER:\n{answer.strip()}")

    parts.append(
        "\nInstructions:"
        "\n1. Study the question and the correct (official) answer carefully."
        "\n2. Reason step by step as if you are solving the question on your own."
        "\n3. Arrive at the same answer (paraphrasing is fine) through your own reasoning."
        "\n4. NEVER reveal that the answer was given to you. Write as if you figured it out yourself (by deriving it on your own or relying on your own knowledge possibly including past public records and/or news sources)."
        "\n5. Put your final answer inside <answer>...</answer> tags."
        "\n\nFor example: <answer>The answer is X.</answer>"
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(model_output: str) -> str:
    """Extract the content inside <answer>...</answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", model_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: return the last non-empty line
    lines = [l.strip() for l in model_output.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def run_evaluation(args):
    # Set env vars for gpt-oss BEFORE importing vllm
    use_harmony = _is_gpt_oss(args.model)
    if use_harmony:
        os.environ["VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS"] = "1"
        os.environ["VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8"] = "0"
        os.environ["VLLM_USE_FLASHINFER_MOE_MXFP4_BF16"] = "0"
        print("GPT-OSS model detected: setting FlashInfer CUTLASS env vars")

    from vllm import LLM, SamplingParams

    # Load QA files
    print(f"\nLoading QA pairs from: {args.questions}")
    source_qa = load_qa_files(args.questions)
    if not source_qa:
        print("No QA files found. Exiting.")
        return

    total_qa = sum(len(v) for v in source_qa.values())
    print(f"Loaded {total_qa} QA pairs from {len(source_qa)} sources")

    # Create output directory
    questions_folder_name = Path(args.questions).name
    eval_model_short = os.path.basename(os.path.normpath(args.model))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build descriptive dir name
    think_suffix = ""
    if not use_harmony and args.no_think:
        think_suffix = "_nothink"
    eval_dir_name = f"{eval_model_short}_with_answer{think_suffix}_{timestamp}"
    if use_harmony:
        eval_dir_name = f"{eval_model_short}_with_answer_{args.effort}_{timestamp}"

    output_dir = Path(args.output_base) / questions_folder_name / eval_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Init model
    print(f"\nLoading vLLM model: {args.model}")
    print(f"Harmony format: {use_harmony}")
    if use_harmony:
        print(f"Reasoning effort: {args.effort}")
    if not use_harmony:
        print(f"Thinking enabled: {not args.no_think}")
    print(f"Tensor parallel size: {args.tp}")

    # Load tokenizer for non-gpt-oss models (needed for chat template)
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

    # If using harmony, configure stop tokens
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

    # Build ALL prompts across all sources
    all_messages = []
    all_records = []  # original QA records, parallel to all_messages

    for source_name, records in source_qa.items():
        for rec in records:
            question = rec.get("question", "")
            background = rec.get("background", "")
            answer = rec.get("answer", "")
            if not question.strip() or not answer.strip():
                continue

            prompt_text = build_eval_with_answer_prompt(question, background, answer)
            all_messages.append([{"role": "user", "content": prompt_text}])
            all_records.append((source_name, rec))

    if not all_messages:
        print("No valid QA pairs to evaluate. Exiting.")
        return

    print(f"\nBuilt {len(all_messages)} eval prompts across {len(source_qa)} sources")

    # Single batch generation
    if use_harmony:
        from vllm.inputs.data import TokensPrompt

        token_prompts = []
        valid_indices = []
        for i, msgs in enumerate(all_messages):
            token_ids = _build_harmony_token_ids(msgs, effort=args.effort)
            if token_ids is not None:
                token_prompts.append(TokensPrompt(prompt_token_ids=token_ids))
                valid_indices.append(i)
            else:
                print(f"  Warning: Harmony encoding failed for prompt {i}, skipping")

        if not token_prompts:
            print("No valid prompts after Harmony encoding. Exiting.")
            return

        print(f"Running batch evaluation ({len(token_prompts)} prompts)...")
        outputs = llm.generate(token_prompts, sampling_params=sampling_params)

        all_outputs = [None] * len(all_messages)
        for out_idx, output in enumerate(outputs):
            orig_idx = valid_indices[out_idx]
            raw_text = output.outputs[0].text
            tok_ids = list(output.outputs[0].token_ids) if hasattr(output.outputs[0], "token_ids") else None
            all_outputs[orig_idx] = _parse_harmony_output(raw_text, tok_ids)
    else:
        # Apply chat template for non-gpt-oss models
        model_name_short = os.path.basename(os.path.normpath(args.model))
        enable_thinking = not args.no_think
        plain_prompts = []
        for msgs in all_messages:
            raw_prompt = msgs[0]["content"]
            formatted = apply_chat_template(
                tokenizer, raw_prompt, model_name_short,
                enable_thinking=enable_thinking,
            )
            plain_prompts.append(formatted)

        print(f"Running batch evaluation ({len(plain_prompts)} prompts)...")
        outputs = llm.generate(plain_prompts, sampling_params)

        all_outputs = []
        for output in outputs:
            text = output.outputs[0].text
            # Strip <think>...</think> reasoning if present
            # if "</think>" in text:
            #     text = text.split("</think>", 1)[1]
            all_outputs.append(text)

    # Parse and save per-source
    source_results = defaultdict(list)
    answer_tag_found = 0
    answer_tag_missing = 0

    for i, (source_name, orig_rec) in enumerate(all_records):
        raw_output = all_outputs[i]
        if raw_output is None:
            raw_output = ""

        model_answer = extract_answer(raw_output)

        # Track answer tag stats
        if "<answer>" in raw_output and "</answer>" in raw_output:
            answer_tag_found += 1
        else:
            answer_tag_missing += 1

        orig_meta = orig_rec.get("metadata", {})

        result = {
            "question": orig_rec.get("question", ""),
            "background": orig_rec.get("background", ""),
            "ground_truth": orig_rec.get("answer", ""),
            "model_answer": model_answer,
            "raw_model_output": raw_output,
            "prompt": all_messages[i][0]["content"],
            "metadata": {
                "eval_type": "with_answer",
                "eval_model": eval_model_short,
                "eval_effort": args.effort if use_harmony else None,
                "eval_temperature": args.temperature,
                "eval_max_tokens": args.max_tokens,
                "thinking_enabled": not args.no_think if not use_harmony else None,
                "gen_model": orig_meta.get("model", ""),
                "gen_effort": orig_meta.get("effort", ""),
                "article_title": orig_meta.get("article_title", ""),
                "article_url": orig_meta.get("article_url", ""),
                "source_domain": orig_meta.get("source_domain", ""),
                "date_publish": orig_meta.get("date_publish", ""),
                "jsonl_source": orig_meta.get("jsonl_source", ""),
                "article_index": orig_meta.get("article_index", ""),
                "questions_folder": questions_folder_name,
            }
        }
        source_results[source_name].append(result)

    # Save
    total_saved = 0
    for source_name in sorted(source_results.keys()):
        output_path = output_dir / source_name
        with open(output_path, "w", encoding="utf-8") as out_f:
            for result in source_results[source_name]:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
        count = len(source_results[source_name])
        total_saved += count
        print(f"  {source_name}: {count} eval results saved")

    print(f"\nTotal: {total_saved} eval results across {len(source_results)} sources")
    print(f"Answer tag stats: {answer_tag_found} found, {answer_tag_missing} missing "
          f"({answer_tag_found / max(total_saved, 1) * 100:.1f}% compliance)")
    print(f"Output at: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Answer-aware eval: give the model the golden answer and ask "
                    "it to reproduce it through its own reasoning."
    )
    parser.add_argument(
        "--model", type=str, default="/fast/nchandak/models/Qwen3-8B",
        help="Path to vLLM model (default: /fast/nchandak/models/Qwen3-8B)",
    )
    parser.add_argument(
        "--questions", type=str, required=True,
        help="Path to syntheticQA output folder containing .jsonl files",
    )
    parser.add_argument(
        "--output_base", type=str,
        default="/fast/nchandak/forecast-sim/news/syntheticqa",
        help="Base output directory (default: /fast/nchandak/forecast-sim/news/syntheticqa)",
    )
    parser.add_argument(
        "--effort", type=str, default="medium", choices=["low", "medium", "high"],
        help="Reasoning effort for gpt-oss Harmony format (default: medium)",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Sampling temperature (default: 0.6, or 1.0 for gpt-oss models)",
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
        help="Disable thinking/reasoning for non-gpt-oss models (e.g. Qwen3). "
             "Has no effect on gpt-oss models.",
    )
    args = parser.parse_args()

    # Set default temperature based on model type if not explicitly provided
    if args.temperature is None:
        if _is_gpt_oss(args.model):
            args.temperature = 1.0
        else:
            args.temperature = 0.6
        print(f"Using default temperature: {args.temperature}")

    # Auto-detect GPU count if --tp not specified
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
