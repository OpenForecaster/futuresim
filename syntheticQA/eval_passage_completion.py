#!/usr/bin/env python3
"""
Evaluate a model on news passage completion: given a partial news passage,
the model is asked to continue/complete it.

Reads articles from a single JSONL file (e.g. aljazeera.jsonl), truncates
each article's maintext at a configurable split point, provides article
metadata (title, description, authors, date, etc.) as context in the prompt,
and asks the model to complete the maintext. The ground-truth continuation
is saved alongside the model's output for later evaluation.

Supports both gpt-oss (Harmony format) and non-gpt-oss models (plain text).

Supports --checkpoints to evaluate all checkpoints from a training run directory.

Usage:
    python syntheticQA/eval_passage_completion.py --articles /path/to/aljazeera.jsonl
    python syntheticQA/eval_passage_completion.py --model /fast/nchandak/models/Qwen3-8B --articles /path/to/aljazeera.jsonl
    python syntheticQA/eval_passage_completion.py --model /fast/nchandak/models/gpt-oss-20b --articles /path/to/source.jsonl --effort medium
    python syntheticQA/eval_passage_completion.py --checkpoints /path/to/training_run/ --articles /path/to/source.jsonl
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from transformers import AutoTokenizer
from dateutil.parser import parse as parse_date

# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

_DATE_FIELDS = ("date_download", "date_modify", "date_publish")


def _ensure_aware(dt: datetime) -> datetime:
    """Make a datetime timezone-aware (assume UTC if naive)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def get_article_max_date(article: dict) -> Optional[datetime]:
    """Return the max of all available date fields in an article, or None."""
    dates = []
    for field in _DATE_FIELDS:
        val = article.get(field)
        if not val:
            continue
        try:
            parsed = parse_date(str(val))
            dates.append(_ensure_aware(parsed))
        except (ValueError, OverflowError):
            continue
    return max(dates) if dates else None


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
    "analysis" channel (wrapped in <think>...</think>) and the "final"
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
# Checkpoint discovery
# ---------------------------------------------------------------------------

def discover_checkpoints(checkpoints_dir: str) -> List[tuple]:
    """
    Discover all checkpoints in a training run directory.

    Expects a layout like:
        Qwen3-8B-1e-5-4096-4096/
            global_step_120/
                Qwen3-8B-1e-5-4096-4096-synthetic-sft-step120/   <-- model weights
                huggingface/                                       <-- tokenizer only
            global_step_180/
                ...

    Returns sorted list of (step_number, model_path) tuples.
    """
    root = Path(checkpoints_dir)
    if not root.is_dir():
        print(f"Error: Checkpoints directory not found: {checkpoints_dir}")
        return []

    run_name = root.name

    checkpoints = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not d.name.startswith("global_step_"):
            continue

        try:
            step = int(d.name.replace("global_step_", ""))
        except ValueError:
            continue

        model_path = None
        for sub in sorted(d.iterdir()):
            if sub.is_dir() and sub.name.startswith(run_name) and sub.name != "huggingface":
                model_path = str(sub)
                break

        if model_path is None:
            for sub in d.iterdir():
                if sub.is_dir() and sub.name != "huggingface":
                    has_safetensors = any(
                        f.suffix == ".safetensors" for f in sub.iterdir() if f.is_file()
                    )
                    if has_safetensors:
                        model_path = str(sub)
                        break

        if model_path is None:
            print(f"  Warning: No model folder found in {d.name}, skipping")
            continue

        checkpoints.append((step, model_path))

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


# ---------------------------------------------------------------------------
# Article loading
# ---------------------------------------------------------------------------

MIN_ARTICLE_LENGTH = 200  # skip articles with less text than this
MAX_ARTICLE_WORDS = 10000  # skip articles with more words than this


def load_articles(articles_path: str, num_article: int,
                  filter_dt: Optional[datetime] = None) -> List[dict]:
    """
    Load up to num_article articles from a single .jsonl file.
    Applies length and date filters. Returns a list of article dicts.
    """
    fpath = Path(articles_path)
    if not fpath.is_file():
        print(f"Error: articles file not found: {articles_path}")
        return []

    articles = []
    skipped_short = 0
    skipped_long = 0
    skipped_date = 0

    try:
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    article = json.loads(line)
                except json.JSONDecodeError:
                    continue

                maintext = article.get("maintext", "") or ""
                description = article.get("description", "") or ""

                # Length filters
                if len(maintext) + len(description) < MIN_ARTICLE_LENGTH:
                    skipped_short += 1
                    continue
                if len(maintext.split()) > MAX_ARTICLE_WORDS:
                    skipped_long += 1
                    continue

                # Date filter
                if filter_dt is not None:
                    art_max_date = get_article_max_date(article)
                    if art_max_date is not None and art_max_date >= filter_dt:
                        skipped_date += 1
                        continue

                articles.append(article)
                if len(articles) >= num_article:
                    break
    except (IOError, OSError) as e:
        print(f"Warning: could not read {fpath}: {e}", file=sys.stderr)

    print(f"  Loaded {len(articles)} articles from {fpath.name}")
    print(f"  Skipped: {skipped_short} too short, {skipped_long} too long, "
          f"{skipped_date} filtered by date")
    return articles


# ---------------------------------------------------------------------------
# Passage splitting
# ---------------------------------------------------------------------------

def split_passage(text: str, split_frac: float) -> tuple:
    """
    Split article text into (prefix, suffix) at approximately split_frac of
    the way through, breaking at a sentence boundary.

    Returns (prefix, suffix). If no good split is found, returns (text, "").
    """
    if not text or not text.strip():
        return (text, "")

    # Target character position
    target_pos = int(len(text) * split_frac)

    # Find sentence boundaries (., !, ?) followed by whitespace
    sentence_ends = [m.end() for m in re.finditer(r'[.!?]\s+', text)]

    if not sentence_ends:
        # No sentence boundaries found — try splitting at paragraph breaks
        para_ends = [m.end() for m in re.finditer(r'\n\s*\n', text)]
        if para_ends:
            # Pick the paragraph break closest to target
            best = min(para_ends, key=lambda p: abs(p - target_pos))
            return (text[:best].rstrip(), text[best:].lstrip())
        # Last resort: split at target position at a word boundary
        # Find the nearest space
        space_pos = text.rfind(' ', 0, target_pos)
        if space_pos > 0:
            return (text[:space_pos], text[space_pos + 1:])
        return (text[:target_pos], text[target_pos:])

    # Pick the sentence boundary closest to the target position
    best = min(sentence_ends, key=lambda p: abs(p - target_pos))
    return (text[:best].rstrip(), text[best:].lstrip())


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_completion_prompt(prefix: str, article: dict,
                            no_think: bool = False) -> str:
    """Build the prompt asking the model to complete a partial news passage.

    The prompt includes full article metadata (title, description, authors,
    date, source, etc.) for context, followed by the truncated maintext
    prefix that the model must continue.
    """
    parts = []
    parts.append(
        "You are given a news article with its metadata and the beginning of "
        "its body text. Your task is to continue the body text from where it "
        "left off. Write a plausible, coherent, and factually grounded "
        "continuation that matches the style, tone, and content of the "
        "passage so far."
    )

    # Include all available article metadata
    parts.append("\n=== ARTICLE METADATA ===")
    if article.get("title"):
        parts.append(f"Title: {article['title']}")
    if article.get("authors"):
        authors = article["authors"]
        if isinstance(authors, list):
            authors = ", ".join(str(a) for a in authors)
        parts.append(f"Authors: {authors}")
    if article.get("date_publish"):
        parts.append(f"Date Published: {article['date_publish']}")
    if article.get("source_domain"):
        parts.append(f"Source: {article['source_domain']}")
    if article.get("url"):
        parts.append(f"URL: {article['url']}")
    if article.get("language"):
        parts.append(f"Language: {article['language']}")
    if article.get("description"):
        parts.append(f"\nDescription: {article['description']}")
    parts.append("=== END METADATA ===")

    parts.append(f"\n=== ARTICLE TEXT (BEGINNING) ===\n{prefix.strip()}\n=== END OF PROVIDED TEXT ===")

    suffix = " /no_think" if no_think else ""
    parts.append(
        "\nContinue writing the article body from where it left off. "
        "Your continuation should:"
        "\n1. Seamlessly follow from the last sentence of the provided text."
        "\n2. Maintain the same writing style, tone, and level of detail."
        "\n3. Be factually plausible — use real-world knowledge and the "
        "metadata above to write a realistic continuation."
        "\n4. Be a reasonable length — roughly match the length of the "
        "provided passage or shorter."
        "\n5. Do NOT repeat or summarize the provided text — only write new "
        "content that continues the article."
        f"\n\nJUST COMPLETE THE ARTICLE (DO NOT OUTPUT ANYTHING ELSE, NO PREAMBLE OR EXPLANATION).{suffix}"
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def _run_single_model(
    model_path: str,
    output_dir: Path,
    articles: List[dict],
    source_name: str,
    args,
    use_harmony: bool,
):
    """Run passage completion evaluation for a single model."""
    from vllm import LLM, SamplingParams

    eval_model_short = os.path.basename(os.path.normpath(model_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Init model
    print(f"\nLoading vLLM model: {model_path}")
    print(f"Harmony format: {use_harmony}")
    if use_harmony:
        print(f"Reasoning effort: {args.effort}")
    if not use_harmony:
        print(f"Thinking enabled: {not args.no_think}")
    print(f"Tensor parallel size: {args.tp}")
    print(f"Split fraction: {args.split_frac}")

    # Load tokenizer for non-gpt-oss models (needed for chat template)
    tokenizer = None
    if not use_harmony:
        print(f"Loading tokenizer from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    llm = LLM(
        model=model_path,
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

    # Build ALL prompts
    all_messages = []
    all_records = []  # (article_index, article, prefix, suffix)

    for idx, article in enumerate(articles):
        maintext = article.get("maintext", "") or ""
        if not maintext.strip():
            continue

        prefix, suffix = split_passage(maintext, args.split_frac)

        # Skip if either part is too short to be meaningful
        if len(prefix.split()) < 30 or len(suffix.split()) < 20:
            continue

        prompt_text = build_completion_prompt(
            prefix, article, no_think=args.no_think,
        )
        all_messages.append([{"role": "user", "content": prompt_text}])
        all_records.append((idx, article, prefix, suffix))

    if not all_messages:
        print("No valid articles to evaluate. Exiting.")
        return

    print(f"\nBuilt {len(all_messages)} completion prompts from {source_name}")

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

        print(f"Running batch completion ({len(token_prompts)} prompts)...")
        outputs = llm.generate(token_prompts, sampling_params=sampling_params)

        all_outputs = [None] * len(all_messages)
        for out_idx, output in enumerate(outputs):
            orig_idx = valid_indices[out_idx]
            raw_text = output.outputs[0].text
            tok_ids = list(output.outputs[0].token_ids) if hasattr(output.outputs[0], "token_ids") else None
            all_outputs[orig_idx] = _parse_harmony_output(raw_text, tok_ids)
    else:
        # Apply chat template for non-gpt-oss models
        model_name_short = os.path.basename(os.path.normpath(model_path))
        enable_thinking = not args.no_think
        plain_prompts = []
        for msgs in all_messages:
            raw_prompt = msgs[0]["content"]
            formatted = apply_chat_template(
                tokenizer, raw_prompt, model_name_short,
                enable_thinking=enable_thinking,
            )
            plain_prompts.append(formatted)

        print(f"Running batch completion ({len(plain_prompts)} prompts)...")
        outputs = llm.generate(plain_prompts, sampling_params)

        all_outputs = []
        for output in outputs:
            text = output.outputs[0].text
            all_outputs.append(text)

    # Collect results
    all_results = []

    for i, (art_idx, article, prefix, suffix) in enumerate(all_records):
        raw_output = all_outputs[i]
        if raw_output is None:
            raw_output = ""

        result = {
            "prefix": prefix,
            "ground_truth_continuation": suffix,
            "model_continuation": raw_output,
            "prompt": all_messages[i][0]["content"],
            "metadata": {
                "eval_type": "passage_completion",
                "eval_model": eval_model_short,
                "eval_model_path": model_path,
                "eval_effort": args.effort if use_harmony else None,
                "eval_temperature": args.temperature,
                "eval_max_tokens": args.max_tokens,
                "split_frac": args.split_frac,
                "thinking_enabled": not args.no_think if not use_harmony else None,
                "article_title": article.get("title", ""),
                "article_url": article.get("url", ""),
                "source_domain": article.get("source_domain", ""),
                "date_publish": article.get("date_publish", ""),
                "jsonl_source": source_name,
                "article_index": art_idx,
                "prefix_words": len(prefix.split()),
                "suffix_words": len(suffix.split()),
                "completion_words": len(raw_output.split()),
            }
        }
        all_results.append(result)

    # Save
    output_path = output_dir / source_name
    with open(output_path, "w", encoding="utf-8") as out_f:
        for result in all_results:
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"  {source_name}: {len(all_results)} completion results saved")

    # Compute and display summary stats
    if all_results:
        all_prefix_words = [r["metadata"]["prefix_words"] for r in all_results]
        all_suffix_words = [r["metadata"]["suffix_words"] for r in all_results]
        all_completion_words = [r["metadata"]["completion_words"] for r in all_results]
        avg_prefix = sum(all_prefix_words) / len(all_prefix_words)
        avg_suffix = sum(all_suffix_words) / len(all_suffix_words)
        avg_completion = sum(all_completion_words) / len(all_completion_words)
        print(f"\nAvg prefix length: {avg_prefix:.0f} words")
        print(f"Avg ground truth continuation: {avg_suffix:.0f} words")
        print(f"Avg model continuation: {avg_completion:.0f} words")

    print(f"\nTotal: {len(all_results)} completion results")
    print(f"Output at: {output_dir}")

    # Explicitly delete LLM to free GPU memory before loading next checkpoint
    del llm
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def run_evaluation(args):
    # Determine model path(s)
    if args.checkpoints:
        checkpoints = discover_checkpoints(args.checkpoints)
        if not checkpoints:
            print(f"No checkpoints found in: {args.checkpoints}")
            return

        print(f"\nFound {len(checkpoints)} checkpoints:")
        for step, path in checkpoints:
            print(f"  step {step}: {path}")

        first_model_path = checkpoints[0][1]
        use_harmony = _is_gpt_oss(first_model_path)
    else:
        checkpoints = None
        use_harmony = _is_gpt_oss(args.model)

    # Set env vars for gpt-oss BEFORE importing vllm
    if use_harmony:
        os.environ["VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS"] = "1"
        os.environ["VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8"] = "0"
        os.environ["VLLM_USE_FLASHINFER_MOE_MXFP4_BF16"] = "0"
        print("GPT-OSS model detected: setting FlashInfer CUTLASS env vars")

    # Parse filter date
    filter_dt = _ensure_aware(parse_date(args.filter_date))

    # Load articles from single JSONL file
    print(f"\nLoading articles from: {args.articles}")
    print(f"Date filter: excluding articles with max date >= {args.filter_date}")
    articles = load_articles(
        args.articles, args.num_article, filter_dt=filter_dt,
    )
    if not articles:
        print("No articles found. Exiting.")
        return

    # Derive source name from the JSONL filename (e.g. "aljazeera.jsonl")
    source_name = Path(args.articles).name
    print(f"Loaded {len(articles)} articles from {source_name}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if checkpoints:
        # --- Checkpoint mode ---
        run_name = os.path.basename(os.path.normpath(args.checkpoints))
        think_suffix = "_no_think" if (not use_harmony and args.no_think) else "_thinking"
        if use_harmony:
            base_eval_dir_name = f"{run_name}_completion_{args.effort}_{timestamp}"
        else:
            base_eval_dir_name = f"{run_name}_completion{think_suffix}_{timestamp}"

        base_output_dir = Path(args.output_base) / base_eval_dir_name

        print(f"\n{'='*60}")
        print(f"Checkpoint evaluation: {len(checkpoints)} checkpoints")
        print(f"Base output: {base_output_dir}")
        print(f"{'='*60}")

        for ckpt_idx, (step, model_path) in enumerate(checkpoints):
            print(f"\n{'='*60}")
            print(f"Checkpoint {ckpt_idx + 1}/{len(checkpoints)}: step {step}")
            print(f"Model path: {model_path}")
            print(f"{'='*60}")

            ckpt_output_dir = base_output_dir / f"step_{step}"

            _run_single_model(
                model_path=model_path,
                output_dir=ckpt_output_dir,
                articles=articles,
                source_name=source_name,
                args=args,
                use_harmony=use_harmony,
            )

        print(f"\n{'='*60}")
        print(f"All {len(checkpoints)} checkpoints evaluated!")
        print(f"Results at: {base_output_dir}")
        print(f"{'='*60}")
    else:
        # --- Single model mode ---
        eval_model_short = os.path.basename(os.path.normpath(args.model))

        think_suffix = "_no_think" if (not use_harmony and args.no_think) else "_thinking"
        eval_dir_name = f"{eval_model_short}_completion{think_suffix}_{timestamp}"
        if use_harmony:
            eval_dir_name = f"{eval_model_short}_completion_{args.effort}_{timestamp}"

        output_dir = Path(args.output_base) / eval_dir_name

        _run_single_model(
            model_path=args.model,
            output_dir=output_dir,
            articles=articles,
            source_name=source_name,
            args=args,
            use_harmony=use_harmony,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on news passage completion using vLLM "
                    "batch generation. Given a partial news article, the "
                    "model continues writing the article."
    )
    parser.add_argument(
        "--model", type=str, default="/fast/nchandak/models/Qwen3-8B",
        help="Path to vLLM model (default: /fast/nchandak/models/Qwen3-8B). "
             "Ignored when --checkpoints is used.",
    )
    parser.add_argument(
        "--checkpoints", type=str, default=None,
        help="Path to a training run directory containing global_step_* "
             "checkpoint subdirectories. All checkpoints will be evaluated "
             "sequentially. Overrides --model.",
    )
    parser.add_argument(
        "--articles", type=str, required=True,
        help="Path to a single .jsonl file containing articles "
             "(e.g. /path/to/passagecompletion/aljazeera.jsonl)",
    )
    parser.add_argument(
        "--num_article", type=int, default=10,
        help="Max articles to take from each JSONL file (default: 10)",
    )
    parser.add_argument(
        "--split_frac", type=float, default=0.3,
        help="Fraction of the article to give as prefix (default: 0.5). "
             "E.g., 0.5 = first half is the prompt, second half is ground truth.",
    )
    parser.add_argument(
        "--output_base", type=str,
        default="/fast/nchandak/forecast-sim/news/passage_completion",
        help="Base output directory (default: /fast/nchandak/forecast-sim/news/passage_completion)",
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
        help="Max tokens to generate per article (default: 4096)",
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
             "Appends /no_think to prompt and disables thinking in chat template.",
    )
    parser.add_argument(
        "--filter_date", type=str, default="2025-04-01",
        help="Remove articles whose max date is >= this date. "
             "Format: YYYY-MM-DD (default: 2025-04-01)",
    )
    args = parser.parse_args()

    # Determine which model path to use for defaults
    if args.checkpoints:
        ckpts = discover_checkpoints(args.checkpoints)
        ref_model = ckpts[0][1] if ckpts else args.model
    else:
        ref_model = args.model

    # Set default temperature based on model type if not explicitly provided
    if args.temperature is None:
        if _is_gpt_oss(ref_model):
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
