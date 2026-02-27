#!/usr/bin/env python3
"""
Generate synthetic Q&A pairs from news articles using vLLM batch generation.

For each JSONL source in the articles folder, reads up to --num_article articles,
sends the full article to the model, and asks it to generate --num_q self-contained
question/answer pairs. Uses Harmony format for gpt-oss models.

Output: {articles_path}/syntheticqa/{timestamp}/{source}.jsonl
Each output line is a JSON object with "question", "answer", and "metadata".

Usage:
    python syntheticQA/generate_qa.py --articles /fast/sgoel/forecasting/news/articles2025
    python syntheticQA/generate_qa.py --articles /path/to/articles --num_q 5 --num_article 3
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
# Harmony helpers (adapted from inference/vllm.py)
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

def apply_chat_template(tokenizer, prompt: str, model_name: str = "") -> str:
    """Apply the tokenizer's chat template with model-specific handling."""
    try:
        chat = [{"role": "user", "content": prompt}]
        if "qwen3" in model_name.lower():
            return tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            )
        else:
            return tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True,
            )
    except Exception as e:
        print(f"  Warning: chat template failed ({e}), using raw prompt")
        return prompt


# ---------------------------------------------------------------------------
# Article loading
# ---------------------------------------------------------------------------

def load_articles(articles_path: str, num_article: int) -> Dict[str, List[dict]]:
    """
    Load up to num_article articles from each top-level .jsonl file.
    Returns {source_filename: [article_dict, ...]}.
    """
    root = Path(articles_path)
    results = {}
    jsonl_files = sorted(f for f in root.iterdir() if f.is_file() and f.suffix == ".jsonl")

    for fpath in jsonl_files:
        articles = []
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        article = json.loads(line)
                        articles.append(article)
                    except json.JSONDecodeError:
                        continue
                    if len(articles) >= num_article:
                        break
        except (IOError, OSError) as e:
            print(f"Warning: could not read {fpath}: {e}", file=sys.stderr)
            continue

        if articles:
            results[fpath.name] = articles
            print(f"  Loaded {len(articles)} articles from {fpath.name}")

    return results


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def format_article_for_prompt(article: dict) -> str:
    """Format an article dict into a readable text block for the LLM prompt."""
    parts = []
    if article.get("title"):
        parts.append(f"Title: {article['title']}")
    if article.get("authors"):
        authors = article["authors"]
        if isinstance(authors, list):
            authors = ", ".join(authors)
        parts.append(f"Authors: {authors}")
    if article.get("date_publish"):
        parts.append(f"Date Published: {article['date_publish']}")
    if article.get("date_download"):
        parts.append(f"Date Downloaded: {article['date_download']}")
    if article.get("source_domain"):
        parts.append(f"Source: {article['source_domain']}")
    if article.get("url"):
        parts.append(f"URL: {article['url']}")
    if article.get("language"):
        parts.append(f"Language: {article['language']}")
    if article.get("description"):
        parts.append(f"\nDescription: {article['description']}")
    if article.get("maintext"):
        parts.append(f"\nFull Article Text:\n{article['maintext']}")
    return "\n".join(parts)


MIN_ARTICLE_LENGTH = 200  # skip articles with less text than this


def build_prompt(article: dict, num_q: int) -> str:
    """Build the full prompt for Q&A generation from a single article."""
    article_text = format_article_for_prompt(article)

    prompt = f"""Below is a news article with its metadata. Read it carefully, then generate exactly {num_q} question-answer pairs based on the information in this article.

=== ARTICLE START ===
{article_text}
=== ARTICLE END ===

Generate exactly {num_q} factual question-answer pairs. These are knowledge-awareness questions — they test whether someone knows specific real-world facts. Follow these rules:

1. FULLY SPECIFIED: Each question must contain ALL context needed to answer it — full names of people, organizations, locations, exact dates, and numbers. A knowledgeable person should be able to answer the question without needing any additional context. The answer should be short (1-2 sentences max). EMBED ALL THE NECESSARY CONTEXT DIRECTLY IN THE QUESTION. PROVIDE BACKGROUND INFORMATION AS REQUIRED.

2. INDEPENDENT: Each question MUST BE SELF-CONTAINED AND SHOULD BE ANSWERABLE INDEPENDENTLY. Do not reference other questions like "In the same article..." or "Related to the above..." or "According to the same article...". MAKE SURE EACH QUESTION IS SELF-CONTAINED AND THERE IS NO CROSS REFERENCE BETWEEN QUESTIONS.

3. FACTUAL: Focus on verifiable real-world facts, events, people, numbers, and consequences. Do NOT ask about the article's metadata (author, publication date, URL). Focus on the substantive news content and most important information provided by the article.

4. REFERENCE THE SOURCE IF NECESSARY: The question should not be vague or ambiguous. If necessary, reference the article/news source (like "According to the article on ABC topic published by XYZ on PQR (date), ..." and provide the minimum required background information to answer the question. A question should not reference other questions or answers. It should NOT BE LIKE "In the same article..." or "Related to the above..." or "According to the same article...", "As per the article...", etc. 

5. SHORT ANSWER: The questions should be designed such that the answer is the single unique answer to the question. The answer should be concise (1-2 sentences max).

6. SPECIFIC: Be precise and unambiguous. Include enough detail that the question has exactly one correct answer.

7. DIVERSE: Cover different aspects of the news — key events, people involved, causes, consequences, statistics, broader context. NO QUESTION SHOULD BE THE SAME.

8. BACKGROUND: For each question, provide a short "background" paragraph (2-4 sentences) that gives relevant context a reader would need to understand the question — e.g. who the key actors are, what the broader situation is, what led up to the event. The background MUST NOT reveal or hint at the answer. It should provide any relevant information required to understand the question without giving it away.

9. FORMAT: Output ONLY a valid JSON array. No other text before or after. Each element must have exactly "question", "answer", and "background" keys.

=== GOOD EXAMPLES ===

Example 1 (specific, fully specified, good background):
{{"question": "What was the total amount of funding that the European Commission approved for Ukraine's energy infrastructure reconstruction in the aid package announced on March 12, 2025?", "answer": "The European Commission approved €2.1 billion in funding for Ukraine's energy infrastructure reconstruction.", "background": "Following extensive damage to Ukraine's power grid during the winter of 2024-2025, the European Commission held an emergency summit to discuss reconstruction efforts. Multiple EU member states had been calling for a comprehensive aid package to help Ukraine rebuild critical infrastructure before the next winter season."}}

Example 2 (factual, diverse, no answer leakage in background):
{{"question": "Who did the U.S. Federal Reserve appoint as the new Vice Chair for Supervision in January 2025, replacing Michael Barr?", "answer": "The U.S. Federal Reserve appointed Michelle Bowman as the new Vice Chair for Supervision.", "background": "The position of Vice Chair for Supervision at the Federal Reserve is responsible for overseeing the regulation of the U.S. banking system. Michael Barr announced his resignation from the role in late 2024 amid political pressure, creating a vacancy that the Fed needed to fill to maintain continuity in bank oversight."}}

Example 3 (references source when needed, self-contained):
{{"question": "According to a Reuters investigation published in February 2025, how many undisclosed meetings did senior executives of TechCorp hold with Chinese government officials between 2023 and 2024?", "answer": "According to the Reuters investigation, senior TechCorp executives held at least 14 undisclosed meetings with Chinese government officials between 2023 and 2024.", "background": "TechCorp, a major U.S. semiconductor manufacturer, had publicly stated that it was complying with all U.S. export control regulations regarding China. Reuters conducted a months-long investigation involving leaked documents and insider interviews to examine the company's actual dealings with Chinese authorities."}}

=== BAD EXAMPLES (DO NOT generate questions like these) ===

Bad Example 1 — vague, not self-contained:
{{"question": "What did the government announce?", "answer": "A new policy on immigration.", "background": "The government made an announcement recently."}}
WHY BAD: No specifics — which government? When? What policy exactly? Background is equally vague.

Bad Example 2 — references other questions:
{{"question": "In the same article mentioned above, what was the reaction to the policy?", "answer": "Opposition leaders criticized it.", "background": "The policy was controversial."}}
WHY BAD: References "the same article mentioned above" — each question must be fully independent.

Bad Example 3 — background leaks the answer:
{{"question": "How many people were displaced by the flooding in Bangladesh in March 2025?", "answer": "Approximately 1.2 million people were displaced.", "background": "Severe monsoon flooding hit Bangladesh in March 2025, displacing over a million residents and causing widespread damage to agricultural land in the Sylhet region."}}
WHY BAD: The background says "displacing over a million residents" which essentially gives away the answer. Background should set the scene without hinting at the specific answer.

Bad Example 4 — asks about article metadata:
{{"question": "Who wrote the article about the earthquake in Turkey?", "answer": "The article was written by John Smith for Al Jazeera.", "background": "Al Jazeera published an article about the earthquake."}}
WHY BAD: Asks about the article's author/metadata rather than the substantive news content.

Bad Example 5 — references "the article" without specifying which one:
{{"question": "According to the European Economic and Social Committee cited in the article, how many euros per year did Bulgaria lose before 1 January 2025 because it was not a full Schengen member?", "answer": "Bulgaria lost 834 million euros per year.", "background": "Bulgaria had been seeking full membership in the Schengen Area for years. The European Economic and Social Committee published estimates of the economic costs of remaining outside the zone, including delays at border crossings and reduced trade efficiency."}}
WHY BAD: Says "cited in the article" without specifying WHICH article — no publication name, no date, no topic identifier. A reader has no way to know which article is being referenced. Should instead say something like "According to the European Economic and Social Committee, as reported by DW News in a January 2025 article on Bulgaria's Schengen accession, ..."

=== END EXAMPLES ===

Output format:
```json
[
  {{"question": "...", "answer": "...", "background": "..."}},
  {{"question": "...", "answer": "...", "background": "..."}}
]
```

Generate exactly {num_q} pairs. EACH GENERATED QUESTION ANSWER PAIR SHOULD FOLLOW *ALL* THE ABOVE RULES. After generating, check the rules and make sure they are satisfied. Rewrite or come up with new question-answer pairs if necessary."""

    return prompt


# ---------------------------------------------------------------------------
# Verifier prompt
# ---------------------------------------------------------------------------

def build_verifier_prompt(question: str, answer: str, background: str) -> str:
    """Build a prompt that asks the model to reason about and verify a Q&A pair."""
    return f"""You are a strict quality-control judge for factual question-answer pairs generated from news articles. You must evaluate whether the following question-answer-background triple satisfies ALL of the rules (quality-checks)listed below.

=== QUESTION ===
{question}

=== BACKGROUND ===
{background}

=== ANSWER ===
{answer}


=== RULES TO CHECK ===
1. FULLY SPECIFIED: The question and the background contain ALL context needed to answer it — full names, organizations, locations, dates, numbers. A knowledgeable person can answer without any additional context.
2. INDEPENDENT & SELF-CONTAINED: The question does NOT reference "the article", "the above", "the same report", or any external context. It stands completely on its own. It should NOT say things like "According to the article...", "In the same article...", "As per the article...", "cited in the article", etc. without specifying WHICH article (date, news source).
3. FACTUAL: The question asks about real-world facts, events, people, numbers, or consequences — NOT about article metadata (author, publication date, URL).
4. SPECIFIC & UNAMBIGUOUS: The question is precise enough to have exactly one correct answer.
5. SHORT ANSWER: The answer is concise (1-2 sentences max).
6. BACKGROUND DOES NOT LEAK ANSWER: The background provides relevant context but does NOT reveal the answer directly.

=== INSTRUCTIONS ===
Think step by step. For EACH rule above, briefly state whether it passes or fails and why. Then give your final verdict.

After your reasoning, output your final verdict inside XML tags:
- If ALL rules pass: <valid>1</valid>
- If ANY rule fails: <valid>0</valid>

You MUST include exactly one <valid>...</valid> tag at the end of your response."""


def parse_verifier_output(text: str) -> Optional[bool]:
    """Extract the <valid>0</valid> or <valid>1</valid> tag from verifier output.
    Returns True if valid, False if invalid, None if unparseable."""
    match = re.search(r"<valid>\s*([01])\s*</valid>", text)
    if match:
        return match.group(1) == "1"
    # Fallback: check if the text ends with just 0 or 1
    stripped = text.strip()
    if stripped.endswith("1"):
        return True
    if stripped.endswith("0"):
        return False
    return None


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_qa_pairs(model_output: str, expected_count: int) -> List[Dict[str, str]]:
    """
    Parse question/answer pairs from model output.
    Tries JSON first, then falls back to regex.
    """
    # Strip any markdown code fences
    cleaned = model_output.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    def _extract_pairs(parsed_list):
        """Extract q/a/background dicts from a parsed JSON list."""
        pairs = []
        for item in parsed_list:
            if isinstance(item, dict) and "question" in item and "answer" in item:
                pairs.append({
                    "question": str(item["question"]).strip(),
                    "answer": str(item["answer"]).strip(),
                    "background": str(item.get("background", "")).strip(),
                })
        return pairs

    # Try JSON parse
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            pairs = _extract_pairs(parsed)
            if pairs:
                return pairs
    except json.JSONDecodeError:
        pass

    # Fallback: try to find JSON array anywhere in the text
    json_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, list):
                pairs = _extract_pairs(parsed)
                if pairs:
                    return pairs
        except json.JSONDecodeError:
            pass

    # Fallback: regex for individual question/answer/background patterns
    q_pattern = r'"question"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"answer"\s*:\s*"((?:[^"\\]|\\.)*)"(?:\s*,\s*"background"\s*:\s*"((?:[^"\\]|\\.)*)")?'
    matches = re.findall(q_pattern, cleaned, re.DOTALL)
    if matches:
        pairs = []
        for match in matches:
            q = match[0].replace('\\"', '"').replace("\\n", "\n").strip()
            a = match[1].replace('\\"', '"').replace("\\n", "\n").strip()
            bg = match[2].replace('\\"', '"').replace("\\n", "\n").strip() if len(match) > 2 else ""
            pairs.append({"question": q, "answer": a, "background": bg})
        if pairs:
            return pairs

    return []


# ---------------------------------------------------------------------------
# Main generation logic
# ---------------------------------------------------------------------------

def run_generation(args):
    # Set env vars for gpt-oss BEFORE importing vllm
    if _is_gpt_oss(args.model):
        os.environ["VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS"] = "1"
        os.environ["VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8"] = "0"
        os.environ["VLLM_USE_FLASHINFER_MOE_MXFP4_BF16"] = "0"
        print("GPT-OSS model detected: setting FlashInfer CUTLASS env vars")

    from vllm import LLM, SamplingParams

    # Load articles
    print(f"Date filter: excluding articles with max date >= {args.filter_date}")
    print(f"\nLoading articles from: {args.articles}")
    source_articles = load_articles(args.articles, args.num_article)
    if not source_articles:
        print("No articles found. Exiting.")
        return

    total_articles = sum(len(v) for v in source_articles.values())
    print(f"Loaded {total_articles} articles from {len(source_articles)} sources")

    # Init model
    use_harmony = _is_gpt_oss(args.model)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = os.path.basename(os.path.normpath(args.model))
    dir_name = f"{model_short}_q{args.num_q}_a{args.num_article}_fd{args.filter_date}"
    if use_harmony:
        dir_name += f"_{args.effort}"
    dir_name += f"_{timestamp}"
    output_dir = Path(args.articles) / "syntheticqa" / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"\nLoading vLLM model: {args.model}")
    print(f"Harmony format: {use_harmony}")
    if use_harmony:
        print(f"Reasoning effort: {args.effort}")

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

    # If using harmony, also get stop token IDs
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

    # Build ALL prompts across all sources in one flat list
    all_messages = []      # list of message dicts for each prompt
    all_meta = []          # parallel list of article metadata
    skipped_articles = 0
    date_filtered = 0
    too_long = 0
    MAX_ARTICLE_WORDS = 10000

    for source_name, articles in source_articles.items():
        for idx, article in enumerate(articles):
            maintext = article.get("maintext", "") or ""
            description = article.get("description", "") or ""
            if len(maintext) + len(description) < MIN_ARTICLE_LENGTH:
                # print(f"  Skipping {source_name} article {idx} (insufficient content: " f"{len(maintext)} chars maintext)")
                skipped_articles += 1
                continue

            # Word count filter: skip articles with > 10k words
            word_count = len(maintext.split())
            if word_count > MAX_ARTICLE_WORDS:
                too_long += 1
                continue

            # Date filter: skip articles whose max date >= filter_date
            art_max_date = get_article_max_date(article)
            if art_max_date is not None and art_max_date >= args._filter_dt:
                date_filtered += 1
                continue

            prompt_text = build_prompt(article, args.num_q)
            all_messages.append([{"role": "user", "content": prompt_text}])
            all_meta.append({
                "article_id": article.get("id", ""),
                "article_title": article.get("title", ""),
                "article_url": article.get("url", ""),
                "source_domain": article.get("source_domain", ""),
                "date_publish": article.get("date_publish", ""),
                "jsonl_source": source_name,
                "article_index": idx,
            })

    if not all_messages:
        print("No valid articles to process. Exiting.")
        return

    print(f"\nBuilt {len(all_messages)} prompts across {len(source_articles)} sources"
          f" ({skipped_articles} skipped for content, {too_long} too long (>{MAX_ARTICLE_WORDS} words),"
          f" {date_filtered} filtered by date >= {args.filter_date})")

    # Single batch generation for ALL prompts
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

        print(f"Running batch generation ({len(token_prompts)} prompts)...")
        outputs = llm.generate(token_prompts, sampling_params=sampling_params)

        # Map outputs back
        all_results = [None] * len(all_messages)
        for out_idx, output in enumerate(outputs):
            orig_idx = valid_indices[out_idx]
            raw_text = output.outputs[0].text
            tok_ids = list(output.outputs[0].token_ids) if hasattr(output.outputs[0], "token_ids") else None
            all_results[orig_idx] = _parse_harmony_output(raw_text, tok_ids)
    else:
        # Apply chat template for non-gpt-oss models
        plain_prompts = []
        for msgs in all_messages:
            raw_prompt = msgs[0]["content"]
            formatted = apply_chat_template(tokenizer, raw_prompt, model_short)
            plain_prompts.append(formatted)

        print(f"Running batch generation ({len(plain_prompts)} prompts)...")
        outputs = llm.generate(plain_prompts, sampling_params)

        all_results = []
        for output in outputs:
            text = output.outputs[0].text
            # Strip <think>...</think> reasoning for Qwen3 models
            if "</think>" in text:
                text = text.split("</think>", 1)[1]
            all_results.append(text)

    # Parse all results into records
    from collections import defaultdict

    all_records = []  # flat list of (source_name, record_dict)
    parse_failures = 0

    for i, (model_output, meta) in enumerate(zip(all_results, all_meta)):
        source_name = meta["jsonl_source"]

        if model_output is None:
            parse_failures += 1
            continue

        qa_pairs = parse_qa_pairs(model_output, args.num_q)

        if not qa_pairs:
            print(f"  Warning: No Q&A pairs parsed for {source_name} article "
                  f"{meta['article_index']} ({meta['article_title'][:60]}...)")
            print(f"  Raw output (first 300 chars): {model_output[:300]}")
            parse_failures += 1
            continue

        if len(qa_pairs) != args.num_q:
            print(f"  Note: Got {len(qa_pairs)}/{args.num_q} pairs for "
                  f"{source_name} article {meta['article_index']}")

        for pair in qa_pairs:
            record = {
                "question": pair["question"],
                "answer": pair["answer"],
                "background": pair.get("background", ""),
                "raw_model_output": model_output,
                "prompt": all_messages[i][0]["content"],
                "metadata": {
                    "model": model_short,
                    "effort": args.effort,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "num_q": args.num_q,
                    "filter_date": args.filter_date,
                    **meta,
                }
            }
            all_records.append((source_name, record))

    print(f"\nParsed {len(all_records)} Q&A pairs total ({parse_failures} article parse failures)")

    # -----------------------------------------------------------------------
    # Verification pass (optional)
    # -----------------------------------------------------------------------
    if args.verify and all_records:
        print(f"\n{'='*60}")
        print(f"Running verification on {len(all_records)} Q&A pairs...")
        print(f"{'='*60}")

        # Build verifier prompts
        verify_messages = []
        for _, rec in all_records:
            vp = build_verifier_prompt(rec["question"], rec["answer"], rec["background"])
            verify_messages.append([{"role": "user", "content": vp}])

        # Allow enough tokens for reasoning + <valid> tag
        verify_sampling = SamplingParams(
            temperature=1.0,
            top_p=0.95,
            max_tokens=args.max_tokens,
        )

        if use_harmony:
            from vllm.inputs.data import TokensPrompt

            verify_token_prompts = []
            verify_valid_indices = []
            for vi, msgs in enumerate(verify_messages):
                tok_ids = _build_harmony_token_ids(msgs, effort="medium")
                if tok_ids is not None:
                    verify_token_prompts.append(TokensPrompt(prompt_token_ids=tok_ids))
                    verify_valid_indices.append(vi)

            # Add harmony stop tokens to verify sampling params
            encoding = _get_harmony_encoding()
            if encoding is not None:
                stop_ids = encoding.stop_tokens_for_assistant_actions()
                verify_sampling = SamplingParams(
                    temperature=1.0,
                    top_p=0.95,
                    max_tokens=args.max_tokens,
                    stop_token_ids=stop_ids,
                )

            print(f"  Verifying {len(verify_token_prompts)} pairs (Harmony)...")
            verify_outputs = llm.generate(verify_token_prompts, sampling_params=verify_sampling)

            # Parse verification results
            verify_raw = [None] * len(all_records)
            for out_idx, output in enumerate(verify_outputs):
                orig_idx = verify_valid_indices[out_idx]
                raw_text = output.outputs[0].text
                tok_ids = list(output.outputs[0].token_ids) if hasattr(output.outputs[0], "token_ids") else None
                verify_raw[orig_idx] = _parse_harmony_output(raw_text, tok_ids)
        else:
            # Apply chat template for non-gpt-oss models
            plain_verify = []
            for msgs in verify_messages:
                raw_prompt = msgs[0]["content"]
                formatted = apply_chat_template(tokenizer, raw_prompt, model_short)
                plain_verify.append(formatted)

            print(f"  Verifying {len(plain_verify)} pairs...")
            verify_outputs = llm.generate(plain_verify, verify_sampling)
            verify_raw = []
            for out in verify_outputs:
                text = out.outputs[0].text
                if "</think>" in text:
                    text = text.split("</think>", 1)[1]
                verify_raw.append(text)

        # Filter records using <valid> tag parsing
        verified_records = []
        rejected = 0
        unparseable = 0
        for idx, (source_name, record) in enumerate(all_records):
            raw_verdict = verify_raw[idx] or ""
            is_valid = parse_verifier_output(raw_verdict)

            if is_valid is None:
                unparseable += 1
                # If we can't parse the verdict, reject by default
                is_valid = False

            record["metadata"]["verified"] = is_valid
            record["verification_reasoning"] = raw_verdict.strip()

            if is_valid:
                verified_records.append((source_name, record))
            else:
                rejected += 1
                if rejected <= 10:  # log first 20 rejections
                    print(f"  Rejected: {record['question'][:80]}...")
                    # Show last 200 chars of reasoning (where the verdict usually is)
                    print(f"    Reasoning (tail): ...{raw_verdict.strip()[-200:]}")

        print(f"\nVerification complete: {len(verified_records)} passed, {rejected} rejected "
              f"({rejected / len(all_records) * 100:.1f}% rejection rate)")
        if unparseable:
            print(f"  ({unparseable} verdicts were unparseable — treated as rejected)")
        all_records = verified_records

    # -----------------------------------------------------------------------
    # Save per-source
    # -----------------------------------------------------------------------
    source_records = defaultdict(list)
    for source_name, record in all_records:
        source_records[source_name].append(record)

    total_pairs = 0
    for source_name in sorted(source_records.keys()):
        output_path = output_dir / source_name
        with open(output_path, "w", encoding="utf-8") as out_f:
            for record in source_records[source_name]:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        count = len(source_records[source_name])
        total_pairs += count
        print(f"  {source_name}: {count} Q&A pairs saved")

    print(f"\nTotal: {total_pairs} Q&A pairs across {len(source_records)} sources")
    if parse_failures:
        print(f"  {parse_failures} articles skipped (parse failures)")
    print(f"Output at: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic Q&A from news articles using vLLM batch generation."
    )
    parser.add_argument(
        "--model", type=str, default="/fast/nchandak/models/gpt-oss-20b",
        help="Path to vLLM model (default: /fast/nchandak/models/gpt-oss-20b)",
    )
    parser.add_argument(
        "--articles", type=str, default="/fast/sgoel/forecasting/news/articles2025/deduped/relevant/",
        help="Path to folder containing .jsonl article files",
    )
    parser.add_argument(
        "--num_q", type=int, default=10,
        help="Number of Q&A pairs to generate per article (default: 10)",
    )
    parser.add_argument(
        "--num_article", type=int, default=2,
        help="Max articles to take from each JSONL file (default: 10)",
    )
    parser.add_argument(
        "--gpu_mem", type=float, default=0.85,
        help="GPU memory utilization (default: 0.85)",
    )
    parser.add_argument(
        "--max_model_len", type=int, default=16384,
        help="Max model context length (default: 8192)",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=8192,
        help="Max tokens to generate per article (default: 8192)",
    )
    parser.add_argument(
        "--effort", type=str, default="medium", choices=["low", "medium", "high"],
        help="Reasoning effort for gpt-oss Harmony format (default: medium)",
    )
    parser.add_argument(
        "--tp", type=int, default=None,
        help="Tensor parallel size (number of GPUs). Default: auto-detect all available GPUs.",
    )
    parser.add_argument(
        "--filter_date", type=str, default="2025-04-01",
        help="Remove articles whose max date (across date_download, date_modify, "
             "date_publish) is >= this date. Format: YYYY-MM-DD (default: 2025-04-01)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Run a verification pass on generated Q&A pairs and keep only valid ones.",
    )
    args = parser.parse_args()

    # Parse filter_date into an aware datetime
    args._filter_dt = _ensure_aware(parse_date(args.filter_date))

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
    run_generation(args)


if __name__ == "__main__":
    main()
