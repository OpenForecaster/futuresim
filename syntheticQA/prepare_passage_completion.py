#!/usr/bin/env python3
"""
Prepare passage-completion train/val/test splits from news articles.

Input can be a single .jsonl file or a folder containing .jsonl files
(non-recursive).

For each article, samples --repeat random split fractions from
U(--range_min, --range_max), splits the maintext at each fraction (snapping
to sentence boundaries), and builds a training record where the prompt asks
the model to continue the article and the ground-truth is the remainder.

Saves as parquet (verl format), plain JSONL (verl format with --dry_run),
or raw JSONL (--raw_jsonl) into a subdirectory of the input path.

Usage:
    python syntheticQA/prepare_passage_completion.py --input /path/to/aljazeera.jsonl
    python syntheticQA/prepare_passage_completion.py --input /path/to/articles_folder/
    python syntheticQA/prepare_passage_completion.py --input /path/to/aljazeera.jsonl --repeat 5 --range_min 0.2 --range_max 0.6
    python syntheticQA/prepare_passage_completion.py --input /path/to/aljazeera.jsonl --raw_jsonl
"""

import argparse
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from dateutil.parser import parse as parse_date


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

_DATE_FIELDS = ("date_download", "date_modify", "date_publish")


def _ensure_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def get_article_max_date(article: dict) -> Optional[datetime]:
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
# Article loading
# ---------------------------------------------------------------------------

MIN_ARTICLE_LENGTH = 200
MAX_ARTICLE_WORDS = 10000


def load_articles(articles_path: str, num_article: int,
                  filter_dt: Optional[datetime] = None) -> List[dict]:
    """Load up to num_article articles from a single .jsonl file."""
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

                if len(maintext) + len(description) < MIN_ARTICLE_LENGTH:
                    skipped_short += 1
                    continue
                if len(maintext.split()) > MAX_ARTICLE_WORDS:
                    skipped_long += 1
                    continue

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
# Passage splitting (same as eval_passage_completion.py)
# ---------------------------------------------------------------------------

def split_passage(text: str, split_frac: float) -> tuple:
    """Split article text into (prefix, suffix) at ~split_frac, snapping to
    a sentence boundary. Returns (prefix, suffix)."""
    if not text or not text.strip():
        return (text, "")

    target_pos = int(len(text) * split_frac)

    sentence_ends = [m.end() for m in re.finditer(r'[.!?]\s+', text)]

    if not sentence_ends:
        para_ends = [m.end() for m in re.finditer(r'\n\s*\n', text)]
        if para_ends:
            best = min(para_ends, key=lambda p: abs(p - target_pos))
            return (text[:best].rstrip(), text[best:].lstrip())
        space_pos = text.rfind(' ', 0, target_pos)
        if space_pos > 0:
            return (text[:space_pos], text[space_pos + 1:])
        return (text[:target_pos], text[target_pos:])

    best = min(sentence_ends, key=lambda p: abs(p - target_pos))
    return (text[:best].rstrip(), text[best:].lstrip())


# ---------------------------------------------------------------------------
# Prompt construction (same as eval_passage_completion.py)
# ---------------------------------------------------------------------------

def build_completion_prompt(prefix: str, article: dict) -> str:
    """Build the prompt for passage completion (always no_think for training)."""
    parts = []
    parts.append(
        "You are given a news article with its metadata and the beginning of "
        "its body text. Your task is to continue the body text from where it "
        "left off. Write a plausible, coherent, and factually grounded "
        "continuation that matches the style, tone, and content of the "
        "passage so far."
    )

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

    parts.append(
        f"\n=== ARTICLE TEXT (BEGINNING) ===\n{prefix.strip()}\n"
        f"=== END OF PROVIDED TEXT ==="
    )

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
        "\n\nJUST COMPLETE THE ARTICLE (DO NOT OUTPUT ANYTHING ELSE, "
        "NO PREAMBLE OR EXPLANATION). /no_think"
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Verl format transformation
# ---------------------------------------------------------------------------

def transform_record(record: dict, idx: int, split: str) -> dict:
    """Transform a passage-completion record into the verl training format."""
    prompt_text = record["prompt"]
    ground_truth = record["ground_truth_continuation"]
    metadata = record.get("metadata", {})

    return {
        "data_source": "syntheticqa/passage_completion",
        "prompt": [
            {
                "role": "user",
                "content": prompt_text,
            }
        ],
        "ability": "passage_completion",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "split": split,
            "index": idx,
            "prompt": prompt_text,
            "response": ground_truth,
            "split_frac": record["split_frac"],
            "article_title": metadata.get("article_title", ""),
            "article_url": metadata.get("article_url", ""),
            "source_domain": metadata.get("source_domain", ""),
            "date_publish": metadata.get("date_publish", ""),
            "jsonl_source": metadata.get("jsonl_source", ""),
            "article_index": metadata.get("article_index", ""),
            "repeat_index": metadata.get("repeat_index", ""),
            "prefix_words": metadata.get("prefix_words", 0),
            "suffix_words": metadata.get("suffix_words", 0),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare passage-completion train/val/test splits from "
                    "a news articles JSONL file."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to a single .jsonl file or a folder containing .jsonl files "
             "(e.g. /path/to/aljazeera.jsonl or /path/to/articles/)",
    )
    parser.add_argument(
        "--num_article", type=int, default=100000,
        help="Max articles to load from the JSONL file (default: 100000 = all)",
    )
    parser.add_argument(
        "--range_min", type=float, default=0.1,
        help="Minimum prefix fraction (default: 0.1)",
    )
    parser.add_argument(
        "--range_max", type=float, default=0.7,
        help="Maximum prefix fraction (default: 0.5)",
    )
    parser.add_argument(
        "--repeat", type=int, default=50,
        help="Number of random splits per article (default: 10)",
    )
    parser.add_argument(
        "--train_frac", type=float, default=0.995,
        help="Fraction of data for training (default: 0.95)",
    )
    parser.add_argument(
        "--val_frac", type=float, default=0.004,
        help="Fraction of data for validation (default: 0.03)",
    )
    parser.add_argument(
        "--test_frac", type=float, default=0.001,
        help="Fraction of data for testing (default: 0.02)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--filter_date", type=str, default="2025-04-01",
        help="Remove articles whose max date >= this date. "
             "Format: YYYY-MM-DD (default: 2025-04-01)",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Save as plain JSONL files (verl format) instead of HF parquet.",
    )
    parser.add_argument(
        "--raw_jsonl", action="store_true",
        help="Save raw JSONL (no verl transformation). "
             "Useful for inspection or non-training purposes.",
    )
    args = parser.parse_args()

    # Validate fractions
    total_frac = args.train_frac + args.val_frac + args.test_frac
    if abs(total_frac - 1.0) > 0.01:
        print(f"Warning: fractions sum to {total_frac:.3f}, not 1.0. Proceeding anyway.")

    if args.range_min >= args.range_max:
        print("Error: --range_min must be < --range_max", file=sys.stderr)
        sys.exit(1)

    # Parse filter date
    filter_dt = _ensure_aware(parse_date(args.filter_date))

    # Determine input files
    input_path = Path(args.input)
    if input_path.is_file():
        jsonl_files = [input_path]
    elif input_path.is_dir():
        jsonl_files = sorted(
            f for f in input_path.iterdir()
            if f.is_file() and f.suffix == ".jsonl"
        )
        if not jsonl_files:
            print(f"No .jsonl files found in {args.input}. Exiting.")
            sys.exit(1)
        print(f"Found {len(jsonl_files)} .jsonl files in {args.input}")
    else:
        print(f"Error: {args.input} is not a file or directory", file=sys.stderr)
        sys.exit(1)

    # Load articles from all input files
    print(f"Date filter: excluding articles with max date >= {args.filter_date}")
    all_articles: List[tuple] = []  # (source_name, article)
    for fpath in jsonl_files:
        print(f"\nLoading articles from: {fpath.name}")
        articles = load_articles(str(fpath), args.num_article, filter_dt=filter_dt)
        for art in articles:
            all_articles.append((fpath.name, art))

    if not all_articles:
        print("No articles found. Exiting.")
        sys.exit(1)

    total_articles = len(all_articles)
    print(f"\nTotal: {total_articles} articles from {len(jsonl_files)} file(s)")

    # Generate records
    print(f"Generating records: {args.repeat} splits per article, "
          f"prefix fraction ~ U({args.range_min}, {args.range_max})")

    all_records = []
    rng = random.Random(args.seed)
    skipped = 0

    for art_idx, (source_name, article) in enumerate(all_articles):
        maintext = article.get("maintext", "") or ""
        if not maintext.strip():
            continue

        for rep in range(args.repeat):
            frac = rng.uniform(args.range_min, args.range_max)
            prefix, suffix = split_passage(maintext, frac)

            if (len(prefix.split()) < 30
                    or len(suffix.split()) < 20
                    or len(suffix.split()) > 5000):
                skipped += 1
                continue

            prompt_text = build_completion_prompt(prefix, article)

            if len(prompt_text.split()) > 5000:
                skipped += 1
                continue

            all_records.append({
                "prefix": prefix,
                "ground_truth_continuation": suffix,
                "prompt": prompt_text,
                "split_frac": round(frac, 4),
                "metadata": {
                    "article_title": article.get("title", ""),
                    "article_url": article.get("url", ""),
                    "source_domain": article.get("source_domain", ""),
                    "date_publish": article.get("date_publish", ""),
                    "jsonl_source": source_name,
                    "article_index": art_idx,
                    "repeat_index": rep,
                    "prefix_words": len(prefix.split()),
                    "suffix_words": len(suffix.split()),
                },
            })

    print(f"  Generated {len(all_records)} records "
          f"({skipped} skipped due to short prefix/suffix)")

    if not all_records:
        print("No records generated. Exiting.")
        sys.exit(1)

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(all_records)

    n = len(all_records)
    n_train = int(args.train_frac * n)
    n_val = int(args.val_frac * n)

    train_records = all_records[:n_train]
    val_records = all_records[n_train:n_train + n_val]
    test_records = all_records[n_train + n_val:]

    print(f"\nSplit sizes: train={len(train_records)}, "
          f"val={len(val_records)}, test={len(test_records)}")

    # Build output directory name
    if input_path.is_file():
        src_stem = input_path.stem
        output_parent = input_path.parent
    else:
        src_stem = input_path.name
        output_parent = input_path
    dir_name = (f"{src_stem}_pc_r{args.repeat}"
                f"_{args.range_min}-{args.range_max}"
                f"_a{total_articles}")
    base_output = output_parent / "passage_completion" / dir_name

    # Save
    if args.raw_jsonl:
        output_dir = base_output / "raw"
        output_dir.mkdir(parents=True, exist_ok=True)

        splits = [("train", train_records), ("val", val_records), ("test", test_records)]
        for split_name, split_data in splits:
            out_path = output_dir / f"{split_name}_{len(split_data)}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for record in split_data:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"Saved {split_name} ({len(split_data)} records) -> {out_path}")
    else:
        train_data = [transform_record(r, i, "train") for i, r in enumerate(train_records)]
        val_data = [transform_record(r, i, "val") for i, r in enumerate(val_records)]
        test_data = [transform_record(r, i, "test") for i, r in enumerate(test_records)]

        output_dir = base_output / "training"
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.dry_run:
            for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
                out_path = output_dir / f"{split_name}_{len(split_data)}.jsonl"
                with open(out_path, "w", encoding="utf-8") as f:
                    for record in split_data:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"Saved {split_name} ({len(split_data)} records) -> {out_path}")
        else:
            import datasets as ds

            train_dataset = ds.Dataset.from_list(train_data)
            val_dataset = ds.Dataset.from_list(val_data)
            test_dataset = ds.Dataset.from_list(test_data)

            train_path = output_dir / f"train_{len(train_dataset)}.parquet"
            val_path = output_dir / f"val_{len(val_dataset)}.parquet"
            test_path = output_dir / f"test_{len(test_dataset)}.parquet"

            train_dataset.to_parquet(str(train_path))
            print(f"Saved train ({len(train_dataset)} records) -> {train_path}")

            val_dataset.to_parquet(str(val_path))
            print(f"Saved val ({len(val_dataset)} records) -> {val_path}")

            test_dataset.to_parquet(str(test_path))
            print(f"Saved test ({len(test_dataset)} records) -> {test_path}")

    print(f"\nDone. Output at: {output_dir}")


if __name__ == "__main__":
    main()
