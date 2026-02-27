#!/usr/bin/env python3
"""
Convert a verl-format training/test JSONL file (produced by prepare_distillation.py)
into the standard QA JSONL format expected by eval_qa.py.

eval_qa.py expects a *folder* of .jsonl files where each record has top-level
"question", "answer", "background", and "metadata" keys.  This script reads a
single verl-format JSONL, extracts those fields from extra_info/reward_model,
writes a compatible .jsonl file inside a new output folder, and prints the
folder path so it can be passed directly to eval_qa.py --questions.

Usage:
    python syntheticQA/convert_to_eval_jsonl.py --input /path/to/test_494.jsonl
    python syntheticQA/convert_to_eval_jsonl.py --input /path/to/test_494.jsonl --output /custom/output/dir/

Then evaluate:
    python syntheticQA/eval_qa.py --questions <output_folder> --model ...
"""

import argparse
import json
import sys
from pathlib import Path


def convert(input_path: str, output_dir: str | None) -> None:
    src = Path(input_path)
    if not src.is_file():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Default output: sibling folder named after the input file (without extension)
    if output_dir is None:
        out_dir = src.parent / f"{src.stem}_eval"
    else:
        out_dir = Path(output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / src.name  # keep the same filename

    converted = 0
    skipped = 0

    with open(src, "r", encoding="utf-8", errors="replace") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            extra = record.get("extra_info", {})
            reward = record.get("reward_model", {})

            question = extra.get("question", "")
            if not question.strip():
                skipped += 1
                continue

            out_record = {
                "question": question,
                "answer": extra.get("ground_truth", "") or reward.get("ground_truth", ""),
                "background": extra.get("background", ""),
                "metadata": {
                    "model": extra.get("gen_model", ""),
                    "effort": extra.get("gen_effort", ""),
                    "article_title": extra.get("article_title", ""),
                    "article_url": extra.get("article_url", ""),
                    "source_domain": extra.get("source_domain", ""),
                    "date_publish": extra.get("date_publish", ""),
                    "jsonl_source": extra.get("source_file", ""),
                    "original_split": extra.get("split", ""),
                    "original_index": extra.get("index", ""),
                },
            }
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            converted += 1

    print(f"Converted {converted} records ({skipped} skipped)")
    print(f"Output file: {out_path}")
    print(f"\nTo evaluate, run:")
    print(f"  python syntheticQA/eval_qa.py --questions {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert verl-format training JSONL to eval_qa.py-compatible "
                    "questions folder."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to the verl-format JSONL file (e.g. test_494.jsonl)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: <input_dir>/<input_stem>_eval/). "
             "The converted .jsonl will be placed inside this folder.",
    )
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
