#!/usr/bin/env python3
"""
Replay matcher prompts from existing matcher.jsonl files with a new matcher model,
then compute alignment against the original matcher outputs.

Target supported:
- A single timestamped run directory containing matcher.jsonl

Per run outputs:
- matcher_{newmatchername}.jsonl
- alignment_mismatches_{newmatchername}.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environment.ansmatching import (
    build_find_match_prompt,
    build_is_equivalent_prompt,
    parse_find_match_response,
    parse_is_equivalent_response,
)
from inference.openrouter import GlobalRateLimiter, OpenRouterInference


CHECK_EQ_TYPES = {"check_guess", "is_equivalent", "expand_set"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay matcher.jsonl prompts with a new matcher and print per-run alignment metrics"
    )
    parser.add_argument(
        "run_dir",
        help="Full path to one timestamped run directory containing matcher.jsonl",
    )
    parser.add_argument(
        "--model",
        default="deepseek/deepseek-v3.2",
        help="OpenRouter model ID to use for replay (default: deepseek/deepseek-v3.2)",
    )
    parser.add_argument(
        "--new-matcher-name",
        default=None,
        help="Filename-safe matcher suffix; defaults to a sanitized form of --model",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=24,
        help="Thread workers for replay requests (default: 24)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=32.0,
        help="Global OpenRouter request rate limit, requests/sec (default: 32)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="OpenRouter retries for transient failures (default: 3)",
    )
    parser.add_argument(
        "--retry-backoff-base-s",
        type=float,
        default=1.0,
        help="Retry backoff base seconds (default: 1.0)",
    )
    parser.add_argument(
        "--retry-backoff-max-s",
        type=float,
        default=16.0,
        help="Retry backoff cap seconds (default: 16.0)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N completed requests per run (default: 100)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing matcher_{newmatchername}.jsonl files",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap for replay rows per run (for smoke tests)",
    )
    return parser.parse_args()


def sanitize_component(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return safe or "newmatcher"


def resolve_run_dir(run_dir: Path) -> Path:
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise ValueError(f"run_dir is not a directory: {run_dir}")
    if not (run_dir / "matcher.jsonl").exists():
        raise ValueError(f"matcher.jsonl not found in run_dir: {run_dir}")
    return run_dir


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{lineno}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def replay_record(old_record: Dict[str, Any], inference: OpenRouterInference) -> Dict[str, Any]:
    input_data = old_record.get("input", {})
    match_type = input_data.get("type", "is_equivalent")

    if match_type in CHECK_EQ_TYPES:
        prompt = build_is_equivalent_prompt(
            predicted=str(input_data.get("predicted", "")),
            ground_truth=str(input_data.get("ground_truth", "")),
            question_title=input_data.get("question_title"),
        )
        response, _ = inference.chat(
            [{"role": "user", "content": prompt}],
            {"temperature": 0.0, "max_tokens": 10},
        )
        output_data: Dict[str, Any] = {
            "response": response,
            "is_equivalent": parse_is_equivalent_response(response),
        }
    elif match_type == "find_match":
        existing = input_data.get("existing") or []
        if not isinstance(existing, list):
            existing = list(existing)
        prompt = build_find_match_prompt(
            candidate=str(input_data.get("candidate", "")),
            existing_outcomes=existing,
            question_title=input_data.get("question_title"),
        )
        response, _ = inference.chat(
            [{"role": "user", "content": prompt}],
            {"temperature": 0.0, "max_tokens": 10},
        )
        output_data = {
            "response": response,
            "matched": parse_find_match_response(response, existing),
        }
    else:
        raise ValueError(f"Unsupported matcher input type: {match_type!r}")

    return {
        "timestamp": datetime.now().isoformat(),
        "input": input_data,
        "output": output_data,
        "metadata": old_record.get("metadata", {}),
    }


def replay_run(
    old_rows: Sequence[Dict[str, Any]],
    inference: OpenRouterInference,
    workers: int,
    progress_every: int,
) -> List[Dict[str, Any]]:
    new_rows: List[Optional[Dict[str, Any]]] = [None] * len(old_rows)

    if workers <= 1:
        for idx, row in enumerate(old_rows, start=1):
            new_rows[idx - 1] = replay_record(row, inference)
            if idx % progress_every == 0 or idx == len(old_rows):
                print(f"    Progress: {idx}/{len(old_rows)}", flush=True)
        return [r for r in new_rows if r is not None]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(replay_record, row, inference): idx
            for idx, row in enumerate(old_rows)
        }

        completed = 0
        for future in as_completed(futures):
            idx = futures[future]
            new_rows[idx] = future.result()
            completed += 1
            if completed % progress_every == 0 or completed == len(old_rows):
                print(f"    Progress: {completed}/{len(old_rows)}", flush=True)

    return [r for r in new_rows if r is not None]


def extract_label(record: Dict[str, Any]) -> Optional[bool]:
    output_data = record.get("output", {})
    val = output_data.get("is_equivalent")
    if isinstance(val, bool):
        return val

    response = output_data.get("response")
    if isinstance(response, str):
        low = response.lower()
        if "yes" in low:
            return True
        if "no" in low:
            return False
    return None


def compute_alignment(old_rows: Sequence[Dict[str, Any]], new_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if len(old_rows) != len(new_rows):
        raise ValueError(f"Row count mismatch: old={len(old_rows)} new={len(new_rows)}")

    n_yy = 0
    n_nn = 0
    n_yn = 0
    n_ny = 0
    comparable = 0

    for old_rec, new_rec in zip(old_rows, new_rows):
        old_label = extract_label(old_rec)
        new_label = extract_label(new_rec)
        if old_label is None or new_label is None:
            continue

        comparable += 1
        if old_label and new_label:
            n_yy += 1
        elif (not old_label) and (not new_label):
            n_nn += 1
        elif old_label and (not new_label):
            n_yn += 1
        else:
            n_ny += 1

    if comparable == 0:
        return {
            "num_rows": len(old_rows),
            "num_comparable": 0,
            "num_agree": 0,
            "num_disagree": 0,
            "raw_agreement_pct": None,
            "scotts_pi": None,
        }

    num_agree = n_yy + n_nn
    num_disagree = n_yn + n_ny
    po = num_agree / comparable

    old_yes = n_yy + n_yn
    new_yes = n_yy + n_ny

    p_yes = (old_yes + new_yes) / (2.0 * comparable)
    p_no = 1.0 - p_yes
    pe = p_yes ** 2 + p_no ** 2

    den = 1.0 - pe
    if math.isclose(den, 0.0, abs_tol=1e-12):
        scotts_pi = 1.0 if math.isclose(po, 1.0, abs_tol=1e-12) else 0.0
    else:
        scotts_pi = (po - pe) / den

    return {
        "num_rows": len(old_rows),
        "num_comparable": comparable,
        "num_agree": num_agree,
        "num_disagree": num_disagree,
        "raw_agreement_pct": 100.0 * po,
        "scotts_pi": scotts_pi,
    }


def collect_mismatches(
    old_rows: Sequence[Dict[str, Any]],
    new_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    mismatches: List[Dict[str, Any]] = []

    for old_rec, new_rec in zip(old_rows, new_rows):
        old_label = extract_label(old_rec)
        new_label = extract_label(new_rec)
        if old_label is None or new_label is None:
            continue
        if old_label == new_label:
            continue

        input_data = old_rec.get("input", {})
        mismatches.append(
            {
                "predicted_outcome": input_data.get("predicted"),
                "groundtruth": input_data.get("ground_truth"),
                "original_matcher": old_label,
                "new_matcher": new_label,
                "question": input_data.get("question_title"),
            }
        )

    return mismatches


def main() -> int:
    args = parse_args()

    run_dir = resolve_run_dir(Path(args.run_dir))

    new_matcher_name = args.new_matcher_name or sanitize_component(args.model)
    print(f"Run: {run_dir}")
    print(f"New matcher name: {new_matcher_name}")
    print(f"OpenRouter model: {args.model}")

    GlobalRateLimiter.configure(args.rate_limit)
    inference = OpenRouterInference(
        args.model,
        max_retries=args.max_retries,
        base_delay=args.retry_backoff_base_s,
        max_delay=args.retry_backoff_max_s,
    )

    matcher_in = run_dir / "matcher.jsonl"
    matcher_out = run_dir / f"matcher_{new_matcher_name}.jsonl"

    old_rows = read_jsonl(matcher_in)
    if args.max_rows is not None:
        old_rows = old_rows[: args.max_rows]

    if matcher_out.exists() and not args.overwrite:
        print(f"Reusing existing: {matcher_out}")
        new_rows = read_jsonl(matcher_out)
        if args.max_rows is not None:
            new_rows = new_rows[: args.max_rows]
        if len(new_rows) != len(old_rows):
            raise ValueError(
                f"Existing output length mismatch in {matcher_out}: "
                f"new={len(new_rows)} old={len(old_rows)}. Use --overwrite."
            )
    else:
        if matcher_out.exists() and args.overwrite:
            print(f"Overwriting: {matcher_out}")
        else:
            print(f"Creating: {matcher_out}")

        print(f"Replaying {len(old_rows)} rows with {args.workers} workers...")
        new_rows = replay_run(
            old_rows=old_rows,
            inference=inference,
            workers=args.workers,
            progress_every=max(1, args.progress_every),
        )
        write_jsonl(matcher_out, new_rows)

    metrics = compute_alignment(old_rows, new_rows)
    mismatches = collect_mismatches(old_rows, new_rows)
    mismatch_jsonl = run_dir / f"alignment_mismatches_{new_matcher_name}.jsonl"
    write_jsonl(mismatch_jsonl, mismatches)

    raw_pct = metrics.get("raw_agreement_pct")
    pi = metrics.get("scotts_pi")
    raw_txt = "n/a" if raw_pct is None else f"{raw_pct:.2f}%"
    pi_txt = "n/a" if pi is None else f"{pi:.6f}"

    print(f"raw_agreement={raw_txt}")
    print(f"scotts_pi={pi_txt}")
    print(f"mismatches={len(mismatches)}")
    print(f"Matcher output: {matcher_out}")
    print(f"Mismatch JSONL: {mismatch_jsonl}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
