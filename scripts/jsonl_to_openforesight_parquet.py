#!/usr/bin/env python3
"""Convert a JSONL question file (OpenForesight-like rows) to split parquet(s) next to HuggingFace-style OpenForesight exports.

Output columns, order, and pandas dtypes match existing disk splits (e.g. aljazeeraSept25) and
``OPENFORESIGHT_COLUMNS`` (see below), as consumed by ``data/fetchqs/openforesight.py``.

Optional --max-rows: after normalizing columns, sort by resolution_date and qid, then keep the first N rows (deterministic subset).

Optional --reference-parquet: assert column order and dtypes match an existing split file before writing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# Columns expected by data/fetchqs/openforesight.py (disk parquet branch) and typical HF export.
OPENFORESIGHT_COLUMNS = [
    "qid",
    "question_title",
    "background",
    "resolution_criteria",
    "answer_type",
    "answer",
    "url",
    "article_maintext",
    "article_publish_date",
    "article_modify_date",
    "article_download_date",
    "article_description",
    "article_title",
    "data_source",
    "news_source",
    "resolution_date",
    "question_start_date",
    "prompt",
    "prompt_without_retrieval",
]


def assert_aligned_with_reference(df: pd.DataFrame, reference_parquet: Path) -> None:
    """Fail fast if this frame does not match an on-disk OpenForesight split layout."""
    ref = pd.read_parquet(reference_parquet)
    if list(df.columns) != list(ref.columns):
        raise SystemExit(
            f"Column mismatch vs reference {reference_parquet}:\n"
            f"  built:  {list(df.columns)}\n"
            f"  ref:    {list(ref.columns)}"
        )
    bad = []
    for c in df.columns:
        if df[c].dtype != ref[c].dtype:
            bad.append(f"  {c}: built={df[c].dtype} ref={ref[c].dtype}")
    if bad:
        raise SystemExit(
            "dtype mismatch vs reference:\n" + "\n".join(bad) + f"\n(ref: {reference_parquet})"
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path, help="Source .jsonl path")
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="OpenForesight directory (same as dataset_path; will write split-*.parquet)",
    )
    p.add_argument(
        "--split",
        required=True,
        help="Split name (file will be {split}-00000-of-00001.parquet)",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="If set, keep only this many rows after sorting by resolution_date, qid (stable subset).",
    )
    p.add_argument(
        "--reference-parquet",
        type=Path,
        default=None,
        help="Existing split parquet to compare columns and dtypes against (e.g. aljazeeraSept25-00000-of-00001.parquet).",
    )
    args = p.parse_args()

    rows = []
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    if "question_id" in df.columns and "qid" not in df.columns:
        df = df.rename(columns={"question_id": "qid"})
    df["qid"] = df["qid"].astype(str)

    for col in OPENFORESIGHT_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df = df[OPENFORESIGHT_COLUMNS]
    for col in ("article_publish_date", "article_modify_date", "article_download_date", "resolution_date", "question_start_date"):
        df[col] = df[col].astype(str).replace({"nan": "", "None": "", "<NA>": ""})

    # Deterministic subset: chronological by resolution_date, then qid.
    df["_rd"] = pd.to_datetime(df["resolution_date"], errors="coerce")
    df = df.sort_values(["_rd", "qid"], na_position="last").drop(columns=["_rd"])
    if args.max_rows is not None:
        if args.max_rows <= 0:
            raise SystemExit("--max-rows must be positive")
        df = df.head(args.max_rows).reset_index(drop=True)

    if args.reference_parquet is not None:
        assert_aligned_with_reference(df, args.reference_parquet)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"{args.split}-00000-of-00001.parquet"
    df.to_parquet(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
