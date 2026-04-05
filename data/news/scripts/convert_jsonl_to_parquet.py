#!/usr/bin/env python3
"""Compatibility wrapper for the shared parquet conversion script."""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    target = repo_root / "scripts" / "convert_jsonl_to_parquet.py"
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
