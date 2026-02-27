#!/usr/bin/env python3
"""
One-time setup utility: download/verify NLTK tokenizer data needed by the CCNews crawl.

This avoids doing network-y downloads inside HTCondor jobs and keeps run_news_crawl.sh simple.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    try:
        import nltk
    except Exception as e:
        print(f"ERROR: Could not import nltk: {e}", file=sys.stderr)
        return 2

    # Prefer a repo-shared location so HTCondor nodes see the same data.
    default_dir = Path.home() / "forecast-sim" / "fsim" / "nltk_data"
    download_dir = Path(os.environ.get("NLTK_DATA", str(default_dir))).expanduser().resolve()
    download_dir.mkdir(parents=True, exist_ok=True)

    # Ensure NLTK looks in this directory first.
    nltk.data.path.insert(0, str(download_dir))

    pkgs = ["punkt_tab", "punkt"]
    missing: list[str] = []
    for pkg in pkgs:
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            missing.append(pkg)

    if missing:
        print(f"NLTK_DATA={download_dir}")
        for pkg in missing:
            ok = False
            try:
                ok = bool(nltk.download(pkg, download_dir=str(download_dir), quiet=True))
            except Exception as e:
                print(f"ERROR: nltk.download({pkg!r}) failed: {e}", file=sys.stderr)
            if not ok:
                print(
                    f"ERROR: Failed to download {pkg!r} into {download_dir}. "
                    "If the machine has no internet, download from a login node and re-run.",
                    file=sys.stderr,
                )
                return 2

    # Verify again.
    still_missing: list[str] = []
    for pkg in pkgs:
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            still_missing.append(pkg)

    if still_missing:
        print(f"ERROR: NLTK packages still missing: {still_missing}", file=sys.stderr)
        print(f"NLTK_DATA={download_dir}", file=sys.stderr)
        return 2

    print(f"OK: NLTK tokenizer data present in {download_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

