#!/usr/bin/env python3
"""Compatibility wrapper to run mpi_scripts/embed/submit_job.py from aisa_scripts."""

import runpy
from pathlib import Path


def main() -> None:
    repo_dir = Path(__file__).resolve().parents[2]
    target = repo_dir / "mpi_scripts" / "embed" / "submit_job.py"
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
