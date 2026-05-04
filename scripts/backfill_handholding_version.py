#!/usr/bin/env python3
"""Backfill handholding_version into existing run config.json files.

Walks every sim dir under <root> (default: final_runs_v37), inspects the
agent's rendered prompt(s), and writes a `handholding_version` field
(`v1`/`v2`/`v3`) at the top level of the run's config.json.

Detection rules (looking at the rendered prompt the agent actually saw):
  - "questions resolve tomorrow" / "No questions resolve tomorrow" present  -> v3
  - else "TW score equally rewards updating" present                        -> v2
  - else                                                                    -> v1

For default prompt_mode, the prompt lives at
    agents/<id>/system_prompt.md
For active_memory prompt_mode, per-day prompts are written to
    agents/<id>/active_memory_prompt_YYYY-MM-DD.md
We pick the latest active_memory_prompt_*.md (post-resolution-date so that the
imminent reminder logic is exercised; same rule across days, so any prompt
suffices, but we use the latest for stability).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


IMMINENT_MARKERS = (
    "question(s) resolve tomorrow",
    "No questions resolve tomorrow",
)
TW_NUDGE_MARKER = "TW score equally rewards updating"


def detect_version_from_text(text: str) -> str:
    if any(m in text for m in IMMINENT_MARKERS):
        return "v3"
    if TW_NUDGE_MARKER in text:
        return "v2"
    return "v1"


def find_prompt_file(agent_dir: Path) -> Optional[Path]:
    """Return the prompt file to inspect for an agent dir, or None if none found.

    Looks, in order, for:
      - system_prompt.md          (default minimal-harness mode)
      - active_memory_prompt_*.md (active_memory mode; latest by date suffix)
      - warmup_prompts/*.md       (rg1 warmup mode; pick the largest, which
                                   is the full system prompt rather than a
                                   short per-question wrapper)
    """
    sp = agent_dir / "system_prompt.md"
    if sp.is_file() and sp.stat().st_size > 0:
        return sp
    am_prompts = sorted(agent_dir.glob("active_memory_prompt_*.md"))
    if am_prompts:
        return am_prompts[-1]
    warmup = list((agent_dir / "warmup_prompts").glob("*.md")) if (agent_dir / "warmup_prompts").is_dir() else []
    if warmup:
        return max(warmup, key=lambda p: p.stat().st_size)
    # Some rg1-warmup runs only persist the rendered prompt under
    # warmup_logs/<wid>/prompt.md (one per warmup unit). Pick the largest.
    wl_root = agent_dir / "warmup_logs"
    if wl_root.is_dir():
        wl_prompts = list(wl_root.glob("*/prompt.md"))
        if wl_prompts:
            return max(wl_prompts, key=lambda p: p.stat().st_size)
    return None


def detect_version_for_run(run_dir: Path) -> tuple[Optional[str], list[str]]:
    """Inspect every agent in run_dir/agents and return the consensus version.

    Returns (version, notes). version is None if no prompt was found.
    notes lists per-agent observations for diagnostics.
    """
    agents_dir = run_dir / "agents"
    if not agents_dir.is_dir():
        return None, ["no agents/ dir"]
    versions: list[tuple[str, str]] = []  # (agent_name, version)
    notes: list[str] = []
    for agent_path in sorted(agents_dir.iterdir()):
        if not agent_path.is_dir():
            continue
        prompt = find_prompt_file(agent_path)
        if prompt is None:
            notes.append(f"{agent_path.name}: no prompt file")
            continue
        try:
            text = prompt.read_text(errors="replace")
        except OSError as e:
            notes.append(f"{agent_path.name}: read error {e}")
            continue
        v = detect_version_from_text(text)
        versions.append((agent_path.name, v))
        notes.append(f"{agent_path.name}: {v} ({prompt.name})")
    if not versions:
        return None, notes
    distinct = {v for _, v in versions}
    if len(distinct) == 1:
        return next(iter(distinct)), notes
    # Mixed — should not happen on a single run; keep the highest (most explicit)
    # version and flag in notes.
    chosen = "v3" if "v3" in distinct else ("v2" if "v2" in distinct else "v1")
    notes.append(f"MIXED versions across agents; chose {chosen}")
    return chosen, notes


def update_config(config_path: Path, version: str, dry_run: bool) -> str:
    """Write handholding_version into config.json. Returns one of:
      'wrote'   — field added
      'updated' — field changed value
      'noop'    — field already correct
      'error'   — exception
    """
    try:
        with config_path.open() as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return f"error: {e}"
    prior = cfg.get("handholding_version")
    if prior == version:
        return "noop"
    cfg["handholding_version"] = version
    if dry_run:
        return "wrote (dry-run)" if prior is None else f"updated (dry-run) {prior} -> {version}"
    tmp = config_path.with_suffix(config_path.suffix + ".tmp")
    try:
        with tmp.open("w") as f:
            json.dump(cfg, f, indent=2, default=str)
        tmp.replace(config_path)
    except OSError as e:
        return f"error: {e}"
    return "wrote" if prior is None else f"updated {prior} -> {version}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="/fast/sgoel/forecasting/current_sim/final_runs_v37",
        help="Root directory holding sim subdirs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect and report but do not modify config.json.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-agent prompt detection notes.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"ERROR: root not found: {root}", file=sys.stderr)
        return 2

    counts = {"v1": 0, "v2": 0, "v3": 0, "no_prompt": 0, "no_config": 0, "error": 0}
    rows: list[tuple[str, str, str]] = []  # (run_path, version, action)

    for sim_dir in sorted(root.iterdir()):
        if not sim_dir.is_dir():
            continue
        for run_dir in sorted(sim_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            cfg_path = run_dir / "config.json"
            if not cfg_path.is_file():
                counts["no_config"] += 1
                rows.append((str(run_dir), "-", "no config.json"))
                continue
            version, notes = detect_version_for_run(run_dir)
            if version is None:
                counts["no_prompt"] += 1
                rows.append((str(run_dir), "-", "no prompt"))
                if args.verbose:
                    for n in notes:
                        print(f"  {n}")
                continue
            counts[version] += 1
            action = update_config(cfg_path, version, dry_run=args.dry_run)
            rows.append((str(run_dir), version, action))
            if args.verbose:
                for n in notes:
                    print(f"  {n}")

    print()
    print(f"Scanned root: {root}")
    print(f"Runs by detected version:")
    for k in ("v1", "v2", "v3"):
        print(f"  {k}: {counts[k]}")
    print(f"  no_prompt (skipped): {counts['no_prompt']}")
    print(f"  no_config (skipped): {counts['no_config']}")
    print(f"  errors: {counts['error']}")
    print()
    print(f"{'VERSION':<8}  {'ACTION':<35}  RUN")
    for run_path, v, action in rows:
        print(f"{v:<8}  {action:<35}  {run_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
