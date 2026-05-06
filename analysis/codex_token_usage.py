#!/usr/bin/env python3
"""Recover per-day token usage for a Codex-based simulation run.

Codex's harness-side ``codex_stdout.jsonl`` only emits a ``turn.completed``
record for the first turn of a thread, so 95-day ``codex exec resume`` runs
log usage for day 0 only. The full picture lives in Codex's own per-thread
session rollout under ``$CODEX_HOME/sessions/<yyyy>/<mm>/<dd>/rollout-*-<tid>.jsonl``,
which records a ``token_count`` event after every LLM call with cumulative
``info.total_token_usage`` and per-call ``info.last_token_usage``.

This script joins those sources retroactively. For each agent under
``<run_dir>/agents/<id>/`` it:

  1. Reads ``codex_stdout.jsonl`` to collect every ``thread.started`` thread_id
     in order of first appearance.
  2. Locates the matching rollout file(s) under ``$CODEX_HOME`` (default
     ``~/.codex``) by recursive name match.
  3. Walks the rollout in order, treating each ``task_started`` event as a day
     boundary and snapshotting the cumulative ``total_token_usage`` between
     adjacent boundaries to compute per-day deltas.
  4. Writes ``<run_dir>/agents/<id>/token_usage.json`` plus a console summary.

Example:
  python analysis/codex_token_usage.py \\
    /fast/sgoel/forecasting/current_sim/final_runs_v37/codex_aljazeeraQ12026v37_gpt54_resume/26-04-29-03-09-57
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


USAGE_KEYS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "total_tokens",
)


def codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME") or (Path.home() / ".codex"))


def thread_ids_from_stdout(stdout_path: Path) -> list[str]:
    """Return thread_ids in first-seen order from a harness codex_stdout.jsonl."""
    seen: list[str] = []
    seen_set: set[str] = set()
    with stdout_path.open() as f:
        for line in f:
            if '"thread.started"' not in line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") != "thread.started":
                continue
            tid = d.get("thread_id")
            if tid and tid not in seen_set:
                seen.append(tid)
                seen_set.add(tid)
    return seen


def find_rollout(thread_id: str, sessions_root: Path) -> Path | None:
    """Find the rollout file for a thread_id under a Codex sessions root."""
    matches = list(sessions_root.rglob(f"rollout-*-{thread_id}.jsonl"))
    if not matches:
        return None
    # If multiple (shouldn't happen, but guard), pick the largest by mtime.
    matches.sort(key=lambda p: p.stat().st_mtime)
    return matches[-1]


def _zero_usage() -> dict:
    return {k: 0 for k in USAGE_KEYS}


def _diff(a: dict, b: dict) -> dict:
    """Return a - b for the standard usage keys, treating missing as 0."""
    return {k: int(a.get(k, 0)) - int(b.get(k, 0)) for k in USAGE_KEYS}


def parse_rollout(rollout_path: Path) -> dict:
    """Parse a Codex rollout file and return a token-usage report.

    Returns a dict with:
      - first_event_ts / last_event_ts
      - token_count_events (count, including the no-info bootstrap event)
      - task_started_count
      - context_compacted_count
      - total_token_usage (final cumulative)
      - last_token_usage (very last per-call usage)
      - per_day: list of {day_index, turn_id, started_at_iso, started_at_epoch,
                          delta, cumulative_after}
    """
    first_ts: str | None = None
    last_ts: str | None = None
    token_count_events = 0
    task_started_count = 0
    context_compacted_count = 0

    final_total: dict | None = None
    final_last: dict | None = None

    # Day-boundary tracking. Each entry is a task_started event with the
    # cumulative usage observed just before it (i.e. the previous day's tail).
    days: list[dict] = []  # incrementally populated

    # Latest cumulative seen so far (used to snapshot at day boundaries and
    # to compute per-day deltas).
    cumulative: dict = _zero_usage()

    with rollout_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue

            ts = d.get("timestamp")
            if ts:
                if first_ts is None:
                    first_ts = ts
                last_ts = ts

            if d.get("type") != "event_msg":
                continue
            payload = d.get("payload") or {}
            ptype = payload.get("type")

            if ptype == "task_started":
                task_started_count += 1
                days.append(
                    {
                        "day_index": len(days),
                        "turn_id": payload.get("turn_id"),
                        "started_at_iso": ts,
                        "started_at_epoch": payload.get("started_at"),
                        "cumulative_before": dict(cumulative),
                        # `cumulative_after` and `delta` filled when next
                        # task_started arrives (or at end of file).
                    }
                )
            elif ptype == "token_count":
                token_count_events += 1
                info = payload.get("info") or {}
                tot = info.get("total_token_usage")
                if tot:
                    cumulative = {k: int(tot.get(k, 0)) for k in USAGE_KEYS}
                    final_total = dict(cumulative)
                last = info.get("last_token_usage")
                if last:
                    final_last = {k: int(last.get(k, 0)) for k in USAGE_KEYS}
            elif ptype == "context_compacted":
                context_compacted_count += 1

    # Close per-day deltas: each day's "cumulative_after" is the next day's
    # "cumulative_before"; the last day's is the final cumulative.
    for i, day in enumerate(days):
        after = days[i + 1]["cumulative_before"] if i + 1 < len(days) else dict(cumulative)
        day["cumulative_after"] = after
        day["delta"] = _diff(after, day["cumulative_before"])
        # Drop the now-redundant before snapshot to keep output small.
        del day["cumulative_before"]

    return {
        "rollout_path": str(rollout_path),
        "first_event_ts": first_ts,
        "last_event_ts": last_ts,
        "token_count_events": token_count_events,
        "task_started_count": task_started_count,
        "context_compacted_count": context_compacted_count,
        "total_token_usage": final_total or _zero_usage(),
        "last_token_usage": final_last or _zero_usage(),
        "per_day": days,
    }


def merge_reports(reports: list[dict]) -> dict:
    """Sum cumulative usage across multiple threads and concatenate per_day."""
    if len(reports) == 1:
        return reports[0]
    total = _zero_usage()
    per_day: list[dict] = []
    for r in reports:
        for k in USAGE_KEYS:
            total[k] += int(r["total_token_usage"].get(k, 0))
        per_day.extend(r.get("per_day", []))
    return {
        "rollouts": [r["rollout_path"] for r in reports],
        "first_event_ts": min((r["first_event_ts"] for r in reports if r["first_event_ts"]), default=None),
        "last_event_ts": max((r["last_event_ts"] for r in reports if r["last_event_ts"]), default=None),
        "token_count_events": sum(r["token_count_events"] for r in reports),
        "task_started_count": sum(r["task_started_count"] for r in reports),
        "context_compacted_count": sum(r["context_compacted_count"] for r in reports),
        "total_token_usage": total,
        "last_token_usage": reports[-1]["last_token_usage"],
        "per_day": per_day,
    }


def process_agent(agent_dir: Path, sessions_root: Path) -> dict | None:
    stdout = agent_dir / "codex_stdout.jsonl"
    if not stdout.exists():
        return None
    tids = thread_ids_from_stdout(stdout)
    if not tids:
        print(f"  [skip] no thread.started events in {stdout}", file=sys.stderr)
        return None

    reports: list[dict] = []
    missing: list[str] = []
    for tid in tids:
        rollout = find_rollout(tid, sessions_root)
        if rollout is None:
            missing.append(tid)
            continue
        reports.append(parse_rollout(rollout))

    if missing:
        print(
            f"  [warn] {len(missing)} thread_id(s) had no rollout under {sessions_root}: "
            + ", ".join(missing),
            file=sys.stderr,
        )
    if not reports:
        return None

    merged = merge_reports(reports)
    merged["agent_id"] = agent_dir.name
    merged["thread_ids"] = tids
    merged["missing_thread_ids"] = missing
    return merged


def fmt_int(n: int) -> str:
    return f"{n:,}"


def print_summary(report: dict) -> None:
    a = report.get("agent_id", "<agent>")
    tot = report["total_token_usage"]
    cached = tot.get("cached_input_tokens", 0)
    inp = tot.get("input_tokens", 0)
    cache_pct = (100.0 * cached / inp) if inp else 0.0
    print(f"  agent={a}")
    print(f"    days (task_started events): {report['task_started_count']}")
    print(f"    token_count events: {fmt_int(report['token_count_events'])}")
    print(f"    input_tokens:     {fmt_int(inp)}  (cached {fmt_int(cached)} / {cache_pct:.1f}%)")
    print(f"    output_tokens:    {fmt_int(tot.get('output_tokens', 0))}")
    print(f"    reasoning_tokens: {fmt_int(tot.get('reasoning_output_tokens', 0))}")
    print(f"    total_tokens:     {fmt_int(tot.get('total_tokens', 0))}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("run_dir", type=Path, help="Run directory (contains agents/<id>/codex_stdout.jsonl)")
    ap.add_argument(
        "--codex-home",
        type=Path,
        default=None,
        help="Override CODEX_HOME (default: env CODEX_HOME or ~/.codex)",
    )
    ap.add_argument(
        "--out-name",
        default="token_usage.json",
        help="Output filename written under each agent dir (default: token_usage.json)",
    )
    ap.add_argument(
        "--stdout-only",
        action="store_true",
        help="Print summary to stdout but do not write per-agent JSON files",
    )
    args = ap.parse_args()

    sessions_root = (args.codex_home or codex_home()) / "sessions"
    if not sessions_root.exists():
        print(f"Codex sessions root does not exist: {sessions_root}", file=sys.stderr)
        return 2

    agents_root = args.run_dir / "agents"
    if not agents_root.exists():
        print(f"No agents/ subdir in {args.run_dir}", file=sys.stderr)
        return 2

    print(f"run_dir: {args.run_dir}")
    print(f"sessions_root: {sessions_root}")

    any_written = False
    for agent_dir in sorted(p for p in agents_root.iterdir() if p.is_dir()):
        report = process_agent(agent_dir, sessions_root)
        if report is None:
            continue
        print_summary(report)
        if not args.stdout_only:
            out = agent_dir / args.out_name
            out.write_text(json.dumps(report, indent=2))
            print(f"    wrote {out}")
            any_written = True

    if not any_written and not args.stdout_only:
        print("No usage reports written.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
