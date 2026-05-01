"""All-in-one bootstrap for an active_memory codex run from a prior gpt-5.5 run.

Creates a ready-to-resume run directory pre-populated with:
  - actions.jsonl   : filtered to sim_date < restart_from_day, agent_id remapped
  - matcher_cache.json (copied)
  - daily_metrics.csv (copied, filtered)
  - agents/<new_id>/workspace/memory/<day0>/{mem.csv, meta.yaml}
  - source_config.json (copy of source's config.json for traceability)

After running this, launch the simulation with:
    python scripts/test_basic_agent.py --config <yaml> --resume <prepared_dir>

The yaml's `agents`, `defaults`, etc. provide the new run's config; the env's
restore_state(args.resume) replays actions.jsonl to rebuild prediction_histories
and fast-forwards env.current_date to last_date + timegap_days.
"""

import argparse
import json
import os
import shutil
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml

from scripts.bootstrap_active_memory_day0 import (
    META_ENTRIES,
    _validate_meta_entries,
    build_mem_csv,
    build_meta_yaml,
)


def _new_agent_id(model: str, idx: int = 1) -> str:
    """Replicate test_basic_agent.create_agents_from_config naming for minimalHarness."""
    return f"minimalHarness_{model.replace('/', '_').replace('.', '')}_{idx:03d}"


def _filter_and_remap_actions(
    src: Path, dst: Path, restart_day: date, source_agent_id: str, target_agent_id: str
) -> tuple[int, int]:
    """Copy src→dst, keeping only sim_date < restart_day and remapping prediction agent_ids."""
    kept = 0
    remapped = 0
    with open(src) as f, open(dst, "w") as g:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            d_str = rec.get("sim_date")
            if not d_str:
                continue
            try:
                d = date.fromisoformat(d_str)
            except ValueError:
                continue
            if d >= restart_day:
                continue
            if rec.get("type") == "prediction" and rec.get("agent_id") == source_agent_id:
                rec["agent_id"] = target_agent_id
                remapped += 1
            g.write(json.dumps(rec) + "\n")
            kept += 1
    return kept, remapped


def _filter_daily_metrics(src: Path, dst: Path, restart_day: date) -> int:
    copied = 0
    with open(src) as f, open(dst, "w") as g:
        for line in f:
            line = line.strip()
            if not line:
                continue
            first = line.split(",", 1)[0]
            if first.lower() == "date":
                g.write(line + "\n")
                continue
            try:
                d = date.fromisoformat(first)
            except ValueError:
                continue
            if d < restart_day:
                g.write(line + "\n")
                copied += 1
    return copied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True, help="Target run YAML config (defines sim_name, output_base, agents, etc.)")
    ap.add_argument("--source-run", required=True, help="Source run directory (e.g. .../codex_aljazeeraQ12026v37_gpt55_resume/26-04-29-02-49-22)")
    ap.add_argument("--source-agent-id", default="minimalHarness_gpt-55_001",
                    help="Source agent_id whose predictions will be carried forward and remapped")
    ap.add_argument("--day0", required=True, help="Bootstrap day (YYYY-MM-DD); typically sim_start - 1 = first sim day")
    ap.add_argument("--restart-from-day", required=True, help="First day the new run will simulate (typically day0 + timegap)")
    ap.add_argument("--target-agent-idx", type=int, default=1, help="Index used for new agent_id (default: 1)")
    ap.add_argument("--meta-source", default=None,
                    help="Optional YAML file with custom META_ENTRIES (list of {name, description, content}). "
                         "Defaults to the built-in gpt-5.5-derived entries.")
    args = ap.parse_args()

    # Parse / validate dates.
    day0 = date.fromisoformat(args.day0)
    restart_day = date.fromisoformat(args.restart_from_day)
    if restart_day <= day0:
        raise SystemExit(f"--restart-from-day ({restart_day}) must be > --day0 ({day0})")

    yaml_path = Path(args.yaml)
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    sim_name = cfg["sim_name"]
    output_base = cfg["output_base"]

    agents_list = cfg.get("agents", [])
    if not agents_list:
        raise SystemExit("YAML missing `agents` list")
    target_model = agents_list[args.target_agent_idx - 1]["model"]
    target_agent_id = _new_agent_id(target_model, args.target_agent_idx)

    # Source paths.
    src_run = Path(args.source_run)
    src_actions = src_run / "actions.jsonl"
    src_matcher_cache = src_run / "matcher_cache.json"
    src_daily_metrics = src_run / "daily_metrics.csv"
    src_config = src_run / "config.json"
    src_agent_dir = src_run / "agents" / args.source_agent_id
    src_predictions = src_agent_dir / "predictions" / f"{day0.isoformat()}.json"
    src_market = src_agent_dir / "workspace" / "market.csv"

    for p in (src_actions, src_predictions, src_market):
        if not p.exists():
            raise SystemExit(f"Required source file missing: {p}")

    # Resolve meta entries (custom or default), and validate before doing
    # anything destructive.
    if args.meta_source:
        meta_path = Path(args.meta_source)
        if not meta_path.exists():
            raise SystemExit(f"--meta-source not found: {meta_path}")
        with open(meta_path) as f:
            meta_entries = yaml.safe_load(f)
        if not isinstance(meta_entries, list) or not meta_entries:
            raise SystemExit(f"--meta-source must contain a non-empty list (got {type(meta_entries).__name__})")
    else:
        meta_entries = META_ENTRIES
    _validate_meta_entries(meta_entries)

    # Create new run dir.
    timestamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    out_dir = Path(output_base) / sim_name / timestamp
    out_dir.mkdir(parents=True, exist_ok=False)
    print(f"[1/5] Created run dir: {out_dir}")

    # Filter + remap actions.jsonl.
    dst_actions = out_dir / "actions.jsonl"
    kept, remapped = _filter_and_remap_actions(
        src_actions, dst_actions, restart_day, args.source_agent_id, target_agent_id
    )
    print(f"[2/5] actions.jsonl: kept {kept} records (sim_date < {restart_day}); remapped {remapped} predictions to agent_id={target_agent_id}")

    # Copy matcher cache.
    if src_matcher_cache.exists():
        shutil.copy(src_matcher_cache, out_dir / "matcher_cache.json")
        print(f"[3/5] Copied matcher_cache.json")
    else:
        print(f"[3/5] No matcher_cache.json in source; skipping")

    # Filter daily_metrics.csv (likely empty since no resolutions yet, but be consistent).
    if src_daily_metrics.exists():
        copied = _filter_daily_metrics(src_daily_metrics, out_dir / "daily_metrics.csv", restart_day)
        print(f"    Copied {copied} daily_metrics rows (date < {restart_day})")

    # Copy source config.json for traceability.
    if src_config.exists():
        shutil.copy(src_config, out_dir / "source_config.json")

    # Build workspace memory for day0 under the new agent_id.
    agent_workspace_memory = out_dir / "agents" / target_agent_id / "workspace" / "memory" / day0.isoformat()
    agent_workspace_memory.mkdir(parents=True, exist_ok=True)

    mem_df = build_mem_csv(src_predictions, src_market, day0.isoformat())
    mem_csv_path = agent_workspace_memory / "mem.csv"
    mem_df.to_csv(mem_csv_path, index=False)
    print(f"[4/5] Wrote {mem_csv_path}  ({len(mem_df)} rows)")

    meta_yaml_text = build_meta_yaml(day0.isoformat(), entries=meta_entries)
    meta_yaml_path = agent_workspace_memory / "meta.yaml"
    meta_yaml_path.write_text(meta_yaml_text)
    print(f"    Wrote {meta_yaml_path}  ({len(meta_entries)} entries)")

    # Save provenance for the bootstrap itself.
    provenance = {
        "bootstrap_timestamp": datetime.now().isoformat(),
        "yaml": str(yaml_path.resolve()),
        "source_run": str(src_run.resolve()),
        "source_agent_id": args.source_agent_id,
        "target_agent_id": target_agent_id,
        "day0": day0.isoformat(),
        "restart_from_day": restart_day.isoformat(),
        "actions_kept": kept,
        "predictions_remapped": remapped,
    }
    with open(out_dir / "bootstrap_provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"[5/5] Wrote bootstrap_provenance.json")

    print()
    print(f"Run dir ready: {out_dir}")
    print()
    print("Next step:")
    print(f"  python scripts/test_basic_agent.py --config {yaml_path} --resume {out_dir}")


if __name__ == "__main__":
    main()
