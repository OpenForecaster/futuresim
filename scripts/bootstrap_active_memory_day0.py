"""Bootstrap an active_memory codex run from a prior gpt-5.5-resume run's day 0.

The active_memory prompt mode (agents/minimalHarnessAgent) expects the workspace
to contain `memory/{prev_date}/mem.csv` (per-question notes) and
`memory/{prev_date}/meta.yaml` (meta-insights) when the agent starts on
{prev_date+1}. The prior gpt-5.5 codex_resume run only stored day-0 reasoning
in a free-form `forecasting_notes.md`. This script converts that into the
structured layout active_memory expects.

Concretely it produces, in TARGET/agents/<agent_id>/workspace/memory/<day0>/:
  - mem.csv : one row per qid with `memory` = compact "Initial forecast: {dist}"
              string derived from the day-0 predictions JSON, plus the
              question text from the day-0 market.csv.
  - meta.yaml : ~5 entries derived from the "<day0> initial strategy" bullets
                in forecasting_notes.md. Names are lowercase-hyphens-numbers.

Usage:
  python scripts/bootstrap_active_memory_day0.py \\
      --source /fast/.../codex_aljazeeraQ12026v37_gpt55_resume/<ts>/agents/minimalHarness_gpt-55_001 \\
      --target <new_run_dir>/agents/<new_agent_id> \\
      --day0 2025-12-24
"""

import argparse
import json
import re
from datetime import date
from pathlib import Path

import pandas as pd
import yaml


def _format_distribution(outcomes: dict, max_outcomes: int = 5) -> str:
    items = sorted(outcomes.items(), key=lambda kv: (-kv[1], kv[0]))[:max_outcomes]
    return ", ".join(f"{k}: {v:.2f}" for k, v in items)


def build_mem_csv(predictions_path: Path, market_csv_path: Path, day0: str) -> pd.DataFrame:
    """One row per qid with day-0 forecast distribution as the memory field."""
    with open(predictions_path) as f:
        preds = json.load(f)
    market = pd.read_csv(market_csv_path, dtype={"qid": str})
    title_by_qid = dict(zip(market["qid"].astype(str), market["title"].astype(str)))

    rows = []
    for p in preds:
        qid = str(p["question_id"])
        dist = _format_distribution(p["outcomes"])
        rows.append({
            "qid": qid,
            "question": title_by_qid.get(qid, ""),
            "last_updated": day0,
            "memory": f"Initial forecast: {dist}",
            "category": "",
        })
    df = pd.DataFrame(rows, columns=["qid", "question", "last_updated", "memory", "category"])
    return df


# Hand-curated meta-insights derived from the gpt-5.5 day-0 strategy bullets.
# Field constraints (ACTIVE_META_FIELD_LIMITS): name <= 32 chars, description
# <= 100 chars, content <= 400 chars. Names: lowercase-hyphens-numbers only.
META_ENTRIES = [
    {
        "name": "initial-distribution-strategy",
        "description": "How to budget probability mass for long-horizon string questions on day 0",
        "content": (
            "For long-horizon string-answer questions, spread 0.03-0.20 total mass across a few "
            "plausible answers rather than concentrating. Confident wrong exact guesses are "
            "heavily penalized."
        ),
    },
    {
        "name": "tight-constraint-anchors",
        "description": "Cases where day-0 evidence already pins down a likely answer",
        "content": (
            "Concentrate mass when the prompt/context tightly constrains the answer (e.g. Kerala "
            "for Nipah, Sri Lanka for Bangladesh T20 relocation, Oman/Muscat for US-Iran talks, "
            "Novichok for Navalny toxin, Canada for Winter Olympics hockey/medal, named actors "
            "like Donald Trump, Michael Carrick, Masayoshi Son, ADNOC)."
        ),
    },
    {
        "name": "string-scoring-discipline",
        "description": "When to escalate confidence on exact-string predictions",
        "content": (
            "Only raise probabilities sharply when articles or official schedules directly "
            "identify the answer. String scoring punishes confident wrong exact guesses."
        ),
    },
    {
        "name": "daily-workflow",
        "description": "Standard order of operations each session",
        "content": (
            "1) Check new articles in articles/. 2) Search near-resolution questions first. "
            "3) Update exact strings with higher probabilities when found. 4) Call "
            "mcp__forecast__next_day to advance."
        ),
    },
    {
        "name": "day0-coverage",
        "description": "Coverage achieved on the bootstrap day",
        "content": (
            "Day 0 (2025-12-24) submitted forecasts for all 331 active questions. mem.csv carries "
            "the per-qid initial distributions; later days should update only when new evidence "
            "appears, per UPDATE RULES."
        ),
    },
]


def build_meta_yaml(day0: str) -> str:
    entries = [{
        "name": e["name"],
        "description": e["description"],
        "content": e["content"],
        "added": day0,
    } for e in META_ENTRIES]
    return yaml.safe_dump(entries, default_flow_style=False, allow_unicode=True)


def _validate_meta_entries(entries: list) -> None:
    for e in entries:
        n = e["name"]
        assert re.fullmatch(r"[a-z0-9-]+", n), f"bad name: {n!r}"
        assert len(n) <= 32, f"name too long ({len(n)}): {n!r}"
        assert len(e["description"]) <= 100, f"desc too long ({len(e['description'])}): {n!r}"
        assert len(e["content"]) <= 400, f"content too long ({len(e['content'])}): {n!r}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Source agent dir (e.g. .../agents/minimalHarness_gpt-55_001)")
    ap.add_argument("--target", required=True, help="Target agent dir for the new active_memory run")
    ap.add_argument("--day0", required=True, help="Bootstrap day (YYYY-MM-DD), typically sim_start - 1")
    args = ap.parse_args()

    src = Path(args.source)
    tgt = Path(args.target)
    day0 = args.day0
    date.fromisoformat(day0)  # validate

    predictions_path = src / "predictions" / f"{day0}.json"
    market_csv_path = src / "workspace" / "market.csv"
    notes_path = src / "workspace" / "memory" / "forecasting_notes.md"

    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)
    if not market_csv_path.exists():
        raise FileNotFoundError(market_csv_path)
    if not notes_path.exists():
        raise FileNotFoundError(notes_path)

    _validate_meta_entries(META_ENTRIES)

    out_dir = tgt / "workspace" / "memory" / day0
    out_dir.mkdir(parents=True, exist_ok=True)

    mem_df = build_mem_csv(predictions_path, market_csv_path, day0)
    mem_csv_path = out_dir / "mem.csv"
    mem_df.to_csv(mem_csv_path, index=False)

    meta_yaml_text = build_meta_yaml(day0)
    meta_yaml_path = out_dir / "meta.yaml"
    meta_yaml_path.write_text(meta_yaml_text)

    print(f"Wrote {mem_csv_path}  ({len(mem_df)} rows)")
    print(f"Wrote {meta_yaml_path}  ({len(META_ENTRIES)} entries)")


if __name__ == "__main__":
    main()
