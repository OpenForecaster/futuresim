"""Render the day-0 system prompt for configs/minimalHarness/aljazeera_sept_opus.yaml.

The MinimalHarness agent writes this once to <workspace>/../system_prompt.md
at simulation start; it does not rebuild it per day because Claude Code keeps
a persistent session across wakeups.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agents.minimalHarnessAgent.prompts.prompt import build_system_prompt  # noqa: E402


def main() -> None:
    workspace = "/fast/nchandak/sims/claude_code_aljazeera_sept_opus/r00/agents/cc_claude-opus-4-6_001/workspace"
    prompt = build_system_prompt(
        workspace=workspace,
        current_date=date(2025, 9, 1),
        start_date=date(2025, 9, 1),
        end_date=date(2025, 9, 30),
        source_context="",
        source_name="openforesight",
        num_questions=108,
        num_active=108,
        num_resolved=0,
        max_outcomes_per_question=5,
        search_cutoff_days=0,
        timegap_days=1,
        new_articles_count=None,
    )

    out = REPO_ROOT / "minimal_harness_day0_prompt.txt"
    out.write_text(prompt)
    print(f"Wrote {len(prompt):,} chars to {out}")


if __name__ == "__main__":
    main()
