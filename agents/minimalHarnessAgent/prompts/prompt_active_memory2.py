"""Daily user prompt builder for the `active_memory2` prompt mode.

Mirrors AllQAgent / BasicAgent's ActiveMemory daily prompt as closely as
possible. The only differences are surgical swaps required because
active_memory2 has no `query_df` sandbox and runs against the minimalHarness
backends:

- `df` / `query_df` references swap to reading `market.csv`.
- Inspecting prior `mem_df` swaps from the sandbox to reading the prior day's
  `memory/{prev}/mem.csv` file directly (read-only).
- All MCP tool names are prefixed with `mcp__forecast__`.
- The end-of-day memory update phase is dropped: the MCP server persists
  today's `mem.csv` + `meta.yaml` automatically when the harness calls
  `mcp__forecast__next_day`. Agents must not write under `memory/`.
- INTERACTION FLOW drops the budget overview (no token budget here) and
  swaps "queries" → "reads".

Everything else (opening line, feedback, source_context, source_rules,
cadence, scoring, AllQ "already-predicted" reminder, submission rules,
tip line, "Begin") matches the AllQ flow.
"""

from datetime import date, timedelta
from typing import Optional

from futuresim_agents.minimalHarnessAgent.prompts.prompt_active_memory import (
    _allq_reminder,
    _normalize_prompt_heading_spacing,
)
from futuresim_agents.minimalHarnessAgent.prompts.prompt import (
    _build_new_articles_text,
    _get_data_notes,
    _get_scoring_section,
    _get_source_rules,
    _iso,
    _search_results_description,
)


# ── UPDATE CADENCE section — AllQ wording projected onto market.csv ────

def _build_allq_cadence_section(
    current_date: date,
    start_date: date,
    end_date: date,
    timegap_days: int,
    *,
    new_articles_count: Optional[int] = None,
    last_active_date: Optional[date] = None,
    next_active_date: Optional[date] = None,
    imminent_qids: Optional[list] = None,
) -> str:
    day_unit = "day" if timegap_days == 1 else "days"
    next_text = (
        f"Next scheduled update: {_iso(next_active_date)}."
        if next_active_date
        else "No later updates are scheduled."
    )
    last_text = (
        f"Last update: {_iso(last_active_date)}. "
        if last_active_date
        else "This is your first update. "
    )
    articles_text = _build_new_articles_text(new_articles_count) if last_active_date else ""

    tomorrow_iso = _iso(current_date + timedelta(days=1))
    if imminent_qids:
        imminent_reminder = (
            f"**IMPORTANT**: {len(imminent_qids)} question(s) resolve tomorrow ({tomorrow_iso}): "
            f"{list(imminent_qids)}. Make sure your prediction on each is up-to-date before calling next_day — "
            "stale forecasts might hurt your performance."
        )
    else:
        imminent_reminder = (
            f"No questions resolve tomorrow ({tomorrow_iso}), but still scan for ones "
            "resolving soon (check the resolution_date column in market.csv)."
        )

    return (
        "## UPDATE CADENCE\n"
        f"You can make updates every {timegap_days} {day_unit}. Your context is cleared after every session "
        "and your memory (along with past predictions) is the only information retained between sessions. "
        f"{articles_text}{last_text}Current date: {_iso(current_date)}. {next_text}\n"
        f"{imminent_reminder}\n\n"
    )


# ── YOUR MEMORY section — mirrors AllQ's ActiveMemory section ──────────

def _memory_section(
    *,
    last_active_date: Optional[date],
    current_date: date,
    meta_index: str,
) -> str:
    meta_block = (
        f"Current meta-insights with their indices:\n{meta_index}\n\n"
        if meta_index
        else ""
    )
    if last_active_date is not None:
        prev_iso = _iso(last_active_date)
        prior_files_line = (
            f"Your prior `mem_df` is saved at `memory/{prev_iso}/mem.csv` (read-only). "
            "Read it early with the Read tool to recall your prior reasoning."
        )
        inspect_line = "Inspect `mem_df` by reading that file."
    else:
        prior_files_line = (
            "No prior memory exists yet — this is your first session, so the memory store is empty."
        )
        inspect_line = "Inspect `mem_df` by reading `memory/<prev-date>/mem.csv` once it exists."
    return f"""## YOUR MEMORY
{meta_block}`mem_df` holds your per-question notes (reasoning, evidence, calibration) — 1 row per question.
Columns: qid (str), question (str), last_updated (str), memory (str), category (str)
{prior_files_line}

{inspect_line} Edit per-question notes with `mcp__forecast__mem_add`, `mcp__forecast__mem_update`, `mcp__forecast__mem_delete`.
Manage meta-insights with `mcp__forecast__memory_retrieve` (using the indices), `mcp__forecast__memory_new`, `mcp__forecast__memory_update`, `mcp__forecast__memory_delete`.
Caps (enforced by silent truncation): meta-insights ≤ 500 entries (name ≤ 64, description ≤ 256, content ≤ 400 chars; oldest dropped if over 500); per-question `mem_df` memory ≤ 1000 chars per row.
"""


def _memory_workflow_section() -> str:
    """AllQ-style guidance for when to use mem_df vs meta-insights."""
    return """## MEMORY WORKFLOW
Treat memory as two layers:
- `mem_df`: question-specific reasoning, evidence, and calibration notes for a single QID.
- meta-insights: reusable cross-question patterns, lessons, and calibration rules that should help on future days.

Before calling `mcp__forecast__next_day()`:
1. Update `mem_df` for questions you researched or forecasted today using `mcp__forecast__mem_add` / `mcp__forecast__mem_update`.
2. If today's work revealed a reusable pattern, lesson, or calibration rule that applies across multiple questions or future days, promote it into a meta-insight with `mcp__forecast__memory_new` or `mcp__forecast__memory_update`.
3. If a prior meta-insight is now stale or contradicted, revise it with `mcp__forecast__memory_update` or remove it with `mcp__forecast__memory_delete`.

Good meta-insights are compact and reusable, for example:
- a pattern across several questions
- a domain-specific calibration rule
- a lesson from a resolved question that should change future forecasting behavior

Do not use meta-insights as a daily activity log. If you learned nothing reusable today, it's fine to skip meta-insight writes for that day.
"""


def build_memory_update_prompt(
    *,
    current_date: date,
    mem_summary: str,
    meta_index: str,
    mem_count: int,
    meta_count: int,
    max_meta_entries: int,
    resolution_recap: str = "",
    touched_qids: Optional[list[str]] = None,
) -> str:
    """Build the AllQ-style memory-update prompt shown after first next_day()."""
    index_block = f"Current meta-insight index:\n{meta_index}\n" if meta_index else ""
    recap_block = f"{resolution_recap}\n\n" if resolution_recap else ""
    touched_block = (
        f"Questions you interacted with today: {touched_qids}\n\n"
        if touched_qids else
        "No question-specific memory or forecast submissions were recorded yet this session.\n\n"
    )
    return _normalize_prompt_heading_spacing(
        f"""End of session {current_date}. Update your memory now.

## MEMORY UPDATE
{recap_block}{touched_block}Your forecasting work for today is done. Do not search or submit more forecasts in this phase.
Use only memory tools plus a final `mcp__forecast__next_day()` when your updates are complete.

## Layer 1: QUESTION-SPECIFIC NOTES (`mem_df`: {mem_count} entries)

Current entries:
{mem_summary}

Use `mcp__forecast__mem_add`, `mcp__forecast__mem_update`, and `mcp__forecast__mem_delete`
for per-question reasoning, evidence, and calibration notes.

## Layer 2: META-INSIGHTS ({meta_count}/{max_meta_entries} entries)

Use this for reusable cross-question patterns, lessons, and calibration rules.
{index_block}
### STEP 1: Extract lessons from resolved questions
If any questions resolved since your last session, create or update meta-insight lesson entries
capturing what happened, why you were right or wrong, and the reusable rule you want future-you to apply.

### STEP 2: Update question-specific notes
For questions you researched or forecasted today, especially the QIDs listed above, update `mem_df` with the current reasoning and key evidence.

### STEP 3: Promote reusable patterns
If today's work revealed a pattern, calibration rule, or cross-question lesson that should matter later,
promote it into a meta-insight with `mcp__forecast__memory_new` or `mcp__forecast__memory_update`.

### STEP 4: Cleanup
Delete stale per-question notes and stale meta-insights that future-you should no longer rely on.

When your memory updates are complete, call `mcp__forecast__next_day()` again to actually advance the simulation.
"""
    )


# ── Main builder ───────────────────────────────────────────────────────

def build_daily_prompt(
    *,
    current_date: date,
    start_date: date,
    end_date: date,
    last_active_date: Optional[date],
    next_active_date: Optional[date],
    source_context: str,
    source_name: str,
    feedback_text: str,
    meta_index: str,
    num_questions: int,
    num_active: int,
    num_resolved: int,
    predicted_count: int,
    active_count: int,
    max_outcomes_per_question: int,
    search_cutoff_days: int,
    timegap_days: int,
    new_articles_count: Optional[int] = None,
    imminent_qids: Optional[list] = None,
) -> str:
    """Build the per-day user prompt for the active_memory2 mode."""
    source_rules = _get_source_rules(source_name)
    scoring_section = _get_scoring_section(
        source_name, max_outcomes_per_question, handholding_version="v3"
    )
    cadence_section = _build_allq_cadence_section(
        current_date,
        start_date,
        end_date,
        timegap_days,
        new_articles_count=new_articles_count,
        last_active_date=last_active_date,
        next_active_date=next_active_date,
        imminent_qids=imminent_qids,
    )
    memory_section = _memory_section(
        last_active_date=last_active_date,
        current_date=current_date,
        meta_index=meta_index,
    )
    reminder = _allq_reminder(predicted_count, active_count)
    data_notes = _get_data_notes()

    search_results_desc = _search_results_description()
    search_advice = (
        "You have access to a news article database which is updated **daily** "
        "through a search tool, that you can use to find evidence for your forecasts."
    )
    cutoff_desc = "today's date"
    if search_cutoff_days > 0:
        cutoff_date = current_date - timedelta(days=search_cutoff_days)
        cutoff_desc = f"{_iso(cutoff_date)} (today - {search_cutoff_days} days)"
    search_tool_line = (
        f"- `mcp__forecast__search_news(query, from_date?, to_date?)`: search the news corpus for evidence. "
        f"`to_date` is capped at {cutoff_desc}. {search_results_desc}\n"
    )

    intro_sections = [
        f"You are a forecasting agent. Today is {current_date}. Your goal is to make accurate and calibrated predictions.",
        feedback_text.strip(),
        source_context.strip(),
        source_rules.strip(),
        f"{cadence_section}{reminder}\n\n{memory_section}{scoring_section}".strip(),
    ]
    intro_block = "\n\n".join(s for s in intro_sections if s)

    available_data_section = "\n".join([
        "## AVAILABLE DATA",
        search_advice,
        f"You also have access to a read-only `market.csv` file in your workspace with {num_questions} questions ({num_active} active/unresolved, {num_resolved} resolved).",
        "",
        "Column descriptions of market.csv:",
        "- qid (str) (Question ID)",
        "- title (str) (Question Content)",
        "- background (object)",
        "- resolution_criteria (object)",
        "- answer_type (object)",
        "- resolution_date (object)",
        "- is_resolved (bool)",
        "- ground_truth (object)",
        "- num_predictions (int64)",
        "- options (object)",
        "- my_prediction (object)",
        "- my_prediction_date (object)",
        "",
        data_notes,
    ])

    tools_section = (
        "## TOOLS AVAILABLE FOR YOUR USE\n"
        f"{search_tool_line}"
        "- `mcp__forecast__memory_retrieve` / `mcp__forecast__memory_new` / `mcp__forecast__memory_update` / `mcp__forecast__memory_delete`: manage meta-insight entries.\n"
        "- `mcp__forecast__mem_add` / `mcp__forecast__mem_update` / `mcp__forecast__mem_delete`: manage question-specific notes in `mem_df`.\n"
        "- `mcp__forecast__submit_forecasts(question_id, outcomes)`: submit exactly one forecast for exactly one question ID (`qid`).\n"
        "- `mcp__forecast__next_day()`: first call enters memory-update mode; call it a second time after your memory updates to actually proceed to the next day.\n"
        "You also have access to native tools Bash/Read/Grep etc. — use them to read market.csv and browse articles/. The MCP server persists today's `mem.csv` + `meta.yaml` automatically on `mcp__forecast__next_day`; do not write under `memory/` yourself."
    )

    workspace_section = (
        "## Workspace:\n"
        "- market.csv — Read-only snapshot of all questions (refreshed each day).\n"
        "- articles/ — Browsable news articles organized by date as articles/YYYY/MM/DD/articles.jsonl (one JSON article per line). New date directories appear after calling `mcp__forecast__next_day`.\n"
        "- memory/ — Read-only persisted notes (`memory/YYYY-MM-DD/{mem.csv, meta.yaml}`), written by the MCP server on each `mcp__forecast__next_day`. Read prior days' files for context; edit memory only through the `mcp__forecast__mem_*` / `mcp__forecast__memory_*` tools."
    )

    if last_active_date is not None:
        interaction_flow = (
            "## INTERACTION FLOW\n"
            "You can interleave reads, searches, memory operations, and submissions as needed. "
            f"Consider reading `memory/{_iso(last_active_date)}/mem.csv` early to recall prior reasoning and identify which questions need attention."
        )
    else:
        interaction_flow = (
            "## INTERACTION FLOW\n"
            "You can interleave reads, searches, memory operations, and submissions as needed."
        )

    submission_rules = (
        "## SUBMISSION RULES\n"
        "- qid must be from an active (`is_resolved=False`) question you identified from market.csv\n"
        "- Each `mcp__forecast__submit_forecasts` call must contain exactly one forecast for one question ID (`qid`).\n"
        "- You may submit again later in the same session to update that `qid`.\n"
        f"- Maximum of {max_outcomes_per_question} outcomes allowed per question.\n"
        "- Outcome names must be REAL predicted answers (e.g. person names, locations, dates, etc.)\n"
        "- NEVER use placeholders like \"Unknown\", \"TBD\", \"Other\", or \"N/A\"\n"
        "- Probabilities must sum to <= 1.0"
    )

    tip_line = (
        "Tip: After submitting a forecast, consider saving your reasoning and key evidence for that QID using "
        "`mcp__forecast__mem_add`/`mcp__forecast__mem_update`. The MCP server persists `mem.csv` + `meta.yaml` "
        "automatically when you finish the memory-update phase and call `mcp__forecast__next_day()` again."
    )

    sections = [
        intro_block,
        available_data_section,
        tools_section,
        workspace_section,
        interaction_flow,
        _memory_workflow_section(),
        submission_rules,
        tip_line,
        "---\nBegin.",
    ]
    return _normalize_prompt_heading_spacing("\n\n".join(s for s in sections if s))
