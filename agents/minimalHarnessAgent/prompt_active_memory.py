"""Daily user prompt builder for the codex `active_memory` prompt mode.

Mirrors the BasicAgent/AllQAgent daily instructions as closely as possible.
Codex spawns fresh each day and reads/writes structured memory files via its
native Bash/Read/Write tools — so the surgical swaps vs. AllQAgent are:

- YOUR MEMORY tool-instruction paragraph points to memory/{prev}/{mem.csv,meta.yaml}
  on disk and asks codex to write memory/{today}/{mem.csv,meta.yaml} before next_day.
- AVAILABLE DATA refers to market.csv (read-only) instead of a DataFrame.
- CODE EXECUTION ENVIRONMENT block is dropped.
- TOOLS list drops query_df / mem_* / memory_*, prefixes search_news /
  submit_forecasts / next_day with mcp__forecast__, and notes that codex has
  native Bash/Read/Write/Edit/Grep etc.
- INTERACTION FLOW drops the budget overview and swaps "queries → reads".
- Tip line drops the second sentence about an end-of-day memory phase.
- End-of-prompt budget seed block is dropped.

Everything else (opening line, feedback, source_context, source_rules, cadence,
scoring, AllQ "already-predicted" reminder, submission rules, "Begin") is kept
verbatim from the BasicAgent/AllQAgent flow.
"""

import re
from datetime import date
from typing import Optional


def _normalize_prompt_heading_spacing(prompt: str) -> str:
    """Ensure `##` section headings are separated by two blank lines.

    Mirrors BasicAgent._normalize_prompt_heading_spacing so the active_memory
    daily prompt has the same heading spacing as AllQAgent's.
    """
    if not prompt:
        return prompt
    return re.sub(r"\n{1,}(?=## )", "\n\n\n", prompt)

from agents.minimalHarnessAgent.prompt import (
    _build_cadence_section,
    _get_data_notes,
    _get_scoring_section,
    _get_source_rules,
    _iso,
    _search_results_description,
)


# ── AllQAgent's already-predicted reminder, kept verbatim ──────────────

def _allq_reminder(predicted_count: int, active_count: int) -> str:
    return (
        "IMPORTANT: You have predictions on "
        f"{predicted_count} out of {active_count} active questions.\n"
        "Tip: You can check your existing predictions by reading market.csv "
        "and filtering rows where `my_prediction` is not null.\n\n"
        "UPDATE RULES:\n"
        "- Do NOT re-predict questions from scratch unless you find specific new evidence.\n"
        "- Only update a prediction if you find SPECIFIC NEW evidence (news, data) that updates your view.\n\n"
        "PRIORITIES FOR UPDATES:\n"
        "1. **Questions resolving the next day** (filter `market.csv` by `resolution_date` == tomorrow) — make sure your prediction is up-to-date before calling next_day.\n"
        "2. Questions without predictions (if any)\n"
        "3. Questions where today's news search reveals new information\n"
        "4. Questions approaching resolution date that you haven't checked recently\n"
        "5. Skip questions where there is no new evidence"
    )


# ── YOUR MEMORY section — surgical swap of AllQAgent's tool paragraph ──

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
            f"Your prior notes live at `memory/{prev_iso}/mem.csv` (per-question) "
            f"and `memory/{prev_iso}/meta.yaml` (meta-insights). Read them with the "
            "Read tool early in the session to recall your prior reasoning."
        )
    else:
        prior_files_line = (
            "No prior memory directory exists yet — you have no notes to load."
        )
    today_iso = _iso(current_date)
    return f"""## YOUR MEMORY
{meta_block}`mem.csv` holds your per-question notes (reasoning, evidence, calibration) — 1 row per question.
Columns: qid (str), question (str), last_updated (str), memory (str), category (str)
{prior_files_line}

Before calling `mcp__forecast__next_day`, write your updated notes to `memory/{today_iso}/mem.csv` and `memory/{today_iso}/meta.yaml`. Carry forward prior rows you still believe in; add/update rows for questions you worked on today; and keep the meta-insight YAML to at most 15 entries with content under ~400 chars each.
"""


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
    """Build the per-day user prompt for codex active_memory mode.

    Section order matches BasicAgent._build_instructions / AllQAgent._build_instructions:
      1. Opening line
      2. Feedback (resolutions + cumulative perf)
      3. Source context
      4. Source rules
      5. (Multi-agent context — single agent here, so empty)
      6. Cadence section
      7. AllQ "already-predicted" reminder
      8. Memory section
      9. Scoring section
     10. AVAILABLE DATA (search + market.csv + notes)  ← codex swap
     11. (CODE EXECUTION ENVIRONMENT — dropped)
     12. TOOLS AVAILABLE FOR YOUR USE                  ← codex swap
     13. INTERACTION FLOW (no budget)                  ← codex swap
     14. SUBMISSION RULES
     15. tip_line
     16. Begin
    """
    source_rules = _get_source_rules(source_name)
    # Active memory has its own handholding (`_allq_reminder`), and per design
    # is decoupled from the build_system_prompt `handholding_version` flag —
    # always render the maximal-handholding sections so its existing prompt
    # (including the "questions resolve tomorrow" reminder and TW-update nudge)
    # is preserved regardless of the user's handholding_version setting.
    scoring_section = _get_scoring_section(
        source_name, max_outcomes_per_question, handholding_version="v3"
    )
    cadence_section = _build_cadence_section(
        current_date,
        start_date,
        end_date,
        timegap_days,
        new_articles_count=new_articles_count,
        last_active_date=last_active_date,
        next_active_date=next_active_date,
        imminent_qids=imminent_qids,
        handholding_version="v3",
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
        from datetime import timedelta
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
        # Multi-agent context omitted (single-agent harness).
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
        "- `mcp__forecast__submit_forecasts(question_id, outcomes)`: submit exactly one forecast for exactly one question ID (`qid`).\n"
        "- `mcp__forecast__next_day()`: end the current session and proceed to the next one.\n"
        "You have access to native tools Bash/Read/Write/Edit/Grep etc. — use them to read market.csv, browse articles/, and read/write memory/."
    )

    workspace_section = (
        "## Workspace:\n"
        "- market.csv — Read-only snapshot of all questions (refreshed each day).\n"
        "- articles/ — Browsable news articles organized by date as articles/YYYY/MM/DD/articles.jsonl (one JSON article per line). New date directories appear after calling `mcp__forecast__next_day`.\n"
        "- predictions/ — Read-only record of your past submissions, one file per day as `predictions/YYYY-MM-DD.json`. Each file is a JSON list of `{\"question_id\": ..., \"outcomes\": {<outcome>: <prob>, ...}}` entries — the predictions you submitted that day. A new file appears after each `mcp__forecast__next_day`.\n"
        "- memory/ — Your structured per-day notes directory (`memory/YYYY-MM-DD/{mem.csv, meta.yaml}`). Read prior days' files at the start of each session and write today's files before calling `mcp__forecast__next_day`."
    )

    interaction_flow = (
        "## INTERACTION FLOW\n"
        "You can interleave reads, searches, memory edits, and submissions as needed. "
        f"Read `memory/{_iso(last_active_date)}/mem.csv` early to recall prior reasoning and identify which questions need attention. "
        if last_active_date is not None
        else
        "## INTERACTION FLOW\n"
        "You can interleave reads, searches, memory edits, and submissions as needed. "
    )
    interaction_flow += (
        f"Before calling `mcp__forecast__next_day`, write today's `memory/{_iso(current_date)}/mem.csv` and `memory/{_iso(current_date)}/meta.yaml`."
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
        "Tip: After submitting a forecast, consider saving your reasoning and key evidence "
        "for that QID by updating `mem.csv` (in-memory or on a scratch file) so the end-of-day write captures it."
    )

    sections = [
        intro_block,
        available_data_section,
        tools_section,
        workspace_section,
        interaction_flow,
        submission_rules,
        tip_line,
        "---\nBegin.",
    ]
    return _normalize_prompt_heading_spacing("\n\n".join(s for s in sections if s))
