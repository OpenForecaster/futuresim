"""System prompt builder for Claude Code forecasting agent.

Mirrors BasicAgent._build_instructions as closely as possible.
Sections that are structurally different for Claude Code (MCP tools instead of
chat-tool function calling, persistent session instead of context-per-day,
workspace files instead of structured memory, no budget system, no DataFrame
sandbox) are marked with "CC-SPECIFIC" comments.
"""

from datetime import date, timedelta
from typing import Optional


def _iso(d) -> str:
    return d.isoformat() if hasattr(d, "isoformat") else str(d)


# ── CC-SPECIFIC: workflow and handholding ──────────────────────────────

WORKFLOW_BASIC = """\
1. Read market.csv to understand active questions (is_resolved == False).
2. Research using `mcp__forecast__search_news` and direct file browsing in articles/.
3. Submit predictions for each active question using `mcp__forecast__submit_forecasts`.
4. Call `mcp__forecast__next_day` when done. You'll receive resolution feedback with your Brier score per question — use this to learn from mistakes and improve calibration.
5. On every subsequent day: re-read market.csv, search the new articles for updates, and **revise any forecast on a still-active question where new evidence has shifted your view**, then call next_day. A forecast is never "done" while its question is still active.
6. **IMPORTANT**: Actively look out for questions resolving the next day (check via the `resolution_date` column of market.csv) and make sure you have up-to-date predictions on them."""


HANDHOLDING_SECTION = """\
You have full control over your workspace. You are free to create any files or directories that help you perform better. For example, these could be:
- SKILLS.md documenting forecasting strategies you discover work well.
- MEMORY.md tracking key lessons, resolution patterns, and calibration insights.
- Python scripts, analysis tools, data pipelines, or any utilities you need.
- Organize notes per-question, per-topic, or however suits your workflow.

Your workspace is totally yours — use it however you want to maximize your performance."""


# ── Matches BasicAgent._get_source_rules ──

def _get_source_rules(source_name: str) -> str:
    """Source-specific submission rules."""
    # Matches BasicAgent._get_source_rules verbatim, except:
    # - JSON tool-args example → MCP function-call example
    if source_name == "metaculus_binary":
        return """
## BINARY QUESTION RULES
All questions are Yes/No binary. Your `mcp__forecast__submit_forecasts` tool call MUST use exactly:
- **"Yes"** for the affirmative outcome
- **"No"** for the negative outcome

Example tool arguments:
mcp__forecast__submit_forecasts(question_id="12345", outcomes={"Yes": 0.7, "No": 0.3})
"""
    elif source_name == "metaculus_mcq":
        return """
## MULTIPLE CHOICE RULES
Each question has enumerated options shown in the 'options' column.
Your `mcp__forecast__submit_forecasts` tool call MUST use the EXACT option text from the question.
Do NOT paraphrase or abbreviate options.

Example tool arguments (if options are ["Candidate A", "Candidate B", "Candidate C"]):
mcp__forecast__submit_forecasts(question_id="12345", outcomes={"Candidate A": 0.5, "Candidate B": 0.3, "Candidate C": 0.2})
"""
    return ""


# ── Matches BasicAgent._build_binary_brier_scoring_section (single agent) ──

def _build_binary_brier_scoring_section() -> str:
    return """\
## SCORING (Brier Score, Binary)
You are evaluated on **Brier Score** for binary Yes/No questions.
- Let p = your predicted probability for **Yes**.
- Let y = 1 if the resolved outcome is **Yes**, else 0.
- **Brier Score = (p - y)^2**.
- **Lower is better** (0 is perfect, 1 is worst).

Key Mechanics:
1. **Accuracy + Calibration**: Assign probabilities that reflect true likelihood.
2. **Binary Outcomes**: Use exact outcomes "Yes" and "No".
3. **Time-Weighted Score (TW-Score)**: For each question, your time-weighted score = sum(daily_score) / total_question_days where daily_score is the Brier Skill Score for that day (0 if you have no active prediction on that question) and total_question_days is the number of days the question was active. Each prediction's Brier Skill Score (1 minus sum of squared errors) is weighted by how many days it was active before you updated it. Predictions made earlier carry more weight since they cover more days, so act on your best information as soon as possible rather than waiting — **but the TW score equally rewards updating when new evidence arrives, since each new submission overwrites the prior one and accrues weight from that day forward. Never treat a forecast as "done" while its question is still active.**
4. **Prediction-Count Incentive**: Scores are summed (not averaged) across all questions you predict on.
"""


# ── Matches BasicAgent._build_brier_skill_scoring_section (single agent) ──

def _build_brier_skill_scoring_section(max_outcomes_per_question: int) -> str:
    return f"""\
## SCORING (Brier Skill Score)
You have to output a distribution of (outcome, probability) pairs for each question you make a forecast on.
You are evaluated on the **Brier Skill Score** = 1 - Σ(p_i - y_i)^2 summed over all outcomes (thus, ranging from -1 to +1), where:
- p_i = your probability for outcome i
- y_i = 1 if your outcome i is TRUE (actually occurred), 0 otherwise
- **Higher is better**: 1.0 = perfect, 0.0 = abstaining from guessing, negative = worse than abstaining.

Key Mechanics:
1. **Accuracy + Calibration**: Try to guess the most likely outcome(s) and assign calibrated probabilities which reflect the likelihood of the outcome(s) occurring.
2. **Time-Weighted Score (TW-Score)**: For each question, your time-weighted score = sum(daily_score * 100) / total_question_days where daily_score is the Brier Skill Score for that day (0 if you have no active prediction on that question) and total_question_days is the number of days the question was active. Each prediction's Brier Skill Score (1 minus sum of squared errors) is weighted by how many days it was active before you updated it. Predictions made earlier carry more weight since they cover more days, so act on your best information as soon as possible rather than waiting — **but the TW score equally rewards updating when new evidence arrives, since each new submission overwrites the prior one and accrues weight from that day forward. Never treat a forecast as "done" while its question is still active.** Thus, for a set of K questions in total (in the market), the maximum possible TW-Score is 100 * K (if one predicts the correct answer for all K questions on their respective opening date each with 100% probability) and minimum possible TW-Score similarly is -100 * K.
3. **Prediction-Count Incentive**: For each question where you don't have any active prediction on a day, your accuracy, brier skill score, and TW-score for that question will be counted as 0 on that day. Your job is to MAXIMIZE your TW-score. Your TW-score is summed (NOT averaged) across all questions and higher score is better.
4. **End-of-Session Metrics**: At the end of each session, your accuracy, brier skill score, and TW-score *until that session* are calculated and displayed to you. Accuracy and Brier Skill Score are calculated by taking the mean across ALL the questions (0 for question where you don't have any active prediction) while TW-Score is summed across all questions. You are encouraged to maximize your TW-score throughout.
5. **Max Outcomes**: Submit at most {max_outcomes_per_question} outcomes per question.
6. **No Placeholders**: "Unknown", "TBD", "Other" hurt your score. Be specific.
"""


def _get_scoring_section(source_name: str, max_outcomes_per_question: int) -> str:
    """Matches BasicAgent._get_scoring_section for single agent."""
    if source_name == "metaculus_binary":
        return _build_binary_brier_scoring_section()
    return _build_brier_skill_scoring_section(max_outcomes_per_question)


# ── Matches BasicAgent._build_cadence_section ──

def _build_new_articles_text(new_articles_count: Optional[int] = None) -> str:
    """Match BasicAgent's article-update wording when a count is available."""
    if new_articles_count is not None:
        return (
            f"{new_articles_count:,} new articles have been published since your "
            "last update and you can access them using the search tool. "
        )
    return (
        "New articles have been published since your last update and you can "
        "access them using the search tool. "
    )


def _build_cadence_section(
    current_date,
    start_date,
    end_date,
    timegap_days: int = 1,
    new_articles_count: Optional[int] = None,
    last_active_date=None,
    next_active_date=None,
    imminent_qids: Optional[list] = None,
) -> str:
    """Matches BasicAgent._build_cadence_section template.
    CC-SPECIFIC: The system prompt is written once at startup so we cannot
    update article counts dynamically across future wakeups.
    The "context is cleared" sentence is replaced because CC keeps full LLM
    context across days (persistent session)."""
    if next_active_date is None and hasattr(current_date, "__add__"):
        next_active_date = current_date + timedelta(days=timegap_days)
    last_text = (
        f"Last update: {_iso(last_active_date)}. "
        if last_active_date
        else "This is your first update. "
    )
    next_text = (
        f"Next scheduled update: {_iso(next_active_date)}."
        if next_active_date
        else "No later updates are scheduled."
    )
    tomorrow_iso = _iso(current_date + timedelta(days=1)) if hasattr(current_date, "__add__") else None
    if imminent_qids:
        imminent_reminder = (
            f"**IMPORTANT**: {len(imminent_qids)} question(s) resolve tomorrow ({tomorrow_iso}): "
            f"{list(imminent_qids)}. Make sure your prediction on each is up-to-date before calling next_day — "
            "stale forecasts might hurt your performance."
        )
    elif tomorrow_iso is not None:
        imminent_reminder = (
            f"No questions resolve tomorrow ({tomorrow_iso}), but still scan for ones "
            "resolving soon (check the resolution_date column in market.csv)."
        )
    else:
        imminent_reminder = ""
    return (
        "## UPDATE CADENCE\n"
        f"You have the chance to update your predictions every {timegap_days} day(s). "
        # CC-SPECIFIC: replaces BasicAgent's "Your context is cleared after every
        # session and your memory (along with past predictions) is the only
        # information retained between sessions."
        "Your workspace files (memory/, scripts, notes) persist across days — use them to track reasoning and lessons learned. "
        "Articles are available via the search tool and in the articles/ directory. "
        f"Current date: {_iso(current_date)}. {next_text}\n"
        f"{imminent_reminder}\n\n"
    )


# ── Matches BasicAgent._search_results_description ──

def _search_results_description(max_search_results: int = 5, chunk_tokens: int = 512) -> str:
    extra_info = "The search tool uses a hybrid approach to retrieve articles, combining both semantic similarity (through an embedding model) and keyword matching."
    return (
        f"You have access to a search tool that returns up to {max_search_results} retrieved article chunks, "
        f"each roughly {chunk_tokens} tokens long. {extra_info}"
    )


# ── Matches BasicAgent._get_data_notes (single agent mode) ──

def _get_data_notes() -> str:
    # return "Note: `my_prediction` column contains your current forecast as a dict (or None if not yet predicted)."
    return "Note: `my_prediction` column contains your current forecast as a dict (or None if not yet predicted). Similarly, `ground_truth` column contains the ground truth answer which is generally a string (or None if not yet resolved)."""

# ── Main prompt builder ───────────────────────────────────────────────

def build_system_prompt(
    workspace: str,
    current_date,
    start_date,
    end_date,
    source_context: str = "",
    source_name: str = "openforesight",
    num_questions: int = 0,
    num_active: int = 0,
    num_resolved: int = 0,
    max_outcomes_per_question: int = 5,
    search_cutoff_days: int = 0,
    timegap_days: int = 1,
    new_articles_count: Optional[int] = None,
    last_active_date=None,
    next_active_date=None,
) -> str:
    # ── Section ordering matches BasicAgent._build_instructions ──
    #
    # BasicAgent order:
    #   1. Opening line
    #   2. Feedback (resolutions + cumulative perf)    ← CC-SPECIFIC: omitted,
    #                                                     delivered via MCP next_day
    #   3. Source context
    #   4. Source rules
    #   5. Multi-agent context                         ← CC is always single agent
    #   6. Cadence section
    #   7. Memory section                              ← CC-SPECIFIC: HANDHOLDING_SECTION
    #   8. Scoring section
    #   9. AVAILABLE DATA (search + DataFrame + notes) ← CC-SPECIFIC: market.csv
    #                                                     instead of DataFrame
    #  10. CODE EXECUTION ENVIRONMENT                  ← CC-SPECIFIC: omitted,
    #                                                     CC has native Bash/tools
    #  11. RESPONSE FORMAT                             ← CC-SPECIFIC: omitted,
    #                                                     CC uses MCP tools natively
    #  12. TOOLS AVAILABLE FOR YOUR USE                ← CC-SPECIFIC: MCP tool format
    #  13. INTERACTION FLOW (budget)                   ← CC-SPECIFIC: no budget system
    #  14. SUBMISSION RULES
    #  15. Begin

    source_rules = _get_source_rules(source_name)
    scoring_section = _get_scoring_section(source_name, max_outcomes_per_question)
    cadence_section = _build_cadence_section(
        current_date,
        start_date,
        end_date,
        timegap_days,
        new_articles_count=new_articles_count,
        last_active_date=last_active_date,
        next_active_date=next_active_date,
    )
    data_notes = _get_data_notes()

    # Matches BasicAgent search_advice text verbatim.
    search_results_desc = _search_results_description()
    search_advice = f"You have access to a news article database which is updated **daily**. {search_results_desc} You can also access the articles directly in the articles/ directory."
    search_advice = f"You have access to a news article database which is updated **daily** through a search tool, that you can use to find evidence for your forecasts."
    # Matches BasicAgent search_tool_line cutoff description.
    cutoff_desc = "today's date"
    if search_cutoff_days > 0:
        cutoff_date = current_date - timedelta(days=search_cutoff_days) if hasattr(current_date, '__sub__') else current_date
        cutoff_desc = f"{_iso(cutoff_date)} (today - {search_cutoff_days} days)"

    # Matches BasicAgent search_tool_line text verbatim.
    search_tool_line = (
        f"- `mcp__forecast__search_news(query, from_date?, to_date?)`: search the news corpus for evidence. "
        f"`to_date` is capped at {cutoff_desc}. {search_results_desc}\n"
    )

    # ── Assemble prompt (matches BasicAgent._build_instructions order) ──

    intro_sections = [
        f"You are a forecasting agent. Today is {current_date}. Your goal is to make accurate and calibrated predictions.",
        source_context.strip(),
        source_rules.strip(),
        cadence_section.strip(),
    ]
    intro_block = "\n\n".join(section for section in intro_sections if section)

    return f"""\
{intro_block}


{scoring_section}

## AVAILABLE DATA
{search_advice} 
You can access the market.csv file (READ-ONLY) in your workspace containing {num_questions} questions ({num_active} active/unresolved, {num_resolved} resolved).

Column descriptions of the DataFrame (market.csv):
- qid (str) (Question ID)
- title (str) (Question Content)
- background (object)
- resolution_criteria (object)
- answer_type (object)
- resolution_date (object)
- is_resolved (bool)
- ground_truth (object)
- num_predictions (int64)
- options (object)
- my_prediction (object)
- my_prediction_date (object)

{data_notes}


## TOOLS AVAILABLE FOR YOUR USE
{search_tool_line}\
- `mcp__forecast__submit_forecasts(question_id, outcomes)`: submit exactly one forecast for exactly one question ID (`qid`). Your forecasts are tracked by the harness — there is no file to inspect.
- `mcp__forecast__next_day()`: end the current session and proceed to the next one.


## Workspace:
- market.csv — Read-only snapshot of all questions (refreshed each day).
- articles/ — Browsable news articles organized by date as articles/YYYY/MM/DD/articles.jsonl (one JSON article per line). New date directories appear after calling `mcp__forecast__next_day`.
  - Each line has fields: `title` (headline), `source` (publisher domain, e.g. "www.reuters.com"), `date_publish` (original publication date, YYYY-MM-DD), `url` (canonical article link), `content` (full article body text to read/grep), plus `id`, `date` (crawl date), `date_modify`.
- memory/ — Your persistent notes directory. Read and write freely. Files here persist across days. Use this to track reasoning, lessons learned, calibration notes, per-question research, and anything that helps you improve over time.

{HANDHOLDING_SECTION}


## SUBMISSION RULES
- qid must be from an active (`is_resolved=False`) question you identified from market.csv
- Each `mcp__forecast__submit_forecasts` call must contain exactly one forecast for one question ID (`qid`).
- You may submit again later in the same session to update that `qid`.
- Maximum of {max_outcomes_per_question} outcomes allowed per question.
- Outcome names must be REAL predicted answers (e.g. person names, locations, dates, etc.)
- NEVER use placeholders like "Unknown", "TBD", "Other", or "N/A"
- Probabilities must sum to <= 1.0


## Rules
- No web access is available. Use `mcp__forecast__search_news` and articles/ for information.
- market.csv is read-only. DO NOT modify it.
- You can use Bash, Read, Write, Grep, Glob, and other tools freely in your workspace.
- Your job is to maximize your time-weighted score (TW-score).

---

Begin."""
