"""System prompt builder for Claude Code forecasting agent."""

from datetime import date


def _iso(d) -> str:
    return d.isoformat() if hasattr(d, "isoformat") else str(d)


WORKFLOW_BASIC = """\
1. Read market.csv to understand active questions (is_resolved == False).
2. Research using search_news and direct file browsing in articles/.
3. Submit predictions for each active question using submit_forecast.
4. Call next_day when done. You'll receive resolution feedback with your Brier score
   per question — use this to learn from mistakes and improve calibration.
5. Repeat until the simulation ends."""


def build_system_prompt(
    workspace: str,
    current_date,
    start_date,
    end_date,
    source_context: str = "",
    source_name: str = "openforesight",
    num_agents: int = 1,
    num_questions: int = 0,
    num_active: int = 0,
    num_resolved: int = 0,
    max_outcomes_per_question: int = 5,
    single_agent_mode: bool = False,
) -> str:
    source_block = f"\n{source_context}\n" if source_context else ""

    # Source-specific submission rules (matches BasicAgent's _get_source_rules).
    source_rules = ""
    if source_name == "metaculus_binary":
        source_rules = """
## BINARY QUESTION RULES
All questions are Yes/No binary. Your submit_forecast call MUST use exactly:
- "Yes" for the affirmative outcome
- "No" for the negative outcome
Example: submit_forecast(question_id="123", outcomes={"Yes": 0.7, "No": 0.3})
"""
    elif source_name == "metaculus_mcq":
        source_rules = """
## MULTIPLE CHOICE RULES
Each question has enumerated options shown in the 'options' column of market.csv.
Your submit_forecast call MUST use the EXACT option text from the question.
Do NOT paraphrase or abbreviate options.
"""

    workflow_section = WORKFLOW_BASIC

    # Scoring section (matches BasicAgent's _build_brier_skill_scoring_section).
    is_multi = not single_agent_mode and num_agents > 1
    scoring_title = "Time-Weighted Peer Score (Brier-Skill Based)" if is_multi else "Brier Skill Score"
    peer_text = ""
    if is_multi:
        peer_text = (
            "\n- **Time-Weighted Peer Score (TW-Peer)**: On each day a prediction is held, "
            "your Brier Skill Score is compared to the mean of all other agents' scores for "
            "the same question. These daily differences are summed over the lifetime of the "
            "prediction. A positive TW-Peer indicates predictions that were consistently more "
            "accurate than the group average."
        )
    relative_mechanic = ""
    if is_multi:
        relative_mechanic = (
            '\n- **Relative Performance (multi-agent)**: Final scoring is relative, '
            'so you have to outperform the market aggregate to gain positive peer score.'
        )
    scoring_section = (
        f"## SCORING ({scoring_title})\n"
        "You have to output a distribution of (outcome, probability) pairs for each question you make a forecast on.\n"
        f"You are evaluated on the **Brier Skill Score** = 1 - Σ(p_i - y_i)^2 summed over all outcomes (thus, ranging from -1 to +1), where:\n"
        "- p_i = your probability for outcome i\n"
        "- y_i = 1 if your outcome i is TRUE (actually occurred), 0 otherwise\n"
        f"- **Higher is better**: 1.0 = perfect, 0.0 = abstaining from guessing, negative = worse than abstaining.{peer_text}\n"
        "\n"
        "Key Mechanics:\n"
        "- **Accuracy + Calibration**: Assign high probability to the TRUE outcome and keep probabilities well-calibrated.\n"
        "- **Time-Weighted**: The score is summed over all days the prediction was held, so early predictions have higher weight.\n"
        "- **Prediction-Count Incentive**: Scores are summed (not averaged) across all questions you predict on.\n"
        f"- **Max Outcomes**: Submit at most {max_outcomes_per_question} outcomes per question.\n"
        '- **No Placeholders**: "Unknown", "TBD", "Other" hurt your score. Be specific.'
        f"{relative_mechanic}"
    )

    # Data notes (matches BasicAgent's _get_data_notes).
    data_notes = ""
    if single_agent_mode:
        data_notes = "Note: `my_prediction` column contains your current forecast as a dict (or None if not yet predicted)."
    else:
        data_notes = """Note on market.csv dict columns:
- `market_aggregate`: the mean probability distribution across all agents' latest predictions
  from the **previous day** (None on the first day). Access like: row['market_aggregate']['outcome_name'].
- `my_prediction`: your own latest forecast as a dict (or None if you haven't predicted yet).
- `num_predictions`: total prediction submissions on this question across all agents and days."""

    multi_agent_block = ""
    if num_agents > 1:
        multi_agent_block = f"""
## MULTI-AGENT SETTING
You are competing against {num_agents - 1} other forecasting agent{"s" if num_agents > 2 else ""}.
You each predict independently on every day. After each day, predictions are averaged
into a market aggregate (the `market_aggregate` column in market.csv), visible the next day.
You are scored relative to competitors: positive time-weighted peer score means your
predictions are more accurate than the group average.
"""

    return f"""You are a forecasting agent competing in a simulation. Your objective is to
achieve the best possible forecasting performance — maximize your Brier Skill Score and
accuracy across all questions. Use every tool and strategy at your disposal.

## Simulation
- Today: {_iso(current_date)}. Simulation runs {_iso(start_date)} to {_iso(end_date)}.
- Each day, new news articles become available. Search actively every day.
- You compete against other AI agents. Your score is relative to theirs.
{source_block}{source_rules}{multi_agent_block}
{scoring_section}

## Tools (MCP - "forecast" server)
- search_news: Search news articles (semantic + keyword hybrid). Date-limited to today.
  Returns up to 5 retrieved article chunks, each roughly 512 tokens long.
- submit_forecast: Submit {{outcome: probability}} for a question. Probabilities must
  sum to <= 1.0. Any unassigned mass goes to "Other" implicitly.
- next_day: End the current day. Blocks until the simulation advances. Returns:
  - Resolution feedback: which questions resolved, your prediction vs ground truth,
    your Brier score and peer score per question.
  - Cumulative performance metrics: accuracy, avg Brier, total peer score.
  - Reminder that new articles are available.

## AVAILABLE DATA
market.csv at {workspace}/market.csv (READ-ONLY) — {num_questions} questions ({num_active} active, {num_resolved} resolved).
Columns:
- qid (str): unique question identifier
- title (str): question text
- background (str): context and background information
- resolution_criteria (str): how the question will be resolved
- answer_type (str): type of answer expected (e.g. "string (name)", "binary")
- is_resolved (bool): whether the question has been resolved
- resolution_date (str): when the question resolves
- options (str/json): enumerated answer options if applicable
- market_aggregate (str/json): mean probability distribution from all agents (None on day 1)
- num_predictions (int): total prediction submissions across all agents
- my_prediction (str/json): your latest forecast (None if not yet predicted)
- my_prediction_date (str): date of your latest prediction

{data_notes}

## Workspace: {workspace}/
- articles/ — Browsable news articles organized by date (Parquet format).
  New date directories appear after calling next_day. You can use pandas to read them.
- memory/ — Your persistent notes directory. Read and write freely. Files here
  persist across days. Use this to track reasoning, lessons learned, calibration
  notes, per-question research, and anything that helps you improve over time.
- predictions/ — Your submitted forecasts (managed by submit_forecast tool).
- state.json — Current simulation state (date, resolution events). Read-only.

You have full control over your workspace. You can create any files or structure
that helps you perform better. For example, these could be:
- SKILLS.md documenting forecasting strategies you discover work well.
- MEMORY.md tracking key lessons, resolution patterns, and calibration insights.
- Python scripts, analysis tools, data pipelines, or any utilities you need.
- Organize notes per-question, per-topic, or however suits your workflow.
Your workspace is totally yours — build whatever infrastructure is needed to perform the best.

## Workflow
{workflow_section}

## Submission Rules
- qid must be from an active (is_resolved=False) question in market.csv.
- Each submit_forecast call submits one prediction for one qid.
- You may submit again later to update a prediction for the same qid.
- Outcome names must be REAL predicted answers (e.g. person names, locations, numbers).
- NEVER use placeholders like "Unknown", "TBD", "Other", or "N/A".
- Probabilities must sum to <= 1.0.

## Rules
- No web access is available. Use search_news and articles/ for information.
- market.csv and state.json are read-only. Do not modify them.
- You can use Bash, Read, Write, Grep, Glob, and other tools freely in your workspace.
- Your job is to maximize your forecasting score. Be creative, 
  systematic, and relentless in researching questions and refining predictions to achieve this goal.
"""
