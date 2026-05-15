"""Daily prompt construction helpers for BasicAgent."""

import re
from datetime import date, timedelta
from typing import Dict, Optional

from agents.utils.memory import ActiveMemory, StructuredMemory


class BasicPromptBuilder:
    def _build_prompt_seed_budget_block(
        self,
        *,
        warmup: bool = False,
        max_actions_override: Optional[int] = None,
        leading_newline: bool = False,
        trailing_newlines: int = 2,
    ) -> str:
        """Render the start-of-prompt budget block with a prompt-seed clarification."""
        status = self._build_start_budget_status(
            warmup=warmup,
            max_actions_override=max_actions_override,
        )
        if not status:
            return ""

        prefix = "\n" if leading_newline else ""
        suffix = "\n" * max(trailing_newlines, 0)
        return (
            f"{prefix}Budget at start:\n"
            f"{status}\n"
            "Note: current message tokens are not accounted for yet."
            f"{suffix}"
        )

    @staticmethod
    def _normalize_prompt_heading_spacing(prompt: str) -> str:
        """Ensure `##` section headings are separated by two blank lines."""
        if not prompt:
            return prompt
        return re.sub(r"\n{1,}(?=## )", "\n\n\n", prompt)

    def _search_results_description(self) -> str:
        chunk_tokens = self._search_handler.chunk_tokens
        extra_info = "The search tool uses a hybrid approach to retrieve articles, combining both semantic similarity (through an embedding model) and keyword matching."
        if chunk_tokens is None:
            return f"You have access to a search tool that returns up to {self.config.max_search_results} retrieved article chunks. {extra_info}"
        return (
            f"You have access to a search tool that returns up to {self.config.max_search_results} retrieved article chunks, "
            f"each roughly {chunk_tokens} tokens long. {extra_info}"
        )

    def _get_timegap_days(self) -> int:
        return max(1, int(getattr(self.config, "timegap_days", 1) or 1))

    def _get_last_active_date(self, current_date: date) -> Optional[date]:
        fi = getattr(self, "_forecast_interface", None)
        last_active = getattr(fi, "last_active_date", None) if fi is not None else None
        if last_active:
            return last_active
        return None

    def _get_next_active_date(self, current_date: date) -> Optional[date]:
        fi = getattr(self, "_forecast_interface", None)
        if fi is not None and hasattr(fi, "next_active_date"):
            next_active = getattr(fi, "next_active_date")
            return next_active
        return current_date + timedelta(days=self._get_timegap_days())

    def _build_cadence_section(self, current_date: date) -> str:
        last_active = self._get_last_active_date(current_date)
        next_active = self._get_next_active_date(current_date)
        timegap_days = self._get_timegap_days()
        day_unit = "day" if timegap_days == 1 else "days"
        next_text = (
            f"Next scheduled update: {next_active}."
            if next_active
            else "No later updates are scheduled."
        )
        last_text = (
            f"Last update: {last_active}. "
            if last_active
            else "This is your first update. "
        )
        articles_text = ""
        if last_active:
            if self._search_handler.is_available:
                count = self._search_handler.count_articles(
                    min_date=last_active,
                    max_date=current_date - timedelta(days=self.config.search_cutoff_days),
                )
                if count is not None:
                    articles_text = f"{count:,} new articles have been published since your last update and you can access them using the search tool. "
            if not articles_text:
                articles_text = "New articles have been published since your last update and you can access them using the search tool. "
        tomorrow = current_date + timedelta(days=1)
        imminent_qids = []
        fi = getattr(self, "_forecast_interface", None)
        if fi is not None and hasattr(fi, "questions"):
            imminent_qids = [
                qid for qid, q in (fi.questions or {}).items()
                if getattr(q, "resolution_date", None) == tomorrow
            ]
        if imminent_qids:
            imminent_reminder = (
                f"**IMPORTANT**: {len(imminent_qids)} question(s) resolve tomorrow ({tomorrow.isoformat()}): "
                f"{imminent_qids}. Make sure your prediction on each is up-to-date before calling next_day — "
                "stale forecasts might hurt your performance."
            )
        else:
            imminent_reminder = (
                f"No questions resolve tomorrow ({tomorrow.isoformat()}), but still scan for ones "
                "resolving soon (filter `df` by `resolution_date`)."
            )
        has_memory = getattr(self, "_memory", None) is not None
        retained_context = (
            "your memory (along with past predictions) is the only information retained between sessions"
            if has_memory
            else "your past predictions are the only information retained between sessions"
        )
        return (
            "## UPDATE CADENCE\n"
            f"You can make updates every {timegap_days} {day_unit}. Your context is cleared after every session and {retained_context}. {articles_text}"
            f"{last_text}Current date: {current_date}. {next_text}\n"
            f"{imminent_reminder}\n\n"
        )

    def _format_and_cache_feedback(self, current_date: date) -> str:
        """Generate feedback, cache it for the memory prompt, and return formatted text."""
        feedback_data = self._feedback_handler.generate_feedback(
            self._forecast_interface, current_date, self.inference
        )
        self._last_feedback_data = feedback_data
        return self._feedback_handler.format_feedback(
            feedback_data,
            show_tw_peer=not self.config.single_agent_mode,
        )

    @staticmethod
    def _render_key_mechanics(
        mechanics: Dict[str, str],
        drop_keys: Optional[set[str]] = None,
    ) -> str:
        """Render numbered key mechanics, preserving insertion order."""
        drop = drop_keys or set()
        lines = [text for key, text in mechanics.items() if key not in drop]
        return "\n".join(f"{idx}. {line}" for idx, line in enumerate(lines, start=1))

    def _build_binary_brier_scoring_section(
        self,
        *,
        include_peer_summary: bool = True,
        drop_mechanics: Optional[set[str]] = None,
    ) -> str:
        """Build binary Brier scoring text with optional mechanic filtering."""
        is_multi_agent = not self.config.single_agent_mode
        show_peer = is_multi_agent and include_peer_summary

        peer_text = ""
        if show_peer:
            peer_text = """
- **Time-Weighted Peer Score (TW-Peer)**: `100 × (avg others' Brier - your Brier)`, summed over each day a prediction is held. A positive TW-Peer indicates predictions that were consistently more accurate than the group average."""

        mechanics: Dict[str, str] = {
            "accuracy_calibration": "**Accuracy + Calibration**: Assign probabilities that reflect true likelihood.",
            "binary_outcomes": "**Binary Outcomes**: Use exact outcomes \"Yes\" and \"No\".",
            "time_weighted": "**Time-Weighted Score (TW-Score)**: For each question, your time-weighted score = sum(daily_score) / total_question_days where daily_score is the Brier Skill Score for that day (0 if you have no active prediction on that question) and total_question_days is the number of days the question was active. Each prediction's Brier Skill Score (1 minus sum of squared errors) is weighted by how many days it was active before you updated it. Predictions made earlier carry more weight since they cover more days, so act on your best information as soon as possible rather than waiting — **but the TW score equally rewards updating when new evidence arrives, since each new submission overwrites the prior one and accrues weight from that day forward. Never treat a forecast as \"done\" while its question is still active.**",
            "question_count": "**Prediction-Count Incentive**: Scores are summed (not averaged) across all questions you predict on.",
        }
        if show_peer:
            mechanics["relative_performance"] = (
                "**Relative Performance (multi-agent)**: Final scoring is relative, "
                "so you have to outperform the market aggregate to gain positive peer score."
            )

        mechanics_text = self._render_key_mechanics(mechanics, drop_mechanics)
        return f"""## SCORING (Brier Score, Binary)
You are evaluated on **Brier Score** for binary Yes/No questions.
- Let p = your predicted probability for **Yes**.
- Let y = 1 if the resolved outcome is **Yes**, else 0.
- **Brier Score = (p - y)^2**.
- **Lower is better** (0 is perfect, 1 is worst).{peer_text}

Key Mechanics:
{mechanics_text}
"""

    def _build_brier_skill_scoring_section(
        self,
        *,
        include_peer_summary: bool = True,
        drop_mechanics: Optional[set[str]] = None,
    ) -> str:
        """Build Brier Skill scoring text with optional mechanic filtering."""
        is_multi_agent = not self.config.single_agent_mode
        show_peer = is_multi_agent and include_peer_summary

        section_title = "Time-Weighted Peer Score (Brier-Skill Based)" if show_peer else "Brier Skill Score"
        peer_text = ""
        if show_peer:
            peer_text = """
- **Time-Weighted Peer Score (TW-Peer)**: On each day a prediction is held, your Brier Skill Score is compared to the mean of all other agents' scores for the same question. These daily differences are summed over the lifetime of the prediction. A positive TW-Peer indicates predictions that were consistently more accurate than the group average."""

        mechanics: Dict[str, str] = {
            "accuracy_calibration": "**Accuracy + Calibration**: Try to guess the most likely outcome(s) and assign calibrated probabilities which reflect the likelihood of the outcome(s) occurring.",
            "time_weighted": "**Time-Weighted Score (TW-Score)**: For each question, your time-weighted score = sum(daily_score * 100) / total_question_days where daily_score is the Brier Skill Score for that day (0 if you have no active prediction on that question) and total_question_days is the number of days the question was active. Each prediction's Brier Skill Score (1 minus sum of squared errors) is weighted by how many days it was active before you updated it. Predictions made earlier carry more weight since they cover more days, so act on your best information as soon as possible rather than waiting — **but the TW score equally rewards updating when new evidence arrives, since each new submission overwrites the prior one and accrues weight from that day forward. Never treat a forecast as \"done\" while its question is still active.** Thus, for a set of K questions in total (in the market), the maximum possible TW-Score is 100 * K (if one predicts the correct answer for all K questions on their respective opening date each with 100% probability) and minimum possible TW-Score similarly is -100 * K.",
            "question_count": "**Prediction-Count Incentive**: For each question where you don't have any active prediction on a day, your accuracy, brier skill score, and TW-score for that question will be counted as 0 on that day. Your job is to MAXIMIZE your TW-score. Your TW-score is summed (NOT averaged) across all questions and higher score is better.",
            "end_of_session_metrics": "**End-of-Session Metrics**: At the end of each session, your accuracy, brier skill score, and TW-score *until that session* are calculated and displayed to you. Accuracy and Brier Skill Score are calculated by taking the mean across ALL the questions (0 for question where you don't have any active prediction) while TW-Score is summed across all questions. You are encouraged to maximize your TW-score throughout.",
            "max_outcomes": f"**Max Outcomes**: Submit at most {self.config.max_outcomes_per_question} outcomes per question.",
            "no_placeholders": "**No Placeholders**: \"Unknown\", \"TBD\", \"Other\" hurt your score. Be specific.",
        }
        if show_peer:
            mechanics["relative_performance"] = (
                "**Relative Performance (multi-agent)**: Final scoring is relative, "
                "so you have to outperform the market aggregate to gain positive peer score."
            )

        mechanics_text = self._render_key_mechanics(mechanics, drop_mechanics)
        return f"""## SCORING ({section_title})
You have to output a distribution of (outcome, probability) pairs for each question you make a forecast on.
You are evaluated on the **Brier Skill Score** = 1 - Σ(p_i - y_i)^2 summed over all outcomes (thus, ranging from -1 to +1), where:
- p_i = your probability for outcome i
- y_i = 1 if your outcome i is TRUE (actually occurred), 0 otherwise
- **Higher is better**: 1.0 = perfect, 0.0 = abstaining from guessing, negative = worse than abstaining.{peer_text}

Key Mechanics:
{mechanics_text}
"""

    def _get_scoring_section(self) -> str:
        """Get scoring description - single vs multi-agent mode."""
        return self._build_brier_skill_scoring_section()

    def _get_data_notes(self) -> str:
        """Get notes about DataFrame columns - conditional on agent mode."""
        if self.config.single_agent_mode:
            return "Note: `my_prediction` column contains your current forecast as a dict (or None if not yet predicted). Similarly, `ground_truth` column contains the ground truth answer which is generally a string (or None if not yet resolved)."
        else:
            return """Note: `market_aggregate` and `my_prediction` columns contain Python dicts (or None). You can access them directly, e.g. `row['market_aggregate']['outcome_name']`.
- `market_aggregate`: the mean probability distribution across all agents' latest predictions from the **previous day**. `None` on the first day (no predictions exist yet).
- `my_prediction`: your own latest forecast (or None if you haven't predicted this question yet).
- `num_predictions`: total number of prediction submissions made on this question across all agents and all days."""

    def _get_multiagent_context(self) -> str:
        """Multi-agent preamble describing the competitive setting."""
        if self.config.single_agent_mode:
            return ""
        n = getattr(self._forecast_interface, "num_agents", 0)
        if n < 2:
            return ""
        return f"""## MULTI-AGENT SETTING
You are competing against {n - 1} other forecasting agent{"s" if n > 2 else ""} on the same set of questions.
You each predict independently on every wakeup day. After each day, your predictions are averaged with the others' into a market aggregate (the `market_aggregate` column), which you can see starting the following day.
You are scored relative to your competitors: to earn a positive time-weighted peer score, your predictions need to be more accurate than the group average.
"""

    def _get_source_rules(self) -> str:
        """Get source-specific submission rules."""
        return ""

    def _build_instructions(self, current_date: date) -> str:
        """Build the daily tool-calling prompt."""
        df_info = self._query_handler.get_info()
        budget_start_block = self._build_prompt_seed_budget_block()
        cadence_section = self._build_cadence_section(current_date)

        memory_section = ""
        memory_flow_note = ""
        if self._memory is not None:
            memory_content = self._memory.get()
            if isinstance(self._memory, ActiveMemory):
                meta_index = self._memory.get_index()
                meta_block = f"Current meta-insights with their indices:\n{meta_index}\n\n" if meta_index else ""
                memory_section = f"""## YOUR MEMORY
{meta_block}`mem_df` holds your per-question notes (reasoning, evidence, calibration) — 1 row per question.
Columns: qid (str), question (str), last_updated (str), memory (str), category (str)
Both `mem_df` and `df` are available in the same `query_df` sandbox. You can join them on qid to find questions worth revisiting.

Inspect `mem_df` via `query_df`. Edit per-question notes with `mem_add`, `mem_update`, `mem_delete`. 
Manage meta-insights with `memory_retrieve` (using the indices), `memory_new`, `memory_update`, `memory_delete`. 
"""
            elif isinstance(self._memory, StructuredMemory):
                memory_index = self._memory.get_index()
                index_block = f"Current memory index:\n{memory_index}\n\n" if memory_index else ""
                memory_section = f"""## YOUR MEMORY ({self._memory.entry_count} entries, max {self._memory._max_entries})
{index_block}Entries should capture question-specific reasoning (include QIDs), post-resolution lessons,
or cross-question calibration patterns. Use the memory tools to retrieve, add, update, or delete entries.

"""
            elif memory_content:
                memory_section = f"""## YOUR MEMORY
{memory_content}

Use the reasoning and insights above to inform today's forecasts.

"""

            if isinstance(self._memory, (StructuredMemory, ActiveMemory)):
                memory_flow_note = (
                    "When you finish forecasting and are ready to move on, call `next_day()` to transition into the memory update phase. "
                    "The transition can also happen automatically if tokens run low."
                )
            else:
                memory_flow_note = "After ending this session, you will be prompted to update your memory."

        search_tool_line = ""
        search_advice = ""
        if self._search_handler.is_available:
            cutoff_desc = "today's date"
            if self.config.search_cutoff_days > 0:
                cutoff_date = current_date - timedelta(days=self.config.search_cutoff_days)
                cutoff_desc = f"{cutoff_date} (today - {self.config.search_cutoff_days} days)"
            search_tool_line = (
                "- `search_news(query, from_date?, to_date?)`: search the news corpus for evidence. "
                f"`to_date` is capped at {cutoff_desc}. {self._search_results_description()}\n"
            )
            search_advice = (
                f"You have access to a news article database which is updated **daily** through a search tool, that you can use to find evidence for your forecasts."
            )

        memory_tools_section = ""
        if isinstance(self._memory, ActiveMemory):
            memory_tools_section = (
                "- `memory_retrieve` / `memory_new` / `memory_update` / `memory_delete`: manage meta-insight entries.\n"
                "- `mem_add` / `mem_update` / `mem_delete`: manage question-specific notes in `mem_df`.\n"
            )
        elif isinstance(self._memory, StructuredMemory):
            memory_tools_section = (
                "- `memory_retrieve` / `memory_new` / `memory_update` / `memory_delete`: manage reusable memory entries.\n"
            )

        if isinstance(self._memory, ActiveMemory):
            interaction_text = (
                "You can interleave queries, searches, memory operations, and submissions as needed. "
                "Consider using `mem_df` early to recall prior reasoning and identify which questions need attention."
            )
        elif isinstance(self._memory, StructuredMemory):
            interaction_text = (
                "You can interleave queries, searches, memory operations, and submissions as needed. "
                "Use memory tools when you need to retrieve or save reusable reasoning."
            )
        elif self._memory is not None:
            interaction_text = (
                "You can interleave queries, searches, and submissions as needed. "
                "Use your memory to inform today's forecasts."
            )
        else:
            interaction_text = "You can interleave queries, searches, and submissions as needed."

        intro_sections = [
            self._format_and_cache_feedback(current_date).strip(),
            str(getattr(self._forecast_interface, 'source_context', '') or '').strip(),
            self._get_source_rules().strip(),
            self._get_multiagent_context().strip(),
            f"{cadence_section}{memory_section}{self._get_scoring_section()}".strip(),
        ]
        intro_block = "\n\n".join(section for section in intro_sections if section)

        available_data_lines = ["## AVAILABLE DATA"]
        if search_advice:
            available_data_lines.append(search_advice)
        available_data_lines.extend(
            [
                f"You also have access to a pandas DataFrame `df` with {df_info['n_rows']} questions ({df_info['n_active']} active/unresolved, {df_info['n_resolved']} resolved).",
                "",
                "Column descriptions of the DataFrame:",
                str(df_info['columns_desc']),
                "",
                self._get_data_notes(),
            ]
        )
        available_data_section = "\n".join(available_data_lines)

        code_env_lines = [
            "## CODE EXECUTION ENVIRONMENT",
            "You have access to a Python code execution environment where your code runs in a sandbox with these variables pre-defined:",
            "- `df`: the DataFrame with the questions and your predictions",
            "- `pd`: pandas module",
            f"- `today`: date object for {current_date}",
            "- `date`, `datetime`, `timedelta`: from datetime module",
        ]
        if isinstance(self._memory, ActiveMemory):
            code_env_lines.append(
                "- `mem_df`: your question-specific memory DataFrame (you can join with df on qid to decide what to revisit)"
            )
        code_env_lines.append(
            "Standard builtins (len, str, int, float, min, max, sum, sorted, range, etc.) are available. "
            "A small safe subset of stdlib imports (for example datetime, json, math, re, ast) is allowed. "
            "External file, network, process, and private-attribute access is blocked; stay within in-memory DataFrame/pandas operations."
        )
        code_env_section = "\n".join(code_env_lines)

        tip_line = ""
        if isinstance(self._memory, ActiveMemory):
            tip_line = (
                "Tip: After submitting a forecast, consider saving your reasoning and key evidence for that QID using mem_add/mem_update. "
                "At end of day, you will get another opportunity to update your memory."
            )
        elif isinstance(self._memory, StructuredMemory):
            tip_line = (
                "Tip: After submitting a forecast, consider saving reusable reasoning and key evidence using memory_new/memory_update. "
                "At end of day, you will get another opportunity to update your memory."
            )

        sections = [
            f"You are a forecasting agent. Today is {current_date}. Your goal is to make accurate and calibrated predictions.",
            intro_block,
            available_data_section,
            code_env_section,
            (
                "## TOOLS AVAILABLE FOR YOUR USE\n"
                "Use the function tools from the tool schema. Call exactly one tool per turn.\n"
                "- `query_df(code)`: inspect questions and your existing predictions. Use `print(...)` for outputs.\n"
                f"{search_tool_line}{memory_tools_section}"
                "- `submit_forecasts(forecasts)`: submit exactly one forecast for exactly one question ID (`qid`).\n"
                "- `next_day()`: end the current session and proceed to the next one."
            ),
            (
                "## INTERACTION FLOW\n"
                f"{self._build_budget_overview()}"
                f"{interaction_text}"
            ),
            (
                "## SUBMISSION RULES\n"
                "- qid must be from an active (`is_resolved=False`) question you identified from `df`\n"
                "- Each `submit_forecasts` call must contain exactly one forecast for one question ID (`qid`).\n"
                "- You may submit again later in the same session to update that `qid`.\n"
                f"- Maximum of {self.config.max_outcomes_per_question} outcomes allowed per question.\n"
                "- Outcome names must be REAL predicted answers (e.g. person names, locations, dates, etc.)\n"
                "- NEVER use placeholders like \"Unknown\", \"TBD\", \"Other\", or \"N/A\"\n"
                "- Probabilities must sum to <= 1.0"
            ),
            tip_line,
            f"---\n{budget_start_block}Begin.",
        ]
        return self._normalize_prompt_heading_spacing(
            "\n\n".join(section for section in sections if section)
        )
