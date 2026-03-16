"""
BasicAgent: Main agent class for LLM-based forecasting.

Uses chain-of-thought with <reasoning> and <action type="..."> tags.
"""

import re
from datetime import date, timedelta
from typing import List, Dict, Any, Optional, Tuple

from agents.base import BaseAgent
from agents.utils.forecast_parser import parse_action, extract_memory, extract_memory_ops, extract_memo_ops
from agents.utils.budget import BudgetSettings, BudgetTracker
from agents.utils.timing import AgentTimer
from environment.interfaces import PredictionSubmission

from .config import AgentConfig
from .memory import BasicMemory
from agents.utils.memory import StructuredMemory, ActiveMemory
from .query import QueryHandler
from .search import SearchHandler
from .feedback import FeedbackHandler


class BasicAgent(BaseAgent):
    """
    Basic forecasting agent using LLM inference.
    
    Interaction flow per day:
    1. Receives system prompt with DataFrame schema and scoring rules
    2. Can take query/search/submit/next actions, subject to configured loop budgets:
       - query: Execute Python code to explore the DataFrame
       - search: Search news articles for context (if enabled)
       - get_article: Retrieve full article content
       - submit: Submit a forecast for exactly one question
       - next: End the current day and proceed to next
    3. Updates memory at end of day (always, regardless of how day ended)
    
    Uses <reasoning> for chain-of-thought and <action type="..."> for actions.
    """
    
    def __init__(self, 
                 agent_id: str,
                 inference_provider,
                 config: AgentConfig = None,
                 model_name: str = "",
                 search_tool=None):
        super().__init__(agent_id, inference_provider, model_name)
        self.config = config or AgentConfig()

        # Timing utilities for performance analysis
        self._timer = AgentTimer()
        
        # Handlers
        if self.config.enable_memory:
            if self.config.memory_format == "active":
                self._memory = ActiveMemory(agent_id, self.config.memory_dir)
            elif self.config.memory_format == "structured":
                self._memory = StructuredMemory(agent_id, self.config.memory_dir)
            else:
                self._memory = BasicMemory(agent_id, self.config.memory_dir)
        else:
            self._memory = None
        self._query_handler = QueryHandler()
        self._search_handler = SearchHandler(
            search_tool,
            snippet_max_chars=self.config.snippet_max_chars,
            article_max_chars=self.config.article_max_chars,
            search_cutoff_days=self.config.search_cutoff_days
        )
        self._feedback_handler = FeedbackHandler(
            agent_id,
            timing_callback=self._record_matcher_timing
        )
        
    def act(self, 
            doc_interface,  # Not used in BasicAgent
            forecast_interface,  # Has get_market_csv_path, submit_prediction, next_day
            current_date: date) -> List[Dict[str, Any]]:
        """
        Execute agent logic for the day.
        
        Flow:
        1. Initialize handlers
        2. Run action loop (query/search/submit/next)
        3. Update memory
        4. Signal day completion
        
        Returns list of submitted forecasts.
        """
        # Start timing for the day
        self._timer.reset()
        self._timer.start_day()
        self._day_qids = set()  # Track QIDs the agent interacts with today

        # Load memory for this date (loads most recent snapshot before current_date)
        if self._memory is not None:
            self._memory.set_date(current_date)
        
        # Setup handlers
        self._setup_day(forecast_interface, current_date)
        
        # Build initial prompt
        messages = [{"role": "user", "content": self._build_instructions(current_date)}]
        
        # Action loop
        all_forecasts = self._run_action_loop(messages, forecast_interface)
        
        # End of day: memory update (always happens)
        if self._memory is not None:
            self._prompt_memory_update(messages, forecast_interface, current_date)
        
        # End timing and save stats
        self._timer.end_day()
        if self.config.memory_dir:
            self._timer.save_day_stats(self.config.memory_dir, current_date)
        
        # Signal day completion
        forecast_interface.next_day()
        
        return all_forecasts
    
    # =========================================================================
    # Setup
    # =========================================================================
    
    def _setup_day(self, forecast_interface, current_date: date) -> None:
        """Initialize handlers for the day."""
        csv_path = forecast_interface.get_market_csv_path()
        self._query_handler.setup(
            csv_path, forecast_interface, self.agent_id, current_date,
            single_agent_mode=self.config.single_agent_mode
        )
        self._search_handler.set_date(current_date)
        self._forecast_interface = forecast_interface

    def _get_budget_settings(self, *, warmup: bool = False) -> BudgetSettings:
        """Resolve loop-budget settings for day or warmup loops."""
        if warmup:
            return BudgetSettings(
                max_actions=self.config.warmup_max_actions,
                max_total_tokens=self.config.warmup_max_total_tokens,
                submit_reserve_tokens=(
                    self.config.warmup_submit_reserve_tokens
                    or self.config.submit_reserve_tokens
                ),
                force_submit_threshold_tokens=(
                    self.config.warmup_force_submit_threshold_tokens
                    or self.config.force_submit_threshold_tokens
                ),
            )
        return BudgetSettings(
            max_actions=self.config.max_actions,
            max_total_tokens=self.config.max_total_tokens,
            submit_reserve_tokens=self.config.submit_reserve_tokens,
            force_submit_threshold_tokens=self.config.force_submit_threshold_tokens,
        )

    def _create_budget_tracker(
        self,
        *,
        warmup: bool = False,
        max_actions_override: Optional[int] = None,
    ) -> BudgetTracker:
        settings = self._get_budget_settings(warmup=warmup)
        if max_actions_override is not None:
            settings = BudgetSettings(
                max_actions=max_actions_override,
                max_total_tokens=settings.max_total_tokens,
                submit_reserve_tokens=settings.submit_reserve_tokens,
                force_submit_threshold_tokens=settings.force_submit_threshold_tokens,
            )
        return BudgetTracker(settings)

    def _build_budget_overview(self, *, warmup: bool = False, per_question: bool = False) -> str:
        """Human-readable budget instructions for prompts."""
        settings = self._get_budget_settings(warmup=warmup)
        lines: List[str] = []
        if settings.max_actions is not None:
            if per_question:
                lines.append(f"You have {settings.max_actions} actions to research and forecast this question.")
            else:
                lines.append(
                    f"You have {settings.max_actions} actions per day. Each query, search, or submission uses 1 action."
                )
        if settings.max_total_tokens is not None:
            scope = "this question loop" if per_question else "this day"
            lines.append(
                f"You have a total token budget of {settings.max_total_tokens} tokens for {scope}. "
                "Every model call consumes its full input + output tokens from this budget."
            )
            lines.append(
                f"Keep at least {settings.submit_reserve_tokens} tokens in reserve for a final submit. "
                f"Force-submit once the remaining token budget is at or below {settings.force_submit_threshold_tokens}."
            )
        if settings.max_actions is not None and settings.max_total_tokens is not None:
            lines.append("If both budgets are configured, both are enforced and the loop ends when either one is exhausted.")
        return "\n".join(lines)

    def _build_start_budget_status(
        self,
        *,
        warmup: bool = False,
        max_actions_override: Optional[int] = None,
    ) -> str:
        """Render the initial remaining-budget status for prompt seeds."""
        tracker = self._create_budget_tracker(
            warmup=warmup,
            max_actions_override=max_actions_override,
        )
        return tracker.status_text()

    def _build_force_submit_preamble(self, budget: BudgetTracker) -> str:
        """Shared force-submit wording for action/token-constrained loops."""
        lines = [
            "FINAL ACTION: You MUST submit your best guess forecast now."
        ]
        status = budget.status_text()
        if status:
            lines.append(status)
        return "\n".join(lines)

    def _search_results_description(self) -> str:
        chunk_tokens = self._search_handler.chunk_tokens
        if chunk_tokens is None:
            return f"Search returns up to {self.config.max_search_results} retrieved article chunks."
        return (
            f"Search returns up to {self.config.max_search_results} retrieved article chunks, "
            f"each roughly {chunk_tokens} tokens long."
        )
    
    # =========================================================================
    # Action Loop
    # =========================================================================

    def _run_action_loop(self, messages: List[Dict], forecast_interface) -> List[Dict]:
        """
        Main action loop: process agent responses until the loop budget is exhausted or the day ends.
        
        Returns list of all submitted forecasts.
        """
        budget = self._create_budget_tracker()
        all_forecasts = []
        empty_retries = 0
        max_empty_retries = 2

        while not budget.is_exhausted():
            # Get agent response (track LLM time)
            try:
                with self._timer.track("llm"):
                    response, usage = self.inference.chat(messages, self.config.sampling_params)
            except Exception as e:
                print(f"  [{self.agent_id}] LLM error: {e}, ending turn")
                self._log_action(
                    forecast_interface,
                    messages,
                    "",
                    "llm_error",
                    budget,
                    error=str(e),
                )
                break
            
            # Handle empty response (API failure with graceful fallback)
            if not response or not response.strip():
                # Retry empty responses a few times before giving up
                if empty_retries < max_empty_retries:
                    empty_retries += 1
                    print(f"  [{self.agent_id}] Empty LLM response, retrying ({empty_retries}/{max_empty_retries})")
                    continue
                print(f"  [{self.agent_id}] Empty LLM response after {max_empty_retries} retries, ending turn")
                self._log_action(forecast_interface, messages, response or "", "api_failure", budget)
                break
            
            # Successful response — reset empty retry counter
            empty_retries = 0

            # Extract reasoning
            reasoning = usage.get("_reasoning_content") if usage else None

            # Record token usage and cost
            self._timer.record_tokens(usage)
            self._timer.record_cost(usage.get("cost", 0), "llm")
            budget.record_usage(usage)

            messages.append({"role": "assistant", "content": response})

            # Track QIDs mentioned in reasoning/code (for active memory expansion)
            if hasattr(self, '_day_qids'):
                self._day_qids.update(re.findall(r'(?:qid|QID)\W{0,10}(\d+)', response))

            # Parse and handle the action
            parsed = parse_action(response, self.config.max_outcomes_per_question)
            
            if parsed.action_type == "next":
                self._log_action(forecast_interface, messages, response, "next_day", budget, reasoning=reasoning)
                break
            
            elif parsed.action_type == "query":
                self._handle_query(
                    messages, forecast_interface, response, parsed, budget, reasoning=reasoning
                )
            
            elif parsed.action_type == "search":
                self._handle_search(
                    messages, forecast_interface, response, parsed, budget, reasoning=reasoning
                )
            
            elif parsed.action_type == "submit":
                forecasts = self._handle_submit(
                    messages, forecast_interface, response, parsed, budget, reasoning=reasoning
                )
                all_forecasts.extend(forecasts)
            
            else:
                self._handle_invalid(
                    messages, forecast_interface, response, parsed, budget, reasoning=reasoning
                )
        
        return all_forecasts
    
    # =========================================================================
    # Action Handlers
    # =========================================================================
    
    def _handle_query(self, messages, forecast_interface, response, parsed, budget: BudgetTracker, qid: str = None, reasoning=None) -> None:
        """Handle query action."""
        budget.consume_action()
        
        if parsed.code:
            extra_ctx = None
            if isinstance(self._memory, ActiveMemory):
                extra_ctx = {"memo_df": self._memory.get_memo_df()}
            with self._timer.track("df_query"):
                result, error = self._query_handler.execute(parsed.code, extra_context=extra_ctx)
            self._log_action(forecast_interface, messages, response, "query", budget, qid=qid, reasoning=reasoning)
            
            if error:
                feedback = f"QUERY ERROR: {error}"
            else:
                feedback = f"QUERY RESULT:\n{result}"
        else:
            self._log_action(forecast_interface, messages, response, "query_error", budget, qid=qid, error=parsed.error, reasoning=reasoning)
            feedback = f"ERROR: {parsed.error}"
        
        messages.append({"role": "user", "content": budget.format_feedback(feedback)})
    
    def _handle_search(self, messages, forecast_interface, response, parsed, budget: BudgetTracker, qid: str = None, reasoning=None) -> None:
        """Handle search action. Returns full chunk content directly."""
        budget.consume_action()
        
        if not self._search_handler.is_available:
            self._log_action(forecast_interface, messages, response, "search_unavailable", budget, qid=qid, reasoning=reasoning)
            feedback = "SEARCH ERROR: Search is not available."
        elif parsed.query:
            # Parse date range if provided
            min_date = None
            max_date = None
            if parsed.search_from:
                try:
                    from datetime import datetime
                    min_date = datetime.strptime(parsed.search_from, "%Y-%m-%d").date()
                except ValueError:
                    pass
            if parsed.search_to:
                try:
                    from datetime import datetime
                    max_date = datetime.strptime(parsed.search_to, "%Y-%m-%d").date()
                except ValueError:
                    pass
            
            with self._timer.track("search"):
                result, error = self._search_handler.search(
                    parsed.query, 
                    max_results=self.config.max_search_results,
                    min_date=min_date,
                    max_date=max_date,
                )
            self._log_action(forecast_interface, messages, response, "search", budget, qid=qid, reasoning=reasoning)
            
            if error:
                feedback = f"SEARCH ERROR: {error}"
            else:
                feedback = f"SEARCH RESULTS:\n{result}"
        else:
            self._log_action(forecast_interface, messages, response, "search_error", budget, qid=qid, error=parsed.error, reasoning=reasoning)
            feedback = "SEARCH ERROR: No query provided."
        
        messages.append({"role": "user", "content": budget.format_feedback(feedback)})
    
    def _handle_submit(self, messages, forecast_interface, response, parsed, budget: BudgetTracker, qid: str = None, reasoning=None) -> List:
        """Handle submit action. Returns list of submitted forecasts."""
        submitted = []
        budget.consume_action()
        dropped_forecasts = 0
        
        # For logging: use provided qid, or infer from forecasts
        log_qid = qid
        if not log_qid and parsed.forecasts and len(parsed.forecasts) == 1:
            log_qid = parsed.forecasts[0]['qid']
        
        if parsed.forecasts:
            # Enforce single-qid submit: one <forecast ...> block per submit action.
            if len(parsed.forecasts) > 1:
                dropped_forecasts = len(parsed.forecasts) - 1
                parsed.forecasts = [parsed.forecasts[0]]

            for f in parsed.forecasts:
                try:
                    pred = PredictionSubmission(question_id=f['qid'], outcomes=f['outcomes'])
                    forecast_interface.submit_prediction(pred)
                    submitted.append(f)
                    outcomes_str = ", ".join(f"{k}: {v:.2f}" for k, v in f['outcomes'].items())
                    print(f"  [{self.agent_id}] Forecast {f['qid']}: {outcomes_str}")
                except Exception as e:
                    print(f"  [{self.agent_id}] Failed to submit {f['qid']}: {e}")

            if submitted:
                # Ensure later same-day df queries reflect newly submitted predictions.
                self._query_handler.invalidate_cache()
            
            # Include submitted qids in log metadata
            submitted_qids = [f['qid'] for f in submitted]
            if hasattr(self, '_day_qids'):
                self._day_qids.update(str(q) for q in submitted_qids)
            self._log_action(forecast_interface, messages, response, "submit", budget, qid=log_qid, submitted_qids=submitted_qids,
                           num_forecasts=len(submitted), dropped_forecasts=dropped_forecasts, reasoning=reasoning)
            if submitted:
                sub = submitted[0]
                outcomes_str = ", ".join(f"{k}: {v:.2f}" for k, v in sub['outcomes'].items())
                title = self._query_handler.get_question_title(sub['qid'])
                title_str = f" ({title})" if title else ""
                feedback = f"Submitted forecast for qid={sub['qid']}{title_str}: {outcomes_str}."
                if dropped_forecasts > 0:
                    feedback += f"\nIgnored {dropped_forecasts} extra forecast block(s); submit exactly one qid per action."
            else:
                feedback = "SUBMIT ERROR: No valid forecast submitted."
        else:
            # Parse error - still consumed action
            self._log_action(forecast_interface, messages, response, "submit_error", budget, qid=log_qid, error=parsed.error, reasoning=reasoning)
            feedback = f"SUBMIT ERROR: {parsed.error}"
        
        messages.append({"role": "user", "content": budget.format_feedback(feedback)})
        return submitted
    
    def _handle_invalid(self, messages, forecast_interface, response, parsed, budget: BudgetTracker, qid: str = None, reasoning=None) -> None:
        """Handle invalid/unknown action."""
        budget.consume_action()
        self._log_action(forecast_interface, messages, response, "invalid", budget, qid=qid, error=parsed.error, reasoning=reasoning)
        
        error_msg = parsed.error or 'Use <action type="...">...</action> format.'
        feedback = f"No valid action found. {error_msg}"
        messages.append({"role": "user", "content": budget.format_feedback(feedback)})
    
    # =========================================================================
    # Helpers
    # =========================================================================
    
    def _log_action(self, forecast_interface, messages, response, phase, 
                   budget: BudgetTracker, qid: str = None, **extra) -> None:
        """Log model output with metadata. qid indicates the question context if known."""
        last_user = messages[-2]["content"] if len(messages) >= 2 else ""
        metadata = {
            "phase": phase, 
            "qid": qid,
            **budget.metadata(),
            **extra
        }
        forecast_interface.log_model_output(last_user, response, metadata)

    def _record_matcher_timing(self, duration: float, cost: float = 0) -> None:
        """Record answer matcher latency and cost in timing stats."""
        self._timer.record("matcher", duration)
        self._timer.record_cost(cost, "matcher")
    
    # =========================================================================
    # Instructions & Memory
    # =========================================================================
    # Prompt Helpers
    # =========================================================================

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
- **Peer Score (relative ranking)**: `100 × (avg others' Brier - your Brier)` (baseline 0 if you are the only predictor). Positive means better than peers."""

        mechanics: Dict[str, str] = {
            "accuracy_calibration": "**Accuracy + Calibration**: Assign probabilities that reflect true likelihood.",
            "binary_outcomes": "**Binary Outcomes**: Use exact outcomes \"Yes\" and \"No\".",
            "time_weighted": "**Time-Weighted**: The score is summed over all days the prediction was held, so early predictions have higher weight.",
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
- **Peer Score**: For each snapshot, your Brier Skill Score is compared to the crowd by subtracting the average of other agents' Brier Skill Scores (baseline 0 if you are the only predictor)."""

        mechanics: Dict[str, str] = {
            "accuracy_calibration": "**Accuracy + Calibration**: Assign high probability to the TRUE outcome and keep probabilities well-calibrated.",
            "time_weighted": "**Time-Weighted**: The score is summed over all days the prediction was held, so early predictions have higher weight.",
            "question_count": "**Prediction-Count Incentive**: Scores are summed (not averaged) across all questions you predict on.",
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
You are evaluated on **Brier Skill Score** = 1 - Σ(pᵢ - yᵢ)² summed over all outcomes.
- pᵢ = your probability for outcome i
- yᵢ = 1 if outcome i is TRUE, 0 otherwise
- **Higher is better**: 1.0 = perfect, 0.0 = no skill, negative = worse than uniform.{peer_text}

Key Mechanics:
{mechanics_text}
"""

    def _get_scoring_section(self) -> str:
        """Get scoring description - single vs multi-agent mode."""
        source_name = getattr(getattr(self, "_forecast_interface", None), "source_name", "openforesight")
        if source_name == "metaculus_binary":
            return self._build_binary_brier_scoring_section()
        return self._build_brier_skill_scoring_section()
    
    def _get_data_notes(self) -> str:
        """Get notes about DataFrame columns - conditional on agent mode."""
        if self.config.single_agent_mode:
            # Single-agent: market_aggregate just reflects own predictions, don't mention it
            return "Note: `my_prediction` column contains your current forecast as a dict (or None if not yet predicted)."
        else:
            # Multi-agent: market_aggregate is meaningful
            return """Note: `market_aggregate` and `my_prediction` columns contain Python dicts (or None). You can access them directly, e.g. `row['market_aggregate']['outcome_name']`.
The `num_predictions` column shows how many predictions have been made on that question."""
    
    def _get_source_rules(self) -> str:
        """Get source-specific submission rules."""
        source_name = getattr(self._forecast_interface, 'source_name', 'openforesight')
        
        if source_name == "metaculus_binary":
            return """
## BINARY QUESTION RULES
All questions are Yes/No binary. Your prediction MUST use exactly:
- **"Yes"** for the affirmative outcome
- **"No"** for the negative outcome

Example submission:
<forecast qid="12345">
  <outcome name="Yes" prob="0.7"/>
  <outcome name="No" prob="0.3"/>
</forecast>
"""
        elif source_name == "metaculus_mcq":
            return """
## MULTIPLE CHOICE RULES
Each question has enumerated options shown in the 'options' column.
Your prediction MUST use the EXACT option text from the question.
Do NOT paraphrase or abbreviate options.

Example (if options are ["Candidate A", "Candidate B", "Candidate C"]):
<forecast qid="12345">
  <outcome name="Candidate A" prob="0.5"/>
  <outcome name="Candidate B" prob="0.3"/>
  <outcome name="Candidate C" prob="0.2"/>
</forecast>
"""
        return ""

    def _build_instructions(self, current_date: date) -> str:
        """Build the instructions for the agent including data info and rules."""
        df_info = self._query_handler.get_info()
        budget_start_status = self._build_start_budget_status()
        budget_start_block = f"Budget at start:\n{budget_start_status}\n\n" if budget_start_status else ""
        
        memory_section = ""
        if self._memory is not None:
            memory_content = self._memory.get()
            if isinstance(self._memory, ActiveMemory):
                # Active memory: meta-insights inline + memo_df documentation
                meta_part = ""
                if memory_content:
                    meta_part = f"""### Meta-Insights ({self._memory.entry_count} entries)
<memory>
{memory_content}
</memory>

"""
                memo_count = self._memory.memo_count
                memory_section = f"""## YOUR MEMORY
{meta_part}### Question-Specific Memory (memo_df: {memo_count} entries)
You have a DataFrame `memo_df` with your question-specific notes (reasoning, evidence, confidence).
Query it using the "query" action alongside `df`. Both DataFrames are available in the same sandbox.
Columns: qid (str), question (str), last_updated (str), memory (str), confidence (float), category (str)

Example:
```python
print(memo_df[memo_df['category'] == 'sports'][['qid', 'memory']].head())
```

Use meta-insights and memo_df to inform today's forecasts. Entry IDs in brackets (e.g. [a1b2c3d4]) can be used to delete stale meta-insight entries at end of day.

"""
            elif memory_content:
                if isinstance(self._memory, StructuredMemory):
                    memory_section = f"""## YOUR MEMORY ({self._memory.entry_count} entries)
<memory>
{memory_content}
</memory>

Use these stored insights to inform today's forecasts. Entry IDs in brackets (e.g. [a1b2c3d4]) can be used to delete stale entries at end of day. The DataFrame includes your latest predictions for active questions and your final prediction snapshot on resolved questions.

"""
                else:
                    memory_section = f"""## YOUR MEMORY (reasoning, patterns, and insights from previous days)
<memory>
{memory_content}
</memory>

Use the reasoning and insights above to inform today's forecasts. The DataFrame includes your latest predictions for active questions and your final prediction snapshot on resolved questions.

"""
            memory_flow_note = "Note: After ending your day, you will be prompted to update your memory."
        else:
            memory_flow_note = ""
        
        # Search section (only if enabled)
        search_section = ""
        search_advice = ""
        end_day_num = 3  # Default without search
        submit_num = 2   # Default without search
        if self._search_handler.is_available:
            end_day_num = 4  # Shift to 4 when search is enabled
            submit_num = 3
            
            cutoff_desc = "today's date"
            if self.config.search_cutoff_days > 0:
                cutoff_date = current_date - timedelta(days=self.config.search_cutoff_days)
                cutoff_desc = f"{cutoff_date} (today - {self.config.search_cutoff_days} days)"

            search_section = f"""
### 2. Search News Articles
<action type="search">
your search query here
</action>

Optional: Specify date range (YYYY-MM-DD format):
<action type="search" from="2024-12-01" to="2024-12-15">
your search query here
</action>

Note: "to" date is capped at {cutoff_desc} (no future leakage).
{self._search_results_description()}
You can use search to gather evidence before submitting forecasts.
"""
            search_advice = f"""

You have access to a news article database. {self._search_results_description()}"""
        
        return f"""You are a forecasting agent. Today is {current_date}. Your goal: make accurate probability predictions.

{self._feedback_handler.format_feedback(
    self._feedback_handler.generate_feedback(self._forecast_interface, current_date, self.inference),
    show_tw_peer=not self.config.single_agent_mode,
)}

{getattr(self._forecast_interface, 'source_context', '')}

{self._get_source_rules()}

{memory_section}{self._get_scoring_section()}
{search_advice}
## AVAILABLE DATA
DataFrame `df` with {df_info['n_rows']} questions ({df_info['n_active']} active/unresolved, {df_info['n_resolved']} resolved).

Column descriptions:
{df_info['columns_desc']}

{self._get_data_notes()}

## CODE EXECUTION ENVIRONMENT
Your Python code runs in a sandbox with these variables pre-defined:
- `df`: the pandas DataFrame
- `pd`: pandas module
- `today`: date object for {current_date}
- `date`, `datetime`, `timedelta`: from datetime module
{('- `memo_df`: your question-specific memory DataFrame (query with print())' + chr(10)) if isinstance(self._memory, ActiveMemory) else ''}Import statements are not available. Standard builtins (len, str, int, float, min, max, sum, sorted, range, etc.) are available.

## RESPONSE FORMAT
Every response must have <reasoning> followed by exactly ONE <action> block.

<reasoning>
Your analysis
</reasoning>
<action type="ACTION_TYPE">
...content based on action type...
</action>

IMPORTANT:
- Only ONE <action> block allowed per turn.
- For "search", this means only 1 search query per turn.
- For "query", you can write complex multi-step Python code in one block.
- For "submit", include exactly one <forecast ...> block (one qid only).

## ACTION TYPES

### 1. Query Questions (explore data)
<action type="query">
```python
print(df[df['is_resolved'] == False][['qid', 'title', 'answer_type']].head())
```
</action>
Use `print()` to ensure you see output. We are not executing in a jupyter notebook, so .head() preview alone can be unreliable.
{search_section}
### {submit_num}. Submit Forecast
<action type="submit">
<forecast qid="QUESTION_ID">
  <outcome name="Answer1" prob="0.5"/>
  <outcome name="Answer2" prob="0.3"/>
</forecast>
</action>

### {end_day_num}. End Day (proceed to next day)
<action type="next"/>

## INTERACTION FLOW
{self._build_budget_overview()}
You can interleave queries, searches, and submissions as needed.
When ready to move on, use <action type="next"/> to end your day.
{memory_flow_note}

## SUBMISSION RULES
- qid must be from an active (is_resolved=False) question you saw in query results
- In one submit action, include exactly one <forecast ...> block for one qid.
- You can submit again in later turns to update that qid.
- Max {self.config.max_outcomes_per_question} outcomes per question.
- Outcome names must be REAL predicted answers (e.g., person names, locations, numbers)
- NEVER use placeholders like "Unknown", "TBD", "Other", "N/A" - these ALWAYS hurt your score
- Probabilities must sum to ≤ 1.0

---
{budget_start_block}Begin."""
    
    def _prompt_memory_update(self, messages: List[Dict[str, str]],
                               forecast_interface, current_date: date) -> None:
        """
        Ask the agent to update its memory at the end of the day.

        For StructuredMemory: uses add/delete operations on individual entries.
        For BasicMemory (legacy): uses full replacement via <memory> tags.
        """
        if isinstance(self._memory, ActiveMemory):
            memory_prompt = self._build_active_memory_prompt(current_date, day_qids=getattr(self, '_day_qids', None))
        elif isinstance(self._memory, StructuredMemory):
            memory_prompt = self._build_structured_memory_prompt(current_date)
        else:
            memory_prompt = self._build_plain_memory_prompt(current_date)

        # Prepend privileged cheat-feedback when enabled
        if self.config.cheat_feedback:
            cheat_data = forecast_interface.get_cheat_feedback(
                detail=self.config.cheat_feedback_detail
            )
            if cheat_data.get("items"):
                cheat_section = FeedbackHandler.format_cheat_feedback(
                    cheat_data, self.config.cheat_feedback_detail
                )
                memory_prompt = cheat_section + "\n\n" + memory_prompt

        messages.append({"role": "user", "content": memory_prompt})
        response, usage = self.inference.chat(messages, self.config.sampling_params)
        self._timer.record_tokens(usage)
        self._timer.record_cost(usage.get("cost", 0), "llm")
        messages.append({"role": "assistant", "content": response})

        reasoning = usage.get("_reasoning_content") if usage else None

        forecast_interface.log_model_output(
            memory_prompt, response,
            {"phase": "memory_update", "current_memory_len": len(self._memory), "reasoning": reasoning}
        )

        if isinstance(self._memory, ActiveMemory):
            self._apply_active_memory_ops(response, current_date)
        elif isinstance(self._memory, StructuredMemory):
            self._apply_structured_memory_ops(response)
        else:
            new_memory = extract_memory(response)
            if new_memory is not None:
                self._memory.update(new_memory)

    def _build_structured_memory_prompt(self, current_date: date) -> str:
        """Build the structured memory update prompt (add/delete entries)."""
        return f"""End of day {current_date}. You can now update your memory.

## MEMORY UPDATE
Your memory entries are the ONLY context retained between days. Everything else resets. Tomorrow you get: search over news articles and the DataFrame (active question predictions, resolved question ground truths, your final predictions on resolved questions).

You currently have {self._memory.entry_count} memory entries ({len(self._memory)} chars). Max 30 entries.

### What to store (add new entries for):
1. **reasoning** — Why you made specific predictions, especially when based on hard-to-find evidence. Example: "Q149: PSG 0.70 because Sky Bet implied 55% and Inter eliminated in semis."
2. **calibration** — Performance patterns across resolved questions. Example: "Bookmaker odds correct 80% across 15 sports Qs; weight them more."
3. **insight** — Non-obvious patterns that search alone would not surface. Example: "'First country to X' Qs almost always resolve to a major economy."
4. **fact** — Critical hard-to-find facts relevant to active questions. Example: "ECB next meeting June 5 — relevant to Q72, Q108."

### What NOT to store:
General forecasting advice (already in instructions), easily searchable facts, prediction outcomes without reasoning.

### How to update:
Add entries (you may add multiple):
<memory_add>
name: Short descriptive title (max 150 chars, include question IDs if relevant)
type: reasoning|calibration|insight|fact
qids: Q72, Q108 (comma-separated, or leave empty)
content: Your content here (max 800 chars). Include specific numbers, sources, and reasoning.
</memory_add>

Delete stale entries by ID (you may delete multiple):
<memory_delete>ENTRY_ID_HERE</memory_delete>

First reflect on today's session, then add/delete entries as needed:
<reasoning>
Reflect on today's forecasting session...
</reasoning>

If no updates needed, just output <reasoning>...</reasoning> without any memory tags."""

    def _build_plain_memory_prompt(self, current_date: date) -> str:
        """Build the legacy plain-text memory update prompt (full replacement)."""
        return f"""End of day {current_date}. You can now update your memory.

## MEMORY UPDATE
Your memory is the ONLY thing that carries over to tomorrow. Everything else resets. Tomorrow you get: search over news articles and access to the DataFrame (active question predictions, resolved question ground truths, and your final predictions on resolved questions).

Store things NOT recoverable from those tools:
1. Reasoning behind predictions and how you did on resolved questions that might help with unresolved questions — once a question resolves, your prediction remains visible in the dataframe, but your reasoning is never stored in the dataframe. Example: "Q149: PSG 0.70 because Sky Bet implied 55% and Inter eliminated in semis."
2. Performance patterns — track your accuracy across resolved questions so you can calibrate. Example: "Bookmaker odds were correct 80% across 15 sports questions; I should weight them more."
3. Non-obvious insights that search alone would not surface. Example: "'First country to X' questions almost always resolve to a major economy."
4. Critical hard-to-find facts directly relevant to active questions. Example: "ECB next meeting June 5 — relevant to Q72, Q108."

Do NOT store: general forecasting advice (already in your instructions), easily searchable facts, prediction outcomes without reasoning, or vague tracking lists without reasoning.
Aim to keep memory under 2000 characters. Prioritize recent and high-impact items and drop stale entries about resolved questions you have already learned from.

To update memory, include it after your reasoning:
<reasoning>
Reflect on today's forecasting session...
</reasoning>
<memory>
Your updated memory content here (complete replacement, not a diff)
</memory>

If you don't want to update memory, just output <reasoning>...</reasoning> without <memory> tags.
Current memory length: {len(self._memory)} characters"""

    def _apply_structured_memory_ops(self, response: str) -> None:
        """Apply structured memory operations from agent response, with fallback."""
        adds, deletes = extract_memory_ops(response)
        if adds or deletes:
            for entry_id in deletes:
                self._memory.delete_entry(entry_id)
            for add in adds:
                self._memory.add_entry(
                    add["name"], add["type"], add.get("qids", ""), add["content"]
                )
        else:
            # Fallback: try old-style <memory> full replacement
            old_memory = extract_memory(response)
            if old_memory is not None:
                self._memory.update(old_memory)

    # =========================================================================
    # Active Memory (memo_df + reduced meta-insights)
    # =========================================================================

    def _build_active_memory_prompt(self, current_date: date, day_qids: set = None) -> str:
        """Build the active memory update prompt (memo_df + meta-insights)."""
        memo_summary = self._memory.memo_summary(expanded_qids=day_qids)
        meta_content = self._memory.get()
        meta_section = ""
        if meta_content:
            meta_section = f"""Current meta-insight entries:
<memory>
{meta_content}
</memory>

"""
        return f"""End of day {current_date}. You can now update your memory.

Your memory has two layers, both retained between days. Everything else resets.
Tomorrow you get: search over news articles and the DataFrame (active question predictions, resolved question ground truths, your final predictions on resolved questions).

## 1. QUESTION-SPECIFIC NOTES (memo_df: {self._memory.memo_count} entries)

Current entries:
{memo_summary}

Store question-specific reasoning, evidence, and confidence. Be concise — max 500 chars per entry.
Only store what is NOT recoverable from the DataFrame or search (e.g., your reasoning, hard-to-find evidence, calibration notes for specific questions).

Add or update entries (you may add/update as many as needed):
<memo_add>
qid: Q123
question: Will X happen by Y?
memory: Your reasoning and key evidence (max 500 chars)
confidence: 0.7
category: politics
</memo_add>

Update an existing entry (only changed fields needed alongside memory):
<memo_update qid="Q123">
memory: Updated reasoning with new evidence (max 500 chars)
confidence: 0.8
</memo_update>

Delete a stale entry:
<memo_delete qid="Q123"/>

## 2. META-INSIGHTS ({self._memory.entry_count}/15 entries, {len(self._memory)} chars)

Cross-question patterns and calibration notes. NOT for question-specific reasoning (use memo_df for that).
{meta_section}Update meta-insights:
<memory_add>
name: Short descriptive title (max 150 chars)
type: reasoning|calibration|insight|fact
qids: Q72, Q108 (comma-separated, or leave empty)
content: Your content here (max 400 chars). Focus on cross-question patterns.
</memory_add>

<memory_delete>ENTRY_ID_HERE</memory_delete>

First reflect on today's session, then update both layers as needed:
<reasoning>
Reflect on today's forecasting session...
</reasoning>

If no updates needed, just output <reasoning>...</reasoning> without any memory tags."""

    def _apply_active_memory_ops(self, response: str, current_date: date) -> None:
        """Apply active memory operations from agent response."""
        # Parse memo_df operations
        memo_adds, memo_updates, memo_deletes = extract_memo_ops(response)

        for qid in memo_deletes:
            self._memory.memo_delete(qid)
        for add in memo_adds:
            self._memory.memo_add(
                qid=add["qid"],
                question=add.get("question", ""),
                memory=add["memory"],
                confidence=add.get("confidence"),
                category=add.get("category", ""),
            )
        for upd in memo_updates:
            self._memory.memo_update(
                qid=upd["qid"],
                memory=upd["memory"],
                confidence=upd.get("confidence"),
                category=upd.get("category"),
            )

        # Parse meta-insight operations (reuse existing parser)
        meta_adds, meta_deletes = extract_memory_ops(response)
        for entry_id in meta_deletes:
            self._memory.delete_entry(entry_id)
        for add in meta_adds:
            self._memory.add_entry(
                add["name"], add["type"], add.get("qids", ""), add["content"]
            )

        # Persist both layers
        self._memory.save(current_date)
