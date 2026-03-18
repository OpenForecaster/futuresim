"""
BasicAgent: Main agent class for LLM-based forecasting.

Uses chain-of-thought with <reasoning> and <action type="..."> tags.
"""

from datetime import date, timedelta
from typing import List, Dict, Any, Optional, Tuple

from agents.base import BaseAgent
from agents.utils.forecast_parser import parse_action, extract_memory, extract_memory_ops
from agents.utils.timing import AgentTimer
from environment.interfaces import PredictionSubmission

from .config import AgentConfig
from .memory import BasicMemory
from agents.utils.memory import StructuredMemory
from .query import QueryHandler
from .search import SearchHandler
from .feedback import FeedbackHandler


class BasicAgent(BaseAgent):
    """
    Basic forecasting agent using LLM inference.
    
    Interaction flow per day:
    1. Receives system prompt with DataFrame schema and scoring rules
    2. Can take up to max_actions per day, including:
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
            if self.config.memory_format == "structured":
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
    
    # =========================================================================
    # Action Loop
    # =========================================================================

    def _run_action_loop(self, messages: List[Dict], forecast_interface) -> List[Dict]:
        """
        Main action loop: process agent responses until actions exhausted or day ended.
        
        Returns list of all submitted forecasts.
        """
        actions_remaining = self.config.max_actions
        all_forecasts = []
        empty_retries = 0
        max_empty_retries = 2
        
        while actions_remaining > 0:
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
                    actions_remaining,
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
                self._log_action(forecast_interface, messages, response or "", "api_failure", actions_remaining)
                break
            
            # Successful response — reset empty retry counter
            empty_retries = 0

            # Extract reasoning
            reasoning = usage.get("_reasoning_content") if usage else None

            # Record token usage and cost
            self._timer.record_tokens(usage)
            self._timer.record_cost(usage.get("cost", 0), "llm")

            messages.append({"role": "assistant", "content": response})
            
            # Parse and handle the action
            parsed = parse_action(response, self.config.max_outcomes_per_question)
            
            if parsed.action_type == "next":
                self._log_action(forecast_interface, messages, response, "next_day", actions_remaining, reasoning=reasoning)
                break
            
            elif parsed.action_type == "query":
                actions_remaining = self._handle_query(
                    messages, forecast_interface, response, parsed, actions_remaining, reasoning=reasoning
                )
            
            elif parsed.action_type == "search":
                actions_remaining = self._handle_search(
                    messages, forecast_interface, response, parsed, actions_remaining, reasoning=reasoning
                )
            
            elif parsed.action_type == "submit":
                actions_remaining, forecasts = self._handle_submit(
                    messages, forecast_interface, response, parsed, actions_remaining, reasoning=reasoning
                )
                all_forecasts.extend(forecasts)
            
            else:
                actions_remaining = self._handle_invalid(
                    messages, forecast_interface, response, parsed, actions_remaining, reasoning=reasoning
                )
        
        return all_forecasts
    
    # =========================================================================
    # Action Handlers
    # =========================================================================
    
    def _handle_query(self, messages, forecast_interface, response, parsed, actions_remaining, qid: str = None, reasoning=None) -> int:
        """Handle query action. Returns updated actions_remaining."""
        actions_remaining -= 1
        
        if parsed.code:
            with self._timer.track("df_query"):
                result, error = self._query_handler.execute(parsed.code)
            self._log_action(forecast_interface, messages, response, "query", actions_remaining, qid=qid, reasoning=reasoning)
            
            if error:
                feedback = f"QUERY ERROR: {error}\n\nActions remaining: {actions_remaining}"
            else:
                feedback = f"QUERY RESULT:\n{result}\n\nActions remaining: {actions_remaining}"
        else:
            self._log_action(forecast_interface, messages, response, "query_error", 
                           actions_remaining, qid=qid, error=parsed.error, reasoning=reasoning)
            feedback = f"ERROR: {parsed.error}\n\nActions remaining: {actions_remaining}"
        
        feedback = self._add_exhaustion_warning(feedback, actions_remaining)
        messages.append({"role": "user", "content": feedback})
        return actions_remaining
    
    def _handle_search(self, messages, forecast_interface, response, parsed, actions_remaining, qid: str = None, reasoning=None) -> int:
        """Handle search action. Returns full chunk content directly."""
        actions_remaining -= 1
        
        if not self._search_handler.is_available:
            self._log_action(forecast_interface, messages, response, "search_unavailable", actions_remaining, qid=qid, reasoning=reasoning)
            feedback = f"SEARCH ERROR: Search is not available.\n\nActions remaining: {actions_remaining}"
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
            self._log_action(forecast_interface, messages, response, "search", actions_remaining, qid=qid, reasoning=reasoning)
            
            if error:
                feedback = f"SEARCH ERROR: {error}\n\nActions remaining: {actions_remaining}"
            else:
                feedback = f"SEARCH RESULTS:\n{result}\n\nActions remaining: {actions_remaining}"
        else:
            self._log_action(forecast_interface, messages, response, "search_error", 
                           actions_remaining, qid=qid, error=parsed.error, reasoning=reasoning)
            feedback = f"SEARCH ERROR: No query provided.\n\nActions remaining: {actions_remaining}"
        
        feedback = self._add_exhaustion_warning(feedback, actions_remaining)
        messages.append({"role": "user", "content": feedback})
        return actions_remaining
    
    def _handle_submit(self, messages, forecast_interface, response, parsed, actions_remaining, qid: str = None, reasoning=None) -> Tuple[int, List]:
        """Handle submit action. Returns (updated actions_remaining, list of submitted forecasts)."""
        submitted = []
        actions_remaining -= 1  # Always consume action
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
            self._log_action(forecast_interface, messages, response, "submit", 
                           actions_remaining, qid=log_qid, submitted_qids=submitted_qids,
                           num_forecasts=len(submitted), dropped_forecasts=dropped_forecasts, reasoning=reasoning)
            if submitted:
                # feedback = f"Submitted forecast for qid={sub['qid']}. Actions remaining: {actions_remaining}"
                sub = submitted[0]
                outcomes_str = ", ".join(f"{k}: {v:.2f}" for k, v in sub['outcomes'].items())
                title = self._query_handler.get_question_title(sub['qid'])
                title_str = f" ({title})" if title else ""
                feedback = f"Submitted forecast for qid={sub['qid']}{title_str}: {outcomes_str}. Actions remaining: {actions_remaining}"
                if dropped_forecasts > 0:
                    feedback += f"\nIgnored {dropped_forecasts} extra forecast block(s); submit exactly one qid per action."
            else:
                feedback = f"SUBMIT ERROR: No valid forecast submitted.\n\nActions remaining: {actions_remaining}"
        else:
            # Parse error - still consumed action
            self._log_action(forecast_interface, messages, response, "submit_error",
                           actions_remaining, qid=log_qid, error=parsed.error, reasoning=reasoning)
            feedback = f"SUBMIT ERROR: {parsed.error}\n\nActions remaining: {actions_remaining}"
        
        feedback = self._add_exhaustion_warning(feedback, actions_remaining)
        messages.append({"role": "user", "content": feedback})
        return actions_remaining, submitted
    
    def _handle_invalid(self, messages, forecast_interface, response, parsed, actions_remaining, qid: str = None, reasoning=None) -> int:
        """Handle invalid/unknown action. Returns updated actions_remaining."""
        actions_remaining -= 1
        self._log_action(forecast_interface, messages, response, "invalid", 
                        actions_remaining, qid=qid, error=parsed.error, reasoning=reasoning)
        
        error_msg = parsed.error or 'Use <action type="...">...</action> format.'
        feedback = f"No valid action found. {error_msg}\n\nActions remaining: {actions_remaining}"
        feedback = self._add_exhaustion_warning(feedback, actions_remaining)
        messages.append({"role": "user", "content": feedback})
        return actions_remaining
    
    # =========================================================================
    # Helpers
    # =========================================================================
    
    def _log_action(self, forecast_interface, messages, response, phase, 
                   actions_remaining, qid: str = None, **extra) -> None:
        """Log model output with metadata. qid indicates the question context if known."""
        last_user = messages[-2]["content"] if len(messages) >= 2 else ""
        metadata = {
            "phase": phase, 
            "actions_remaining": actions_remaining, 
            "qid": qid,
            **extra
        }
        forecast_interface.log_model_output(last_user, response, metadata)
    
    def _add_exhaustion_warning(self, feedback: str, actions_remaining: int) -> str:
        """Add warning if actions are exhausted."""
        if actions_remaining == 0:
            feedback += "\n\nNo more actions available. Your day ends now."
        return feedback

    def _record_matcher_timing(self, duration: float, cost: float = 0) -> None:
        """Record answer matcher latency and cost in timing stats."""
        self._timer.record("matcher", duration)
        self._timer.record_cost(cost, "matcher")

    def _build_calibration_summary(self) -> str:
        """
        Build a brief performance summary. Kept minimal to avoid
        inducing self-doubt or probability-capping behavior.
        """
        fh = self._feedback_handler
        if fh.total_resolved_count == 0:
            return ""

        accuracy = fh.total_accuracy_count / fh.total_resolved_count * 100

        recent = getattr(fh, '_last_resolved', [])
        today_str = f" ({len(recent)} resolved today)" if recent else ""

        return f"\n{fh.total_resolved_count} questions resolved so far{today_str}, accuracy {accuracy:.1f}%."

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
        
        memory_section = ""
        if self._memory is not None:
            memory_content = self._memory.get()
            if memory_content:
                if isinstance(self._memory, StructuredMemory):
                    memory_section = f"""## YOUR MEMORY ({self._memory.entry_count} entries)
<memory>
{memory_content}
</memory>

**How to use your memory today:**
- **reasoning** entries: These explain WHY you predicted something with key evidence. If the question is still active, use this to stay consistent or update based on new info. This saves you from re-searching.
- **insight** entries: Transferable patterns you discovered. Apply these when making predictions on similar questions.
- **fact** entries: Hard-to-find facts relevant to active questions. Use them directly.

**CRITICAL — Outcome consolidation before every prediction:**
Before submitting any prediction, check if your outcomes contain duplicates or near-duplicates. Common patterns:
- Acronym vs full name: "WFP" and "World Food Programme" are the SAME outcome — merge them.
- Hyphenation variants: "counter-terrorism" and "counterterrorism" are the SAME — merge them.
- Name variations: "SAVE Act" and "Safeguard American Voter Eligibility Act" — merge them.
- Person + title: "Pope Leo XIV" and "Cardinal Prevost" (if same person) — merge them.
Splitting probability across duplicate outcomes DESTROYS your Brier score. Always merge before submitting.

Entry IDs in brackets (e.g. [a1b2c3d4]) can be used to delete stale entries at end of day.

"""
                else:
                    memory_section = f"""## YOUR MEMORY (reasoning, patterns, and insights from previous days)
<memory>
{memory_content}
</memory>

Use the reasoning and insights above to inform today's forecasts. The DataFrame includes your latest predictions for active questions and your final prediction snapshot on resolved questions.

CRITICAL - Outcome consolidation: Before every prediction, merge duplicate outcomes (acronym vs full name, hyphenation variants, name variations). Splitting probability across duplicates destroys Brier score.

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
You can use search to gather evidence before submitting forecasts.
"""
            search_advice = """

You have access to a news article database. You can search it to gather information before making predictions."""
        
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

Import statements are not available. Standard builtins (len, str, int, float, min, max, sum, sorted, range, etc.) are available.

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
You have {self.config.max_actions} actions per day. Each query, search, or submission uses 1 action.
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

You have {self.config.max_actions} actions available. Begin."""
    
    def _prompt_memory_update(self, messages: List[Dict[str, str]],
                               forecast_interface, current_date: date) -> None:
        """
        Ask the agent to update its memory at the end of the day.

        For StructuredMemory: uses add/delete operations on individual entries.
        For BasicMemory (legacy): uses full replacement via <memory> tags.
        """
        if isinstance(self._memory, StructuredMemory):
            memory_prompt = self._build_structured_memory_prompt(current_date)
        else:
            memory_prompt = self._build_plain_memory_prompt(current_date)

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

        if isinstance(self._memory, StructuredMemory):
            self._apply_structured_memory_ops(response)
        else:
            new_memory = extract_memory(response)
            if new_memory is not None:
                self._memory.update(new_memory)

    def _build_structured_memory_prompt(self, current_date: date) -> str:
        """Build the structured memory update prompt (add/delete entries)."""
        return f"""End of day {current_date}. You can now update your memory.

## MEMORY UPDATE
Your memory is the ONLY context that persists between days. Everything else (conversation, searches, reasoning) resets. Tomorrow you get: news search and the DataFrame (with your latest predictions and resolved ground truths).

You currently have {self._memory.entry_count} memory entries ({len(self._memory)} chars). Max 30 entries.

**Your goal is to maximize prediction accuracy and Brier score.** Memory should help you be RIGHT more often, not less confident.

## WHAT TO STORE (prioritized by value)

1. **reasoning** (HIGHEST PRIORITY) — Your evidence chain for active predictions that would take multiple searches to reconstruct. This is your most valuable memory: it prevents wasted search effort and keeps predictions consistent.
   - Good: "Q149: PSG 0.70 — Sky Bet implied 55%, Inter eliminated in semis (May 14 article), PSG home advantage in CL final."
   - Good: "Q24: Riggs 0.85 — Federal judge Myers ordered NC Board to certify by 734 votes (Fox News May 6), 7-day appeal window passed."
   - Bad: "Q149: I think PSG will win." (no evidence)

2. **insight** — Transferable patterns that help you predict CORRECTLY, discovered from resolved questions or cross-question analysis.
   - Good: "Questions asking 'first country to X' resolved to US/China/India in 6/7 cases. Weight major economies heavily."
   - Good: "When 2025 news articles explicitly confirm an outcome, that outcome resolved correctly 9/10 times. Trust direct evidence."
   - Good: "Outcome consolidation: always merge acronym vs full name (WFP=World Food Programme), hyphenation variants (counter-terrorism=counterterrorism), name variants (SAVE Act=Safeguard American Voter Eligibility Act). Splitting probability across duplicates destroys Brier."
   - Bad: "Pay attention to geopolitics." (too generic to act on)
   - Bad: "Probabilities must sum to 1.0." (you already know this — don't waste memory on obvious rules)

3. **fact** — Hard-to-find facts that are relevant to active questions and cannot be easily re-discovered via search.
   - Good: "Qatar gifted Boeing 747-8 to Trump admin (JPost May 11). Relevant to Q314, Q790."
   - Bad: "The Super Bowl is in February." (easily searchable)

### What NOT to store:
- **Basic math/formatting rules** you already know ("probabilities must sum to 1.0", "avoid Other outcomes"). These waste memory slots on things you'll do correctly anyway.
- Generic probability-capping rules (these hurt accuracy)
- Resolved question outcomes without a transferable lesson
- Facts you could re-find with one search query
- General advice ("be more careful", "research more", "verify before submitting")
- Self-doubt or rules that make you less confident without evidence

### How to update:
<memory_add>
name: Short descriptive title (max 150 chars, include question IDs if relevant)
type: reasoning|insight|fact
qids: Q72, Q108 (comma-separated, or leave empty)
content: Your content here (max 800 chars). Include specific numbers, sources, and reasoning.
</memory_add>

Delete stale/resolved entries:
<memory_delete>ENTRY_ID_HERE</memory_delete>

### Deletion guidelines:
- Delete **reasoning** entries when their question has resolved (unless there's a transferable lesson — convert to insight first).
- Delete **fact** entries you could re-find via search.
- Do NOT delete **insight** entries unless they've been proven wrong. Insights compound over time — they are your most durable advantage.
- **Always keep at least 3 entries.** An empty memory wastes the advantage of persistence. If you have fewer than 3 entries, add new ones before deleting.

<reasoning>
Analyze today's session. What evidence and insights should you preserve?
</reasoning>"""

    def _build_plain_memory_prompt(self, current_date: date) -> str:
        """Build the legacy plain-text memory update prompt (full replacement)."""
        calibration_summary = self._build_calibration_summary()
        return f"""End of day {current_date}. You can now update your memory.

## MEMORY UPDATE
Your memory is the ONLY context that persists between days. Everything else resets.
{calibration_summary}
Prioritize storing (in order of value):
1. **Evidence chains** for active predictions that would take multiple searches to reconstruct. E.g., "Q149: PSG 0.70 — Sky Bet implied 55%, Inter eliminated May 14, PSG home advantage."
2. **Transferable insights** — Patterns that help you predict correctly. E.g., "'First country to X' resolved to major economy 6/7 times."
3. **Hard-to-find facts** relevant to active questions.

Do NOT store: probability-capping rules, generic advice, easily searchable facts.
Aim to keep memory under 2000 chars. Drop resolved-question reasoning but keep transferable insights.
**Always keep at least 3 entries.** Insights compound over time — never let memory go empty.

<reasoning>
What evidence and insights should you preserve?
</reasoning>
<memory>
Your updated memory content here (complete replacement)
</memory>

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
