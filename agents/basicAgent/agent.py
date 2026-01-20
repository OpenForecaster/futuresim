"""
BasicAgent: Main agent class for LLM-based forecasting.

Uses chain-of-thought with <reasoning> and <action type="..."> tags.
"""

from datetime import date, timedelta
from typing import List, Dict, Any, Optional, Tuple

from agents.base import BaseAgent
from agents.utils.forecast_parser import parse_action, extract_memory
from agents.utils.timing import AgentTimer
from environment.interfaces import PredictionSubmission

from .config import AgentConfig
from .memory import BasicMemory
from .query import QueryHandler
from .search import SearchHandler


class BasicAgent(BaseAgent):
    """
    Basic forecasting agent using LLM inference.
    
    Interaction flow per day:
    1. Receives system prompt with DataFrame schema and scoring rules
    2. Can take up to max_actions per day, including:
       - query: Execute Python code to explore the DataFrame
       - search: Search news articles for context (if enabled)
       - get_article: Retrieve full article content
       - submit: Submit forecasts for one or more questions
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
        
        # Handlers
        self._memory = BasicMemory(agent_id, self.config.memory_dir)
        self._query_handler = QueryHandler()
        self._search_handler = SearchHandler(
            search_tool,
            snippet_max_chars=self.config.snippet_max_chars,
            article_max_chars=self.config.article_max_chars,
            search_cutoff_days=self.config.search_cutoff_days
        )
        
        # Timing utilities for performance analysis
        self._timer = AgentTimer()
        
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
        
        # Setup handlers
        self._setup_day(forecast_interface, current_date)
        
        # Build initial prompt
        messages = [{"role": "user", "content": self._build_instructions(current_date)}]
        
        # Action loop
        all_forecasts = self._run_action_loop(messages, forecast_interface)
        
        # End of day: memory update (always happens)
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
            csv_path, forecast_interface, self.agent_id, current_date
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
        
        while actions_remaining > 0:
            # Get agent response (track LLM time)
            # Get agent response (track LLM time)
            with self._timer.track("llm"):
                response, usage = self.inference.chat(messages, self.config.sampling_params)
            
            # Extract reasoning
            reasoning = usage.get("_reasoning_content") if usage else None

            # Record token usage
            self._timer.record_tokens(usage)
            
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
    
    def _handle_query(self, messages, forecast_interface, response, parsed, actions_remaining, reasoning=None) -> int:
        """Handle query action. Returns updated actions_remaining."""
        actions_remaining -= 1
        
        if parsed.code:
            with self._timer.track("df_query"):
                result, error = self._query_handler.execute(parsed.code)
            self._log_action(forecast_interface, messages, response, "query", actions_remaining, reasoning=reasoning)
            
            if error:
                feedback = f"QUERY ERROR: {error}\n\nActions remaining: {actions_remaining}"
            else:
                feedback = f"QUERY RESULT:\n{result}\n\nActions remaining: {actions_remaining}"
        else:
            self._log_action(forecast_interface, messages, response, "query_error", 
                           actions_remaining, error=parsed.error, reasoning=reasoning)
            feedback = f"ERROR: {parsed.error}\n\nActions remaining: {actions_remaining}"
        
        feedback = self._add_exhaustion_warning(feedback, actions_remaining)
        messages.append({"role": "user", "content": feedback})
        return actions_remaining
    
    def _handle_search(self, messages, forecast_interface, response, parsed, actions_remaining, reasoning=None) -> int:
        """Handle search action. Returns full chunk content directly."""
        actions_remaining -= 1
        
        if not self._search_handler.is_available:
            self._log_action(forecast_interface, messages, response, "search_unavailable", actions_remaining, reasoning=reasoning)
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
                    min_date=min_date
                )
            self._log_action(forecast_interface, messages, response, "search", actions_remaining, reasoning=reasoning)
            
            if error:
                feedback = f"SEARCH ERROR: {error}\n\nActions remaining: {actions_remaining}"
            else:
                feedback = f"SEARCH RESULTS:\n{result}\n\nActions remaining: {actions_remaining}"
        else:
            self._log_action(forecast_interface, messages, response, "search_error", 
                           actions_remaining, error=parsed.error, reasoning=reasoning)
            feedback = f"SEARCH ERROR: No query provided.\n\nActions remaining: {actions_remaining}"
        
        feedback = self._add_exhaustion_warning(feedback, actions_remaining)
        messages.append({"role": "user", "content": feedback})
        return actions_remaining
    
    def _handle_submit(self, messages, forecast_interface, response, parsed, actions_remaining, reasoning=None) -> Tuple[int, List]:
        """Handle submit action. Returns (updated actions_remaining, list of submitted forecasts)."""
        submitted = []
        actions_remaining -= 1  # Always consume action
        
        if parsed.forecasts:
            for f in parsed.forecasts:
                try:
                    pred = PredictionSubmission(question_id=f['qid'], outcomes=f['outcomes'])
                    forecast_interface.submit_prediction(pred)
                    submitted.append(f)
                    outcomes_str = ", ".join(f"{k}: {v:.2f}" for k, v in f['outcomes'].items())
                    print(f"  [{self.agent_id}] Forecast {f['qid']}: {outcomes_str}")
                except Exception as e:
                    print(f"  [{self.agent_id}] Failed to submit {f['qid']}: {e}")
            
            self._log_action(forecast_interface, messages, response, "submit", 
                           actions_remaining, num_forecasts=len(submitted), reasoning=reasoning)
            feedback = f"Submitted {len(submitted)} forecast(s). Actions remaining: {actions_remaining}"
        else:
            # Parse error - still consumed action
            self._log_action(forecast_interface, messages, response, "submit_error",
                           actions_remaining, error=parsed.error, reasoning=reasoning)
            feedback = f"SUBMIT ERROR: {parsed.error}\n\nActions remaining: {actions_remaining}"
        
        feedback = self._add_exhaustion_warning(feedback, actions_remaining)
        messages.append({"role": "user", "content": feedback})
        return actions_remaining, submitted
    
    def _handle_invalid(self, messages, forecast_interface, response, parsed, actions_remaining, reasoning=None) -> int:
        """Handle invalid/unknown action. Returns updated actions_remaining."""
        actions_remaining -= 1
        self._log_action(forecast_interface, messages, response, "invalid", 
                        actions_remaining, error=parsed.error, reasoning=reasoning)
        
        feedback = f"No valid action found. {parsed.error or 'Use <action type=\"...\">...</action> format.'}\n\nActions remaining: {actions_remaining}"
        feedback = self._add_exhaustion_warning(feedback, actions_remaining)
        messages.append({"role": "user", "content": feedback})
        return actions_remaining
    
    # =========================================================================
    # Helpers
    # =========================================================================
    
    def _log_action(self, forecast_interface, messages, response, phase, 
                   actions_remaining, **extra) -> None:
        """Log model output with metadata."""
        last_user = messages[-2]["content"] if len(messages) >= 2 else ""
        metadata = {"phase": phase, "actions_remaining": actions_remaining, **extra}
        forecast_interface.log_model_output(last_user, response, metadata)
    
    def _add_exhaustion_warning(self, feedback: str, actions_remaining: int) -> str:
        """Add warning if actions are exhausted."""
        if actions_remaining == 0:
            feedback += "\n\nNo more actions available. Your day ends now."
        return feedback
    
    # =========================================================================
    # Instructions & Memory
    # =========================================================================
    
    def _build_instructions(self, current_date: date) -> str:
        """Build the instructions for the agent including data info and rules."""
        df_info = self._query_handler.get_info()
        
        memory_section = ""
        if self._memory:
            memory_section = f"""## YOUR MEMORY FROM PREVIOUS DAYS
<memory>
{self._memory.get()}
</memory>

"""
        
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

{memory_section}## SCORING (Time-Weighted Peer Score)
You are evaluated on your **Time-Weighted Peer Score**.
- **Peer Score**: How much better your forecast is compared to the crowd (average of other agents). 
  - If you are the only predictor, you are compared against a baseline of 0.
- **Time-Weighted**: Your score is accumulated over time.
  - **Incentive**: Predict **early** and **accurately**, and on **more** questions. If you hold a prediction better than the crowd on D days, your improvement multiples D times!

Key Mechanics:
1. **Accuracy**: Brier score (squared error). Assign probability to the TRUE outcome.
2. **Relative Performance**: You gain points by being more accurate than the market average.
3. **Speed**: Establish a good position early to maximize the duration of your high score.
4. **Number of predictions**: Your score is accumulated (summed, not averaged!) across all questions you predict on
5. **No Placeholders**: "Unknown", "TBD", "Other" are detrimental. Predict specific outcomes.
{search_advice}
## AVAILABLE DATA
DataFrame `df` with {df_info['n_rows']} questions ({df_info['n_active']} active/unresolved, {df_info['n_resolved']} resolved).

Column descriptions:
{df_info['columns_desc']}

Note: `market_aggregate` and `my_prediction` columns contain Python dicts (or None). You can access them directly, e.g. `row['market_aggregate']['outcome_name']`.
The `num_predictions` column shows how many predictions have been made on that question.

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
- For "submit", you can include multiple <forecast> items in one block.

## ACTION TYPES

### 1. Query Questions (explore data)
<action type="query">
```python
df[df['is_resolved'] == False][['qid', 'title', 'answer_type']].head()
```
</action>
{search_section}
### {submit_num}. Submit Forecasts
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
Note: After ending your day, you will be prompted to update your memory.

## SUBMISSION RULES
- qid must be from an active (is_resolved=False) question you saw in query results
- One forecast per QID per submit action (can submit multiple times for different questions)
- Outcome names must be REAL predicted answers (e.g., person names, locations, numbers)
- NEVER use placeholders like "Unknown", "TBD", "Other", "N/A" - these ALWAYS hurt your score
- Probabilities must sum to ≤ 1.0
- Submit at least one forecast before ending your day

---

You have {self.config.max_actions} actions available. Begin."""
    
    def _prompt_memory_update(self, messages: List[Dict[str, str]],  
                               forecast_interface, current_date: date) -> None:
        """
        Ask the agent to update its memory at the end of the day.
        
        Memory is the only context retained between days, allowing the agent to
        track insights, patterns, and strategies across simulation days.
        """
        memory_prompt = f"""End of day {current_date}. You can now update your memory.

## MEMORY UPDATE
Your memory is the ONLY context you will retain tomorrow. Everything else (this conversation, 
queries, forecasts) will be forgotten. Tomorrow you'll receive a fresh system prompt with 
your memory included.

Use memory to record:
- Insights about recurring question patterns
- Strategies that worked well or poorly
- Important observations about data/trends
- Notes about specific questions you're tracking

Keep it CONCISE (a few paragraphs max) as it consumes your context window.

To update memory, include it after your reasoning:
<reasoning>
Reflect on today's forecasting session...
</reasoning>
<memory>
Your updated memory content here (complete replacement, not a diff)
</memory>

If you don't want to update memory, just output <reasoning>...</reasoning> without <memory> tags.
Current memory length: {len(self._memory)} characters"""
        
        messages.append({"role": "user", "content": memory_prompt})
        response, usage = self.inference.chat(messages, self.config.sampling_params)
        self._timer.record_tokens(usage)
        messages.append({"role": "assistant", "content": response})
        
        reasoning = usage.get("_reasoning_content") if usage else None

        forecast_interface.log_model_output(
            memory_prompt, response, 
            {"phase": "memory_update", "current_memory_len": len(self._memory), "reasoning": reasoning}
        )
        
        new_memory = extract_memory(response)
        if new_memory is not None:
            self._memory.update(new_memory)
