"""
BasicAgent: A simple LLM-based forecasting agent.

This agent:
1. Receives a DataFrame of active/resolved questions
2. Writes Python code to explore the data
3. Submits probability forecasts in XML format

Uses chain-of-thought with <reasoning> and <action type="..."> tags.
"""

import re
import json
import os
from datetime import date
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from agents.base import BaseAgent
from agents.utils.memory import BasicMemory
from agents.utils.df_interface import DfInterface
from agents.utils.forecast_parser import parse_action, extract_memory
from environment.interfaces import PredictionSubmission


@dataclass
class AgentConfig:
    """Configuration for BasicAgent."""
    max_actions: int = 10  # Max actions per day (queries + submits)
    max_submit_retries: int = 3
    max_outcomes_per_question: int = 5
    memory_dir: Optional[str] = None  # Directory to persist memory, None = no persistence
    sampling_params: dict = None
    
    def __post_init__(self):
        if self.sampling_params is None:
            self.sampling_params = {
                'temperature': 0.7,
                'max_tokens': 2048,
            }


class BasicAgent(BaseAgent):
    """
    Basic forecasting agent using LLM inference.
    
    Interaction flow per day:
    1. Receives system prompt with DataFrame schema and scoring rules
    2. Can take up to max_actions per day, including:
       - query: Execute Python code to explore the DataFrame
       - submit: Submit forecasts for one or more questions
       - next: End the current day and proceed to next
    3. Updates memory at end of day (always, regardless of how day ended)
    
    Uses <reasoning> for chain-of-thought and <action type="..."> for actions.
    """
    
    def __init__(self, 
                 agent_id: str,
                 inference_provider,
                 config: AgentConfig = None,
                 model_name: str = ""):
        super().__init__(agent_id, inference_provider, model_name)
        self.config = config or AgentConfig()
        self._memory = BasicMemory(agent_id, self.config.memory_dir)
        
    def act(self, 
            doc_interface,  # Not used in BasicAgent
            forecast_interface,  # Has get_market_csv_path, submit_prediction, next_day
            current_date: date) -> List[Dict[str, Any]]:
        """
        Execute agent logic for the day.
        
        Flow:
        1. Initialize DataFrame interface
        2. Run action loop (query/submit/next)
        3. Update memory
        4. Signal day completion
        
        Returns list of submitted forecasts.
        """
        # Setup
        self._setup_day(forecast_interface, current_date)
        messages = [{"role": "user", "content": self._build_instructions(current_date)}]
        
        # Action loop
        all_forecasts = self._run_action_loop(messages, forecast_interface)
        
        # End of day: memory update (always happens)
        self._prompt_memory_update(messages, forecast_interface, current_date)
        
        # Signal day completion
        forecast_interface.next_day()
        
        return all_forecasts
    
    # =========================================================================
    # Setup
    # =========================================================================
    
    def _setup_day(self, forecast_interface, current_date: date) -> None:
        """Initialize interfaces for the day."""
        csv_path = forecast_interface.get_market_csv_path()
        self._df_interface = DfInterface(
            csv_path, forecast_interface, self.agent_id, current_date
        )
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
            # Get agent response
            response = self.inference.chat(messages, self.config.sampling_params)
            messages.append({"role": "assistant", "content": response})
            
            # Parse and handle the action
            parsed = parse_action(response, self.config.max_outcomes_per_question)
            
            if parsed.action_type == "next":
                self._log_action(forecast_interface, messages, response, "next_day", actions_remaining)
                break
            
            elif parsed.action_type == "query":
                actions_remaining = self._handle_query(
                    messages, forecast_interface, response, parsed, actions_remaining
                )
            
            elif parsed.action_type == "submit":
                actions_remaining, forecasts = self._handle_submit(
                    messages, forecast_interface, response, parsed, actions_remaining
                )
                all_forecasts.extend(forecasts)
            
            else:
                actions_remaining = self._handle_invalid(
                    messages, forecast_interface, response, parsed, actions_remaining
                )
        
        return all_forecasts
    
    # =========================================================================
    # Action Handlers
    # =========================================================================
    
    def _handle_query(self, messages, forecast_interface, response, parsed, actions_remaining) -> int:
        """Handle query action. Returns updated actions_remaining."""
        if parsed.code:
            result, error = self._df_interface.execute_query(parsed.code)
            actions_remaining -= 1
            self._log_action(forecast_interface, messages, response, "query", actions_remaining)
            
            if error:
                feedback = f"QUERY ERROR: {error}\n\nActions remaining: {actions_remaining}"
            else:
                feedback = f"QUERY RESULT:\n{result}\n\nActions remaining: {actions_remaining}"
        else:
            actions_remaining -= 1
            self._log_action(forecast_interface, messages, response, "query_error", 
                           actions_remaining, error=parsed.error)
            feedback = f"ERROR: {parsed.error}\n\nActions remaining: {actions_remaining}"
        
        feedback = self._add_exhaustion_warning(feedback, actions_remaining)
        messages.append({"role": "user", "content": feedback})
        return actions_remaining
    
    def _handle_submit(self, messages, forecast_interface, response, parsed, actions_remaining) -> Tuple[int, List]:
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
                           actions_remaining, num_forecasts=len(submitted))
            feedback = f"Submitted {len(submitted)} forecast(s). Actions remaining: {actions_remaining}"
        else:
            # Parse error - still consumed action
            self._log_action(forecast_interface, messages, response, "submit_error",
                           actions_remaining, error=parsed.error)
            feedback = f"SUBMIT ERROR: {parsed.error}\n\nActions remaining: {actions_remaining}"
        
        feedback = self._add_exhaustion_warning(feedback, actions_remaining)
        messages.append({"role": "user", "content": feedback})
        return actions_remaining, submitted
    
    def _handle_invalid(self, messages, forecast_interface, response, parsed, actions_remaining) -> int:
        """Handle invalid/unknown action. Returns updated actions_remaining."""
        actions_remaining -= 1
        self._log_action(forecast_interface, messages, response, "invalid", 
                        actions_remaining, error=parsed.error)
        
        feedback = f"No valid action found. {parsed.error or 'Use <action type=\"...\"> format.'}\n\nActions remaining: {actions_remaining}"
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
        df_info = self._df_interface.get_info()
        
        memory_section = ""
        if self._memory:
            memory_section = f"""## YOUR MEMORY FROM PREVIOUS DAYS
<memory>
{self._memory.get()}
</memory>

"""
        
        return f"""You are a forecasting agent. Today is {current_date}. Your goal: make accurate probability predictions.

{memory_section}## SCORING (Brier Score)
You are scored on how well your predicted probabilities match reality.

Key insight: You are ONLY rewarded for assigning probability to the ACTUAL correct answer.
- Assigning probability to wrong answers HURTS your score (you lose points)
- Assigning probability to placeholders like "Unknown", "TBD", "Other" is USELESS and hurts your score
- NOT naming the correct answer = maximum penalty

Strategy: Predict specific, plausible answers based on the question context. Use the question title, 
resolution criteria, and any available data to make educated guesses about likely outcomes.

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
Every response must have <reasoning> then <action type="...">.

<reasoning>
Your analysis
</reasoning>
<action type="ACTION_TYPE">
...content based on action type...
</action>

## ACTION TYPES

### 1. Query Tasks (explore data)
<action type="query">
```python
df[df['is_resolved'] == False][['qid', 'title', 'answer_type']].head()
```
</action>

### 2. Submit Forecasts
<action type="submit">
<forecast qid="QUESTION_ID">
  <outcome name="Answer1" prob="0.5"/>
  <outcome name="Answer2" prob="0.3"/>
</forecast>
</action>

### 3. End Day (proceed to next day)
<action type="next"/>

## INTERACTION FLOW
You have {self.config.max_actions} actions per day. Each query or submission uses 1 action.
You can interleave queries and submissions as needed.
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
        response = self.inference.chat(messages, self.config.sampling_params)
        messages.append({"role": "assistant", "content": response})
        
        forecast_interface.log_model_output(
            memory_prompt, response, 
            {"phase": "memory_update", "current_memory_len": len(self._memory)}
        )
        
        new_memory = extract_memory(response)
        if new_memory is not None:
            self._memory.update(new_memory)
