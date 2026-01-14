"""
BasicAgent: A simple LLM-based forecasting agent.

This agent:
1. Receives a DataFrame of active/resolved questions
2. Writes Python code to explore the data
3. Submits probability forecasts in XML format

Uses chain-of-thought with <reasoning> and <action> tags.
"""

import re
import json
import os
from datetime import date
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from agents.base import BaseAgent
from environment.interfaces import PredictionSubmission


@dataclass
class AgentConfig:
    """Configuration for BasicAgent."""
    max_queries: int = 3
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
    2. Can query the DataFrame up to max_queries times
    3. Submits forecasts in XML format
    
    Uses <reasoning> for chain-of-thought and <action> for executable content.
    """
    
    def __init__(self, 
                 agent_id: str,
                 inference_provider,
                 config: AgentConfig = None,
                 model_name: str = ""):
        super().__init__(agent_id, inference_provider, model_name)
        self.config = config or AgentConfig()
        
        # Memory: persisted text the agent can update between days
        self.memory: str = ""
        self._memory_path: Optional[Path] = None
        
        # Load memory from disk if configured
        if self.config.memory_dir:
            self._memory_path = Path(self.config.memory_dir) / f"{agent_id}_memory.txt"
            if self._memory_path.exists():
                self.memory = self._memory_path.read_text().strip()
        
    def act(self, 
            doc_interface,  # AgentInterface - not used in BasicAgent
            forecast_interface,  # Has execute_query, submit_prediction, get_dataframe
            current_date: date) -> List[Dict[str, Any]]:
        """
        Execute agent logic for the day.
        
        Returns list of submitted forecasts.
        """
        # Build the instructions (formerly "system prompt") as first user message
        instructions = self._build_instructions(forecast_interface, current_date)
        messages = [{"role": "user", "content": instructions}]
        
        # Query loop
        queries_remaining = self.config.max_queries
        response = ""
        
        while queries_remaining > 0:
            # Generate response using chat
            response = self.inference.chat(messages, self.config.sampling_params)
            messages.append({"role": "assistant", "content": response})
            
            # Check if agent wants to submit
            if "<submit>" in response.lower():
                # Log the submit response
                last_user = messages[-2]["content"] if len(messages) >= 2 else ""
                forecast_interface.log_model_output(
                    last_user, response, 
                    {"phase": "submit", "queries_remaining": queries_remaining}
                )
                break
            
            # Extract and execute code from <action> tag
            code = self._extract_action_code(response)
            if code:
                result, error = forecast_interface.execute_query(code, current_date)
                queries_remaining -= 1
                
                # Log after decrement so count is accurate
                last_user = messages[-2]["content"] if len(messages) >= 2 else ""
                forecast_interface.log_model_output(
                    last_user, response, 
                    {"phase": "query", "queries_remaining": queries_remaining}
                )
                
                if error:
                    feedback = f"QUERY ERROR: {error}\n\nQueries remaining: {queries_remaining}"
                else:
                    feedback = f"QUERY RESULT:\n{result}\n\nQueries remaining: {queries_remaining}"
                
                if queries_remaining == 0:
                    feedback += "\n\nNo more queries available. You must submit your forecasts now."
                
                messages.append({"role": "user", "content": feedback})
            else:
                # No action found
                queries_remaining -= 1
                last_user = messages[-2]["content"] if len(messages) >= 2 else ""
                forecast_interface.log_model_output(
                    last_user, response, 
                    {"phase": "invalid", "queries_remaining": queries_remaining}
                )
                messages.append({"role": "user", "content": "No valid <action> block found. See instructions for format."})
        
        # Submit loop
        forecasts = []
        retries_remaining = self.config.max_submit_retries
        
        while retries_remaining > 0:
            # Try to parse forecasts from last response
            forecasts, error = self._parse_forecasts(response)
            
            if forecasts:
                # Successfully parsed - submit all forecasts
                for f in forecasts:
                    try:
                        pred = PredictionSubmission(
                            question_id=f['qid'],
                            outcomes=f['outcomes']
                        )
                        forecast_interface.submit_prediction(pred)
                        # Print the submission
                        outcomes_str = ", ".join(f"{k}: {v:.2f}" for k, v in f['outcomes'].items())
                        print(f"  [{self.agent_id}] Forecast {f['qid']}: {outcomes_str}")
                    except Exception as e:
                        print(f"  [{self.agent_id}] Failed to submit {f['qid']}: {e}")
                break
            else:
                retries_remaining -= 1
                if retries_remaining > 0:
                    retry_msg = f"PARSE ERROR: {error}\n\nRetries remaining: {retries_remaining}"
                    messages.append({"role": "user", "content": retry_msg})
                    response = self.inference.chat(messages, self.config.sampling_params)
                    messages.append({"role": "assistant", "content": response})
                    forecast_interface.log_model_output(
                        messages[-2]["content"], 
                        response, 
                        {"phase": "submit_retry", "retries_remaining": retries_remaining}
                    )
        
        # Memory update phase (end of day)
        self._prompt_memory_update(messages, forecast_interface, current_date)
        
        return forecasts
    
    def _build_instructions(self, forecast_interface, current_date: date) -> str:
        """Build the instructions for the agent including data info and rules."""
        df_info = forecast_interface.get_dataframe_info()
        
        # Memory section (if agent has memory from previous days)
        memory_section = ""
        if self.memory:
            memory_section = f"""## YOUR MEMORY FROM PREVIOUS DAYS
<memory>
{self.memory}
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

Example query:
```python
df[df['is_resolved'] == False][['qid', 'title', 'answer_type']].head()
```

## RESPONSE FORMAT
Every response must have <reasoning> then <action>:
<reasoning>
Your analysis
</reasoning>
<action>
Python code in ```python ... ``` OR forecast submission in <submit>...</submit>
</action>

## INTERACTION FLOW
1. QUERY PHASE: You have {self.config.max_queries} queries. One query per turn.
2. SUBMIT PHASE: Submit forecasts when ready.

## SUBMISSION FORMAT
<action>
<submit>
  <forecast qid="QUESTION_ID">
    <outcome name="Answer1" prob="0.5"/>
    <outcome name="Answer2" prob="0.3"/>
  </forecast>
  ...
  <forecast qid="QUESTION_ID">
    ...
  </forecast>
</submit>
</action>

Rules:
- qid must be from an active (is_resolved=False) question you saw in query results
- One forecast per QID (do not submit multiple forecasts for the same question)
- Outcome names must be REAL predicted answers (e.g., person names, locations, numbers)
- NEVER use placeholders like "Unknown", "TBD", "Other", "N/A" - these ALWAYS hurt your score
- Probabilities must sum to ≤ 1.0
- You must submit at least one forecast

---

You have {self.config.max_queries} query turns. Begin."""
    
    def _extract_action_code(self, response: str) -> Optional[str]:
        """Extract Python code from <action> tag."""
        # Look for <action>...</action>
        action_pattern = r'<action>(.*?)</action>'
        action_match = re.search(action_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if not action_match:
            # Fallback: look for ```python blocks directly
            code_pattern = r'```python\s*(.*?)\s*```'
            code_matches = re.findall(code_pattern, response, re.DOTALL)
            if code_matches:
                return code_matches[-1]
            return None
        
        action_content = action_match.group(1)
        
        # If it contains <submit>, return None (handled separately)
        if '<submit>' in action_content.lower():
            return None
        
        # Extract python code from action
        code_pattern = r'```python\s*(.*?)\s*```'
        code_matches = re.findall(code_pattern, action_content, re.DOTALL)
        if code_matches:
            return code_matches[-1]
        
        # Maybe it's just code without markdown
        code = action_content.strip()
        if code and ('df' in code or 'pd.' in code):
            return code
        
        return None
    
    def _print_action(self, response: str) -> None:
        """Print the content of <action> tags for terminal visibility."""
        action_pattern = r'<action>(.*?)</action>'
        action_match = re.search(action_pattern, response, re.DOTALL | re.IGNORECASE)
        if action_match:
            action_content = action_match.group(1).strip()
            # Truncate if too long
            if len(action_content) > 200:
                action_content = action_content[:200] + "..."
            print(f"  [{self.agent_id}] {action_content}")
    
    def _parse_forecasts(self, response: str) -> Tuple[List[Dict], Optional[str]]:
        """
        Parse XML forecasts from agent response.
        
        Returns (forecasts, error_message).
        """
        # Extract from <action> block if present
        action_pattern = r'<action>(.*?)</action>'
        action_match = re.search(action_pattern, response, re.DOTALL | re.IGNORECASE)
        if action_match:
            response = action_match.group(1)
        
        # Extract <submit>...</submit> block
        pattern = r'<submit>(.*?)</submit>'
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if not match:
            if '<forecast' not in response.lower():
                return [], "No <submit> block or <forecast> tags found"
            submit_content = response
        else:
            submit_content = match.group(1)
        
        # Parse individual forecasts
        forecast_pattern = r'<forecast\s+qid=["\']([^"\']+)["\']>(.*?)</forecast>'
        forecast_matches = re.findall(forecast_pattern, submit_content, re.DOTALL | re.IGNORECASE)
        
        if not forecast_matches:
            return [], "No valid <forecast qid=\"...\">...</forecast> blocks found"
        
        forecasts = []
        for qid, content in forecast_matches:
            # Parse outcomes
            outcome_pattern = r'<outcome\s+name=["\']([^"\']+)["\']\s+prob=["\']([^"\']+)["\']'
            outcome_matches = re.findall(outcome_pattern, content, re.IGNORECASE)
            
            if not outcome_matches:
                return [], f"No valid outcomes found for question {qid}"
            
            if len(outcome_matches) > self.config.max_outcomes_per_question:
                return [], f"Too many outcomes for {qid}: {len(outcome_matches)} > {self.config.max_outcomes_per_question}"
            
            outcomes = {}
            for name, prob_str in outcome_matches:
                try:
                    prob = float(prob_str)
                except ValueError:
                    return [], f"Invalid probability '{prob_str}' for outcome '{name}' in {qid}"
                
                if prob < 0 or prob > 1:
                    return [], f"Probability {prob} out of range [0,1] for '{name}' in {qid}"
                
                outcomes[name] = prob
            
            # Check sum
            total = sum(outcomes.values())
            if total > 1.0 + 1e-6:
                return [], f"Probabilities sum to {total:.3f} > 1 for question {qid}"
            
            forecasts.append({'qid': qid, 'outcomes': outcomes})
        
        # Deduplicate by QID (keep last)
        unique_forecasts = {}
        for f in forecasts:
            unique_forecasts[f['qid']] = f
        
        return list(unique_forecasts.values()), None
    
    def _prompt_memory_update(self, messages: List[Dict[str, str]], 
                               forecast_interface, current_date: date) -> None:
        """
        Ask the agent if it wants to update its memory at the end of the day.
        
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
Current memory length: {len(self.memory)} characters"""
        
        messages.append({"role": "user", "content": memory_prompt})
        response = self.inference.chat(messages, self.config.sampling_params)
        messages.append({"role": "assistant", "content": response})
        
        # Log the memory update exchange
        forecast_interface.log_model_output(
            memory_prompt, 
            response, 
            {"phase": "memory_update", "current_memory_len": len(self.memory)}
        )
        
        # Extract and save memory if present
        new_memory = self._extract_memory(response)
        if new_memory is not None:
            self.memory = new_memory
            self._save_memory()
    
    def _extract_memory(self, response: str) -> Optional[str]:
        """
        Extract memory content from <memory></memory> tags.
        
        Returns None if no memory tags found, empty string if tags are empty.
        """
        pattern = r'<memory>(.*?)</memory>'
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None
    
    def _save_memory(self) -> None:
        """Persist memory to disk if configured."""
        if self._memory_path:
            self._memory_path.parent.mkdir(parents=True, exist_ok=True)
            self._memory_path.write_text(self.memory)

