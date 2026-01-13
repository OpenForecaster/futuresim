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
from datetime import date
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from agents.base import BaseAgent
from environment.interfaces import PredictionSubmission


@dataclass
class AgentConfig:
    """Configuration for BasicAgent."""
    max_queries: int = 3
    max_submit_retries: int = 3
    max_outcomes_per_question: int = 5
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
        
    def act(self, 
            doc_interface,  # AgentInterface - not used in BasicAgent
            forecast_interface,  # Has execute_query, submit_prediction, get_dataframe
            current_date: date) -> List[Dict[str, Any]]:
        """
        Execute agent logic for the day.
        
        Returns list of submitted forecasts.
        """
        # Build messages
        system_prompt = self._build_system_prompt(forecast_interface, current_date)
        messages = [{"role": "system", "content": system_prompt}]
        
        # Initial user message
        messages.append({"role": "user", "content": f"Begin. You have {self.config.max_queries} queries to explore data, then you must submit forecasts."})
        
        # Query loop
        queries_remaining = self.config.max_queries
        response = ""
        
        while queries_remaining > 0:
            # Generate response using chat
            response = self.inference.chat(messages, self.config.sampling_params)
            messages.append({"role": "assistant", "content": response})
            
            # Log just the last exchange (not full conversation with system prompt)
            last_user = messages[-2]["content"] if len(messages) >= 2 else ""
            forecast_interface.log_model_output(
                last_user, 
                response, 
                {"phase": "query", "queries_remaining": queries_remaining}
            )
            
            # Check if agent wants to submit
            if "<submit>" in response.lower():
                break
            
            # Extract and execute code from <action> tag
            code = self._extract_action_code(response)
            if code:
                result, error = forecast_interface.execute_query(code, current_date)
                queries_remaining -= 1
                
                if error:
                    feedback = f"QUERY ERROR: {error}\nTries remaining: {queries_remaining}"
                else:
                    feedback = f"QUERY RESULT:\n{result}\n\nTries remaining: {queries_remaining}"
                    if queries_remaining == 0:
                        feedback += "\n\nNo more queries. Submit your forecasts now."
                
                messages.append({"role": "user", "content": feedback})
            else:
                # No action found
                messages.append({"role": "user", "content": "I didn't find an <action> block. Please use <action>```python\n...\n```</action> for code or <action><submit>...</submit></action> for forecasts."})
                queries_remaining -= 1
        
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
                    except Exception as e:
                        print(f"  [{self.agent_id}] Failed to submit {f['qid']}: {e}")
                break
            else:
                retries_remaining -= 1
                if retries_remaining > 0:
                    messages.append({"role": "user", "content": f"PARSE ERROR: {error}\nSubmit retries remaining: {retries_remaining}. Fix and resubmit."})
                    response = self.inference.chat(messages, self.config.sampling_params)
                    messages.append({"role": "assistant", "content": response})
                    forecast_interface.log_model_output(
                        messages[-2]["content"], 
                        response, 
                        {"phase": "submit_retry", "retries_remaining": retries_remaining}
                    )
        
        return forecasts
    
    def _build_system_prompt(self, forecast_interface, current_date: date) -> str:
        """Build the system prompt with DataFrame info and rules."""
        df_info = forecast_interface.get_dataframe_info()
        
        return f"""You are a forecasting agent. Your goal: make accurate probability predictions on questions.

## SCORING (Brier Skill Score)
Score = 1 - Σ(p_i - y_i)² where sum is over ALL outcomes you name PLUS the true answer.
- y_i = 1 if outcome i is the true answer, 0 otherwise  
- p_i = probability you assigned (0 if you didn't name it)

If you don't name the true answer, you get penalized by (0-1)² = 1. Name plausible answers!

## AVAILABLE DATA
DataFrame `df` with {df_info['n_rows']} rows ({df_info['n_active']} active, {df_info['n_resolved']} resolved).

Columns:
{df_info['columns_desc']}

Current date: `today` = {current_date}

## RESPONSE FORMAT (IMPORTANT!)
Always structure your response with first reasoning, then the final action in XML tags:
<reasoning>
Your reasoning here
</reasoning>
<action>
Your executable action here. Either:
1. Python code: ```python
df[...].head()
```
2. Forecast submission: <submit>...</submit>
</action>

## QUERY PHASE ({self.config.max_queries} queries max)
Code has access to: df, pd, today, date, datetime, timedelta

## SUBMIT PHASE  
Outcome names must be SPECIFIC answers matching the question type:
- "string (name)" → person names like "John Smith", "Elon Musk"
- "string (location)" → places like "Tokyo", "California"
- DO NOT use generic "Other" - it won't match!

Format:
<action>
<submit>
  <forecast qid="12345">
    <outcome name="Tokyo" prob="0.4"/>
    <outcome name="Beijing" prob="0.3"/>
  </forecast>
</submit>
</action>

Rules:
- qid MUST be from df where is_resolved=False
- Probabilities sum ≤ 1.0
- Only use qids you see in query results!"""
    
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
        
        return forecasts, None
