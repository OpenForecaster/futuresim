from datetime import date
from typing import List, Dict, Any, Optional
import time

from agents.basicAgent.agent import BasicAgent
from agents.basicAgent.config import AgentConfig
from agents.utils.forecast_parser import parse_action

class AllQAgent(BasicAgent):
    """
    AllQAgent: Extends BasicAgent to perform an initial "warmup" phase on Day 0.
    
    During warmup (Day 0), the agent iterates through ALL active questions one by one,
    makes a targeted prediction for each using a focused prompt and a limited action loop
    (default 10 actions, configurable).
    
    On subsequent days, it behaves like BasicAgent but with a reminder that initial
    predictions have already been made.
    """
    
    def __init__(self, 
                 agent_id: str,
                 inference_provider,
                 config: AgentConfig = None,
                 model_name: str = "",
                 search_tool=None,
                 start_date: date = None):
        super().__init__(agent_id, inference_provider, config, model_name, search_tool)
        self.start_date = start_date
        self.warmed_up = False
        
    def warmup(self, forecast_interface, current_date: date) -> None:
        """
        Execute warmup phase: predict on ALL active questions individually.
        This effectively replaces the standard 'act' loop for Day 0.
        """
        print(f"[{self.agent_id}] Starting WARMUP phase on {current_date}")
        
        # Start timing for Day 0
        self._timer.start_day()
        
        # Get all active questions
        questions = list(forecast_interface.questions.values())
        
        # Setup handlers for this "day" (Day 0)
        self._setup_warmup_day(forecast_interface, current_date)
        
        # Set agent context so submissions are recorded correctly
        forecast_interface.current_agent_id = self.agent_id
        
        # Parallel execution for warmup
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Use configurable parallelism (default 20 from config)
        max_workers = getattr(self.config, 'warmup_parallelism', 20)
        
        print(f"[{self.agent_id}] Parallelizing warmup with {max_workers} threads...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_qid = {
                executor.submit(self._process_single_question, q, current_date, forecast_interface): q.qid 
                for q in questions
            }
            
            for i, future in enumerate(as_completed(future_to_qid)):
                qid = future_to_qid[future]
                try:
                    future.result()
                    if (i+1) % 10 == 0:
                        print(f"[{self.agent_id}] Warmup Progress: {i+1}/{len(questions)}")
                except Exception as e:
                    print(f"[{self.agent_id}] Error processing question {qid}: {e}")
            
        self.warmed_up = True
        
        # End timing and save stats for Day 0
        self._timer.end_day()
        if self.config.memory_dir:
            self._timer.save_day_stats(self.config.memory_dir, current_date)
            
        # Reset timer for subsequent standard days
        self._timer.reset()
        
        print(f"[{self.agent_id}] Warmup complete.")

    def _process_single_question(self, q, current_date, forecast_interface):
        """Process a single question validation loop (thread-safe)."""
        # customized prompt for single-question focus
        system_prompt = self._build_warmup_system_prompt(current_date, q)
        messages = [{"role": "user", "content": system_prompt}]
        
        # Run mini-loop for this question
        self._run_warmup_loop(messages, forecast_interface, q.qid)

    def act(self, 
            doc_interface, 
            forecast_interface, 
            current_date: date) -> List[Dict[str, Any]]:
        """
        Override act to skip Day 0 if warmup was done.
        """
        if current_date == self.start_date and self.warmed_up:
            print(f"[{self.agent_id}] Skipping standard act() on Day 0 (Warmup already completed).")
            forecast_interface.next_day()
            return []
            
        return super().act(doc_interface, forecast_interface, current_date)

    def _setup_warmup_day(self, forecast_interface, current_date: date) -> None:
        """
        Special setup for warmup that skips DfInterface since market CSV doesn't exist yet.
        """
        # We purposely skip self._query_handler.setup() because query is disabled in warmup
        self._search_handler.set_date(current_date)
        self._forecast_interface = forecast_interface

    def _build_instructions(self, current_date: date) -> str:
        """
        Add reminder about initial predictions to standard instructions.
        """
        base_instructions = super()._build_instructions(current_date)
        
        if self.warmed_up or (self.start_date and current_date > self.start_date):
            reminder = "\n\nIMPORTANT: You have already made initial predictions on ALL active questions during Day 0. Now focus on UPDATING these predictions if new information is available or if you missed something."
            # Insert before "You have {max_actions} actions..."
            if "You have" in base_instructions:
                base_instructions = base_instructions.replace("You have", reminder + "\n\nYou have", 1)
            else:
                base_instructions += reminder
                
        return base_instructions

    def _run_warmup_loop(self, messages: List[Dict], forecast_interface, target_qid: str) -> None:
        """
        Specialized action loop for warmup:
        - Max actions determined by config.warmup_max_actions (default 10)
        - Submit MUST be for target_qid (enforced by prompt mostly, but logic handles generic submit)
        - All actions are tagged with target_qid for searchable logs
        """
        actions_remaining = self.config.warmup_max_actions
        
        while actions_remaining > 0:
            # Get response
            with self._timer.track("llm"):
                response, usage = self.inference.chat(messages, self.config.sampling_params)
            
            if not response or not response.strip():
                print(f"  [{self.agent_id}] Empty response in warmup, skipping Q.")
                break
                
            reasoning = usage.get("_reasoning_content") if usage else None
            self._timer.record_tokens(usage)
            messages.append({"role": "assistant", "content": response})
            
            # Parse
            parsed = parse_action(response, self.config.max_outcomes_per_question)
            
            if parsed.action_type == "next":
                # In warmup, 'next' isn't really appropriate as we want them to submit.
                # But if they insist on skipping, we break.
                self._log_action(forecast_interface, messages, response, "warmup_skip", 
                               actions_remaining, qid=target_qid, reasoning=reasoning)
                print(f"  [{self.agent_id}] Agent chose 'next' (skipping submission).")
                break
            
            elif parsed.action_type == "query":
                # Queries are disabled in warmup (no DataFrame access)
                self._log_action(forecast_interface, messages, response, "warmup_query_disabled",
                               actions_remaining, qid=target_qid, reasoning=reasoning)
                feedback = "Error: Database queries are not available during the warmup phase. Please use search or submit your forecast."
                messages.append({"role": "user", "content": feedback})
                actions_remaining -= 1
            
            elif parsed.action_type == "search":
                actions_remaining = self._handle_search(
                    messages, forecast_interface, response, parsed, actions_remaining, 
                    qid=target_qid, reasoning=reasoning
                )
            
            elif parsed.action_type == "submit":
                # Enforce strict single-QID submission
                valid_forecasts = []
                if parsed.forecasts:
                    for f in parsed.forecasts:
                        if f['qid'] == target_qid:
                            valid_forecasts.append(f)
                        else:
                            print(f"  [{self.agent_id}] Ignoring submission for {f['qid']} (target: {target_qid})")
                
                # Replace with filtered list
                parsed.forecasts = valid_forecasts
                
                if valid_forecasts:
                    self._handle_submit(
                        messages, forecast_interface, response, parsed, actions_remaining, 
                        qid=target_qid, reasoning=reasoning
                    )
                    # End interaction immediately after successful submission
                    break 
                else:
                    # If no valid forecasts, warn agent
                    self._log_action(forecast_interface, messages, response, "warmup_submit_wrong_qid",
                                   actions_remaining, qid=target_qid, reasoning=reasoning)
                    feedback = f"Error: You must submit a forecast for question {target_qid}. You submitted for different IDs or none."
                    messages.append({"role": "user", "content": feedback})
                    actions_remaining -= 1
                
            else:
                actions_remaining = self._handle_invalid(
                    messages, forecast_interface, response, parsed, actions_remaining, 
                    qid=target_qid, reasoning=reasoning
                )

    def _build_warmup_system_prompt(self, current_date: date, q) -> str:
        """
        Builds a focused prompt for Day 0 warmup.
        Includes Brier score context and question details.
        """
        
        # Borrowing structure from BasicAgent._build_instructions but simplified
        
        search_section = ""
        if self._search_handler.is_available:
             search_section = """
### 2. Search News Articles
<action type="search">
your search query here
</action>

Optional date filtering:
<action type="search" from="YYYY-MM-DD" to="YYYY-MM-DD">
your search query here
</action>

You can search for recent news to inform your forecast.
"""

        # Detailed instructions as requested, based on BasicAgent but stripped of memory/peer score
        return f"""You are a forecasting agent. Today is {current_date}. 
Target Question: {q.title} (ID: {q.qid})

Background: {q.background}
Resolution Criteria: {q.resolution_criteria}
Answer Type: {q.answer_type}

## SCORING (Brier Score)
For this initial phase, you are evaluated strictly on **Brier Score** (accuracy).
- **Brier Score**: Measures the squared difference between your predicted probability and the actual outcome (0 or 1).
- **Goal**: Assign high probability to the TRUE outcome and low probability to FALSE outcomes.
- Minimize your Brier score (lower is better).

Key Mechanics:
1. **Brier Score**: You are scored on the squared difference between your predicted probabilities and the resolution (0 or 1).
2. **Distribution**: You must submit a list of (Outcome, Probability) pairs. 
3. **Max Outcomes**: You can submit at most {self.config.max_outcomes_per_question} outcomes per question. Focus on the most likely ones.
4. **No Placeholders**: "Unknown", "TBD", "Other" are detrimental. Predict specific outcomes.

## ACTIONS
You have {self.config.warmup_max_actions} actions to research and forecast this question.

{search_section}

### 3. Submit Forecast
<action type="submit">
<forecast qid="{q.qid}">
  <outcome name="Answer1" prob="0.5"/>
  <outcome name="Answer2" prob="0.3"/>
</forecast>
</action>
(Submitting ends your turn for this question).

## SUBMISSION RULES
- qid must be {q.qid}
- Outcome names must be REAL predicted answers (e.g., person names, locations, numbers)
- You can submit up to {self.config.max_outcomes_per_question} outcomes per question.
- NEVER use placeholders like "Unknown", "TBD", "Other", "N/A" - these ALWAYS hurt your score
- Probabilities must sum to ≤ 1.0

## RESPONSE FORMAT
<reasoning>
...
</reasoning>
<action type="...">
...
</action>
"""
