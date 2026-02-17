from datetime import date
from typing import List, Dict, Any, Optional
import time

from agents.basicAgent.agent import BasicAgent
from agents.basicAgent.config import AgentConfig
from agents.utils.forecast_parser import parse_action, parse_answer_probability, ParsedAction

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

    @staticmethod
    def _is_fatal_inference_failure(error: Exception) -> bool:
        """Detect inference failures that should abort the run immediately."""
        msg = str(error).lower()
        fatal_markers = (
            "vllm server died on port",
            "vllm server failed to start",
            "vllm server process died immediately",
            "engine core initialization failed",
            "cuda out of memory occurred when warming up sampler",
        )
        return any(marker in msg for marker in fatal_markers)

    def _build_final_submit_instruction(self, target_qid: str) -> str:
        """
        Build a strict, last-action instruction that forces a submission attempt.
        This is injected as an extra user turn when only one action remains.
        """
        if getattr(self.config, "singleans", False):
            return (
                "FINAL ACTION (last chance): You have exactly 1 action remaining and must submit now.\n"
                f"Target question ID: {target_qid}\n"
                "Do NOT search, query, or skip.\n"
                "Respond ONLY with:\n"
                "<answer>...</answer>\n"
                "<probability>...</probability>\n"
                "No extra text."
            )

        return (
            "FINAL ACTION (last chance): You have exactly 1 action remaining and must submit now.\n"
            f"Target question ID: {target_qid}\n"
            "Do NOT use search, query, or next.\n"
            "Return exactly one submit action for this qid only:\n"
            f"<action type=\"submit\"><forecast qid=\"{target_qid}\">...</forecast></action>"
        )

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

        # Critical for local vLLM: start the agent server ONCE before fanning out
        # to many threads. Otherwise, thread-level failures/timeouts can lead to
        # repeated server start attempts and instability.
        from inference.vllm import VLLMInference
        if isinstance(self.inference, VLLMInference):
            print(f"[{self.agent_id}] Warming up agent vLLM server before parallel warmup...")
            # Fail fast: if startup/warmup fails, abort instead of spending hours on retries.
            self.inference.chat([{"role": "user", "content": "ping"}], {"temperature": 0.0, "max_tokens": 1})
        
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
                    if self._is_fatal_inference_failure(e):
                        print(
                            f"[{self.agent_id}] Fatal inference failure detected. Aborting warmup immediately.",
                            flush=True,
                        )
                        for pending in future_to_qid:
                            if not pending.done():
                                pending.cancel()
                        raise RuntimeError(
                            f"[{self.agent_id}] Warmup aborted due to fatal inference failure: {e}"
                        ) from e
            
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
        system_prompt = self._build_warmup_system_prompt(current_date, q, forecast_interface=forecast_interface)
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
        final_submit_prompt_injected = False

        while actions_remaining > 0:
            # Last-action guardrail: explicitly force a submit attempt.
            if actions_remaining == 1 and not final_submit_prompt_injected:
                messages.append({"role": "user", "content": self._build_final_submit_instruction(target_qid)})
                final_submit_prompt_injected = True

            # Get response
            with self._timer.track("llm"):
                response, usage = self.inference.chat(messages, self.config.sampling_params)
            
            if not response or not response.strip():
                print(f"  [{self.agent_id}] Empty response in warmup, skipping Q.")
                break
                
            reasoning = usage.get("_reasoning_content") if usage else None
            self._timer.record_tokens(usage)
            messages.append({"role": "assistant", "content": response})

            # singleans mode: accept <answer>/<probability> tags and submit a single-outcome forecast.
            if getattr(self.config, "singleans", False):
                answer, prob, err = parse_answer_probability(response)
                if err:
                    actions_remaining -= 1
                    self._log_action(
                        forecast_interface,
                        messages,
                        response,
                        "warmup_singleans_parse_error",
                        actions_remaining,
                        qid=target_qid,
                        error=err,
                        reasoning=reasoning,
                    )
                    feedback = (
                        "Format error: In singleans mode you must output:\n"
                        "<answer>...</answer>\n"
                        "<probability>...</probability>\n\n"
                        f"Parser error: {err}\n\nActions remaining: {actions_remaining}"
                    )
                    feedback = self._add_exhaustion_warning(feedback, actions_remaining)
                    messages.append({"role": "user", "content": feedback})
                    continue

                parsed = ParsedAction(
                    action_type="submit",
                    code=None,
                    forecasts=[{"qid": target_qid, "outcomes": {answer: prob}}],
                    query=None,
                    error=None,
                )
                self._handle_submit(
                    messages,
                    forecast_interface,
                    response,
                    parsed,
                    actions_remaining,
                    qid=target_qid,
                    reasoning=reasoning,
                )
                break
            
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
                feedback = "Error: Database queries are not available in this per-question focused mode. Please use search or submit your forecast."
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

    def _format_forecast_dict(self, outcomes: Optional[Dict[str, float]]) -> str:
        """Compact, stable formatting for probability dicts in prompts."""
        if not outcomes:
            return "None"
        items = sorted(outcomes.items(), key=lambda kv: (-kv[1], kv[0]))
        return "{" + ", ".join(f"{k}: {v:.3f}" for k, v in items) + "}"

    def _build_warmup_system_prompt(self, current_date: date, q, forecast_interface=None) -> str:
        """
        Builds a focused prompt for Day 0 warmup.
        Includes Brier score context and question details.
        """

        # Borrowing structure from BasicAgent._build_instructions but simplified
        
        search_section = ""
        submit_num = 1
        if self._search_handler.is_available:
             submit_num = 2
             search_section = """
### 1. Search News Articles
<action type="search">
your search query here
</action>

Optional date filtering:
<action type="search" from="YYYY-MM-DD" to="YYYY-MM-DD">
your search query here
</action>

You can search for recent news to inform your forecast.
"""

        # Optional: include existing prediction + market aggregate (without DF access).
        # We only include this section if there's something non-empty to show; on day 0
        # it is typically None/empty and can distract the model.
        existing_section = ""
        if forecast_interface is not None:
            my_pred = None
            history = forecast_interface.histories.get(q.qid) if hasattr(forecast_interface, "histories") else None
            if history and getattr(forecast_interface, "current_agent_id", None):
                my_pred = history.get_latest_prediction(forecast_interface.current_agent_id)

            my_pred_text = ""
            if my_pred:
                my_pred_text = f"\nYour latest forecast: {self._format_forecast_dict(my_pred.outcomes)} (as of {my_pred.day})"

            market_text = ""
            if not self.config.single_agent_mode:
                market = getattr(forecast_interface, "aggregates", {}).get(q.qid, {})
                if market:
                    market_text = f"\nMarket aggregate (as of {current_date}): {self._format_forecast_dict(market)}"

            if my_pred_text or market_text:
                existing_section = f"""
## EXISTING FORECASTS (NO DF ACCESS){my_pred_text}{market_text}
"""

        # Detailed instructions as requested, based on BasicAgent but stripped of memory/peer score
        return f"""You are a forecasting agent. Today is {current_date}. 
Target Question: {q.title} (ID: {q.qid})

Background: {q.background}
Resolution Criteria: {q.resolution_criteria}
Answer Type: {q.answer_type}
{existing_section}

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

### {submit_num}. Submit Forecast
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


class AllQDailyAgent(AllQAgent):
    """
    AllQDailyAgent ("allqd"): predict each active question independently each day.

    - Per-question mini-loop (default 10 actions via warmup_max_actions)
    - No DataFrame queries; existing prediction is provided in the prompt instead
    - Per-day execution parallelized across all questions
    """

    def act(self, doc_interface, forecast_interface, current_date: date):
        print(f"[{self.agent_id}] Starting ALLQD daily run on {current_date}")

        # Start timing for the day
        self._timer.reset()
        self._timer.start_day()

        # Active questions for this day (already safe-filtered by env)
        questions = list(forecast_interface.questions.values())

        # Setup handlers for this day (no DF interface)
        self._setup_warmup_day(forecast_interface, current_date)

        # Ensure submissions/logs are tagged correctly
        forecast_interface.current_agent_id = self.agent_id

        # Start the agent server before parallelizing across questions (same rationale
        # as AllQ warmup).
        from inference.vllm import VLLMInference
        if isinstance(self.inference, VLLMInference):
            print(f"[{self.agent_id}] Warming up agent vLLM server before parallel allqd...")
            # Fail fast: if startup/warmup fails, abort instead of wasting the whole day loop.
            self.inference.chat([{"role": "user", "content": "ping"}], {"temperature": 0.0, "max_tokens": 1})

        # Parallel execution across questions
        from concurrent.futures import ThreadPoolExecutor, as_completed
        max_workers = getattr(self.config, 'warmup_parallelism', 20)

        print(f"[{self.agent_id}] Parallelizing allqd with {max_workers} threads over {len(questions)} questions...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_qid = {
                executor.submit(self._process_single_question, q, current_date, forecast_interface): q.qid
                for q in questions
            }
            for i, future in enumerate(as_completed(future_to_qid)):
                qid = future_to_qid[future]
                try:
                    future.result()
                    if (i + 1) % 10 == 0:
                        print(f"[{self.agent_id}] ALLQD Progress: {i+1}/{len(questions)}")
                except Exception as e:
                    print(f"[{self.agent_id}] Error processing question {qid}: {e}")
                    if self._is_fatal_inference_failure(e):
                        print(
                            f"[{self.agent_id}] Fatal inference failure detected. Aborting allqd immediately.",
                            flush=True,
                        )
                        for pending in future_to_qid:
                            if not pending.done():
                                pending.cancel()
                        raise RuntimeError(
                            f"[{self.agent_id}] ALLQD aborted due to fatal inference failure: {e}"
                        ) from e

        # End timing and save stats
        self._timer.end_day()
        if self.config.memory_dir:
            self._timer.save_day_stats(self.config.memory_dir, current_date)

        forecast_interface.next_day()
        return []
