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

    def _build_warmup_final_submit_instruction(self, target_qid: str) -> str:
        """
        Build a strict, last-action instruction that forces a submission attempt.
        This is injected as an extra user turn when only one action remains.

        This is warmup-only and always targets a single question id.
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

    @staticmethod
    def _with_actions_remaining(content: str, actions_remaining: int) -> str:
        """Ensure user-feedback messages explicitly carry the remaining-action count."""
        if "Actions remaining:" in content:
            return content
        if content.endswith("\n"):
            return f"{content}Actions remaining: {actions_remaining}"
        return f"{content}\n\nActions remaining: {actions_remaining}"

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

        # Collect per-question memo entries from parallel warmup threads.
        # Python list.append is thread-safe (GIL), so no lock needed.
        from agents.utils.memory import ActiveMemory
        if isinstance(self._memory, ActiveMemory):
            self._warmup_memo_entries = []

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

        # Active memory: seed memo_df from per-question warmup conversations.
        from agents.utils.memory import ActiveMemory
        if isinstance(self._memory, ActiveMemory) and hasattr(self, '_warmup_memo_entries'):
            entries = self._warmup_memo_entries
            print(f"[{self.agent_id}] Seeding memo_df with {len(entries)} warmup entries...")
            for entry in entries:
                self._memory.memo_add(
                    qid=entry["qid"],
                    question=entry["question"],
                    memory=entry["memory"],
                    confidence=entry.get("confidence"),
                    category=entry.get("category"),
                )
            self._memory.save(current_date)
            del self._warmup_memo_entries

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

        # Ask the LLM to produce a memo entry from this warmup conversation.
        # The model has full context (search results, reasoning, prediction) and
        # decides what's worth remembering, just like end-of-day memory updates.
        if hasattr(self, '_warmup_memo_entries'):
            memo_entry = self._request_warmup_memo(messages, q.qid, q.title)
            if memo_entry:
                self._warmup_memo_entries.append(memo_entry)

    def _request_warmup_memo(self, messages: List[Dict], qid, question_title: str) -> Optional[Dict]:
        """Ask the LLM to produce a memo entry from the warmup conversation.

        Makes one additional LLM call with the full conversation context,
        letting the model decide what's worth remembering — same as end-of-day
        memory updates.
        """
        from agents.utils.forecast_parser import extract_memo_ops

        memo_prompt = f"""You just finished researching and predicting question {qid}: "{question_title}".

Write a concise memo entry capturing your key reasoning, evidence, and any calibration notes worth remembering for future updates. Max 500 chars.

<memo_add>
qid: {qid}
question: {question_title}
memory: Your key reasoning and evidence here (max 500 chars)
confidence: your_confidence_float
category: topic_category
</memo_add>

Output exactly one <memo_add> block. No other text needed."""

        memo_messages = messages + [{"role": "user", "content": memo_prompt}]
        try:
            response, usage = self.inference.chat(memo_messages, self.config.sampling_params)
            self._timer.record_tokens(usage)
            self._timer.record_cost(usage.get("cost", 0), "llm")
        except Exception as e:
            print(f"[{self.agent_id}] Warmup memo LLM call failed for qid {qid}: {e}")
            return None

        adds, _, _ = extract_memo_ops(response)
        if adds:
            add = adds[0]
            return {
                "qid": str(add.get("qid", qid)),
                "question": str(add.get("question", question_title)),
                "memory": str(add.get("memory", ""))[:500],
                "confidence": add.get("confidence"),
                "category": add.get("category"),
            }
        return None

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

        predicted_count = 0
        active_count = 0
        active_qids = set()
        fi = getattr(self, "_forecast_interface", None)
        if fi is not None and hasattr(fi, "questions"):
            questions = fi.questions or {}
            if isinstance(questions, dict):
                active_count = len(questions)
                active_qids = set(questions.keys())
        if fi is not None and hasattr(fi, "get_agent_predictions"):
            preds = fi.get_agent_predictions(self.agent_id)
            if isinstance(preds, dict):
                # Count predictions only for currently active questions.
                predicted_count = sum(1 for qid in preds.keys() if qid in active_qids)

        if self.warmed_up or (self.start_date and current_date > self.start_date):
            reminder = (
                "\n\nIMPORTANT: You currently have predictions on "
                f"{predicted_count} out of {active_count} active questions. "
                "You can both update past predictions if new information became available, "
                "or make new ones."
            )
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
        final_submit_retry_used = False
        empty_retries = 0
        max_empty_retries = 2  # retry up to 2 times on empty responses mid-loop

        while actions_remaining > 0:
            is_final_turn = actions_remaining == 1
            # Last-action guardrail: explicitly force a submit attempt.
            if is_final_turn and not final_submit_prompt_injected:
                messages.append(
                    {
                        "role": "user",
                        "content": self._build_warmup_final_submit_instruction(target_qid),
                    }
                )
                final_submit_prompt_injected = True

            # Get response
            with self._timer.track("llm"):
                response, usage = self.inference.chat(messages, self.config.sampling_params)

            if not response or not response.strip():
                if is_final_turn and not final_submit_retry_used:
                    final_submit_retry_used = True
                    continue
                # Retry empty responses a few times before giving up
                if empty_retries < max_empty_retries:
                    empty_retries += 1
                    print(f"  [{self.agent_id}] Empty response in warmup (retry {empty_retries}/{max_empty_retries})")
                    continue
                print(f"  [{self.agent_id}] Empty response in warmup after {max_empty_retries} retries, skipping Q.")
                self._log_action(
                    forecast_interface, messages, response or "", "warmup_empty_skip",
                    actions_remaining, qid=target_qid,
                )
                break
                
            # Successful response — reset empty retry counter
            empty_retries = 0

            reasoning = usage.get("_reasoning_content") if usage else None
            self._timer.record_tokens(usage)
            self._timer.record_cost(usage.get("cost", 0), "llm")

            # singleans mode: accept <answer>/<probability> tags and submit a single-outcome forecast.
            if getattr(self.config, "singleans", False):
                answer, prob, err = parse_answer_probability(response)
                if err:
                    if is_final_turn and not final_submit_retry_used:
                        final_submit_retry_used = True
                        continue
                    messages.append({"role": "assistant", "content": response})
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
                    messages.append(
                        {
                            "role": "user",
                            "content": self._with_actions_remaining(feedback, actions_remaining),
                        }
                    )
                    continue

                messages.append({"role": "assistant", "content": response})
                parsed = ParsedAction(
                    action_type="submit",
                    code=None,
                    forecasts=[{"qid": target_qid, "outcomes": {answer: prob}}],
                    query=None,
                    error=None,
                )
                # Only end this question if submission actually succeeded.
                actions_remaining, submitted = self._handle_submit(
                    messages,
                    forecast_interface,
                    response,
                    parsed,
                    actions_remaining,
                    qid=target_qid,
                    reasoning=reasoning,
                )
                if submitted:
                    break
                continue
            
            # Parse
            parsed = parse_action(response, self.config.max_outcomes_per_question)
            valid_forecasts = []
            if parsed.action_type == "submit" and parsed.forecasts:
                valid_forecasts = [f for f in parsed.forecasts if f.get("qid") == target_qid]

            has_valid_final_submit = (
                parsed.action_type == "submit"
                and bool(valid_forecasts)
                and self._forecasts_within_probability_bounds(valid_forecasts)
            )
            if is_final_turn and not final_submit_retry_used and not has_valid_final_submit:
                final_submit_retry_used = True
                continue

            messages.append({"role": "assistant", "content": response})
            
            if parsed.action_type == "next":
                # In warmup, 'next' isn't really appropriate as we want them to submit.
                # But if they insist on skipping, we break.
                self._log_action(forecast_interface, messages, response, "warmup_skip", 
                               actions_remaining, qid=target_qid, reasoning=reasoning)
                print(f"  [{self.agent_id}] Agent chose 'next' (skipping submission).")
                break
            
            elif parsed.action_type == "query":
                # Queries are disabled in warmup (no DataFrame access)
                actions_remaining -= 1
                self._log_action(forecast_interface, messages, response, "warmup_query_disabled",
                               actions_remaining, qid=target_qid, reasoning=reasoning)
                feedback = "Error: Database queries are not available in this per-question focused mode. Please use search or submit your forecast."
                messages.append(
                    {
                        "role": "user",
                        "content": self._with_actions_remaining(feedback, actions_remaining),
                    }
                )
            
            elif parsed.action_type == "search":
                actions_remaining = self._handle_search(
                    messages, forecast_interface, response, parsed, actions_remaining, 
                    qid=target_qid, reasoning=reasoning
                )
            
            elif parsed.action_type == "submit":
                # Enforce strict single-QID submission
                if parsed.forecasts:
                    for f in parsed.forecasts:
                        if f['qid'] != target_qid:
                            print(f"  [{self.agent_id}] Ignoring submission for {f['qid']} (target: {target_qid})")
                
                # Replace with filtered list
                parsed.forecasts = valid_forecasts
                
                if valid_forecasts:
                    actions_remaining, submitted = self._handle_submit(
                        messages, forecast_interface, response, parsed, actions_remaining, 
                        qid=target_qid, reasoning=reasoning
                    )
                    # End interaction immediately after successful submission.
                    if submitted:
                        break
                    continue
                else:
                    # If no valid forecasts, warn agent
                    actions_remaining -= 1
                    self._log_action(forecast_interface, messages, response, "warmup_submit_wrong_qid",
                                   actions_remaining, qid=target_qid, reasoning=reasoning)
                    feedback = f"Error: You must submit a forecast for question {target_qid}. You submitted for different IDs or none."
                    messages.append(
                        {
                            "role": "user",
                            "content": self._with_actions_remaining(feedback, actions_remaining),
                        }
                    )
                
            else:
                actions_remaining = self._handle_invalid(
                    messages, forecast_interface, response, parsed, actions_remaining, 
                    qid=target_qid, reasoning=reasoning
                )

    @staticmethod
    def _forecasts_within_probability_bounds(forecasts: List[Dict[str, Any]]) -> bool:
        """
        Validate forecast payload structure against environment probability constraints.
        """
        if not forecasts:
            return False
        for f in forecasts:
            outcomes = f.get("outcomes")
            if not isinstance(outcomes, dict) or not outcomes:
                return False
            total = 0.0
            for p in outcomes.values():
                try:
                    pv = float(p)
                except Exception:
                    return False
                if pv < 0.0 or pv > 1.0:
                    return False
                total += pv
            if total > 1.0 + 1e-6:
                return False
        return True

    def _format_forecast_dict(self, outcomes: Optional[Dict[str, float]]) -> str:
        """Compact, stable formatting for probability dicts in prompts."""
        if not outcomes:
            return "None"
        items = sorted(outcomes.items(), key=lambda kv: (-kv[1], kv[0]))
        return "{" + ", ".join(f"{k}: {v:.3f}" for k, v in items) + "}"

    def _get_warmup_source_name(self, forecast_interface=None) -> str:
        """Resolve source name for warmup/allqd prompts."""
        source_name = None
        if forecast_interface is not None:
            source_name = getattr(forecast_interface, "source_name", None)
        if not source_name:
            fi = getattr(self, "_forecast_interface", None)
            source_name = getattr(fi, "source_name", None) if fi is not None else None
        return (source_name or "openforesight").lower()

    def _get_warmup_scoring_section(self, forecast_interface=None) -> str:
        """
        Dataset-conditional scoring instructions for warmup/allqd prompts.

        - metaculus_binary: keep legacy binary raw-Brier wording
        - all other datasets: mirror BasicAgent scoring section
        """
        if self._get_warmup_source_name(forecast_interface) == "metaculus_binary":
            return """## SCORING (Brier Score)
For this initial phase, you are evaluated strictly on **Brier Score** (accuracy).
- **Brier Score**: Measures the squared difference between your predicted probability and the actual outcome (0 or 1).
- **Goal**: Assign high probability to the TRUE outcome and low probability to FALSE outcomes.
- Minimize your Brier score (lower is better).

Key Mechanics:
1. **Brier Score**: You are scored on the squared difference between your predicted probabilities and the resolution (0 or 1).
2. **Distribution**: You must submit a list of (Outcome, Probability) pairs.
3. **Max Outcomes**: You can submit at most {self.config.max_outcomes_per_question} outcomes per question. Focus on the most likely ones.
4. **No Placeholders**: "Unknown", "TBD", "Other" are detrimental. Predict specific outcomes.
"""

        return self._build_brier_skill_scoring_section(
            include_peer_summary=False,
            drop_mechanics={"time_weighted", "question_count"},
        )

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
        scoring_section = self._get_warmup_scoring_section(forecast_interface)
        return f"""You are a forecasting agent. Today is {current_date}. 
Target Question: {q.title} (ID: {q.qid})

Background: {q.background}
Resolution Criteria: {q.resolution_criteria}
Answer Type: {q.answer_type}
{existing_section}

{scoring_section}

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
