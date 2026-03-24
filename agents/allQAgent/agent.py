from datetime import date
from typing import List, Dict, Any, Optional
import time

from agents.basicAgent.agent import BasicAgent
from agents.basicAgent.config import AgentConfig
from agents.utils.forecast_parser import parse_action, parse_answer_probability, ParsedAction
from agents.utils.budget import BudgetTracker

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

    def _build_warmup_final_submit_instruction(self, target_qid: str, budget: BudgetTracker) -> str:
        """
        Build a strict final-submit instruction once the loop budget becomes binding.

        This is warmup-only and always targets a single question id.
        """
        preamble = self._build_force_submit_preamble(budget)
        if self.config.singleans:
            return (
                f"{preamble}\n"
                f"Target question ID: {target_qid}\n"
                "Do NOT search, query, or skip.\n"
                "Respond ONLY with:\n"
                "<answer>...</answer>\n"
                "<probability>...</probability>\n"
                "No extra text."
            )

        return (
            f"{preamble}\n"
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

        # Ensure memory object has correct date for entry timestamps and saves.
        # Safe on Day 0: no prior files exist, so set_date() just sets the date.
        if self._memory is not None:
            self._memory.set_date(current_date)

        # Collect per-question mem entries from parallel warmup threads.
        # Python list.append is thread-safe (GIL), so no lock needed.
        from agents.utils.memory import ActiveMemory
        if isinstance(self._memory, ActiveMemory):
            self._warmup_mem_entries = []

        # Collect per-question structured memory entries from parallel warmup threads.
        from agents.utils.memory import StructuredMemory
        if isinstance(self._memory, StructuredMemory):
            self._warmup_structured_entries = []

        # Set agent context so submissions are recorded correctly
        forecast_interface.current_agent_id = self.agent_id

        # Critical for local vLLM: start the agent server ONCE before fanning out
        # to many threads. Otherwise, thread-level failures/timeouts can lead to
        # repeated server start attempts and instability.
        from inference.vllm import VLLMInference
        if isinstance(self.inference, VLLMInference):
            print(f"[{self.agent_id}] Warming up agent vLLM server before parallel warmup...")
            # Fail fast: do not fan out 64 warmup threads until a real generation succeeds.
            probe_messages = [{"role": "user", "content": "ping"}]
            probe_params = {"temperature": 0.0, "max_tokens": 1}
            probe_response = ""
            for attempt in range(3):
                probe_response, _ = self.inference.chat(probe_messages, probe_params)
                if probe_response and probe_response.strip():
                    break
                if attempt < 2:
                    print(
                        f"[{self.agent_id}] Initial vLLM probe returned empty output; retrying ({attempt + 1}/3)...",
                        flush=True,
                    )
                    time.sleep(5)
            if not probe_response or not probe_response.strip():
                raise RuntimeError(
                    "Initial vLLM warmup probe failed; aborting before parallel warmup."
                )
        
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
        forecast_interface.flush_warmup_raw_logs()

        # Active memory: seed mem_df from per-question warmup conversations.
        from agents.utils.memory import ActiveMemory
        if isinstance(self._memory, ActiveMemory) and hasattr(self, '_warmup_mem_entries'):
            entries = self._warmup_mem_entries
            print(f"[{self.agent_id}] Seeding mem_df with {len(entries)} warmup entries...")
            for entry in entries:
                self._memory.mem_add(
                    qid=entry["qid"],
                    question=entry["question"],
                    memory=entry["memory"],
                    category=entry.get("category"),
                )
            self._memory.save(current_date)
            del self._warmup_mem_entries
            # Also write flat YAML so this warmup can be restarted with structured memory.
            self._save_warmup_interop(current_date)

        # Structured memory: seed entries from per-question warmup conversations.
        from agents.utils.memory import StructuredMemory
        if isinstance(self._memory, StructuredMemory) and hasattr(self, '_warmup_structured_entries'):
            entries = self._warmup_structured_entries
            print(f"[{self.agent_id}] Seeding structured memory with {len(entries)} warmup entries...")
            for entry in entries:
                try:
                    self._memory.add_entry(
                        name=entry["name"],
                        description=entry["description"],
                        content=entry["content"],
                    )
                except ValueError as e:
                    print(f"[{self.agent_id}] Warmup memory seed skipped: {e}")
            del self._warmup_structured_entries

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

        # Ask the LLM to produce a mem_df entry from this warmup conversation.
        # The model has full context (search results, reasoning, prediction) and
        # decides what's worth remembering. Falls back to placeholder to guarantee
        # every question gets an entry.
        if hasattr(self, '_warmup_mem_entries'):
            mem_entry = self._request_warmup_mem(messages, q.qid, q.title)
            if mem_entry:
                self._warmup_mem_entries.append(mem_entry)

        # Structured memory: extract question-specific memory entry from warmup.
        if hasattr(self, '_warmup_structured_entries'):
            structured_entry = self._request_warmup_structured_memory(messages, q.qid, q.title)
            if structured_entry:
                self._warmup_structured_entries.append(structured_entry)

    def _request_warmup_mem(self, messages: List[Dict], qid, question_title: str) -> Optional[Dict]:
        """Ask the LLM to produce a mem entry from the warmup conversation.

        Makes one additional LLM call with the full conversation context,
        letting the model decide what's worth remembering — same as end-of-day
        memory updates. Retries on parse failure and falls back to a placeholder
        to guarantee an entry for every question.
        """
        from agents.utils.forecast_parser import extract_mem_ops

        mem_prompt = f"""You just finished reasoning about and predicting question {qid}: "{question_title}".

Store one mem_df entry capturing your key reasoning, evidence, and calibration insights for this question. This will help you on future forecasting days.

Guidelines:
- Focus on: key evidence found, reasoning chain, your confidence level, what would change your mind
- Be specific: include numbers, sources, dates — not vague summaries
- Include your confidence estimate in the memory text (e.g., "Confidence: 0.65")
- Max 1000 chars

<mem_add>
qid: {qid}
question: {question_title}
memory: Your key reasoning, evidence, confidence, and triggers for updating (max 1000 chars)
category: topic_category
</mem_add>

Output exactly one <mem_add> block. No other text needed."""

        mem_messages = messages + [{"role": "user", "content": mem_prompt}]
        try:
            response, usage = self.inference.chat(mem_messages, self.config.sampling_params)
            self._timer.record_tokens(usage)
            self._timer.record_cost(usage.get("cost", 0), "llm")
        except Exception as e:
            print(f"[{self.agent_id}] Warmup mem LLM call failed for qid {qid}: {e}")
            return self._warmup_mem_placeholder(qid, question_title)

        adds, _, _ = extract_mem_ops(response)
        if adds:
            add = adds[0]
            return {
                "qid": str(add.get("qid", qid)),
                "question": str(add.get("question", question_title)),
                "memory": str(add.get("memory", ""))[:1000],
                "category": add.get("category"),
            }

        # Retry up to 3 times on parse failure
        print(
            f"[{self.agent_id}] Warmup mem parse failed for qid {qid}, retrying: "
            f"response_preview={response[:200]!r}"
        )
        for attempt in range(3):
            try:
                response, usage = self.inference.chat(mem_messages, self.config.sampling_params)
                self._timer.record_tokens(usage)
                self._timer.record_cost(usage.get("cost", 0), "llm")
            except Exception as e:
                print(f"[{self.agent_id}] Warmup mem retry {attempt+1} LLM error for qid {qid}: {e}")
                continue

            adds, _, _ = extract_mem_ops(response)
            if adds:
                add = adds[0]
                return {
                    "qid": str(add.get("qid", qid)),
                    "question": str(add.get("question", question_title)),
                    "memory": str(add.get("memory", ""))[:1000],
                    "category": add.get("category"),
                }
            print(
                f"[{self.agent_id}] Warmup mem retry {attempt+1} failed for qid {qid}: "
                f"response_preview={response[:200]!r}"
            )

        # All retries exhausted — use placeholder to guarantee coverage
        print(f"[{self.agent_id}] Warmup mem all retries failed for qid {qid}, using placeholder.")
        return self._warmup_mem_placeholder(qid, question_title)

    @staticmethod
    def _warmup_mem_placeholder(qid, question_title: str) -> Dict:
        """Deterministic placeholder so every question gets a mem_df entry."""
        return {
            "qid": str(qid),
            "question": question_title[:200],
            "memory": "No memory created during warmup. Update this entry with reasoning and evidence.",
            "category": "",
        }

    def _save_warmup_interop(self, current_date: date) -> None:
        """After ActiveMemory warmup, also write flat YAML for StructuredMemory restart compat.

        Converts mem_df rows to StructuredMemory entries so that
        --restart_from this warmup works with memory_format=structured.
        """
        from pathlib import Path
        from agents.utils.memory import ActiveMemory

        if not isinstance(self._memory, ActiveMemory) or not self.config.memory_dir:
            return

        memory_dir = Path(self.config.memory_dir) / "memory"
        flat_yaml_path = memory_dir / f"{current_date}.yaml"
        if flat_yaml_path.exists():
            return

        entries = []
        for _, row in self._memory.get_mem_df().iterrows():
            qid = str(row["qid"])
            entries.append({
                "name": f"q{qid}-warmup",
                "description": f"Q{qid}: {str(row['question'])[:200]}",
                "content": str(row["memory"])[:1024],
                "added": str(current_date),
            })
        if entries:
            import yaml
            flat_yaml_path.write_text(
                yaml.safe_dump(entries, default_flow_style=False, allow_unicode=True)
            )
            print(f"[{self.agent_id}] Warmup interop: wrote {len(entries)} entries to {flat_yaml_path.name}")

    def _request_warmup_structured_memory(self, messages: List[Dict], qid, question_title: str) -> Optional[Dict]:
        """Ask the LLM to produce a structured memory entry from the warmup conversation.

        Makes one additional LLM call with the full conversation context.
        Returns a dict with {name, description, content} or None on failure.
        """
        from agents.utils.forecast_parser import parse_action

        memory_prompt = f"""You just finished reasoning about and predicting question {qid}: "{question_title}".

Store one memory entry capturing your key reasoning, evidence, and calibration insights for this question. This will help you on future forecasting days.

Guidelines:
- Include the Question ID ({qid}) in the name or description so you can find it later
- Focus on: key evidence found, reasoning chain, confidence level, what would change your mind
- Be specific: include numbers, sources, dates — not vague summaries
- Descriptions should answer: "Why would future-me want to read this?" not just "What did I do today"


Memory tool call to add the entry:
<action type="memory_new">
<name>q{qid}-lowercase-hyphenated-topic (max 64 chars, a-z 0-9 hyphens only)</name>
<description>What this stores and when to use it (max 256 chars, include Question ID)</description>
<content>Your key reasoning, evidence, confidence, and triggers for updating (max 1024 chars)</content>
</action>

Output only the action block above. No other text needed."""

        mem_messages = messages + [{"role": "user", "content": memory_prompt}]
        try:
            response, usage = self.inference.chat(mem_messages, self.config.sampling_params)
            self._timer.record_tokens(usage)
            self._timer.record_cost(usage.get("cost", 0), "llm")
        except Exception as e:
            print(f"[{self.agent_id}] Warmup structured memory LLM call failed for qid {qid}: {e}")
            return None

        parsed = parse_action(response, self.config.max_outcomes_per_question)
        if parsed.action_type == "memory_new" and parsed.memory_new_data:
            return parsed.memory_new_data  # dict with {name, description, content}

        # Retry up to 3 times with the same context (strip failed attempt each time)
        print(
            f"[{self.agent_id}] Warmup structured memory parse failed for qid {qid}, retrying: "
            f"action_type={parsed.action_type}, has_data={parsed.memory_new_data is not None}, "
            f"response_preview={response[:200]!r}"
        )
        for attempt in range(3):
            try:
                response, usage = self.inference.chat(mem_messages, self.config.sampling_params)
                self._timer.record_tokens(usage)
                self._timer.record_cost(usage.get("cost", 0), "llm")
            except Exception as e:
                print(f"[{self.agent_id}] Warmup structured memory retry {attempt+1} LLM error for qid {qid}: {e}")
                continue

            parsed = parse_action(response, self.config.max_outcomes_per_question)
            if parsed.action_type == "memory_new" and parsed.memory_new_data:
                return parsed.memory_new_data
            print(
                f"[{self.agent_id}] Warmup structured memory retry {attempt+1} failed for qid {qid}: "
                f"response_preview={response[:200]!r}"
            )

        # All retries exhausted — create a deterministic placeholder entry
        print(f"[{self.agent_id}] Warmup structured memory all retries failed for qid {qid}, using placeholder.")
        return {
            "name": f"q{qid}-placeholder",
            "description": f"Question {qid}: {question_title[:200]}",
            "content": "No memory created during warmup. Update this entry with reasoning and evidence.",
        }

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
        budget = self._create_budget_tracker(warmup=True)
        budget.bootstrap_context(messages)
        raw_stream = "warmup"
        final_submit_prompt_injected = False
        final_submit_retry_used = False
        empty_retries = 0
        max_empty_retries = 2  # retry up to 2 times on empty responses mid-loop

        while not budget.is_exhausted():
            force_submit_turn = budget.should_force_submit()
            if force_submit_turn and not final_submit_prompt_injected:
                self._append_with_budget(
                    messages,
                    budget,
                    {
                        "role": "user",
                        "content": self._build_warmup_final_submit_instruction(target_qid, budget),
                    },
                )
                final_submit_prompt_injected = True

            # Get response
            with self._timer.track("llm"):
                response, usage = self.inference.chat(messages, self.config.sampling_params)

            if not response or not response.strip():
                if force_submit_turn and not final_submit_retry_used:
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
                    budget, qid=target_qid, raw_stream=raw_stream,
                )
                break
                
            # Successful response — reset empty retry counter
            empty_retries = 0

            reasoning = usage.get("_reasoning_content") if usage else None
            self._timer.record_tokens(usage)
            self._timer.record_cost(usage.get("cost", 0), "llm")
            budget.record_usage(usage)

            # singleans mode: accept <answer>/<probability> tags and submit a single-outcome forecast.
            if self.config.singleans:
                answer, prob, err = parse_answer_probability(response)
                if err:
                    if force_submit_turn and not final_submit_retry_used:
                        final_submit_retry_used = True
                        continue
                    self._append_with_budget(messages, budget, {"role": "assistant", "content": response})
                    budget.consume_action()
                    self._log_action(
                        forecast_interface,
                        messages,
                        response,
                        "warmup_singleans_parse_error",
                        budget,
                        qid=target_qid,
                        error=err,
                        reasoning=reasoning,
                        raw_stream=raw_stream,
                    )
                    feedback = (
                        "Format error: In singleans mode you must output:\n"
                        "<answer>...</answer>\n"
                        "<probability>...</probability>\n\n"
                        f"Parser error: {err}"
                    )
                    self._append_with_budget(
                        messages,
                        budget,
                        {"role": "user", "content": budget.format_feedback(feedback)},
                    )
                    continue

                self._append_with_budget(messages, budget, {"role": "assistant", "content": response})
                parsed = ParsedAction(
                    action_type="submit",
                    code=None,
                    forecasts=[{"qid": target_qid, "outcomes": {answer: prob}}],
                    query=None,
                    error=None,
                )
                # Only end this question if submission actually succeeded.
                submitted = self._handle_submit(
                    messages,
                    forecast_interface,
                    response,
                    parsed,
                    budget,
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
            if force_submit_turn and not final_submit_retry_used and not has_valid_final_submit:
                final_submit_retry_used = True
                continue

            self._append_with_budget(messages, budget, {"role": "assistant", "content": response})
            
            if parsed.action_type == "next":
                # In warmup, 'next' isn't really appropriate as we want them to submit.
                # But if they insist on skipping, we break.
                self._log_action(forecast_interface, messages, response, "warmup_skip", 
                               budget, qid=target_qid, reasoning=reasoning, raw_stream=raw_stream)
                print(f"  [{self.agent_id}] Agent chose 'next' (skipping submission).")
                break
            
            elif parsed.action_type == "query":
                # Queries are disabled in warmup (no DataFrame access)
                budget.consume_action()
                self._log_action(forecast_interface, messages, response, "warmup_query_disabled",
                               budget, qid=target_qid, reasoning=reasoning, raw_stream=raw_stream)
                feedback = "Error: Database queries are not available in this per-question focused mode. Please use search or submit your forecast."
                self._append_with_budget(
                    messages,
                    budget,
                    {"role": "user", "content": budget.format_feedback(feedback)},
                )
            
            elif parsed.action_type == "search":
                self._handle_search(
                    messages, forecast_interface, response, parsed, budget, 
                    qid=target_qid, reasoning=reasoning, raw_stream=raw_stream
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
                    submitted = self._handle_submit(
                        messages, forecast_interface, response, parsed, budget, 
                        qid=target_qid, reasoning=reasoning, raw_stream=raw_stream
                    )
                    # End interaction immediately after successful submission.
                    if submitted:
                        break
                    continue
                else:
                    # If no valid forecasts, warn agent
                    budget.consume_action()
                    self._log_action(forecast_interface, messages, response, "warmup_submit_wrong_qid",
                                   budget, qid=target_qid, reasoning=reasoning, raw_stream=raw_stream)
                    feedback = f"Error: You must submit a forecast for question {target_qid}. You submitted for different IDs or none."
                    self._append_with_budget(
                        messages,
                        budget,
                        {"role": "user", "content": budget.format_feedback(feedback)},
                    )
                
            else:
                self._handle_invalid(
                    messages, forecast_interface, response, parsed, budget, 
                    qid=target_qid, reasoning=reasoning, raw_stream=raw_stream
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
        budget_start_status = self._build_start_budget_status(warmup=True)
        budget_start_block = f"\nBudget at start:\n{budget_start_status}" if budget_start_status else ""

        # Borrowing structure from BasicAgent._build_instructions but simplified
        
        search_section = ""
        submit_num = 1
        if self._search_handler.is_available:
             submit_num = 2
             search_section = f"""
### 1. Search News Articles
<action type="search">
your search query here
</action>

Optional date filtering:
<action type="search" from="YYYY-MM-DD" to="YYYY-MM-DD">
your search query here
</action>

{self._search_results_description()}
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
{self._build_budget_overview(warmup=True, per_question=True)}

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

{budget_start_block}
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
