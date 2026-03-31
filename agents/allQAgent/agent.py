from datetime import date
from typing import List, Dict, Any, Optional, NamedTuple
import time

from agents.basicAgent.agent import BasicAgent
from agents.basicAgent.config import AgentConfig
from agents.utils.forecast_parser import parse_action, parse_answer_probability, ParsedAction
from agents.utils.budget import BudgetSettings, BudgetTracker


class WarmupLoopResult(NamedTuple):
    submitted: bool
    context_limit_hit: bool = False
    submitted_forecasts: Optional[List[Dict[str, Any]]] = None

class AllQAgent(BasicAgent):
    """
    AllQAgent: Extends BasicAgent to perform an initial "warmup" phase on Day 0.
    
    During warmup (Day 0), the agent iterates through ALL active questions one by one,
    makes a targeted prediction for each using a focused prompt and a limited action loop
    (default 10 actions, configurable).
    
    On subsequent days, it behaves like BasicAgent but with a reminder that initial
    predictions have already been made.
    """

    WARMUP_MEMORY_TOKEN_RESERVE = 4096
    WARMUP_MEMORY_MAX_OUTPUT_TOKENS = 512
    WARMUP_MEMORY_MAX_RETRIES = 2
    
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

    def _warmup_memory_mode(self) -> Optional[str]:
        from agents.utils.memory import ActiveMemory, StructuredMemory

        if isinstance(self._memory, ActiveMemory):
            return "active"
        if isinstance(self._memory, StructuredMemory):
            return "structured"
        return None

    def _create_warmup_budget_tracker(self) -> BudgetTracker:
        settings = self._get_budget_settings(warmup=True)
        if self._warmup_memory_mode() and settings.max_total_tokens is not None:
            reserve = self.WARMUP_MEMORY_TOKEN_RESERVE
            settings = BudgetSettings(
                max_actions=settings.max_actions,
                max_total_tokens=settings.max_total_tokens,
                submit_reserve_tokens=settings.submit_reserve_tokens + reserve,
                force_submit_threshold_tokens=settings.force_submit_threshold_tokens + reserve,
            )
        return BudgetTracker(settings, token_estimator=self._estimate_budget_tokens)

    def _warmup_memory_sampling_params(self) -> Dict[str, Any]:
        sampling_params = dict(self.config.sampling_params or {})
        sampling_params["temperature"] = min(float(sampling_params.get("temperature", 0.2) or 0.2), 0.2)
        max_tokens = int(sampling_params.get("max_tokens", self.WARMUP_MEMORY_MAX_OUTPUT_TOKENS) or self.WARMUP_MEMORY_MAX_OUTPUT_TOKENS)
        sampling_params["max_tokens"] = min(max_tokens, self.WARMUP_MEMORY_MAX_OUTPUT_TOKENS)
        return sampling_params

    def _build_warmup_memory_prompt(self, qid: str, question_title: str) -> str:
        mode = self._warmup_memory_mode()
        if mode == "active":
            return f"""You just finished forecasting question {qid}: "{question_title}".

Using the conversation above, write exactly one question-specific memory entry for this qid.
- Capture only the key reasoning, evidence, calibration, and what would change your mind.
- Keep the memory concise and specific.
- Do not add meta-lessons across multiple questions.
- Do not mention future cleanup steps.

Output exactly one block:
<mem_add>
qid: {qid}
question: {question_title}
memory: Key reasoning, evidence, prediction, and update triggers (max 1000 chars)
category: topic_category
</mem_add>"""

        return f"""You just finished forecasting question {qid}: "{question_title}".

Using the conversation above, create exactly one structured memory entry for this question only.
- Store question-specific reasoning, evidence, calibration, and what would change your mind.
- Do not add cross-question meta-lessons.
- Include the question id in the entry name or description.

Output exactly one action block:
<action type="memory_new">
<name>q{qid}-lowercase-hyphenated-topic</name>
<description>Question {qid}: what this stores and when to use it</description>
<content>Key reasoning, evidence, prediction, and update triggers (max 1024 chars)</content>
</action>"""

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
                try:
                    probe_response, _ = self.inference.chat(probe_messages, probe_params)
                except Exception as e:
                    if attempt < 2:
                        print(
                            f"[{self.agent_id}] Initial vLLM probe failed; retrying ({attempt + 1}/3): {e}",
                            flush=True,
                        )
                        time.sleep(5)
                        continue
                    raise
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
        result = self._run_warmup_loop(messages, forecast_interface, q.qid)

        if hasattr(self, '_warmup_mem_entries'):
            if result.context_limit_hit:
                self._warmup_mem_entries.append(
                    self._warmup_mem_placeholder(q.qid, q.title, result.submitted_forecasts)
                )
            elif result.submitted:
                mem_entry = self._request_warmup_mem(
                    messages, q.qid, q.title, result.submitted_forecasts
                )
                if mem_entry:
                    self._warmup_mem_entries.append(mem_entry)

        if hasattr(self, '_warmup_structured_entries'):
            if result.context_limit_hit:
                self._warmup_structured_entries.append(
                    self._warmup_structured_placeholder(
                        q.qid, q.title, result.submitted_forecasts
                    )
                )
            elif result.submitted:
                structured_entry = self._request_warmup_structured_memory(
                    messages, q.qid, q.title, result.submitted_forecasts
                )
                if structured_entry:
                    self._warmup_structured_entries.append(structured_entry)

    def _request_warmup_mem(
        self,
        messages: List[Dict],
        qid,
        question_title: str,
        submitted_forecasts: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict]:
        """Finalize one same-context warmup mem_df entry for a submitted question."""
        from agents.utils.forecast_parser import extract_mem_ops

        mem_messages = messages + [{"role": "user", "content": self._build_warmup_memory_prompt(qid, question_title)}]
        sampling_params = self._warmup_memory_sampling_params()
        for attempt in range(self.WARMUP_MEMORY_MAX_RETRIES + 1):
            try:
                response, usage = self.inference.chat(mem_messages, sampling_params)
                self._timer.record_tokens(usage)
                self._timer.record_cost(usage.get("cost", 0), "llm")
            except Exception as e:
                if attempt < self.WARMUP_MEMORY_MAX_RETRIES:
                    print(f"[{self.agent_id}] Warmup mem finalize retry {attempt+1} for qid {qid} after error: {e}")
                    continue
                print(f"[{self.agent_id}] Warmup mem finalize failed for qid {qid}: {e}")
                continue

            adds, _, _ = extract_mem_ops(response)
            if adds:
                add = adds[0]
                add_qid = str(add.get("qid", qid))
                if add_qid != str(qid):
                    if attempt < self.WARMUP_MEMORY_MAX_RETRIES:
                        print(
                            f"[{self.agent_id}] Warmup mem finalize retry {attempt+1} for qid {qid}: "
                            f"got qid={add_qid!r}"
                        )
                        continue
                    print(
                        f"[{self.agent_id}] Warmup mem finalize invalid for qid {qid}: "
                        f"got qid={add_qid!r}"
                    )
                    break
                return {
                    "qid": add_qid,
                    "question": str(add.get("question", question_title)),
                    "memory": str(add.get("memory", ""))[:1000],
                    "category": add.get("category"),
                }

            if attempt < self.WARMUP_MEMORY_MAX_RETRIES:
                print(
                    f"[{self.agent_id}] Warmup mem finalize retry {attempt+1} for qid {qid}: "
                    f"response_preview={response[:200]!r}"
                )
                continue
            print(
                f"[{self.agent_id}] Warmup mem finalize invalid for qid {qid}: "
                f"response_preview={response[:200]!r}"
            )

        return self._warmup_mem_placeholder(qid, question_title, submitted_forecasts)

    @staticmethod
    def _warmup_forecast_summary(submitted_forecasts: Optional[List[Dict[str, Any]]]) -> str:
        if not submitted_forecasts:
            return ""
        forecast = submitted_forecasts[0] if submitted_forecasts else None
        if not isinstance(forecast, dict):
            return ""
        outcomes = forecast.get("outcomes")
        if not isinstance(outcomes, dict) or not outcomes:
            return ""

        def _sort_key(item):
            try:
                return float(item[1])
            except Exception:
                return float("-inf")

        parts = []
        for name, prob in sorted(outcomes.items(), key=_sort_key, reverse=True):
            try:
                prob_text = f"{float(prob):.2f}"
            except Exception:
                prob_text = str(prob)
            parts.append(f"{name}={prob_text}")
        if not parts:
            return ""
        return " Submitted forecast: " + "; ".join(parts[:5]) + "."

    @classmethod
    def _warmup_placeholder_content(
        cls,
        submitted_forecasts: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        summary = cls._warmup_forecast_summary(submitted_forecasts)
        return (
            "WARMUP PLACEHOLDER: The agent reasoned about this question but did not create a memory entry."
            f"{summary} Review the warmup logs and replace this placeholder with reasoning, evidence, "
            "calibration, and update triggers."
        )[:1000]

    @classmethod
    def _warmup_mem_placeholder(
        cls,
        qid,
        question_title: str,
        submitted_forecasts: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict:
        """Deterministic placeholder so every question gets a mem_df entry."""
        return {
            "qid": str(qid),
            "question": question_title[:200],
            "memory": cls._warmup_placeholder_content(submitted_forecasts),
            "category": "warmup-placeholder",
        }

    @classmethod
    def _warmup_structured_placeholder(
        cls,
        qid,
        question_title: str,
        submitted_forecasts: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict:
        return {
            "name": f"q{qid}-placeholder",
            "description": f"Question {qid}: warmup placeholder for {question_title[:160]}",
            "content": cls._warmup_placeholder_content(submitted_forecasts),
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

    def _request_warmup_structured_memory(
        self,
        messages: List[Dict],
        qid,
        question_title: str,
        submitted_forecasts: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict]:
        """Finalize one same-context warmup structured-memory entry."""
        from agents.utils.forecast_parser import parse_action

        mem_messages = messages + [{"role": "user", "content": self._build_warmup_memory_prompt(qid, question_title)}]
        sampling_params = self._warmup_memory_sampling_params()
        for attempt in range(self.WARMUP_MEMORY_MAX_RETRIES + 1):
            try:
                response, usage = self.inference.chat(mem_messages, sampling_params)
                self._timer.record_tokens(usage)
                self._timer.record_cost(usage.get("cost", 0), "llm")
            except Exception as e:
                if attempt < self.WARMUP_MEMORY_MAX_RETRIES:
                    print(f"[{self.agent_id}] Warmup structured memory retry {attempt+1} for qid {qid} after error: {e}")
                    continue
                print(f"[{self.agent_id}] Warmup structured memory finalize failed for qid {qid}: {e}")
                continue

            parsed = parse_action(response, self.config.max_outcomes_per_question)
            if parsed.action_type == "memory_new" and parsed.memory_new_data:
                return parsed.memory_new_data

            if attempt < self.WARMUP_MEMORY_MAX_RETRIES:
                print(
                    f"[{self.agent_id}] Warmup structured memory retry {attempt+1} for qid {qid}: "
                    f"response_preview={response[:200]!r}"
                )
                continue
            print(
                f"[{self.agent_id}] Warmup structured memory invalid for qid {qid}: "
                f"response_preview={response[:200]!r}"
            )

        return self._warmup_structured_placeholder(qid, question_title, submitted_forecasts)

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
                "\n\nIMPORTANT: You have predictions on "
                f"{predicted_count} out of {active_count} active questions.\n"
                "BEFORE making any forecasts, query your existing predictions:\n"
                "  df[df['my_prediction'].notna()][['qid','title','my_prediction']]\n\n"
                "UPDATE RULES:\n"
                "- Do NOT re-predict questions from scratch unless you find specific new evidence.\n"
                "- Only update a prediction if you find SPECIFIC NEW evidence (news, data) that changes your view.\n"
                "- Anchor to your current prediction and adjust incrementally.\n\n"
                "PRIORITIES:\n"
                "1. Questions without predictions (if any)\n"
                "2. Questions where today's news search reveals new information\n"
                "3. Questions approaching resolution date that you haven't checked recently\n"
                "4. Skip questions where nothing has changed"
            )
            # Insert before "You have {max_actions} actions..."
            if "You have" in base_instructions:
                base_instructions = base_instructions.replace("You have", reminder + "\n\nYou have", 1)
            else:
                base_instructions += reminder
                
        return base_instructions

    def _run_warmup_loop(self, messages: List[Dict], forecast_interface, target_qid: str) -> WarmupLoopResult:
        """
        Specialized action loop for warmup:
        - Max actions determined by config.warmup_max_actions (default 10)
        - Submit MUST be for target_qid (enforced by prompt mostly, but logic handles generic submit)
        - All actions are tagged with target_qid for searchable logs
        """
        budget = self._create_warmup_budget_tracker()
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
            try:
                with self._timer.track("llm"):
                    response, usage = self.inference.chat(messages, self.config.sampling_params)
            except Exception as e:
                print(f"  [{self.agent_id}] Warmup LLM error for qid {target_qid}: {e}")
                if self._is_fatal_inference_failure(e):
                    raise
                self._log_action(
                    forecast_interface,
                    messages,
                    "",
                    "llm_error",
                    budget,
                    qid=target_qid,
                    error=str(e),
                    raw_stream=raw_stream,
                )
                return WarmupLoopResult(submitted=False)

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
                return WarmupLoopResult(submitted=False)
                
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
                    return WarmupLoopResult(
                        submitted=True,
                        submitted_forecasts=list(parsed.forecasts or []),
                    )
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
                        return WarmupLoopResult(
                            submitted=True,
                            submitted_forecasts=list(valid_forecasts),
                        )
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

        return WarmupLoopResult(submitted=False)

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
