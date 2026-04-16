from datetime import date, timedelta
from typing import List, Dict, Any, Optional, NamedTuple
import time

from agents.basicAgent.agent import BasicAgent
from agents.basicAgent.config import AgentConfig
from agents.basicAgent.tools import build_memory_phase_tools, chat_response_to_action
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
        """Build the strict final-submit instruction for one warmup question."""
        return self._build_final_submit_tool_instruction(target_qid, budget)

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

    @staticmethod
    def _build_warmup_mem_tool_prompt(qid: str, question_title: str) -> str:
        return f"""You just finished forecasting question {qid}: "{question_title}".

Call exactly one tool now: `mem_add`.

Requirements:
- qid must be exactly "{qid}"
- question should be "{question_title}"
- memory should store only question-specific reasoning, evidence, calibration, and update triggers
- keep memory concise and specific"""

    @staticmethod
    def _build_warmup_structured_memory_tool_prompt(qid: str, question_title: str) -> str:
        return f"""You just finished forecasting question {qid}: "{question_title}".

Call exactly one tool now: `memory_new`.

Requirements:
- store only question-specific reasoning, evidence, calibration, and update triggers
- do not store cross-question meta-lessons
- include question id {qid} in the entry name or description
- use a short lowercase hyphenated name"""

    def warmup(self, forecast_interface, current_date: date) -> None:
        """
        Execute warmup phase: predict on ALL active questions individually.
        This effectively replaces the standard 'act' loop for Day 0.
        """
        print(f"[{self.agent_id}] Starting WARMUP phase on {current_date}")
        self._require_chat_tools()
        
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
        self._flush_warmup_raw_logs()

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
        effective_current_date = self._get_warmup_current_date(current_date, q)
        self._set_warmup_search_context(current_date, q)
        try:
            # customized prompt for single-question focus
            system_prompt = self._build_warmup_system_prompt(
                effective_current_date,
                q,
                forecast_interface=forecast_interface,
            )
            messages = [{"role": "user", "content": system_prompt}]

            # Run mini-loop for this question
            result = self._run_warmup_loop(messages, forecast_interface, q.qid)

            if hasattr(self, '_warmup_mem_entries'):
                if result.context_limit_hit:
                    self._warmup_mem_entries.append(
                        self._warmup_mem_placeholder(
                            q.qid,
                            q.title,
                            result.submitted_forecasts,
                        )
                    )
                elif result.submitted:
                    mem_entry = self._request_warmup_mem(
                        messages,
                        q.qid,
                        q.title,
                        result.submitted_forecasts,
                    )
                    if mem_entry:
                        self._warmup_mem_entries.append(mem_entry)

            if hasattr(self, '_warmup_structured_entries'):
                if result.context_limit_hit:
                    self._warmup_structured_entries.append(
                        self._warmup_structured_placeholder(
                            q.qid,
                            q.title,
                            result.submitted_forecasts,
                        )
                    )
                elif result.submitted:
                    structured_entry = self._request_warmup_structured_memory(
                        messages,
                        q.qid,
                        q.title,
                        result.submitted_forecasts,
                    )
                    if structured_entry:
                        self._warmup_structured_entries.append(structured_entry)
        finally:
            self._clear_warmup_search_context()

    def _request_warmup_mem(
        self,
        messages: List[Dict],
        qid,
        question_title: str,
        submitted_forecasts: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict]:
        """Finalize one same-context warmup mem_df entry for a submitted question."""
        mem_messages = messages + [{"role": "user", "content": self._build_warmup_mem_tool_prompt(qid, question_title)}]
        allowed_tools = [
            tool
            for tool in build_memory_phase_tools(enable_memory=True, enable_mem_df=True)
            if tool.get("function", {}).get("name") == "mem_add"
        ]
        sampling_params = self._warmup_memory_sampling_params()
        sampling_params["tool_choice"] = "required"

        for attempt in range(self.WARMUP_MEMORY_MAX_RETRIES + 1):
            try:
                resp_json = self._call_chat_json_with_retries(
                    messages=mem_messages,
                    tools=allowed_tools,
                    sampling_params=sampling_params,
                )
                usage = self._normalize_chat_usage(resp_json)
                self._timer.record_tokens(usage)
                self._timer.record_cost(usage.get("cost", 0), "llm")
            except Exception as e:
                if attempt < self.WARMUP_MEMORY_MAX_RETRIES:
                    print(f"[{self.agent_id}] Warmup mem finalize retry {attempt+1} for qid {qid} after error: {e}")
                    continue
                print(f"[{self.agent_id}] Warmup mem finalize failed for qid {qid}: {e}")
                continue

            parsed, _, _ = chat_response_to_action(resp_json)
            if parsed is not None and parsed.action_type == "mem_add" and parsed.mem_data:
                add = dict(parsed.mem_data)
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
                    f"action={getattr(parsed, 'action_type', None)}"
                )
                continue
            print(
                f"[{self.agent_id}] Warmup mem finalize invalid for qid {qid}: "
                f"action={getattr(parsed, 'action_type', None)}"
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
        mem_messages = messages + [{"role": "user", "content": self._build_warmup_structured_memory_tool_prompt(qid, question_title)}]
        allowed_tools = [
            tool
            for tool in build_memory_phase_tools(enable_memory=True, enable_mem_df=False)
            if tool.get("function", {}).get("name") == "memory_new"
        ]
        sampling_params = self._warmup_memory_sampling_params()
        sampling_params["tool_choice"] = "required"

        for attempt in range(self.WARMUP_MEMORY_MAX_RETRIES + 1):
            try:
                resp_json = self._call_chat_json_with_retries(
                    messages=mem_messages,
                    tools=allowed_tools,
                    sampling_params=sampling_params,
                )
                usage = self._normalize_chat_usage(resp_json)
                self._timer.record_tokens(usage)
                self._timer.record_cost(usage.get("cost", 0), "llm")
            except Exception as e:
                if attempt < self.WARMUP_MEMORY_MAX_RETRIES:
                    print(f"[{self.agent_id}] Warmup structured memory retry {attempt+1} for qid {qid} after error: {e}")
                    continue
                print(f"[{self.agent_id}] Warmup structured memory finalize failed for qid {qid}: {e}")
                continue

            parsed, _, _ = chat_response_to_action(resp_json)
            if parsed is not None and parsed.action_type == "memory_new" and parsed.memory_new_data:
                return dict(parsed.memory_new_data)

            if attempt < self.WARMUP_MEMORY_MAX_RETRIES:
                print(
                    f"[{self.agent_id}] Warmup structured memory retry {attempt+1} for qid {qid}: "
                    f"action={getattr(parsed, 'action_type', None)}"
                )
                continue
            print(
                f"[{self.agent_id}] Warmup structured memory invalid for qid {qid}: "
                f"action={getattr(parsed, 'action_type', None)}"
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
        self._set_log_date(current_date)
        self._search_handler.set_date(current_date)
        self._forecast_interface = forecast_interface

    def _get_warmup_current_date(self, current_date: date, q) -> date:
        resolution_guard = getattr(self.config, "resolution_guard", None)
        resolution_date = getattr(q, "resolution_date", None)
        if resolution_guard is None or resolution_date is None:
            return current_date
        return resolution_date - timedelta(days=resolution_guard)

    def _set_warmup_search_context(self, current_date: date, q) -> None:
        self._search_handler.set_date(current_date)
        effective_current_date = self._get_warmup_current_date(current_date, q)
        if effective_current_date == current_date:
            self._search_handler.clear_current_date_override()
            return
        self._search_handler.set_current_date_override(effective_current_date)

    def _clear_warmup_search_context(self) -> None:
        self._search_handler.clear_current_date_override()

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
                "IMPORTANT: You have predictions on "
                f"{predicted_count} out of {active_count} active questions.\n"
                "Tip: You can check your existing predictions by making the following query - `df[df['my_prediction'].notna()][['qid','title','my_prediction']]`\n\n"
                "UPDATE RULES:\n"
                "- Do NOT re-predict questions from scratch unless you find specific new evidence.\n"
                "- Only update a prediction if you find SPECIFIC NEW evidence (news, data) that updates your view.\n\n"
                "PRIORITIES FOR UPDATES:\n"
                "1. Questions without predictions (if any)\n"
                "2. Questions where today's news search reveals new information\n"
                "3. Questions approaching resolution date that you haven't checked recently\n"
                "4. Skip questions where there is no new evidence"
            )
            cadence_section = self._build_cadence_section(current_date)
            if cadence_section in base_instructions:
                base_instructions = base_instructions.replace(
                    cadence_section,
                    cadence_section + reminder + "\n\n",
                    1,
                )
            else:
                base_instructions += reminder
                
        return base_instructions

    def _run_warmup_loop(self, messages: List[Dict], forecast_interface, target_qid: str) -> WarmupLoopResult:
        """
        Per-question warmup loop using the same chat-tools protocol as the daily agent.
        """
        if self.config.singleans:
            raise RuntimeError(
                "AllQAgent warmup now requires chat tools. singleans is no longer supported here; "
                "use the dedicated og scaffold if you need the answer/probability format."
            )

        forecasts, context_limit_hit = self._run_chat_tools_action_loop(
            messages=messages,
            forecast_interface=forecast_interface,
            target_qid=target_qid,
            enable_query=False,
            warmup_budget=True,
            max_actions=self.config.warmup_max_actions,
        )
        return WarmupLoopResult(
            submitted=bool(forecasts),
            context_limit_hit=context_limit_hit,
            submitted_forecasts=list(forecasts),
        )
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
        budget_start_block = self._build_prompt_seed_budget_block(
            warmup=True,
            leading_newline=True,
            trailing_newlines=0,
        )

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
        search_block = ""
        if self._search_handler.is_available:
            search_block = (
                "- `search_news(query, from_date?, to_date?)` for evidence. "
                f"{self._search_results_description()}\n"
            )

        prompt = f"""You are a forecasting agent. Your goal is to make accurate and calibrated predictions. Today is {current_date}.

Target Question: {q.title} (ID: {q.qid})

Background: {q.background}
Resolution Criteria: {q.resolution_criteria}
Answer Type: {q.answer_type}
{existing_section}

{scoring_section}

## INTERACTION FLOW
{self._build_budget_overview(warmup=True, per_question=True)}


## TOOLS YOU SHOULD USE
Use the function tools from the tool schema. Call exactly one tool per turn.
{search_block}- `submit_forecasts(forecasts)`: submit exactly one forecast for qid `{q.qid}`.
- `next_day()`: skip this question only if you truly have no forecast to submit.


## SUBMISSION RULES
- qid must be {q.qid}
- Outcome names must be REAL predicted answers (e.g., person names, locations, numbers)
- You can submit up to {self.config.max_outcomes_per_question} outcomes per question.
- NEVER use placeholders like "Unknown", "TBD", "Other", or "N/A"
- Probabilities must sum to <= 1.0
- A successful `submit_forecasts` call ends this question's warmup interaction.

{budget_start_block}

Begin.
"""
        return self._normalize_prompt_heading_spacing(prompt)


class AllQDailyAgent(AllQAgent):
    """
    AllQDailyAgent ("allqd"): predict each active question independently each day.

    - Per-question mini-loop (default 10 actions via warmup_max_actions)
    - No DataFrame queries; existing prediction is provided in the prompt instead
    - Per-day execution parallelized across all questions
    """

    def act(self, doc_interface, forecast_interface, current_date: date):
        print(f"[{self.agent_id}] Starting ALLQD daily run on {current_date}")
        self._require_chat_tools()

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
