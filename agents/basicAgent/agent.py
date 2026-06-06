"""
BasicAgent: Main agent class for LLM-based forecasting.

Uses Chat Completions tool-calling for all action and structured-memory loops.
"""

from datetime import date
from typing import List, Dict, Any, Optional, Tuple

from futuresim_agents.base import BaseAgent
from futuresim_agents.utils.budget import (
    BudgetSettings,
    BudgetTracker,
    build_budget_overview,
    build_start_budget_status,
    estimate_budget_tokens,
)
from futuresim_agents.utils.timing import AgentTimer

from .actions import BasicActionHandlers
from .chat_protocol import BasicChatProtocol
from .config import AgentConfig
from .memory import BasicMemory
from .memory_prompts import BasicMemoryPromptBuilder
from .prompts import BasicPromptBuilder
from .tools import (
    build_action_tools,
    build_memory_phase_tools,
    chat_response_to_action,
    extract_assistant_message,
    extract_finish_reason,
    final_submit_tool_instruction_text,
    single_call_to_parsed_action,
)
from futuresim_agents.utils.memory import StructuredMemory, ActiveMemory
from futuresim_agents.utils.output_logger import AgentOutputLogger
from .query import QueryHandler
from .search import SearchHandler
from .feedback import FeedbackHandler


class BasicAgent(BasicChatProtocol, BasicPromptBuilder, BasicMemoryPromptBuilder, BasicActionHandlers, BaseAgent):
    """
    Basic forecasting agent using LLM inference.
    
    Interaction flow per day:
    1. Receives system prompt with DataFrame schema and scoring rules
    2. Can take query/search/submit/next actions, subject to configured loop budgets:
       - query: Execute Python code to explore the DataFrame
       - search: Search news articles for context (if enabled)
       - submit: Submit a forecast for exactly one question
       - next: End the current day and proceed to next
    3. Updates memory at end of day (always, regardless of how day ended)
    
    Uses Chat Completions tools for the forecasting loop and structured-memory
    updates. Providers must expose `chat_json(...)`, and local vLLM runs must be
    started with tool calling enabled.
    """

    @staticmethod
    def _forecasts_within_probability_bounds(forecasts: List[Dict[str, Any]]) -> bool:
        """Each forecast must have a non-empty outcomes dict: values in [0, 1], sum ≤ 1 (+ε)."""
        if not forecasts:
            return False
        for forecast in forecasts:
            outcomes = forecast.get("outcomes")
            if not isinstance(outcomes, dict) or not outcomes:
                return False
            total = 0.0
            for prob in outcomes.values():
                try:
                    value = float(prob)
                except Exception:
                    return False
                if value < 0.0 or value > 1.0:
                    return False
                total += value
            if total > 1.0 + 1e-6:
                return False
        return True

    @staticmethod
    def extract_reasoning_token_count(response_json: Dict[str, Any]) -> int:
        """Extract per-response reasoning token count from usage details when available."""
        usage = response_json.get("usage") or {}
        if not isinstance(usage, dict):
            return 0

        out_details = usage.get("output_tokens_details")
        if isinstance(out_details, dict):
            rt = out_details.get("reasoning_tokens")
            if isinstance(rt, int):
                return rt
            try:
                return int(rt)
            except Exception:
                pass

        comp_details = usage.get("completion_tokens_details")
        if isinstance(comp_details, dict):
            rt = comp_details.get("reasoning_tokens")
            if isinstance(rt, int):
                return rt
            try:
                return int(rt)
            except Exception:
                pass

        rt = usage.get("reasoning_tokens")
        if isinstance(rt, int):
            return rt
        try:
            return int(rt)
        except Exception:
            return 0

    def __init__(self, 
                 agent_id: str,
                 inference_provider,
                 config: AgentConfig = None,
                 model_name: str = "",
                 search_tool=None):
        super().__init__(agent_id, inference_provider, model_name)
        self.config = config or AgentConfig()

        # Timing utilities for performance analysis
        self._timer = AgentTimer()
        
        # Handlers
        if self.config.enable_memory:
            if self.config.memory_format == "active":
                self._memory = ActiveMemory(agent_id, self.config.memory_dir,
                                            max_entries=self.config.memory_max_entries)
            elif self.config.memory_format == "structured":
                self._memory = StructuredMemory(agent_id, self.config.memory_dir,
                                                   max_entries=self.config.memory_max_entries)
            else:
                self._memory = BasicMemory(agent_id, self.config.memory_dir)
        else:
            self._memory = None
        self._query_handler = QueryHandler()
        self._search_handler = SearchHandler(
            search_tool,
            snippet_max_chars=self.config.snippet_max_chars,
            article_max_chars=self.config.article_max_chars,
            search_cutoff_days=self.config.search_cutoff_days
        )
        self._feedback_handler = FeedbackHandler(
            agent_id,
            timing_callback=self._record_matcher_timing
        )
        self._output_logger = AgentOutputLogger(
            agent_id,
            self.config.memory_dir,
            append=getattr(self.config, "append_model_output_logs", False),
        )
        self._log_date: Optional[date] = None

    def _build_session_instructions(self, current_date: date) -> str:
        return self._build_instructions(current_date)
        
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
        self._day_qids = set()  # Track QIDs the agent interacts with today
        self._context_limit_hit = False
        self._require_chat_tools()

        # Load memory for this date (loads most recent snapshot before current_date)
        if self._memory is not None:
            self._memory.set_date(current_date)
        
        # Setup handlers
        self._setup_day(forecast_interface, current_date)
        
        # Build initial prompt
        messages = [{"role": "user", "content": self._build_session_instructions(current_date)}]
        
        # Unified action loop (includes memory phase for structured/active memory)
        all_forecasts, _ = self._run_chat_tools_action_loop(
            messages=messages,
            forecast_interface=forecast_interface,
            current_date=current_date,
        )

        # If the unified loop did not complete an in-loop memory phase, fall back
        # to the explicit end-of-day memory update. This keeps structured/active
        # memory working even when max_total_tokens is unset.
        if (
            self._memory is not None
            and not getattr(self, "_context_limit_hit", False)
            and not getattr(self, '_memory_phase_completed', False)
        ):
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
        self._set_log_date(current_date)
        csv_path = forecast_interface.get_market_csv_path()
        self._query_handler.setup(
            csv_path, forecast_interface, self.agent_id, current_date,
            single_agent_mode=self.config.single_agent_mode
        )
        self._search_handler.set_date(current_date)
        self._forecast_interface = forecast_interface

    def _get_budget_settings(self, *, warmup: bool = False) -> BudgetSettings:
        """Resolve loop-budget settings for day or warmup loops."""
        if warmup:
            return BudgetSettings(
                max_actions=self.config.warmup_max_actions,
                max_total_tokens=self.config.warmup_max_total_tokens,
                submit_reserve_tokens=(
                    self.config.warmup_submit_reserve_tokens
                    if self.config.warmup_submit_reserve_tokens is not None
                    else self.config.submit_reserve_tokens
                ),
                force_submit_threshold_tokens=(
                    self.config.warmup_force_submit_threshold_tokens
                    if self.config.warmup_force_submit_threshold_tokens is not None
                    else self.config.force_submit_threshold_tokens
                ),
            )
        # Non-warmup: no submit_reserve or force_submit.
        # Memory phase threshold set when structured/active memory is configured.
        memory_threshold = None
        if (
            self._memory is not None
            and isinstance(self._memory, (StructuredMemory, ActiveMemory))
            and self.config.max_total_tokens is not None
        ):
            memory_threshold = self.config.memory_update_max_total_tokens
        return BudgetSettings(
            max_actions=self.config.max_actions,
            max_total_tokens=self.config.max_total_tokens,
            submit_reserve_tokens=0,
            force_submit_threshold_tokens=0,
            memory_phase_threshold_tokens=memory_threshold,
        )

    def _create_budget_tracker(
        self,
        *,
        warmup: bool = False,
        max_actions_override: Optional[int] = None,
    ) -> BudgetTracker:
        settings = self._get_budget_settings(warmup=warmup)
        if max_actions_override is not None:
            settings = BudgetSettings(
                max_actions=max_actions_override,
                max_total_tokens=settings.max_total_tokens,
                submit_reserve_tokens=settings.submit_reserve_tokens,
                force_submit_threshold_tokens=settings.force_submit_threshold_tokens,
                memory_phase_threshold_tokens=settings.memory_phase_threshold_tokens,
            )
        return BudgetTracker(settings, token_estimator=self._estimate_budget_tokens)

    def _build_budget_overview(self, *, warmup: bool = False, per_question: bool = False) -> str:
        """Human-readable budget instructions for prompts."""
        settings = self._get_budget_settings(warmup=warmup)
        return build_budget_overview(settings, per_question=per_question)

    def _build_start_budget_status(
        self,
        *,
        warmup: bool = False,
        max_actions_override: Optional[int] = None,
    ) -> str:
        """Render the initial remaining-budget status for prompt seeds."""
        tracker = self._create_budget_tracker(
            warmup=warmup,
            max_actions_override=max_actions_override,
        )
        return build_start_budget_status(tracker.settings)

    def _build_final_submit_tool_instruction(self, target_qid: str, budget: BudgetTracker) -> str:
        return final_submit_tool_instruction_text(target_qid, budget)

    def _estimate_budget_tokens(self, payload: Any) -> int:
        return estimate_budget_tokens(payload, model_name=str(self.model_name or ""))

    @staticmethod
    def _append_with_budget(messages: List[Dict[str, Any]], budget: BudgetTracker, message: Dict[str, Any]) -> None:
        messages.append(message)
        budget.record_appended_item(message)

    def _run_chat_tools_action_loop(
        self,
        *,
        messages: List[Dict[str, Any]],
        forecast_interface,
        target_qid: Optional[str] = None,
        enable_query: bool = True,
        enable_search: Optional[bool] = None,
        warmup_budget: bool = False,
        max_actions: Optional[int] = None,
        current_date: Optional[date] = None,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        self._context_limit_hit = False
        budget = self._create_budget_tracker(
            warmup=warmup_budget,
            max_actions_override=max_actions,
        )
        has_structured_memory = isinstance(self._memory, (StructuredMemory, ActiveMemory))
        has_active_memory = isinstance(self._memory, ActiveMemory)

        if enable_search is None:
            enable_search = bool(self._search_handler.is_available)

        tools = build_action_tools(
            enable_query=enable_query,
            enable_search=enable_search,
            max_outcomes_per_question=self.config.max_outcomes_per_question,
            max_search_results=self.config.max_search_results,
            search_chunk_tokens=self._search_handler.chunk_tokens,
            enable_memory=has_structured_memory,
            enable_mem_df=has_active_memory,
        )
        budget.bootstrap_context({"messages": messages, "tools": tools})

        all_forecasts: List[Dict[str, Any]] = []
        context_limit_hit = False
        memory_phase = False
        memory_phase_eligible = (
            current_date is not None
            and has_structured_memory
            and budget.settings.memory_phase_threshold_tokens is not None
        )
        memory_tools = (
            build_memory_phase_tools(
                enable_memory=has_structured_memory,
                enable_mem_df=has_active_memory,
            )
            if has_structured_memory
            else None
        )
        raw_stream = "warmup" if warmup_budget else "daily"
        final_submit_prompt_injected = False
        final_submit_retry_used = False
        consecutive_llm_failures = 0
        consecutive_content_filters = 0
        pending_input_delta: List[Any] = list(messages)

        while not budget.is_exhausted():
            if not memory_phase and memory_phase_eligible and budget.should_enter_memory_phase():
                memory_phase = True
                budget.memory_phase = True
                memory_prompt = self._build_memory_phase_prompt(current_date)
                message = {"role": "user", "content": memory_prompt}
                self._append_with_budget(messages, budget, message)
                pending_input_delta.append(message)

            force_final_submit_turn = (
                target_qid is not None
                and not memory_phase
                and budget.should_force_submit()
            )
            if force_final_submit_turn and not final_submit_prompt_injected:
                message = {
                    "role": "user",
                    "content": self._build_final_submit_tool_instruction(target_qid, budget),
                }
                self._append_with_budget(messages, budget, message)
                pending_input_delta.append(message)
                final_submit_prompt_injected = True

            if memory_phase:
                effective_tools = memory_tools
            elif force_final_submit_turn:
                submit_only = [tool for tool in tools if tool.get("function", {}).get("name") == "submit_forecasts"]
                effective_tools = submit_only if submit_only else tools
            else:
                effective_tools = tools

            model_input_delta = list(pending_input_delta)
            pending_input_delta = []
            sampling_params = dict(self.config.sampling_params or {})
            sampling_params.setdefault("tool_choice", "auto")
            if memory_phase and self.config.parallel_tool_calls:
                sampling_params["parallel_tool_calls"] = True

            try:
                with self._timer.track("llm"):
                    resp_json = self._call_chat_json_with_retries(
                        messages=messages,
                        tools=effective_tools,
                        sampling_params=sampling_params,
                    )
            except Exception as e:
                if self._is_fatal_inference_failure(e):
                    raise RuntimeError(f"Fatal inference failure in action loop: {e}") from e
                if self._is_context_limit_error(e):
                    context_limit_hit = True
                    self._context_limit_hit = True
                    self._log_chat_tools_action(
                        messages=messages,
                        rendered_response=f"CONTEXT LIMIT: {e}",
                        phase="llm_context_limit",
                        budget=budget,
                        qid=target_qid,
                        error=str(e),
                        raw_stream=raw_stream,
                        prompt_override=model_input_delta,
                    )
                    if target_qid is None:
                        print(f"[{self.agent_id}] Context limit reached; ending this wakeup early.", flush=True)
                    else:
                        print(
                            f"[{self.agent_id}] Context limit reached for qid {target_qid}; "
                            "skipping the rest of this question.",
                            flush=True,
                        )
                    break

                if force_final_submit_turn and not final_submit_retry_used:
                    final_submit_retry_used = True
                    continue
                if force_final_submit_turn:
                    break

                budget.consume_action()
                consecutive_llm_failures += 1
                err_msg = f"LLM ERROR after retries: {e}"
                self._log_chat_tools_action(
                    messages=messages,
                    rendered_response=err_msg,
                    phase="llm_error",
                    budget=budget,
                    qid=target_qid,
                    error=str(e),
                    consecutive_llm_failures=consecutive_llm_failures,
                    raw_stream=raw_stream,
                    prompt_override=model_input_delta,
                )
                message = {"role": "user", "content": budget.format_feedback(err_msg)}
                self._append_with_budget(messages, budget, message)
                pending_input_delta.append(message)
                if consecutive_llm_failures >= 3:
                    print(
                        f"[{self.agent_id}] Stopping after {consecutive_llm_failures} consecutive LLM failures.",
                        flush=True,
                    )
                    break
                continue

            consecutive_llm_failures = 0
            usage = self._normalize_chat_usage(resp_json)
            self._timer.record_tokens(usage)
            self._timer.record_cost(usage.get("cost", 0), "llm")
            budget.record_usage(usage)
            reasoning = usage.get("_reasoning_content") if usage else None
            reasoning_tokens_turn = self.extract_reasoning_token_count({"usage": usage})

            parsed, assistant_text, tool_calls = chat_response_to_action(resp_json)
            assistant_message = extract_assistant_message(resp_json)
            finish_reason = extract_finish_reason(resp_json)
            if finish_reason == "content_filter":
                consecutive_content_filters += 1
                if consecutive_content_filters >= self.config.content_filter_circuit_breaker:
                    print(f"  [{self.agent_id}] Circuit breaker: {consecutive_content_filters} "
                          f"consecutive content_filter responses, ending phase")
                    break
            else:
                consecutive_content_filters = 0
            appended = self._append_assistant_message(messages=messages, assistant_message=assistant_message)
            if appended is not None:
                budget.record_appended_item(appended)

            rendered = self._render_turn_for_logging(assistant_text, tool_calls)
            # turn_log_extra collects handler-level metadata (submitted_qids,
            # dropped_forecasts, errors, etc.) so a single post-dispatch log
            # call captures everything without double-logging.
            turn_log_extra: Dict[str, Any] = {}

            valid_final_forecasts: List[Dict[str, Any]] = []
            has_valid_final_submit = False
            if parsed is not None and parsed.action_type == "submit" and parsed.forecasts:
                valid_final_forecasts = list(parsed.forecasts)
                if target_qid is not None:
                    valid_final_forecasts = [f for f in valid_final_forecasts if f.get("qid") == target_qid]
                has_valid_final_submit = bool(valid_final_forecasts) and self._forecasts_within_probability_bounds(
                    valid_final_forecasts
                )
            if force_final_submit_turn and not has_valid_final_submit:
                if not final_submit_retry_used:
                    final_submit_retry_used = True
                    continue
                break

            if parsed is None:
                budget.consume_action()
                self._log_chat_tools_action(
                    messages=messages, rendered_response=rendered,
                    phase="memory_update" if memory_phase else "llm",
                    budget=budget, qid=target_qid, reasoning=reasoning,
                    reasoning_tokens=reasoning_tokens_turn, finish_reason=finish_reason,
                    usage=usage, raw_stream=raw_stream, prompt_override=model_input_delta,
                )
                message = {
                    "role": "user",
                    "content": budget.format_feedback("No tool call detected. Call at least one tool each turn."),
                }
                self._append_with_budget(messages, budget, message)
                pending_input_delta.append(message)
                continue

            if parsed.action_type == "next":
                if not memory_phase and memory_phase_eligible:
                    memory_phase = True
                    budget.memory_phase = True
                    self._log_chat_tools_action(
                        messages=messages,
                        rendered_response=rendered,
                        phase="next_entering_memory",
                        budget=budget,
                        qid=target_qid,
                        raw_stream=raw_stream,
                    )
                    memory_prompt = self._build_memory_phase_prompt(current_date)
                    message = {"role": "user", "content": memory_prompt}
                    self._append_with_budget(messages, budget, message)
                    pending_input_delta.append(message)
                    continue

                phase_label = "memory_update_done" if memory_phase else "next_day"
                self._log_chat_tools_action(
                    messages=messages,
                    rendered_response=rendered,
                    phase=phase_label,
                    budget=budget,
                    qid=target_qid,
                    raw_stream=raw_stream,
                )
                break

            if memory_phase:
                before_len = len(messages)
                saw_next_day = False
                _mem_actions: List[str] = []
                for tc in tool_calls:
                    tc_parsed, _ = single_call_to_parsed_action(tc)
                    if tc_parsed is None:
                        continue
                    if tc_parsed.action_type == "next":
                        saw_next_day = True
                        break
                    _mem_actions.append(tc_parsed.action_type)
                    if tc_parsed.action_type in ("memory_retrieve", "memory_new", "memory_add",
                                                  "memory_update", "memory_delete"):
                        self._handle_memory_action(
                            messages, forecast_interface, rendered, tc_parsed,
                            budget, reasoning=reasoning, tool_call=tc,
                        )
                    elif tc_parsed.action_type in ("mem_add", "mem_update", "mem_delete"):
                        self._handle_mem_action(
                            messages, forecast_interface, rendered, tc_parsed,
                            budget, reasoning=reasoning, tool_call=tc,
                        )
                    else:
                        self._handle_invalid(
                            messages, forecast_interface, rendered, tc_parsed,
                            budget, reasoning=reasoning, tool_call=tc,
                        )
                        if tc_parsed.error:
                            turn_log_extra["error"] = tc_parsed.error
                pending_input_delta.extend(messages[before_len:])
                _mem_phase = "memory_update_done" if saw_next_day else "memory_update"
                if _mem_actions:
                    turn_log_extra["actions"] = _mem_actions
                self._log_chat_tools_action(
                    messages=messages, rendered_response=rendered,
                    phase=_mem_phase, budget=budget,
                    qid=target_qid, reasoning=reasoning,
                    reasoning_tokens=reasoning_tokens_turn, finish_reason=finish_reason,
                    usage=usage, raw_stream=raw_stream,
                    prompt_override=model_input_delta, **turn_log_extra,
                )
                if saw_next_day:
                    break
                continue

            before_len = len(messages)
            break_outer = False
            _turn_actions: List[str] = []
            for tc in tool_calls:
                tc_parsed, _ = single_call_to_parsed_action(tc)
                if tc_parsed is None:
                    continue
                _turn_actions.append(tc_parsed.action_type)
                if tc_parsed.action_type == "query":
                    self._handle_query(
                        messages, forecast_interface, rendered, tc_parsed,
                        budget, qid=target_qid, reasoning=reasoning,
                        raw_stream=raw_stream, tool_call=tc,
                    )
                    if tc_parsed.error:
                        turn_log_extra["error"] = tc_parsed.error
                elif tc_parsed.action_type == "search":
                    self._handle_search(
                        messages, forecast_interface, rendered, tc_parsed,
                        budget, qid=target_qid, reasoning=reasoning,
                        raw_stream=raw_stream, tool_call=tc,
                    )
                    if tc_parsed.error:
                        turn_log_extra["error"] = tc_parsed.error
                elif tc_parsed.action_type in ("memory_retrieve", "memory_new", "memory_add", "memory_update", "memory_delete"):
                    self._handle_memory_action(
                        messages, forecast_interface, rendered, tc_parsed,
                        budget, reasoning=reasoning, tool_call=tc,
                    )
                elif tc_parsed.action_type in ("mem_add", "mem_update", "mem_delete"):
                    self._handle_mem_action(
                        messages, forecast_interface, rendered, tc_parsed,
                        budget, reasoning=reasoning, tool_call=tc,
                    )
                elif tc_parsed.action_type == "submit":
                    if target_qid is not None and tc_parsed.forecasts:
                        tc_parsed.forecasts = [f for f in tc_parsed.forecasts if f.get("qid") == target_qid]
                    n_before = len(tc_parsed.forecasts) if tc_parsed.forecasts else 0
                    forecasts = self._handle_submit(
                        messages, forecast_interface, rendered, tc_parsed,
                        budget, qid=target_qid, reasoning=reasoning,
                        raw_stream=raw_stream, tool_call=tc,
                    )
                    turn_log_extra["submitted_qids"] = [f["qid"] for f in forecasts]
                    turn_log_extra["num_forecasts"] = len(forecasts)
                    turn_log_extra["dropped_forecasts"] = n_before - len(forecasts)
                    if tc_parsed.error:
                        turn_log_extra["error"] = tc_parsed.error
                    all_forecasts.extend(forecasts)
                    if target_qid is not None and forecasts:
                        break_outer = True
                        break
                else:
                    self._handle_invalid(
                        messages, forecast_interface, rendered, tc_parsed,
                        budget, qid=target_qid, reasoning=reasoning,
                        raw_stream=raw_stream, tool_call=tc,
                    )
                    if tc_parsed.error:
                        turn_log_extra["error"] = tc_parsed.error
            # Build phase from all action types seen this turn
            if _turn_actions:
                _log_phase = "+".join(_turn_actions) if len(_turn_actions) > 1 else _turn_actions[0]
                turn_log_extra["actions"] = _turn_actions
            else:
                _log_phase = "llm"
            self._log_chat_tools_action(
                messages=messages, rendered_response=rendered,
                phase=_log_phase, budget=budget, qid=target_qid,
                reasoning=reasoning, reasoning_tokens=reasoning_tokens_turn,
                finish_reason=finish_reason, usage=usage, raw_stream=raw_stream,
                prompt_override=model_input_delta, **turn_log_extra,
            )
            pending_input_delta.extend(messages[before_len:])
            if break_outer:
                break

        if memory_phase and self._memory is not None:
            self._memory._save(current_date)
        elif memory_phase_eligible and not memory_phase:
            print(f"  [{self.agent_id}] Budget exhausted before memory phase. Memory not updated in-loop.")

        self._memory_phase_completed = memory_phase
        return all_forecasts, context_limit_hit

    def _run_memory_update_loop_with_chat_tools(
        self,
        messages: List[Dict[str, Any]],
        forecast_interface,
        memory_prompt: str,
        current_date: date,
    ) -> None:
        mem_budget = BudgetTracker(
            BudgetSettings(
                max_total_tokens=self.config.memory_update_max_total_tokens,
                submit_reserve_tokens=self.config.submit_reserve_tokens,
                force_submit_threshold_tokens=self.config.force_submit_threshold_tokens,
            )
        )

        tool_prompt = memory_prompt
        messages.append({"role": "user", "content": tool_prompt})
        tools = build_memory_phase_tools(
            enable_memory=isinstance(self._memory, (StructuredMemory, ActiveMemory)),
            enable_mem_df=isinstance(self._memory, ActiveMemory),
        )
        mem_budget.bootstrap_context({"messages": [{"role": "user", "content": tool_prompt}], "tools": tools})
        sampling_params = dict(self.config.sampling_params or {})
        sampling_params.setdefault("tool_choice", "auto")
        if self.config.parallel_tool_calls:
            sampling_params["parallel_tool_calls"] = True
        consecutive_content_filters = 0

        while not mem_budget.is_exhausted():
            try:
                with self._timer.track("llm"):
                    resp_json = self._call_chat_json_with_retries(
                        messages=messages,
                        tools=tools,
                        sampling_params=sampling_params,
                    )
            except Exception as e:
                print(f"  [{self.agent_id}] Memory update LLM error in chat-tools mode: {e}")
                break

            usage = self._normalize_chat_usage(resp_json)
            self._timer.record_tokens(usage)
            self._timer.record_cost(usage.get("cost", 0), "llm")
            mem_budget.record_usage(usage)
            reasoning = usage.get("_reasoning_content") if usage else None

            finish_reason = extract_finish_reason(resp_json)
            if finish_reason == "content_filter":
                consecutive_content_filters += 1
                if consecutive_content_filters >= self.config.content_filter_circuit_breaker:
                    print(f"  [{self.agent_id}] Circuit breaker: {consecutive_content_filters} "
                          f"consecutive content_filter responses, ending memory phase")
                    break
            else:
                consecutive_content_filters = 0

            parsed, assistant_text, tool_calls = chat_response_to_action(resp_json)
            assistant_message = extract_assistant_message(resp_json)
            appended = self._append_assistant_message(messages=messages, assistant_message=assistant_message)
            if appended is not None:
                mem_budget.record_appended_item(appended)

            rendered = self._render_turn_for_logging(assistant_text, tool_calls)
            self._log_model_output(
                tool_prompt,
                rendered,
                {"phase": "memory_update", "current_memory_entries": self._memory.entry_count, "reasoning": reasoning},
            )

            if parsed is None:
                mem_budget.consume_action()
                self._append_feedback_message(
                    messages,
                    mem_budget,
                    "No tool call detected. Call at least one memory tool, or next_day() to finish.",
                )
                continue

            # Process all tool calls in the batch (parallel tool calls)
            saw_next_day = False
            for tc in tool_calls:
                tc_parsed, _ = single_call_to_parsed_action(tc)
                if tc_parsed is None:
                    continue
                if tc_parsed.action_type == "next":
                    saw_next_day = True
                    break
                if tc_parsed.action_type in ("memory_retrieve", "memory_new", "memory_add",
                                              "memory_update", "memory_delete"):
                    self._handle_memory_action(
                        messages, forecast_interface, rendered, tc_parsed,
                        mem_budget, reasoning=reasoning, tool_call=tc,
                    )
                elif tc_parsed.action_type in ("mem_add", "mem_update", "mem_delete"):
                    self._handle_mem_action(
                        messages, forecast_interface, rendered, tc_parsed,
                        mem_budget, reasoning=reasoning, tool_call=tc,
                    )
                else:
                    self._handle_invalid(
                        messages, forecast_interface, rendered, tc_parsed,
                        mem_budget, reasoning=reasoning, tool_call=tc,
                    )
            if saw_next_day:
                break

        self._memory._save(current_date)

    # =========================================================================
    # Helpers
    # =========================================================================

    def _record_matcher_timing(self, duration: float, cost: float = 0) -> None:
        """Record answer matcher latency and cost in timing stats."""
        self._timer.record("matcher", duration)
        self._timer.record_cost(cost, "matcher")

    def _set_log_date(self, current_date: date) -> None:
        self._log_date = current_date

    def _log_model_output(
        self,
        prompt: Any,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._log_date is None:
            return
        self._output_logger.log_model_output(self._log_date, prompt, response, metadata)

    def _flush_warmup_raw_logs(self) -> None:
        self._output_logger.flush_warmup_raw()
    
    # =========================================================================
    # Memory Update Loop
    # =========================================================================

    def _prompt_memory_update(self, messages: List[Dict[str, str]],
                               forecast_interface, current_date: date) -> None:
        """
        Ask the agent to update its memory at the end of the day.

        Structured and active memory reuse the chat-tools memory loop.
        Plain memory uses a full-text replacement prompt.
        """
        if isinstance(self._memory, ActiveMemory):
            memory_prompt = self._build_active_memory_prompt(current_date, day_qids=getattr(self, '_day_qids', None))
        elif isinstance(self._memory, StructuredMemory):
            memory_prompt = self._build_structured_memory_prompt(current_date)
        else:
            memory_prompt = self._build_plain_memory_prompt(current_date)

        # Structured/active memory use the same chat-tools protocol as the main loop.
        if isinstance(self._memory, (StructuredMemory, ActiveMemory)):
            self._run_memory_update_loop(messages, forecast_interface, memory_prompt, current_date)
            return

        # Plain memory is the only remaining non-tool path: return the full replacement
        # memory text directly, or `NO_MEMORY_UPDATE` to keep the current memory.
        messages.append({"role": "user", "content": memory_prompt})
        response, usage = self.inference.chat(messages, self.config.sampling_params)
        self._timer.record_tokens(usage)
        self._timer.record_cost(usage.get("cost", 0), "llm")

        if not response:
            print(f"  [{self.agent_id}] Memory update got empty response from LLM, skipping.")
            return

        messages.append({"role": "assistant", "content": response})

        reasoning = usage.get("_reasoning_content") if usage else None

        self._log_model_output(
            memory_prompt, response,
            {"phase": "memory_update", "current_memory_len": len(self._memory), "reasoning": reasoning}
        )

        new_memory = response.strip()
        if new_memory and new_memory != "NO_MEMORY_UPDATE":
            self._memory.update(new_memory)

    def _run_memory_update_loop(self, messages: List[Dict[str, str]],
                                 forecast_interface, memory_prompt: str,
                                 current_date: date) -> None:
        """Fallback end-of-day memory loop for structured/active memory."""
        if isinstance(self._memory, (StructuredMemory, ActiveMemory)):
            self._run_memory_update_loop_with_chat_tools(
                messages, forecast_interface, memory_prompt, current_date
            )
            return
        raise TypeError("Structured/active memory loop called without structured or active memory enabled.")
