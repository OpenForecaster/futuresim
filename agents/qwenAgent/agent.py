from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from agents.allQAgent.agent import AllQAgent, WarmupLoopResult
from agents.basicAgent.agent import BasicAgent
from agents.basicAgent.search import SearchHandler
from agents.utils.budget import BudgetTracker
from agents.utils.forecast_parser import ParsedAction
from agents.utils.memory import ActiveMemory, StructuredMemory
from environment.interfaces import PredictionSubmission

from .tools import (
    build_action_tools,
    build_memory_phase_tools,
    chat_response_to_action,
    extract_assistant_message,
    extract_finish_reason,
)


def qwen_final_submit_instruction_text(
    target_qid: str,
    budget: Optional[BudgetTracker],
    *,
    forbid_query_df: bool = True,
) -> str:
    """Force-submit user message for Qwen loops (eval) and SkyRL warmup.

    ``budget`` only adds remaining-budget lines (same idea as ``BasicAgent._build_force_submit_preamble``).

    When ``query_df`` is present in the tool schema (full daily eval), set ``forbid_query_df=True``
    so the model is told not to call it on the force-submit turn. SkyRL warmup builds tools with
    ``enable_query=False`` — there is no ``query_df`` tool — so use ``forbid_query_df=False`` and
    only forbid ``search_news`` / ``next_day``.
    """
    lines = ["FINAL ACTION: You MUST submit your best guess forecast now."]
    if budget is not None:
        status = budget.status_text()
        if status:
            lines.append(status)
    preamble = "\n".join(lines)
    if forbid_query_df:
        forbid = "Do NOT call query_df, search_news, or next_day.\n"
    else:
        forbid = "Do NOT call search_news or next_day.\n"
    return (
        f"{preamble}\n"
        f"Target question ID: {target_qid}\n"
        "Call exactly one tool: submit_forecasts.\n"
        f"{forbid}"
        "Submit only this question id with concrete outcomes and probabilities (sum <= 1.0)."
    )


class QwenBasicAgent(BasicAgent):
    """
    BasicAgent-compatible behavior, but obtains actions via Chat Completions tool
    calls (assistant.tool_calls + role=tool outputs).

    This is intended for Qwen models served by vLLM with native parser flags such
    as `--tool-call-parser qwen3_coder`.
    """

    def act(self, doc_interface, forecast_interface, current_date: date) -> List[Dict[str, Any]]:
        self._timer.reset()
        self._timer.start_day()
        self._day_qids = set()

        if self._memory is not None:
            self._memory.set_date(current_date)

        self._setup_day(forecast_interface, current_date)

        messages = [{"role": "user", "content": self._build_qwen_instructions(current_date)}]
        all_forecasts, context_limit_hit = self._run_qwen_action_loop(
            messages=messages,
            forecast_interface=forecast_interface,
            current_date=current_date,
        )

        # Separate memory update only for BasicMemory (plain text, single-shot).
        # StructuredMemory/ActiveMemory are handled inside _run_qwen_action_loop
        # via the next-triggered or token-threshold-triggered memory phase.
        if (
            self._memory is not None
            and not context_limit_hit
            and not getattr(self, '_memory_phase_completed', False)
            and not isinstance(self._memory, (StructuredMemory, ActiveMemory))
        ):
            self._prompt_memory_update(messages, forecast_interface, current_date)

        self._timer.end_day()
        if self.config.memory_dir:
            self._timer.save_day_stats(self.config.memory_dir, current_date)

        forecast_interface.next_day()
        return all_forecasts

    def _build_qwen_instructions(self, current_date: date) -> str:
        base_prompt = self._build_instructions(current_date)
        return self._rewrite_basic_prompt_for_qwen(base_prompt)

    def _rewrite_basic_prompt_for_qwen(self, base_prompt: str) -> str:
        prefix = base_prompt
        if "## RESPONSE FORMAT" in base_prompt:
            prefix = base_prompt.split("## RESPONSE FORMAT", 1)[0].rstrip()
        start_budget = self._build_start_budget_status()
        start_block = f"\nBudget at start:\n{start_budget}\n\nBegin." if start_budget else "\nBegin."

        return (
            f"{prefix}\n\n"
            "## RESPONSE FORMAT\n"
            "Use function tools as defined in the tool schema, following the signatures, "
            "constraints, and examples there. Call exactly one tool per turn.\n\n"
            "## INTERACTION FLOW\n"
            f"{self._build_budget_overview()}\n"
            "When ready to move on, call `next_day()` to end this session.\n\n"
            "## SUBMISSION RULES\n"
            "- You may call `submit_forecasts` multiple times in a single session until the active loop budget runs out.\n"
            "- Each `submit_forecasts` call must include exactly one forecast for exactly one qid.\n"
            "- You may update earlier same-session forecasts by submitting again for the same qid.\n"
            f"{start_block}"
        )

    def _build_final_submit_tool_instruction(self, target_qid: str, budget: BudgetTracker) -> str:
        return qwen_final_submit_instruction_text(target_qid, budget)

    def _run_qwen_action_loop(
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
        budget = self._create_budget_tracker(
            warmup=warmup_budget,
            max_actions_override=max_actions,
        )
        all_forecasts: List[Dict[str, Any]] = []
        context_limit_hit = False

        if enable_search is None:
            enable_search = bool(self._search_handler.is_available)

        has_structured_memory = isinstance(getattr(self, '_memory', None), (StructuredMemory, ActiveMemory))
        has_active_memory = isinstance(getattr(self, '_memory', None), ActiveMemory)

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

        # Memory phase state
        memory_phase = False
        memory_phase_eligible = (
            current_date is not None
            and has_structured_memory
            and budget.settings.memory_phase_threshold_tokens is not None
        )
        memory_tools = None  # lazy-built on first transition

        raw_stream = "warmup" if warmup_budget else "daily"
        final_submit_prompt_injected = False
        final_submit_retry_used = False
        consecutive_llm_failures = 0
        pending_input_delta: List[Any] = list(messages)

        while not budget.is_exhausted():
            # --- Memory phase auto-transition on token threshold ---
            if not memory_phase and memory_phase_eligible and budget.should_enter_memory_phase():
                memory_phase = True
                budget.memory_phase = True
                if memory_tools is None:
                    memory_tools = build_memory_phase_tools(
                        enable_memory=has_structured_memory, enable_mem_df=has_active_memory,
                    )
                memory_prompt = self._build_memory_phase_prompt(current_date)
                msg = {"role": "user", "content": memory_prompt}
                self._append_with_budget(messages, budget, msg)
                pending_input_delta.append(msg)

            force_final_submit_turn = target_qid is not None and budget.should_force_submit() and not memory_phase
            if force_final_submit_turn and not final_submit_prompt_injected:
                message = {
                    "role": "user",
                    "content": self._build_final_submit_tool_instruction(target_qid, budget),
                }
                self._append_with_budget(messages, budget, message)
                pending_input_delta.append(message)
                final_submit_prompt_injected = True

            # Select tools for this turn
            if memory_phase:
                if memory_tools is None:
                    memory_tools = build_memory_phase_tools(
                        enable_memory=has_structured_memory, enable_mem_df=has_active_memory,
                    )
                effective_tools = memory_tools
            elif force_final_submit_turn:
                submit_only = [t for t in tools if t.get("function", {}).get("name") == "submit_forecasts"]
                effective_tools = submit_only if submit_only else tools
            else:
                effective_tools = tools

            model_input_delta = list(pending_input_delta)
            pending_input_delta = []

            try:
                with self._timer.track("llm"):
                    resp_json = self._call_chat_json_with_retries(
                        messages=messages,
                        tools=effective_tools,
                        sampling_params=self.config.sampling_params,
                    )
            except Exception as e:
                if self._is_fatal_inference_failure(e):
                    raise RuntimeError(f"Fatal inference failure in action loop: {e}") from e
                if self._is_context_limit_error(e):
                    context_limit_hit = True
                    self._log_qwen_action(
                        forecast_interface=forecast_interface,
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
                        print(
                            f"[{self.agent_id}] Context limit reached; ending this wakeup early.",
                            flush=True,
                        )
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
                self._log_qwen_action(
                    forecast_interface=forecast_interface,
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
            budget.record_usage(usage)
            reasoning_tokens_turn = BasicAgent.extract_reasoning_token_count({"usage": usage})

            parsed, assistant_text, tool_calls = chat_response_to_action(resp_json)
            assistant_message = extract_assistant_message(resp_json)
            finish_reason = extract_finish_reason(resp_json)

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

            appended = self._append_assistant_message(messages=messages, assistant_message=assistant_message)
            if appended is not None:
                budget.record_appended_item(appended)

            rendered = self._render_turn_for_logging(assistant_text, tool_calls)
            self._log_qwen_action(
                forecast_interface=forecast_interface,
                messages=messages,
                rendered_response=rendered,
                phase="llm" if not memory_phase else "memory_update",
                budget=budget,
                qid=target_qid,
                finish_reason=finish_reason,
                usage=usage,
                reasoning_tokens=reasoning_tokens_turn,
                raw_stream=raw_stream,
                prompt_override=model_input_delta,
            )

            if parsed is None:
                budget.consume_action()
                message = {
                    "role": "user",
                    "content": budget.format_feedback("No tool call detected. You MUST call exactly one tool each turn."),
                }
                self._append_with_budget(messages, budget, message)
                pending_input_delta.append(message)
                continue

            if parsed.action_type == "next":
                if not memory_phase and memory_phase_eligible:
                    # Transition to memory phase instead of ending the day
                    memory_phase = True
                    budget.memory_phase = True
                    if memory_tools is None:
                        memory_tools = build_memory_phase_tools(
                            enable_memory=has_structured_memory, enable_mem_df=has_active_memory,
                        )
                    self._log_qwen_action(
                        forecast_interface=forecast_interface,
                        messages=messages,
                        rendered_response=rendered,
                        phase="next_entering_memory",
                        budget=budget,
                        qid=target_qid,
                        raw_stream=raw_stream,
                    )
                    memory_prompt = self._build_memory_phase_prompt(current_date)
                    msg = {"role": "user", "content": memory_prompt}
                    self._append_with_budget(messages, budget, msg)
                    pending_input_delta.append(msg)
                    continue
                else:
                    phase_label = "memory_update_done" if memory_phase else "next_day"
                    self._log_qwen_action(
                        forecast_interface=forecast_interface,
                        messages=messages,
                        rendered_response=rendered,
                        phase=phase_label,
                        budget=budget,
                        qid=target_qid,
                        raw_stream=raw_stream,
                    )
                    break

            # --- Memory action routing ---
            if parsed.action_type in ("memory_retrieve", "memory_new", "memory_update", "memory_delete"):
                before_len = len(messages)
                self._qwen_handle_memory_action(
                    messages=messages,
                    parsed=parsed,
                    tool_call=tool_calls[0] if tool_calls else None,
                    budget=budget,
                )
                pending_input_delta.extend(messages[before_len:])
                continue

            if parsed.action_type in ("mem_add", "mem_update", "mem_delete"):
                before_len = len(messages)
                self._qwen_handle_mem_action(
                    messages=messages,
                    parsed=parsed,
                    tool_call=tool_calls[0] if tool_calls else None,
                    budget=budget,
                )
                pending_input_delta.extend(messages[before_len:])
                continue

            if parsed.action_type == "query":
                before_len = len(messages)
                self._qwen_handle_query(
                    messages=messages,
                    parsed=parsed,
                    tool_call=tool_calls[0] if tool_calls else None,
                    budget=budget,
                )
                pending_input_delta.extend(messages[before_len:])
                continue

            if parsed.action_type == "search":
                before_len = len(messages)
                self._qwen_handle_search(
                    messages=messages,
                    parsed=parsed,
                    tool_call=tool_calls[0] if tool_calls else None,
                    budget=budget,
                )
                pending_input_delta.extend(messages[before_len:])
                continue

            if parsed.action_type == "submit":
                before_len = len(messages)
                submitted = self._qwen_handle_submit(
                    messages=messages,
                    forecast_interface=forecast_interface,
                    parsed=parsed,
                    tool_call=tool_calls[0] if tool_calls else None,
                    budget=budget,
                    target_qid=target_qid,
                )
                pending_input_delta.extend(messages[before_len:])
                all_forecasts.extend(submitted)
                if target_qid is not None and submitted:
                    break
                continue

            budget.consume_action()
            err = parsed.error or "Unknown action/tool."
            message = {"role": "user", "content": budget.format_feedback(f"Invalid action: {err}")}
            self._append_with_budget(messages, budget, message)
            pending_input_delta.append(message)

        # Post-loop: persist memory if memory phase was entered
        if memory_phase and self._memory is not None:
            self._memory._save(current_date)
        elif memory_phase_eligible and not memory_phase:
            print(f"  [{self.agent_id}] Budget exhausted before memory phase. Memory not updated in-loop.")

        self._memory_phase_completed = memory_phase
        return all_forecasts, context_limit_hit

    def _call_chat_json(
        self,
        *,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        sampling_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        from inference.vllm import VLLMInference

        if not isinstance(self.inference, VLLMInference):
            raise TypeError("QwenBasicAgent currently requires provider=vllm (VLLMInference).")

        sp = dict(sampling_params or {})
        sp["tools"] = tools
        sp["tool_choice"] = sp.get("tool_choice", "auto")
        return self.inference.chat_json(messages, sp)

    @staticmethod
    def _is_fatal_inference_failure(error: Exception) -> bool:
        msg = str(error).lower()
        fatal_markers = (
            "vllm server died on port",
            "vllm server failed to start",
            "vllm server process died immediately",
            "engine core initialization failed",
            "cuda out of memory occurred when warming up sampler",
        )
        return any(marker in msg for marker in fatal_markers)

    @staticmethod
    def _is_context_limit_error(error: Exception) -> bool:
        msg = str(error).lower()
        if "400 bad request" not in msg:
            return False
        context_markers = (
            "maximum input length",
            "maximum context length",
            "context length is only",
            "param=input_tokens",
            "parameter=input_tokens",
        )
        return any(marker in msg for marker in context_markers)

    def _call_chat_json_with_retries(
        self,
        *,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        sampling_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        # VLLMInference already retries chat/completions transport failures up to 3 times.
        # Do not stack another retry loop on top of that.
        return self._call_chat_json(messages=messages, tools=tools, sampling_params=sampling_params)

    @staticmethod
    def _normalize_chat_usage(resp_json: Dict[str, Any]) -> Dict[str, Any]:
        raw_usage = resp_json.get("usage") or {}
        usage = dict(raw_usage) if isinstance(raw_usage, dict) else {}
        if "prompt_tokens" not in usage:
            usage["prompt_tokens"] = usage.get("input_tokens", 0)
        if "completion_tokens" not in usage:
            usage["completion_tokens"] = usage.get("output_tokens", 0)
        if "total_tokens" not in usage:
            usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        if "completion_tokens_details" not in usage and "output_tokens_details" in usage:
            usage["completion_tokens_details"] = usage.get("output_tokens_details")
        if "prompt_tokens_details" not in usage and "input_tokens_details" in usage:
            usage["prompt_tokens_details"] = usage.get("input_tokens_details")
        return usage

    @staticmethod
    def _append_assistant_message(
        *,
        messages: List[Dict[str, Any]],
        assistant_message: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(assistant_message, dict):
            return None
        tool_calls = assistant_message.get("tool_calls")
        content = assistant_message.get("content")
        has_tool_calls = isinstance(tool_calls, list) and len(tool_calls) > 0
        has_text = isinstance(content, str) and bool(content.strip())

        if not has_tool_calls and not has_text:
            return None

        out: Dict[str, Any] = {"role": "assistant"}
        if "content" in assistant_message:
            out["content"] = content if content is not None else ""
        if has_tool_calls:
            out["tool_calls"] = tool_calls
        messages.append(out)
        return out

    @staticmethod
    def _append_tool_output_message(
        messages: List[Dict[str, Any]],
        *,
        tool_call: Optional[Dict[str, Any]],
        tool_name: str,
        output: str,
    ) -> Dict[str, Any]:
        call_id = tool_call.get("call_id") if isinstance(tool_call, dict) else None
        if isinstance(call_id, str) and call_id:
            message = {
                "role": "tool",
                "tool_call_id": call_id,
                "name": tool_name,
                "content": output,
            }
            messages.append(message)
            return message

        # Fallback for parser/provider variants that do not expose call id.
        message = {"role": "user", "content": f"{tool_name} output:\n{output}"}
        messages.append(message)
        return message

    def _qwen_handle_query(
        self,
        *,
        messages: List[Dict[str, Any]],
        parsed: ParsedAction,
        tool_call: Optional[Dict[str, Any]],
        budget: BudgetTracker,
    ) -> None:
        budget.consume_action()
        if parsed.code:
            with self._timer.track("df_query"):
                result, error = self._query_handler.execute(parsed.code)
            if error:
                feedback = f"QUERY ERROR: {error}"
            else:
                feedback = f"QUERY RESULT:\n{result}"
        else:
            feedback = f"QUERY ERROR: {parsed.error or 'Missing code'}"

        appended = self._append_tool_output_message(
            messages,
            tool_call=tool_call,
            tool_name="query_df",
            output=budget.format_feedback(feedback),
        )
        budget.record_appended_item(appended)

    def _qwen_handle_search(
        self,
        *,
        messages: List[Dict[str, Any]],
        parsed: ParsedAction,
        tool_call: Optional[Dict[str, Any]],
        budget: BudgetTracker,
    ) -> None:
        budget.consume_action()
        min_date, max_date = qwen_optional_search_dates_from_parsed(parsed)
        with self._timer.track("search"):
            effect = qwen_execute_news_search(
                parsed,
                self._search_handler,
                max_results=self.config.max_search_results,
                min_date=min_date,
                max_date=max_date,
            )
        appended = self._append_tool_output_message(
            messages,
            tool_call=tool_call,
            tool_name="search_news",
            output=budget.format_feedback(effect.feedback),
        )
        budget.record_appended_item(appended)

    def _qwen_handle_submit(
        self,
        *,
        messages: List[Dict[str, Any]],
        forecast_interface,
        parsed: ParsedAction,
        tool_call: Optional[Dict[str, Any]],
        budget: BudgetTracker,
        target_qid: Optional[str],
    ) -> List[Dict[str, Any]]:
        budget.consume_action()
        submitted: List[Dict[str, Any]] = []
        submit_errors: List[str] = []
        dropped_forecasts = 0

        forecasts = list(parsed.forecasts or [])
        if target_qid is not None:
            forecasts = [f for f in forecasts if f.get("qid") == target_qid]

        if len(forecasts) > 1:
            dropped_forecasts = len(forecasts) - 1
            forecasts = [forecasts[0]]

        if forecasts:
            for f in forecasts:
                try:
                    pred = PredictionSubmission(question_id=f["qid"], outcomes=f["outcomes"])
                    forecast_interface.submit_prediction(pred)
                    submitted.append(f)
                except Exception as e:
                    submit_errors.append(f"{f.get('qid', 'unknown_qid')}: {e}")
            if submitted:
                self._query_handler.invalidate_cache()
                sub = submitted[0]
                outcomes_str = ", ".join(f"{k}: {v:.2f}" for k, v in sub["outcomes"].items())
                title = self._query_handler.get_question_title(sub["qid"])
                title_str = f" ({title})" if title else ""
                feedback = (
                    f"Submitted forecast for qid={sub['qid']}{title_str}: {outcomes_str}."
                )
                if dropped_forecasts > 0:
                    feedback += (
                        f"\nIgnored {dropped_forecasts} extra forecast item(s); "
                        "submit exactly one qid per call."
                    )
            else:
                detail = "; ".join(submit_errors) if submit_errors else "Forecast submission failed."
                feedback = f"SUBMIT ERROR: {detail}"
        else:
            feedback = f"SUBMIT ERROR: {parsed.error or 'No valid forecasts'}"

        appended = self._append_tool_output_message(
            messages,
            tool_call=tool_call,
            tool_name="submit_forecasts",
            output=budget.format_feedback(feedback),
        )
        budget.record_appended_item(appended)
        return submitted

    def _qwen_handle_memory_action(
        self,
        *,
        messages: List[Dict[str, Any]],
        parsed: ParsedAction,
        tool_call: Optional[Dict[str, Any]],
        budget: BudgetTracker,
    ) -> None:
        """Handle memory tool calls (retrieve/new/update/delete) via Qwen tool output."""
        budget.consume_action()
        tool_name = {
            "memory_retrieve": "memory_retrieve",
            "memory_new": "memory_new",
            "memory_update": "memory_update",
            "memory_delete": "memory_delete",
        }.get(parsed.action_type, "memory_retrieve")

        if not isinstance(self._memory, (StructuredMemory, ActiveMemory)):
            feedback = "MEMORY ERROR: Structured memory is not enabled."
        elif parsed.error:
            feedback = f"MEMORY ERROR: {parsed.error}"
        elif parsed.action_type == "memory_retrieve":
            entry = self._memory.retrieve(parsed.memory_entry_name)
            if entry is None:
                feedback = f"MEMORY ERROR: No entry with name '{parsed.memory_entry_name}'."
            else:
                feedback = f"MEMORY ENTRY:\n{entry}"
        elif parsed.action_type == "memory_new":
            data = parsed.memory_new_data
            try:
                entry_name = self._memory.add_entry(data["name"], data["description"], data["content"])
                max_ent = self._memory._max_entries if hasattr(self._memory, '_max_entries') else '?'
                feedback = f"MEMORY: Added entry [{entry_name}]. Total entries: {self._memory.entry_count}/{max_ent}."
            except ValueError as exc:
                feedback = f"MEMORY ERROR: {exc}"
        elif parsed.action_type == "memory_update":
            update_data = {k: v for k, v in parsed.memory_update_data.items() if k != "name"}
            ok = self._memory.update_entry(parsed.memory_entry_name, **update_data)
            if ok:
                feedback = f"MEMORY: Updated [{parsed.memory_entry_name}]."
            else:
                feedback = f"MEMORY ERROR: No entry with name '{parsed.memory_entry_name}'."
        elif parsed.action_type == "memory_delete":
            ok = self._memory.delete_entry(parsed.memory_entry_name)
            if ok:
                feedback = f"MEMORY: Deleted [{parsed.memory_entry_name}]. Remaining: {self._memory.entry_count}."
            else:
                feedback = f"MEMORY ERROR: No entry with name '{parsed.memory_entry_name}'."
        else:
            feedback = f"MEMORY ERROR: Unknown memory action '{parsed.action_type}'."

        appended = self._append_tool_output_message(
            messages,
            tool_call=tool_call,
            tool_name=tool_name,
            output=budget.format_feedback(feedback),
        )
        budget.record_appended_item(appended)

    def _qwen_handle_mem_action(
        self,
        *,
        messages: List[Dict[str, Any]],
        parsed: ParsedAction,
        tool_call: Optional[Dict[str, Any]],
        budget: BudgetTracker,
    ) -> None:
        """Handle mem_df tool calls (mem_add/update/delete) for ActiveMemory via Qwen tool output."""
        budget.consume_action()
        tool_name = {
            "mem_add": "mem_add",
            "mem_update": "mem_update",
            "mem_delete": "mem_delete",
        }.get(parsed.action_type, "mem_add")

        if not isinstance(self._memory, ActiveMemory):
            feedback = "MEM ERROR: Active memory is not enabled."
        elif parsed.error:
            feedback = f"MEM ERROR: {parsed.error}"
        elif parsed.action_type == "mem_add":
            data = parsed.mem_data
            self._memory.mem_add(
                qid=data["qid"], question=data.get("question", ""),
                memory=data["memory"], category=data.get("category", ""),
            )
            feedback = f"MEM: Added entry for Q{data['qid']}. Total: {self._memory.mem_count} entries."
        elif parsed.action_type == "mem_update":
            self._memory.mem_update(
                qid=parsed.mem_qid, memory=parsed.mem_data["memory"],
                category=parsed.mem_data.get("category"),
            )
            feedback = f"MEM: Updated entry for Q{parsed.mem_qid}."
        elif parsed.action_type == "mem_delete":
            ok = self._memory.mem_delete(parsed.mem_qid)
            if ok:
                feedback = f"MEM: Deleted Q{parsed.mem_qid}. Remaining: {self._memory.mem_count}."
            else:
                feedback = f"MEM ERROR: No entry for Q{parsed.mem_qid}."
        else:
            feedback = f"MEM ERROR: Unknown mem action '{parsed.action_type}'."

        appended = self._append_tool_output_message(
            messages,
            tool_call=tool_call,
            tool_name=tool_name,
            output=budget.format_feedback(feedback),
        )
        budget.record_appended_item(appended)

    @staticmethod
    def _render_turn_for_logging(assistant_text: str, tool_calls: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        if assistant_text and assistant_text.strip():
            parts.append(assistant_text.strip())
        if tool_calls:
            parts.append("TOOL_CALLS:\n" + json.dumps(tool_calls[:3], indent=2, sort_keys=True))
        return "\n\n".join(parts).strip()

    @staticmethod
    def _forecasts_within_probability_bounds(forecasts: List[Dict[str, Any]]) -> bool:
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

    def _log_qwen_action(
        self,
        *,
        forecast_interface,
        messages: List[Dict[str, Any]],
        rendered_response: str,
        phase: str,
        budget: Optional[BudgetTracker] = None,
        qid: Optional[str] = None,
        prompt_override: Any = None,
        **extra,
    ) -> None:
        input_delta = prompt_override
        if input_delta is None:
            for m in reversed(messages):
                if m.get("role") == "user":
                    input_delta = m
                    break
        metadata = {"phase": phase, "qid": qid, **extra}
        if budget is not None:
            metadata.update(budget.metadata())
        forecast_interface.log_model_output(input_delta, rendered_response, metadata)


class QwenAllQAgent(AllQAgent, QwenBasicAgent):
    """
    AllQ warmup + subsequent daily loop, but actions are obtained via Qwen-native
    Chat Completions tool calling.
    """

    def _build_warmup_system_prompt(self, current_date: date, q, forecast_interface=None) -> str:
        # Reuse AllQ warmup prompt structure and only replace response format.
        base_prompt = AllQAgent._build_warmup_system_prompt(
            self,
            current_date,
            q,
            forecast_interface=forecast_interface,
        )
        return self._rewrite_allq_warmup_prompt_for_qwen(base_prompt)

    @staticmethod
    def _build_qwen_warmup_mem_prompt(qid: str, question_title: str) -> str:
        return f"""You just finished forecasting question {qid}: "{question_title}".

Call exactly one tool now: `mem_add`.
Do not answer with plain text.

Requirements:
- qid must be exactly "{qid}"
- question should be "{question_title}"
- memory should store only question-specific reasoning, evidence, calibration, and update triggers
- do not store cross-question lessons or cleanup notes
- keep memory concise and specific"""

    @staticmethod
    def _build_qwen_warmup_structured_prompt(qid: str, question_title: str) -> str:
        return f"""You just finished forecasting question {qid}: "{question_title}".

Call exactly one tool now: `memory_new`.
Do not answer with plain text.

Requirements:
- store only question-specific reasoning, evidence, calibration, and update triggers
- do not store cross-question lessons
- include question id {qid} in the entry name or description
- use a short lowercase hyphenated name"""

    def _request_warmup_mem(
        self,
        messages: List[Dict],
        qid,
        question_title: str,
        submitted_forecasts: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict]:
        prompt = self._build_qwen_warmup_mem_prompt(qid, question_title)
        mem_messages = messages + [{"role": "user", "content": prompt}]
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
            except Exception as e:
                if attempt < self.WARMUP_MEMORY_MAX_RETRIES:
                    print(f"[{self.agent_id}] Warmup mem finalize retry {attempt+1} for qid {qid} after error: {e}")
                    continue
                print(f"[{self.agent_id}] Warmup mem finalize failed for qid {qid}: {e}")
                break

            usage = self._normalize_chat_usage(resp_json)
            self._timer.record_tokens(usage)
            self._timer.record_cost(usage.get("cost", 0), "llm")
            parsed, _, _ = chat_response_to_action(resp_json)
            if parsed is not None and parsed.action_type == "mem_add" and parsed.mem_data:
                data = dict(parsed.mem_data)
                data_qid = str(data.get("qid", qid))
                if data_qid != str(qid):
                    if attempt < self.WARMUP_MEMORY_MAX_RETRIES:
                        print(f"[{self.agent_id}] Warmup mem finalize retry {attempt+1} for qid {qid}: got qid={data_qid!r}")
                        continue
                    print(f"[{self.agent_id}] Warmup mem finalize invalid for qid {qid}: got qid={data_qid!r}")
                    break
                data["qid"] = data_qid
                data["question"] = str(data.get("question", question_title) or question_title)
                data["memory"] = str(data.get("memory", ""))[:1000]
                data["category"] = data.get("category")
                return data

            if attempt < self.WARMUP_MEMORY_MAX_RETRIES:
                print(f"[{self.agent_id}] Warmup mem finalize retry {attempt+1} for qid {qid}: action={getattr(parsed, 'action_type', None)}")
                continue
            print(f"[{self.agent_id}] Warmup mem finalize invalid for qid {qid}: action={getattr(parsed, 'action_type', None)}")

        print(f"[{self.agent_id}] Warmup mem finalize falling back to XML parser for qid {qid}")
        return super()._request_warmup_mem(messages, qid, question_title, submitted_forecasts)

    def _request_warmup_structured_memory(
        self,
        messages: List[Dict],
        qid,
        question_title: str,
        submitted_forecasts: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict]:
        prompt = self._build_qwen_warmup_structured_prompt(qid, question_title)
        mem_messages = messages + [{"role": "user", "content": prompt}]
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
            except Exception as e:
                if attempt < self.WARMUP_MEMORY_MAX_RETRIES:
                    print(f"[{self.agent_id}] Warmup structured memory retry {attempt+1} for qid {qid} after error: {e}")
                    continue
                print(f"[{self.agent_id}] Warmup structured memory finalize failed for qid {qid}: {e}")
                break

            usage = self._normalize_chat_usage(resp_json)
            self._timer.record_tokens(usage)
            self._timer.record_cost(usage.get("cost", 0), "llm")
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

        print(f"[{self.agent_id}] Warmup structured memory falling back to XML parser for qid {qid}")
        return super()._request_warmup_structured_memory(
            messages,
            qid,
            question_title,
            submitted_forecasts,
        )

    def _run_warmup_loop(self, messages: List[Dict], forecast_interface, target_qid: str) -> WarmupLoopResult:
        # Delegate warmup action handling to the shared Qwen tool-calling loop.
        forecasts, context_limit_hit = self._run_qwen_action_loop(
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

    def _rewrite_allq_warmup_prompt_for_qwen(self, base_prompt: str) -> str:
        prefix = base_prompt
        if "## RESPONSE FORMAT" in base_prompt:
            prefix = base_prompt.split("## RESPONSE FORMAT", 1)[0].rstrip()
        start_budget = self._build_start_budget_status(warmup=True)
        start_block = f"\nBudget at start:\n{start_budget}" if start_budget else ""
        return (
            f"{prefix}\n\n"
            "## RESPONSE FORMAT\n"
            "Use function tools as defined in the tool schema, following the signatures, "
            "constraints, and examples there. Call exactly one tool per turn.\n"
            f"{start_block}\n"
        )


@dataclass(frozen=True)
class QwenNewsSearchStepResult:
    """search_news tool step: user-facing feedback + env metrics (SkyRL warmup uses phase/hit)."""

    feedback: str
    phase: str
    successful_hit: bool


def qwen_optional_search_dates_from_parsed(parsed: ParsedAction) -> Tuple[Optional[date], Optional[date]]:
    """Parse YYYY-MM-DD bounds from ParsedAction (same logic as former _qwen_handle_search)."""
    min_date: Optional[date] = None
    max_date: Optional[date] = None
    if parsed.search_from:
        try:
            from datetime import datetime as _dt

            min_date = _dt.strptime(parsed.search_from, "%Y-%m-%d").date()
        except Exception:
            min_date = None
    if parsed.search_to:
        try:
            from datetime import datetime as _dt

            max_date = _dt.strptime(parsed.search_to, "%Y-%m-%d").date()
        except Exception:
            max_date = None
    return min_date, max_date


def qwen_execute_news_search(
    parsed: ParsedAction,
    search_handler: SearchHandler,
    *,
    max_results: int,
    search_type: str = "hybrid",
    min_date: Optional[date] = None,
    max_date: Optional[date] = None,
) -> QwenNewsSearchStepResult:
    """
    Core search_news behavior for QwenBasicAgent and SkyRL OpenForesightSearchWarmupEnv.

    Matches QwenBasicAgent._qwen_handle_search (default search_type hybrid when omitted there).
    """
    if not search_handler.is_available:
        return QwenNewsSearchStepResult(
            feedback="SEARCH ERROR: Search is not available.",
            phase="search_unavailable",
            successful_hit=False,
        )
    if not parsed.query:
        return QwenNewsSearchStepResult(
            feedback="SEARCH ERROR: Missing query.",
            phase="search_error",
            successful_hit=False,
        )
    result, error = search_handler.search(
        parsed.query,
        max_results=max_results,
        search_type=search_type,
        min_date=min_date,
        max_date=max_date,
    )
    if error:
        return QwenNewsSearchStepResult(
            feedback=f"SEARCH ERROR: {error}",
            phase="search_error",
            successful_hit=False,
        )
    successful_hit = bool(result and not result.startswith("No articles found"))
    return QwenNewsSearchStepResult(
        feedback=f"SEARCH RESULTS:\n{result}",
        phase="search",
        successful_hit=successful_hit,
    )


def qwen_parse_warmup_submit_outcomes(
    parsed: ParsedAction,
    target_qid: str,
) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    """
    Select one forecast for target_qid like QwenBasicAgent._qwen_handle_submit.

    Validates with QwenBasicAgent._forecasts_within_probability_bounds (same gate as the
    eval loop’s force-submit path). Error strings match _qwen_handle_submit when applicable.
    """
    forecasts = list(parsed.forecasts or [])
    forecasts = [f for f in forecasts if f.get("qid") == target_qid]
    if len(forecasts) > 1:
        forecasts = forecasts[:1]

    if not forecasts:
        return None, (parsed.error or "No valid forecasts")

    outcomes_raw = forecasts[0].get("outcomes")
    if not isinstance(outcomes_raw, dict):
        return None, "No valid forecasts"
    try:
        outcomes = {k: float(v) for k, v in outcomes_raw.items()}
    except Exception:
        return None, "No valid forecasts"

    qid = forecasts[0].get("qid", target_qid)
    if not QwenBasicAgent._forecasts_within_probability_bounds([{"qid": qid, "outcomes": outcomes}]):
        return None, "Forecast submission failed."

    return outcomes, None
