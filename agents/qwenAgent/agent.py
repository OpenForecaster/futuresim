from __future__ import annotations

import json
import random
import time
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from agents.allQAgent.agent import AllQAgent
from agents.basicAgent.agent import BasicAgent
from agents.gptossAgent.tools import extract_reasoning_token_count
from agents.utils.forecast_parser import ParsedAction
from environment.interfaces import PredictionSubmission

from .tools import (
    build_action_tools,
    chat_response_to_action,
    extract_assistant_message,
    extract_finish_reason,
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

        if self._memory is not None:
            self._memory.set_date(current_date)

        self._setup_day(forecast_interface, current_date)

        messages = [{"role": "user", "content": self._build_qwen_instructions(current_date)}]
        all_forecasts = self._run_qwen_action_loop(messages=messages, forecast_interface=forecast_interface)

        if self._memory is not None:
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

        return (
            f"{prefix}\n\n"
            "## RESPONSE FORMAT\n"
            "Use function tools as defined in the tool schema. Call exactly one tool per turn.\n\n"
            "## INTERACTION FLOW\n"
            f"You have {self.config.max_actions} actions per day. Each query/search/submit consumes 1 action.\n"
            "When ready to move on, call `next_day()`.\n\n"
            "## SUBMISSION RULES\n"
            "- You may call `submit_forecasts` multiple times in a single day until actions run out.\n"
            "- Each `submit_forecasts` call must include exactly one forecast for exactly one qid.\n"
            "- You may update earlier same-day forecasts by submitting again for the same qid.\n"
        )

    @staticmethod
    def _build_final_submit_tool_instruction(target_qid: str) -> str:
        return (
            "FINAL ACTION (last chance): You have exactly 1 action remaining and MUST submit your best guess forecast now.\n"
            f"Target question ID: {target_qid}\n"
            "Call exactly one tool: submit_forecasts.\n"
            "Do NOT call query_df, search_news, or next_day.\n"
            "Submit only this question id with concrete outcomes and probabilities (sum <= 1.0)."
        )

    def _run_qwen_action_loop(
        self,
        *,
        messages: List[Dict[str, Any]],
        forecast_interface,
        target_qid: Optional[str] = None,
        enable_query: bool = True,
        enable_search: Optional[bool] = None,
        max_actions: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        actions_remaining = int(max_actions if max_actions is not None else self.config.max_actions)
        all_forecasts: List[Dict[str, Any]] = []

        if enable_search is None:
            enable_search = bool(self._search_handler.is_available)
        tools = build_action_tools(
            enable_query=enable_query,
            enable_search=enable_search,
            max_outcomes_per_question=self.config.max_outcomes_per_question,
        )

        final_submit_prompt_injected = False
        final_submit_retry_used = False

        while actions_remaining > 0:
            if target_qid is not None and actions_remaining == 1 and not final_submit_prompt_injected:
                messages.append(
                    {
                        "role": "user",
                        "content": self._build_final_submit_tool_instruction(target_qid),
                    }
                )
                final_submit_prompt_injected = True

            force_final_submit_turn = target_qid is not None and actions_remaining == 1
            effective_tools = tools
            if force_final_submit_turn:
                submit_only = [t for t in tools if t.get("function", {}).get("name") == "submit_forecasts"]
                if submit_only:
                    effective_tools = submit_only

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

                if force_final_submit_turn and not final_submit_retry_used:
                    final_submit_retry_used = True
                    continue

                if force_final_submit_turn:
                    break

                actions_remaining -= 1
                err_msg = f"LLM ERROR after retries: {e}"
                self._log_qwen_action(
                    forecast_interface=forecast_interface,
                    messages=messages,
                    rendered_response=err_msg,
                    phase="llm_error",
                    actions_remaining=actions_remaining,
                    qid=target_qid,
                    error=str(e),
                )
                messages.append(
                    {
                        "role": "user",
                        "content": f"{err_msg}\n\nActions remaining: {actions_remaining}",
                    }
                )
                continue

            usage = self._normalize_chat_usage(resp_json)
            self._timer.record_tokens(usage)
            reasoning_tokens_turn = extract_reasoning_token_count({"usage": usage})

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

            self._append_assistant_message(messages=messages, assistant_message=assistant_message)

            rendered = self._render_turn_for_logging(assistant_text, tool_calls)
            self._log_qwen_action(
                forecast_interface=forecast_interface,
                messages=messages,
                rendered_response=rendered,
                phase="llm",
                actions_remaining=actions_remaining,
                qid=target_qid,
                finish_reason=finish_reason,
                usage=usage,
                reasoning_tokens=reasoning_tokens_turn,
            )

            if parsed is None:
                actions_remaining -= 1
                feedback = (
                    "No tool call detected. You MUST call exactly one tool each turn.\n\n"
                    f"Actions remaining: {actions_remaining}"
                )
                messages.append({"role": "user", "content": feedback})
                continue

            if parsed.action_type == "next":
                break

            if parsed.action_type == "query":
                actions_remaining = self._qwen_handle_query(
                    messages=messages,
                    parsed=parsed,
                    tool_call=tool_calls[0] if tool_calls else None,
                    actions_remaining=actions_remaining,
                )
                continue

            if parsed.action_type == "search":
                actions_remaining = self._qwen_handle_search(
                    messages=messages,
                    parsed=parsed,
                    tool_call=tool_calls[0] if tool_calls else None,
                    actions_remaining=actions_remaining,
                )
                continue

            if parsed.action_type == "submit":
                actions_remaining, submitted = self._qwen_handle_submit(
                    messages=messages,
                    forecast_interface=forecast_interface,
                    parsed=parsed,
                    tool_call=tool_calls[0] if tool_calls else None,
                    actions_remaining=actions_remaining,
                    target_qid=target_qid,
                )
                all_forecasts.extend(submitted)
                if target_qid is not None and submitted:
                    break
                continue

            actions_remaining -= 1
            err = parsed.error or "Unknown action/tool."
            messages.append(
                {
                    "role": "user",
                    "content": f"Invalid action: {err}\n\nActions remaining: {actions_remaining}",
                }
            )

        return all_forecasts

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
        sp["tool_choice"] = "auto"
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

    def _call_chat_json_with_retries(
        self,
        *,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        sampling_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        max_retries = max(0, int(getattr(self.config, "gptoss_responses_max_retries", 3)))
        base_backoff = max(0.0, float(getattr(self.config, "gptoss_retry_backoff_base_s", 1.0)))
        max_backoff = max(base_backoff, float(getattr(self.config, "gptoss_retry_backoff_max_s", 16.0)))
        attempts = max_retries + 1

        last_error: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                return self._call_chat_json(messages=messages, tools=tools, sampling_params=sampling_params)
            except Exception as e:
                last_error = e
                if self._is_fatal_inference_failure(e):
                    raise
                if attempt >= max_retries:
                    break
                delay = min(max_backoff, base_backoff * (2 ** attempt))
                delay += delay * 0.2 * random.random()
                print(
                    f"[{self.agent_id}] /v1/chat/completions error (attempt {attempt + 1}/{attempts}): {e}. "
                    f"Retrying in {delay:.1f}s...",
                    flush=True,
                )
                time.sleep(delay)

        assert last_error is not None
        raise last_error

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
    def _append_assistant_message(*, messages: List[Dict[str, Any]], assistant_message: Dict[str, Any]) -> None:
        if not isinstance(assistant_message, dict):
            return
        tool_calls = assistant_message.get("tool_calls")
        content = assistant_message.get("content")
        has_tool_calls = isinstance(tool_calls, list) and len(tool_calls) > 0
        has_text = isinstance(content, str) and bool(content.strip())

        if not has_tool_calls and not has_text:
            return

        out: Dict[str, Any] = {"role": "assistant"}
        if "content" in assistant_message:
            out["content"] = content if content is not None else ""
        if has_tool_calls:
            out["tool_calls"] = tool_calls
        messages.append(out)

    @staticmethod
    def _append_tool_output_message(
        messages: List[Dict[str, Any]],
        *,
        tool_call: Optional[Dict[str, Any]],
        tool_name: str,
        output: str,
    ) -> None:
        call_id = tool_call.get("call_id") if isinstance(tool_call, dict) else None
        if isinstance(call_id, str) and call_id:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name,
                    "content": output,
                }
            )
            return

        # Fallback for parser/provider variants that do not expose call id.
        messages.append({"role": "user", "content": f"{tool_name} output:\n{output}"})

    def _qwen_handle_query(
        self,
        *,
        messages: List[Dict[str, Any]],
        parsed: ParsedAction,
        tool_call: Optional[Dict[str, Any]],
        actions_remaining: int,
    ) -> int:
        actions_remaining -= 1
        if parsed.code:
            with self._timer.track("df_query"):
                result, error = self._query_handler.execute(parsed.code)
            if error:
                feedback = f"QUERY ERROR: {error}\n\nActions remaining: {actions_remaining}"
            else:
                feedback = f"QUERY RESULT:\n{result}\n\nActions remaining: {actions_remaining}"
        else:
            feedback = f"QUERY ERROR: {parsed.error or 'Missing code'}\n\nActions remaining: {actions_remaining}"

        self._append_tool_output_message(
            messages,
            tool_call=tool_call,
            tool_name="query_df",
            output=feedback,
        )
        return actions_remaining

    def _qwen_handle_search(
        self,
        *,
        messages: List[Dict[str, Any]],
        parsed: ParsedAction,
        tool_call: Optional[Dict[str, Any]],
        actions_remaining: int,
    ) -> int:
        actions_remaining -= 1
        if not self._search_handler.is_available:
            feedback = f"SEARCH ERROR: Search is not available.\n\nActions remaining: {actions_remaining}"
            self._append_tool_output_message(
                messages,
                tool_call=tool_call,
                tool_name="search_news",
                output=feedback,
            )
            return actions_remaining

        if not parsed.query:
            feedback = f"SEARCH ERROR: Missing query.\n\nActions remaining: {actions_remaining}"
            self._append_tool_output_message(
                messages,
                tool_call=tool_call,
                tool_name="search_news",
                output=feedback,
            )
            return actions_remaining

        min_date = None
        max_date = None
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

        with self._timer.track("search"):
            result, error = self._search_handler.search(
                parsed.query,
                max_results=self.config.max_search_results,
                min_date=min_date,
                max_date=max_date,
            )
        if error:
            feedback = f"SEARCH ERROR: {error}\n\nActions remaining: {actions_remaining}"
        else:
            feedback = f"SEARCH RESULTS:\n{result}\n\nActions remaining: {actions_remaining}"
        self._append_tool_output_message(
            messages,
            tool_call=tool_call,
            tool_name="search_news",
            output=feedback,
        )
        return actions_remaining

    def _qwen_handle_submit(
        self,
        *,
        messages: List[Dict[str, Any]],
        forecast_interface,
        parsed: ParsedAction,
        tool_call: Optional[Dict[str, Any]],
        actions_remaining: int,
        target_qid: Optional[str],
    ) -> Tuple[int, List[Dict[str, Any]]]:
        actions_remaining -= 1
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
                    f"Submitted forecast for qid={sub['qid']}{title_str}: {outcomes_str}. "
                    f"Actions remaining: {actions_remaining}"
                )
                if dropped_forecasts > 0:
                    feedback += (
                        f"\nIgnored {dropped_forecasts} extra forecast item(s); "
                        "submit exactly one qid per call."
                    )
            else:
                detail = "; ".join(submit_errors) if submit_errors else "Forecast submission failed."
                feedback = f"SUBMIT ERROR: {detail}\n\nActions remaining: {actions_remaining}"
        else:
            feedback = f"SUBMIT ERROR: {parsed.error or 'No valid forecasts'}\n\nActions remaining: {actions_remaining}"

        self._append_tool_output_message(
            messages,
            tool_call=tool_call,
            tool_name="submit_forecasts",
            output=feedback,
        )
        return actions_remaining, submitted

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
        actions_remaining: int,
        qid: Optional[str] = None,
        **extra,
    ) -> None:
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                content = m.get("content")
                if isinstance(content, str):
                    last_user = content
                else:
                    last_user = str(content)
                break
        metadata = {"phase": phase, "actions_remaining": actions_remaining, "qid": qid, **extra}
        forecast_interface.log_model_output(last_user, rendered_response, metadata)


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

    def _run_warmup_loop(self, messages: List[Dict], forecast_interface, target_qid: str) -> None:
        # Delegate warmup action handling to the shared Qwen tool-calling loop.
        self._run_qwen_action_loop(
            messages=messages,
            forecast_interface=forecast_interface,
            target_qid=target_qid,
            enable_query=False,
            max_actions=getattr(self.config, "warmup_max_actions", 10),
        )

    @staticmethod
    def _rewrite_allq_warmup_prompt_for_qwen(base_prompt: str) -> str:
        prefix = base_prompt
        if "## RESPONSE FORMAT" in base_prompt:
            prefix = base_prompt.split("## RESPONSE FORMAT", 1)[0].rstrip()
        return (
            f"{prefix}\n\n"
            "## RESPONSE FORMAT\n"
            "Use function tools as defined in the tool schema. Call exactly one tool per turn.\n"
        )
