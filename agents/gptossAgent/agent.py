from __future__ import annotations

import json
import hashlib
import random
import time
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from agents.basicAgent.agent import BasicAgent
from agents.allQAgent.agent import AllQAgent
from agents.utils.forecast_parser import ParsedAction
from environment.interfaces import PredictionSubmission

from .tools import (
    build_action_tools,
    build_memory_tools,
    response_to_action,
    extract_function_calls,
    extract_assistant_messages,
    extract_replay_items_for_tool_turn,
    extract_reasoning_text,
    extract_reasoning_token_count,
)


def _to_responses_input(conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert local Harmony-like conversation items into Responses API input.

    Supports:
    - message-like items: {"role", "content", ...}
    - preformatted Responses items: {"type": ...}
    """
    out: List[Dict[str, Any]] = []
    for m in conversation or []:
        if not isinstance(m, dict):
            continue
        # Pass through preformatted Responses items (e.g. function_call_output).
        if isinstance(m.get("type"), str):
            out.append(m)
            continue

        role = (m.get("role") or "user").strip()
        content = m.get("content", "")
        blocks: List[Dict[str, str]] = []
        if content is not None and str(content) != "":
            blocks.append({"type": "input_text", "text": str(content)})
        item: Dict[str, Any] = {"role": role, "content": blocks}

        # Some vLLM /v1/responses builds reject assistant headers containing
        # function recipients ("to=functions.*"). Treat those as malformed
        # echoes and drop them from replay context.
        recipient = m.get("recipient")
        recipient_str = recipient.strip() if isinstance(recipient, str) else None
        if (
            role == "assistant"
            and isinstance(recipient_str, str)
            and recipient_str.startswith("functions.")
        ):
            continue

        name = m.get("name")
        if isinstance(name, str) and name.strip():
            item["name"] = name.strip()
        channel = m.get("channel")
        if isinstance(channel, str) and channel.strip():
            item["channel"] = channel.strip()
        if isinstance(recipient_str, str) and recipient_str:
            item["recipient"] = recipient_str
        content_type = m.get("content_type")
        if isinstance(content_type, str) and content_type.strip():
            item["content_type"] = content_type.strip()
        out.append(item)
    return out


class GPTOSSBasicAgent(BasicAgent):
    """
    GPTOSSBasicAgent: BasicAgent-compatible behavior, but requests actions via Harmony
    tool calling on the OpenAI Responses API (vLLM).

    Key design constraints:
    - Do not change environment/simulation code.
    - Keep the same config knobs (AgentConfig, sampling params, handlers).
    - Avoid putting legacy <action ...> XML into the model's context (it triggers
      strict Harmony parsing issues in some vLLM versions).
    """

    def _get_gptoss_prompt_mode(self) -> str:
        mode = str(getattr(self.config, "gptoss_prompt_mode", "instructions") or "instructions").strip().lower()
        if mode not in ("instructions", "first_user"):
            return "instructions"
        return mode

    def _seed_harmony_conversation(
        self,
        instructions: str,
        current_date: date,
        *,
        actions_remaining: int,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Build initial Responses payload in either baseline mode (`instructions`) or
        A/B mode (`first_user`).
        """
        del current_date  # current date is already included in developer instructions
        begin = f"Begin. Actions remaining {int(actions_remaining)}"
        mode = self._get_gptoss_prompt_mode()
        if mode == "first_user":
            first_user = (instructions or "").strip()
            if first_user:
                first_user = f"{first_user}\n\n{begin}"
            else:
                first_user = begin
            return "", [{"role": "user", "content": first_user}]
        return instructions, [{"role": "user", "content": begin}]

    def act(self, doc_interface, forecast_interface, current_date: date) -> List[Dict[str, Any]]:
        self._timer.reset()
        self._timer.start_day()

        if self._memory is not None:
            self._memory.set_date(current_date)

        self._setup_day(forecast_interface, current_date)

        instructions = self._build_harmony_instructions(current_date)
        instructions_for_api, conversation = self._seed_harmony_conversation(
            instructions,
            current_date,
            actions_remaining=int(self.config.max_actions),
        )

        all_forecasts = self._run_harmony_action_loop(
            instructions=instructions_for_api,
            conversation=conversation,
            forecast_interface=forecast_interface,
        )

        if self._memory is not None:
            self._harmony_memory_update(
                instructions=instructions_for_api,
                conversation=conversation,
                forecast_interface=forecast_interface,
                current_date=current_date,
            )

        self._timer.end_day()
        if self.config.memory_dir:
            self._timer.save_day_stats(self.config.memory_dir, current_date)

        forecast_interface.next_day()
        return all_forecasts

    # =========================================================================
    # Harmony prompt
    # =========================================================================

    def _build_harmony_instructions(self, current_date: date) -> str:
        """
        Keep BasicAgent instructions intact (scoring, feedback, rules, data notes),
        but replace XML action format with Harmony tool-call guidance.
        """
        # Use class-specific prompt builder (e.g., AllQ reminder text) and only
        # rewrite the output/action format portion for Harmony tool calls.
        base_prompt = self._build_instructions(current_date)
        return self._rewrite_basic_prompt_for_harmony(base_prompt)

    def _rewrite_basic_prompt_for_harmony(self, base_prompt: str) -> str:
        prefix = base_prompt
        if "## RESPONSE FORMAT" in base_prompt:
            prefix = base_prompt.split("## RESPONSE FORMAT", 1)[0].rstrip()

        return (
            f"{prefix}\n\n"
            "## RESPONSE FORMAT\n"
            "Use function tools only (no XML tags).\n"
            "- Call exactly ONE tool per turn.\n"
            "- Do not output <reasoning> or <action> tags.\n"
            "- Tool signatures, constraints, and examples are defined in the tool schema.\n\n"
            "## INTERACTION FLOW\n"
            f"You have {self.config.max_actions} actions per day. Each query/search/submit consumes 1 action.\n"
            "When ready to move on, call `next_day()`.\n\n"
            "## SUBMISSION RULES\n"
            "- You may call `submit_forecasts` multiple times in a single day until actions run out.\n"
            "- One `submit_forecasts` call may include forecasts for multiple question IDs.\n"
            "- You may update earlier same-day forecasts by submitting again for the same qid.\n"
            "- In peer-scoring mode, broader accurate coverage improves total score.\n"
        )

    @staticmethod
    def _build_final_submit_tool_instruction(target_qid: str) -> str:
        """
        Strict final-turn instruction for per-question loops.
        """
        return (
            "FINAL ACTION (last chance): You have exactly 1 action remaining and MUST submit your best guess forecast now.\n"
            f"Target question ID: {target_qid}\n"
            "Call exactly one tool: submit_forecasts.\n"
            "Do NOT call query_df, search_news, or next_day.\n"
            "Submit only this question id with concrete outcomes and probabilities (sum <= 1.0)."
        )

    # =========================================================================
    # Harmony action loop
    # =========================================================================

    def _run_harmony_action_loop(
        self,
        *,
        instructions: str,
        conversation: List[Dict[str, Any]],
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
        prompt_mode = self._get_gptoss_prompt_mode()
        reasoning_effort = str(getattr(self.config, "gptoss_reasoning_effort", "medium") or "medium").strip().lower()
        include_reasoning = bool(getattr(self.config, "gptoss_include_reasoning", True))
        final_submit_prompt_injected = False
        final_submit_retry_used = False

        while actions_remaining > 0:
            # Last-action guardrail for per-question runs: force a submit attempt.
            if target_qid is not None and actions_remaining == 1 and not final_submit_prompt_injected:
                conversation.append(
                    {"role": "user", "content": self._build_final_submit_tool_instruction(target_qid)}
                )
                final_submit_prompt_injected = True

            # Hard-enforce final submit in per-question loops by exposing only the
            # submit tool. Keep tool_choice="auto" because vLLM Harmony rejects
            # non-auto tool_choice values.
            force_final_submit_turn = target_qid is not None and actions_remaining == 1
            effective_tools = tools
            if force_final_submit_turn:
                submit_only = [t for t in tools if t.get("name") == "submit_forecasts"]
                if submit_only:
                    effective_tools = submit_only

            model_input_payload = self._build_model_input_for_logging(instructions=instructions, conversation=conversation)
            try:
                with self._timer.track("llm"):
                    resp_json = self._call_responses_with_retries(
                        instructions=instructions,
                        conversation=conversation,
                        tools=effective_tools,
                        sampling_params=self.config.sampling_params,
                    )
            except Exception as e:
                if self._is_fatal_inference_failure(e):
                    raise RuntimeError(f"Fatal inference failure in action loop: {e}") from e
                if force_final_submit_turn:
                    if not final_submit_retry_used:
                        final_submit_retry_used = True
                        continue
                    # Final turn retry already used: end this question attempt without submission.
                    break
                actions_remaining -= 1
                err_msg = f"LLM ERROR after retries: {e}"
                self._log_harmony_action(
                    forecast_interface=forecast_interface,
                    conversation=conversation,
                    rendered_response=err_msg,
                    phase="llm_error",
                    actions_remaining=actions_remaining,
                    qid=target_qid,
                    prompt_mode=prompt_mode,
                    reasoning_effort=reasoning_effort,
                    include_reasoning=include_reasoning,
                    error=str(e),
                    prompt_override=model_input_payload,
                )
                conversation.append(
                    {
                        "role": "user",
                        "content": f"{err_msg}\n\nActions remaining: {actions_remaining}",
                    }
                )
                continue

            parsed, assistant_text, tool_calls = response_to_action(resp_json)
            reasoning_text = extract_reasoning_text(resp_json)
            assistant_messages = extract_assistant_messages(resp_json)
            replay_items = extract_replay_items_for_tool_turn(resp_json)
            reasoning_tokens_turn = extract_reasoning_token_count(resp_json)
            status = resp_json.get("status")
            incomplete_details = resp_json.get("incomplete_details") or {}
            incomplete_reason = incomplete_details.get("reason") if isinstance(incomplete_details, dict) else None

            # Normalize Responses-API usage fields to the timer's expected schema.
            raw_usage = resp_json.get("usage") or {}
            usage = dict(raw_usage) if isinstance(raw_usage, dict) else {}
            if "prompt_tokens" not in usage and "input_tokens" in usage:
                usage["prompt_tokens"] = usage.get("input_tokens", 0)
            if "completion_tokens" not in usage and "output_tokens" in usage:
                usage["completion_tokens"] = usage.get("output_tokens", 0)
            # Map Responses-style token detail fields to ChatCompletions-style keys
            # expected by AgentTimer/TokenStats.
            if "completion_tokens_details" not in usage and "output_tokens_details" in usage:
                usage["completion_tokens_details"] = usage.get("output_tokens_details")
            if "prompt_tokens_details" not in usage and "input_tokens_details" in usage:
                usage["prompt_tokens_details"] = usage.get("input_tokens_details")
            if "total_tokens" not in usage:
                usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
            if "reasoning_tokens" not in usage and reasoning_tokens_turn > 0:
                usage["reasoning_tokens"] = reasoning_tokens_turn
            self._timer.record_tokens(usage)

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
                # Final turn retry already used: end this question attempt without submission.
                break

            # Persist assistant side of the turn in Harmony-like structure.
            self._append_assistant_turns(
                conversation=conversation,
                assistant_messages=assistant_messages,
                assistant_text=assistant_text,
                replay_items=replay_items,
                tool_calls=tool_calls,
                reasoning_text=reasoning_text,
                replay_mode=str(getattr(self.config, "gptoss_replay_mode", "raw_recommended") or "raw_recommended"),
            )

            # Log a compact representation (don’t shove huge JSON into logs).
            rendered = self._render_turn_for_logging(assistant_text, tool_calls)
            self._log_harmony_action(
                forecast_interface=forecast_interface,
                conversation=conversation,
                rendered_response=rendered,
                phase="llm",
                actions_remaining=actions_remaining,
                qid=target_qid,
                prompt_mode=prompt_mode,
                reasoning_effort=reasoning_effort,
                include_reasoning=include_reasoning,
                reasoning=reasoning_text if reasoning_text else None,
                reasoning_tokens=reasoning_tokens_turn,
                status=status,
                incomplete_reason=incomplete_reason,
                usage=usage,
                prompt_override=model_input_payload,
            )

            if parsed is None:
                # Model didn't call a tool. Consume an action and push back a firm reminder.
                actions_remaining -= 1
                if status == "incomplete" and incomplete_reason == "max_output_tokens":
                    feedback = (
                        "Response was incomplete because max_output_tokens was reached before a tool call. "
                        "Call exactly one tool with concise arguments.\n\n"
                        f"Actions remaining: {actions_remaining}"
                    )
                else:
                    feedback = (
                        "No tool call detected. You MUST call exactly one tool each turn.\n\n"
                        f"Actions remaining: {actions_remaining}"
                    )
                conversation.append({"role": "user", "content": feedback})
                continue

            if parsed.action_type == "next":
                break

            if parsed.action_type == "query":
                actions_remaining = self._harmony_handle_query(
                    conversation=conversation,
                    forecast_interface=forecast_interface,
                    parsed=parsed,
                    tool_call=tool_calls[0] if tool_calls else None,
                    actions_remaining=actions_remaining,
                    qid=target_qid,
                )
                continue

            if parsed.action_type == "search":
                actions_remaining = self._harmony_handle_search(
                    conversation=conversation,
                    forecast_interface=forecast_interface,
                    parsed=parsed,
                    tool_call=tool_calls[0] if tool_calls else None,
                    actions_remaining=actions_remaining,
                    qid=target_qid,
                )
                continue

            if parsed.action_type == "submit":
                actions_remaining, submitted = self._harmony_handle_submit(
                    conversation=conversation,
                    forecast_interface=forecast_interface,
                    parsed=parsed,
                    tool_call=tool_calls[0] if tool_calls else None,
                    actions_remaining=actions_remaining,
                    qid=target_qid,
                    target_qid=target_qid,
                )
                all_forecasts.extend(submitted)
                # In AllQ warmup, a submit should end the interaction for that question.
                if target_qid is not None and submitted:
                    break
                continue

            # Invalid / unknown
            actions_remaining -= 1
            err = parsed.error or "Unknown action/tool."
            conversation.append(
                {
                    "role": "user",
                    "content": f"Invalid action: {err}\n\nActions remaining: {actions_remaining}",
                }
            )

        return all_forecasts

    def _call_responses(
        self,
        *,
        instructions: str,
        conversation: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        sampling_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        from inference.vllm import VLLMInference

        if not isinstance(self.inference, VLLMInference):
            raise TypeError("GPTOSSBasicAgent currently requires provider=vllm (VLLMInference).")

        sp = dict(sampling_params or {})
        min_output = int(getattr(self.config, "gptoss_min_max_output_tokens", 25000) or 0)
        if min_output > 0:
            current_max = int(sp.get("max_tokens", 0) or 0)
            if current_max < min_output:
                sp["max_tokens"] = min_output
        sp["tools"] = tools
        sp["tool_choice"] = "auto"

        overrides: Dict[str, Any] = {
            # Keep the model from trying to emit many tool calls at once.
            "parallel_tool_calls": False,
            "max_tool_calls": 1,
        }
        effort = str(getattr(self.config, "gptoss_reasoning_effort", "medium") or "medium").strip().lower()
        if effort in ("low", "medium", "high"):
            # /v1/responses expects reasoning controls under `reasoning`.
            overrides["reasoning"] = {"effort": effort}

        return self.inference.responses_json(
            instructions=instructions,
            input_messages=_to_responses_input(conversation),
            sampling_params=sp,
            request_overrides=overrides,
        )

    @staticmethod
    def _is_fatal_inference_failure(error: Exception) -> bool:
        """Detect inference failures that should abort immediately."""
        msg = str(error).lower()
        fatal_markers = (
            "vllm server died on port",
            "vllm server failed to start",
            "vllm server process died immediately",
            "engine core initialization failed",
            "cuda out of memory occurred when warming up sampler",
        )
        return any(marker in msg for marker in fatal_markers)

    def _call_responses_with_retries(
        self,
        *,
        instructions: str,
        conversation: List[Dict[str, Any]],
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
                return self._call_responses(
                    instructions=instructions,
                    conversation=conversation,
                    tools=tools,
                    sampling_params=sampling_params,
                )
            except Exception as e:
                last_error = e
                if self._is_fatal_inference_failure(e):
                    raise
                if attempt >= max_retries:
                    break
                delay = min(max_backoff, base_backoff * (2 ** attempt))
                # Add light jitter to prevent synchronized retries.
                delay += delay * 0.2 * random.random()
                print(
                    f"[{self.agent_id}] /v1/responses error (attempt {attempt + 1}/{attempts}): {e}. "
                    f"Retrying in {delay:.1f}s...",
                    flush=True,
                )
                time.sleep(delay)

        assert last_error is not None
        raise last_error

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

    def _log_harmony_action(
        self,
        *,
        forecast_interface,
        conversation: List[Dict[str, Any]],
        rendered_response: str,
        phase: str,
        actions_remaining: int,
        qid: Optional[str] = None,
        prompt_override: Optional[str] = None,
        **extra,
    ) -> None:
        last_user = prompt_override if isinstance(prompt_override, str) else ""
        if not last_user:
            for m in reversed(conversation):
                if m.get("role") == "user":
                    last_user = m.get("content") or ""
                    break
        metadata = {"phase": phase, "actions_remaining": actions_remaining, "qid": qid, **extra}
        forecast_interface.log_model_output(last_user, rendered_response, metadata)

    @staticmethod
    def _build_model_input_for_logging(*, instructions: str, conversation: List[Dict[str, Any]]) -> str:
        """
        Render the exact Responses request-side context for this turn.
        """
        payload = {
            "instructions": instructions or "",
            "input": _to_responses_input(conversation),
        }
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _append_tool_output_message(
        conversation: List[Dict[str, Any]],
        *,
        tool_call: Optional[Dict[str, Any]],
        tool_name: str,
        output: str,
    ) -> None:
        if isinstance(tool_call, dict) and isinstance(tool_call.get("call_id"), str):
            conversation.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call["call_id"],
                    "output": output,
                }
            )
            return

        # Fallback for providers that do not expose call_id.
        conversation.append(
            {
                "role": "tool",
                "name": f"functions.{tool_name}",
                "channel": "commentary",
                "recipient": "assistant",
                "content": output,
            }
        )

    @staticmethod
    def _append_assistant_turns(
        *,
        conversation: List[Dict[str, Any]],
        assistant_messages: List[Dict[str, Any]],
        assistant_text: str,
        replay_items: List[Dict[str, Any]],
        tool_calls: List[Dict[str, Any]],
        reasoning_text: str,
        replay_mode: str,
    ) -> None:
        if tool_calls and replay_mode == "raw_recommended" and replay_items:
            conversation.extend(replay_items)
            return

        # If a final channel message exists, drop earlier analysis messages to
        # avoid replaying stale chain-of-thought into later turns.
        if assistant_messages:
            has_final = any((m.get("channel") == "final") for m in assistant_messages)
            if has_final:
                cleaned: List[Dict[str, Any]] = []
                seen_final = False
                for m in assistant_messages:
                    channel = m.get("channel")
                    if not seen_final and channel == "analysis":
                        continue
                    cleaned.append(m)
                    if channel == "final":
                        seen_final = True
                assistant_messages = cleaned

        has_analysis_message = any((m.get("channel") == "analysis") for m in assistant_messages)
        # Keep CoT for subsequent sampling when tools are being called.
        if tool_calls and reasoning_text and not has_analysis_message:
            conversation.append(
                {
                    "role": "assistant",
                    "channel": "analysis",
                    "content": reasoning_text,
                }
            )

        # Preserve explicit assistant messages from model output when available.
        if assistant_messages:
            for m in assistant_messages:
                content = m.get("content")
                if not isinstance(content, str) or not content.strip():
                    continue
                conversation.append(
                    {
                        "role": "assistant",
                        "channel": m.get("channel"),
                        "recipient": m.get("recipient"),
                        "content_type": m.get("content_type"),
                        "content": content,
                    }
                )
        elif assistant_text and assistant_text.strip():
            # Fallback: if we only got flattened output text, preserve it with a sensible channel.
            conversation.append(
                {
                    "role": "assistant",
                    "channel": "commentary" if tool_calls else "final",
                    "content": assistant_text.strip(),
                }
            )

    # =========================================================================
    # Tool handlers
    # =========================================================================

    def _harmony_handle_query(
        self,
        *,
        conversation: List[Dict[str, Any]],
        forecast_interface,
        parsed: ParsedAction,
        tool_call: Optional[Dict[str, Any]],
        actions_remaining: int,
        qid: Optional[str],
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
            conversation,
            tool_call=tool_call,
            tool_name="query_df",
            output=feedback,
        )
        return actions_remaining

    def _harmony_handle_search(
        self,
        *,
        conversation: List[Dict[str, Any]],
        forecast_interface,
        parsed: ParsedAction,
        tool_call: Optional[Dict[str, Any]],
        actions_remaining: int,
        qid: Optional[str],
    ) -> int:
        actions_remaining -= 1
        if not self._search_handler.is_available:
            feedback = f"SEARCH ERROR: Search is not available.\n\nActions remaining: {actions_remaining}"
            self._append_tool_output_message(
                conversation,
                tool_call=tool_call,
                tool_name="search_news",
                output=feedback,
            )
            return actions_remaining

        if not parsed.query:
            feedback = f"SEARCH ERROR: Missing query.\n\nActions remaining: {actions_remaining}"
            self._append_tool_output_message(
                conversation,
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
            conversation,
            tool_call=tool_call,
            tool_name="search_news",
            output=feedback,
        )
        return actions_remaining

    def _harmony_handle_submit(
        self,
        *,
        conversation: List[Dict[str, Any]],
        forecast_interface,
        parsed: ParsedAction,
        tool_call: Optional[Dict[str, Any]],
        actions_remaining: int,
        qid: Optional[str],
        target_qid: Optional[str],
    ) -> Tuple[int, List[Dict[str, Any]]]:
        actions_remaining -= 1
        submitted: List[Dict[str, Any]] = []

        forecasts = list(parsed.forecasts or [])
        if target_qid is not None:
            # Warmup: enforce target qid.
            forecasts = [f for f in forecasts if f.get("qid") == target_qid]

        if forecasts:
            for f in forecasts:
                try:
                    pred = PredictionSubmission(question_id=f["qid"], outcomes=f["outcomes"])
                    forecast_interface.submit_prediction(pred)
                    submitted.append(f)
                except Exception as e:
                    pass
            if submitted:
                # Ensure later same-day df queries reflect newly submitted predictions.
                self._query_handler.invalidate_cache()
            feedback = f"Submitted {len(submitted)} forecast(s). Actions remaining: {actions_remaining}"
        else:
            feedback = f"SUBMIT ERROR: {parsed.error or 'No valid forecasts'}\n\nActions remaining: {actions_remaining}"

        self._append_tool_output_message(
            conversation,
            tool_call=tool_call,
            tool_name="submit_forecasts",
            output=feedback,
        )
        return actions_remaining, submitted

    # =========================================================================
    # Memory update
    # =========================================================================

    def _harmony_memory_update(
        self,
        *,
        instructions: str,
        conversation: List[Dict[str, Any]],
        forecast_interface,
        current_date: date,
    ) -> None:
        if self._memory is None:
            return

        memory_prompt = f"""End of day {current_date}. You can now update your memory.

## MEMORY UPDATE
Your memory is the ONLY thing that carries over to tomorrow. Everything else resets. Tomorrow you get: search over news articles and access to the DataFrame (active question predictions and ground truths for resolved ones, but NOT your predictions for resolved questions — those are deleted on resolution).

Store things NOT recoverable from those tools:
1. Reasoning behind predictions and how you did on resolved questions that might help with unresolved questions — once a question resolves, both your prediction and reasoning are lost from the DataFrame. Example: "Q149: PSG 0.70 because Sky Bet implied 55% and Inter eliminated in semis."
2. Performance patterns — track your accuracy across resolved questions so you can calibrate. Example: "Bookmaker odds were correct 80% across 15 sports questions; I should weight them more."
3. Non-obvious insights that search alone would not surface. Example: "'First country to X' questions almost always resolve to a major economy."
4. Critical hard-to-find facts directly relevant to active questions. Example: "ECB next meeting June 5 — relevant to Q72, Q108."

Do NOT store: general forecasting advice (already in your instructions), easily searchable facts, prediction outcomes without reasoning, or vague tracking lists without reasoning.
Aim to keep memory under 2000 characters. Prioritize recent and high-impact items and drop stale entries about resolved questions you have already learned from.

To update memory, call this tool exactly once:
`update_memory(memory="Your updated memory content here (complete replacement, not a diff)")`

If you don't want to update memory, respond without calling any tool.
Current memory length: {len(self._memory)} characters"""
        conversation.append({"role": "user", "content": memory_prompt})
        model_input_payload = self._build_model_input_for_logging(instructions=instructions, conversation=conversation)

        tools = build_memory_tools()
        try:
            with self._timer.track("llm"):
                resp_json = self._call_responses_with_retries(
                    instructions=instructions,
                    conversation=conversation,
                    tools=tools,
                    sampling_params=self.config.sampling_params,
                )
        except Exception as e:
            self._log_harmony_action(
                forecast_interface=forecast_interface,
                conversation=conversation,
                rendered_response=f"MEMORY UPDATE ERROR after retries: {e}",
                phase="memory_update_error",
                actions_remaining=-1,
                current_memory_len=len(self._memory),
                error=str(e),
                prompt_override=model_input_payload,
            )
            print(f"[{self.agent_id}] Memory update skipped after retries: {e}", flush=True)
            return

        reasoning_text = extract_reasoning_text(resp_json)
        reasoning_tokens_turn = extract_reasoning_token_count(resp_json)
        replay_items = extract_replay_items_for_tool_turn(resp_json)
        status = resp_json.get("status")
        incomplete_details = resp_json.get("incomplete_details") or {}
        incomplete_reason = incomplete_details.get("reason") if isinstance(incomplete_details, dict) else None
        raw_usage = resp_json.get("usage") or {}
        usage = dict(raw_usage) if isinstance(raw_usage, dict) else {}
        if "prompt_tokens" not in usage and "input_tokens" in usage:
            usage["prompt_tokens"] = usage.get("input_tokens", 0)
        if "completion_tokens" not in usage and "output_tokens" in usage:
            usage["completion_tokens"] = usage.get("output_tokens", 0)
        if "completion_tokens_details" not in usage and "output_tokens_details" in usage:
            usage["completion_tokens_details"] = usage.get("output_tokens_details")
        if "prompt_tokens_details" not in usage and "input_tokens_details" in usage:
            usage["prompt_tokens_details"] = usage.get("input_tokens_details")
        if "total_tokens" not in usage:
            usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        if "reasoning_tokens" not in usage and reasoning_tokens_turn > 0:
            usage["reasoning_tokens"] = reasoning_tokens_turn
        self._timer.record_tokens(usage)

        tool_calls = extract_function_calls(resp_json)
        assistant_messages = extract_assistant_messages(resp_json)
        self._append_assistant_turns(
            conversation=conversation,
            assistant_messages=assistant_messages,
            assistant_text="",
            replay_items=replay_items,
            tool_calls=tool_calls,
            reasoning_text=reasoning_text,
            replay_mode=str(getattr(self.config, "gptoss_replay_mode", "raw_recommended") or "raw_recommended"),
        )
        rendered = self._render_turn_for_logging("", tool_calls)
        self._log_harmony_action(
            forecast_interface=forecast_interface,
            conversation=conversation,
            rendered_response=rendered,
            phase="memory_update",
            actions_remaining=-1,
            current_memory_len=len(self._memory),
            reasoning=reasoning_text if reasoning_text else None,
            reasoning_tokens=reasoning_tokens_turn,
            status=status,
            incomplete_reason=incomplete_reason,
            usage=usage,
            prompt_override=model_input_payload,
        )

        # Apply update if present.
        if tool_calls:
            for tc in tool_calls:
                if tc.get("name") != "update_memory":
                    continue
                args = tc.get("arguments") or {}
                mem = args.get("memory")
                if isinstance(mem, str) and mem.strip():
                    self._memory.update(mem)
                    self._append_tool_output_message(
                        conversation,
                        tool_call=tc,
                        tool_name="update_memory",
                        output=f"Memory updated ({len(mem)} characters).",
                    )
                    break


class GPTOSSAllQAgent(AllQAgent, GPTOSSBasicAgent):
    """
    GPTOSSAllQAgent: AllQ warmup + subsequent daily loop, but actions are obtained via
    Harmony tool calling (GPT-OSS) rather than XML parsing.
    """

    def warmup(self, forecast_interface, current_date: date) -> None:
        print(f"[{self.agent_id}] Starting WARMUP phase on {current_date}")

        self._timer.start_day()
        questions = list(forecast_interface.questions.values())
        self._setup_warmup_day(forecast_interface, current_date)
        forecast_interface.current_agent_id = self.agent_id

        # Warm server before high parallelism.
        from inference.vllm import VLLMInference
        if isinstance(self.inference, VLLMInference):
            # Fail fast if the agent vLLM server cannot start.
            self.inference.chat([{"role": "user", "content": "ping"}], {"temperature": 0.0, "max_tokens": 1})

        from concurrent.futures import ThreadPoolExecutor, as_completed

        max_workers = getattr(self.config, "warmup_parallelism", 20)
        print(f"[{self.agent_id}] Parallelizing warmup with {max_workers} threads...")

        def _run_one(q):
            instructions = self._build_harmony_warmup_instructions(current_date, q, forecast_interface=forecast_interface)
            instructions_for_api, convo = self._seed_harmony_conversation(
                instructions,
                current_date,
                actions_remaining=int(getattr(self.config, "warmup_max_actions", 10)),
            )
            self._run_harmony_action_loop(
                instructions=instructions_for_api,
                conversation=convo,
                forecast_interface=forecast_interface,
                target_qid=q.qid,
                enable_query=False,  # warmup has no df query handler setup
                max_actions=getattr(self.config, "warmup_max_actions", 10),
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futs = {executor.submit(_run_one, q): q.qid for q in questions}
            for i, fut in enumerate(as_completed(futs)):
                qid = futs[fut]
                try:
                    fut.result()
                    if (i + 1) % 10 == 0:
                        print(f"[{self.agent_id}] Warmup Progress: {i+1}/{len(questions)}")
                except Exception as e:
                    print(f"[{self.agent_id}] Error processing question {qid}: {e}")
                    if self._is_fatal_inference_failure(e):
                        print(
                            f"[{self.agent_id}] Fatal inference failure detected. Aborting warmup immediately.",
                            flush=True,
                        )
                        for pending in futs:
                            if not pending.done():
                                pending.cancel()
                        raise RuntimeError(
                            f"[{self.agent_id}] Warmup aborted due to fatal inference failure: {e}"
                        ) from e

        self.warmed_up = True
        self._timer.end_day()
        if self.config.memory_dir:
            self._timer.save_day_stats(self.config.memory_dir, current_date)
        self._timer.reset()
        print(f"[{self.agent_id}] Warmup complete.")

    def _build_harmony_warmup_instructions(self, current_date: date, q, forecast_interface=None) -> str:
        base_prompt = AllQAgent._build_warmup_system_prompt(self, current_date, q, forecast_interface=forecast_interface)
        return self._rewrite_allq_warmup_prompt_for_harmony(base_prompt, q.qid)

    def _rewrite_allq_warmup_prompt_for_harmony(self, base_prompt: str, target_qid: str) -> str:
        del target_qid
        prefix = base_prompt
        if "## RESPONSE FORMAT" in base_prompt:
            prefix = base_prompt.split("## RESPONSE FORMAT", 1)[0].rstrip()
        return (
            f"{prefix}\n\n"
            "## RESPONSE FORMAT\n"
            "Use function tools only (no XML tags).\n"
            "- Call exactly ONE tool per turn.\n"
            "- Tool signatures, constraints, and examples are defined in the tool schema.\n"
        )
