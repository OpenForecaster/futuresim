"""Chat Completions protocol helpers for BasicAgent."""

import json
from typing import Any, Dict, List, Optional

from futuresim_agents.utils.budget import BudgetTracker


class BasicChatProtocol:
    def _should_use_chat_tools(self) -> bool:
        """Return True when the provider can serve raw Chat Completions tool calls."""
        if not hasattr(self.inference, "chat_json"):
            return False
        if hasattr(self.inference, "enable_tools") and not bool(getattr(self.inference, "enable_tools", False)):
            return False
        return True

    def _require_chat_tools(self) -> None:
        if self._should_use_chat_tools():
            return

        provider_name = type(self.inference).__name__
        model_name = self.model_name or getattr(self.inference, "model_name", "") or "<unknown>"
        if not hasattr(self.inference, "chat_json"):
            raise RuntimeError(
                f"BasicAgent now requires a provider with chat_json() for OpenAI-style tools. "
                f"Current provider: {provider_name}."
            )
        if hasattr(self.inference, "enable_tools") and not bool(getattr(self.inference, "enable_tools", False)):
            raise RuntimeError(
                "BasicAgent now requires chat tools. This vLLM run was started without "
                "`vllm_enable_tools=true`."
            )
        raise RuntimeError(
            f"BasicAgent could not enable chat tools for provider={provider_name}, model={model_name}."
        )

    def _call_chat_json(
        self,
        *,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        sampling_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        self._require_chat_tools()
        sp = dict(sampling_params or {})
        sp["tools"] = tools
        sp["tool_choice"] = sp.get("tool_choice", "auto")
        sp["parallel_tool_calls"] = sp.get("parallel_tool_calls", self.config.parallel_tool_calls)
        return self.inference.chat_json(messages, sp)

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
        # DeepSeek reasoning models (e.g. deepseek-v4-pro) reject follow-up
        # requests unless the prior reasoning_content is echoed back.
        reasoning = assistant_message.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            out["reasoning"] = reasoning
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

        message = {"role": "user", "content": f"{tool_name} output:\n{output}"}
        messages.append(message)
        return message

    def _append_feedback_message(
        self,
        messages: List[Dict[str, Any]],
        budget: BudgetTracker,
        feedback: str,
        *,
        tool_call: Optional[Dict[str, Any]] = None,
        tool_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        output = budget.format_feedback(feedback)
        if tool_call is not None and tool_name:
            appended = self._append_tool_output_message(
                messages,
                tool_call=tool_call,
                tool_name=tool_name,
                output=output,
            )
            budget.record_appended_item(appended)
            return appended

        message = {"role": "user", "content": output}
        self._append_with_budget(messages, budget, message)
        return message

    @staticmethod
    def _render_turn_for_logging(assistant_text: str, tool_calls: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        if assistant_text and assistant_text.strip():
            parts.append(assistant_text.strip())
        if tool_calls:
            parts.append("TOOL_CALLS:\n" + json.dumps(tool_calls, indent=2, sort_keys=True))
        return "\n\n".join(parts).strip()

    def _log_chat_tools_action(
        self,
        *,
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
            for message in reversed(messages):
                if message.get("role") != "assistant":
                    input_delta = message
                    break
        metadata = {"phase": phase, "qid": qid, **extra}
        if budget is not None:
            metadata.update(budget.metadata())
        self._log_model_output(input_delta, rendered_response, metadata)

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
        # VLLMInference already retries transport failures internally.
        return self._call_chat_json(messages=messages, tools=tools, sampling_params=sampling_params)
