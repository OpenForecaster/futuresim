"""
BasicAgent: Main agent class for LLM-based forecasting.

Uses Chat Completions tool-calling for all action and structured-memory loops.
"""

import json
import re
from datetime import date, timedelta
from typing import List, Dict, Any, Optional, Tuple

from agents.base import BaseAgent
from agents.utils.forecast_parser import ParsedAction
from agents.utils.budget import (
    BudgetSettings,
    BudgetTracker,
    build_budget_overview,
    build_start_budget_status,
    estimate_budget_tokens,
)
from agents.utils.timing import AgentTimer
from environment.interfaces import PredictionSubmission

from .config import AgentConfig
from .memory import BasicMemory
from .tools import (
    build_action_tools,
    build_memory_phase_tools,
    chat_response_to_action,
    execute_news_search,
    extract_assistant_message,
    extract_finish_reason,
    final_submit_tool_instruction_text,
    optional_search_dates_from_parsed,
    single_call_to_parsed_action,
    tool_calls_to_parsed_action as normalize_tool_calls_to_parsed_action,
)
from agents.utils.memory import StructuredMemory, ActiveMemory
from agents.utils.output_logger import AgentOutputLogger
from .query import QueryHandler
from .search import SearchHandler
from .feedback import FeedbackHandler


class BasicAgent(BaseAgent):
    """
    Basic forecasting agent using LLM inference.
    
    Interaction flow per day:
    1. Receives system prompt with DataFrame schema and scoring rules
    2. Can take query/search/submit/next actions, subject to configured loop budgets:
       - query: Execute Python code to explore the DataFrame
       - search: Search news articles for context (if enabled)
       - get_article: Retrieve full article content
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

    @staticmethod
    def tool_calls_to_parsed_action(
        tool_calls: Optional[List[Dict[str, Any]]],
        *,
        assistant_text: str = "",
    ) -> Tuple[Optional[ParsedAction], str, List[Dict[str, Any]]]:
        return normalize_tool_calls_to_parsed_action(
            tool_calls,
            assistant_text=assistant_text,
        )

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

    def _should_use_chat_tools(self) -> bool:
        """Return True when the provider can serve raw Chat Completions tool calls."""
        model_l = str(self.model_name or "").lower()
        if "gpt-oss" in model_l:
            return False
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
        if "gpt-oss" in str(model_name).lower():
            raise RuntimeError(
                "BasicAgent no longer supports GPT-OSS Harmony/Responses. "
                "Use the dedicated gptossbasic/gptossallq scaffold."
            )
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
        all_forecasts = self._run_action_loop(messages, forecast_interface, current_date)

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

    def _build_force_submit_preamble(self, budget: BudgetTracker) -> str:
        """Shared force-submit wording for action/token-constrained loops."""
        lines = [
            "FINAL ACTION: You MUST submit your best guess forecast now."
        ]
        status = budget.status_text()
        if status:
            lines.append(status)
        return "\n".join(lines)

    def _build_final_submit_tool_instruction(self, target_qid: str, budget: BudgetTracker) -> str:
        return final_submit_tool_instruction_text(target_qid, budget)

    def _search_results_description(self) -> str:
        chunk_tokens = self._search_handler.chunk_tokens
        extra_info = "The search tool uses a hybrid approach to retrieve articles, combining both semantic similarity (through an embedding model) and keyword matching."
        if chunk_tokens is None:
            return f"You have access to a search tool that returns up to {self.config.max_search_results} retrieved article chunks. {extra_info}"
        return (
            f"You have access to a search tool that returns up to {self.config.max_search_results} retrieved article chunks, "
            f"each roughly {chunk_tokens} tokens long. {extra_info}"
        )

    def _estimate_budget_tokens(self, payload: Any) -> int:
        return estimate_budget_tokens(payload, model_name=str(self.model_name or ""))

    @staticmethod
    def _append_with_budget(messages: List[Dict[str, Any]], budget: BudgetTracker, message: Dict[str, Any]) -> None:
        messages.append(message)
        budget.record_appended_item(message)

    def _get_timegap_days(self) -> int:
        return max(1, int(getattr(self.config, "timegap_days", 1) or 1))

    def _get_last_active_date(self, current_date: date) -> Optional[date]:
        fi = getattr(self, "_forecast_interface", None)
        last_active = getattr(fi, "last_active_date", None) if fi is not None else None
        if last_active:
            return last_active
        return None

    def _get_next_active_date(self, current_date: date) -> Optional[date]:
        fi = getattr(self, "_forecast_interface", None)
        if fi is not None and hasattr(fi, "next_active_date"):
            next_active = getattr(fi, "next_active_date")
            return next_active
        return current_date + timedelta(days=self._get_timegap_days())

    def _build_cadence_section(self, current_date: date) -> str:
        last_active = self._get_last_active_date(current_date)
        next_active = self._get_next_active_date(current_date)
        next_text = (
            f"Next scheduled update: {next_active}."
            if next_active
            else "No later updates are scheduled."
        )
        last_text = (
            f"Last update: {last_active}. "
            if last_active
            else "This is your first update. "
        )
        articles_text = ""
        if last_active:
            if self._search_handler.is_available:
                count = self._search_handler.count_articles(
                    min_date=last_active,
                    max_date=current_date - timedelta(days=self.config.search_cutoff_days),
                )
                if count is not None:
                    articles_text = f"{count:,} new articles have been published since your last update and you can access them using the search tool. "
            if not articles_text:
                articles_text = "New articles have been published since your last update and you can access them using the search tool. "
        return (
            "## UPDATE CADENCE\n"
            f"You have the chance to update your predictions every {self._get_timegap_days()} day(s). Your context is cleared after every session and your memory (along with past predictions) is the only information retained between sessions. {articles_text}"
            f"{last_text}Current date: {current_date}. {next_text}\n\n"
        )

    def _build_memory_carryover_note(self, current_date: date) -> str:
        next_active = self._get_next_active_date(current_date)
        next_text = (
            f"Next scheduled update: {next_active}."
            if next_active
            else "No later updates are scheduled."
        )
        return (
            "Your memory entries are the ONLY context retained between sessions. "
            f"You make updates every {self._get_timegap_days()} days. {next_text} "
            "In each session you get: search over news articles and the DataFrame "
            "(active question predictions, resolved question ground truths, your final "
            "predictions on resolved questions)."
        )
    
    def _format_and_cache_feedback(self, current_date: date) -> str:
        """Generate feedback, cache it for the memory prompt, and return formatted text."""
        feedback_data = self._feedback_handler.generate_feedback(
            self._forecast_interface, current_date, self.inference
        )
        self._last_feedback_data = feedback_data
        return self._feedback_handler.format_feedback(
            feedback_data,
            show_tw_peer=not self.config.single_agent_mode,
        )

    def _build_resolution_recap_for_memory(self) -> str:
        """Build a compact recap of this session's resolutions for the memory update prompt."""
        feedback = getattr(self, '_last_feedback_data', None)
        if not feedback:
            return "No questions resolved this session.\n"
        resolved = feedback.get('resolved_today', [])
        if not resolved:
            return "No questions resolved this session.\n"

        lines = ["## QUESTIONS RESOLVED THIS SESSION (extract lessons from these)"]
        for item in resolved:
            dist = FeedbackHandler._format_distribution(
                item.get('my_pred_distribution') or {}
            )
            lines.append(
                f"- Q{item['qid']}: \"{item['title']}\"\n"
                f"  Predicted: {dist} | Truth: {item['ground_truth']} | Brier: {item['brier']:+.2f}"
            )
        lines.append("")
        return "\n".join(lines)

    # =========================================================================
    # Memory Phase Helpers (unified loop)
    # =========================================================================

    def _build_memory_phase_prompt(self, current_date: date) -> str:
        """Build the memory-update prompt injected into the unified action loop."""
        if isinstance(self._memory, ActiveMemory):
            return self._build_active_memory_prompt(
                current_date, day_qids=getattr(self, '_day_qids', None)
            )
        if isinstance(self._memory, StructuredMemory):
            return self._build_structured_memory_prompt(current_date)
        return ""

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
        memory_tools: Optional[List[Dict[str, Any]]] = None
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
                if memory_tools is None:
                    memory_tools = build_memory_phase_tools(
                        enable_memory=has_structured_memory,
                        enable_mem_df=has_active_memory,
                    )
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
                if memory_tools is None:
                    memory_tools = build_memory_phase_tools(
                        enable_memory=has_structured_memory,
                        enable_mem_df=has_active_memory,
                    )
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
                    if memory_tools is None:
                        memory_tools = build_memory_phase_tools(
                            enable_memory=has_structured_memory,
                            enable_mem_df=has_active_memory,
                        )
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

    def _run_action_loop_with_chat_tools(
        self,
        messages: List[Dict[str, Any]],
        forecast_interface,
        current_date: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        forecasts, context_limit_hit = self._run_chat_tools_action_loop(
            messages=messages,
            forecast_interface=forecast_interface,
            current_date=current_date,
        )
        self._context_limit_hit = context_limit_hit
        return forecasts

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
    # Action Loop
    # =========================================================================

    def _run_action_loop(self, messages: List[Dict], forecast_interface, current_date: date = None) -> List[Dict]:
        """Main daily loop using OpenAI-style function tools."""
        return self._run_action_loop_with_chat_tools(messages, forecast_interface, current_date)
    
    # =========================================================================
    # Action Handlers
    # =========================================================================
    
    def _handle_query(
        self,
        messages,
        forecast_interface,
        response,
        parsed,
        budget: BudgetTracker,
        qid: str = None,
        reasoning=None,
        raw_stream: Optional[str] = None,
        tool_call: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Handle query action."""
        budget.consume_action()
        
        if parsed.code:
            extra_ctx = None
            if isinstance(self._memory, ActiveMemory):
                extra_ctx = {"mem_df": self._memory.get_mem_df()}
            with self._timer.track("df_query"):
                result, error = self._query_handler.execute(parsed.code, extra_context=extra_ctx)

            if error:
                feedback = f"QUERY ERROR: {error}"
            else:
                feedback = f"QUERY RESULT:\n{result}"
        else:
            feedback = f"ERROR: {parsed.error}"

        self._append_feedback_message(messages, budget, feedback, tool_call=tool_call, tool_name="query_df")
    
    def _handle_search(
        self,
        messages,
        forecast_interface,
        response,
        parsed,
        budget: BudgetTracker,
        qid: str = None,
        reasoning=None,
        raw_stream: Optional[str] = None,
        tool_call: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Handle search action. Returns full chunk content directly."""
        budget.consume_action()
        
        if not self._search_handler.is_available:
            feedback = "SEARCH ERROR: Search is not available."
        elif parsed.query:
            min_date, max_date = optional_search_dates_from_parsed(parsed)
            with self._timer.track("search"):
                effect = execute_news_search(
                    parsed,
                    self._search_handler,
                    max_results=self.config.max_search_results,
                    search_type="hybrid",
                    min_date=min_date,
                    max_date=max_date,
                )
            feedback = effect.feedback
        else:
            feedback = "SEARCH ERROR: No query provided."

        self._append_feedback_message(messages, budget, feedback, tool_call=tool_call, tool_name="search_news")

    def _handle_memory_action(
        self,
        messages,
        forecast_interface,
        response,
        parsed,
        budget: BudgetTracker,
        reasoning=None,
        tool_call: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Handle memory tool calls (retrieve/add/update/delete). Follows search handler pattern."""
        budget.consume_action()
        tool_name = {
            "memory_retrieve": "memory_retrieve",
            "memory_new": "memory_new",
            "memory_add": "memory_new",
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
        elif parsed.action_type in ("memory_new", "memory_add"):
            data = parsed.memory_new_data
            try:
                entry_name = self._memory.add_entry(data["name"], data["description"], data["content"])
                feedback = f"MEMORY: Added entry [{entry_name}]. Total entries: {self._memory.entry_count}/{self._memory._max_entries if hasattr(self._memory, '_max_entries') else '?'}."
            except ValueError as exc:
                feedback = f"MEMORY ERROR: {exc}"
        elif parsed.action_type == "memory_update":
            ok = self._memory.update_entry(parsed.memory_entry_name, **parsed.memory_update_data)
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

        self._append_feedback_message(messages, budget, feedback, tool_call=tool_call, tool_name=tool_name)

    def _handle_mem_action(
        self,
        messages,
        forecast_interface,
        response,
        parsed,
        budget: BudgetTracker,
        reasoning=None,
        tool_call: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Handle mem_df tool calls (mem_add/update/delete) for ActiveMemory."""
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

        self._append_feedback_message(messages, budget, feedback, tool_call=tool_call, tool_name=tool_name)

    def _handle_submit(
        self,
        messages,
        forecast_interface,
        response,
        parsed,
        budget: BudgetTracker,
        qid: str = None,
        reasoning=None,
        raw_stream: Optional[str] = None,
        tool_call: Optional[Dict[str, Any]] = None,
    ) -> List:
        """Handle submit action. Returns list of submitted forecasts."""
        submitted = []
        budget.consume_action()
        dropped_forecasts = 0
        
        # For logging: use provided qid, or infer from forecasts
        log_qid = qid
        if not log_qid and parsed.forecasts and len(parsed.forecasts) == 1:
            log_qid = parsed.forecasts[0]['qid']
        
        if parsed.forecasts:
            # Enforce single-qid submit: one <forecast ...> block per submit action.
            if len(parsed.forecasts) > 1:
                dropped_forecasts = len(parsed.forecasts) - 1
                parsed.forecasts = [parsed.forecasts[0]]

            for f in parsed.forecasts:
                try:
                    pred = PredictionSubmission(question_id=f['qid'], outcomes=f['outcomes'])
                    forecast_interface.submit_prediction(pred)
                    submitted.append(f)
                    outcomes_str = ", ".join(f"{k}: {v:.2f}" for k, v in f['outcomes'].items())
                    print(f"  [{self.agent_id}] Forecast {f['qid']}: {outcomes_str}")
                except Exception as e:
                    print(f"  [{self.agent_id}] Failed to submit {f['qid']}: {e}")

            if submitted:
                # Ensure later same-day df queries reflect newly submitted predictions.
                self._query_handler.invalidate_cache()
            
            # Include submitted qids in log metadata
            submitted_qids = [f['qid'] for f in submitted]
            if hasattr(self, '_day_qids'):
                self._day_qids.update(str(q) for q in submitted_qids)
            if submitted:
                sub = submitted[0]
                outcomes_str = ", ".join(f"{k}: {v:.2f}" for k, v in sub['outcomes'].items())
                title = self._query_handler.get_question_title(sub['qid'])
                title_str = f" ({title})" if title else ""
                feedback = f"Submitted forecast for qid={sub['qid']}{title_str}: {outcomes_str}."
                if dropped_forecasts > 0:
                    feedback += f"\nIgnored {dropped_forecasts} extra forecast block(s); submit exactly one qid per action."
            else:
                feedback = "SUBMIT ERROR: No valid forecast submitted."
        else:
            # Parse error - still consumed action
            feedback = f"SUBMIT ERROR: {parsed.error}"

        self._append_feedback_message(messages, budget, feedback, tool_call=tool_call, tool_name="submit_forecasts")
        return submitted
    
    def _handle_invalid(
        self,
        messages,
        forecast_interface,
        response,
        parsed,
        budget: BudgetTracker,
        qid: str = None,
        reasoning=None,
        raw_stream: Optional[str] = None,
        tool_call: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Handle invalid/unknown action."""
        budget.consume_action()

        error_msg = parsed.error or "Call exactly one function tool."
        feedback = f"No valid action found. {error_msg}"
        tool_name = tool_call.get("name") if isinstance(tool_call, dict) else None
        self._append_feedback_message(messages, budget, feedback, tool_call=tool_call, tool_name=tool_name)
    
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
    # Instructions & Memory
    # =========================================================================
    # Prompt Helpers
    # =========================================================================

    @staticmethod
    def _render_key_mechanics(
        mechanics: Dict[str, str],
        drop_keys: Optional[set[str]] = None,
    ) -> str:
        """Render numbered key mechanics, preserving insertion order."""
        drop = drop_keys or set()
        lines = [text for key, text in mechanics.items() if key not in drop]
        return "\n".join(f"{idx}. {line}" for idx, line in enumerate(lines, start=1))

    def _build_binary_brier_scoring_section(
        self,
        *,
        include_peer_summary: bool = True,
        drop_mechanics: Optional[set[str]] = None,
    ) -> str:
        """Build binary Brier scoring text with optional mechanic filtering."""
        is_multi_agent = not self.config.single_agent_mode
        show_peer = is_multi_agent and include_peer_summary

        peer_text = ""
        if show_peer:
            peer_text = """
- **Time-Weighted Peer Score (TW-Peer)**: `100 × (avg others' Brier - your Brier)`, summed over each day a prediction is held. A positive TW-Peer indicates predictions that were consistently more accurate than the group average."""

        mechanics: Dict[str, str] = {
            "accuracy_calibration": "**Accuracy + Calibration**: Assign probabilities that reflect true likelihood.",
            "binary_outcomes": "**Binary Outcomes**: Use exact outcomes \"Yes\" and \"No\".",
            "time_weighted": "**Time-Weighted Score**: For each question, your time-weighted score = sum(daily_score) / total_question_days where daily_score is the Brier Skill Score for that day (0 if you have no active prediction on that question) and total_question_days is the number of days the question was active. Each prediction's Brier Skill Score (1 minus sum of squared errors) is weighted by how many days it was active before you updated it. Predictions made earlier carry more weight since they cover more days, so act on your best information as soon as possible rather than waiting.",
            "question_count": "**Prediction-Count Incentive**: Scores are summed (not averaged) across all questions you predict on.",
        }
        if show_peer:
            mechanics["relative_performance"] = (
                "**Relative Performance (multi-agent)**: Final scoring is relative, "
                "so you have to outperform the market aggregate to gain positive peer score."
            )

        mechanics_text = self._render_key_mechanics(mechanics, drop_mechanics)
        return f"""## SCORING (Brier Score, Binary)
You are evaluated on **Brier Score** for binary Yes/No questions.
- Let p = your predicted probability for **Yes**.
- Let y = 1 if the resolved outcome is **Yes**, else 0.
- **Brier Score = (p - y)^2**.
- **Lower is better** (0 is perfect, 1 is worst).{peer_text}

Key Mechanics:
{mechanics_text}
"""

    def _build_brier_skill_scoring_section(
        self,
        *,
        include_peer_summary: bool = True,
        drop_mechanics: Optional[set[str]] = None,
    ) -> str:
        """Build Brier Skill scoring text with optional mechanic filtering."""
        is_multi_agent = not self.config.single_agent_mode
        show_peer = is_multi_agent and include_peer_summary

        section_title = "Time-Weighted Peer Score (Brier-Skill Based)" if show_peer else "Brier Skill Score"
        peer_text = ""
        if show_peer:
            peer_text = """
- **Time-Weighted Peer Score (TW-Peer)**: On each day a prediction is held, your Brier Skill Score is compared to the mean of all other agents' scores for the same question. These daily differences are summed over the lifetime of the prediction. A positive TW-Peer indicates predictions that were consistently more accurate than the group average."""

        mechanics: Dict[str, str] = {
            "accuracy_calibration": "**Accuracy + Calibration**: Try to guess the most likely outcome(s) and assign calibrated probabilities which reflect the likelihood of the outcome(s) occurring.",
            "time_weighted": "**Time-Weighted Score**: For each question, your time-weighted score = sum(daily_score) / total_question_days where daily_score is the Brier Skill Score for that day (0 if you have no active prediction on that question) and total_question_days is the number of days the question was active. Each prediction's Brier Skill Score (1 minus sum of squared errors) is weighted by how many days it was active before you updated it. Predictions made earlier carry more weight since they cover more days, so act on your best information as soon as possible rather than waiting.",
            "question_count": "**Prediction-Count Incentive**: Your score for each of the metrics like accuracy, brier skill score, time-weighted score is summed (NOT averaged) across all questions you predict on and higher score is better.",
            "max_outcomes": f"**Max Outcomes**: Submit at most {self.config.max_outcomes_per_question} outcomes per question.",
            "no_placeholders": "**No Placeholders**: \"Unknown\", \"TBD\", \"Other\" hurt your score. Be specific.",
        }
        if show_peer:
            mechanics["relative_performance"] = (
                "**Relative Performance (multi-agent)**: Final scoring is relative, "
                "so you have to outperform the market aggregate to gain positive peer score."
            )

        mechanics_text = self._render_key_mechanics(mechanics, drop_mechanics)
        return f"""## SCORING ({section_title})
You have to output a distribution of (outcome, probability) pairs for each question you make a forecast on.
You are evaluated on the **Brier Skill Score** = 1 - Σ(p_i - y_i)^2 summed over all outcomes (thus, ranging from -1 to +1), where:
- p_i = your probability for outcome i
- y_i = 1 if your outcome i is TRUE (actually occurred), 0 otherwise
- **Higher is better**: 1.0 = perfect, 0.0 = abstaining from guessing, negative = worse than abstaining.{peer_text}

Key Mechanics:
{mechanics_text}
"""

    def _get_scoring_section(self) -> str:
        """Get scoring description - single vs multi-agent mode."""
        source_name = getattr(getattr(self, "_forecast_interface", None), "source_name", "openforesight")
        if source_name == "metaculus_binary":
            return self._build_binary_brier_scoring_section()
        return self._build_brier_skill_scoring_section()
    
    def _get_data_notes(self) -> str:
        """Get notes about DataFrame columns - conditional on agent mode."""
        if self.config.single_agent_mode:
            return "Note: `my_prediction` column contains your current forecast as a dict (or None if not yet predicted)."
        else:
            return """Note: `market_aggregate` and `my_prediction` columns contain Python dicts (or None). You can access them directly, e.g. `row['market_aggregate']['outcome_name']`.
- `market_aggregate`: the mean probability distribution across all agents' latest predictions from the **previous day**. `None` on the first day (no predictions exist yet).
- `my_prediction`: your own latest forecast (or None if you haven't predicted this question yet).
- `num_predictions`: total number of prediction submissions made on this question across all agents and all days."""
    
    def _get_multiagent_context(self) -> str:
        """Multi-agent preamble describing the competitive setting."""
        if self.config.single_agent_mode:
            return ""
        n = getattr(self._forecast_interface, "num_agents", 0)
        if n < 2:
            return ""
        return f"""## MULTI-AGENT SETTING
You are competing against {n - 1} other forecasting agent{"s" if n > 2 else ""} on the same set of questions.
You each predict independently on every wakeup day. After each day, your predictions are averaged with the others' into a market aggregate (the `market_aggregate` column), which you can see starting the following day.
You are scored relative to your competitors: to earn a positive time-weighted peer score, your predictions need to be more accurate than the group average.
"""

    def _get_source_rules(self) -> str:
        """Get source-specific submission rules."""
        source_name = getattr(self._forecast_interface, 'source_name', 'openforesight')
        
        if source_name == "metaculus_binary":
            return """
## BINARY QUESTION RULES
All questions are Yes/No binary. Your `submit_forecasts` tool call MUST use exactly:
- **"Yes"** for the affirmative outcome
- **"No"** for the negative outcome

Example tool arguments:
{"forecasts":[{"qid":"12345","outcomes":{"Yes":0.7,"No":0.3}}]}
"""
        elif source_name == "metaculus_mcq":
            return """
## MULTIPLE CHOICE RULES
Each question has enumerated options shown in the 'options' column.
Your `submit_forecasts` tool call MUST use the EXACT option text from the question.
Do NOT paraphrase or abbreviate options.

Example tool arguments (if options are ["Candidate A", "Candidate B", "Candidate C"]):
{"forecasts":[{"qid":"12345","outcomes":{"Candidate A":0.5,"Candidate B":0.3,"Candidate C":0.2}}]}
"""
        return ""

    def _build_instructions(self, current_date: date) -> str:
        """Build the daily tool-calling prompt."""
        df_info = self._query_handler.get_info()
        budget_start_status = self._build_start_budget_status()
        budget_start_block = f"Budget at start:\n{budget_start_status}\n\n" if budget_start_status else ""
        cadence_section = self._build_cadence_section(current_date)

        memory_section = ""
        memory_flow_note = ""
        if self._memory is not None:
            memory_content = self._memory.get()
            if isinstance(self._memory, ActiveMemory):
                meta_index = self._memory.get_index()
                meta_block = f"Current meta-insights with their indices:\n{meta_index}\n\n" if meta_index else ""
                memory_section = f"""## YOUR MEMORY
{meta_block}`mem_df` holds your per-question notes (reasoning, evidence, calibration) — 1 row per question.
Columns: qid (str), question (str), last_updated (str), memory (str), category (str)
Both `mem_df` and `df` are available in the same `query_df` sandbox. You can join them on qid to find questions worth revisiting.

Inspect `mem_df` via `query_df`. Edit per-question notes with `mem_add`, `mem_update`, `mem_delete`. 
Manage meta-insights with `memory_retrieve` (using the indices), `memory_new`, `memory_update`, `memory_delete`. 
"""
            elif isinstance(self._memory, StructuredMemory):
                memory_index = self._memory.get_index()
                index_block = f"Current memory index:\n{memory_index}\n\n" if memory_index else ""
                memory_section = f"""## YOUR MEMORY ({self._memory.entry_count} entries, max {self._memory._max_entries})
{index_block}Entries should capture question-specific reasoning (include QIDs), post-resolution lessons,
or cross-question calibration patterns. Use the memory tools to retrieve, add, update, or delete entries.

"""
            elif memory_content:
                memory_section = f"""## YOUR MEMORY
{memory_content}

Use the reasoning and insights above to inform today's forecasts.

"""

            if isinstance(self._memory, (StructuredMemory, ActiveMemory)):
                memory_flow_note = (
                    "When you finish forecasting and are ready to move on, call `next_day()` to transition into the memory update phase. "
                    "The transition can also happen automatically if tokens run low."
                )
            else:
                memory_flow_note = "After ending this session, you will be prompted to update your memory."

        search_tool_line = ""
        search_advice = ""
        if self._search_handler.is_available:
            cutoff_desc = "today's date"
            if self.config.search_cutoff_days > 0:
                cutoff_date = current_date - timedelta(days=self.config.search_cutoff_days)
                cutoff_desc = f"{cutoff_date} (today - {self.config.search_cutoff_days} days)"
            search_tool_line = (
                "- `search_news(query, from_date?, to_date?)`: search the news corpus for evidence. "
                f"`to_date` is capped at {cutoff_desc}. {self._search_results_description()}\n"
            )
            search_advice = f"\nYou have access to a news article database which is updated **daily**. {self._search_results_description()}."

        memory_tools_section = ""
        if isinstance(self._memory, ActiveMemory):
            memory_tools_section = (
                "- `memory_retrieve` / `memory_new` / `memory_update` / `memory_delete`: manage meta-insight entries.\n"
                "- `mem_add` / `mem_update` / `mem_delete`: manage question-specific notes in `mem_df`.\n"
            )
        elif isinstance(self._memory, StructuredMemory):
            memory_tools_section = (
                "- `memory_retrieve` / `memory_new` / `memory_update` / `memory_delete`: manage reusable memory entries.\n"
            )

        return f"""You are a forecasting agent. Today is {current_date}. Your goal is to make accurate and calibrated predictions.

{self._format_and_cache_feedback(current_date)}

{getattr(self._forecast_interface, 'source_context', '')}

{self._get_source_rules()}

{self._get_multiagent_context()}
{cadence_section}{memory_section}{self._get_scoring_section()}
## AVAILABLE DATA
{search_advice}
You also have access to a pandas DataFrame `df` with {df_info['n_rows']} questions ({df_info['n_active']} active/unresolved, {df_info['n_resolved']} resolved).

Column descriptions of the DataFrame:
{df_info['columns_desc']}

{self._get_data_notes()}

## CODE EXECUTION ENVIRONMENT
You have access to a Python code execution environment where your code runs in a sandbox with these variables pre-defined:
- `df`: the DataFrame with the questions and your predictions
- `pd`: pandas module
- `today`: date object for {current_date}
- `date`, `datetime`, `timedelta`: from datetime module
{('- `mem_df`: your question-specific memory DataFrame (you can join with df on qid to decide what to revisit)' + chr(10)) if isinstance(self._memory, ActiveMemory) else ''}Standard builtins (len, str, int, float, min, max, sum, sorted, range, etc.) are available. A small safe subset of stdlib imports (for example datetime, json, math, re, ast) is allowed. External file, network, process, and private-attribute access is blocked; stay within in-memory DataFrame/pandas operations.

## RESPONSE FORMAT
Use the function tools from the tool schema. {'You may call multiple tools per turn.' if self.config.parallel_tool_calls else 'Call exactly one tool per turn.'} 

## TOOLS AVAILABLE FOR YOUR USE
- `query_df(code)`: inspect questions and your existing predictions. Use `print(...)` for outputs.
{search_tool_line}{memory_tools_section}- `submit_forecasts(forecasts)`: submit exactly one forecast for exactly one question ID (`qid`).
- `next_day()`: end the current session and proceed to the next one.

## INTERACTION FLOW
{self._build_budget_overview()}
You can interleave queries, searches, memory operations, and submissions as needed. Consider using `mem_df` early to recall prior reasoning and identify which questions need attention.

## SUBMISSION RULES
- qid must be from an active (`is_resolved=False`) question you identified from `df`
- Each `submit_forecasts` call must contain exactly one forecast for one question ID (`qid`).
- You may submit again later in the same session to update that `qid`.
- Maximum of {self.config.max_outcomes_per_question} outcomes allowed per question.
- Outcome names must be REAL predicted answers (e.g. person names, locations, dates, etc.)
- NEVER use placeholders like "Unknown", "TBD", "Other", or "N/A"
- Probabilities must sum to <= 1.0

{('Tip: After submitting a forecast, consider saving your reasoning and key evidence for that QID using mem_add/mem_update. At end of day, you will get another opportunity to update your memory.' + chr(10)) if isinstance(self._memory, (StructuredMemory, ActiveMemory)) else ''}

---
{budget_start_block}Begin."""
    
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

    def _build_structured_memory_prompt(self, current_date: date) -> str:
        """Build the structured-memory update prompt."""
        memory_index = self._memory.get_index()
        max_ent = self._memory._max_entries
        index_block = ""
        if memory_index:
            index_block = f"Current memory index:\n{memory_index}\n"
        resolution_recap = self._build_resolution_recap_for_memory()

        return f"""End of session {current_date}. Update your memory now.

## MEMORY UPDATE
{self._build_memory_carryover_note(current_date)}

{resolution_recap}You currently have {self._memory.entry_count} memory entries (max {max_ent}).
{index_block}

### STEP 1: Extract lessons from resolved questions
For each question resolved this session, create a lesson entry:
- Name it `q<QID>-lesson-<topic>` (e.g., `q247-lesson-ceremony-precedent`)
- Content: What you predicted, what actually happened, WHY you were wrong/right, and a transferable rule for similar future questions
- Delete the old forecast entry for that QID — replace it with the lesson
- If no questions resolved this session, skip to Step 2.

Example lesson entry:
  name: q247-lesson-ceremony-precedent
  description: Lesson from Q247 resolution — weight historical precedent for rituals
  content: Predicted St. Peter's Square 0.70, Lateran 0.10. Truth: St. Peter's Square. Brier +0.85. Lesson: For ceremonial events with strong historical patterns (all recent popes used same venue), assign 0.85+ to the precedent option.

### STEP 2: Update meta-patterns
If you see a pattern across 2+ resolved questions (or resolved + active), add or update a meta-pattern entry:
- Name it `meta-<pattern-topic>`
- Content: The pattern, supporting evidence (which QIDs), and how to apply it

### STEP 3: Update active question entries
For questions you researched or updated today, store your current reasoning and key evidence. Merge duplicate entries about the same question.

### STEP 4: Cleanup
Delete stale entries (resolved questions with no useful lesson, outdated facts). Descriptions should answer: "Why would future-me read this?" — not just "What did I do today."

### Rules
- Every entry must contain a reusable insight or reasoning chain — NOT a log of what you did
- Do NOT just create entries like `2025-05-14-updates-summary` that just list probability changes. You must include a reusable insight or reasoning chain.
- Do NOT just store general forecasting advice or easily searchable facts. You must include a reusable insight or reasoning chain.

## RESPONSE FORMAT
Use the function tools from the tool schema. {'You may call multiple tools per turn for efficiency.' if self.config.parallel_tool_calls else 'Call one tool per turn.'}
Use `memory_retrieve`, `memory_new`, `memory_update`, and `memory_delete` for memory work.
{chr(10) + 'To avoid read/write conflicts, batch your calls by type:' + chr(10) + '- First turn(s): batch all `memory_retrieve` calls to review entries you need' + chr(10) + '- Next turn(s): batch all write operations (`memory_new`, `memory_update`, `memory_delete`) based on what you read' + chr(10) if self.config.parallel_tool_calls else ''}
You do NOT need to exhaust your remaining token budget — call `next_day()` as soon as your essential updates are complete.

Start with Step 1. If questions resolved this session, create lesson entries first."""

    def _build_plain_memory_prompt(self, current_date: date) -> str:
        """Build the plain-memory update prompt (full replacement text)."""
        return f"""End of session {current_date}. You can now update your memory.

## MEMORY UPDATE
{self._build_memory_carryover_note(current_date)}

Store things NOT recoverable from those tools:
1. Reasoning behind predictions and how you did on resolved questions that might help with unresolved questions — once a question resolves, your prediction remains visible in the dataframe, but your reasoning is never stored in the dataframe. Example: "Q149: PSG 0.70 because Sky Bet implied 55% and Inter eliminated in semis."
2. Performance patterns — track your accuracy across resolved questions so you can calibrate. Example: "Bookmaker odds were correct 80% across 15 sports questions; I should weight them more."
3. Non-obvious insights that search alone would not surface. Example: "'First country to X' questions almost always resolve to a major economy."
4. Critical hard-to-find facts directly relevant to active questions. Example: "ECB next meeting June 5 — relevant to Q72, Q108."

Do NOT store: general forecasting advice (already in your instructions), easily searchable facts, prediction outcomes without reasoning, or vague tracking lists without reasoning.
Aim to keep memory under 2000 characters. Prioritize recent and high-impact items and drop stale entries about resolved questions you have already learned from.

Respond with exactly one of:
- The full updated memory text only, with no extra wrapper tags or commentary.
- `NO_MEMORY_UPDATE` if you want to keep the current memory unchanged.

Current memory length: {len(self._memory)} characters"""

    # =========================================================================
    # Active Memory (mem_df + reduced meta-insights)
    # =========================================================================

    def _build_active_memory_prompt(self, current_date: date, day_qids: set = None) -> str:
        """Build the active-memory update prompt."""
        mem_summary = self._memory.mem_summary(expanded_qids=day_qids)
        meta_index = self._memory.get_index()
        max_ent = self._memory._max_entries
        index_block = ""
        if meta_index:
            index_block = f"Current meta-insight index:\n{meta_index}\n"
        resolution_recap = self._build_resolution_recap_for_memory()

        return f"""End of session {current_date}. Update your memory now.

## MEMORY UPDATE
{self._build_memory_carryover_note(current_date)}

{resolution_recap}Your memory has two layers, both retained between sessions. Everything else resets.

## Layer 1: QUESTION-SPECIFIC NOTES (mem_df: {self._memory.mem_count} entries)

Current entries:
{mem_summary}

Per-question reasoning, evidence, and calibration notes. Max 1000 chars per entry.
Only store what is NOT recoverable from the DataFrame or search.

## Layer 2: META-INSIGHTS ({self._memory.entry_count}/{max_ent} entries)

Cross-question patterns and calibration notes. NOT for question-specific reasoning (use mem_df for that).
{index_block}

### STEP 1: Extract lessons from resolved questions
For each question resolved this session:
- Create a meta-insight lesson entry (not mem_df — lessons are cross-question):
  Name it `q<QID>-lesson-<topic>` (e.g., `q247-lesson-ceremony-precedent`)
  Content: What you predicted, what actually happened, WHY you were wrong/right, and a transferable rule for similar future questions
- Delete the old mem_df entry for that QID — it's stale now
- If no questions resolved this session, skip to Step 2.

Example lesson entry:
  name: q247-lesson-ceremony-precedent
  description: Lesson from Q247 resolution — weight historical precedent for rituals
  content: Predicted St. Peter's Square 0.70, Lateran 0.10. Truth: St. Peter's Square. Brier +0.85. Lesson: For ceremonial events with strong historical patterns (all recent popes used same venue), assign 0.85+ to the precedent option.

### STEP 2: Update mem_df for questions you interacted with today
For each question you researched or forecasted today, add or update a mem_df entry with your current reasoning and key evidence.
- Include: your prediction, key evidence, sources, calibration notes
- Max 1000 chars — be concise but specific

### STEP 3: Update meta-patterns
If you see a pattern across 2+ resolved questions (or resolved + active), add or update a meta-insight entry:
- Name it `meta-<pattern-topic>`
- Content: The pattern, supporting evidence (which QIDs), and how to apply it

### STEP 4: Cleanup
Delete stale entries from both layers (resolved questions with no useful lesson, outdated facts, duplicates).
Descriptions should answer: "Why would future-me read this?" — not just "What did I do today."

### Rules
- Every entry must contain a reusable insight or reasoning chain — NOT a log of what you did
- Do NOT just create entries like `2025-05-14-updates-summary` that just list probability changes
- Do NOT just store general forecasting advice or easily searchable facts

## RESPONSE FORMAT
Use the function tools from the tool schema. {'You may call multiple tools per turn for efficiency.' if self.config.parallel_tool_calls else 'Call one tool per turn.'}
Use `mem_add`, `mem_update`, and `mem_delete` for question-specific notes.
Use `memory_retrieve`, `memory_new`, `memory_update`, and `memory_delete` for meta-insights.
{chr(10) + 'To avoid read/write conflicts, batch your calls by type:' + chr(10) + '- First turn(s): batch all `memory_retrieve` calls to review entries you need' + chr(10) + '- Next turn(s): batch all write operations based on what you read' + chr(10) if self.config.parallel_tool_calls else ''}
You do NOT need to exhaust your remaining token budget — call `next_day()` as soon as your essential updates are complete.

Start with Step 1. If questions resolved this session, create lesson entries first."""
