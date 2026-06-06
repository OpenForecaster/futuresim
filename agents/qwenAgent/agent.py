from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from futuresim_agents.allQAgent.agent import AllQAgent
from futuresim_agents.basicAgent.agent import BasicAgent
from futuresim_agents.basicAgent.tools import (
    final_submit_tool_instruction_text,
)
from futuresim_agents.utils.budget import BudgetTracker
from futuresim_agents.utils.forecast_parser import ParsedAction


def qwen_final_submit_instruction_text(
    target_qid: str,
    budget: Optional[BudgetTracker],
    *,
    forbid_query_df: bool = True,
) -> str:
    return final_submit_tool_instruction_text(
        target_qid,
        budget,
        forbid_query_df=forbid_query_df,
    )


class QwenBasicAgent(BasicAgent):
    """Thin compatibility scaffold that inherits the base chat-tools implementation."""

    def _build_session_instructions(self, current_date: date) -> str:
        return self._build_qwen_instructions(current_date)

    def _build_qwen_instructions(self, current_date: date) -> str:
        return super()._build_session_instructions(current_date)

    def _rewrite_basic_prompt_for_qwen(self, base_prompt: str) -> str:
        return base_prompt

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
        return self._run_chat_tools_action_loop(
            messages=messages,
            forecast_interface=forecast_interface,
            target_qid=target_qid,
            enable_query=enable_query,
            enable_search=enable_search,
            warmup_budget=warmup_budget,
            max_actions=max_actions,
            current_date=current_date,
        )

    def _qwen_handle_query(
        self,
        *,
        messages: List[Dict[str, Any]],
        parsed: ParsedAction,
        tool_call: Optional[Dict[str, Any]],
        budget: BudgetTracker,
    ) -> None:
        self._handle_query(messages, None, "", parsed, budget, tool_call=tool_call)

    def _qwen_handle_search(
        self,
        *,
        messages: List[Dict[str, Any]],
        parsed: ParsedAction,
        tool_call: Optional[Dict[str, Any]],
        budget: BudgetTracker,
    ) -> None:
        self._handle_search(messages, None, "", parsed, budget, tool_call=tool_call)

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
        if target_qid is not None and parsed.forecasts:
            parsed.forecasts = [forecast for forecast in parsed.forecasts if forecast.get("qid") == target_qid]
        return self._handle_submit(
            messages,
            forecast_interface,
            "",
            parsed,
            budget,
            qid=target_qid,
            tool_call=tool_call,
        )

    def _qwen_handle_memory_action(
        self,
        *,
        messages: List[Dict[str, Any]],
        parsed: ParsedAction,
        tool_call: Optional[Dict[str, Any]],
        budget: BudgetTracker,
    ) -> None:
        self._handle_memory_action(messages, None, "", parsed, budget, tool_call=tool_call)

    def _qwen_handle_mem_action(
        self,
        *,
        messages: List[Dict[str, Any]],
        parsed: ParsedAction,
        tool_call: Optional[Dict[str, Any]],
        budget: BudgetTracker,
    ) -> None:
        self._handle_mem_action(messages, None, "", parsed, budget, tool_call=tool_call)

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
        self._log_chat_tools_action(
            messages=messages,
            rendered_response=rendered_response,
            phase=phase,
            budget=budget,
            qid=qid,
            prompt_override=prompt_override,
            **extra,
        )


class QwenAllQAgent(AllQAgent, QwenBasicAgent):
    """
    AllQ warmup + subsequent daily loop, but actions are obtained via Qwen-native
    Chat Completions tool calling.
    """

    def _rewrite_allq_warmup_prompt_for_qwen(self, base_prompt: str) -> str:
        return base_prompt

def qwen_parse_warmup_submit_outcomes(
    parsed: ParsedAction,
    target_qid: str,
) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    """
    Select one forecast for target_qid like QwenBasicAgent._qwen_handle_submit.

    Validates with BasicAgent._forecasts_within_probability_bounds (same gate as the
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
    if not BasicAgent._forecasts_within_probability_bounds([{"qid": qid, "outcomes": outcomes}]):
        return None, "Forecast submission failed."

    return outcomes, None
