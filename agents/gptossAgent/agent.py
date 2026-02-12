from __future__ import annotations

import json
import hashlib
from datetime import date, timedelta
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
    extract_reasoning_text,
)


def _to_responses_input(conversation: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Convert a simple chat history [{role, content:str}, ...] into Responses API "input".
    """
    out: List[Dict[str, Any]] = []
    for m in conversation or []:
        role = (m.get("role") or "user").strip()
        content = m.get("content", "")
        blocks: List[Dict[str, str]] = []
        if content is not None and str(content) != "":
            blocks.append({"type": "input_text", "text": str(content)})
        out.append({"role": role, "content": blocks})
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

    def _seed_harmony_conversation(self, instructions: str) -> Tuple[str, List[Dict[str, str]]]:
        """
        Build initial Responses payload in either baseline mode (`instructions`) or
        A/B mode (`first_user`).
        """
        mode = self._get_gptoss_prompt_mode()
        if mode == "first_user":
            first_user = (instructions or "").strip()
            if first_user:
                first_user = f"{first_user}\n\nBegin."
            else:
                first_user = "Begin."
            return "", [{"role": "user", "content": first_user}]
        return instructions, [{"role": "user", "content": "Begin."}]

    def act(self, doc_interface, forecast_interface, current_date: date) -> List[Dict[str, Any]]:
        self._timer.reset()
        self._timer.start_day()

        if self._memory:
            self._memory.set_date(current_date)

        self._setup_day(forecast_interface, current_date)

        instructions = self._build_harmony_instructions(current_date)
        instructions_for_api, conversation = self._seed_harmony_conversation(instructions)

        all_forecasts = self._run_harmony_action_loop(
            instructions=instructions_for_api,
            conversation=conversation,
            forecast_interface=forecast_interface,
        )

        if self._memory:
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
        base_prompt = BasicAgent._build_instructions(self, current_date)
        return self._rewrite_basic_prompt_for_harmony(base_prompt, current_date)

    def _rewrite_basic_prompt_for_harmony(self, base_prompt: str, current_date: date) -> str:
        # Keep everything before RESPONSE FORMAT as-is (scoring, data notes, etc.).
        if "## RESPONSE FORMAT" in base_prompt:
            prefix = base_prompt.split("## RESPONSE FORMAT", 1)[0].rstrip()
        else:
            prefix = base_prompt.rstrip()

        memory_flow_note = (
            "Note: After ending your day, you will be prompted to update your memory."
            if self._memory
            else ""
        )

        has_search = bool(self._search_handler.is_available)
        end_day_num = 4 if has_search else 3
        submit_num = 3 if has_search else 2

        cutoff_desc = "today's date"
        if has_search and self.config.search_cutoff_days > 0:
            cutoff_date = current_date - timedelta(days=self.config.search_cutoff_days)
            cutoff_desc = f"{cutoff_date} (today - {self.config.search_cutoff_days} days)"

        search_tool_section = ""
        if has_search:
            search_tool_section = f"""
### 2. Search News Articles
Call tool:
`search_news(query="...", from_date="YYYY-MM-DD" (optional), to_date="YYYY-MM-DD" (optional))`

Notes:
- `to_date` is capped at {cutoff_desc} (no future leakage).
- Use at most one `search_news` call per turn.
"""

        suffix = f"""
## RESPONSE FORMAT

## ACTION TYPES

### 1. Query Questions (explore data)
Call tool:
`query_df(code="print(df[df['is_resolved'] == False][['qid', 'title', 'answer_type']].head())")`
`query_df` supports multi-line Python code in one call; statements execute sequentially.

Use `print()` to ensure you see output. We are not executing in a jupyter notebook, so `.head()` preview alone can be unreliable.
{search_tool_section}
### {submit_num}. Submit Forecasts
Call tool:
`submit_forecasts(forecasts=[{{"qid":"QUESTION_ID","outcomes":{{"Answer1":0.5,"Answer2":0.3}}}}])`
`submit_forecasts` accepts forecasts on multiple questions in one call via the `forecasts=[...]` list.

### {end_day_num}. End Day (proceed to next day)
Call tool:
`next_day()`

## INTERACTION FLOW
You have {self.config.max_actions} actions per day. Each query, search, or submission uses 1 action.
You can interleave queries, searches, and submissions as needed.
When ready to move on, call `next_day()` to end your day.
{memory_flow_note}

## SUBMISSION RULES
- qid must be from an active (is_resolved=False) question you saw in query results
- One forecast per QID per submit action (can submit multiple times for different questions)
- Max {self.config.max_outcomes_per_question} outcomes per question.
- Outcome names must be REAL predicted answers (e.g., person names, locations, numbers)
- NEVER use placeholders like "Unknown", "TBD", "Other", "N/A" - these ALWAYS hurt your score
- Probabilities must sum to ≤ 1.0
- Submit at least one forecast before ending your day

---

You have {self.config.max_actions} actions available. Begin."""
        return f"{prefix}\n\n{suffix.strip()}"

    # =========================================================================
    # Harmony action loop
    # =========================================================================

    def _run_harmony_action_loop(
        self,
        *,
        instructions: str,
        conversation: List[Dict[str, str]],
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
        # Record the actual instructions used for this loop in compact form.
        instructions_text = instructions or ""
        instructions_sha256 = hashlib.sha256(instructions_text.encode("utf-8")).hexdigest()
        instructions_preview = instructions_text[:1500]
        prompt_mode = self._get_gptoss_prompt_mode()
        reasoning_effort = str(getattr(self.config, "gptoss_reasoning_effort", "medium") or "medium").strip().lower()
        include_reasoning = bool(getattr(self.config, "gptoss_include_reasoning", True))

        while actions_remaining > 0:
            with self._timer.track("llm"):
                resp_json = self._call_responses(
                    instructions=instructions,
                    conversation=conversation,
                    tools=tools,
                    sampling_params=self.config.sampling_params,
                )

            parsed, assistant_text, tool_calls = response_to_action(resp_json)
            reasoning_text = extract_reasoning_text(resp_json)

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
            self._timer.record_tokens(usage)

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
                instructions_sha256=instructions_sha256,
                instructions_preview=instructions_preview,
                reasoning=reasoning_text if reasoning_text else None,
                usage=usage,
            )

            if parsed is None:
                # Model didn't call a tool. Consume an action and push back a firm reminder.
                actions_remaining -= 1
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
                    actions_remaining=actions_remaining,
                    qid=target_qid,
                )
                continue

            if parsed.action_type == "search":
                actions_remaining = self._harmony_handle_search(
                    conversation=conversation,
                    forecast_interface=forecast_interface,
                    parsed=parsed,
                    actions_remaining=actions_remaining,
                    qid=target_qid,
                )
                continue

            if parsed.action_type == "submit":
                actions_remaining, submitted = self._harmony_handle_submit(
                    conversation=conversation,
                    forecast_interface=forecast_interface,
                    parsed=parsed,
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
        conversation: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        sampling_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        from inference.vllm import VLLMInference

        if not isinstance(self.inference, VLLMInference):
            raise TypeError("GPTOSSBasicAgent currently requires provider=vllm (VLLMInference).")

        sp = dict(sampling_params or {})
        sp["tools"] = tools
        sp["tool_choice"] = "auto"  # vLLM Harmony requires this

        overrides: Dict[str, Any] = {
            # Keep the model from trying to emit many tool calls at once.
            "parallel_tool_calls": False,
            "max_tool_calls": 1,
        }
        effort = str(getattr(self.config, "gptoss_reasoning_effort", "medium") or "medium").strip().lower()
        if effort in ("low", "medium", "high"):
            overrides["reasoning_effort"] = effort
        overrides["include_reasoning"] = bool(getattr(self.config, "gptoss_include_reasoning", True))

        return self.inference.responses_json(
            instructions=instructions,
            input_messages=_to_responses_input(conversation),
            sampling_params=sp,
            request_overrides=overrides,
        )

    @staticmethod
    def _render_turn_for_logging(assistant_text: str, tool_calls: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        if assistant_text and assistant_text.strip():
            parts.append(assistant_text.strip())
        if tool_calls:
            parts.append("TOOL_CALLS:\n" + json.dumps(tool_calls[:3], indent=2, sort_keys=True))
        return "\n\n".join(parts).strip()

    def _log_harmony_action(
        self,
        *,
        forecast_interface,
        conversation: List[Dict[str, str]],
        rendered_response: str,
        phase: str,
        actions_remaining: int,
        qid: Optional[str] = None,
        **extra,
    ) -> None:
        last_user = ""
        for m in reversed(conversation):
            if m.get("role") == "user":
                last_user = m.get("content") or ""
                break
        metadata = {"phase": phase, "actions_remaining": actions_remaining, "qid": qid, **extra}
        forecast_interface.log_model_output(last_user, rendered_response, metadata)

    # =========================================================================
    # Tool handlers
    # =========================================================================

    def _harmony_handle_query(
        self,
        *,
        conversation: List[Dict[str, str]],
        forecast_interface,
        parsed: ParsedAction,
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

        conversation.append({"role": "user", "content": feedback})
        return actions_remaining

    def _harmony_handle_search(
        self,
        *,
        conversation: List[Dict[str, str]],
        forecast_interface,
        parsed: ParsedAction,
        actions_remaining: int,
        qid: Optional[str],
    ) -> int:
        actions_remaining -= 1
        if not self._search_handler.is_available:
            feedback = f"SEARCH ERROR: Search is not available.\n\nActions remaining: {actions_remaining}"
            conversation.append({"role": "user", "content": feedback})
            return actions_remaining

        if not parsed.query:
            feedback = f"SEARCH ERROR: Missing query.\n\nActions remaining: {actions_remaining}"
            conversation.append({"role": "user", "content": feedback})
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
            )
        if error:
            feedback = f"SEARCH ERROR: {error}\n\nActions remaining: {actions_remaining}"
        else:
            feedback = f"SEARCH RESULTS:\n{result}\n\nActions remaining: {actions_remaining}"
        conversation.append({"role": "user", "content": feedback})
        return actions_remaining

    def _harmony_handle_submit(
        self,
        *,
        conversation: List[Dict[str, str]],
        forecast_interface,
        parsed: ParsedAction,
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
            feedback = f"Submitted {len(submitted)} forecast(s). Actions remaining: {actions_remaining}"
        else:
            feedback = f"SUBMIT ERROR: {parsed.error or 'No valid forecasts'}\n\nActions remaining: {actions_remaining}"

        conversation.append({"role": "user", "content": feedback})
        return actions_remaining, submitted

    # =========================================================================
    # Memory update
    # =========================================================================

    def _harmony_memory_update(
        self,
        *,
        instructions: str,
        conversation: List[Dict[str, str]],
        forecast_interface,
        current_date: date,
    ) -> None:
        if not self._memory:
            return

        # Ask for memory update as a separate tool-call turn.
        prompt = (
            f"End of day {current_date}. If you want to update memory for tomorrow, "
            "call update_memory(memory=...) with a complete replacement. "
            "If you do not want to update memory, respond without calling any tool."
        )
        conversation.append({"role": "user", "content": prompt})

        tools = build_memory_tools()
        with self._timer.track("llm"):
            resp_json = self._call_responses(
                instructions=instructions,
                conversation=conversation,
                tools=tools,
                sampling_params=self.config.sampling_params,
            )

        tool_calls = extract_function_calls(resp_json)
        rendered = self._render_turn_for_logging("", tool_calls)
        self._log_harmony_action(
            forecast_interface=forecast_interface,
            conversation=conversation,
            rendered_response=rendered,
            phase="memory_update",
            actions_remaining=-1,
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
        try:
            from inference.vllm import VLLMInference
            if isinstance(self.inference, VLLMInference):
                self.inference.chat([{"role": "user", "content": "ping"}], {"temperature": 0.0, "max_tokens": 1})
        except Exception:
            pass

        from concurrent.futures import ThreadPoolExecutor, as_completed

        max_workers = getattr(self.config, "warmup_parallelism", 20)
        print(f"[{self.agent_id}] Parallelizing warmup with {max_workers} threads...")

        def _run_one(q):
            instructions = self._build_harmony_warmup_instructions(current_date, q, forecast_interface=forecast_interface)
            instructions_for_api, convo = self._seed_harmony_conversation(instructions)
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

        self.warmed_up = True
        self._timer.end_day()
        if self.config.memory_dir:
            self._timer.save_day_stats(self.config.memory_dir, current_date)
        self._timer.reset()
        print(f"[{self.agent_id}] Warmup complete.")

    def _build_harmony_warmup_instructions(self, current_date: date, q, forecast_interface=None) -> str:
        base_prompt = AllQAgent._build_warmup_system_prompt(self, current_date, q, forecast_interface=forecast_interface)
        return self._rewrite_allq_warmup_prompt_for_harmony(base_prompt, current_date, q.qid)

    def _rewrite_allq_warmup_prompt_for_harmony(self, base_prompt: str, current_date: date, target_qid: str) -> str:
        # Keep question context/scoring exactly as AllQ, replace action/response formatting section.
        if "## ACTIONS" in base_prompt:
            prefix = base_prompt.split("## ACTIONS", 1)[0].rstrip()
        else:
            prefix = base_prompt.rstrip()

        has_search = bool(self._search_handler.is_available)
        cutoff_desc = "today's date"
        if has_search and self.config.search_cutoff_days > 0:
            cutoff_date = current_date - timedelta(days=self.config.search_cutoff_days)
            cutoff_desc = f"{cutoff_date} (today - {self.config.search_cutoff_days} days)"

        search_section = ""
        submit_num = 1
        if has_search:
            submit_num = 2
            search_section = f"""
### 1. Search News Articles
Call tool:
`search_news(query="...", from_date="YYYY-MM-DD" (optional), to_date="YYYY-MM-DD" (optional))`

Notes:
- `to_date` is capped at {cutoff_desc} (no future leakage).
- Use at most one `search_news` call per turn.
"""

        suffix = f"""
## ACTIONS
You have {self.config.warmup_max_actions} actions to research and forecast this question.

{search_section}
### {submit_num}. Submit Forecast
Call tool:
`submit_forecasts(forecasts=[{{"qid":"{target_qid}","outcomes":{{"Answer1":0.5,"Answer2":0.3}}}}])`
`submit_forecasts` supports forecasts on multiple questions, but in warmup you should submit for qid `{target_qid}` only.

(Submitting ends your turn for this question).

## SUBMISSION RULES
- qid must be {target_qid}
- Outcome names must be REAL predicted answers (e.g., person names, locations, numbers)
- You can submit up to {self.config.max_outcomes_per_question} outcomes per question.
- NEVER use placeholders like "Unknown", "TBD", "Other", "N/A" - these ALWAYS hurt your score
- Probabilities must sum to ≤ 1.0
"""
        return f"{prefix}\n\n{suffix.strip()}\n"
