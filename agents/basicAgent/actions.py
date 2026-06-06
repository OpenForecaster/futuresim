"""Action handler methods for BasicAgent."""

from typing import Any, Dict, List, Optional

from futuresim_agents.utils.budget import BudgetTracker
from futuresim_agents.utils.memory import ActiveMemory, StructuredMemory
from environment.interfaces import PredictionSubmission

from .tools import execute_news_search, optional_search_dates_from_parsed


class BasicActionHandlers:
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
