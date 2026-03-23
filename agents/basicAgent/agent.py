"""
BasicAgent: Main agent class for LLM-based forecasting.

Uses chain-of-thought with <reasoning> and <action type="..."> tags.
"""

import json
import os
import re
from functools import lru_cache
from datetime import date, timedelta
from typing import List, Dict, Any, Optional, Tuple

from agents.base import BaseAgent
from agents.utils.forecast_parser import parse_action, extract_memory, extract_memory_ops, extract_mem_ops
from agents.utils.budget import BudgetSettings, BudgetTracker
from agents.utils.timing import AgentTimer
from environment.interfaces import PredictionSubmission

from .config import AgentConfig
from .memory import BasicMemory
from agents.utils.memory import StructuredMemory, ActiveMemory
from .query import QueryHandler
from .search import SearchHandler
from .feedback import FeedbackHandler


@lru_cache(maxsize=16)
def _load_budget_tokenizer(model_name: str):
    if not model_name or not os.path.exists(model_name):
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        return None


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
    
    Uses <reasoning> for chain-of-thought and <action type="..."> for actions.
    """
    
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

        # Load memory for this date (loads most recent snapshot before current_date)
        if self._memory is not None:
            self._memory.set_date(current_date)
        
        # Setup handlers
        self._setup_day(forecast_interface, current_date)
        
        # Build initial prompt
        messages = [{"role": "user", "content": self._build_instructions(current_date)}]
        
        # Unified action loop (includes memory phase for structured/active memory)
        all_forecasts = self._run_action_loop(messages, forecast_interface, current_date)

        # Separate memory update only for BasicMemory (plain text, single-shot).
        # StructuredMemory/ActiveMemory are handled inside _run_action_loop via
        # the next-triggered or token-threshold-triggered memory phase.
        if (
            self._memory is not None
            and not getattr(self, '_memory_phase_completed', False)
            and not isinstance(self._memory, (StructuredMemory, ActiveMemory))
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
                    or self.config.submit_reserve_tokens
                ),
                force_submit_threshold_tokens=(
                    self.config.warmup_force_submit_threshold_tokens
                    or self.config.force_submit_threshold_tokens
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
            )
        return BudgetTracker(settings, token_estimator=self._estimate_budget_tokens)

    def _build_budget_overview(self, *, warmup: bool = False, per_question: bool = False) -> str:
        """Human-readable budget instructions for prompts."""
        settings = self._get_budget_settings(warmup=warmup)
        lines: List[str] = []
        if settings.max_actions is not None:
            if per_question:
                lines.append(f"You have {settings.max_actions} actions to research and forecast this question.")
            else:
                lines.append(
                    f"You have {settings.max_actions} actions per day. Each query, search, or submission uses 1 action."
                )
        if settings.max_total_tokens is not None:
            scope = "this question" if per_question else "this session"
            lines.append(
                f"You have a context budget of {settings.max_total_tokens} tokens for {scope}. "
                "This tracks the current prompt length, not cumulative tokens spent."
            )
            if warmup:
                # Warmup keeps submit_reserve / force_submit semantics
                lines.append(
                    f"Keep at least {settings.submit_reserve_tokens} tokens free for a final submit. "
                    f"Force-submit once the remaining context budget is at or below {settings.force_submit_threshold_tokens}."
                )
            elif settings.memory_phase_threshold_tokens is not None:
                # Non-warmup unified loop: explain memory phase transition
                action_budget = settings.max_total_tokens - settings.memory_phase_threshold_tokens
                lines.append(
                    "When you finish forecasting, call <action type=\"next\"/> to transition to the memory update phase. "
                    f"The transition also happens automatically when ~{action_budget} tokens have been used. "
                    f"The remaining ~{settings.memory_phase_threshold_tokens} tokens are reserved for memory updates."
                )
        if settings.max_actions is not None and settings.max_total_tokens is not None:
            lines.append("If both budgets are configured, both are enforced and the session ends when either one is exhausted.")
        return "\n".join(lines)

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
        return tracker.status_text()

    def _build_force_submit_preamble(self, budget: BudgetTracker) -> str:
        """Shared force-submit wording for action/token-constrained loops."""
        lines = [
            "FINAL ACTION: You MUST submit your best guess forecast now."
        ]
        status = budget.status_text()
        if status:
            lines.append(status)
        return "\n".join(lines)

    def _search_results_description(self) -> str:
        chunk_tokens = self._search_handler.chunk_tokens
        if chunk_tokens is None:
            return f"Search returns up to {self.config.max_search_results} retrieved article chunks."
        return (
            f"Search returns up to {self.config.max_search_results} retrieved article chunks, "
            f"each roughly {chunk_tokens} tokens long."
        )

    def _estimate_budget_tokens(self, payload: Any) -> int:
        if payload is None:
            return 0
        text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        if not text:
            return 0

        tokenizer = _load_budget_tokenizer(str(self.model_name or ""))
        if tokenizer is not None:
            try:
                return len(tokenizer.encode(text, add_special_tokens=False))
            except TypeError:
                return len(tokenizer.encode(text))
            except Exception:
                pass

        return max(1, (len(text) + 3) // 4)

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
        return (
            "## UPDATE CADENCE\n"
            f"You make updates every {self._get_timegap_days()} days. "
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
            return ""
        resolved = feedback.get('resolved_today', [])
        if not resolved:
            return ""

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
            memory_prompt = self._build_active_memory_prompt(
                current_date, day_qids=getattr(self, '_day_qids', None)
            )
        elif isinstance(self._memory, StructuredMemory):
            memory_prompt = self._build_structured_memory_prompt(current_date)
        else:
            return ""

        # Prepend privileged cheat-feedback when enabled
        if self.config.cheat_feedback:
            cheat_data = self._forecast_interface.get_cheat_feedback(
                detail=self.config.cheat_feedback_detail
            )
            if cheat_data.get("items"):
                cheat_section = FeedbackHandler.format_cheat_feedback(
                    cheat_data, self.config.cheat_feedback_detail
                )
                memory_prompt = cheat_section + "\n\n" + memory_prompt

        return memory_prompt

    def _handle_memory_phase_action(
        self, messages, forecast_interface, response, parsed,
        budget: BudgetTracker, reasoning=None,
    ) -> None:
        """Handle a single response during the memory-update phase of the unified loop."""
        forecast_interface.log_model_output(
            "(memory_update_loop)", response,
            {"phase": "memory_update", "current_memory_entries": self._memory.entry_count, "reasoning": reasoning}
        )

        # For ActiveMemory: extract and apply mem_df XML ops from every response
        mem_ops_applied = 0
        if isinstance(self._memory, ActiveMemory):
            mem_adds, mem_updates, mem_deletes = extract_mem_ops(response)
            mem_ops_applied = len(mem_adds) + len(mem_updates) + len(mem_deletes)
            for qid in mem_deletes:
                self._memory.mem_delete(qid)
            for add in mem_adds:
                self._memory.mem_add(
                    qid=add["qid"], question=add.get("question", ""),
                    memory=add["memory"], category=add.get("category", ""),
                )
            for upd in mem_updates:
                self._memory.mem_update(
                    qid=upd["qid"], memory=upd["memory"],
                    category=upd.get("category"),
                )

        if parsed.action_type in ("memory_retrieve", "memory_new", "memory_update", "memory_delete"):
            self._handle_memory_action(messages, forecast_interface, response, parsed, budget, reasoning=reasoning)
        elif parsed.action_type in ("mem_add", "mem_update", "mem_delete"):
            self._handle_mem_action(messages, forecast_interface, response, parsed, budget, reasoning=reasoning)
        elif mem_ops_applied > 0:
            feedback = f"MEM: Applied {mem_ops_applied} mem_df operations ({self._memory.mem_count} entries). Use <action type=\"next\"/> when done."
            self._append_with_budget(messages, budget, {"role": "user", "content": budget.format_feedback(feedback)})
        else:
            # Legacy XML ops fallback for meta-insights
            adds, deletes = extract_memory_ops(response)
            if adds or deletes:
                for entry_name in deletes:
                    self._memory.delete_entry(entry_name)
                for add in adds:
                    try:
                        self._memory.add_entry(
                            add["name"], add.get("description", add.get("name", "")), add.get("content", "")
                        )
                    except ValueError:
                        pass
                feedback = f"MEMORY: Applied {len(adds)} adds, {len(deletes)} deletes. Total entries: {self._memory.entry_count}."
                self._append_with_budget(messages, budget, {"role": "user", "content": budget.format_feedback(feedback)})
            else:
                feedback = "No recognized memory action. Use memory_new/update/delete or <action type=\"next\"/> to finish."
                self._append_with_budget(messages, budget, {"role": "user", "content": budget.format_feedback(feedback)})

    # =========================================================================
    # Action Loop
    # =========================================================================

    def _run_action_loop(self, messages: List[Dict], forecast_interface, current_date: date = None) -> List[Dict]:
        """
        Main action loop: process agent responses until the loop budget is exhausted or the day ends.

        When a memory_phase_threshold is configured, the loop automatically transitions
        to a memory-update phase (same conversation, same budget tracker) once remaining
        tokens drop to that threshold.  No separate memory loop is needed.

        Returns list of all submitted forecasts.
        """
        budget = self._create_budget_tracker()
        budget.bootstrap_context(messages)
        all_forecasts = []
        empty_retries = 0
        max_empty_retries = 2
        memory_phase = False
        memory_phase_eligible = (
            current_date is not None
            and self._memory is not None
            and isinstance(self._memory, (StructuredMemory, ActiveMemory))
            and budget.settings.memory_phase_threshold_tokens is not None
        )

        while not budget.is_exhausted():
            # --- Memory phase transition ---
            if not memory_phase and memory_phase_eligible and budget.should_enter_memory_phase():
                memory_phase = True
                budget.memory_phase = True
                memory_prompt = self._build_memory_phase_prompt(current_date)
                self._append_with_budget(
                    messages, budget,
                    {"role": "user", "content": memory_prompt},
                )

            # Get agent response (track LLM time)
            try:
                with self._timer.track("llm"):
                    response, usage = self.inference.chat(messages, self.config.sampling_params)
            except Exception as e:
                print(f"  [{self.agent_id}] LLM error: {e}, ending turn")
                self._log_action(
                    forecast_interface,
                    messages,
                    "",
                    "llm_error",
                    budget,
                    error=str(e),
                )
                break

            # Handle empty response (API failure with graceful fallback)
            if not response or not response.strip():
                if empty_retries < max_empty_retries:
                    empty_retries += 1
                    print(f"  [{self.agent_id}] Empty LLM response, retrying ({empty_retries}/{max_empty_retries})")
                    continue
                print(f"  [{self.agent_id}] Empty LLM response after {max_empty_retries} retries, ending turn")
                self._log_action(forecast_interface, messages, response or "", "api_failure", budget)
                break

            empty_retries = 0
            reasoning = usage.get("_reasoning_content") if usage else None

            # Record token usage and cost
            self._timer.record_tokens(usage)
            self._timer.record_cost(usage.get("cost", 0), "llm")
            budget.record_usage(usage)

            self._append_with_budget(messages, budget, {"role": "assistant", "content": response})

            # Track QIDs mentioned in reasoning/code (for active memory expansion)
            if hasattr(self, '_day_qids'):
                self._day_qids.update(re.findall(r'(?:qid|QID)\W{0,10}(\d+)', response))

            # Parse and handle the action
            parsed = parse_action(response, self.config.max_outcomes_per_question)

            if parsed.action_type == "next":
                if not memory_phase and memory_phase_eligible:
                    # Transition to memory phase instead of ending the day
                    memory_phase = True
                    budget.memory_phase = True
                    self._log_action(
                        forecast_interface, messages, response,
                        "next_entering_memory", budget, reasoning=reasoning,
                    )
                    memory_prompt = self._build_memory_phase_prompt(current_date)
                    self._append_with_budget(
                        messages, budget,
                        {"role": "user", "content": memory_prompt},
                    )
                    continue
                else:
                    # Actually end the day (memory phase done, or no memory)
                    phase_label = "memory_update_done" if memory_phase else "next_day"
                    self._log_action(
                        forecast_interface, messages, response,
                        phase_label, budget, reasoning=reasoning,
                    )
                    break

            if memory_phase:
                # --- Memory phase: only memory actions ---
                self._handle_memory_phase_action(
                    messages, forecast_interface, response, parsed, budget, reasoning=reasoning,
                )
            elif parsed.action_type == "query":
                self._handle_query(
                    messages, forecast_interface, response, parsed, budget, reasoning=reasoning
                )
            elif parsed.action_type == "search":
                self._handle_search(
                    messages, forecast_interface, response, parsed, budget, reasoning=reasoning
                )
            elif parsed.action_type in ("memory_retrieve", "memory_new", "memory_update", "memory_delete"):
                self._handle_memory_action(
                    messages, forecast_interface, response, parsed, budget, reasoning=reasoning
                )
            elif parsed.action_type in ("mem_add", "mem_update", "mem_delete"):
                self._handle_mem_action(
                    messages, forecast_interface, response, parsed, budget, reasoning=reasoning
                )
            elif parsed.action_type == "submit":
                forecasts = self._handle_submit(
                    messages, forecast_interface, response, parsed, budget, reasoning=reasoning
                )
                all_forecasts.extend(forecasts)
            else:
                self._handle_invalid(
                    messages, forecast_interface, response, parsed, budget, reasoning=reasoning
                )

        # Post-loop: persist memory if memory phase was entered
        if memory_phase and self._memory is not None:
            self._memory._save(current_date)
        elif memory_phase_eligible and not memory_phase:
            print(f"  [{self.agent_id}] Budget exhausted before memory phase. Memory not updated in-loop.")

        self._memory_phase_completed = memory_phase
        return all_forecasts
    
    # =========================================================================
    # Action Handlers
    # =========================================================================
    
    def _handle_query(self, messages, forecast_interface, response, parsed, budget: BudgetTracker, qid: str = None, reasoning=None, raw_stream: Optional[str] = None) -> None:
        """Handle query action."""
        budget.consume_action()
        
        if parsed.code:
            extra_ctx = None
            if isinstance(self._memory, ActiveMemory):
                extra_ctx = {"mem_df": self._memory.get_mem_df()}
            with self._timer.track("df_query"):
                result, error = self._query_handler.execute(parsed.code, extra_context=extra_ctx)
            self._log_action(forecast_interface, messages, response, "query", budget, qid=qid, reasoning=reasoning, raw_stream=raw_stream)
            
            if error:
                feedback = f"QUERY ERROR: {error}"
            else:
                feedback = f"QUERY RESULT:\n{result}"
        else:
            self._log_action(forecast_interface, messages, response, "query_error", budget, qid=qid, error=parsed.error, reasoning=reasoning, raw_stream=raw_stream)
            feedback = f"ERROR: {parsed.error}"
        
        self._append_with_budget(messages, budget, {"role": "user", "content": budget.format_feedback(feedback)})
    
    def _handle_search(self, messages, forecast_interface, response, parsed, budget: BudgetTracker, qid: str = None, reasoning=None, raw_stream: Optional[str] = None) -> None:
        """Handle search action. Returns full chunk content directly."""
        budget.consume_action()
        
        if not self._search_handler.is_available:
            self._log_action(forecast_interface, messages, response, "search_unavailable", budget, qid=qid, reasoning=reasoning, raw_stream=raw_stream)
            feedback = "SEARCH ERROR: Search is not available."
        elif parsed.query:
            # Parse date range if provided
            min_date = None
            max_date = None
            if parsed.search_from:
                try:
                    from datetime import datetime
                    min_date = datetime.strptime(parsed.search_from, "%Y-%m-%d").date()
                except ValueError:
                    pass
            if parsed.search_to:
                try:
                    from datetime import datetime
                    max_date = datetime.strptime(parsed.search_to, "%Y-%m-%d").date()
                except ValueError:
                    pass
            
            with self._timer.track("search"):
                result, error = self._search_handler.search(
                    parsed.query, 
                    max_results=self.config.max_search_results,
                    min_date=min_date,
                    max_date=max_date,
                )
            self._log_action(forecast_interface, messages, response, "search", budget, qid=qid, reasoning=reasoning, raw_stream=raw_stream)
            
            if error:
                feedback = f"SEARCH ERROR: {error}"
            else:
                feedback = f"SEARCH RESULTS:\n{result}"
        else:
            self._log_action(forecast_interface, messages, response, "search_error", budget, qid=qid, error=parsed.error, reasoning=reasoning, raw_stream=raw_stream)
            feedback = "SEARCH ERROR: No query provided."
        
        self._append_with_budget(messages, budget, {"role": "user", "content": budget.format_feedback(feedback)})

    def _handle_memory_action(self, messages, forecast_interface, response, parsed,
                              budget: BudgetTracker, reasoning=None) -> None:
        """Handle memory tool calls (retrieve/add/update/delete). Follows search handler pattern."""
        budget.consume_action()

        if not isinstance(self._memory, (StructuredMemory, ActiveMemory)):
            feedback = "MEMORY ERROR: Structured memory is not enabled."
            self._log_action(forecast_interface, messages, response, "memory_unavailable", budget, reasoning=reasoning)
        elif parsed.error:
            feedback = f"MEMORY ERROR: {parsed.error}"
            self._log_action(forecast_interface, messages, response, f"{parsed.action_type}_error", budget, error=parsed.error, reasoning=reasoning)
        elif parsed.action_type == "memory_retrieve":
            entry = self._memory.retrieve(parsed.memory_entry_name)
            if entry is None:
                feedback = f"MEMORY ERROR: No entry with name '{parsed.memory_entry_name}'."
            else:
                feedback = f"MEMORY ENTRY:\n{entry}"
            self._log_action(forecast_interface, messages, response, "memory_retrieve", budget, reasoning=reasoning)
        elif parsed.action_type == "memory_new":
            data = parsed.memory_new_data
            try:
                entry_name = self._memory.add_entry(data["name"], data["description"], data["content"])
                feedback = f"MEMORY: Added entry [{entry_name}]. Total entries: {self._memory.entry_count}/{self._memory._max_entries if hasattr(self._memory, '_max_entries') else '?'}."
            except ValueError as exc:
                feedback = f"MEMORY ERROR: {exc}"
            self._log_action(forecast_interface, messages, response, "memory_new", budget, reasoning=reasoning)
        elif parsed.action_type == "memory_update":
            update_data = {k: v for k, v in parsed.memory_update_data.items() if k != "name"}
            ok = self._memory.update_entry(parsed.memory_entry_name, **update_data)
            if ok:
                feedback = f"MEMORY: Updated [{parsed.memory_entry_name}]."
            else:
                feedback = f"MEMORY ERROR: No entry with name '{parsed.memory_entry_name}'."
            self._log_action(forecast_interface, messages, response, "memory_update", budget, reasoning=reasoning)
        elif parsed.action_type == "memory_delete":
            ok = self._memory.delete_entry(parsed.memory_entry_name)
            if ok:
                feedback = f"MEMORY: Deleted [{parsed.memory_entry_name}]. Remaining: {self._memory.entry_count}."
            else:
                feedback = f"MEMORY ERROR: No entry with name '{parsed.memory_entry_name}'."
            self._log_action(forecast_interface, messages, response, "memory_delete", budget, reasoning=reasoning)
        else:
            feedback = f"MEMORY ERROR: Unknown memory action '{parsed.action_type}'."
            self._log_action(forecast_interface, messages, response, "memory_unknown", budget, reasoning=reasoning)

        self._append_with_budget(messages, budget, {"role": "user", "content": budget.format_feedback(feedback)})

    def _handle_mem_action(self, messages, forecast_interface, response, parsed,
                           budget: BudgetTracker, reasoning=None) -> None:
        """Handle mem_df tool calls (mem_add/update/delete) for ActiveMemory."""
        budget.consume_action()

        if not isinstance(self._memory, ActiveMemory):
            feedback = "MEM ERROR: Active memory is not enabled."
            self._log_action(forecast_interface, messages, response, "mem_unavailable", budget, reasoning=reasoning)
        elif parsed.error:
            feedback = f"MEM ERROR: {parsed.error}"
            self._log_action(forecast_interface, messages, response, f"{parsed.action_type}_error", budget, error=parsed.error, reasoning=reasoning)
        elif parsed.action_type == "mem_add":
            data = parsed.mem_data
            self._memory.mem_add(
                qid=data["qid"], question=data.get("question", ""),
                memory=data["memory"], category=data.get("category", ""),
            )
            feedback = f"MEM: Added entry for Q{data['qid']}. Total: {self._memory.mem_count} entries."
            self._log_action(forecast_interface, messages, response, "mem_add", budget, reasoning=reasoning)
        elif parsed.action_type == "mem_update":
            self._memory.mem_update(
                qid=parsed.mem_qid, memory=parsed.mem_data["memory"],
                category=parsed.mem_data.get("category"),
            )
            feedback = f"MEM: Updated entry for Q{parsed.mem_qid}."
            self._log_action(forecast_interface, messages, response, "mem_update", budget, reasoning=reasoning)
        elif parsed.action_type == "mem_delete":
            ok = self._memory.mem_delete(parsed.mem_qid)
            if ok:
                feedback = f"MEM: Deleted Q{parsed.mem_qid}. Remaining: {self._memory.mem_count}."
            else:
                feedback = f"MEM ERROR: No entry for Q{parsed.mem_qid}."
            self._log_action(forecast_interface, messages, response, "mem_delete", budget, reasoning=reasoning)
        else:
            feedback = f"MEM ERROR: Unknown mem action '{parsed.action_type}'."
            self._log_action(forecast_interface, messages, response, "mem_unknown", budget, reasoning=reasoning)

        self._append_with_budget(messages, budget, {"role": "user", "content": budget.format_feedback(feedback)})

    def _handle_submit(self, messages, forecast_interface, response, parsed, budget: BudgetTracker, qid: str = None, reasoning=None, raw_stream: Optional[str] = None) -> List:
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
            self._log_action(forecast_interface, messages, response, "submit", budget, qid=log_qid, submitted_qids=submitted_qids,
                           num_forecasts=len(submitted), dropped_forecasts=dropped_forecasts, reasoning=reasoning, raw_stream=raw_stream)
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
            self._log_action(forecast_interface, messages, response, "submit_error", budget, qid=log_qid, error=parsed.error, reasoning=reasoning, raw_stream=raw_stream)
            feedback = f"SUBMIT ERROR: {parsed.error}"
        
        self._append_with_budget(messages, budget, {"role": "user", "content": budget.format_feedback(feedback)})
        return submitted
    
    def _handle_invalid(self, messages, forecast_interface, response, parsed, budget: BudgetTracker, qid: str = None, reasoning=None, raw_stream: Optional[str] = None) -> None:
        """Handle invalid/unknown action."""
        budget.consume_action()
        self._log_action(forecast_interface, messages, response, "invalid", budget, qid=qid, error=parsed.error, reasoning=reasoning, raw_stream=raw_stream)
        
        error_msg = parsed.error or 'Use <action type="...">...</action> format.'
        feedback = f"No valid action found. {error_msg}"
        self._append_with_budget(messages, budget, {"role": "user", "content": budget.format_feedback(feedback)})
    
    # =========================================================================
    # Helpers
    # =========================================================================
    
    def _log_action(self, forecast_interface, messages, response, phase, 
                   budget: BudgetTracker, qid: str = None, **extra) -> None:
        """Log model output with metadata. qid indicates the question context if known."""
        last_user = messages[-2]["content"] if len(messages) >= 2 else ""
        metadata = {
            "phase": phase, 
            "qid": qid,
            **budget.metadata(),
            **extra
        }
        forecast_interface.log_model_output(last_user, response, metadata)

    def _record_matcher_timing(self, duration: float, cost: float = 0) -> None:
        """Record answer matcher latency and cost in timing stats."""
        self._timer.record("matcher", duration)
        self._timer.record_cost(cost, "matcher")
    
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
- **Peer Score (relative ranking)**: `100 × (avg others' Brier - your Brier)` (baseline 0 if you are the only predictor). Positive means better than peers."""

        mechanics: Dict[str, str] = {
            "accuracy_calibration": "**Accuracy + Calibration**: Assign probabilities that reflect true likelihood.",
            "binary_outcomes": "**Binary Outcomes**: Use exact outcomes \"Yes\" and \"No\".",
            "time_weighted": "**Time-Weighted**: The score is summed over all days the prediction was held, so early predictions have higher weight.",
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
- **Peer Score**: For each snapshot, your Brier Skill Score is compared to the crowd by subtracting the average of other agents' Brier Skill Scores (baseline 0 if you are the only predictor)."""

        mechanics: Dict[str, str] = {
            "accuracy_calibration": "**Accuracy + Calibration**: Assign high probability to the TRUE outcome and keep probabilities well-calibrated.",
            "time_weighted": "**Time-Weighted**: The score is summed over all days the prediction was held, so early predictions have higher weight.",
            "question_count": "**Prediction-Count Incentive**: Scores are summed (not averaged) across all questions you predict on.",
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
You are evaluated on **Brier Skill Score** = 1 - Σ(pᵢ - yᵢ)² summed over all outcomes.
- pᵢ = your probability for outcome i
- yᵢ = 1 if outcome i is TRUE, 0 otherwise
- **Higher is better**: 1.0 = perfect, 0.0 = no skill, negative = worse than uniform.{peer_text}

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
            # Single-agent: market_aggregate just reflects own predictions, don't mention it
            return "Note: `my_prediction` column contains your current forecast as a dict (or None if not yet predicted)."
        else:
            # Multi-agent: market_aggregate is meaningful
            return """Note: `market_aggregate` and `my_prediction` columns contain Python dicts (or None). You can access them directly, e.g. `row['market_aggregate']['outcome_name']`.
The `num_predictions` column shows how many predictions have been made on that question."""
    
    def _get_source_rules(self) -> str:
        """Get source-specific submission rules."""
        source_name = getattr(self._forecast_interface, 'source_name', 'openforesight')
        
        if source_name == "metaculus_binary":
            return """
## BINARY QUESTION RULES
All questions are Yes/No binary. Your prediction MUST use exactly:
- **"Yes"** for the affirmative outcome
- **"No"** for the negative outcome

Example submission:
<forecast qid="12345">
  <outcome name="Yes" prob="0.7"/>
  <outcome name="No" prob="0.3"/>
</forecast>
"""
        elif source_name == "metaculus_mcq":
            return """
## MULTIPLE CHOICE RULES
Each question has enumerated options shown in the 'options' column.
Your prediction MUST use the EXACT option text from the question.
Do NOT paraphrase or abbreviate options.

Example (if options are ["Candidate A", "Candidate B", "Candidate C"]):
<forecast qid="12345">
  <outcome name="Candidate A" prob="0.5"/>
  <outcome name="Candidate B" prob="0.3"/>
  <outcome name="Candidate C" prob="0.2"/>
</forecast>
"""
        return ""

    def _build_instructions(self, current_date: date) -> str:
        """Build the instructions for the agent including data info and rules."""
        df_info = self._query_handler.get_info()
        budget_start_status = self._build_start_budget_status()
        budget_start_block = f"Budget at start:\n{budget_start_status}\n\n" if budget_start_status else ""
        cadence_section = self._build_cadence_section(current_date)
        
        memory_section = ""
        if self._memory is not None:
            memory_content = self._memory.get()
            if isinstance(self._memory, ActiveMemory):
                # Active memory: meta-insights index + mem_df documentation
                max_ent = self._memory._max_entries
                meta_index = self._memory.get_index()
                if meta_index:
                    meta_part = f"""### Meta-Insights ({self._memory.entry_count} entries, max {max_ent})
<memory_index>
{meta_index}
</memory_index>

Use the memory tools below to retrieve full content, add/update/delete meta-insight entries.
"""
                else:
                    meta_part = f"""### Meta-Insights (0 entries, max {max_ent})
No meta-insight entries yet. Use the memory tools below to add entries.
"""
                mem_count = self._memory.mem_count
                memory_section = f"""## YOUR MEMORY
{meta_part}
### Question-Specific Memory (mem_df: {mem_count} entries)
You have a DataFrame `mem_df` with your per-question notes (reasoning, evidence, calibration).
It lives in the same query sandbox as `df` — you can join, filter, and use any pandas operation.
Columns: qid (str), question (str), last_updated (str), memory (str), category (str)

Tip: `df` tells you what questions exist and their current state; `mem_df` holds your accumulated reasoning and evidence. Joining them on qid lets you find questions worth revisiting — e.g. stale notes, categories where you've been wrong, or questions approaching resolution that you haven't checked recently.

Use the mem_add/update/delete tools below to write to mem_df, or query it with the query tool.

"""
            elif isinstance(self._memory, StructuredMemory):
                max_ent = self._memory._max_entries
                memory_index = self._memory.get_index()
                if memory_index:
                    memory_section = f"""## YOUR MEMORY ({self._memory.entry_count} entries, max {max_ent})
<memory_index>
{memory_index}
</memory_index>

The index above shows entry names and short descriptions. Use the memory tools below to retrieve full content, add/update/delete entries.
Entries should be: question-specific reasoning (include QIDs), post-resolution lessons (what went wrong/right), or meta-patterns (cross-question calibration insights).

"""
                else:
                    memory_section = f"""## YOUR MEMORY (0 entries, max {max_ent})
No memory entries yet. Use the memory tools below to add entries during this session.
Entries should be: question-specific reasoning (include QIDs), post-resolution lessons (what went wrong/right), or meta-patterns (cross-question calibration insights).

"""
            elif memory_content:
                # BasicMemory (legacy)
                memory_section = f"""## YOUR MEMORY (reasoning, patterns, and insights from previous days)
<memory>
{memory_content}
</memory>

Use the reasoning and insights above to inform today's forecasts. The DataFrame includes your latest predictions for active questions and your final prediction snapshot on resolved questions.

"""
            if isinstance(self._memory, (StructuredMemory, ActiveMemory)):
                memory_flow_note = "Note: When you finish forecasting (via <action type=\"next\"/>), the session transitions to a memory update phase. The transition also happens automatically if tokens run low."
            else:
                memory_flow_note = "Note: After ending this session, you will be prompted to update your memory."
        else:
            memory_flow_note = ""
        
        # Search section (only if enabled)
        search_section = ""
        search_advice = ""
        has_memory_tools = isinstance(self._memory, (StructuredMemory, ActiveMemory))
        memory_tool_section = ""
        # Action numbering: 1=query, then optionally search, then optionally memory tools, then submit, then end
        next_num = 2
        if self._search_handler.is_available:
            cutoff_desc = "today's date"
            if self.config.search_cutoff_days > 0:
                cutoff_date = current_date - timedelta(days=self.config.search_cutoff_days)
                cutoff_desc = f"{cutoff_date} (today - {self.config.search_cutoff_days} days)"

            search_section = f"""
### {next_num}. Search News Articles
<action type="search">
your search query here
</action>

Optional: Specify date range (YYYY-MM-DD format):
<action type="search" from="2024-12-01" to="2024-12-15">
your search query here
</action>

Note: "to" date is capped at {cutoff_desc} (no future leakage).
{self._search_results_description()}
You can use search to gather evidence before submitting forecasts.
"""
            search_advice = f"""

You have access to a news article database. {self._search_results_description()}"""
            next_num += 1

        if has_memory_tools:
            max_ent = getattr(self._memory, '_max_entries', 500)
            memory_tool_section = f"""
### {next_num}. Retrieve Memory Entry
<action type="memory_retrieve">ENTRY_NAME</action>
Retrieves the full content of a meta-insight entry by its name from the index above.

### {next_num + 1}. Add a new Memory Entry
<action type="memory_new">
<name>lowercase-hyphenated-name-with-qid (max 64 chars, a-z 0-9 hyphens only)</name>
<description>What it stores + when to use it + QIDs (max 256 chars)</description>
<content>Full content (max {self._memory._meta._field_limits['content'] if isinstance(self._memory, ActiveMemory) else 1024} chars)</content>
</action>

### {next_num + 2}. Update an existing Memory Entry
<action type="memory_update" name="ENTRY_NAME">
<description>Updated description</description>
<content>Updated content</content>
</action>
Only include the fields you want to change.

### {next_num + 3}. Delete an existing Memory Entry
<action type="memory_delete">ENTRY_NAME</action>
"""
            next_num += 4

            # Additional mem_df tools for ActiveMemory
            if isinstance(self._memory, ActiveMemory):
                memory_tool_section += f"""
### {next_num}. Add Question Memory
<action type="mem_add">
<qid>Q123</qid>
<question>Will X happen by Y?</question>
<memory>Your reasoning and key evidence (max 1000 chars)</memory>
<category>politics</category>
</action>

### {next_num + 1}. Update Question Memory
<action type="mem_update" qid="Q123">
<memory>Updated reasoning (max 1000 chars)</memory>
</action>

### {next_num + 2}. Delete Question Memory
<action type="mem_delete">Q123</action>

Tip: After submitting a forecast, save your reasoning and key evidence using mem_add/update. At end of day, you will get another chance to update both memory layers.
"""
                next_num += 3
            else:
                memory_tool_section += """
Tip: After submitting a forecast, consider updating or adding the memory entry for that QID with your current reasoning and key evidence. At end of day, you will get another chance to update your memory.
"""

        submit_num = next_num
        end_day_num = next_num + 1

        return f"""You are a forecasting agent. Today is {current_date}. Your goal: make accurate probability predictions.

{self._format_and_cache_feedback(current_date)}

{getattr(self._forecast_interface, 'source_context', '')}

{self._get_source_rules()}

{cadence_section}{memory_section}{self._get_scoring_section()}
{search_advice}
## AVAILABLE DATA
DataFrame `df` with {df_info['n_rows']} questions ({df_info['n_active']} active/unresolved, {df_info['n_resolved']} resolved).

Column descriptions:
{df_info['columns_desc']}

{self._get_data_notes()}

## CODE EXECUTION ENVIRONMENT
Your Python code runs in a sandbox with these variables pre-defined:
- `df`: the pandas DataFrame
- `pd`: pandas module
- `today`: date object for {current_date}
- `date`, `datetime`, `timedelta`: from datetime module
{('- `mem_df`: your question-specific memory DataFrame (join with df on qid to guide which questions to revisit)' + chr(10)) if isinstance(self._memory, ActiveMemory) else ''}Import statements are not available. Standard builtins (len, str, int, float, min, max, sum, sorted, range, etc.) are available.

## RESPONSE FORMAT
Every response must have <reasoning> followed by exactly ONE <action> block.

<reasoning>
Your analysis
</reasoning>
<action type="ACTION_TYPE">
...content based on action type...
</action>

IMPORTANT:
- Only ONE <action> block allowed per turn.
- For "search", this means only 1 search query per turn.
- For "query", you can write complex multi-step Python code in one block.
- For "submit", include exactly one <forecast ...> block (one qid only).

## ACTION TYPES

### 1. Query Questions (explore data)
<action type="query">
```python
print(df[df['is_resolved'] == False][['qid', 'title', 'answer_type']].head())
```
</action>
Use `print()` to ensure you see output. We are not executing in a jupyter notebook, so .head() preview alone can be unreliable.
{search_section}{memory_tool_section}
### {submit_num}. Submit Forecast
<action type="submit">
<forecast qid="QUESTION_ID">
  <outcome name="Answer1" prob="0.5"/>
  <outcome name="Answer2" prob="0.3"/>
</forecast>
</action>

### {end_day_num}. End Session
<action type="next"/>

## INTERACTION FLOW
{self._build_budget_overview()}
You can interleave queries, searches, and submissions as needed. Consider using mem_df early to recall your prior reasoning and identify which questions need attention.
When ready to move on, use <action type="next"/> to end this session.
{memory_flow_note}

## SUBMISSION RULES
- qid must be from an active (is_resolved=False) question you saw in query results
- In one submit action, include exactly one <forecast ...> block for one qid.
- You can submit again in later turns to update that qid.
- Max {self.config.max_outcomes_per_question} outcomes per question.
- Outcome names must be REAL predicted answers (e.g., person names, locations, numbers)
- NEVER use placeholders like "Unknown", "TBD", "Other", "N/A" - these ALWAYS hurt your score
- Probabilities must sum to ≤ 1.0

---
{budget_start_block}Begin."""
    
    def _prompt_memory_update(self, messages: List[Dict[str, str]],
                               forecast_interface, current_date: date) -> None:
        """
        Ask the agent to update its memory at the end of the day.

        For StructuredMemory: runs a mini action loop with memory tool calls.
        For ActiveMemory: single-shot with XML ops (existing behavior).
        For BasicMemory (legacy): uses full replacement via <memory> tags.
        """
        if isinstance(self._memory, ActiveMemory):
            memory_prompt = self._build_active_memory_prompt(current_date, day_qids=getattr(self, '_day_qids', None))
        elif isinstance(self._memory, StructuredMemory):
            memory_prompt = self._build_structured_memory_prompt(current_date)
        else:
            memory_prompt = self._build_plain_memory_prompt(current_date)

        # Prepend privileged cheat-feedback when enabled
        if self.config.cheat_feedback:
            cheat_data = forecast_interface.get_cheat_feedback(
                detail=self.config.cheat_feedback_detail
            )
            if cheat_data.get("items"):
                cheat_section = FeedbackHandler.format_cheat_feedback(
                    cheat_data, self.config.cheat_feedback_detail
                )
                memory_prompt = cheat_section + "\n\n" + memory_prompt

        # For StructuredMemory and ActiveMemory: run a mini action loop with memory tools
        if isinstance(self._memory, (StructuredMemory, ActiveMemory)):
            self._run_memory_update_loop(messages, forecast_interface, memory_prompt, current_date)
            return

        # For BasicMemory: single-shot (existing behavior)
        messages.append({"role": "user", "content": memory_prompt})
        response, usage = self.inference.chat(messages, self.config.sampling_params)
        self._timer.record_tokens(usage)
        self._timer.record_cost(usage.get("cost", 0), "llm")

        if not response:
            print(f"  [{self.agent_id}] Memory update got empty response from LLM, skipping.")
            return

        messages.append({"role": "assistant", "content": response})

        reasoning = usage.get("_reasoning_content") if usage else None

        forecast_interface.log_model_output(
            memory_prompt, response,
            {"phase": "memory_update", "current_memory_len": len(self._memory), "reasoning": reasoning}
        )

        new_memory = extract_memory(response)
        if new_memory is not None:
            self._memory.update(new_memory)

    def _run_memory_update_loop(self, messages: List[Dict[str, str]],
                                 forecast_interface, memory_prompt: str,
                                 current_date: date) -> None:
        """Run a mini action loop for structured memory updates with its own budget.

        NOTE: For StructuredMemory/ActiveMemory, the primary path now handles
        memory inside _run_action_loop (triggered by the `next` action or token
        threshold).  This method is retained as a fallback for BasicMemory
        within _prompt_memory_update.
        """
        mem_budget = BudgetTracker(BudgetSettings(
            max_total_tokens=self.config.memory_update_max_total_tokens,
            submit_reserve_tokens=self.config.submit_reserve_tokens,
            force_submit_threshold_tokens=self.config.force_submit_threshold_tokens,
        ))

        messages.append({"role": "user", "content": memory_prompt})
        # Bootstrap only from the memory prompt — NOT the full conversation.
        # The main loop already consumed its own budget; this mini-loop has an
        # independent token budget (memory_update_max_total_tokens).
        mem_budget.bootstrap_context(memory_prompt)

        while not mem_budget.is_exhausted():
            try:
                with self._timer.track("llm"):
                    response, usage = self.inference.chat(messages, self.config.sampling_params)
            except Exception as e:
                print(f"  [{self.agent_id}] Memory update LLM error: {e}")
                break

            if not response or not response.strip():
                print(f"  [{self.agent_id}] Memory update got empty response, ending.")
                break

            self._timer.record_tokens(usage)
            self._timer.record_cost(usage.get("cost", 0), "llm")
            mem_budget.record_usage(usage)

            reasoning = usage.get("_reasoning_content") if usage else None

            self._append_with_budget(messages, mem_budget, {"role": "assistant", "content": response})

            forecast_interface.log_model_output(
                "(memory_update_loop)", response,
                {"phase": "memory_update", "current_memory_entries": self._memory.entry_count, "reasoning": reasoning}
            )

            parsed = parse_action(response, self.config.max_outcomes_per_question)

            if parsed.action_type == "next":
                break

            # For ActiveMemory: also parse and apply mem_df ops from every response
            mem_ops_applied = 0
            if isinstance(self._memory, ActiveMemory):
                mem_adds, mem_updates, mem_deletes = extract_mem_ops(response)
                mem_ops_applied = len(mem_adds) + len(mem_updates) + len(mem_deletes)
                for qid in mem_deletes:
                    self._memory.mem_delete(qid)
                for add in mem_adds:
                    self._memory.mem_add(
                        qid=add["qid"], question=add.get("question", ""),
                        memory=add["memory"], category=add.get("category", ""),
                    )
                for upd in mem_updates:
                    self._memory.mem_update(
                        qid=upd["qid"], memory=upd["memory"],
                        category=upd.get("category"),
                    )

            if parsed.action_type in ("memory_retrieve", "memory_new", "memory_update", "memory_delete"):
                self._handle_memory_action(messages, forecast_interface, response, parsed, mem_budget, reasoning=reasoning)
            elif parsed.action_type in ("mem_add", "mem_update", "mem_delete"):
                self._handle_mem_action(messages, forecast_interface, response, parsed, mem_budget, reasoning=reasoning)
            elif mem_ops_applied > 0:
                # Mem XML ops applied but no action-type tag — give feedback
                feedback = f"MEM: Applied {mem_ops_applied} mem_df operations ({self._memory.mem_count} entries). Use <action type=\"next\"/> when done."
                self._append_with_budget(messages, mem_budget, {"role": "user", "content": mem_budget.format_feedback(feedback)})
            else:
                # Try legacy XML ops as fallback for meta-insights
                adds, deletes = extract_memory_ops(response)
                if adds or deletes:
                    for entry_name in deletes:
                        self._memory.delete_entry(entry_name)
                    for add in adds:
                        try:
                            self._memory.add_entry(
                                add["name"], add.get("description", add.get("name", "")), add.get("content", "")
                            )
                        except ValueError:
                            pass  # skip duplicates in legacy fallback
                    feedback = f"MEMORY: Applied {len(adds)} adds, {len(deletes)} deletes. Total entries: {self._memory.entry_count}."
                    self._append_with_budget(messages, mem_budget, {"role": "user", "content": mem_budget.format_feedback(feedback)})
                else:
                    # No recognized action — end the loop
                    break

        # Persist memory to disk
        self._memory._save(current_date)

    def _build_structured_memory_prompt(self, current_date: date) -> str:
        """Build the structured memory update prompt (tool-call-based)."""
        memory_index = self._memory.get_index()
        max_ent = self._memory._max_entries
        index_block = ""
        if memory_index:
            index_block = f"""<memory_index>
{memory_index}
</memory_index>
"""
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

### Memory tools (use one per turn):

Retrieve an existing entry: <action type="memory_retrieve">ENTRY_NAME</action>

Add a new entry:
<action type="memory_new">
<name>lowercase-hyphenated-name-with-qid (max 64 chars, a-z 0-9 hyphens only)</name>
<description>What it stores + when to use it (max 256 chars, include QIDs)</description>
<content>Full content (max 1024 chars). Include specific numbers, sources, reasoning.</content>
</action>

Update an existing entry:
<action type="memory_update" name="ENTRY_NAME">
<description>Updated description</description>
<content>Updated content</content>
</action>
Only include the fields you want to change.

Delete an existing entry: <action type="memory_delete">ENTRY_NAME</action>

Done: <action type="next"/>

Start with Step 1. If questions resolved this session, create lesson entries first."""

    def _build_plain_memory_prompt(self, current_date: date) -> str:
        """Build the legacy plain-text memory update prompt (full replacement)."""
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

To update memory, include it after your reasoning:
<reasoning>
Reflect on today's forecasting session...
</reasoning>
<memory>
Your updated memory content here (complete replacement, not a diff)
</memory>

If you don't want to update memory, just output <reasoning>...</reasoning> without <memory> tags.
Current memory length: {len(self._memory)} characters"""

    def _apply_structured_memory_ops(self, response: str) -> None:
        """Apply structured memory operations from agent response, with fallback.

        Kept for backward compatibility (ActiveMemory meta-insight path).
        The primary StructuredMemory path now uses ``_run_memory_update_loop``.
        """
        adds, deletes = extract_memory_ops(response)
        if adds or deletes:
            for entry_name in deletes:
                self._memory.delete_entry(entry_name)
            for add in adds:
                try:
                    self._memory.add_entry(
                        add["name"], add.get("description", add.get("name", "")), add.get("content", "")
                    )
                except ValueError:
                    pass  # skip duplicates
        else:
            # Fallback: try old-style <memory> full replacement
            old_memory = extract_memory(response)
            if old_memory is not None:
                self._memory.update(old_memory)

    # =========================================================================
    # Active Memory (mem_df + reduced meta-insights)
    # =========================================================================

    def _build_active_memory_prompt(self, current_date: date, day_qids: set = None) -> str:
        """Build the active memory update prompt (tool-call-based, like structured)."""
        mem_summary = self._memory.mem_summary(expanded_qids=day_qids)
        meta_index = self._memory.get_index()
        max_ent = self._memory._max_entries
        index_block = ""
        if meta_index:
            index_block = f"""<memory_index>
{meta_index}
</memory_index>
"""
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

### mem_df tools (can include multiple per turn):

Add:
<action type="mem_add">
<qid>Q123</qid>
<question>Will X happen by Y?</question>
<memory>Your reasoning and key evidence (max 1000 chars)</memory>
<category>politics</category>
</action>

Update:
<action type="mem_update" qid="Q123">
<memory>Updated reasoning (max 1000 chars)</memory>
</action>

Delete: <action type="mem_delete">Q123</action>

### Meta-insight tools (use one per turn):

Retrieve an existing entry: <action type="memory_retrieve">ENTRY_NAME</action>

Add a new entry:
<action type="memory_new">
<name>lowercase-hyphenated-name (max 64 chars, a-z 0-9 hyphens only)</name>
<description>What it stores + when to use it (max 256 chars, include QIDs)</description>
<content>Full content (max 400 chars). Cross-question patterns only.</content>
</action>

Update an existing entry:
<action type="memory_update" name="ENTRY_NAME">
<description>Updated description</description>
<content>Updated content</content>
</action>
Only include the fields you want to change.

Delete an existing entry: <action type="memory_delete">ENTRY_NAME</action>

Done: <action type="next"/>

Start with Step 1. If questions resolved this session, create lesson entries first. Use <action type="next"/> when done."""
