"""Shared budget tracking for agent action loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class BudgetSettings:
    """Configuration for an action loop budget."""

    max_actions: Optional[int] = None
    max_total_tokens: Optional[int] = None
    submit_reserve_tokens: int = 8192
    force_submit_threshold_tokens: int = 16384
    # When set, the unified loop transitions to memory-update mode once
    # token_budget_remaining() drops to this value.
    memory_phase_threshold_tokens: Optional[int] = None


class BudgetTracker:
    """Tracks optional action and token budgets for a single loop."""

    def __init__(
        self,
        settings: BudgetSettings,
        token_estimator: Optional[Callable[[Any], int]] = None,
    ):
        self.settings = settings
        self.actions_remaining = settings.max_actions
        self.current_context_tokens = 0
        self.cached_prompt_tokens = 0
        self.memory_phase: bool = False
        self._token_estimator = token_estimator

    def bootstrap_context(self, payload: Any) -> int:
        """Seed the context estimate before the first model call in a loop."""
        if self.settings.max_total_tokens is None:
            return self.current_context_tokens
        self.current_context_tokens = self._estimate_tokens(payload)
        return self.current_context_tokens

    def record_usage(self, usage: Optional[Dict[str, Any]]) -> int:
        """Anchor the context estimate to exact prompt tokens from a model call."""
        if self.settings.max_total_tokens is None or not usage:
            return self.current_context_tokens

        prompt_tokens = usage.get("prompt_tokens")
        if prompt_tokens is None:
            prompt_tokens = usage.get("input_tokens", 0)
        self.current_context_tokens = int(prompt_tokens or 0)

        prompt_details = usage.get("prompt_tokens_details")
        if isinstance(prompt_details, dict):
            self.cached_prompt_tokens = int(prompt_details.get("cached_tokens", 0) or 0)
        else:
            self.cached_prompt_tokens = int(usage.get("cached_tokens", 0) or 0)

        return self.current_context_tokens

    def record_appended_item(self, item: Any) -> int:
        """Estimate newly appended transcript content after a model call."""
        if self.settings.max_total_tokens is None:
            return self.current_context_tokens
        self.current_context_tokens += self._estimate_tokens(item)
        return self.current_context_tokens

    def consume_action(self) -> None:
        """Consume one action if an action budget is active."""
        if self.actions_remaining is None:
            return
        self.actions_remaining -= 1

    def token_budget_remaining(self) -> Optional[int]:
        if self.settings.max_total_tokens is None:
            return None
        return self.settings.max_total_tokens - self.current_context_tokens

    # ------------------------------------------------------------------
    # Memory-phase helpers
    # ------------------------------------------------------------------

    def tokens_until_memory_phase(self) -> Optional[int]:
        """Tokens remaining before the memory phase kicks in. None if not configured."""
        if self.settings.memory_phase_threshold_tokens is None:
            return None
        remaining = self.token_budget_remaining()
        if remaining is None:
            return None
        return max(0, remaining - self.settings.memory_phase_threshold_tokens)

    def should_enter_memory_phase(self) -> bool:
        """True when remaining tokens have dropped to the memory-phase threshold."""
        if self.memory_phase:
            return False
        until = self.tokens_until_memory_phase()
        return until is not None and until <= 0

    # ------------------------------------------------------------------
    # Exhaustion / force-submit
    # ------------------------------------------------------------------

    def is_exhausted(self) -> bool:
        if self.actions_remaining is not None and self.actions_remaining <= 0:
            return True
        remaining_tokens = self.token_budget_remaining()
        if remaining_tokens is not None and remaining_tokens < self.settings.submit_reserve_tokens:
            return True
        return False

    def should_force_submit(self) -> bool:
        if self.is_exhausted():
            return False
        if self.actions_remaining is not None and self.actions_remaining <= 1:
            return True
        remaining_tokens = self.token_budget_remaining()
        if remaining_tokens is not None and remaining_tokens <= self.settings.force_submit_threshold_tokens:
            return True
        return False

    # ------------------------------------------------------------------
    # Status / feedback
    # ------------------------------------------------------------------

    def status_text(self, *, include_exhaustion_warning: bool = False) -> str:
        lines: List[str] = []
        if self.actions_remaining is not None:
            lines.append(f"Actions remaining: {max(self.actions_remaining, 0)}")
        remaining_tokens = self.token_budget_remaining()
        if remaining_tokens is not None and self.settings.max_total_tokens is not None:
            lines.append(
                "Context tokens remaining: "
                f"{max(remaining_tokens, 0)} "
                f"(estimated current context {max(self.current_context_tokens, 0)} / {self.settings.max_total_tokens})"
            )
        # Show tokens-until-memory-phase only during the action phase
        if not self.memory_phase:
            until_mem = self.tokens_until_memory_phase()
            if until_mem is not None:
                lines.append(f"Tokens remaining until memory phase: {until_mem}")
        if include_exhaustion_warning and self.is_exhausted():
            lines.append("No more budget available. Your day ends now.")
        return "\n".join(lines)

    def format_feedback(self, feedback: str, *, include_exhaustion_warning: bool = True) -> str:
        status = self.status_text(include_exhaustion_warning=include_exhaustion_warning)
        if not status:
            return feedback
        if feedback.endswith("\n"):
            return f"{feedback}{status}"
        return f"{feedback}\n\n{status}"

    def metadata(self) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "budget_force_submit": self.should_force_submit(),
            "budget_exhausted": self.is_exhausted(),
        }
        if self.settings.max_actions is not None:
            metadata["actions_remaining"] = self.actions_remaining
        if self.settings.max_total_tokens is not None:
            metadata["token_budget_used"] = self.current_context_tokens
            metadata["token_budget_remaining"] = self.token_budget_remaining()
            metadata["prompt_cached_tokens"] = self.cached_prompt_tokens
        if self.settings.memory_phase_threshold_tokens is not None:
            metadata["tokens_until_memory_phase"] = self.tokens_until_memory_phase()
            metadata["in_memory_phase"] = self.memory_phase
        return metadata

    def _estimate_tokens(self, item: Any) -> int:
        if item is None:
            return 0
        if self._token_estimator is not None:
            return max(0, int(self._token_estimator(item) or 0))
        text = item if isinstance(item, str) else str(item)
        if not text:
            return 0
        return max(1, (len(text) + 3) // 4)
