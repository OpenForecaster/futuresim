"""Shared budget tracking for agent action loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class BudgetSettings:
    """Configuration for an action loop budget."""

    max_actions: Optional[int] = None
    max_total_tokens: Optional[int] = None
    submit_reserve_tokens: int = 4096
    force_submit_threshold_tokens: int = 8192


class BudgetTracker:
    """Tracks optional action and token budgets for a single loop."""

    def __init__(self, settings: BudgetSettings):
        self.settings = settings
        self.actions_remaining = settings.max_actions
        self.total_tokens_used = 0

    def record_usage(self, usage: Optional[Dict[str, Any]]) -> int:
        """Consume total token usage for a completed model call."""
        if self.settings.max_total_tokens is None or not usage:
            return self.total_tokens_used

        total_tokens = usage.get("total_tokens")
        if total_tokens is None:
            total_tokens = int(usage.get("prompt_tokens", 0) or 0) + int(
                usage.get("completion_tokens", 0) or 0
            )
        self.total_tokens_used += int(total_tokens or 0)
        return self.total_tokens_used

    def consume_action(self) -> None:
        """Consume one action if an action budget is active."""
        if self.actions_remaining is None:
            return
        self.actions_remaining -= 1

    def token_budget_remaining(self) -> Optional[int]:
        if self.settings.max_total_tokens is None:
            return None
        return self.settings.max_total_tokens - self.total_tokens_used

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

    def status_text(self, *, include_exhaustion_warning: bool = False) -> str:
        lines: List[str] = []
        if self.actions_remaining is not None:
            lines.append(f"Actions remaining: {max(self.actions_remaining, 0)}")
        remaining_tokens = self.token_budget_remaining()
        if remaining_tokens is not None and self.settings.max_total_tokens is not None:
            lines.append(
                "Token budget remaining: "
                f"{max(remaining_tokens, 0)} (used {self.total_tokens_used} / {self.settings.max_total_tokens})"
            )
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
            metadata["token_budget_used"] = self.total_tokens_used
            metadata["token_budget_remaining"] = self.token_budget_remaining()
        return metadata
