"""BasicAgent configuration."""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class AgentConfig:
    max_actions: Optional[int] = None
    warmup_max_actions: Optional[int] = None
    max_total_tokens: Optional[int] = None
    warmup_max_total_tokens: Optional[int] = None
    submit_reserve_tokens: int = 8192
    warmup_submit_reserve_tokens: Optional[int] = None
    force_submit_threshold_tokens: int = 16384
    warmup_force_submit_threshold_tokens: Optional[int] = None
    warmup_parallelism: int = 20  # Default higher parallelism for warmup
    max_submit_retries: int = 3
    max_outcomes_per_question: int = 5
    # Legacy OG-style single-answer format; not supported by the tools-only Basic/AllQ scaffolds.
    singleans: bool = False
    memory_dir: Optional[str] = None
    enable_memory: bool = True
    memory_format: str = "structured"  # "structured" (YAML entries), "plain" (legacy text), or "active" (mem_df + meta-insights)
    memory_max_entries: int = 500  # Max number of structured memory entries
    memory_update_max_total_tokens: int = 50000  # Token budget for end-of-day memory mini-loop
    content_filter_circuit_breaker: int = 5  # Break out of loop after N consecutive content_filter responses
    append_model_output_logs: bool = False
    sampling_params: Optional[Dict[str, Any]] = None
    
    # Search
    search_enabled: bool = False
    max_search_results: int = 5
    snippet_max_chars: int = 2000
    article_max_chars: int = 4000
    search_cutoff_days: int = 0
    resolution_guard: Optional[int] = None
    timegap_days: int = 1
    # Keep only the K most recent user/tool-result messages when replaying the
    # conversation to the model. -1 keeps all results.
    tool_result_keep_last: int = -1
    
    # Allow the model to emit multiple tool calls per turn.
    parallel_tool_calls: bool = False

    # Single agent mode - adjusts prompt to focus on accuracy only (no peer/market language)
    single_agent_mode: bool = False

    # GPT-OSS Harmony prompt placement:
    # - "instructions" (default): task text goes to Responses API `instructions`.
    # - "first_user": task text is prepended to the first user message.
    gptoss_prompt_mode: str = "instructions"

    # GPT-OSS reasoning controls for Responses API.
    # reasoning_effort: "low" | "medium" | "high"
    gptoss_reasoning_effort: str = "medium"
    gptoss_include_reasoning: bool = True
    # Replay policy for GPT-OSS tool turns:
    # - "sanitized": rebuild assistant/tool transcript from parsed fields
    # - "raw_recommended": replay only reasoning/function_call/function_call_output items
    gptoss_replay_mode: str = "raw_recommended"
    # Reserve output budget for reasoning+visible tokens on GPT-OSS Responses calls.
    # 0 disables clamping; otherwise max_output_tokens is raised to at least this value.
    gptoss_min_max_output_tokens: int = 25000
    # GPT-OSS Responses retry controls.
    gptoss_responses_max_retries: int = 3
    gptoss_retry_backoff_base_s: float = 1.0
    gptoss_retry_backoff_max_s: float = 16.0
    
    def __post_init__(self):
        if self.sampling_params is None:
            self.sampling_params = {'temperature': 0.7, 'max_tokens': 2048}
        for name, value in (
            ("max_actions", self.max_actions),
            ("warmup_max_actions", self.warmup_max_actions),
            ("max_total_tokens", self.max_total_tokens),
            ("warmup_max_total_tokens", self.warmup_max_total_tokens),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be > 0 when provided")
        for name, value in (
            ("submit_reserve_tokens", self.submit_reserve_tokens),
            ("warmup_submit_reserve_tokens", self.warmup_submit_reserve_tokens),
            ("force_submit_threshold_tokens", self.force_submit_threshold_tokens),
            ("warmup_force_submit_threshold_tokens", self.warmup_force_submit_threshold_tokens),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be >= 0 when provided")

        if self.tool_result_keep_last < -1:
            raise ValueError("tool_result_keep_last must be >= -1")

        if self.resolution_guard is not None and self.resolution_guard < 0:
            raise ValueError("resolution_guard must be >= 0 when provided")

        if self.force_submit_threshold_tokens < self.submit_reserve_tokens:
            raise ValueError("force_submit_threshold_tokens must be >= submit_reserve_tokens")

        warmup_submit_reserve = (
            self.warmup_submit_reserve_tokens
            if self.warmup_submit_reserve_tokens is not None
            else self.submit_reserve_tokens
        )
        warmup_force_submit = (
            self.warmup_force_submit_threshold_tokens
            if self.warmup_force_submit_threshold_tokens is not None
            else self.force_submit_threshold_tokens
        )
        if warmup_force_submit < warmup_submit_reserve:
            raise ValueError(
                "warmup_force_submit_threshold_tokens must be >= warmup_submit_reserve_tokens"
            )

        if self.timegap_days <= 0:
            raise ValueError("timegap_days must be >= 1")
