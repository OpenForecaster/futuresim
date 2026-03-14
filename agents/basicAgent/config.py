"""BasicAgent configuration."""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class AgentConfig:
    max_actions: int = 10
    warmup_max_actions: int = 10
    warmup_parallelism: int = 20  # Default higher parallelism for warmup
    max_submit_retries: int = 3
    max_outcomes_per_question: int = 5
    # If True, agents are expected to output <answer>...</answer><probability>...</probability>
    # instead of <action type="submit"> XML. Currently supported for per-question warmup loops.
    singleans: bool = False
    memory_dir: Optional[str] = None
    enable_memory: bool = True
    memory_format: str = "structured"  # "structured" (YAML entries) or "plain" (legacy text)
    sampling_params: Optional[Dict[str, Any]] = None
    
    # Search
    search_enabled: bool = False
    max_search_results: int = 5
    snippet_max_chars: int = 2000
    article_max_chars: int = 4000
    search_cutoff_days: int = 0
    
    # Single agent mode - adjusts prompt to focus on accuracy only (no peer/market language)
    single_agent_mode: bool = False

    # Cheat feedback: provide privileged directional-correctness feedback before memory update
    cheat_feedback: bool = False
    cheat_feedback_detail: str = "full"  # "full" (show Brier scores) or "direction" (only improved/worsened)

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
