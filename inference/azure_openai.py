"""
Azure OpenAI inference provider (via openai SDK).

Drop-in replacement for OpenRouterInference - same chat() interface.
Features:
- Automatic retries with exponential backoff
- Global rate limiting (shared across all instances)
- Compatible sampling_params interface

Credentials (in priority order — first match wins):
    1. Constructor args  (api_key=, endpoint=, model=)
    2. Environment vars  AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_DEPLOYMENT
    3. configs/azure_openai_config.py  (AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT)
"""

from __future__ import annotations

import os
import sys
import time
import random
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import OpenAI, AzureOpenAI, APIStatusError, APIConnectionError, APITimeoutError
except ImportError:
    raise ImportError("openai package not found. Install with: pip install openai")

# Try to import credentials from config file
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from configs.azure_openai_config import (
        AZURE_OPENAI_API_KEY    as CONFIG_API_KEY,
        AZURE_OPENAI_ENDPOINT   as CONFIG_ENDPOINT,
        AZURE_OPENAI_DEPLOYMENT as CONFIG_DEPLOYMENT,
    )
except ImportError:
    CONFIG_API_KEY    = None
    CONFIG_ENDPOINT   = None
    CONFIG_DEPLOYMENT = None


API_VERSION = "2024-12-01-preview"

# ---------------------------------------------------------------------------
# Rate limiter (identical singleton pattern to openrouter.py)
# ---------------------------------------------------------------------------

class GlobalRateLimiter:
    """Thread-safe token-bucket rate limiter shared across all instances."""

    _instance: Optional["GlobalRateLimiter"] = None
    _lock = Lock()

    def __new__(cls, requests_per_second: float = 32.0):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, requests_per_second: float = 32.0):
        if self._initialized:
            return
        self._initialized = True
        self.requests_per_second = requests_per_second
        self.interval = 1.0 / requests_per_second
        self.last_request = 0.0
        self._acquire_lock = Lock()

    def acquire(self):
        with self._acquire_lock:
            now = time.time()
            wait_time = self.last_request + self.interval - now
            if wait_time > 0:
                time.sleep(wait_time)
            self.last_request = time.time()

    @classmethod
    def configure(cls, requests_per_second: float):
        with cls._lock:
            if cls._instance is not None:
                cls._instance.requests_per_second = requests_per_second
                cls._instance.interval = 1.0 / requests_per_second
            else:
                instance = super().__new__(cls)
                instance._initialized = False
                instance.requests_per_second = requests_per_second
                instance.interval = 1.0 / requests_per_second
                instance.last_request = 0.0
                instance._acquire_lock = Lock()
                instance._initialized = True
                cls._instance = instance


# ---------------------------------------------------------------------------
# Main inference class
# ---------------------------------------------------------------------------

class AzureOpenAIInference:
    """
    Azure-hosted OpenAI-compatible inference provider.

    Usage:
        inference = AzureOpenAIInference(
            model="DeepSeek-V3.2",
            endpoint="https://<resource>.services.ai.azure.com/openai/v1/",
            api_key="...",
        )
        text, usage = inference.chat(messages, {"temperature": 0.7, "max_tokens": 2048})
    """

    def __init__(
        self,
        model: str,
        endpoint: str = None,
        api_key: str = None,
        max_retries: int = 3,
        base_delay: float = 10.0,
        max_delay: float = 60.0,
        **kwargs,
    ):
        """
        Args:
            model:       Deployment / model name (e.g. "DeepSeek-V3.2")
            endpoint:    Azure endpoint URL. Falls back to AZURE_OPENAI_ENDPOINT env var.
            api_key:     API key. Falls back to AZURE_OPENAI_API_KEY env var.
            max_retries: Maximum retry attempts on transient errors.
            base_delay:  Base delay (s) for exponential backoff.
            max_delay:   Maximum backoff cap (s).
            **kwargs:    Extra default parameters forwarded to every completion call.
        """
        # If model looks like an OpenRouter-style name (e.g. "openai/gpt-5.2"), it is
        # not a valid Azure deployment name — fall back to the configured deployment.
        _raw_model = model or os.environ.get("AZURE_OPENAI_DEPLOYMENT") or CONFIG_DEPLOYMENT
        if _raw_model and "/" in _raw_model:
            _raw_model = os.environ.get("AZURE_OPENAI_DEPLOYMENT") or CONFIG_DEPLOYMENT or _raw_model
        self.model = _raw_model
        print(f"Model: {self.model}")
        self.model_name = self.model  # compatibility with VLLM interface

        self.endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT") or CONFIG_ENDPOINT
        if not self.endpoint:
            raise ValueError(
                "Azure endpoint required. Pass endpoint=, set AZURE_OPENAI_ENDPOINT, "
                "or add to configs/azure_openai_config.py"
            )

        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY") or CONFIG_API_KEY
        if not self.api_key:
            raise ValueError(
                "Azure API key required. Pass api_key=, set AZURE_OPENAI_API_KEY, "
                "or add to configs/azure_openai_config.py"
            )

        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.default_kwargs = kwargs

        # self._client = OpenAI(
        #     base_url=self.endpoint,
        #     api_key=self.api_key,
        # )
        self._client = AzureOpenAI(
            api_version=API_VERSION,
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
        )

        GlobalRateLimiter()  # ensure singleton is initialised

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        sampling_params: Dict[str, Any],
    ) -> Tuple[str, Dict]:
        """
        Chat completion.

        Args:
            messages:        List of {"role": ..., "content": ...} dicts.
            sampling_params: Supported keys: temperature, max_tokens, top_p,
                             frequency_penalty, presence_penalty, stop, reasoning.

        Returns:
            (response_text, usage_dict)
        """
        kwargs: Dict[str, Any] = {}

        for key in ("temperature", "max_tokens", "top_p",
                    "frequency_penalty", "presence_penalty", "stop"):
            if key in sampling_params:
                kwargs[key] = sampling_params[key]

        # # Reasoning effort: OpenRouter uses {"reasoning": {"effort": "high"}} but the
        # # standard OpenAI SDK uses the top-level "reasoning_effort" param (o-series only).
        # # Map it here; non-o-series models will simply ignore an unsupported effort level
        # # rather than raising a local TypeError.
        # reasoning_cfg = sampling_params.get("reasoning")
        # if reasoning_cfg is not None:
        #     effort = reasoning_cfg if isinstance(reasoning_cfg, str) else reasoning_cfg.get("effort")
        #     if effort:
        #         kwargs["reasoning_effort"] = effort

        # Merge caller-level defaults (lowest priority)
        for k, v in self.default_kwargs.items():
            kwargs.setdefault(k, v)

        return self._request_with_retry(messages, kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request_with_retry(
        self,
        messages: List[Dict[str, str]],
        kwargs: Dict[str, Any],
    ) -> Tuple[str, Dict]:
        RETRYABLE_STATUS = {429, 500, 502, 503, 504, 524}
        last_error: Optional[BaseException] = None
        rate_limiter = GlobalRateLimiter()

        for attempt in range(self.max_retries + 1):
            try:
                rate_limiter.acquire()

                completion = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    # timeout=120,
                    # **kwargs,
                    max_completion_tokens=16384,
                )

                choice = completion.choices[0]
                content = choice.message.content or ""
                usage: Dict[str, Any] = {}
                if completion.usage:
                    usage = {
                        "prompt_tokens": completion.usage.prompt_tokens,
                        "completion_tokens": completion.usage.completion_tokens,
                        "total_tokens": completion.usage.total_tokens,
                    }

                # Extract reasoning tokens if present (e.g. DeepSeek-R1 style)
                reasoning = getattr(choice.message, "reasoning_content", None)
                if reasoning:
                    usage["_reasoning_content"] = reasoning

                finish_reason = choice.finish_reason
                if finish_reason:
                    usage["_finish_reason"] = finish_reason

                # Retry if content is empty but reasoning exists
                if (not content or not content.strip()) and reasoning:
                    last_error = Exception(
                        f"Empty content with reasoning ({len(reasoning)} chars), "
                        f"finish_reason={finish_reason}"
                    )
                    if attempt < self.max_retries:
                        delay = self._backoff(attempt)
                        print(
                            f"  [AzureOpenAI] Empty content (reasoning={len(reasoning)}c, "
                            f"finish={finish_reason}), retrying in {delay:.1f}s "
                            f"(attempt {attempt + 1}/{self.max_retries})"
                        )
                        time.sleep(delay)
                        continue
                    print(
                        f"  [AzureOpenAI] Empty content persisted after {self.max_retries} retries, "
                        f"using reasoning as content ({len(reasoning)} chars)"
                    )
                    return reasoning, usage

                if not content or not content.strip():
                    comp_tokens = usage.get("completion_tokens", "?")
                    print(
                        f"  [AzureOpenAI] Warning: empty content "
                        f"(finish={finish_reason}, tokens={comp_tokens})"
                    )

                return content, usage

            except APIStatusError as e:
                last_error = e
                if e.status_code in RETRYABLE_STATUS and attempt < self.max_retries:
                    delay = self._backoff(attempt)
                    print(
                        f"  [AzureOpenAI] HTTP {e.status_code}, retrying in "
                        f"{delay:.1f}s (attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(delay)
                    continue
                if e.status_code in (401, 403):
                    raise ValueError(
                        f"AzureOpenAI auth error (HTTP {e.status_code}). Check API key / permissions."
                    ) from e
                if e.status_code in (400, 402):
                    raise ValueError(
                        f"AzureOpenAI request error (HTTP {e.status_code}). Check model/params/billing."
                    ) from e
                raise

            except (APIConnectionError, APITimeoutError) as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = self._backoff(attempt)
                    print(
                        f"  [AzureOpenAI] {type(e).__name__}, retrying in "
                        f"{delay:.1f}s (attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(delay)
                    continue

        print(
            f"  [AzureOpenAI] Request failed after {self.max_retries} retries: "
            f"{last_error}. Returning empty output."
        )
        return "", {}

    def _backoff(self, attempt: int) -> float:
        delay = min(self.max_delay, self.base_delay * (2 ** attempt))
        return delay * random.uniform(0.75, 1.25)
