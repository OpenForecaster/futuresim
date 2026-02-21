"""
OpenRouter inference provider.

Drop-in replacement for VLLMInference - use chat() method with same interface.
Features:
- Automatic retries with exponential backoff
- Prompt caching support (automatic for most providers)
- Global rate limiting (shared across all instances)
- Session reuse for connection pooling
- Compatible sampling_params interface
"""

import os
import time
import random
from threading import Lock
from typing import Dict, Any, List, Optional, Tuple

try:
    import requests
except ImportError:
    raise ImportError(
        "requests module not found. Install with: pip install requests"
    )


class GlobalRateLimiter:
    """
    Thread-safe rate limiter shared across all OpenRouter instances.
    Uses token bucket algorithm.
    """
    _instance: Optional['GlobalRateLimiter'] = None
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
        """Block until a request slot is available."""
        with self._acquire_lock:
            now = time.time()
            wait_time = self.last_request + self.interval - now
            if wait_time > 0:
                time.sleep(wait_time)
            self.last_request = time.time()
    
    @classmethod
    def configure(cls, requests_per_second: float):
        """Reconfigure the global rate limiter."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.requests_per_second = requests_per_second
                cls._instance.interval = 1.0 / requests_per_second
            else:
                # Create instance directly without calling __new__ again (would deadlock)
                instance = super().__new__(cls)
                instance._initialized = False
                instance.requests_per_second = requests_per_second
                instance.interval = 1.0 / requests_per_second
                instance.last_request = 0.0
                instance._acquire_lock = Lock()
                instance._initialized = True
                cls._instance = instance


# Global session for connection pooling
_session: Optional[requests.Session] = None
_session_lock = Lock()

_pool_connections: int = 10
_pool_maxsize: int = 20


def configure_http_pool(pool_maxsize: int, pool_connections: Optional[int] = None) -> None:
    """
    Configure the shared HTTP connection pool used by all OpenRouterInference instances.

    This should be called once at startup (before high-parallelism runs) to avoid
    urllib3 warnings like "Connection pool is full" under ThreadPool workloads.
    """
    global _pool_connections, _pool_maxsize, _session

    pool_maxsize = int(pool_maxsize)
    if pool_maxsize <= 0:
        raise ValueError("pool_maxsize must be > 0")

    if pool_connections is None:
        pool_connections = pool_maxsize
    pool_connections = int(pool_connections)
    if pool_connections <= 0:
        raise ValueError("pool_connections must be > 0")

    with _session_lock:
        _pool_connections = pool_connections
        _pool_maxsize = pool_maxsize
        if _session is not None:
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=_pool_connections,
                pool_maxsize=_pool_maxsize,
                max_retries=0,  # We handle retries ourselves
            )
            _session.mount('https://', adapter)


def get_session() -> requests.Session:
    """Get or create the shared requests session."""
    global _session
    with _session_lock:
        if _session is None:
            _session = requests.Session()
            # Configure connection pooling
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=_pool_connections,
                pool_maxsize=_pool_maxsize,
                max_retries=0  # We handle retries ourselves
            )
            _session.mount('https://', adapter)
        return _session


class OpenRouterInference:
    """
    OpenRouter API inference provider.
    
    Environment variable required:
        OPENROUTER_API_KEY: Your OpenRouter API key
    
    Usage:
        inference = OpenRouterInference("xiaomi/mimo-v2-flash:free")
        response = inference.chat(messages, {"temperature": 0.7, "max_tokens": 2048})
    """
    
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    
    def __init__(self, 
                 model: str,
                 api_key: str = None,
                 max_retries: int = 3,
                 base_delay: float = 10.0,
                 max_delay: float = 60.0,
                 enable_caching: bool = True,
                 **kwargs):
        """
        Initialize OpenRouter inference provider.
        
        Args:
            model: Model identifier (e.g., "xiaomi/mimo-v2-flash:free")
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            max_retries: Maximum retry attempts on transient errors (default 3)
            base_delay: Base delay in seconds for exponential backoff (default 1.0)
            max_delay: Maximum delay cap in seconds (default 30.0)
            enable_caching: Enable prompt caching routing (default True)
            **kwargs: Additional default parameters for requests
        """
        self.model = model
        self.model_name = model  # For compatibility with VLLM interface
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.enable_caching = enable_caching
        self.default_kwargs = kwargs
        
        # Request headers
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/forecast-sim",  # Optional: for rankings
        }
        
        # Ensure rate limiter is initialized
        GlobalRateLimiter()
    
    def chat(self, messages: List[Dict[str, str]], sampling_params: Dict[str, Any]) -> Tuple[str, Dict]:
        """
        Chat completion using messages format.
        
        Args:
            messages: List of {"role": "system"|"user"|"assistant", "content": "..."}
            sampling_params: Dict with temperature, max_tokens, etc.
                Supported keys: temperature, max_tokens, top_p, top_k, 
                frequency_penalty, presence_penalty, stop
        
        Returns:
            The assistant's response text.
        """
        # Build request payload
        payload = {
            "model": self.model,
            "messages": messages,
            "usage": {"include": True},
        }
        
        # Map sampling_params to OpenRouter parameters
        param_mapping = {
            "temperature": "temperature",
            "max_tokens": "max_tokens",
            "top_p": "top_p",
            "top_k": "top_k",
            "frequency_penalty": "frequency_penalty",
            "presence_penalty": "presence_penalty",
            "stop": "stop",
        }
        
        for local_key, api_key in param_mapping.items():
            if local_key in sampling_params:
                payload[api_key] = sampling_params[local_key]
        
        # Add any default kwargs
        for key, value in self.default_kwargs.items():
            if key not in payload:
                payload[key] = value
        
        # Make request with retries
        return self._request_with_retry(payload)
    
    def _request_with_retry(self, payload: Dict[str, Any]) -> Tuple[str, Dict]:
        """
        Make API request with exponential backoff retry logic.
        
        Retries on:
            - HTTP 429 (rate limit)
            - HTTP 500, 502, 503, 504, 524 (server errors)
            - Connection errors
            - JSON decode errors (malformed responses)
        """
        RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504, 524}
        last_error: Optional[BaseException] = None
        rate_limiter = GlobalRateLimiter()
        session = get_session()
        
        for attempt in range(self.max_retries + 1):
            try:
                # Apply rate limiting
                rate_limiter.acquire()
                
                response = session.post(
                    self.API_URL,
                    headers=self.headers,
                    json=payload,
                    timeout=120,  # 2 minute timeout for long generations
                )
                
                # Success
                if response.status_code == 200:
                    # Handle malformed JSON responses
                    try:
                        data = response.json()
                    except Exception as e:
                        last_error = e
                        if attempt < self.max_retries:
                            delay = self._calculate_backoff(attempt)
                            print(f"  [OpenRouter] Malformed JSON response, retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                            time.sleep(delay)
                            continue
                        # Exhausted retries
                        print(f"  [OpenRouter] Malformed JSON after {self.max_retries} retries. Returning empty output.")
                        return "", {}
                    
                    # Check for provider errors returned as 200 (e.g., Xiaomi 524 timeout)
                    if "error" in data and "choices" not in data:
                        error_detail = data.get("error", data)
                        error_code = error_detail.get("code", 0) if isinstance(error_detail, dict) else 0
                        
                        # Retry on provider timeouts/errors
                        if error_code in (524, 500, 502, 503, 504) or "timeout" in str(error_detail).lower():
                            last_error = Exception(f"Provider error: {error_detail}")
                            if attempt < self.max_retries:
                                delay = self._calculate_backoff(attempt)
                                print(f"  [OpenRouter] Provider error {error_code}, retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                                time.sleep(delay)
                                continue
                        
                        raise RuntimeError(f"OpenRouter returned error: {error_detail}")
                    
                    # Check for valid response structure
                    if "choices" not in data:
                        error_detail = data.get("error", data)
                        raise RuntimeError(f"OpenRouter returned invalid response: {error_detail}")
                    
                    content = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})

                    # Extract reasoning if available
                    reasoning = data["choices"][0]["message"].get("reasoning")
                    if reasoning:
                        usage["_reasoning_content"] = reasoning

                    finish_reason = data["choices"][0].get("finish_reason")
                    if finish_reason:
                        usage["_finish_reason"] = finish_reason

                    # Treat null/empty content as retryable when reasoning exists
                    if (not content or not content.strip()) and reasoning:
                        last_error = Exception(
                            f"Empty content with reasoning ({len(reasoning)} chars), finish_reason={finish_reason}"
                        )
                        if attempt < self.max_retries:
                            delay = self._calculate_backoff(attempt)
                            print(
                                f"  [OpenRouter] Empty content (reasoning={len(reasoning)}c, "
                                f"finish={finish_reason}), retrying in {delay:.1f}s "
                                f"(attempt {attempt + 1}/{self.max_retries})"
                            )
                            time.sleep(delay)
                            continue
                        # Exhausted retries — fall through with the reasoning as content
                        print(
                            f"  [OpenRouter] Empty content persisted after {self.max_retries} retries, "
                            f"using reasoning as content ({len(reasoning)} chars)"
                        )
                        return reasoning, usage

                    # Log empty content without reasoning (unexpected)
                    if not content or not content.strip():
                        comp_tokens = usage.get("completion_tokens", "?")
                        print(
                            f"  [OpenRouter] Warning: empty content "
                            f"(finish={finish_reason}, tokens={comp_tokens}, "
                            f"reasoning={'yes' if reasoning else 'no'})"
                        )

                    return content, usage
                
                # Rate limit or server error - retry (including 524 Cloudflare timeout)
                if response.status_code in RETRYABLE_STATUS_CODES:
                    error_msg = f"HTTP {response.status_code}"
                    try:
                        error_data = response.json()
                        if "error" in error_data:
                            error_msg = f"{error_msg}: {error_data['error']}"
                    except:
                        pass
                    
                    last_error = Exception(error_msg)
                    
                    if attempt < self.max_retries:
                        delay = self._calculate_backoff(attempt)
                        print(f"  [OpenRouter] {error_msg}, retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                        time.sleep(delay)
                        continue
                
                # Other HTTP errors: treat as fatal configuration/usage errors.
                # We intentionally do NOT retry these.
                if response.status_code in (401, 403):
                    raise ValueError(f"OpenRouter auth error (HTTP {response.status_code}). Check OPENROUTER_API_KEY / permissions.")
                if response.status_code in (400, 402):
                    raise ValueError(f"OpenRouter request error (HTTP {response.status_code}). Check model/params/billing.")

                # Best-effort message for unknown status codes.
                try:
                    body = response.text[:500]
                except Exception:
                    body = "<unreadable body>"
                raise RuntimeError(f"OpenRouter HTTP {response.status_code}: {body}")
                
            except requests.exceptions.RequestException as e:
                # Broad catch for transient network/proxy failures:
                # - Timeout / ReadTimeout
                # - ConnectionError
                # - ChunkedEncodingError ("Response ended prematurely")
                # - etc.
                #
                # NOTE: We avoid using raise_for_status() so HTTPError isn't expected here.
                last_error = e
                if attempt < self.max_retries:
                    delay = self._calculate_backoff(attempt)
                    print(f"  [OpenRouter] {type(e).__name__}, retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(delay)
                    continue
        
        # All retries exhausted - return empty output to allow simulation to continue
        print(f"  [OpenRouter] Request failed after {self.max_retries} retries: {last_error}. Returning empty output.")
        return "", {}
    
    def _calculate_backoff(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay with jitter.
        
        Formula: min(max_delay, base_delay * 2^attempt) * (0.75 to 1.25 jitter)
        """
        delay = min(self.max_delay, self.base_delay * (2 ** attempt))
        # Add ±25% jitter to prevent thundering herd
        jitter = random.uniform(0.75, 1.25)
        return delay * jitter
