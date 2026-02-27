"""
VLLM inference provider.

Uses VLLM OpenAI-compatible server internally to avoid GPU contention issues.
The server is started automatically on first use and cleaned up on exit.
"""

from typing import Dict, Any, List, Tuple, Optional
import os
import time
import atexit
import subprocess
import socket
import re
import sys

try:
    import requests
except ImportError:
    raise ImportError("requests module not found. Install with: pip install requests")


# Global server management
_VLLM_SERVERS: Dict[str, dict] = {}  # port -> {process, model_path}
_NEXT_PORT = 8001


def _find_free_port(start_port: int = 8001) -> int:
    """Find a free port starting from start_port."""
    port = start_port
    while port < 65535:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except OSError:
                port += 1
    raise RuntimeError("No free ports available")


def _cleanup_servers():
    """Cleanup all VLLM servers on exit."""
    for port, info in list(_VLLM_SERVERS.items()):
        proc = info.get('process')
        if proc and proc.poll() is None:
            print(f"Shutting down VLLM server on port {port}...")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


atexit.register(_cleanup_servers)


class VLLMInference:
    """VLLM inference using OpenAI-compatible server internally."""
    
    @staticmethod
    def _is_large_gpu() -> bool:
        """Detect if current GPU has >100GB total memory (e.g. B200).
        
        Uses nvidia-smi to avoid importing torch (which may not see the GPU
        depending on when CUDA modules are loaded).
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                # Take the first GPU's memory (in MiB)
                mem_mib = int(result.stdout.strip().split('\n')[0].strip())
                mem_gb = mem_mib / 1024
                return mem_gb > 100
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass
        return False

    def __init__(self, model_path: str, model_name: str = None, 
                 max_model_len: int = 8192,
                 gpu_memory_utilization: float = 0.3,
                 timeout: float = 120.0,
                 enforce_eager: bool = None,
                 device: int = None,
                 **kwargs):
        """
        Initialize VLLM inference.
        
        Args:
            model_path: Path to the model
            model_name: Optional display name
            max_model_len: Maximum context length (default 8192 for matcher)
            gpu_memory_utilization: GPU memory fraction (default 0.3)
            timeout: Request timeout in seconds
            enforce_eager: Disable CUDA graphs. If None (default), auto-detects:
                           enabled on large GPUs (>100GB, e.g. B200) to avoid
                           flashinfer CUDA graph bugs, disabled otherwise.
            device: GPU device index to run on (e.g. 0, 1, 2). If None, inherits
                    parent process CUDA_VISIBLE_DEVICES.
            **kwargs: Additional args (ignored for server mode)
        """
        global _NEXT_PORT, _VLLM_SERVERS
        
        # Normalize local paths to avoid trailing-slash issues in vLLM config.
        self.model_path = os.path.normpath(model_path) if os.path.isabs(model_path) else model_path
        self.model_name = model_name or os.path.basename(os.path.normpath(model_path))
        self.max_model_len = max_model_len
        self.gpu_mem = gpu_memory_utilization
        self.timeout = timeout
        
        # Auto-detect enforce_eager based on GPU type
        if enforce_eager is None:
            self.enforce_eager = self._is_large_gpu()
            if self.enforce_eager:
                print(f"  Auto-detected large GPU (>100GB), enabling enforce_eager to avoid CUDA graph bugs", flush=True)
        else:
            self.enforce_eager = enforce_eager
        
        self._device = device
        self._port: Optional[int] = None
        self._session: Optional[requests.Session] = None
        self._server_started = False
        self._harmony_encoding = None
        self._harmony_import_error: Optional[Exception] = None
        # GPT-OSS Harmony completions route currently enforces 2048-token context in our setup.
        self._harmony_max_context = 2048
        
        # Check if there's already a server for this model
        for port, info in _VLLM_SERVERS.items():
            if info.get('model_path') == model_path:
                self._port = int(port)
                self._server_started = True
                break

    def _is_gpt_oss_model(self) -> bool:
        """Return True when model path/id points to GPT-OSS."""
        return "gpt-oss" in (self.model_path or "").lower()

    def _get_harmony_encoding(self):
        """Lazy-load Harmony encoder for GPT-OSS models."""
        if self._harmony_encoding is not None:
            return self._harmony_encoding
        if self._harmony_import_error is not None:
            return None
        try:
            from openai_harmony import HarmonyEncodingName, load_harmony_encoding
            self._harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
            return self._harmony_encoding
        except Exception as exc:
            self._harmony_import_error = exc
            print(f"  [VLLM] Harmony unavailable, falling back to chat API: {exc}", flush=True)
            return None

    @staticmethod
    def _harmony_role(role_name: str):
        """Map string role to openai_harmony Role enum."""
        from openai_harmony import Role
        role_map = {
            "system": Role.SYSTEM,
            "developer": Role.DEVELOPER,
            "assistant": Role.ASSISTANT,
            "tool": Role.TOOL,
            "user": Role.USER,
        }
        return role_map.get((role_name or "").lower(), Role.USER)

    def _build_harmony_prompt_token_ids(
        self,
        messages: List[Dict[str, str]],
        sampling_params: Dict[str, Any],
    ) -> Optional[List[int]]:
        """Build Harmony prompt token ids from OpenAI-style messages."""
        encoding = self._get_harmony_encoding()
        if encoding is None:
            return None
        try:
            from openai_harmony import (
                Conversation,
                Message,
                ReasoningEffort,
                Role,
                SystemContent,
            )

            effort_raw = str(sampling_params.get("reasoning_effort", "medium")).lower()
            effort_map = {
                "low": ReasoningEffort.LOW,
                "medium": ReasoningEffort.MEDIUM,
                "high": ReasoningEffort.HIGH,
            }
            effort = effort_map.get(effort_raw, ReasoningEffort.MEDIUM)

            harmony_messages = [
                Message.from_role_and_content(
                    Role.SYSTEM,
                    SystemContent.new().with_reasoning_effort(effort),
                )
            ]

            for msg in messages:
                role = self._harmony_role(msg.get("role", "user"))
                content = msg.get("content")
                if content is None:
                    continue
                harmony_messages.append(Message.from_role_and_content(role, content))

            convo = Conversation.from_messages(harmony_messages)
            return encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
        except Exception as exc:
            print(f"  [VLLM] Failed to build Harmony prompt, falling back to chat API: {exc}", flush=True)
            return None

    @staticmethod
    def _sanitize_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Drop invalid/empty message entries before sending to API."""
        cleaned: List[Dict[str, str]] = []
        for msg in messages:
            role = str(msg.get("role", "user"))
            content = msg.get("content")
            if content is None:
                continue
            content_str = str(content)
            if not content_str.strip():
                # Empty assistant/tool outputs can trigger 400s in strict servers.
                continue
            cleaned.append({"role": role, "content": content_str})
        if not cleaned:
            cleaned.append({"role": "user", "content": "Continue."})
        return cleaned

    def _parse_harmony_completion(
        self,
        completion_token_ids: Optional[List[int]],
        raw_text: str,
        usage: Dict[str, Any],
    ) -> str:
        """Extract assistant final text (and reasoning) from Harmony completion."""
        if not completion_token_ids:
            return raw_text or ""

        encoding = self._get_harmony_encoding()
        if encoding is None:
            return raw_text or ""
        try:
            from openai_harmony import Role

            entries = encoding.parse_messages_from_completion_tokens(
                completion_token_ids, Role.ASSISTANT
            )
        except Exception:
            return raw_text or ""

        finals: List[str] = []
        analyses: List[str] = []
        all_assistant_texts: List[str] = []
        for entry in entries:
            msg = entry.to_dict()
            role_value = msg.get("role")
            role_name = role_value.value if hasattr(role_value, "value") else str(role_value)
            if role_name != "assistant":
                continue
            channel = msg.get("channel")
            content = msg.get("content")

            text_value = ""
            if isinstance(content, str):
                text_value = content
            elif isinstance(content, list):
                parts: List[str] = []
                for part in content:
                    if isinstance(part, str):
                        parts.append(part)
                    elif isinstance(part, dict):
                        if isinstance(part.get("text"), str):
                            parts.append(part["text"])
                        elif isinstance(part.get("content"), str):
                            parts.append(part["content"])
                text_value = "".join(parts)
            elif content is not None:
                text_value = str(content)

            if not text_value:
                continue
            all_assistant_texts.append(text_value)
            if channel == "analysis":
                analyses.append(text_value)
            elif channel == "final":
                finals.append(text_value)

        if analyses:
            usage["_reasoning_content"] = "\n".join(analyses).strip()
        parsed_final = "\n".join([text for text in finals if text]).strip()
        if parsed_final:
            return parsed_final
        # Some GPT-OSS completions may not emit explicit "final" channel text.
        parsed_any = "\n".join([text for text in all_assistant_texts if text]).strip()
        return parsed_any or (raw_text or "")

    @staticmethod
    def _normalize_gpt_oss_response_text(text: str) -> str:
        """Normalize GPT-OSS channel-marked text into parser-friendly action text."""
        if not text:
            return ""
        normalized = text.strip()

        # Remove obvious transport/channel wrappers that occasionally leak into text.
        normalized = re.sub(r"(?is)<\s*/?\s*assistant[^>]*>", "", normalized)
        normalized = re.sub(r"(?is)<\s*/?\s*analysis[^>]*>", "", normalized)

        # vLLM may emit channel markers inline, e.g.:
        # "analysis ... assistantfinal ...", or "assistantaction type=..."
        if "assistantfinal" in normalized:
            tail = normalized.rsplit("assistantfinal", 1)[-1].strip()
            if tail:
                normalized = tail

        # Convert malformed action opener variants to canonical '<action ...>'
        normalized = re.sub(
            r"(?i)\bassistantaction\s+type\s*=",
            "<action type=",
            normalized,
        )
        normalized = re.sub(
            r"(?i)(?<!<)\baction\s+type\s*=",
            "<action type=",
            normalized,
        )

        # Harmony tool-call trace leakage can look like:
        # "... assistantanalysis to=python code<python...>"
        # Recover this into a proper query action.
        tool_call_match = re.search(r"(?is)\bto\s*=\s*python\s+code\s*(.+)$", normalized)
        if tool_call_match and "<action" not in normalized.lower():
            code = tool_call_match.group(1).strip()
            if code:
                return f"<action type=\"query\">\n```python\n{code}\n```\n</action>"

        # If there are multiple chunks, keep the last action block only.
        lower = normalized.lower()
        action_idx = lower.rfind("<action")
        if action_idx != -1:
            normalized = normalized[action_idx:]
            end_tag = "</action>"
            end_idx = normalized.lower().rfind(end_tag)
            if end_idx != -1:
                normalized = normalized[: end_idx + len(end_tag)]
                return normalized.strip()

            # Common malformed tails: "... </assistantanalysis..." or inline channel tokens.
            bad_tail_patterns = [
                r"(?i)</assistant",
                r"(?i)\bassistantanalysis\b",
                r"(?i)\bassistantfinal\b",
                r"(?i)\banalysis\b",
            ]
            cut_idx = len(normalized)
            for pat in bad_tail_patterns:
                m = re.search(pat, normalized)
                if m:
                    cut_idx = min(cut_idx, m.start())
            candidate = normalized[:cut_idx].rstrip()
            if candidate and not candidate.lower().endswith("</action>"):
                candidate = f"{candidate}\n</action>"
            # Broken submit actions without a parseable <forecast> block are common
            # when the model emits planning prose and gets cut off.
            if re.search(r'(?is)^<action\s+type\s*=\s*"submit"\s*>', candidate) and "<forecast" not in candidate.lower():
                return '<action type="next"/>'
            return candidate.strip()

        # Recover common malformed-but-salvageable generations:
        # 1) raw forecast XML without outer action wrapper
        if "<forecast" in lower and "</forecast>" in lower:
            return f"<action type=\"submit\">{normalized}</action>"

        # 2) python fenced code without action wrapper
        code_match = re.search(r"```python\s*(.*?)\s*```", normalized, re.DOTALL | re.IGNORECASE)
        if code_match:
            code = code_match.group(1).strip()
            # Remove leaked channel markers inside code fences.
            code = re.sub(r"(?is)\bassistantanalysis\b.*?\bto\s*=\s*python\s+code", "", code).strip()
            if code:
                return f"<action type=\"query\">\n```python\n{code}\n```\n</action>"

        # 3) direct dataframe-like code lines without fences
        if ("df[" in normalized or "print(" in normalized or "len(df" in normalized) and "<action" not in lower:
            candidate = normalized.strip()
            if candidate:
                return f"<action type=\"query\">\n```python\n{candidate}\n```\n</action>"

        # If the model emits only a self-closing next action marker without '<'
        if re.search(r'(?i)\bnext\s*/?>\s*$', normalized):
            return '<action type="next"/>'

        # As fallback, strip leading channel token if present.
        normalized = re.sub(r"(?i)^\s*analysis", "", normalized).strip()
        normalized = re.sub(r"(?i)</assistant.*$", "", normalized).strip()
        # Long planning prose without any action is usually unusable/truncated.
        # Convert to explicit next to keep turns parser-safe.
        if "<action" not in normalized.lower():
            prose_signals = ("we need to", "let's", "the system expects", "output:")
            if len(normalized) > 240 or any(sig in normalized.lower() for sig in prose_signals):
                return '<action type="next"/>'
        return normalized
    
    @staticmethod
    def _kill_stale_vllm_servers():
        """Kill any stale VLLM server processes from previous runs."""
        import signal
        try:
            result = subprocess.run(
                ["pgrep", "-f", "vllm.entrypoints.openai.api_server"],
                capture_output=True, text=True, timeout=5
            )
            if result.stdout.strip():
                stale_pids = result.stdout.strip().split('\n')
                # Filter out PIDs we're already tracking
                tracked_pids = {
                    info['process'].pid for info in _VLLM_SERVERS.values()
                    if info.get('process') and info['process'].poll() is None
                }
                for pid_str in stale_pids:
                    pid = int(pid_str.strip())
                    if pid not in tracked_pids:
                        print(f"  Killing stale VLLM server process (PID {pid})...", flush=True)
                        try:
                            os.kill(pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                if any(int(p.strip()) not in tracked_pids for p in stale_pids):
                    time.sleep(3)  # Wait for processes to release GPU memory
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass  # pgrep not available or other issue, skip cleanup

    def _ensure_server(self):
        """Start VLLM server if not already running."""
        global _NEXT_PORT, _VLLM_SERVERS
        
        if self._server_started:
            return
        
        # Kill any stale VLLM servers from previous runs to free GPU memory
        self._kill_stale_vllm_servers()
        
        # Find a free port
        self._port = _find_free_port(_NEXT_PORT)
        _NEXT_PORT = self._port + 1
        
        print(f"Starting VLLM server for {self.model_name} on port {self._port}...", flush=True)
        
        # Start server process
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--port", str(self._port),
            "--gpu-memory-utilization", str(self.gpu_mem),
            "--max-model-len", str(self.max_model_len),
            "--disable-log-stats",
            "--trust-remote-code",
            "--host", "0.0.0.0",  # Bind to all interfaces
        ]
        
        # Add enforce-eager flag if enabled (avoids CUDA graph bugs with newer models)
        if self.enforce_eager:
            cmd.append("--enforce-eager")
        
        # Log to a proper location - try output_dir from environment, fallback to /tmp
        import os
        log_dir = os.environ.get('SIM_OUTPUT_DIR', '/tmp')
        log_file = os.path.join(log_dir, f"matcher_server_{self._port}.log")
        self._log_file_path = log_file
        print(f"  Server log: {log_file}", flush=True)
        
        # Open log file to capture both stdout and stderr
        self._log_file = open(log_file, 'w')
        
        env = os.environ.copy()
        # Pin subprocess to a specific GPU if requested
        if self._device is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self._device)
            print(f"  Pinning server to GPU {self._device}", flush=True)
        # For MXFP4 models (like gpt-oss) on SM100 (Blackwell/B200):
        # FlashInfer 0.6.3's TRTLLM fused_moe runner doesn't support gpt-oss
        # routing method (enum 64), and the CUTLASS/BF16 backends trigger
        # massive JIT compilation (30+ min). We hide flashinfer from VLLM's
        # detection so it falls back to the Marlin/Triton MXFP4 backend.
        if "gpt-oss" in self.model_path.lower():
            # Force CUTLASS MXFP8 backend for GPT-OSS on Blackwell/SM100.
            # Default BF16 FlashInfer backend hits unsupported routing method.
            env["VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS"] = "1"
            env["VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8"] = "0"
            env["VLLM_USE_FLASHINFER_MOE_MXFP4_BF16"] = "0"
            print("  GPT-OSS model: forcing FlashInfer CUTLASS MXFP8 MoE backend", flush=True)
        
        proc = subprocess.Popen(
            cmd,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            env=env,
        )
        
        _VLLM_SERVERS[str(self._port)] = {
            'process': proc,
            'model_path': self.model_path,
        }
        
        # Immediately check if process died
        time.sleep(0.5)
        if proc.poll() is not None:
            self._log_file.flush()
            with open(log_file, 'r') as f:
                content = f.read()
            print(f"  ERROR: Server process died immediately! Exit code: {proc.returncode}", flush=True)
            print(f"  Server log:\n{content[:1000]}", flush=True)
            raise RuntimeError(f"VLLM server process died immediately on port {self._port}")
        
        # Wait for server to be ready with detailed progress
        print(f"  Waiting for server to be ready...", flush=True)
        start_time = time.time()
        timeout = 300  # 5 minutes to account for torch.compile and initialization
        
        ready = False
        session = self._get_session()
        while time.time() - start_time < timeout:
            # Check if process is still alive
            if proc.poll() is not None:
                self._log_file.flush()
                with open(log_file, 'r') as f:
                    content = f.read()
                elapsed = time.time() - start_time
                print(f"  ERROR: Server died after {elapsed:.1f}s! Exit code: {proc.returncode}", flush=True)
                print(f"  Last 1000 chars of log:\n{content[-1000:]}", flush=True)
                # Give a helpful hint for common GPU memory errors
                if "Free memory on device" in content and "less than desired" in content:
                    print(f"  HINT: GPU memory is insufficient. Try lowering --matcher_gpu_mem or kill other GPU processes.", flush=True)
                raise RuntimeError(f"VLLM server died on port {self._port}")
            
            try:
                response = session.get(
                    f"http://127.0.0.1:{self._port}/v1/models",
                    timeout=5
                )
                if response.status_code == 200:
                    ready = True
                    break
            except requests.exceptions.RequestException:
                pass
            
            # Progress update every 30 seconds
            elapsed = time.time() - start_time
            if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                print(f"    Still waiting... ({elapsed:.0f}s)", flush=True)
            
            time.sleep(2)
        
        if not ready:
            elapsed = time.time() - start_time
            self._log_file.flush()
            with open(log_file, 'r') as f:
                content = f.read()
            print(f"  ERROR: Server timeout after {elapsed:.1f}s", flush=True)
            print(f"  Last 1000 chars of log:\n{content[-1000:]}", flush=True)
            proc.terminate()
            raise RuntimeError(f"VLLM server failed to start on port {self._port} (timeout)")
        
        elapsed = time.time() - start_time
        print(f"  VLLM server ready on port {self._port} (took {elapsed:.1f}s)", flush=True)
        self._server_started = True
    
    def _wait_for_server(self, timeout: float = 120.0) -> bool:
        """Wait for server to become ready."""
        start = time.time()
        session = self._get_session()
        
        while time.time() - start < timeout:
            try:
                response = session.get(
                    f"http://127.0.0.1:{self._port}/v1/models",
                    timeout=5
                )
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)
        
        return False
    
    def _get_session(self) -> requests.Session:
        """Get or create requests session with proxy disabled for localhost."""
        if self._session is None:
            self._session = requests.Session()
            # Disable proxy for localhost - cluster has Squid proxy that intercepts all traffic
            self._session.proxies = {'http': None, 'https': None}
            self._session.trust_env = False  # Don't use system proxy settings
        return self._session
    
    def chat(self, messages: List[Dict[str, str]], sampling_params: Dict[str, Any]) -> Tuple[str, Dict]:
        """
        Chat completion using messages format.
        
        Args:
            messages: List of {"role": "system"|"user"|"assistant", "content": "..."}
            sampling_params: Dict with temperature, max_tokens, etc.
            
        Returns:
            Tuple of (response_text, usage_dict)
        """
        self._ensure_server()
        
        messages = self._sanitize_messages(messages)

        payload = {
            "model": self.model_path,
            "messages": messages,
            "temperature": sampling_params.get("temperature", 0.7),
            "max_tokens": sampling_params.get("max_tokens", 1024),
        }
        
        # Add optional params
        if "top_p" in sampling_params:
            payload["top_p"] = sampling_params["top_p"]
        if "stop" in sampling_params:
            payload["stop"] = sampling_params["stop"]
        
        session = self._get_session()

        # GPT-OSS models work best with Harmony-rendered prompt token ids.
        harmony_payload = None
        if self._is_gpt_oss_model():
            prompt_token_ids = self._build_harmony_prompt_token_ids(messages, sampling_params)
            if prompt_token_ids:
                requested_max_tokens = int(payload["max_tokens"])
                # Reserve meaningful decode budget; leaving only 1 token causes empty outputs.
                # Reserve a larger completion budget to reduce mid-sentence truncation.
                reserve_for_completion = max(128, min(requested_max_tokens, 512))
                max_prompt_tokens = max(1, self._harmony_max_context - reserve_for_completion - 1)
                # Keep within effective context limit for Harmony completions endpoint.
                if len(prompt_token_ids) > max_prompt_tokens:
                    keep = max_prompt_tokens
                    prompt_token_ids = prompt_token_ids[-keep:]
                allowed_max_tokens = max(1, self._harmony_max_context - len(prompt_token_ids) - 1)
                harmony_payload = {
                    "model": self.model_path,
                    # OpenAI-compatible completions endpoint accepts tokenized prompts via `prompt`.
                    "prompt": prompt_token_ids,
                    "temperature": payload["temperature"],
                    "max_tokens": min(requested_max_tokens, allowed_max_tokens),
                    # Use Harmony stop tokens to end on assistant action boundaries.
                    "stop_token_ids": self._get_harmony_encoding().stop_tokens_for_assistant_actions(),
                }
                if "top_p" in payload:
                    harmony_payload["top_p"] = payload["top_p"]
        
        for attempt in range(3):
            try:
                if harmony_payload is not None:
                    response = session.post(
                        f"http://127.0.0.1:{self._port}/v1/completions",
                        json=harmony_payload,
                        timeout=self.timeout,
                    )
                    if response.status_code == 200:
                        data = response.json()
                        choice = data["choices"][0]
                        usage = data.get("usage", {})
                        # vLLM can expose token ids in non-standard fields; use if present.
                        completion_token_ids = (
                            choice.get("token_ids")
                            or choice.get("completion_token_ids")
                            or choice.get("output_token_ids")
                        )
                        content = self._parse_harmony_completion(
                            completion_token_ids=completion_token_ids,
                            raw_text=choice.get("text", ""),
                            usage=usage,
                        )
                        content = self._normalize_gpt_oss_response_text(content)
                        return content, usage
                    if response.status_code in (400, 404, 422):
                        # Endpoint may not accept prompt_token_ids in some server versions.
                        try:
                            err_text = response.text[:300]
                            print(f"  [VLLM] Harmony completions rejected ({response.status_code}): {err_text}", flush=True)
                        except Exception:
                            pass
                        harmony_payload = None
                        continue
                else:
                    response = session.post(
                        f"http://127.0.0.1:{self._port}/v1/chat/completions",
                        json=payload,
                        timeout=self.timeout,
                    )
                
                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    if self._is_gpt_oss_model():
                        content = self._normalize_gpt_oss_response_text(content)
                    usage = data.get("usage", {})
                    return content, usage
                
                # Server errors - retry
                if response.status_code in (500, 502, 503, 504, 529):
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                        continue
                # Invalid request payload should not crash the whole simulation.
                if response.status_code == 400:
                    try:
                        err_text = response.text[:500]
                        print(f"  [VLLM] Bad request from server: {err_text}", flush=True)
                        # Auto-reduce max_tokens when context + max_tokens exceeds max_model_len
                        if "'max_tokens' or 'max_completion_tokens' is too large" in err_text:
                            current_max = payload.get("max_tokens", 1024)
                            reduced = max(128, current_max // 2)
                            if reduced < current_max and attempt < 2:
                                print(f"  [VLLM] Reducing max_tokens from {current_max} to {reduced} and retrying", flush=True)
                                payload["max_tokens"] = reduced
                                continue
                    except Exception:
                        pass
                    return "", {}
                
                response.raise_for_status()
                
            except requests.exceptions.Timeout:
                if attempt < 2:
                    continue
                print(f"  [VLLM] Timeout after {self.timeout}s")
                return "", {}
                
            except requests.exceptions.ConnectionError:
                if attempt < 2:
                    time.sleep(2)
                    continue
                print("  [VLLM] Connection error")
                return "", {}
        
        return "", {}
