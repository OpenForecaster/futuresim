"""
VLLM inference provider.

Uses VLLM OpenAI-compatible server internally to avoid GPU contention issues.
The server is started automatically on first use and cleaned up on exit.
"""

from typing import Dict, Any, List, Tuple, Optional
import os
import time
import json
import atexit
import subprocess
import socket
import sys
from threading import Lock, Condition
from dataclasses import dataclass

try:
    import requests
except ImportError:
    raise ImportError("requests module not found. Install with: pip install requests")


# Global server management
_VLLM_SERVERS: Dict[str, dict] = {}  # port -> {process, model_path}
_NEXT_PORT = 8001
_SERVERS_LOCK = Lock()
_PORT_LOCK = Lock()


def _is_gpt_oss_model(model_path: str, model_name: str) -> bool:
    """
    Heuristic to detect OpenAI GPT-OSS weights.

    gpt-oss models use the "Harmony" prompt/tooling format. vLLM supports them
    (including MXFP4 weights) in recent versions, but tool calling is expected
    via the OpenAI Responses API rather than Chat Completions.
    """
    p = (model_path or "").lower()
    n = (model_name or "").lower()
    return ("gpt-oss" in p) or ("gpt-oss" in n)


def _safe_filename(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "model"
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)[:120]


@dataclass
class _EmbedOutputs:
    embedding: List[float]


@dataclass
class _EmbedItem:
    outputs: _EmbedOutputs


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
    
    def __init__(self, model_path: str, model_name: str = None, 
                 max_model_len: int = 8192,
                 gpu_memory_utilization: float = 0.3,
                 tensor_parallel_size: int = 1,
                 rope_scaling: Optional[Dict[str, Any]] = None,
                 timeout: float = 120.0,
                 startup_timeout: float = 300.0,
                 enable_tools: bool = False,
                 cuda_visible_devices: Optional[str] = None,
                 **kwargs):
        """
        Initialize VLLM inference.
        
        Args:
            model_path: Path to the model
            model_name: Optional display name
            max_model_len: Maximum context length (default 8192 for matcher)
            gpu_memory_utilization: GPU memory fraction (default 0.3)
            rope_scaling: Optional RoPE scaling config for long-context extension.
            timeout: Request timeout in seconds
            startup_timeout: Server startup timeout in seconds (default 600).
            enable_tools: Start vLLM with tool-calling flags (requires newer vLLM).
            **kwargs: Additional args (ignored for server mode)
        """
        global _NEXT_PORT, _VLLM_SERVERS
        
        self.model_path = model_path
        self.model_name = model_name or os.path.basename(model_path)
        self.max_model_len = max_model_len
        self.gpu_mem = gpu_memory_utilization
        self.tensor_parallel_size = max(1, int(tensor_parallel_size))
        self.rope_scaling = rope_scaling if isinstance(rope_scaling, dict) else None
        self.timeout = timeout
        self.startup_timeout = startup_timeout
        self.enable_tools = enable_tools
        self.cuda_visible_devices = cuda_visible_devices

        # GPT-OSS uses the Harmony format. vLLM supports it, but tool calling is
        # expected via /v1/responses rather than /v1/chat/completions.
        self._is_gpt_oss = _is_gpt_oss_model(self.model_path, self.model_name)
        
        self._port: Optional[int] = None
        self._session: Optional[requests.Session] = None
        self._server_started = False
        self._start_lock = Lock()
        self._start_cond = Condition(self._start_lock)
        self._starting = False
        self._start_error: Optional[str] = None
        
        # Check if there's already a server for this model
        with _SERVERS_LOCK:
            for port, info in _VLLM_SERVERS.items():
                if (
                    info.get('model_path') == model_path
                    and info.get('cuda_visible_devices') == self.cuda_visible_devices
                    and int(info.get('tensor_parallel_size', 1)) == self.tensor_parallel_size
                ):
                    self._port = int(port)
                    self._server_started = True
                    break
    
    def _ensure_server(self):
        """Start VLLM server if not already running."""
        global _NEXT_PORT, _VLLM_SERVERS

        # AllQ warmup / AllQD can issue many concurrent requests. Ensure only
        # one thread performs server startup per VLLMInference instance, and
        # propagate startup failures to all waiters (avoid thrash-restarts).
        while True:
            with self._start_lock:
                if self._server_started:
                    return
                if self._start_error is not None:
                    raise RuntimeError(self._start_error)
                if self._starting:
                    self._start_cond.wait(timeout=5.0)
                    continue
                self._starting = True
                break

        try:
            # Find a free port (global allocator).
            with _PORT_LOCK:
                self._port = _find_free_port(_NEXT_PORT)
                _NEXT_PORT = self._port + 1
        
            print(f"Starting VLLM server for {self.model_name} on port {self._port}...", flush=True)
        
            # Start server process through a local wrapper so repo-level
            # compatibility patches can be applied reproducibly.
            cmd = [
                sys.executable, "-m", "inference.vllm_api_server_wrapper",
                "--model", self.model_path,
                "--port", str(self._port),
                "--gpu-memory-utilization", str(self.gpu_mem),
                "--max-model-len", str(self.max_model_len),
                "--tensor-parallel-size", str(self.tensor_parallel_size),
                "--disable-log-stats",
                "--trust-remote-code",
                "--host", "0.0.0.0",  # Bind to all interfaces
            ]
            if self.rope_scaling:
                cmd += ["--rope-scaling", json.dumps(self.rope_scaling, separators=(",", ":"))]
        
            # Tool calling on vLLM's OpenAI server requires extra flags in newer vLLM.
            # We keep this behind a toggle because older vLLM versions will error on
            # unknown flags.
            if self.enable_tools:
                cmd += [
                    "--enable-auto-tool-choice",
                    "--tool-call-parser", "openai",
                ]
        
            # Log to a proper location - try output_dir from environment, fallback to /tmp
            import os
            log_dir = os.environ.get('SIM_OUTPUT_DIR', '/tmp')
            log_file = os.path.join(log_dir, f"vllm_server_{_safe_filename(self.model_name)}_{self._port}.log")
            self._log_file_path = log_file
            print(f"  Server log: {log_file}", flush=True)
        
            # Open log file to capture both stdout and stderr
            self._log_file = open(log_file, 'w')
        
            proc = subprocess.Popen(
                cmd,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                env=self._build_subprocess_env(),
            )
        
            with _SERVERS_LOCK:
                _VLLM_SERVERS[str(self._port)] = {
                    'process': proc,
                    'model_path': self.model_path,
                    'cuda_visible_devices': self.cuda_visible_devices,
                    'tensor_parallel_size': self.tensor_parallel_size,
                }
        
            # Immediately check if process died
            time.sleep(0.5)
            if proc.poll() is not None:
                self._log_file.flush()
                with open(log_file, 'r') as f:
                    content = f.read()
                print(f"  ERROR: Server process died immediately! Exit code: {proc.returncode}", flush=True)
                print(f"  Server log:\n{content[:1000]}", flush=True)
                if self.rope_scaling and "--rope-scaling" in " ".join(cmd) and "unrecognized arguments: --rope-scaling" in content:
                    # Compatibility fallback for older vLLM builds:
                    # pass rope_scaling through HF config overrides.
                    print("  [VLLM] --rope-scaling unsupported; retrying with --hf-overrides rope_scaling.", flush=True)
                    try:
                        self._log_file.close()
                    except Exception:
                        pass
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    cmd2 = list(cmd)
                    if "--rope-scaling" in cmd2:
                        i = cmd2.index("--rope-scaling")
                        # Remove flag and its JSON argument.
                        del cmd2[i:i+2]
                    cmd2 += ["--hf-overrides", json.dumps({"rope_scaling": self.rope_scaling}, separators=(",", ":"))]
                    self._log_file = open(log_file, 'w')
                    proc = subprocess.Popen(
                        cmd2,
                        stdout=self._log_file,
                        stderr=subprocess.STDOUT,
                        env=self._build_subprocess_env(),
                    )
                    with _SERVERS_LOCK:
                        _VLLM_SERVERS[str(self._port)] = {
                            'process': proc,
                            'model_path': self.model_path,
                            'cuda_visible_devices': self.cuda_visible_devices,
                            'tensor_parallel_size': self.tensor_parallel_size,
                        }
                    time.sleep(0.5)
                    if proc.poll() is not None:
                        self._log_file.flush()
                        with open(log_file, 'r') as f:
                            content2 = f.read()
                        print(f"  ERROR: Retry with --hf-overrides also failed! Exit code: {proc.returncode}", flush=True)
                        print(f"  Server log:\n{content2[:1000]}", flush=True)
                        raise RuntimeError(
                            "vLLM does not support configured rope scaling on this node "
                            "(both --rope-scaling and --hf-overrides path failed)."
                        )
                else:
                    raise RuntimeError(f"VLLM server process died immediately on port {self._port}")
        
            # Wait for server to be ready with detailed progress
            print(f"  Waiting for server to be ready...", flush=True)
            start_time = time.time()
            timeout = float(self.startup_timeout)
        
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
            with self._start_lock:
                self._server_started = True
                self._start_error = None
        except Exception as e:
            with self._start_lock:
                self._start_error = str(e)
            raise
        finally:
            with self._start_lock:
                self._starting = False
                self._start_cond.notify_all()
    
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

    def _build_subprocess_env(self) -> Dict[str, str]:
        """
        Build the environment for the vLLM server subprocess.

        This is how we pin a specific server to a specific GPU (or set of GPUs)
        for multi-GPU local runs.
        """
        env = os.environ.copy()
        if self.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.cuda_visible_devices)
        # Ensure subprocess can import local modules (inference.* wrapper).
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        existing_pythonpath = env.get("PYTHONPATH", "")
        path_items = [p for p in existing_pythonpath.split(os.pathsep) if p] if existing_pythonpath else []
        if repo_root not in path_items:
            env["PYTHONPATH"] = (
                repo_root if not existing_pythonpath else f"{repo_root}{os.pathsep}{existing_pythonpath}"
            )
        # Enable permissive Harmony parser mode by default for GPT-OSS.
        # This mitigates malformed-header parse failures without editing
        # site-packages directly.
        if self._is_gpt_oss and "FSIM_VLLM_HARMONY_NON_STRICT" not in env:
            env["FSIM_VLLM_HARMONY_NON_STRICT"] = "1"
        return env

    def embed(self, texts: List[str], use_tqdm: bool = False) -> List[_EmbedItem]:
        """
        Embedding API via vLLM OpenAI server (/v1/embeddings).

        Returns a lightweight structure compatible with the existing LanceDB code:
            items[i].outputs.embedding -> List[float]
        """
        del use_tqdm  # kept for signature compatibility with vLLM LLM.embed

        self._ensure_server()
        payload: Dict[str, Any] = {"model": self.model_path, "input": texts}

        session = self._get_session()
        for attempt in range(3):
            try:
                response = session.post(
                    f"http://127.0.0.1:{self._port}/v1/embeddings",
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code == 200:
                    data = response.json()
                    out: List[_EmbedItem] = []
                    for row in (data.get("data") or []):
                        emb = row.get("embedding")
                        if isinstance(emb, list):
                            out.append(_EmbedItem(outputs=_EmbedOutputs(embedding=emb)))
                    return out

                if response.status_code in (500, 502, 503, 504, 529):
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                        continue
                response.raise_for_status()
            except requests.exceptions.Timeout:
                if attempt < 2:
                    continue
                print(f"  [VLLM] Embeddings timeout after {self.timeout}s")
                return []
            except requests.exceptions.ConnectionError:
                if attempt < 2:
                    time.sleep(2)
                    continue
                print("  [VLLM] Embeddings connection error")
                return []
        return []

    @staticmethod
    def _extract_chat_text_and_usage(data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        # OpenAI-like: {"choices":[{"message":{"content":"..."}}], "usage":{...}}
        content: Any = ""
        try:
            content = data["choices"][0]["message"].get("content") or ""
        except Exception:
            content = ""
        # Some servers/models return content as a list of parts.
        if isinstance(content, list):
            parts: List[str] = []
            for p in content:
                if isinstance(p, str):
                    parts.append(p)
                elif isinstance(p, dict):
                    t = p.get("text") or p.get("content")  # tolerate different schemas
                    if isinstance(t, str):
                        parts.append(t)
            content = "".join(parts)
        if content is None:
            content = ""
        usage = data.get("usage", {}) or {}
        return str(content), usage

    @staticmethod
    def _split_instructions_and_input(messages: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Convert Chat Completions messages -> Responses API (instructions + input messages).

        vLLM's /v1/responses expects OpenAI Responses-style content blocks:
            {"role": "...", "content": [{"type": "input_text", "text": "..."}]}
        """
        instructions_parts: List[str] = []
        input_messages: List[Dict[str, Any]] = []
        for m in messages or []:
            role = m.get("role") or "user"
            content = m.get("content", "")
            if role == "system":
                if content:
                    instructions_parts.append(str(content))
                continue

            blocks: List[Dict[str, str]] = []
            if content is not None and str(content) != "":
                blocks.append({"type": "input_text", "text": str(content)})
            input_messages.append({"role": role, "content": blocks})
        return "\n\n".join(instructions_parts), input_messages

    @staticmethod
    def _extract_responses_text_and_usage(data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        # Prefer vLLM/OpenAI-style convenience field when present.
        if isinstance(data, dict) and isinstance(data.get("output_text"), str):
            return data["output_text"], (data.get("usage") or {})

        # Otherwise, try to walk a Responses payload.
        # Typical OpenAI Responses: {"output":[{"type":"message","content":[{"type":"output_text","text":"..."}]}], "usage":{...}}
        chunks: List[str] = []
        try:
            for item in data.get("output", []) or []:
                if not isinstance(item, dict):
                    continue
                if item.get("type") != "message":
                    continue
                for c in item.get("content", []) or []:
                    if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                        t = c.get("text")
                        if isinstance(t, str):
                            chunks.append(t)
        except Exception:
            pass
        usage = data.get("usage", {}) or {}
        return "".join(chunks), usage

    @staticmethod
    def _normalize_responses_overrides(overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Normalize caller-side /v1/responses overrides to the schema expected by
        current vLLM Responses endpoints.

        Notes:
        - vLLM expects reasoning effort under `reasoning: {"effort": ...}` for
          GPT-OSS Harmony system message construction.
        - Some callers still pass legacy top-level keys like `reasoning_effort`
          and `include_reasoning`; map/drop those to avoid ignored-field warnings.
        """
        if not overrides:
            return {}

        normalized = dict(overrides)

        effort_raw = normalized.pop("reasoning_effort", None)
        if isinstance(effort_raw, str):
            effort = effort_raw.strip().lower()
            if effort in ("none", "minimal", "low", "medium", "high", "xhigh"):
                reasoning_cfg = normalized.get("reasoning")
                if not isinstance(reasoning_cfg, dict):
                    reasoning_cfg = {}
                reasoning_cfg = dict(reasoning_cfg)
                reasoning_cfg.setdefault("effort", effort)
                normalized["reasoning"] = reasoning_cfg

        # /v1/responses does not accept this top-level field on vLLM 0.13.
        normalized.pop("include_reasoning", None)

        return normalized
    
    def chat(self, messages: List[Dict[str, str]], sampling_params: Dict[str, Any]) -> Tuple[str, Dict]:
        """
        Chat completion using messages format.
        
        Args:
            messages: List of {"role": "system"|"user"|"assistant", "content": "..."}
            sampling_params: Dict with temperature, max_tokens, etc.
                Optional tool keys: tools, tool_choice.
            
        Returns:
            Tuple of (response_text, usage_dict)
        """
        self._ensure_server()

        # GPT-OSS:
        # - Plain text generation works via /v1/chat/completions (vLLM will internally
        #   convert ChatML -> Harmony as needed).
        # - If you want OpenAI-style tool calling, route through /v1/responses.
        if self._is_gpt_oss and (sampling_params.get("tools") is not None or sampling_params.get("tool_choice") is not None):
            instructions, input_messages = self._split_instructions_and_input(messages)
            return self.responses(
                instructions=instructions,
                input_messages=input_messages,
                sampling_params=sampling_params,
            )
        
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

        # Pass-through for non-gpt-oss models that support tools via chat completions.
        if "tools" in sampling_params:
            payload["tools"] = sampling_params["tools"]
        if "tool_choice" in sampling_params:
            payload["tool_choice"] = sampling_params["tool_choice"]
        session = self._get_session()
        
        for attempt in range(3):
            try:
                response = session.post(
                    f"http://127.0.0.1:{self._port}/v1/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return self._extract_chat_text_and_usage(data)

                if response.status_code == 400:
                    # 400s are usually request-shape/content issues; don't crash the whole sim.
                    body = response.text if isinstance(response.text, str) else ""
                    tail = body[-600:] if body else "unknown 400 body"
                    print(f"  [VLLM] 400 Bad Request on chat/completions: {tail}")
                    return "", {}
                
                # Server errors - retry
                if response.status_code in (500, 502, 503, 504, 529):
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                        continue
                
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

    def responses(
        self,
        *,
        instructions: str,
        input_messages: List[Dict[str, Any]],
        sampling_params: Dict[str, Any],
    ) -> Tuple[str, Dict]:
        """
        OpenAI Responses API-style call against vLLM's /v1/responses.

        This is the recommended path for GPT-OSS tool calling on vLLM.

        Args:
            instructions: System/developer instructions string.
            input_messages: Conversation as OpenAI-style message dicts (non-system).
            sampling_params: Supports temperature, max_tokens (mapped to max_output_tokens),
                tools, tool_choice, top_p, stop.

        Returns:
            Tuple of (response_text, usage_dict)
        """
        self._ensure_server()

        payload: Dict[str, Any] = {
            "model": self.model_path,
            "instructions": instructions or "",
            "input": input_messages,
            "temperature": sampling_params.get("temperature", 0.7),
            "max_output_tokens": sampling_params.get("max_tokens", 1024),
        }
        if "top_p" in sampling_params:
            payload["top_p"] = sampling_params["top_p"]
        if "stop" in sampling_params:
            payload["stop"] = sampling_params["stop"]
        if "tools" in sampling_params:
            payload["tools"] = sampling_params["tools"]
        if "tool_choice" in sampling_params:
            payload["tool_choice"] = sampling_params["tool_choice"]
        responses_overrides = {
            k: sampling_params[k]
            for k in ("reasoning", "reasoning_effort", "include_reasoning")
            if k in sampling_params
        }
        payload.update(self._normalize_responses_overrides(responses_overrides))

        session = self._get_session()
        for attempt in range(3):
            try:
                response = session.post(
                    f"http://127.0.0.1:{self._port}/v1/responses",
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code == 200:
                    data = response.json()
                    return self._extract_responses_text_and_usage(data)

                # Helpful diagnostics: vLLM returns JSON validation errors on 400.
                if response.status_code == 400:
                    body = response.text
                    body = body[-1500:] if isinstance(body, str) else ""
                    raise RuntimeError(f"vLLM /v1/responses 400 Bad Request: {body}")

                if response.status_code in (500, 502, 503, 504, 529):
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                        continue
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

    def responses_json(
        self,
        *,
        instructions: str,
        input_messages: List[Dict[str, Any]],
        sampling_params: Dict[str, Any],
        request_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Like responses(), but returns the full JSON payload from vLLM's /v1/responses.

        This is useful for GPT-OSS Harmony tool calling, where the response may include
        structured output items like {"type":"function_call", "name":..., "arguments":...}.
        """
        self._ensure_server()

        payload: Dict[str, Any] = {
            "model": self.model_path,
            "instructions": instructions or "",
            "input": input_messages,
            "temperature": sampling_params.get("temperature", 0.7),
            "max_output_tokens": sampling_params.get("max_tokens", 1024),
        }
        if "top_p" in sampling_params:
            payload["top_p"] = sampling_params["top_p"]
        if "stop" in sampling_params:
            payload["stop"] = sampling_params["stop"]
        if "tools" in sampling_params:
            payload["tools"] = sampling_params["tools"]
        if "tool_choice" in sampling_params:
            payload["tool_choice"] = sampling_params["tool_choice"]
        if request_overrides:
            # e.g. {"parallel_tool_calls": False, "max_tool_calls": 1}
            payload.update(self._normalize_responses_overrides(request_overrides))

        session = self._get_session()
        for attempt in range(3):
            try:
                response = session.post(
                    f"http://127.0.0.1:{self._port}/v1/responses",
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code == 200:
                    return response.json()

                if response.status_code == 400:
                    body = response.text
                    body = body[-1500:] if isinstance(body, str) else ""
                    raise RuntimeError(f"vLLM /v1/responses 400 Bad Request: {body}")

                if response.status_code in (500, 502, 503, 504, 529):
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                        continue
                response.raise_for_status()
            except requests.exceptions.Timeout:
                if attempt < 2:
                    continue
                print(f"  [VLLM] Timeout after {self.timeout}s")
                return {}
            except requests.exceptions.ConnectionError:
                if attempt < 2:
                    time.sleep(2)
                    continue
                print("  [VLLM] Connection error")
                return {}
        return {}
