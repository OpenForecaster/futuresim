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
    
    def __init__(self, model_path: str, model_name: str = None, 
                 max_model_len: int = 8192,
                 gpu_memory_utilization: float = 0.3,
                 timeout: float = 120.0,
                 **kwargs):
        """
        Initialize VLLM inference.
        
        Args:
            model_path: Path to the model
            model_name: Optional display name
            max_model_len: Maximum context length (default 8192 for matcher)
            gpu_memory_utilization: GPU memory fraction (default 0.3)
            timeout: Request timeout in seconds
            **kwargs: Additional args (ignored for server mode)
        """
        global _NEXT_PORT, _VLLM_SERVERS
        
        self.model_path = model_path
        self.model_name = model_name or os.path.basename(model_path)
        self.max_model_len = max_model_len
        self.gpu_mem = gpu_memory_utilization
        self.timeout = timeout
        
        self._port: Optional[int] = None
        self._session: Optional[requests.Session] = None
        self._server_started = False
        
        # Check if there's already a server for this model
        for port, info in _VLLM_SERVERS.items():
            if info.get('model_path') == model_path:
                self._port = int(port)
                self._server_started = True
                break
    
    def _ensure_server(self):
        """Start VLLM server if not already running."""
        global _NEXT_PORT, _VLLM_SERVERS
        
        if self._server_started:
            return
        
        # Find a free port
        self._port = _find_free_port(_NEXT_PORT)
        _NEXT_PORT = self._port + 1
        
        print(f"Starting VLLM server for {self.model_name} on port {self._port}...", flush=True)
        
        # Start server process
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--port", str(self._port),
            "--gpu-memory-utilization", str(self.gpu_mem),
            "--max-model-len", str(self.max_model_len),
            "--disable-log-stats",
            "--trust-remote-code",
            "--host", "0.0.0.0",  # Bind to all interfaces
        ]
        
        # Log to a proper location - try output_dir from environment, fallback to /tmp
        import os
        log_dir = os.environ.get('SIM_OUTPUT_DIR', '/tmp')
        log_file = os.path.join(log_dir, f"matcher_server_{self._port}.log")
        self._log_file_path = log_file
        print(f"  Server log: {log_file}", flush=True)
        
        # Open log file to capture both stdout and stderr
        self._log_file = open(log_file, 'w')
        
        proc = subprocess.Popen(
            cmd,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
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
        timeout = 180  # 3 minutes should be plenty for a 4B model
        
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
        
        for attempt in range(3):
            try:
                response = session.post(
                    f"http://127.0.0.1:{self._port}/v1/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                )
                
                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    return content, usage
                
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
