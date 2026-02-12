#!/usr/bin/env python
"""
Start a vLLM OpenAI-compatible server from a small YAML config.

Intended for HTCondor jobs and consistent startup across nodes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def _as_bool(v, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "on")
    return default


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())

    model = cfg["model"]
    host = cfg.get("host", "0.0.0.0")
    port = int(cfg.get("port", 8000))
    max_model_len = int(cfg.get("max_model_len", 32768))
    gpu_mem = float(cfg.get("gpu_memory_utilization", 0.90))
    tp = int(cfg.get("tensor_parallel_size", 1))
    trust_remote_code = _as_bool(cfg.get("trust_remote_code"), default=True)
    served_model_name = cfg.get("served_model_name")

    enable_tools = _as_bool(cfg.get("enable_tools"), default=False)
    tool_call_parser = cfg.get("tool_call_parser", "openai")

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(gpu_mem),
        "--max-model-len",
        str(max_model_len),
        "--tensor-parallel-size",
        str(tp),
        "--disable-log-stats",
    ]

    if trust_remote_code:
        cmd.append("--trust-remote-code")
    if served_model_name:
        cmd += ["--served-model-name", str(served_model_name)]

    # Tool calling flags (require newer vLLM).
    if enable_tools:
        cmd += ["--enable-auto-tool-choice", "--tool-call-parser", str(tool_call_parser)]

    print("Launching:", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

