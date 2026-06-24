"""MCP configuration and host-side relay helpers for MinimalHarnessAgent."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

from futuresim_agents.minimalHarnessAgent.sandbox_common import (
    resolve_socat_cmd,
    terminate_process,
    write_mcp_connector,
)

logger = logging.getLogger(__name__)


def _resolve_socat_cmd() -> str:
    return resolve_socat_cmd(prefer_path=True)


class McpHelpers:
    """Build MCP command lines and manage the sandbox relay process."""

    def _write_mcp_config(self) -> None:
        """Write Claude Code MCP config.

        Repo root goes through --repo-root, not config `env`, because Claude
        replaces rather than merges env when `env` is set. With sandbox=True,
        the config points at a socat bridge and MCP stays host-side.
        """
        cmd, args = self._build_mcp_invocation()
        config = {
            "mcpServers": {
                "forecast": {
                    "command": cmd,
                    "args": args,
                }
            }
        }
        config_path = self._internal_dir / "mcp_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    def _mcp_tool_filter_args(self) -> List[str]:
        if self.config.prompt_mode == "static_search":
            return ["--enabled-tools", "submit_forecasts"]
        if self.config.prompt_mode == "warmup":
            return ["--enabled-tools", "search_news,submit_forecasts"]
        return []

    def _mcp_server_args(self, embed_server_url: str = "") -> List[str]:
        repo_root = str(Path(__file__).resolve().parents[2])
        mcp_server_script = str(Path(__file__).resolve().parent / "mcp_server.py")
        args = [
            mcp_server_script,
            "--workspace", str(self.workspace),
            "--internal-dir", str(self._internal_dir),
            "--repo-root", repo_root,
            "--search-backend", self.config.search_backend or "",
            "--search-db", self.config.search_db or "",
            "--embedding-model", self.config.embedding_model or "",
            "--openreward-search-url", self.config.openreward_search_url or "",
            "--openreward-fetch-url", self.config.openreward_fetch_url or "",
            "--handholding-version", self.config.handholding_version,
            "--agent-id", self.agent_id,
        ]
        args.extend(self._mcp_tool_filter_args())
        if embed_server_url:
            args.extend(["--embedding-server-url", embed_server_url])
        if self.config.prompt_mode == "active_memory2":
            args.append("--enable-memory-tools")
        return args

    def _build_mcp_invocation(self) -> Tuple[str, List[str]]:
        """Return (command, args) for harness MCP config.

        sandbox=False keeps pre-refactor behavior: the harness directly spawns
        mcp_server.py with full repo access. sandbox=True uses
        `socat - UNIX-CONNECT:<sock>` to bridge harness stdio to a host relay
        that forks one mcp_server.py per connection. Keeping MCP, search_db,
        lancedb, and the embedding model host-side prevents direct index reads
        from bypassing date-capping.
        """
        if self.config.sandbox:
            if self._mcp_relay_sock is None:
                raise RuntimeError(
                    "MCP relay socket not initialized; "
                    "_start_mcp_relay must run before _build_mcp_invocation"
                )
            if self._mcp_bridge_wrapper is not None:
                return (str(self._mcp_bridge_wrapper), [])
            return (resolve_socat_cmd(), ["-", f"UNIX-CONNECT:{self._mcp_relay_sock}"])

        repo_root = str(Path(__file__).resolve().parents[2])
        venv_python = os.path.join(repo_root, ".venv", "bin", "python")
        python_cmd = venv_python if os.path.exists(venv_python) else "python"
        return (python_cmd, self._mcp_server_args(self._detect_embedding_server()))

    def _start_mcp_relay(self) -> None:
        """Start the idempotent host MCP relay used only with sandbox=True.

        The listener is `socat UNIX-LISTEN:<sock>,fork EXEC:<wrapper.sh>`;
        each sandbox connection forks a fresh mcp_server.py over wrapper stdio.
        The wrapper avoids socat's whitespace-split EXEC parser for paths with
        spaces.
        """
        if self._mcp_relay_proc is not None and self._mcp_relay_proc.poll() is None:
            return

        scratch = os.path.realpath(os.environ.get("_CONDOR_SCRATCH_DIR") or tempfile.gettempdir())
        run_tag = hashlib.sha1(
            str(Path(self._internal_dir).resolve()).encode()
        ).hexdigest()[:8]
        d = Path(scratch) / f"mcp_{self.agent_id}_{run_tag}"
        d.mkdir(parents=True, exist_ok=True)
        os.chmod(d, 0o700)
        self._mcp_relay_dir = d
        self._mcp_relay_sock = d / "mcp.sock"

        # Stale socket from a prior run will block bind; remove it.
        if self._mcp_relay_sock.exists() or self._mcp_relay_sock.is_symlink():
            self._mcp_relay_sock.unlink()

        repo_root = str(Path(__file__).resolve().parents[2])
        venv_python = os.path.join(repo_root, ".venv", "bin", "python")
        python_cmd = venv_python if os.path.exists(venv_python) else sys.executable
        mcp_args = self._mcp_server_args(self._detect_embedding_server())

        wrapper = d / "launch_mcp.sh"
        wrapper.write_text(
            "#!/bin/sh\nexec " + shlex.join([python_cmd, *mcp_args]) + "\n"
        )
        wrapper.chmod(0o755)

        bridge_wrapper = d / "connect_mcp.sh"
        write_mcp_connector(bridge_wrapper, self._mcp_relay_sock)
        self._mcp_bridge_wrapper = bridge_wrapper

        # Do not add EXEC `stderr`: it dup2s stderr onto stdout and corrupts
        # JSON-RPC. Omit it so stderr stays on socat's parent stderr, redirected
        # into mcp_relay.log below.
        cmd = [
            _resolve_socat_cmd(),
            f"UNIX-LISTEN:{self._mcp_relay_sock},fork",
            f"EXEC:{wrapper}",
        ]
        log_path = self._internal_dir / "mcp_relay.log"
        log_f = open(log_path, "a")
        logger.info("Starting MCP relay: %s", " ".join(cmd))
        self._mcp_relay_proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_f,
            stderr=log_f,
        )

        # socat creates the socket synchronously before accepting; wait up to 5s.
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if self._mcp_relay_sock.exists():
                return
            if self._mcp_relay_proc.poll() is not None:
                raise RuntimeError(
                    f"MCP relay died before listening (rc={self._mcp_relay_proc.returncode}); "
                    f"see {log_path}"
                )
            time.sleep(0.05)
        raise RuntimeError(
            f"MCP relay socket {self._mcp_relay_sock} did not appear within 5s; see {log_path}"
        )

    def _stop_mcp_relay(self) -> None:
        terminate_process(self._mcp_relay_proc)
        self._mcp_relay_proc = None
        if self._mcp_relay_sock and self._mcp_relay_sock.exists():
            try:
                self._mcp_relay_sock.unlink()
            except OSError:
                pass
        self._mcp_bridge_wrapper = None
