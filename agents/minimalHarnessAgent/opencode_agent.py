"""OpenCode backend for MinimalHarnessAgent."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from datetime import date
from pathlib import Path
from typing import List, Optional

from futuresim_agents.minimalHarnessAgent.agent import MinimalHarnessAgent

logger = logging.getLogger(__name__)


class OpenCodeAgent(MinimalHarnessAgent):
    """MinimalHarness backend that launches OpenCode sessions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._session_id: Optional[str] = None

    def _respawns_each_day(self) -> bool:
        return self.config.prompt_mode in ("active_memory", "active_memory2", "no_memory")

    def _prepare_harness_launch(self) -> None:
        self._write_opencode_config()

    def _start_harness(self) -> None:
        self._start_opencode()

    def _resume_harness(self, current_date: date) -> bool:
        return self._resume_opencode(current_date)

    def _add_sandbox_harness_install_binds(self, args: List[str]) -> None:
        bin_real = os.path.realpath(self.config.opencode_path)
        if os.path.exists(bin_real):
            oc_bin_dir = os.path.dirname(bin_real)
            oc_install = os.path.dirname(oc_bin_dir)
            if os.path.basename(oc_install) == ".opencode":
                args.extend(["--ro-bind", oc_install, oc_install])
            else:
                args.extend(["--ro-bind", oc_bin_dir, oc_bin_dir])

    def _sandbox_harness_home_subdirs(self) -> tuple[str, ...]:
        return (".cache/opencode", ".local/share/opencode", ".config/opencode")

    def _write_opencode_config(self) -> None:
        """Write opencode.json into the workspace.

        Mirrors the Claude Code harness as closely as opencode permits:
        - MCP server key is `mcp__forecast_` so opencode's `<key>_<tool>`
          naming produces `mcp__forecast__search_news` etc., matching CC.
        - The forecasting system prompt is inlined into the custom agent
          `forecast` so the model receives the same instructions as CC.
          (opencode's built-in system prompt always prepends; this is an
          opencode limitation — agent.prompt augments rather than replaces.)
        - Web tools and opencode-only meta tools are disabled so the tool
          surface matches CC's forecasting scaffold as closely as possible.
        """
        mcp_cmd, mcp_args = self._build_mcp_invocation()

        # active_memory{,2} mode delivers the full daily instructions as the user
        # prompt each day (mirroring claude_code/codex). opencode requires a
        # non-empty agent.prompt, so use a minimal one-liner.
        if self.config.prompt_mode in ("active_memory", "active_memory2"):
            system_prompt = "You are a forecasting agent. Follow the per-day instructions in the user message."
        else:
            system_prompt = (self._internal_dir / "system_prompt.md").read_text()

        config = {
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                "openrouter": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "OpenRouter",
                    "options": {"baseURL": "https://openrouter.ai/api/v1"},
                    "models": {self.config.model: {"name": self.config.model}},
                }
            },
            "mcp": {
                "mcp__forecast_": {
                    "type": "local",
                    "command": [mcp_cmd] + mcp_args,
                    "enabled": True,
                }
            },
            # Disable opencode-only meta tools that CC's forecasting harness
            # doesn't expose, plus the web tools CC disallows explicitly.
            "tools": {
                "webfetch": False,
                "websearch": False,
                "task": False,
                "todowrite": False,
                "todoread": False,
                "question": False,
                "patch": False,
            },
            "agent": {
                "forecast": {
                    "prompt": system_prompt,
                    "mode": "primary",
                }
            },
        }
        with open(self.workspace / "opencode.json", "w") as f:
            json.dump(config, f, indent=2)

    def _opencode_variant_flags(self) -> List[str]:
        # Default opencode reasoning effort to high; callers can override by
        # passing --variant in extra_flags.
        if any(f == "--variant" for f in self.config.extra_flags):
            return []
        return ["--variant", "high"]

    def _start_opencode(self) -> None:
        if (
            self.config.prompt_mode in ("active_memory", "active_memory2")
            and self._initial_prompt_override
        ):
            initial_prompt = self._initial_prompt_override
        else:
            initial_prompt = (
                "Begin forecasting. Read market.csv to see your questions, "
                "then research and submit predictions."
            )
        model_id = f"openrouter/{self.config.model}"
        cmd = [
            self.config.opencode_path,
            "run",
            "--model", model_id,
            "--agent", "forecast",
            "--dangerously-skip-permissions",
            "--format", "json",
            "--print-logs",
            "--log-level", "INFO",
            initial_prompt,
        ]
        cmd.extend(self._opencode_variant_flags())
        cmd.extend(self.config.extra_flags)

        env = os.environ.copy()
        if self.config.openrouter_api_key:
            env["OPENROUTER_API_KEY"] = self.config.openrouter_api_key
        # Isolate opencode's SQLite session DB per agent. Must live on local
        # disk — shared FS (Lustre/NFS) breaks SQLite WAL ("disk I/O error").
        scratch = os.path.realpath(os.environ.get("_CONDOR_SCRATCH_DIR") or tempfile.gettempdir())
        run_tag = hashlib.sha1(
            str(Path(self._internal_dir).resolve()).encode()
        ).hexdigest()[:8]
        data_home = Path(scratch) / f"opencode_{self.agent_id}_{run_tag}"
        data_home.mkdir(parents=True, exist_ok=True)
        env["XDG_DATA_HOME"] = str(data_home)
        self._opencode_data_home = data_home
        if self.config.network_isolation:
            env.update(self._egress_env_overrides())

        self._launch_harness_process(
            cmd,
            stdout_log_name="opencode_stdout.jsonl",
            stderr_log_name="opencode_stderr.log",
            env=env,
            label="opencode",
        )

    def _find_session_id(self) -> Optional[str]:
        """Scan opencode_stdout.jsonl for the first sessionID field.

        opencode emits the sessionID on its first event after launch; once found
        we cache it so resumes use `opencode run --session <id>`.
        """
        if self._session_id:
            return self._session_id
        self._session_id = self._scan_jsonl_for_field(
            self._internal_dir / "opencode_stdout.jsonl",
            lambda ev: True,
            "sessionID",
        )
        return self._session_id

    def _resume_opencode(self, current_date: date) -> bool:
        """Relaunch opencode resuming the same session.

        Returns True if relaunch succeeded (new process running), False on
        unrecoverable error (no session id / exec failure).
        """
        session_id = self._find_session_id()
        if not session_id:
            logger.error("opencode died and no session_id found — cannot resume")
            return False

        model_id = f"openrouter/{self.config.model}"
        resume_prompt = "Please continue from where you left."
        cmd = [
            self.config.opencode_path,
            "run",
            "--session", session_id,
            "--model", model_id,
            "--agent", "forecast",
            "--dangerously-skip-permissions",
            "--format", "json",
            "--print-logs",
            "--log-level", "INFO",
            resume_prompt,
        ]
        cmd.extend(self._opencode_variant_flags())
        cmd.extend(self.config.extra_flags)

        env = os.environ.copy()
        if self.config.openrouter_api_key:
            env["OPENROUTER_API_KEY"] = self.config.openrouter_api_key
        env["XDG_DATA_HOME"] = str(self._opencode_data_home)
        if self.config.network_isolation:
            env.update(self._egress_env_overrides())

        logger.warning(
            "opencode exited (rc=%s) — resuming session %s for %s%s",
            self._harness_proc.returncode if self._harness_proc else "?",
            session_id, current_date,
            " [sandboxed]" if self.config.sandbox else "",
        )
        self._launch_harness_process(
            cmd,
            stdout_log_name="opencode_stdout.jsonl",
            stderr_log_name="opencode_stderr.log",
            env=env,
            label="opencode",
        )
        return True
