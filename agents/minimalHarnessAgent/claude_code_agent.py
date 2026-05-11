"""Claude Code backend for MinimalHarnessAgent."""

from __future__ import annotations

import logging
import os
from datetime import date
from typing import List, Optional

from agents.minimalHarnessAgent.agent import MinimalHarnessAgent

logger = logging.getLogger(__name__)


class ClaudeCodeAgent(MinimalHarnessAgent):
    """MinimalHarness backend that launches Claude Code sessions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._claude_code_session_id: Optional[str] = None

    def _respawns_each_day(self) -> bool:
        return self.config.prompt_mode in ("active_memory", "active_memory2", "no_memory")

    def _prepare_harness_launch(self) -> None:
        self._write_mcp_config()

    def _start_harness(self) -> None:
        self._start_claude_code()

    def _resume_harness(self, current_date: date) -> bool:
        return self._resume_claude_code(current_date)

    def _add_sandbox_harness_install_binds(self, args: List[str]) -> None:
        bin_real = os.path.realpath(self.config.claude_code_path)
        if os.path.exists(bin_real):
            install = os.path.dirname(os.path.dirname(bin_real))
            args.extend(["--ro-bind", install, install])

    def _sandbox_harness_home_subdirs(self) -> tuple[str, ...]:
        return (".claude", ".claude.json", ".cache/claude", ".cache/claude-cli-nodejs")

    def _start_claude_code(self) -> None:
        # Resume mode: if claude_code_resume is set and a previous session_id
        # is recoverable from claude_code_stdout.jsonl, reattach to that
        # conversation via `claude --resume <id>`. Falls back to a fresh spawn
        # when no prior session is found (safe default for new runs).
        resume_session_id: Optional[str] = None
        if self.config.claude_code_resume:
            resume_session_id = self._find_claude_code_session_id()
            if resume_session_id:
                self._claude_code_session_id = resume_session_id

        if resume_session_id:
            day_iso = (
                self._current_date.isoformat()
                if self._current_date else ""
            )
            initial_prompt = (
                f"Simulation resumed at {day_iso}. The previous session was "
                "interrupted; re-read state.json and market.csv for the "
                "current day's questions, resolutions, and your existing "
                "predictions, then continue research, update predictions "
                "where new info changes your view, and call next_day."
            )
        elif (
            self.config.prompt_mode in ("active_memory", "active_memory2")
            and self._initial_prompt_override
        ):
            initial_prompt = self._initial_prompt_override
        else:
            initial_prompt = (
                "Begin forecasting. Read market.csv to see your questions, "
                "then research and submit predictions."
            )
        # When sandboxed, pass the realpath of the claude binary as argv[0] so
        # the subprocess can exec it without needing the ~/.local/bin/claude
        # symlink to be preserved inside the sandbox.
        claude_bin = (
            os.path.realpath(self.config.claude_code_path)
            if self.config.sandbox
            else self.config.claude_code_path
        )
        cmd = [
            claude_bin,
        ]
        if resume_session_id:
            cmd.extend(["--resume", resume_session_id])
        if self.config.prompt_mode == "no_memory":
            disallowed_tools = "WebSearch,WebFetch,Write,Edit,MultiEdit,NotebookEdit"
        else:
            disallowed_tools = "WebSearch,WebFetch"

        cmd.extend([
            "-p", initial_prompt,
            "--verbose",
            "--effort", "max",
            "--model", self.config.model,
            "--output-format", "stream-json",
            "--dangerously-skip-permissions",
            "--mcp-config", str(self._internal_dir / "mcp_config.json"),
            "--strict-mcp-config",
            "--disallowedTools",
            disallowed_tools,
        ])
        # active_memory{,2} mode delivers the full daily instructions via -p, so
        # skip the standalone system prompt file (which isn't written in this
        # mode anyway).
        if self.config.prompt_mode not in ("active_memory", "active_memory2"):
            cmd.extend([
                "--system-prompt-file", str(self._internal_dir / "system_prompt.md"),
            ])
        cmd.extend([
            "--add-dir", str(self.workspace),
        ])

        if self.config.max_budget_usd is not None:
            cmd.extend(["--max-budget-usd", str(self.config.max_budget_usd)])

        cmd.extend(self.config.extra_flags)

        env = os.environ.copy()
        if self.config.anthropic_base_url:
            env["ANTHROPIC_BASE_URL"] = self.config.anthropic_base_url
        if self.config.anthropic_auth_token:
            env["ANTHROPIC_AUTH_TOKEN"] = self.config.anthropic_auth_token
        if self.config.network_isolation:
            env.update(self._egress_env_overrides())

        self._launch_harness_process(
            cmd,
            stdout_log_name="claude_code_stdout.jsonl",
            stderr_log_name="claude_code_stderr.log",
            env=env,
            label="Claude Code",
        )

    def _find_claude_code_session_id(self) -> Optional[str]:
        """Scan claude_code_stdout.jsonl for the first system/init session_id.

        claude emits {"type":"system","subtype":"init","session_id":"<uuid>",...}
        as its first event; we cache it so claude_code_resume mode can pass
        that uuid to `claude --resume <session_id>` on relaunch.
        """
        if self._claude_code_session_id:
            return self._claude_code_session_id
        self._claude_code_session_id = self._scan_jsonl_for_field(
            self._internal_dir / "claude_code_stdout.jsonl",
            lambda ev: ev.get("type") == "system" and ev.get("subtype") == "init",
            "session_id",
        )
        return self._claude_code_session_id

    def _resume_claude_code(self, current_date: date) -> bool:
        """Relaunch claude with `--resume <session_id>` after early exit.

        Reuses _start_claude_code's resume branch by temporarily forcing
        config.claude_code_resume=True. Returns False if no prior session_id
        is recoverable from claude_code_stdout.jsonl.
        """
        if not self._find_claude_code_session_id():
            logger.error("claude_code died and no session_id found — cannot resume")
            return False
        rc = self._harness_proc.returncode if self._harness_proc else "?"
        prev = self.config.claude_code_resume
        self.config.claude_code_resume = True
        try:
            self._start_claude_code()
        finally:
            self.config.claude_code_resume = prev
        logger.warning(
            "claude_code exited (rc=%s) — resumed session %s for %s",
            rc, self._claude_code_session_id, current_date,
        )
        return True
