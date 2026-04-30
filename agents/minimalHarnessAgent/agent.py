"""
MinimalHarnessAgent — forecasting agent that delegates to a single persistent
harness CLI session (claude_code, opencode, or codex).

The first call to act() spawns the harness + its MCP server.  Subsequent calls
signal the MCP server (via a file-based continue signal) so the already-running
harness session can resume with the next simulation day.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.base import BaseAgent
from agents.minimalHarnessAgent.prompt import build_system_prompt
from environment.interfaces import PredictionSubmission

logger = logging.getLogger(__name__)


@dataclass
class MinimalHarnessConfig:
    model: str = "claude-opus-4-6"
    timeout_seconds: int = 7200
    max_budget_usd: Optional[float] = None  # None = unconstrained
    search_db: str = ""
    embedding_model: str = ""
    search_type: str = "hybrid"
    search_cutoff_days: int = 0
    articles_base: str = ""
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    claude_code_path: str = "claude"
    extra_flags: List[str] = field(default_factory=list)
    # "claude_code" (default), "opencode", or "codex".  When "opencode", `model`
    # should be a provider-qualified id like "deepseek/deepseek-v3.2" and the
    # call is routed through OpenRouter configured in the workspace-local
    # opencode.json.  When "codex", `model` is an OpenAI model id like "gpt-5.4"
    # and auth is taken from the user's ~/.codex login (CODEX_HOME).
    harness_backend: str = "claude_code"
    opencode_path: str = "opencode"
    codex_path: str = "codex"
    # Codex reasoning effort: "none" | "minimal" | "low" | "medium" | "high" |
    # "xhigh".  "xhigh" is the max level Codex accepts for gpt-5.x.
    reasoning_effort: str = "xhigh"
    # When true and harness_backend == "codex", resume codex's thread on each
    # subsequent simulation day via `codex exec resume <thread_id>` instead of
    # spawning a fresh `codex exec`. The model keeps yesterday's reasoning in
    # context (auto-compacted by codex when needed). Day 0 still spawns
    # normally; the thread_id is captured from the first session's stdout.
    codex_resume: bool = False
    # When true and harness_backend == "claude_code", relaunch claude with
    # `--resume <session_id>` instead of a fresh `-p "Begin forecasting..."`.
    # session_id is captured from the first system/init line of an existing
    # claude_code_stdout.jsonl (e.g., from a `--resume <output_dir>` env-level
    # restart). When no prior stdout/session_id is found, falls back to a fresh
    # spawn — so this flag is safe to leave on for new runs.
    claude_code_resume: bool = False
    # Optional: opencode reads OPENROUTER_API_KEY from env.  When set here, we
    # pass it into the subprocess environment instead of inheriting.
    openrouter_api_key: str = ""
    # Optional: route Claude Code through a non-Anthropic backend (e.g. z.ai for
    # GLM) by setting ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN in the subprocess
    # env. Empty => default Anthropic endpoint.
    anthropic_base_url: str = ""
    anthropic_auth_token: str = ""
    # When true, wrap the harness subprocess in bwrap so the filesystem is
    # restricted to just {workspace, _internal_dir, repo_root+venv, search_db,
    # harness runtime} plus /usr,/etc,/lib*. articles_base, dataset path,
    # sibling sim dirs, and /home (outside the harness' own state dirs) stay
    # invisible. Network is kept via --share-net so Anthropic/OpenRouter +
    # vLLM embedding server remain reachable. Articles are staged via hardlink
    # instead of symlink (required: workspace and articles_base must be on the
    # same filesystem). Supported for both claude_code and opencode backends.
    sandbox: bool = False


class MinimalHarnessAgent(BaseAgent):
    """Forecasting agent backed by a single persistent harness session
    (claude_code, opencode, or codex)."""

    def __init__(
        self,
        agent_id: str,
        config: MinimalHarnessConfig,
        search_tool: Any = None,
        agent_dir: str = "",
        articles_base: str = "",
    ):
        super().__init__(agent_id)
        self.config = config
        self.search_tool = search_tool
        # agent_dir is the top-level agent directory.
        # _internal_dir: driver/MCP coordination (signals, mcp_config, logs) — CC doesn't see this.
        # workspace: CC's working directory (--add-dir) — only what CC needs.
        self._internal_dir = Path(agent_dir)
        self.workspace = self._internal_dir / "workspace"
        self.articles_base = Path(articles_base or config.articles_base)
        self._harness_proc: Optional[subprocess.Popen] = None
        self._started = False
        self._last_symlink_date: Optional[date] = None
        self._current_date: Optional[date] = None
        self._session_id: Optional[str] = None  # opencode session id, captured from stdout
        self._resume_attempts: Dict[date, int] = {}
        self._codex_thread_id: Optional[str] = None  # codex thread_id, captured from Day 0 stdout
        self._claude_code_session_id: Optional[str] = None  # claude session_id, captured from Day 0 stdout

    # ── BaseAgent contract ─────────────────────────────────────────────

    def act(
        self,
        doc_interface: Any,
        forecast_interface: Any,
        current_date: date,
    ) -> List[Dict[str, Any]]:
        self._current_date = current_date

        # Internal dirs (driver/MCP coordination — harness doesn't see these).
        for d in ("signals",):
            (self._internal_dir / d).mkdir(parents=True, exist_ok=True)

        # Internal dirs (driver/MCP coordination — harness doesn't see these).
        (self._internal_dir / "predictions").mkdir(parents=True, exist_ok=True)

        # Workspace dirs (harness' working directory — only what the harness needs).
        for d in ("memory", "articles"):
            (self.workspace / d).mkdir(parents=True, exist_ok=True)

        # 1. Write state.json for MCP server to read.
        self._write_state(forecast_interface, current_date)

        # 2. chmod market.csv read-only so the harness can't corrupt it.
        self._protect_market_csv(forecast_interface)

        # 3. Update article symlinks up to current date.
        self._update_article_symlinks(current_date)

        # codex has no persistent session: each `codex exec` runs once and
        # exits. So we re-spawn codex (and its child MCP server) every day.
        # claude_code and opencode use a single long-lived process whose
        # MCP server polls for a per-day continue signal.
        codex_per_day = self.config.harness_backend == "codex"

        if not self._started or codex_per_day:
            if codex_per_day and self._started:
                # Reap the previous day's codex if still alive (should have
                # exited after writing next_day, but be defensive).
                if self._harness_proc and self._harness_proc.poll() is None:
                    self._harness_proc.terminate()
                    try:
                        self._harness_proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self._harness_proc.kill()
                # Clear stale next_day signals so _wait_for_next_day blocks
                # until the freshly-spawned codex writes a new one.
                signal_dir = self._internal_dir / "signals"
                signal_dir.mkdir(parents=True, exist_ok=True)
                for old in signal_dir.glob("next_day_*.json"):
                    old.unlink(missing_ok=True)

            self._write_system_prompt(forecast_interface)
            if self.config.harness_backend == "opencode":
                self._write_opencode_config()
                self._start_opencode()
            elif codex_per_day:
                self._start_codex()
            else:
                self._write_mcp_config()
                self._start_claude_code()
            self._started = True
        else:
            # Subsequent days for persistent backends — unblock the MCP
            # server's next_day poll.
            self._write_continue_signal(current_date)

        # 4. Wait for the harness to call next_day (MCP server writes signal).
        predictions = self._wait_for_next_day(current_date)

        # codex_resume mode: capture thread_id from Day 0 stdout so
        # subsequent days can run `codex exec resume <thread_id>`.
        if (
            self.config.harness_backend == "codex"
            and self.config.codex_resume
            and self._codex_thread_id is None
        ):
            self._find_codex_thread_id()

        # 5. Submit predictions to the environment.
        actions: List[Dict[str, Any]] = []
        for pred in predictions:
            try:
                forecast_interface.submit_prediction(PredictionSubmission(
                    question_id=pred["question_id"],
                    outcomes=pred["outcomes"],
                ))
                actions.append(pred)
            except Exception as e:
                logger.warning("Failed to submit prediction %s: %s", pred, e)

        # 6. Restore market.csv write permission so the env can overwrite next day.
        self._unprotect_market_csv()

        # 7. Signal day complete.
        forecast_interface.next_day()
        return actions

    # ── state / config writers ─────────────────────────────────────────

    @staticmethod
    def _to_iso(d) -> str:
        """Convert a date, datetime, or string to ISO date string."""
        if d is None:
            return ""
        if hasattr(d, "isoformat"):
            return d.isoformat()
        return str(d)

    def _write_state(self, forecast_interface: Any, current_date: date) -> None:
        """Serialize simulation state for the MCP server."""
        questions = []
        for q in forecast_interface.list_questions():
            questions.append({
                "qid": q.qid,
                "title": q.title,
                "background": q.background,
                "resolution_criteria": q.resolution_criteria,
                "answer_type": q.answer_type,
                "resolution_date": str(q.resolution_date) if q.resolution_date else None,
                "options": q.options,
            })

        # Include agent's prediction distributions so the MCP server can show
        # the full distribution in resolution feedback (matching BasicAgent).
        agent_predictions = {}
        get_preds = getattr(forecast_interface, "get_agent_predictions", None)
        if callable(get_preds):
            try:
                raw = get_preds(self.agent_id) or {}
                for qid, record in raw.items():
                    if isinstance(record, dict) and record.get("outcomes"):
                        agent_predictions[qid] = record["outcomes"]
            except Exception:
                pass

        # Count total active predictions.
        total_predictions = len(agent_predictions)

        state = {
            "current_date": current_date.isoformat(),
            "start_date": self._to_iso(self.config.start_date or current_date),
            "end_date": self._to_iso(self.config.end_date or current_date),
            "market_csv": forecast_interface.get_market_csv_path(),
            "search_db": self.config.search_db,
            "embedding_model": self.config.embedding_model,
            "search_type": self.config.search_type,
            "search_cutoff_days": self.config.search_cutoff_days,
            "timeout_seconds": self.config.timeout_seconds,
            "questions": questions,
            "resolution_events": forecast_interface.resolution_events,
            "agent_predictions": agent_predictions,
            "total_predictions": total_predictions,
        }
        state_path = self._internal_dir / "state.json"
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def _protect_market_csv(self, forecast_interface: Any) -> None:
        csv_path = forecast_interface.get_market_csv_path()
        if csv_path and os.path.exists(csv_path):
            self._market_csv_path = csv_path
            os.chmod(csv_path, 0o444)
            # Expose market.csv inside workspace. With sandbox=False a symlink
            # is fine. With sandbox=True the run_dir parent is not bound into
            # the sandbox, so a symlink would resolve to a nonexistent path;
            # hardlink instead (pd.to_csv truncates in place, so the link
            # stays in sync across daily rewrites).
            link = self.workspace / "market.csv"
            if not link.exists() and not link.is_symlink():
                if self.config.sandbox:
                    try:
                        os.link(csv_path, link)
                    except OSError as e:
                        if e.errno == 18:  # EXDEV
                            raise RuntimeError(
                                f"sandbox=True requires workspace and market.csv on the same filesystem "
                                f"(hardlink failed: {csv_path} -> {link})."
                            ) from e
                        raise
                else:
                    link.symlink_to(csv_path)

    def _unprotect_market_csv(self) -> None:
        """Restore write permission so the env can overwrite on the next day."""
        csv_path = getattr(self, "_market_csv_path", None)
        if csv_path and os.path.exists(csv_path):
            os.chmod(csv_path, 0o644)

    def _write_mcp_config(self) -> None:
        """Write the MCP config JSON that Claude Code will use to spawn the server.

        We pass the repo root via the server's own --repo-root flag rather than
        using the 'env' key in the MCP config, because Claude Code replaces
        (rather than merges) the process environment when 'env' is set.
        """
        repo_root = str(Path(__file__).resolve().parents[2])
        mcp_server_script = str(Path(__file__).resolve().parent / "mcp_server.py")
        venv_python = os.path.join(repo_root, ".venv", "bin", "python")
        python_cmd = venv_python if os.path.exists(venv_python) else "python"

        # Detect running vLLM embedding server so the MCP server can reuse it.
        embed_server_url = self._detect_embedding_server()

        mcp_args = [
            mcp_server_script,
            "--workspace", str(self.workspace),
            "--internal-dir", str(self._internal_dir),
            "--repo-root", repo_root,
            "--search-db", self.config.search_db or "",
            "--embedding-model", self.config.embedding_model or "",
        ]
        if embed_server_url:
            mcp_args.extend(["--embedding-server-url", embed_server_url])

        config = {
            "mcpServers": {
                "forecast": {
                    "command": python_cmd,
                    "args": mcp_args,
                }
            }
        }
        config_path = self._internal_dir / "mcp_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    def _write_system_prompt(self, forecast_interface: Any = None) -> None:
        source_context = ""
        source_name = "openforesight"
        num_questions = 0
        num_active = 0
        num_resolved = 0
        new_articles_count = None
        last_active_date = None
        next_active_date = None
        if forecast_interface is not None:
            source_context = getattr(forecast_interface, "source_context", "")
            source_name = getattr(forecast_interface, "source_name", "openforesight")
            questions = forecast_interface.list_questions()
            num_active = len(questions)
            num_resolved = len(getattr(forecast_interface, "resolved_questions", []))
            num_questions = num_active + num_resolved
            last_active_date = getattr(forecast_interface, "last_active_date", None)
            next_active_date = getattr(forecast_interface, "next_active_date", None)
            prompt_date = self.config.start_date or self._current_date
            new_articles_count = self._count_new_articles(last_active_date, prompt_date)
        prompt = build_system_prompt(
            workspace=str(self.workspace),
            current_date=self.config.start_date or self._current_date,
            start_date=self.config.start_date or self._current_date,
            end_date=self.config.end_date or self._current_date,
            source_context=source_context,
            source_name=source_name,
            num_questions=num_questions,
            num_active=num_active,
            num_resolved=num_resolved,
            search_cutoff_days=self.config.search_cutoff_days,
            new_articles_count=new_articles_count,
            last_active_date=last_active_date,
            next_active_date=next_active_date,
        )
        prompt_path = self._internal_dir / "system_prompt.md"
        with open(prompt_path, "w") as f:
            f.write(prompt)

    def _count_new_articles(
        self,
        last_active_date: Optional[date],
        current_date: Optional[date],
    ) -> Optional[int]:
        """Match BasicAgent's article-count window when search supports it."""
        if (
            self.search_tool is None
            or not getattr(self.search_tool, "is_available", False)
            or last_active_date is None
            or current_date is None
        ):
            return None
        max_date = current_date - timedelta(days=self.config.search_cutoff_days)
        try:
            return self.search_tool.count_articles(
                min_date=last_active_date,
                max_date=max_date,
            )
        except Exception:
            return None

    def _detect_embedding_server(self) -> str:
        """Check if a vLLM embedding server is already running (started by the driver).
        Uses 127.0.0.1 to bypass any HTTP proxy that intercepts 'localhost'.
        Returns the server URL or empty string."""
        import urllib.request
        # Bypass proxy for local connections.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for port in range(8001, 8010):
            url = f"http://127.0.0.1:{port}"
            try:
                req = urllib.request.Request(f"{url}/v1/models", method="GET")
                with opener.open(req, timeout=2) as resp:
                    if resp.status == 200:
                        logger.info("Found running embedding server at %s", url)
                        return url
            except Exception:
                continue
        return ""

    # ── article staging ────────────────────────────────────────────────

    def _update_article_symlinks(self, current_date: date) -> None:
        """Incrementally expose per-day articles.jsonl files up to current_date.

        When sandbox=False, uses symlinks (fast, but requires articles_base
        to be accessible in the process's filesystem view).
        When sandbox=True, uses hardlinks so the files appear as real entries
        inside the workspace without needing articles_base to be bound — this
        keeps articles_base invisible from inside the sandbox, enforcing the
        date-gate via the driver instead of by filesystem ACLs.

        Only articles.jsonl is exposed; the sibling parquet and headlines JSON
        files are intentionally omitted (parquet isn't greppable and the
        headlines JSON duplicates info available in articles.jsonl)."""
        if not self.articles_base or not self.articles_base.exists():
            return

        articles_dir = self.workspace / "articles"
        start = (self._last_symlink_date + timedelta(days=1)) if self._last_symlink_date else (
            self.config.start_date or current_date - timedelta(days=30)
        )

        d = start
        while d <= current_date:
            src = self.articles_base / f"{d.year}" / f"{d.month:02d}" / f"{d.day:02d}" / "articles.jsonl"
            if src.exists():
                dst = articles_dir / f"{d.year}" / f"{d.month:02d}" / f"{d.day:02d}" / "articles.jsonl"
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    if self.config.sandbox:
                        try:
                            os.link(src, dst)
                        except OSError as e:
                            if e.errno == 18:  # EXDEV
                                raise RuntimeError(
                                    f"sandbox=True requires workspace and articles_base on the same filesystem "
                                    f"(hardlink failed: {src} -> {dst}). Move one of them or disable sandbox."
                                ) from e
                            raise
                    else:
                        dst.symlink_to(src)
            d += timedelta(days=1)

        self._last_symlink_date = current_date

    # ── Claude Code subprocess ─────────────────────────────────────────

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
        cmd.extend([
            "-p", initial_prompt,
            "--verbose",
            "--effort", "max",
            "--model", self.config.model,
            "--output-format", "stream-json",
            "--dangerously-skip-permissions",
            "--mcp-config", str(self._internal_dir / "mcp_config.json"),
            "--strict-mcp-config",
            "--disallowedTools", "WebSearch,WebFetch",
            "--system-prompt-file", str(self._internal_dir / "system_prompt.md"),
            "--add-dir", str(self.workspace),
        ])

        if self.config.max_budget_usd is not None:
            cmd.extend(["--max-budget-usd", str(self.config.max_budget_usd)])

        cmd.extend(self.config.extra_flags)

        # Append mode so a resume relaunch doesn't truncate the prior stdout
        # (which is what we read session_id from on subsequent restarts).
        log_stdout = open(self._internal_dir / "claude_code_stdout.jsonl", "a")
        log_stderr = open(self._internal_dir / "claude_code_stderr.log", "a")

        env = os.environ.copy()
        if self.config.anthropic_base_url:
            env["ANTHROPIC_BASE_URL"] = self.config.anthropic_base_url
        if self.config.anthropic_auth_token:
            env["ANTHROPIC_AUTH_TOKEN"] = self.config.anthropic_auth_token

        launch_cmd = self._maybe_sandbox(cmd)
        # When sandboxed, bwrap sets cwd via --chdir; passing cwd= to Popen
        # would chdir in the host namespace before exec and can fail if the
        # path is visible only through symlinks that bwrap reshapes.
        popen_cwd = None if self.config.sandbox else str(self.workspace)

        logger.info("Starting Claude Code%s: %s",
                    " [sandboxed]" if self.config.sandbox else "",
                    " ".join(launch_cmd))
        self._harness_proc = subprocess.Popen(
            launch_cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_stdout,
            stderr=log_stderr,
            cwd=popen_cwd,
            env=env,
        )
        self._stdout_log = log_stdout
        self._stderr_log = log_stderr

    # ── opencode backend ───────────────────────────────────────────────

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
        repo_root = str(Path(__file__).resolve().parents[2])
        mcp_server_script = str(Path(__file__).resolve().parent / "mcp_server.py")
        venv_python = os.path.join(repo_root, ".venv", "bin", "python")
        python_cmd = venv_python if os.path.exists(venv_python) else "python"

        embed_server_url = self._detect_embedding_server()

        mcp_args = [
            mcp_server_script,
            "--workspace", str(self.workspace),
            "--internal-dir", str(self._internal_dir),
            "--repo-root", repo_root,
            "--search-db", self.config.search_db or "",
            "--embedding-model", self.config.embedding_model or "",
        ]
        if embed_server_url:
            mcp_args.extend(["--embedding-server-url", embed_server_url])

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
                    "command": [python_cmd] + mcp_args,
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

    def _build_bwrap_cmd(self, inner_cmd: List[str]) -> List[str]:
        """Wrap `inner_cmd` in bwrap for filesystem isolation.

        The sandbox hides everything by default and only exposes:
          - workspace (rw) — harness cwd; articles/, memory/, market.csv,
            plus any files the model writes
          - _internal_dir (rw) — MCP signals/, predictions/, state.json
          - repo_root (ro) — agents/, environment/, .venv symlink
          - realpath(venv) (ro) — when .venv is symlinked out of the repo
          - sys.base_prefix (ro) — Python interpreter + stdlib
          - search_db (ro) — LanceDB articles index
          - harness binary tree (ro), harness state dirs under ~ (rw):
              claude_code: ~/.claude, ~/.claude.json, ~/.cache/claude*
              opencode:    ~/.cache|.local|.config/opencode plus an
                           XDG_DATA_HOME scratch for its SQLite session DB
          - /usr, /etc, /lib, /lib64, /bin, /sbin (ro) — OS tools/libs
          - /proc (new namespace), /dev (stripped), /tmp (tmpfs)

        NOT bound (stays invisible to the model):
          - articles_base — date-gating is enforced by hardlinks we placed in
            workspace/articles; articles_base itself is unreachable
          - FSIM_DATASET_PATH — HF dataset with resolution labels
          - sibling sim dirs, daily_metrics.csv, test_daily_metrics.csv,
            actions.jsonl, matcher.jsonl in parent output dir
          - /home outside explicitly bound harness subdirs

        Every path we care about is bound at its realpath, and top-level
        symlink aliases (/fast, /is/cluster/fast) are recreated via
        --symlink so code using aliased paths still resolves.

        Network is kept via --share-net so OpenRouter + local vLLM embedding
        server at 127.0.0.1:800x remain reachable.
        """
        repo_root_real = os.path.realpath(str(Path(__file__).resolve().parents[2]))
        workspace_real = os.path.realpath(str(self.workspace))
        internal_real = os.path.realpath(str(self._internal_dir))

        args = [
            "bwrap",
            "--share-net",
            "--die-with-parent",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--proc", "/proc",
            "--dev", "/dev",
            "--tmpfs", "/tmp",
            "--ro-bind", "/usr", "/usr",
            "--ro-bind", "/etc", "/etc",
            "--ro-bind-try", "/lib", "/lib",
            "--ro-bind-try", "/lib64", "/lib64",
            "--ro-bind-try", "/bin", "/bin",
            "--ro-bind-try", "/sbin", "/sbin",
        ]

        # Recreate top-level path aliases BEFORE any binds whose DEST would
        # implicitly create the alias dir as tmpfs. Each alias only resolves
        # to paths we separately bind below, so this leaks no data.
        # (e.g. on this cluster /home -> /lustre/home, /fast -> /lustre/fast/fast.)
        for alias in ("/home", "/fast", "/is/cluster/fast"):
            if os.path.islink(alias):
                target = os.readlink(alias)
                if not os.path.isabs(target):
                    target = os.path.normpath(os.path.join(os.path.dirname(alias), target))
                # Strip any trailing slash that readlink may return.
                target = target.rstrip("/") or "/"
                args.extend(["--symlink", target, alias])

        # Workspace + internal dir (rw), at realpath so DEST dirs don't clash
        # with the alias symlinks above.
        args.extend(["--bind", workspace_real, workspace_real])
        args.extend(["--bind", internal_real, internal_real])

        # Overlay workspace/articles as read-only. Articles are hardlinked from
        # articles_base — since hardlinks share permissions with the source,
        # a write through the hardlink would corrupt articles_base. The ro
        # overlay blocks writes regardless. Driver writes on the host side are
        # unaffected (they go straight to the filesystem, not through bwrap).
        articles_dst = os.path.join(workspace_real, "articles")
        if os.path.exists(articles_dst):
            args.extend(["--ro-bind", articles_dst, articles_dst])

        # Repo (ro).
        args.extend(["--ro-bind", repo_root_real, repo_root_real])

        # If .venv is a symlink out of the repo (common when the venv lives on
        # fast storage), bind the real venv location too.
        venv = os.path.join(repo_root_real, ".venv")
        if os.path.islink(venv):
            venv_real = os.path.realpath(venv)
            if venv_real and os.path.exists(venv_real):
                args.extend(["--ro-bind", venv_real, venv_real])

        # Python interpreter + stdlib (base_prefix of the venv).
        base_prefix = os.path.realpath(sys.base_prefix) if sys.base_prefix else ""
        if base_prefix and os.path.exists(base_prefix) and base_prefix != repo_root_real:
            args.extend(["--ro-bind", base_prefix, base_prefix])

        # LanceDB search index.
        if self.config.search_db:
            sdb_real = os.path.realpath(self.config.search_db)
            if os.path.exists(sdb_real):
                args.extend(["--ro-bind", sdb_real, sdb_real])

        # Embedding model path (used by LanceDB as a fallback when no vLLM
        # embedding server is detected; harmless to bind when unused).
        if self.config.embedding_model:
            em_real = os.path.realpath(self.config.embedding_model)
            if os.path.exists(em_real):
                args.extend(["--ro-bind", em_real, em_real])

        # Harness binary install tree (ro).
        if self.config.harness_backend == "claude_code":
            # claude lives at <install>/versions/<v>, where <install> is
            # ~/.local/share/claude. Bind the install dir; _start_claude_code
            # passes the realpath as argv[0] so the bin symlink isn't needed.
            bin_real = os.path.realpath(self.config.claude_code_path)
            if os.path.exists(bin_real):
                install = os.path.dirname(os.path.dirname(bin_real))
                args.extend(["--ro-bind", install, install])
            home_subs = (".claude", ".claude.json",
                         ".cache/claude", ".cache/claude-cli-nodejs")
        elif self.config.harness_backend == "codex":
            # codex lives at <install>/versions/<v>/codex (e.g.
            # ~/.local/share/codex/versions/0.125.0/codex). Bind the install
            # dir RO. Auth/session/cache live in ~/.codex (RW).
            bin_real = os.path.realpath(self.config.codex_path)
            if os.path.exists(bin_real):
                install = os.path.dirname(os.path.dirname(bin_real))
                args.extend(["--ro-bind", install, install])
            home_subs = (".codex",)
        else:
            bin_real = os.path.realpath(self.config.opencode_path)
            if os.path.exists(bin_real):
                oc_bin_dir = os.path.dirname(bin_real)
                oc_install = os.path.dirname(oc_bin_dir)
                if os.path.basename(oc_install) == ".opencode":
                    args.extend(["--ro-bind", oc_install, oc_install])
                else:
                    args.extend(["--ro-bind", oc_bin_dir, oc_bin_dir])
            home_subs = (".cache/opencode", ".local/share/opencode", ".config/opencode")

        # Harness runtime state in user home (reads+writes).
        home = os.path.expanduser("~")
        for sub in home_subs:
            p = os.path.realpath(os.path.join(home, sub))
            if os.path.exists(p):
                args.extend(["--bind-try", p, p])

        # Per-agent XDG_DATA_HOME scratch for opencode's SQLite session DB.
        data_home = getattr(self, "_opencode_data_home", None)
        if data_home:
            dh = os.path.realpath(str(data_home))
            if os.path.exists(dh):
                args.extend(["--bind", dh, dh])

        # chdir inside the sandbox to the workspace realpath (matches the bind).
        args.extend(["--chdir", workspace_real])
        args.append("--")
        args.extend(inner_cmd)
        return args

    def _maybe_sandbox(self, cmd: List[str]) -> List[str]:
        """Return cmd wrapped in bwrap when sandbox is enabled, else unchanged."""
        if self.config.sandbox:
            return self._build_bwrap_cmd(cmd)
        return cmd

    def _start_opencode(self) -> None:
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
        scratch = os.environ.get("_CONDOR_SCRATCH_DIR") or tempfile.gettempdir()
        data_home = Path(scratch) / f"opencode_{self.agent_id}"
        data_home.mkdir(parents=True, exist_ok=True)
        env["XDG_DATA_HOME"] = str(data_home)
        self._opencode_data_home = data_home

        log_stdout = open(self._internal_dir / "opencode_stdout.jsonl", "w")
        log_stderr = open(self._internal_dir / "opencode_stderr.log", "w")

        launch_cmd = self._maybe_sandbox(cmd)
        # When sandboxed, bwrap sets cwd via --chdir; passing cwd= to Popen
        # would chdir in the host namespace before exec and can fail if the
        # path is visible only through symlinks that bwrap reshapes.
        popen_cwd = None if self.config.sandbox else str(self.workspace)

        logger.info("Starting opencode%s: %s",
                    " [sandboxed]" if self.config.sandbox else "",
                    " ".join(launch_cmd))
        self._harness_proc = subprocess.Popen(
            launch_cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_stdout,
            stderr=log_stderr,
            cwd=popen_cwd,
            env=env,
        )
        self._stdout_log = log_stdout
        self._stderr_log = log_stderr

    # ── codex backend ──────────────────────────────────────────────────

    def _start_codex(self) -> None:
        """Spawn a `codex exec` process for the current sim day.

        Codex doesn't have a persistent `--continue` mode like Claude Code:
        `codex exec` runs once and exits.  Two strategies are supported:

        - Default (codex_resume=False): act() calls _start_codex each day
          with a freshly-rewritten state.json; every day starts a NEW codex
          thread that picks up workspace state from disk only.

        - codex_resume=True: Day 0 starts a fresh thread (captured into
          self._codex_thread_id), and subsequent days run
          `codex exec resume <thread_id>` so the model retains yesterday's
          reasoning in context (codex auto-compacts when needed).

        System prompt: codex auto-loads `AGENTS.md` from cwd, so we copy
        system_prompt.md → workspace/AGENTS.md.
        MCP server: configured via `-c mcp_servers.forecast.command/args`.
        """
        repo_root = str(Path(__file__).resolve().parents[2])
        mcp_server_script = str(Path(__file__).resolve().parent / "mcp_server.py")
        venv_python = os.path.join(repo_root, ".venv", "bin", "python")
        python_cmd = venv_python if os.path.exists(venv_python) else "python"

        embed_server_url = self._detect_embedding_server()

        mcp_args = [
            mcp_server_script,
            "--workspace", str(self.workspace),
            "--internal-dir", str(self._internal_dir),
            "--repo-root", repo_root,
            "--search-db", self.config.search_db or "",
            "--embedding-model", self.config.embedding_model or "",
        ]
        if embed_server_url:
            mcp_args.extend(["--embedding-server-url", embed_server_url])

        # Codex reads AGENTS.md from cwd at startup as user instructions.
        system_prompt = (self._internal_dir / "system_prompt.md").read_text()
        (self.workspace / "AGENTS.md").write_text(system_prompt)

        # When sandboxed, pass the realpath of the codex binary as argv[0] so
        # the subprocess can exec it without needing the ~/.local/bin/codex
        # symlink to be preserved inside the sandbox.
        codex_bin = (
            os.path.realpath(self.config.codex_path)
            if self.config.sandbox
            else self.config.codex_path
        )

        # `codex exec` accepts -C (cwd); `codex exec resume` does NOT — it
        # inherits cwd from the spawning process instead. Keep -C in the
        # fresh-spawn branch only; resume relies on Popen cwd / bwrap --chdir.
        common_args = [
            "-m", self.config.model,
            "-c", f'model_reasoning_effort="{self.config.reasoning_effort}"',
            "-c", f'mcp_servers.forecast.command="{python_cmd}"',
            "-c", f"mcp_servers.forecast.args={json.dumps(mcp_args)}",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "--json",
        ]

        use_resume = (
            self.config.codex_resume
            and self._codex_thread_id is not None
        )

        if use_resume:
            day_iso = (
                self._current_date.isoformat()
                if self._current_date else ""
            )
            resume_prompt = (
                f"Simulation advanced to {day_iso}. Re-read state.json and "
                "market.csv for the new day's questions, resolution events, "
                "and your existing predictions; update predictions where "
                "new info changes your view, then call next_day."
            )
            cmd = [
                codex_bin, "exec", "resume", self._codex_thread_id,
                *common_args, resume_prompt,
            ]
        else:
            initial_prompt = (
                "Begin forecasting. Read market.csv to see your questions, "
                "then research and submit predictions."
            )
            cmd = [
                codex_bin, "exec",
                *common_args,
                "-C", str(self.workspace),
                initial_prompt,
            ]
        cmd.extend(self.config.extra_flags)

        # Append mode so multi-day re-spawns don't truncate prior days' logs.
        log_stdout = open(self._internal_dir / "codex_stdout.jsonl", "a")
        log_stderr = open(self._internal_dir / "codex_stderr.log", "a")

        env = os.environ.copy()

        launch_cmd = self._maybe_sandbox(cmd)
        # When sandboxed, bwrap sets cwd via --chdir; passing cwd= to Popen
        # would chdir in the host namespace before exec.
        popen_cwd = None if self.config.sandbox else str(self.workspace)

        logger.info("Starting codex%s: %s",
                    " [sandboxed]" if self.config.sandbox else "",
                    " ".join(launch_cmd))
        self._harness_proc = subprocess.Popen(
            launch_cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_stdout,
            stderr=log_stderr,
            cwd=popen_cwd,
            env=env,
        )
        self._stdout_log = log_stdout
        self._stderr_log = log_stderr

    def _find_session_id(self) -> Optional[str]:
        """Scan opencode_stdout.jsonl for the first sessionID field.

        opencode emits the sessionID on its first event after launch; once found
        we cache it so resumes use `opencode run --session <id>`.
        """
        if self._session_id:
            return self._session_id
        path = self._internal_dir / "opencode_stdout.jsonl"
        if not path.exists():
            return None
        with open(path) as f:
            for line in f:
                try:
                    sid = json.loads(line).get("sessionID")
                except Exception:
                    continue
                if sid:
                    self._session_id = sid
                    return sid
        return None

    def _find_claude_code_session_id(self) -> Optional[str]:
        """Scan claude_code_stdout.jsonl for the first system/init session_id.

        claude emits {"type":"system","subtype":"init","session_id":"<uuid>",...}
        as its first event; we cache it so claude_code_resume mode can pass
        that uuid to `claude --resume <session_id>` on relaunch.
        """
        if self._claude_code_session_id:
            return self._claude_code_session_id
        path = self._internal_dir / "claude_code_stdout.jsonl"
        if not path.exists():
            return None
        with open(path) as f:
            for line in f:
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if ev.get("type") == "system" and ev.get("subtype") == "init":
                    sid = ev.get("session_id")
                    if sid:
                        self._claude_code_session_id = sid
                        return sid
        return None

    def _find_codex_thread_id(self) -> Optional[str]:
        """Scan codex_stdout.jsonl for the first thread.started event.

        codex emits {"type":"thread.started","thread_id":"<uuid>"} as its
        first event; we cache it so codex_resume mode can pass that uuid to
        `codex exec resume <thread_id>` on subsequent days.
        """
        if self._codex_thread_id:
            return self._codex_thread_id
        path = self._internal_dir / "codex_stdout.jsonl"
        if not path.exists():
            return None
        with open(path) as f:
            for line in f:
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if ev.get("type") == "thread.started":
                    tid = ev.get("thread_id")
                    if tid:
                        self._codex_thread_id = tid
                        return tid
        return None

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

        try:
            self._stdout_log.close()
            self._stderr_log.close()
        except Exception:
            pass
        log_stdout = open(self._internal_dir / "opencode_stdout.jsonl", "a")
        log_stderr = open(self._internal_dir / "opencode_stderr.log", "a")

        launch_cmd = self._maybe_sandbox(cmd)
        popen_cwd = None if self.config.sandbox else str(self.workspace)

        logger.warning(
            "opencode exited (rc=%s) — resuming session %s for %s%s",
            self._harness_proc.returncode if self._harness_proc else "?",
            session_id, current_date,
            " [sandboxed]" if self.config.sandbox else "",
        )
        self._harness_proc = subprocess.Popen(
            launch_cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_stdout,
            stderr=log_stderr,
            cwd=popen_cwd,
            env=env,
        )
        self._stdout_log = log_stdout
        self._stderr_log = log_stderr
        return True

    # ── signal coordination ────────────────────────────────────────────

    def _wait_for_next_day(self, current_date: date) -> List[Dict[str, Any]]:
        """Poll for the MCP server's next_day signal, then return predictions.

        Reads predictions from the signal file (authoritative snapshot from
        _today_predictions at the moment next_day was called), falling back to
        the predictions/{date}.json file if the signal doesn't contain them.
        """
        signal_path = self._internal_dir / "signals" / f"next_day_{current_date.isoformat()}.json"
        poll_interval = 1.0
        timeout = self.config.timeout_seconds
        start = time.time()

        MAX_RESUMES_PER_DAY = 3

        while not signal_path.exists():
            # Check if harness process died.
            if self._harness_proc and self._harness_proc.poll() is not None:
                if self.config.harness_backend == "opencode":
                    attempts = self._resume_attempts.get(current_date, 0)
                    if attempts < MAX_RESUMES_PER_DAY and self._resume_opencode(current_date):
                        self._resume_attempts[current_date] = attempts + 1
                        continue
                    logger.error(
                        "opencode exited and could not be resumed for %s (attempts=%d)",
                        current_date, attempts,
                    )
                else:
                    logger.warning(
                        "%s exited with code %d before signaling next_day",
                        self.config.harness_backend,
                        self._harness_proc.returncode,
                    )
                return self._read_predictions(current_date)
            if time.time() - start > timeout:
                logger.error("Timeout waiting for next_day signal on %s", current_date)
                return self._read_predictions(current_date)
            time.sleep(poll_interval)

        # Read predictions from the predictions file — this is the durable record
        # of all submit_forecasts MCP calls. Each call writes to this file immediately,
        # so it survives MCP server restarts (unlike the in-memory _today_predictions).
        return self._read_predictions(current_date)

    def _read_predictions(self, current_date: date) -> List[Dict[str, Any]]:
        pred_path = self._internal_dir / "predictions" / f"{current_date.isoformat()}.json"
        if not pred_path.exists():
            logger.warning("No predictions file for %s", current_date)
            return []
        with open(pred_path) as f:
            preds = json.load(f)
        # Normalize: CC may use "question_id" or "qid" depending on whether
        # predictions were submitted via MCP or written directly via Bash.
        for p in preds:
            if "qid" in p and "question_id" not in p:
                p["question_id"] = p["qid"]
        return preds

    def _write_continue_signal(self, current_date: date) -> None:
        """Tell the MCP server's next_day poll that a new day is ready."""
        signal_dir = self._internal_dir / "signals"
        # Clean up old signals to avoid confusion.
        for old in signal_dir.glob("continue_*.json"):
            old.unlink(missing_ok=True)
        for old in signal_dir.glob("next_day_*.json"):
            old.unlink(missing_ok=True)

        signal_path = signal_dir / f"continue_{current_date.isoformat()}.json"
        with open(signal_path, "w") as f:
            json.dump({"status": "day_advanced", "date": current_date.isoformat()}, f)

    def signal_complete(self) -> None:
        """Write a simulation_complete continue signal so the MCP server's
        next_day poll unblocks and tells the harness the sim is over."""
        if not self._started:
            return
        signal_dir = self._internal_dir / "signals"
        signal_dir.mkdir(parents=True, exist_ok=True)
        # Clean old signals.
        for old in signal_dir.glob("continue_*.json"):
            old.unlink(missing_ok=True)
        for old in signal_dir.glob("next_day_*.json"):
            old.unlink(missing_ok=True)
        # Write a terminal state.json so the MCP server can read it.
        end_state = {
            "current_date": self._to_iso(self._current_date or date.today()),
            "start_date": self._to_iso(self.config.start_date or date.today()),
            "end_date": self._to_iso(self.config.end_date or date.today()),
            "questions": [],
            "resolution_events": [],
        }
        with open(self._internal_dir / "state.json", "w") as f:
            json.dump(end_state, f, default=str)
        cur = self._to_iso(self._current_date or date.today())
        signal_path = signal_dir / f"continue_{cur}.json"
        with open(signal_path, "w") as f:
            json.dump({"status": "simulation_complete", "date": cur}, f)

    # ── cleanup ────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Signal simulation complete and terminate the harness process."""
        self.signal_complete()
        if self._harness_proc and self._harness_proc.poll() is None:
            logger.info(
                "Terminating %s process (pid=%d)",
                self.config.harness_backend, self._harness_proc.pid,
            )
            self._harness_proc.terminate()
            try:
                self._harness_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._harness_proc.kill()

        for f in (getattr(self, "_stdout_log", None), getattr(self, "_stderr_log", None)):
            if f and not f.closed:
                f.close()

    def __del__(self):
        self.cleanup()
