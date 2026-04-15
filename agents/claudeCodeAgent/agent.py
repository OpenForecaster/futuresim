"""
ClaudeCodeAgent — forecasting agent that delegates to a single persistent
Claude Code CLI session.

The first call to act() spawns Claude Code + its MCP server.  Subsequent calls
signal the MCP server (via a file-based continue signal) so the already-running
Claude Code session can resume with the next simulation day.
"""

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.base import BaseAgent
from agents.claudeCodeAgent.prompt import build_system_prompt
from environment.interfaces import PredictionSubmission

logger = logging.getLogger(__name__)


@dataclass
class ClaudeCodeConfig:
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


class ClaudeCodeAgent(BaseAgent):
    """Forecasting agent backed by a single persistent Claude Code session."""

    def __init__(
        self,
        agent_id: str,
        config: ClaudeCodeConfig,
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
        self._cc_process: Optional[subprocess.Popen] = None
        self._started = False
        self._last_symlink_date: Optional[date] = None
        self._current_date: Optional[date] = None

    # ── BaseAgent contract ─────────────────────────────────────────────

    def act(
        self,
        doc_interface: Any,
        forecast_interface: Any,
        current_date: date,
    ) -> List[Dict[str, Any]]:
        self._current_date = current_date

        # Internal dirs (driver/MCP coordination — CC doesn't see these).
        for d in ("signals",):
            (self._internal_dir / d).mkdir(parents=True, exist_ok=True)

        # Internal dirs (driver/MCP coordination — CC doesn't see these).
        (self._internal_dir / "predictions").mkdir(parents=True, exist_ok=True)

        # Workspace dirs (CC's working directory — only what CC needs).
        for d in ("memory", "articles"):
            (self.workspace / d).mkdir(parents=True, exist_ok=True)

        # 1. Write state.json for MCP server to read.
        self._write_state(forecast_interface, current_date)

        # 2. chmod market.csv read-only so Claude Code can't corrupt it.
        self._protect_market_csv(forecast_interface)

        # 3. Update article symlinks up to current date.
        self._update_article_symlinks(current_date)

        if not self._started:
            # First day — start Claude Code and MCP server.
            self._write_mcp_config()
            self._write_system_prompt(forecast_interface)
            self._start_claude_code()
            self._started = True
        else:
            # Subsequent days — unblock the MCP server's next_day poll.
            self._write_continue_signal(current_date)

        # 4. Wait for Claude Code to call next_day (MCP server writes signal).
        predictions = self._wait_for_next_day(current_date)

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
        if forecast_interface is not None:
            source_context = getattr(forecast_interface, "source_context", "")
            source_name = getattr(forecast_interface, "source_name", "openforesight")
            questions = forecast_interface.list_questions()
            num_active = len(questions)
            num_resolved = len(getattr(forecast_interface, "resolved_questions", []))
            num_questions = num_active + num_resolved
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
        )
        prompt_path = self._internal_dir / "system_prompt.md"
        with open(prompt_path, "w") as f:
            f.write(prompt)

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
        """Incrementally add symlinks for article directories up to current_date."""
        if not self.articles_base or not self.articles_base.exists():
            return

        articles_dir = self.workspace / "articles"
        start = (self._last_symlink_date + timedelta(days=1)) if self._last_symlink_date else (
            self.config.start_date or current_date - timedelta(days=30)
        )

        d = start
        while d <= current_date:
            src = self.articles_base / f"{d.year}" / f"{d.month:02d}" / f"{d.day:02d}"
            if src.exists():
                dst = articles_dir / f"{d.year}" / f"{d.month:02d}" / f"{d.day:02d}"
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    dst.symlink_to(src)
            d += timedelta(days=1)

        self._last_symlink_date = current_date

    # ── Claude Code subprocess ─────────────────────────────────────────

    def _start_claude_code(self) -> None:
        initial_prompt = (
            "Begin forecasting. Read market.csv to see your questions, "
            "then research and submit predictions."
        )
        cmd = [
            self.config.claude_code_path,
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
        ]

        if self.config.max_budget_usd is not None:
            cmd.extend(["--max-budget-usd", str(self.config.max_budget_usd)])

        cmd.extend(self.config.extra_flags)

        log_stdout = open(self._internal_dir / "claude_code_stdout.jsonl", "w")
        log_stderr = open(self._internal_dir / "claude_code_stderr.log", "w")

        logger.info("Starting Claude Code: %s", " ".join(cmd))
        self._cc_process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_stdout,
            stderr=log_stderr,
            cwd=str(self.workspace),
        )
        self._stdout_log = log_stdout
        self._stderr_log = log_stderr

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

        while not signal_path.exists():
            # Check if Claude Code process died.
            if self._cc_process and self._cc_process.poll() is not None:
                logger.warning(
                    "Claude Code exited with code %d before signaling next_day",
                    self._cc_process.returncode,
                )
                return self._read_predictions(current_date)
            if time.time() - start > timeout:
                logger.error("Timeout waiting for next_day signal on %s", current_date)
                return self._read_predictions(current_date)
            time.sleep(poll_interval)

        # Read predictions from the predictions file — this is the durable record
        # of all submit_forecast MCP calls. Each call writes to this file immediately,
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
        next_day poll unblocks and tells Claude Code the sim is over."""
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
        """Signal simulation complete and terminate Claude Code process."""
        self.signal_complete()
        if self._cc_process and self._cc_process.poll() is None:
            logger.info("Terminating Claude Code process (pid=%d)", self._cc_process.pid)
            self._cc_process.terminate()
            try:
                self._cc_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._cc_process.kill()

        for f in (getattr(self, "_stdout_log", None), getattr(self, "_stderr_log", None)):
            if f and not f.closed:
                f.close()

    def __del__(self):
        self.cleanup()
