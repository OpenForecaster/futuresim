"""Shared MinimalHarness driver for external coding-agent CLIs.

The base class owns Futuresim state, workspace prep, MCP coordination,
sandbox/egress plumbing, warmup orchestration, and cleanup. Backend subclasses
own CLI-specific launch and resume behavior.
"""

import json
import logging
import subprocess
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from futuresim_agents.base import BaseAgent
from futuresim_agents.minimalHarnessAgent.config import MinimalHarnessConfig
from futuresim_agents.minimalHarnessAgent.mcp_helpers import McpHelpers
from futuresim_agents.minimalHarnessAgent.prompts.prompt import build_system_prompt
from futuresim_agents.minimalHarnessAgent.sandbox import SandboxHelpers
from futuresim_agents.minimalHarnessAgent.state import StateHelpers
from environment.interfaces import PredictionSubmission

logger = logging.getLogger(__name__)


class MinimalHarnessAgent(StateHelpers, McpHelpers, SandboxHelpers, BaseAgent):
    """Shared driver for external coding-agent forecast harnesses.

    Direct construction dispatches to the backend subclass named by
    ``config.harness_backend``. Subclasses own CLI-specific launch/resume
    behavior; this base class owns simulation state, workspace prep, MCP
    coordination, warmup orchestration, and cleanup.
    """

    def __new__(cls, agent_id: str, config: MinimalHarnessConfig, *args, **kwargs):
        if cls is MinimalHarnessAgent:
            backend = getattr(config, "harness_backend", "claude_code")
            if backend == "claude_code":
                from futuresim_agents.minimalHarnessAgent.claude_code_agent import ClaudeCodeAgent

                cls = ClaudeCodeAgent
            elif backend == "opencode":
                from futuresim_agents.minimalHarnessAgent.opencode_agent import OpenCodeAgent

                cls = OpenCodeAgent
            elif backend == "codex":
                from futuresim_agents.minimalHarnessAgent.codex_agent import CodexAgent

                cls = CodexAgent
            else:
                raise ValueError(
                    f"Unknown harness_backend={backend!r}; expected 'claude_code', 'opencode', or 'codex'."
                )
        return super().__new__(cls)

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
        # _internal_dir: driver/MCP coordination (signals, mcp_config, logs).
        # workspace: CLI working directory; only agent-visible files live here.
        self._internal_dir = Path(agent_dir)
        self.workspace = self._internal_dir / "workspace"
        self.articles_base = Path(articles_base or config.articles_base)
        self._harness_proc: Optional[subprocess.Popen] = None
        self._started = False
        self._current_date: Optional[date] = None
        self._resume_attempts: Dict[date, int] = {}
        self._bootstrap_applied: bool = False

        # Egress proxy state (network_isolation=True). Started lazily on first act().
        self._egress_proc: Optional[subprocess.Popen] = None
        self._egress_dir: Optional[Path] = None
        self._egress_proxy_sock: Optional[Path] = None
        self._egress_embed_sock: Optional[Path] = None
        self._egress_embed_port: Optional[int] = None
        # In-sandbox loopback port the bridge sidecar listens on for the proxy.
        self._egress_proxy_port = 9080

        # MCP relay state (sandbox=True). Started lazily on first act() — keeps
        # the MCP server (and thus search_db / lancedb / embedding_model) on
        # the host so the sandboxed harness can't bypass date-capping by
        # querying the index directly. The harness reaches MCP via a socat
        # stdio bridge to a per-agent unix socket.
        self._mcp_relay_proc: Optional[subprocess.Popen] = None
        self._mcp_relay_dir: Optional[Path] = None
        self._mcp_relay_sock: Optional[Path] = None
        self._mcp_bridge_wrapper: Optional[Path] = None

        if self.config.network_isolation and not self.config.sandbox:
            raise ValueError("network_isolation=True requires sandbox=True.")
        if self.config.sandbox_proc_mode not in {"new", "host_ro"}:
            raise ValueError(
                "sandbox_proc_mode must be 'new' or 'host_ro' "
                f"(got {self.config.sandbox_proc_mode!r})."
            )

        # Per-day initial prompt for fresh-context backend launches.
        self._initial_prompt_override: Optional[str] = None

        # active_memory / active_memory2 mode wiring.
        if self.config.prompt_mode in ("active_memory", "active_memory2"):
            if self.config.harness_backend not in ("codex", "claude_code", "opencode"):
                raise ValueError(
                    f"prompt_mode={self.config.prompt_mode!r} requires harness_backend in "
                    "{'codex', 'claude_code', 'opencode'} "
                    f"(got {self.config.harness_backend!r})."
                )
            if self.config.harness_backend == "codex" and self.config.codex_resume:
                raise ValueError(
                    f"prompt_mode={self.config.prompt_mode!r} requires codex_resume=False; "
                    "this mode delivers AllQ-style fresh-context-per-day with "
                    "memory files, not a persistent codex thread."
                )
            if self.config.harness_backend == "claude_code" and self.config.claude_code_resume:
                raise ValueError(
                    f"prompt_mode={self.config.prompt_mode!r} requires claude_code_resume=False; "
                    "this mode delivers fresh-context-per-day with memory files, "
                    "not a persistent claude session."
                )
            # Stateful feedback aggregator (shared across days, dedups on qid).
            from futuresim_agents.basicAgent.feedback import FeedbackHandler
            self._feedback_handler: Optional[FeedbackHandler] = FeedbackHandler(self.agent_id)
        elif self.config.prompt_mode in {"warmup", "static_search"}:
            if self.config.harness_backend != "codex":
                raise ValueError(
                    f"prompt_mode={self.config.prompt_mode!r} currently requires harness_backend='codex' "
                    f"(got {self.config.harness_backend!r})."
                )
            if self.config.codex_resume:
                raise ValueError(
                    f"prompt_mode={self.config.prompt_mode!r} requires codex_resume=False; each question "
                    "gets a fresh Codex session."
                )
            self._feedback_handler = None
        elif self.config.prompt_mode not in ("default", "no_memory"):
            raise ValueError(
                f"Unknown prompt_mode={self.config.prompt_mode!r}; "
                "expected 'default', 'no_memory', 'active_memory', 'warmup', 'static_search', or 'active_memory2'."
            )
        else:
            self._feedback_handler = None
        self.warmed_up = False

        if self.config.handholding_version is None:
            # Per-model default. deepseek/qwen/glm need the v3 nudges (TW-update
            # reminder + imminent-resolution callout) to keep submitting; without
            # them they drift into "no new evidence, skip" silence (e.g. glm-5.1
            # only submitted on 33/95 days under v1). Other models keep v1.
            model_lc = (self.config.model or "").lower()
            if any(fam in model_lc for fam in ("deepseek", "qwen", "glm")):
                self.config.handholding_version = "v3"
            else:
                self.config.handholding_version = "v1"
        if self.config.handholding_version not in ("v1", "v2", "v3"):
            raise ValueError(
                f"Unknown handholding_version={self.config.handholding_version!r}; "
                "expected one of 'v1', 'v2', 'v3'."
            )
        # Active memory's prompt builder hardcodes the maximal-handholding
        # shared sections (TW nudge + imminent reminder). Force the field to
        # v3 so the runtime stays consistent everywhere it's read (mcp server,
        # logs, build_system_prompt fallbacks, future audits).
        if (
            self.config.prompt_mode in ("active_memory", "active_memory2")
            and self.config.handholding_version != "v3"
        ):
            print(
                f"[minimalHarness] prompt_mode={self.config.prompt_mode!r} coerces "
                f"handholding_version {self.config.handholding_version!r} -> 'v3'."
            )
            self.config.handholding_version = "v3"

    def article_workspace_config(self) -> Dict[str, Any]:
        """Describe this agent's article workspace for environment-owned staging."""
        return {
            "articles_base": str(self.articles_base) if self.articles_base else "",
            "articles_dir": str(self.workspace / "articles"),
            "mode": "hardlink" if self.config.sandbox else "symlink",
            "search_cutoff_days": self.config.search_cutoff_days,
            "freeze_after_start": self.config.freeze_search_after_start,
            "start_date": self.config.start_date,
        }

    # ── Backend hooks ──────────────────────────────────────────────────
    # Optional hooks default to no-op/False so backends only override real differences.

    def _respawns_each_day(self) -> bool:
        return False

    def _prepare_harness_launch(self) -> None:
        pass

    def _start_harness(self) -> None:
        raise NotImplementedError

    def _resume_harness(self, current_date: date) -> bool:
        raise NotImplementedError

    def _after_next_day(self, current_date: date) -> None:
        pass

    def _next_day_returns_immediately(self) -> bool:
        return False

    def _after_next_day_signal(self, current_date: date, timeout: float, start: float) -> None:
        pass

    def _add_sandbox_harness_install_binds(self, args: List[str]) -> None:
        pass

    def _sandbox_harness_home_subdirs(self) -> Tuple[str, ...]:
        return ()

    @staticmethod
    def _scan_jsonl_for_field(
        path: Path,
        predicate: Callable[[Dict[str, Any]], bool],
        field: str,
        *,
        last_match: bool = False,
    ) -> Optional[str]:
        if not path.exists():
            return None
        matched = None
        with open(path) as f:
            for line in f:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict) or not predicate(event):
                    continue
                value = event.get(field)
                if not value:
                    continue
                if not last_match:
                    return value
                matched = value
        return matched

    def _close_harness_logs(self) -> None:
        for f in (getattr(self, "_stdout_log", None), getattr(self, "_stderr_log", None)):
            if f and not f.closed:
                f.close()

    def _launch_harness_process(
        self,
        cmd: List[str],
        *,
        stdout_log_name: str,
        stderr_log_name: str,
        env: Dict[str, str],
        label: str,
    ) -> None:
        self._close_harness_logs()
        log_stdout = open(self._internal_dir / stdout_log_name, "a")
        log_stderr = open(self._internal_dir / stderr_log_name, "a")

        launch_cmd = self._maybe_sandbox(cmd)
        popen_cwd = None if self.config.sandbox else str(self.workspace)
        debug_path = self._internal_dir / f"{label}_launch.json"
        try:
            with open(debug_path, "w") as f:
                json.dump({
                    "cmd": launch_cmd,
                    "cwd": popen_cwd,
                    "codex_home": env.get("CODEX_HOME", ""),
                    "sandbox": bool(self.config.sandbox),
                    "network_isolation": bool(self.config.network_isolation),
                    "sandbox_proc_mode": self.config.sandbox_proc_mode,
                }, f, indent=2)
        except OSError:
            logger.warning("failed to write harness launch debug file", exc_info=True)

        logger.info(
            "Starting %s%s%s: %s",
            label,
            " [sandboxed]" if self.config.sandbox else "",
            " [net-isolated]" if self.config.network_isolation else "",
            " ".join(launch_cmd),
        )
        try:
            self._harness_proc = subprocess.Popen(
                launch_cmd,
                stdin=subprocess.DEVNULL,
                stdout=log_stdout,
                stderr=log_stderr,
                cwd=popen_cwd,
                env=env,
            )
        except Exception:
            log_stdout.close()
            log_stderr.close()
            raise
        self._stdout_log = log_stdout
        self._stderr_log = log_stderr

    # ── BaseAgent contract ─────────────────────────────────────────────

    def act(
        self,
        doc_interface: Any,
        forecast_interface: Any,
        current_date: date,
    ) -> List[Dict[str, Any]]:
        self._current_date = current_date

        if (
            self.config.prompt_mode in {"warmup", "static_search"}
            and self.warmed_up
            and self.config.start_date == current_date
        ):
            print(f"[{self.agent_id}] Skipping standard act() on Day 0 (Warmup already completed).")
            forecast_interface.next_day()
            return []

        # Internal dirs (driver/MCP coordination — harness doesn't see these).
        for d in ("signals",):
            (self._internal_dir / d).mkdir(parents=True, exist_ok=True)

        # Workspace dirs (harness' working directory — only what the harness needs).
        for d in ("memory", "articles"):
            (self.workspace / d).mkdir(parents=True, exist_ok=True)

        # 0. Bootstrap from fixedWarmup only on the true first sim day.
        # Env-level resumes may start mid-run with copied memory already present.
        if (
            self.config.bootstrap_dir
            and not self._bootstrap_applied
            and (self.config.start_date is None or current_date == self.config.start_date)
        ):
            self._apply_bootstrap(forecast_interface, current_date)
            self._bootstrap_applied = True

        # 1. Write state.json for MCP server to read.
        search_current_date = (
            self.config.start_date
            if self.config.freeze_search_after_start
            else None
        )
        self._write_state(
            forecast_interface,
            current_date,
            search_current_date=search_current_date,
        )

        # 2. chmod market.csv read-only so the harness can't corrupt it.
        self._protect_market_csv(forecast_interface)

        # 3. If network_isolation is on, ensure the host-side egress proxy is
        # running before we spawn (or re-spawn) the harness. Idempotent.
        if self.config.network_isolation:
            self._start_egress_proxy()

        # 3b. If sandboxed, start the host-side MCP relay so search_db and
        # the embedding model stay out of the sandbox. The harness reaches
        # MCP only via the bind-mounted unix socket. Idempotent.
        if self.config.sandbox:
            self._start_mcp_relay()

        # Backends decide whether their CLI process is persistent or
        # per-day. Codex always respawns; Claude Code/OpenCode respawn only
        # for fresh-context prompt modes.
        respawn_per_day = self._respawns_each_day()

        if not self._started or respawn_per_day:
            if respawn_per_day and self._started:
                # Reap the previous day's harness if still alive (should have
                # exited after writing next_day, but be defensive).
                if self._harness_proc and self._harness_proc.poll() is None:
                    self._harness_proc.terminate()
                    try:
                        self._harness_proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self._harness_proc.kill()
                # Clear stale next_day signals so _wait_for_next_day blocks
                # until the freshly-spawned harness writes a new one.
                signal_dir = self._internal_dir / "signals"
                signal_dir.mkdir(parents=True, exist_ok=True)
                for old in signal_dir.glob("next_day_*.json"):
                    old.unlink(missing_ok=True)

            if self.config.prompt_mode in ("active_memory", "active_memory2"):
                # active_memory{,2} mode replaces AGENTS.md + 2-line per-day prompt
                # with a full daily user prompt (mirroring AllQAgent). Skip the
                # system_prompt.md write entirely.
                self._build_active_memory_prompt(forecast_interface, current_date)
            else:
                self._write_system_prompt(forecast_interface)
            self._prepare_harness_launch()
            self._start_harness()
            self._started = True
        else:
            # Subsequent days for persistent backends — unblock the MCP
            # server's next_day poll.
            self._write_continue_signal(current_date)

        # 4. Wait for the harness to call next_day (MCP server writes signal).
        predictions = self._wait_for_next_day(current_date)

        self._after_next_day(current_date)

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

    def _terminate_harness_process(self) -> None:
        if self._harness_proc and self._harness_proc.poll() is None:
            self._harness_proc.terminate()
            try:
                self._harness_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._harness_proc.kill()
        self._harness_proc = None
        self._close_harness_logs()

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
            prompt_date = self._current_date or self.config.start_date
            new_articles_count = self._count_new_articles(last_active_date, prompt_date)
        if self.config.prompt_mode == "no_memory":
            from futuresim_agents.minimalHarnessAgent.prompts.prompt_no_memory import (
                build_system_prompt as _build_system_prompt,
            )
        else:
            _build_system_prompt = build_system_prompt
        prompt = _build_system_prompt(
            workspace=str(self.workspace),
            current_date=self._current_date or self.config.start_date,
            start_date=self.config.start_date or self._current_date,
            end_date=self.config.end_date or self._current_date,
            source_context=source_context,
            source_name=source_name,
            num_questions=num_questions,
            num_active=num_active,
            num_resolved=num_resolved,
            max_outcomes_per_question=self.config.max_outcomes_per_question,
            search_cutoff_days=self.config.search_cutoff_days,
            timegap_days=self.config.timegap_days,
            new_articles_count=new_articles_count,
            last_active_date=last_active_date,
            next_active_date=next_active_date,
            handholding_version=self.config.handholding_version,
        )
        prompt_path = self._internal_dir / "system_prompt.md"
        with open(prompt_path, "w") as f:
            f.write(prompt)

    def _build_active_memory_prompt(
        self, forecast_interface: Any, current_date: date
    ) -> None:
        """Compose the per-day active_memory{,2} user prompt for backend launch.

        Mirrors BasicAgent / AllQAgent daily prompt construction:
          - feedback aggregated via FeedbackHandler (cumulative across days),
          - prior day's meta-insight index loaded from disk via ActiveMemory,
          - AllQ "already-predicted" reminder with current predicted/active counts.

        prompt_mode="active_memory2" routes to prompts.prompt_active_memory2.build_daily_prompt,
        which advertises the MCP memory tools instead of native filesystem reads/writes.
        """
        if self.config.prompt_mode == "active_memory2":
            from futuresim_agents.minimalHarnessAgent.prompts.prompt_active_memory2 import build_daily_prompt
        else:
            from futuresim_agents.minimalHarnessAgent.prompts.prompt_active_memory import build_daily_prompt
        from futuresim_agents.utils.memory import ActiveMemory

        source_context = getattr(forecast_interface, "source_context", "") or ""
        source_name = getattr(forecast_interface, "source_name", "openforesight")
        questions = forecast_interface.list_questions()
        num_active = len(questions)
        num_resolved = len(getattr(forecast_interface, "resolved_questions", []))
        num_questions = num_active + num_resolved
        last_active_date = getattr(forecast_interface, "last_active_date", None)
        if last_active_date is None:
            last_active_date = self._find_latest_memory_date_before(current_date)
        next_active_date = getattr(forecast_interface, "next_active_date", None)
        new_articles_count = self._count_new_articles(last_active_date, current_date)

        # Feedback (consumes resolution_events from the env; dedups on qid).
        assert self._feedback_handler is not None
        feedback_data = self._feedback_handler.generate_feedback(
            forecast_interface, current_date, inference_provider=None
        )
        feedback_text = self._feedback_handler.format_feedback(
            feedback_data, show_tw_peer=False
        )

        # Prior-day meta-insight index from memory/{prev}/meta.yaml on disk.
        meta_index = ""
        am = ActiveMemory(self.agent_id, memory_dir=str(self.workspace))
        am.set_date(current_date)
        meta_index = am.get_index() or ""

        # Predicted / active counts for the AllQ reminder.
        active_qids = {q.qid for q in questions}
        predicted_count = 0
        get_preds = getattr(forecast_interface, "get_agent_predictions", None)
        if callable(get_preds):
            preds = get_preds(self.agent_id) or {}
            predicted_count = sum(1 for qid in preds.keys() if qid in active_qids)

        # Questions resolving tomorrow — surfaced inline in the cadence section.
        tomorrow = current_date + timedelta(days=1)
        imminent_qids = [
            q.qid for q in questions
            if getattr(q, "resolution_date", None) == tomorrow
        ]

        prompt = build_daily_prompt(
            current_date=current_date,
            start_date=self.config.start_date or current_date,
            end_date=self.config.end_date or current_date,
            last_active_date=last_active_date,
            next_active_date=next_active_date,
            source_context=source_context,
            source_name=source_name,
            feedback_text=feedback_text,
            meta_index=meta_index,
            num_questions=num_questions,
            num_active=num_active,
            num_resolved=num_resolved,
            predicted_count=predicted_count,
            active_count=num_active,
            max_outcomes_per_question=self.config.max_outcomes_per_question,
            search_cutoff_days=self.config.search_cutoff_days,
            timegap_days=self.config.timegap_days,
            new_articles_count=new_articles_count,
            imminent_qids=imminent_qids,
        )
        # Persist for debugging / offline review.
        prompt_path = (
            self._internal_dir
            / f"{self.config.prompt_mode}_prompt_{current_date.isoformat()}.md"
        )
        with open(prompt_path, "w") as f:
            f.write(prompt)
        self._initial_prompt_override = prompt

    def _wait_for_next_day(self, current_date: date) -> List[Dict[str, Any]]:
        """Poll for the MCP server's next_day signal, then return predictions.

        The signal file is the authoritative snapshot from _today_predictions
        at the moment next_day was called.
        """
        signal_path = self._internal_dir / "signals" / f"next_day_{current_date.isoformat()}.json"
        poll_interval = 1.0
        timeout = self.config.timeout_seconds
        start = time.time()

        MAX_RESUMES_PER_DAY = 3

        while not signal_path.exists():
            # Check if harness process died.
            if self._harness_proc and self._harness_proc.poll() is not None:
                backend = self.config.harness_backend
                attempts = self._resume_attempts.get(current_date, 0)
                resumed = False
                if attempts < MAX_RESUMES_PER_DAY:
                    resumed = self._resume_harness(current_date)
                if resumed:
                    self._resume_attempts[current_date] = attempts + 1
                    continue
                logger.error(
                    "%s exited with code %d before signaling next_day "
                    "(resume attempts=%d)",
                    backend, self._harness_proc.returncode, attempts,
                )
                return []
            if time.time() - start > timeout:
                logger.error("Timeout waiting for next_day signal on %s", current_date)
                return []
            time.sleep(poll_interval)

        self._after_next_day_signal(current_date, timeout, start)
        try:
            data = json.loads(signal_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read next_day signal for %s: %s", current_date, exc)
            return []
        preds = data.get("predictions", [])
        if not isinstance(preds, list):
            return []
        for p in preds:
            if isinstance(p, dict) and "qid" in p and "question_id" not in p:
                p["question_id"] = p["qid"]
        return [p for p in preds if isinstance(p, dict)]

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

        self._stop_egress_proxy()
        self._stop_mcp_relay()

        self._close_harness_logs()

    def __del__(self):
        self.cleanup()
