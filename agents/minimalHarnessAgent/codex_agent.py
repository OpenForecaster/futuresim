"""Codex CLI backend for MinimalHarnessAgent."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from futuresim_agents.minimalHarnessAgent.agent import MinimalHarnessAgent
from futuresim_agents.minimalHarnessAgent.sandbox_common import real_path
from futuresim_agents.minimalHarnessAgent.static_search import safe_qid
from futuresim_agents.utils.output_logger import AgentOutputLogger
from environment.interfaces import PredictionSubmission

logger = logging.getLogger(__name__)


_CODEX_MCP_NAME_REPLACEMENTS = (
    (
        "`mcp__forecast__search_news(query, from_date?, to_date?)`",
        "`search_news(query, from_date?, to_date?)` on the Forecast MCP server",
    ),
    (
        "`mcp__forecast__submit_forecasts(question_id, outcomes)`",
        "`submit_forecasts(question_id, outcomes)` on the Forecast MCP server",
    ),
    (
        "`mcp__forecast__next_day()`",
        "`next_day()` on the Forecast MCP server",
    ),
    ("`mcp__forecast__search_news`", "`search_news` on the Forecast MCP server"),
    ("`mcp__forecast__submit_forecasts`", "`submit_forecasts` on the Forecast MCP server"),
    ("`mcp__forecast__next_day`", "`next_day` on the Forecast MCP server"),
)


def _use_codex_mcp_tool_names(prompt: str) -> str:
    for old, new in _CODEX_MCP_NAME_REPLACEMENTS:
        prompt = prompt.replace(old, new)
    return prompt


def _read_json_file(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def _read_text_file(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text()
    except OSError:
        return ""


def _read_jsonl_file(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    if not path.exists():
        return events
    try:
        with open(path) as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    events.append({
                        "type": "parse_error",
                        "line_no": line_no,
                        "text": line,
                    })
    except OSError:
        return events
    return events


def _render_codex_response(events: List[Dict[str, Any]], predictions: List[Dict[str, Any]]) -> str:
    messages: List[str] = []
    for ev in events:
        item = ev.get("item") if isinstance(ev, dict) else None
        if not isinstance(item, dict):
            continue
        if ev.get("type") == "item.completed" and item.get("type") == "agent_message":
            text = item.get("text")
            if text:
                messages.append(str(text))

    if predictions:
        messages.append("Submitted predictions: " + json.dumps(predictions, sort_keys=True))
    return "\n\n".join(messages)


def _rewrite_warmup_index_for_aggregated_logs(agent_dir: Path) -> None:
    index_path = agent_dir / "warmup_logs" / "index.jsonl"
    if not index_path.exists():
        return
    records = _read_jsonl_file(index_path)
    with open(index_path, "w") as f:
        for record in records:
            record.pop("log_dir", None)
            record["processed_output_log"] = str(agent_dir / "model_outputs.jsonl")
            record["raw_output_log"] = str(agent_dir / "model_raw_warmup.jsonl")
            f.write(json.dumps(record) + "\n")


def aggregate_codex_warmup_logs(
    agent_id: str,
    agent_dir: Any,
    *,
    remove_per_question_logs: bool = True,
    append: bool = False,
) -> int:
    """Aggregate per-question Codex warmup transcripts into standard agent logs."""
    agent_dir = Path(agent_dir)
    warmup_logs_dir = agent_dir / "warmup_logs"
    if not warmup_logs_dir.exists():
        return 0

    log_dirs = sorted(p for p in warmup_logs_dir.iterdir() if p.is_dir())
    if not log_dirs:
        _rewrite_warmup_index_for_aggregated_logs(agent_dir)
        return 0

    output_logger = AgentOutputLogger(agent_id, str(agent_dir), append=append)
    written = 0
    try:
        for log_dir in log_dirs:
            state = _read_json_file(log_dir / "state.json", {}) or {}
            questions = state.get("questions") or []
            qid = str((questions[0] or {}).get("qid") if questions else log_dir.name)
            sim_date = state.get("current_date", "")
            prompt = _read_text_file(log_dir / "prompt.md")
            predictions = _read_json_file(log_dir / "predictions.json", []) or []
            events = _read_jsonl_file(log_dir / "codex_stdout.jsonl")
            stderr_text = _read_text_file(log_dir / "codex_stderr.log")
            prompt_mode = state.get("prompt_mode", "warmup")
            raw_stream = "warmup"
            thread_id = next(
                (
                    ev.get("thread_id")
                    for ev in events
                    if ev.get("type") == "thread.started" and ev.get("thread_id")
                ),
                None,
            )
            tool_calls = [
                ev.get("item", {})
                for ev in events
                if isinstance(ev.get("item"), dict)
                and ev.get("item", {}).get("type") == "mcp_tool_call"
                and ev.get("type") == "item.completed"
            ]
            response = _render_codex_response(events, predictions)
            metadata = {
                "backend": "codex",
                "harness_backend": "codex",
                "prompt_mode": prompt_mode,
                "raw_stream": raw_stream,
                "qid": qid,
                "current_date": sim_date,
                "search_current_date": state.get("search_current_date", sim_date),
                "thread_id": thread_id,
                "predictions": predictions,
                "codex_event_count": len(events),
                "codex_tool_call_count": len(tool_calls),
                "codex_submit_call_count": sum(
                    1 for item in tool_calls if item.get("tool") == "submit_forecasts"
                ),
                "_logger_raw_input_delta": prompt,
                "_logger_raw_response": events,
                "_logger_raw_metadata": {
                    "backend": "codex",
                    "harness_backend": "codex",
                    "prompt_mode": prompt_mode,
                    "raw_stream": raw_stream,
                    "qid": qid,
                    "current_date": sim_date,
                    "search_current_date": state.get("search_current_date", sim_date),
                    "thread_id": thread_id,
                    "predictions": predictions,
                    "codex_stderr": stderr_text,
                    "mcp_relay_log": _read_text_file(log_dir / "mcp_relay.log"),
                    "state": state,
                },
            }
            output_logger.log_model_output(sim_date, prompt, response, metadata)
            written += 1
            if remove_per_question_logs:
                shutil.rmtree(log_dir, ignore_errors=True)
        output_logger.flush_warmup_raw()
    finally:
        output_logger.close()

    _rewrite_warmup_index_for_aggregated_logs(agent_dir)
    return written


class CodexAgent(MinimalHarnessAgent):
    """MinimalHarness backend that launches Codex CLI sessions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._codex_thread_id: Optional[str] = None

    def _respawns_each_day(self) -> bool:
        return True

    def _start_harness(self) -> None:
        self._start_codex()

    def _resume_harness(self, current_date: date) -> bool:
        return self._resume_codex(current_date)

    def _after_next_day(self, current_date: date) -> None:
        if self.config.codex_resume and self._codex_thread_id is None:
            self._find_codex_thread_id()

    def _next_day_returns_immediately(self) -> bool:
        return True

    def _after_next_day_signal(self, current_date: date, timeout: float, start: float) -> None:
        if not (self._harness_proc and self._harness_proc.poll() is None):
            return
        drain_timeout = min(30.0, max(0.0, timeout - (time.time() - start)))
        try:
            self._harness_proc.wait(timeout=drain_timeout)
        except subprocess.TimeoutExpired:
            logger.warning(
                "codex did not exit within %.1fs after next_day signal on %s; "
                "reading durable predictions anyway",
                drain_timeout,
                current_date,
            )

    def _add_sandbox_harness_install_binds(self, args: List[str]) -> None:
        bin_real = os.path.realpath(self.config.codex_path)
        if os.path.exists(bin_real):
            install = os.path.dirname(os.path.dirname(bin_real))
            args.extend(["--ro-bind", install, install])
        codex_home = os.environ.get("CODEX_HOME")
        if not codex_home:
            return
        home_real = os.path.realpath(os.path.expanduser("~"))
        codex_home_real = real_path(os.path.expanduser(codex_home))
        try:
            rel = os.path.relpath(codex_home_real, home_real)
        except ValueError:
            rel = os.pardir
        if rel and rel != os.curdir and not rel.startswith(os.pardir + os.sep) and rel != os.pardir:
            return
        if os.path.exists(codex_home_real):
            args.extend(["--bind", codex_home_real, codex_home_real])

    def _sandbox_harness_home_subdirs(self) -> tuple[str, ...]:
        codex_home = os.environ.get("CODEX_HOME")
        if codex_home:
            home_real = os.path.realpath(os.path.expanduser("~"))
            codex_home_real = os.path.realpath(os.path.expanduser(codex_home))
            try:
                rel = os.path.relpath(codex_home_real, home_real)
            except ValueError:
                rel = ""
            if rel and rel != os.curdir and not rel.startswith(os.pardir + os.sep):
                return (rel,)
        return (".codex",)

    def _get_warmup_current_date(self, current_date: date, q: Any) -> date:
        resolution_guard = self.config.resolution_guard
        resolution_date = getattr(q, "resolution_date", None)
        if resolution_guard is None or resolution_date is None:
            return current_date
        return resolution_date - timedelta(days=resolution_guard)

    def _build_warmup_prompt(
        self,
        forecast_interface: Any,
        q: Any,
        effective_current_date: date,
    ) -> str:
        from futuresim_agents.minimalHarnessAgent.prompts.prompt import build_warmup_prompt

        source_context = getattr(forecast_interface, "source_context", "") or ""
        source_name = getattr(forecast_interface, "source_name", "openforesight")
        static_search_text = None
        if self.config.prompt_mode == "static_search":
            static_search_text = self._get_static_search_text(q, effective_current_date)
        prompt = build_warmup_prompt(
            current_date=effective_current_date,
            q=q,
            source_context=source_context,
            source_name=source_name,
            max_outcomes_per_question=self.config.max_outcomes_per_question,
            search_cutoff_days=self.config.search_cutoff_days,
            static_search_text=static_search_text,
        )
        qid_slug = safe_qid(q.qid)
        prompt_path = self._internal_dir / "warmup_prompts" / f"{qid_slug}.md"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(prompt_path, "w") as f:
            f.write(prompt)
        return prompt

    def _warmup_runtime_dir(self, safe_qid: str) -> Path:
        return self._internal_dir / "warmup_runtime" / safe_qid

    def _warmup_log_dir(self, safe_qid: str) -> Path:
        return self._internal_dir / "warmup_logs" / safe_qid

    def _static_search_root(self) -> Path:
        if self.config.static_search_dir:
            return Path(os.path.expanduser(os.path.expandvars(self.config.static_search_dir)))
        return self._internal_dir / "static_search"

    def _ensure_static_search_file(self, q: Any, effective_current_date: date) -> Tuple[Path, str]:
        from futuresim_agents.minimalHarnessAgent.static_search import ensure_static_search_file

        return ensure_static_search_file(
            output_dir=self._static_search_root(),
            search_tool=self.search_tool,
            q=q,
            search_date=effective_current_date,
            search_type=self.config.search_type,
            search_cutoff_days=self.config.search_cutoff_days,
            max_results=int(self.config.static_search_max_results or 5),
        )

    def _get_static_search_text(self, q: Any, effective_current_date: date) -> str:
        _, text = self._ensure_static_search_file(q, effective_current_date)
        return text

    def _prepare_static_search_files(self, questions: List[Any], current_date: date) -> None:
        root = self._static_search_root()
        root.mkdir(parents=True, exist_ok=True)
        index_path = root / "index.jsonl"
        index_path.unlink(missing_ok=True)

        total = len(questions)
        print(f"[{self.agent_id}] Preparing static title-search cache in {root}...")
        with open(index_path, "a") as index_f:
            for i, q in enumerate(questions, start=1):
                effective_current_date = self._get_warmup_current_date(current_date, q)
                path, _ = self._ensure_static_search_file(q, effective_current_date)
                index_f.write(json.dumps({
                    "qid": str(q.qid),
                    "title": str(q.title),
                    "resolution_date": self._to_iso(getattr(q, "resolution_date", "")),
                    "search_date": effective_current_date.isoformat(),
                    "query": str(q.title),
                    "path": str(path),
                }) + "\n")
                if i % 10 == 0 or i == total:
                    print(f"[{self.agent_id}] Static Search Progress: {i}/{total}", flush=True)

    def _copy_warmup_runtime_logs(
        self,
        runtime_dir: Path,
        log_dir: Path,
        qid: str,
        current_date: date,
        predictions: List[Dict[str, Any]],
    ) -> None:
        """Preserve per-question raw logs, then allow the runtime dir to be deleted."""
        del current_date
        log_dir.mkdir(parents=True, exist_ok=True)
        copy_map = {
            runtime_dir / "codex_stdout.jsonl": log_dir / "codex_stdout.jsonl",
            runtime_dir / "codex_stderr.log": log_dir / "codex_stderr.log",
            runtime_dir / "mcp_relay.log": log_dir / "mcp_relay.log",
            runtime_dir / "state.json": log_dir / "state.json",
            runtime_dir / "warmup_prompts" / f"{safe_qid(qid)}.md": log_dir / "prompt.md",
        }
        for src, dst in copy_map.items():
            if src.exists():
                shutil.copy2(src, dst)
        (log_dir / "predictions.json").write_text(json.dumps(predictions, default=str))

    def _stop_warmup_worker(self, worker: "MinimalHarnessAgent") -> None:
        worker._terminate_harness_process()
        worker._stop_egress_proxy()
        worker._stop_mcp_relay()
        worker._started = False

    def _run_single_warmup_question(
        self,
        q: Any,
        current_date: date,
        forecast_interface: Any,
    ) -> Tuple[str, date, List[Dict[str, Any]]]:
        qid = str(q.qid)
        qid_slug = safe_qid(qid)
        runtime_dir = self._warmup_runtime_dir(qid_slug)
        log_dir = self._warmup_log_dir(qid_slug)

        if runtime_dir.exists():
            shutil.rmtree(runtime_dir)
        if log_dir.exists():
            shutil.rmtree(log_dir)
        runtime_dir.mkdir(parents=True, exist_ok=True)

        worker_config = replace(self.config, warmup_parallelism=1)
        worker = MinimalHarnessAgent(
            agent_id=self.agent_id,
            config=worker_config,
            search_tool=self.search_tool,
            agent_dir=str(runtime_dir),
            articles_base=str(self.articles_base),
        )

        effective_current_date = self._get_warmup_current_date(current_date, q)
        predictions: List[Dict[str, Any]] = []
        try:
            for d in ("signals",):
                (worker._internal_dir / d).mkdir(parents=True, exist_ok=True)
            for d in ("memory", "articles"):
                (worker.workspace / d).mkdir(parents=True, exist_ok=True)

            if worker.config.network_isolation:
                worker._start_egress_proxy()
            if worker.config.sandbox:
                worker._start_mcp_relay()

            worker._current_date = current_date
            worker._write_state(
                forecast_interface,
                current_date,
                questions_override=[q],
                search_current_date=effective_current_date,
            )
            worker._initial_prompt_override = worker._build_warmup_prompt(
                forecast_interface,
                q,
                effective_current_date,
            )

            worker._start_harness()
            worker._started = True
            predictions = worker._wait_for_next_day(current_date)
            predictions = [
                pred for pred in predictions
                if str(pred.get("question_id", pred.get("qid", ""))) == qid
            ]
            return qid, effective_current_date, predictions
        finally:
            self._stop_warmup_worker(worker)
            self._copy_warmup_runtime_logs(runtime_dir, log_dir, qid, current_date, predictions)
            shutil.rmtree(runtime_dir, ignore_errors=True)

    def _append_warmup_log_index(
        self,
        qid: str,
        effective_current_date: date,
        predictions: List[Dict[str, Any]],
        error: Optional[str] = None,
    ) -> None:
        index_path = self._internal_dir / "warmup_logs" / "index.jsonl"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "qid": qid,
            "search_current_date": effective_current_date.isoformat(),
            "num_predictions": len(predictions),
            "error": error,
            "processed_output_log": str(self._internal_dir / "model_outputs.jsonl"),
            "raw_output_log": str(self._internal_dir / "model_raw_warmup.jsonl"),
        }
        with open(index_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def warmup(self, forecast_interface: Any, current_date: date) -> None:
        """Run AllQ-style per-question warmup with one fresh Codex session per question."""
        if self.config.prompt_mode not in {"warmup", "static_search"}:
            return

        mode_label = self.config.prompt_mode.upper()
        print(f"[{self.agent_id}] Starting MinimalHarness {mode_label} phase on {current_date}")

        forecast_interface.current_agent_id = self.agent_id

        questions = list(forecast_interface.questions.values())
        questions.sort(key=lambda q: (self._get_warmup_current_date(current_date, q), str(q.qid)))
        if self.config.warmup_qids:
            requested_qids = {str(qid) for qid in self.config.warmup_qids}
            questions = [q for q in questions if str(q.qid) in requested_qids]
            missing_qids = requested_qids - {str(q.qid) for q in questions}
            if missing_qids:
                raise ValueError(
                    f"warmup_qids not found in active questions: {sorted(missing_qids)}"
                )
        total = len(questions)
        if self.config.prompt_mode == "static_search":
            self._prepare_static_search_files(questions, current_date)

        max_workers = max(1, min(int(self.config.warmup_parallelism or 1), total or 1))
        print(f"[{self.agent_id}] Parallelizing MinimalHarness warmup with {max_workers} worker(s)...")

        runtime_root = self._internal_dir / "warmup_runtime"
        runtime_root.mkdir(parents=True, exist_ok=True)
        index_path = self._internal_dir / "warmup_logs" / "index.jsonl"
        index_path.unlink(missing_ok=True)

        completed = 0
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_q = {
                    executor.submit(
                        self._run_single_warmup_question,
                        q,
                        current_date,
                        forecast_interface,
                    ): q
                    for q in questions
                }
                for future in as_completed(future_to_q):
                    q = future_to_q[future]
                    qid = str(q.qid)
                    effective_current_date = self._get_warmup_current_date(current_date, q)
                    error = None
                    try:
                        _, effective_current_date, predictions = future.result()
                    except Exception as e:
                        logger.warning("Warmup question %s failed: %s", qid, e)
                        predictions = []
                        error = str(e)

                    submitted = False
                    for pred in predictions:
                        try:
                            forecast_interface.submit_prediction(PredictionSubmission(
                                question_id=pred["question_id"],
                                outcomes=pred["outcomes"],
                            ))
                            submitted = True
                        except Exception as e:
                            logger.warning("Failed to submit warmup prediction %s: %s", pred, e)

                    self._append_warmup_log_index(qid, effective_current_date, predictions, error=error)
                    if not submitted:
                        logger.warning("No warmup prediction submitted for qid %s", qid)

                    completed += 1
                    if completed % 10 == 0 or completed == total:
                        print(f"[{self.agent_id}] Warmup Progress: {completed}/{total}", flush=True)
        finally:
            shutil.rmtree(runtime_root, ignore_errors=True)

        aggregated = aggregate_codex_warmup_logs(
            self.agent_id,
            self._internal_dir,
            remove_per_question_logs=True,
        )
        if aggregated:
            print(f"[{self.agent_id}] Aggregated {aggregated} Codex warmup transcript(s).", flush=True)

        self._initial_prompt_override = None
        self.warmed_up = True
        print(f"[{self.agent_id}] MinimalHarness {self.config.prompt_mode} complete.")

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
        With sandbox=True the command becomes a socat bridge to the
        host-side MCP relay (see _build_mcp_invocation).
        """
        mcp_cmd, mcp_args = self._build_mcp_invocation()

        # These modes deliver the full task as the user prompt instead of
        # AGENTS.md; clear any stale copy from a prior config so Codex can't
        # pick it up.
        if self.config.prompt_mode in {"active_memory", "warmup", "static_search", "active_memory2"}:
            stale_agents_md = self.workspace / "AGENTS.md"
            if stale_agents_md.exists():
                stale_agents_md.unlink()
        else:
            system_prompt = (self._internal_dir / "system_prompt.md").read_text()
            system_prompt = _use_codex_mcp_tool_names(system_prompt)
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
            # Codex exposes native web_search by default in current CLIs. Forecast
            # sims must use the date-capped MCP search_news tool / staged articles
            # only, so remove native web search from every codex exec/resume call.
            "-c", 'web_search="disabled"',
            "-c", f'mcp_servers.forecast.command="{mcp_cmd}"',
            "-c", f"mcp_servers.forecast.args={json.dumps(mcp_args)}",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "--json",
        ]
        if self.config.network_isolation:
            proxy_url = f"http://127.0.0.1:{self._egress_proxy_port}"
            common_args.extend(["-c", f'network.proxy_url="{proxy_url}"'])
        if self.config.prompt_mode == "static_search":
            common_args.extend([
                "--disable", "shell_tool",
                "--disable", "shell_snapshot",
                "--disable", "apply_patch_freeform",
                "--disable", "browser_use",
                "--disable", "computer_use",
                "--disable", "tool_search",
                "--disable", "tool_suggest",
                "--disable", "image_generation",
                "--disable", "multi_agent",
            ])

        if self.config.codex_resume and self._codex_thread_id is None:
            self._find_codex_thread_id()
        use_resume = (
            self.config.codex_resume
            and self._codex_thread_id is not None
        )

        if use_resume:
            day_iso = (
                self._current_date.isoformat()
                if self._current_date else ""
            )
            if self.config.prompt_mode in {"warmup", "static_search"} and self._initial_prompt_override:
                resume_prompt = (
                    "The previous per-question warmup session was interrupted. "
                    "Continue the same target question from the current state, "
                    "submit the forecast if needed.\n\n"
                    f"{self._initial_prompt_override}"
                )
            else:
                resume_prompt = (
                    f"Simulation advanced to {day_iso}. Re-read state.json and "
                    "market.csv for the new day's questions, resolution events, "
                    "and your existing predictions; update predictions where "
                    "new info changes your view, then call next_day."
                )
            resume_prompt = _use_codex_mcp_tool_names(resume_prompt)
            cmd = [
                codex_bin, "exec", "resume", self._codex_thread_id,
                *common_args, resume_prompt,
            ]
        else:
            if (
                self.config.prompt_mode in {"active_memory", "warmup", "static_search", "active_memory2"}
                and self._initial_prompt_override
            ):
                initial_prompt = self._initial_prompt_override
            else:
                initial_prompt = (
                    "Begin forecasting. Read market.csv to see your questions, "
                    "then research and submit predictions."
                )
            initial_prompt = _use_codex_mcp_tool_names(initial_prompt)
            cmd = [
                codex_bin, "exec",
                *common_args,
                "-C", str(self.workspace),
                initial_prompt,
            ]
        cmd.extend(self.config.extra_flags)

        env = os.environ.copy()
        if self.config.network_isolation:
            env.update(self._egress_env_overrides())

        self._launch_harness_process(
            cmd,
            stdout_log_name="codex_stdout.jsonl",
            stderr_log_name="codex_stderr.log",
            env=env,
            label="codex",
        )

    def _find_codex_thread_id(self) -> Optional[str]:
        """Scan codex_stdout.jsonl for the first thread.started event.

        codex emits {"type":"thread.started","thread_id":"<uuid>"} as its
        first event; we cache it so codex_resume mode can pass that uuid to
        `codex exec resume <thread_id>` on subsequent days.
        """
        if self._codex_thread_id:
            return self._codex_thread_id
        self._codex_thread_id = self._scan_jsonl_for_field(
            self._internal_dir / "codex_stdout.jsonl",
            lambda ev: ev.get("type") == "thread.started",
            "thread_id",
            last_match=True,
        )
        return self._codex_thread_id

    def _resume_codex(self, current_date: date) -> bool:
        """Relaunch codex via `exec resume <thread_id>` after early exit.

        Reuses _start_codex's resume branch by ensuring _codex_thread_id is
        populated and temporarily forcing config.codex_resume=True. Returns
        False if no prior thread_id is recoverable from codex_stdout.jsonl.
        """
        if self._codex_thread_id is None:
            self._find_codex_thread_id()
        if self._codex_thread_id is None:
            logger.error("codex died and no thread_id found — cannot resume")
            return False
        rc = self._harness_proc.returncode if self._harness_proc else "?"
        prev = self.config.codex_resume
        self.config.codex_resume = True
        try:
            self._start_codex()
        finally:
            self.config.codex_resume = prev
        logger.warning(
            "codex exited (rc=%s) — resumed thread %s for %s",
            rc, self._codex_thread_id, current_date,
        )
        return True
