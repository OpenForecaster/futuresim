"""Run MinimalHarness-compatible agents through the existing Futuresim MCP server."""

from __future__ import annotations

import hashlib
import json
import shlex
import tempfile
from dataclasses import asdict, dataclass, field, fields
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any, Optional, Protocol

import pandas as pd

from futuresim_agents.minimalHarnessAgent.prompts.prompt import build_system_prompt
from integrations.adapter_runtime import (
    FuturesimAdapterConfig,
    FuturesimAdapterRuntime,
    SandboxCommandResult,
    SandboxUpload,
    parse_iso_date,
)


class SandboxController(Protocol):
    async def upload_file(self, local_path: Path, remote_path: str) -> None:
        ...

    async def run(
        self,
        command: str,
        *,
        timeout_seconds: Optional[int] = None,
    ) -> SandboxCommandResult:
        ...


@dataclass
class MinimalHarnessRunnerConfig:
    """Configuration for the optional MCP-backed MinimalHarness runner."""

    futuresim: FuturesimAdapterConfig = field(default_factory=FuturesimAdapterConfig)
    harness_backend: str = "codex"
    model: str = "gpt-5.4"
    prompt_mode: str = "default"
    handholding_version: Optional[str] = None
    timeout_seconds: int = 7200
    python_path: str = "python"
    repo_root_remote: str = ""
    workspace_subdir: str = "workspace"
    internal_subdir: str = ""
    codex_path: str = "codex"
    codex_resume: bool = False
    codex_thread_id: str = ""
    reasoning_effort: str = "xhigh"
    claude_code_path: str = "claude"
    claude_code_resume: bool = False
    claude_session_id: str = ""
    max_budget_usd: Optional[float] = None
    extra_flags: list[str] = field(default_factory=list)
    search_db_remote: str = ""
    embedding_model_remote: str = ""
    embedding_server_url: str = ""
    allow_raw_search_artifacts: bool = False
    agent_filesystem_sandbox: bool = True
    network_isolation: bool = True
    sandbox_proc_mode: str = "none"
    egress_allowlist: list[str] = field(default_factory=list)
    egress_proxy_port: int = 18765
    host_runtime_dir: str = ""

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "MinimalHarnessRunnerConfig":
        value = dict(value or {})
        futuresim_value = value.get("futuresim") if isinstance(value.get("futuresim"), dict) else value
        futuresim = FuturesimAdapterConfig.from_mapping(futuresim_value)

        runner_values: dict[str, Any] = {}
        for key in ("minimal_harness", "harness", "agent"):
            nested = value.get(key)
            if isinstance(nested, dict):
                runner_values.update(nested)
        runner_values.update({key: val for key, val in value.items() if key not in {"futuresim"}})

        allowed = {item.name for item in fields(cls)} - {"futuresim"}
        config = cls(
            futuresim=futuresim,
            **{key: val for key, val in runner_values.items() if key in allowed},
        )
        config.normalize()
        return config

    def normalize(self) -> None:
        if self.prompt_mode not in {"default", "no_memory"}:
            raise ValueError(
                "Platform MCP runner currently supports prompt_mode='default' or 'no_memory'. "
                f"Use the local MinimalHarnessAgent for {self.prompt_mode!r} until its "
                "memory/warmup bootstrapping is lifted into this runner."
            )
        if self.harness_backend not in {"codex", "claude_code"}:
            raise ValueError(
                "Platform MCP runner currently supports harness_backend='codex' or 'claude_code'."
            )
        if self.handholding_version is None:
            model_lc = (self.model or "").lower()
            self.handholding_version = (
                "v3"
                if any(fam in model_lc for fam in ("deepseek", "qwen", "glm"))
                else "v1"
            )
        if self.handholding_version not in {"v1", "v2", "v3"}:
            raise ValueError("handholding_version must be one of 'v1', 'v2', or 'v3'.")
        if self.extra_flags is None:
            self.extra_flags = []
        if self.sandbox_proc_mode not in {"none", "new", "host_ro"}:
            raise ValueError("sandbox_proc_mode must be 'none', 'new', or 'host_ro'.")
        if self.egress_allowlist is None:
            self.egress_allowlist = []
        if (
            self.futuresim.enable_hybrid_search
            and not self.agent_filesystem_sandbox
            and not self.allow_raw_search_artifacts
        ):
            raise ValueError(
                "Hosted MinimalHarness hybrid search needs raw LanceDB/search artifacts for MCP. "
                "Those artifacts must not be readable by the agent CLI. The current hosted MCP "
                "runner can do this with agent_filesystem_sandbox=True, which starts MCP/search "
                "outside the CLI bubblewrap sandbox and exposes only a Unix-socket relay. Set "
                "allow_raw_search_artifacts=True only for a trusted local/dev direct-MCP run."
            )

    def to_task_spec(self) -> dict[str, Any]:
        data = asdict(self)
        data["futuresim"] = self.futuresim.to_task_spec()
        return data

    @property
    def agent_root_path(self) -> str:
        return self.futuresim.sandbox_workspace.rstrip("/") or "/workspace"

    @property
    def workspace_path(self) -> str:
        subdir = str(self.workspace_subdir or "").strip("/")
        if not subdir:
            return self.agent_root_path
        return f"{self.agent_root_path}/{subdir}"

    @property
    def search_db_for_mcp(self) -> str:
        if not self.futuresim.enable_hybrid_search:
            return ""
        return self.search_db_remote or self.futuresim.hybrid_search.search_db

    @property
    def embedding_model_for_mcp(self) -> str:
        if not self.futuresim.enable_hybrid_search:
            return ""
        return self.embedding_model_remote or self.futuresim.hybrid_search.embedding_model

    @property
    def embedding_server_for_mcp(self) -> str:
        if not self.futuresim.enable_hybrid_search:
            return ""
        return self.embedding_server_url or self.futuresim.hybrid_search.embedding_server_url

    @property
    def search_cutoff_days(self) -> int:
        if self.futuresim.hybrid_search.search_cutoff_days:
            return max(0, int(self.futuresim.hybrid_search.search_cutoff_days))
        return max(0, int(self.futuresim.article_search_cutoff_days or 0))


class FuturesimMcpRunner:
    """Drive a sandboxed MinimalHarness CLI through Futuresim's MCP files."""

    def __init__(
        self,
        config: MinimalHarnessRunnerConfig,
        *,
        output_dir: str | None = None,
    ):
        self.config = config
        runner_output_dir = output_dir or config.futuresim.output_base or None
        self.output_dir = Path(runner_output_dir) if runner_output_dir else None
        self.runtime = FuturesimAdapterRuntime(
            config.futuresim,
            output_dir=runner_output_dir,
            workspace_path=config.workspace_path,
        )
        self.workspace_path = self.runtime.workspace_path
        suffix = str(self.config.internal_subdir or "").strip("/")
        self.internal_path = (
            f"{self.config.agent_root_path}/{suffix}" if suffix else self.config.agent_root_path
        )
        if config.repo_root_remote:
            self._mcp_repo_root = config.repo_root_remote
            self._mcp_entry_args = [
                str(
                    PurePosixPath(config.repo_root_remote)
                    / "agents"
                    / "minimalHarnessAgent"
                    / "mcp_server.py"
                )
            ]
        else:
            self._mcp_repo_root = ""
            self._mcp_entry_args = ["-m", "futuresim_agents.minimalHarnessAgent.mcp_server"]
        self._stage = tempfile.TemporaryDirectory(prefix="futuresim-mcp-runner-")
        self._codex_thread_id: Optional[str] = config.codex_thread_id or None
        self._claude_session_id: Optional[str] = config.claude_session_id or None

    def close(self) -> None:
        self.runtime.close()
        self._stage.cleanup()

    async def run_to_completion(self, sandbox: SandboxController) -> float:
        try:
            while not self.runtime.env.is_complete():
                self.runtime.begin_day()
                current_date = self.runtime.env.current_date
                await self.prepare_day(sandbox)

                command = self.build_harness_shell_command(current_date)
                result = await sandbox.run(command, timeout_seconds=self.config.timeout_seconds)
                self._write_harness_output(current_date, command, result)
                self._capture_harness_ids(result.output)

                signal = await self._read_remote_json(sandbox, self._signal_path(current_date))
                if signal is None:
                    detail = result.output[-2000:] if result.output else ""
                    raise RuntimeError(
                        f"Harness exited without next_day signal for {current_date.isoformat()} "
                        f"(exit {result.exit_code}).\n{detail}"
                    )

                predictions = signal.get("predictions", [])
                self.runtime.submit_predictions(list(predictions or []))
                day_result = self.runtime.finish_day()
                if day_result.done:
                    return day_result.reward
            return self.runtime.reward()
        finally:
            self.close()

    async def prepare_day(self, sandbox: SandboxController) -> None:
        current_date = self.runtime.env.current_date
        await sandbox.run(
            " && ".join(
                [
                    f"mkdir -p {shlex.quote(self.workspace_path)}",
                    f"mkdir -p {shlex.quote(self.internal_path)}",
                    f"mkdir -p {shlex.quote(self.internal_path + '/signals')}",
                    f"mkdir -p {shlex.quote(self.runtime.remote_path('memory'))}",
                    f"mkdir -p {shlex.quote(self.runtime.articles_remote_dir)}",
                    f"rm -f {shlex.quote(self.internal_path + '/signals')}/next_day_*.json",
                    f"rm -f {shlex.quote(self.internal_path + '/signals')}/continue_*.json",
                ]
            )
        )

        uploads, article_marker = self._prepare_day_uploads()
        for upload in uploads:
            parent = str(PurePosixPath(upload.remote_path).parent)
            await sandbox.run(f"mkdir -p {shlex.quote(parent)}")
            await sandbox.upload_file(upload.local_path, upload.remote_path)
        self.runtime.commit_article_uploads(article_marker)

    def _prepare_day_uploads(self) -> tuple[list[SandboxUpload], Optional[date]]:
        current_date = self.runtime.env.current_date
        stage_root = Path(self._stage.name)
        stage_root.mkdir(parents=True, exist_ok=True)

        market_path = stage_root / "market.csv"
        state_path = stage_root / "state.json"
        prompt_path = stage_root / "system_prompt.md"
        mcp_config_path = stage_root / "mcp_config.json"
        agents_path = stage_root / "AGENTS.md"
        harness_spec_path = stage_root / f"harness_spec_{current_date.isoformat()}.json"

        self._write_market_csv(market_path)
        self._write_state(state_path)
        prompt_text = self._build_system_prompt()
        if self.config.agent_filesystem_sandbox:
            mcp_command, mcp_args = f"{self._remote_host_runtime_dir(current_date)}/connect_mcp.sh", []
        else:
            mcp_command, mcp_args = self._mcp_command_and_args()
        prompt_path.write_text(prompt_text)
        agents_path.write_text(prompt_text)
        mcp_config_path.write_text(
            json.dumps(
                {"mcpServers": {"forecast": {"command": mcp_command, "args": mcp_args}}},
                indent=2,
            )
        )
        if self.config.agent_filesystem_sandbox:
            harness_spec_path.write_text(json.dumps(self._sandbox_harness_spec(current_date), indent=2))

        uploads = [
            SandboxUpload(market_path, self.runtime.market_remote_path),
            SandboxUpload(state_path, self._remote_internal_path("state.json")),
            SandboxUpload(prompt_path, self._remote_internal_path("system_prompt.md")),
            SandboxUpload(mcp_config_path, self._remote_internal_path("mcp_config.json")),
            SandboxUpload(agents_path, self.runtime.remote_path("AGENTS.md")),
        ]
        if self.config.agent_filesystem_sandbox:
            uploads.append(
                SandboxUpload(
                    harness_spec_path,
                    f"{self._remote_host_runtime_dir(current_date)}/harness_spec.json",
                )
            )
        article_uploads, marker = self.runtime.prepare_article_uploads()
        uploads.extend(article_uploads)
        return uploads, marker

    def _write_market_csv(self, path: Path) -> None:
        forecast_interface = self.runtime.forecast_interface()
        src = forecast_interface.get_market_csv_path()
        if not src:
            raise RuntimeError("Simulation did not produce a market.csv path.")
        df = pd.read_csv(src, dtype={"qid": str})
        agent_preds = forecast_interface.get_agent_predictions(self.runtime.agent_id) or {}
        df["my_prediction"] = df["qid"].apply(
            lambda qid: json.dumps(agent_preds[qid]["outcomes"])
            if qid in agent_preds and agent_preds[qid].get("outcomes") else None
        )
        df["my_prediction_date"] = df["qid"].apply(
            lambda qid: str(agent_preds[qid]["date"])
            if qid in agent_preds and agent_preds[qid].get("date") else None
        )
        df.to_csv(path, index=False)

    def _write_state(self, path: Path) -> None:
        forecast_interface = self.runtime.forecast_interface()
        current_date = self.runtime.env.current_date
        search_current_date = (
            parse_iso_date(self.config.futuresim.start_date)
            if self.config.futuresim.article_freeze_after_start
            else current_date
        )
        questions = []
        for q in forecast_interface.list_questions():
            questions.append(
                {
                    "qid": q.qid,
                    "title": q.title,
                    "background": q.background,
                    "resolution_criteria": q.resolution_criteria,
                    "answer_type": q.answer_type,
                    "resolution_date": str(q.resolution_date) if q.resolution_date else None,
                    "options": q.options,
                }
            )

        agent_predictions = {}
        raw_predictions = forecast_interface.get_agent_predictions(self.runtime.agent_id) or {}
        for qid, record in raw_predictions.items():
            if isinstance(record, dict) and record.get("outcomes"):
                agent_predictions[qid] = record["outcomes"]

        state = {
            "current_date": current_date.isoformat(),
            "search_current_date": search_current_date.isoformat() if search_current_date else "",
            "start_date": self.runtime.env.start_date.isoformat(),
            "end_date": self.config.futuresim.end_date,
            "market_csv": self.runtime.market_remote_path,
            "search_db": "" if self.config.agent_filesystem_sandbox else self.config.search_db_for_mcp,
            "embedding_model": "" if self.config.agent_filesystem_sandbox else self.config.embedding_model_for_mcp,
            "search_type": "hybrid" if self.config.futuresim.enable_hybrid_search else "",
            "search_cutoff_days": self.config.search_cutoff_days,
            "timeout_seconds": self.config.timeout_seconds,
            "questions": questions,
            "resolution_events": forecast_interface.resolution_events,
            "agent_predictions": agent_predictions,
            "total_predictions": len(agent_predictions),
            "prompt_mode": self.config.prompt_mode,
            "submit_ends_session": False,
            "next_day_returns_immediately": True,
            "max_outcomes_per_question": int(self.config.futuresim.max_outcomes_per_question),
        }
        path.write_text(json.dumps(state, indent=2, default=str))

    def _build_system_prompt(self) -> str:
        forecast_interface = self.runtime.forecast_interface()
        questions = forecast_interface.list_questions()
        if self.config.prompt_mode == "no_memory":
            from futuresim_agents.minimalHarnessAgent.prompts.prompt_no_memory import (
                build_system_prompt as prompt_builder,
            )
        else:
            prompt_builder = build_system_prompt
        return prompt_builder(
            workspace=self.workspace_path,
            current_date=self.runtime.env.current_date,
            start_date=self.runtime.env.start_date,
            end_date=parse_iso_date(self.config.futuresim.end_date) or self.runtime.env.current_date,
            source_context=getattr(forecast_interface, "source_context", "") or "",
            source_name=getattr(forecast_interface, "source_name", "openforesight"),
            num_questions=len(questions) + len(getattr(forecast_interface, "resolved_questions", [])),
            num_active=len(questions),
            num_resolved=len(getattr(forecast_interface, "resolved_questions", [])),
            max_outcomes_per_question=self.config.futuresim.max_outcomes_per_question,
            search_cutoff_days=self.config.search_cutoff_days,
            timegap_days=self.config.futuresim.timegap_days,
            new_articles_count=None,
            last_active_date=getattr(forecast_interface, "last_active_date", None),
            next_active_date=getattr(forecast_interface, "next_active_date", None),
            handholding_version=self.config.handholding_version,
        )

    def _agent_prompt(self, current_date: date, backend: str) -> str:
        if backend == "codex" and self.config.codex_resume and self._codex_thread_id:
            return (
                f"Simulation advanced to {current_date.isoformat()}. Re-read state.json "
                "and market.csv for the new day's questions, resolution events, and "
                "your existing predictions; update predictions where new info changes "
                "your view, then call next_day."
            )
        if backend == "claude_code" and self.config.claude_code_resume and self._claude_session_id:
            return (
                f"Simulation resumed at {current_date.isoformat()}. The previous session was "
                "interrupted; re-read state.json and market.csv for the "
                "current day's questions, resolutions, and your existing "
                "predictions, then continue research, update predictions "
                "where new info changes your view, and call next_day."
            )
        return (
            "Begin forecasting. Read market.csv to see your questions, "
            "then research and submit predictions."
        )

    def build_harness_shell_command(self, current_date: date) -> str:
        if self.config.agent_filesystem_sandbox:
            cmd = [
                self.config.python_path,
                "-m",
                "integrations.sandbox_harness",
                "--spec",
                f"{self._remote_host_runtime_dir(current_date)}/harness_spec.json",
            ]
        elif self.config.harness_backend == "codex":
            cmd = self._codex_command(current_date)
        elif self.config.harness_backend == "claude_code":
            cmd = self._claude_code_command(current_date)
        else:
            raise ValueError(f"Unsupported harness_backend={self.config.harness_backend!r}")
        log_path = self._remote_internal_path(
            "harness_outputs", f"{current_date.isoformat()}.stdout.log"
        )
        log_dir = str(PurePosixPath(log_path).parent)
        quoted_log = shlex.quote(log_path)
        quoted_signal = shlex.quote(self._signal_path(current_date))
        timeout = max(1, int(self.config.timeout_seconds))
        drain_timeout = min(30, timeout)
        return (
            f"mkdir -p {shlex.quote(log_dir)} && "
            f"cd {shlex.quote(self.workspace_path)} && "
            f"rm -f {quoted_log} || exit $?; "
            f"({shlex.join(cmd)}) > {quoted_log} 2>&1 & "
            "pid=$!; "
            "status=0; "
            f"signal_path={quoted_signal}; "
            f"deadline=$(( $(date +%s) + {timeout} )); "
            'while kill -0 "$pid" 2>/dev/null && [ ! -f "$signal_path" ]; do '
            f"if [ -f {quoted_log} ] && grep -q '\"type\":\"turn.failed\"' {quoted_log}; then "
            "status=1; "
            'kill -TERM "$pid" 2>/dev/null || true; '
            "sleep 2; "
            'kill -KILL "$pid" 2>/dev/null || true; '
            'wait "$pid" 2>/dev/null || true; '
            "break; "
            "fi; "
            'if [ "$(date +%s)" -ge "$deadline" ]; then '
            "status=124; "
            'kill -TERM "$pid" 2>/dev/null || true; '
            "sleep 2; "
            'kill -KILL "$pid" 2>/dev/null || true; '
            'wait "$pid" 2>/dev/null || true; '
            "break; "
            "fi; "
            "sleep 1; "
            "done; "
            'if [ "$status" -eq 0 ] && [ -f "$signal_path" ]; then '
            f"drain_deadline=$(( $(date +%s) + {drain_timeout} )); "
            'while kill -0 "$pid" 2>/dev/null && [ "$(date +%s)" -lt "$drain_deadline" ]; do '
            "sleep 1; "
            "done; "
            'if kill -0 "$pid" 2>/dev/null; then '
            'kill -TERM "$pid" 2>/dev/null || true; '
            "sleep 2; "
            'kill -KILL "$pid" 2>/dev/null || true; '
            "fi; "
            'wait "$pid" 2>/dev/null || true; '
            'elif [ "$status" -eq 0 ]; then '
            'wait "$pid"; '
            "status=$?; "
            "fi; "
            f"if [ -f {quoted_log} ]; then cat {quoted_log}; fi; "
            'exit "$status"'
        )

    def _codex_command(self, current_date: date) -> list[str]:
        mcp_command, mcp_args = self._mcp_command_and_args()
        prompt = self._agent_prompt(current_date, "codex")
        common = [
            "-m",
            self.config.model,
            "-c",
            f'model_reasoning_effort="{self.config.reasoning_effort}"',
            "-c",
            'web_search="disabled"',
            "-c",
            f'mcp_servers.forecast.command="{mcp_command}"',
            "-c",
            f"mcp_servers.forecast.args={json.dumps(mcp_args)}",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "--json",
        ]
        if self.config.codex_resume and self._codex_thread_id:
            cmd = [
                self.config.codex_path,
                "exec",
                "resume",
                self._codex_thread_id,
                *common,
                prompt,
            ]
        else:
            cmd = [
                self.config.codex_path,
                "exec",
                *common,
                "-C",
                self.workspace_path,
                prompt,
            ]
        cmd.extend(self.config.extra_flags)
        return cmd

    def _claude_code_command(self, current_date: date) -> list[str]:
        resume_session = self._claude_session_id if self.config.claude_code_resume else None
        prompt = self._agent_prompt(current_date, "claude_code")
        disallowed_tools = (
            "WebSearch,WebFetch,Write,Edit,MultiEdit,NotebookEdit"
            if self.config.prompt_mode == "no_memory"
            else "WebSearch,WebFetch"
        )
        cmd = [self.config.claude_code_path]
        if resume_session:
            cmd.extend(["--resume", resume_session])
        cmd.extend(
            [
                "-p",
                prompt,
                "--verbose",
                "--effort",
                "max",
                "--model",
                self.config.model,
                "--output-format",
                "stream-json",
                "--dangerously-skip-permissions",
                "--mcp-config",
                self._remote_internal_path("mcp_config.json"),
                "--strict-mcp-config",
                "--disallowedTools",
                disallowed_tools,
                "--system-prompt-file",
                self._remote_internal_path("system_prompt.md"),
                "--add-dir",
                self.workspace_path,
            ]
        )
        if self.config.max_budget_usd is not None:
            cmd.extend(["--max-budget-usd", str(self.config.max_budget_usd)])
        cmd.extend(self.config.extra_flags)
        return cmd

    def _mcp_command_and_args(self) -> tuple[str, list[str]]:
        args = [
            *self._mcp_entry_args,
            "--workspace",
            self.workspace_path,
            "--internal-dir",
            self.internal_path,
            "--search-db",
            self.config.search_db_for_mcp,
            "--embedding-model",
            self.config.embedding_model_for_mcp,
            "--handholding-version",
            str(self.config.handholding_version),
            "--agent-id",
            self.runtime.agent_id,
        ]
        if self._mcp_repo_root:
            args.extend(["--repo-root", self._mcp_repo_root])
        if self.config.embedding_server_for_mcp:
            args.extend(["--embedding-server-url", self.config.embedding_server_for_mcp])
        return self.config.python_path, args

    def _sandbox_harness_spec(self, current_date: date) -> dict[str, Any]:
        return {
            "backend": self.config.harness_backend,
            "workspace": self.workspace_path,
            "internal_dir": self.internal_path,
            "host_runtime_dir": self._remote_host_runtime_dir(current_date),
            "mcp_command": self.config.python_path,
            "mcp_entry_args": list(self._mcp_entry_args),
            "mcp_server_config": {
                "workspace": self.workspace_path,
                "internal_dir": self.internal_path,
                "repo_root": self.config.repo_root_remote,
                "search_db": self.config.search_db_for_mcp,
                "embedding_model": self.config.embedding_model_for_mcp,
                "embedding_server_url": self.config.embedding_server_for_mcp,
                "handholding_version": str(self.config.handholding_version),
                "agent_id": self.runtime.agent_id,
            },
            "model": self.config.model,
            "prompt": self._agent_prompt(current_date, self.config.harness_backend),
            "prompt_mode": self.config.prompt_mode,
            "codex_path": self.config.codex_path,
            "codex_resume": self.config.codex_resume,
            "codex_thread_id": self._codex_thread_id or "",
            "reasoning_effort": self.config.reasoning_effort,
            "claude_code_path": self.config.claude_code_path,
            "claude_code_resume": self.config.claude_code_resume,
            "claude_session_id": self._claude_session_id or "",
            "max_budget_usd": self.config.max_budget_usd,
            "extra_flags": list(self.config.extra_flags or []),
            "network_isolation": self.config.network_isolation,
            "egress_allowlist": list(self.config.egress_allowlist or []),
            "egress_proxy_port": int(self.config.egress_proxy_port),
            "sandbox_proc_mode": self.config.sandbox_proc_mode,
        }

    async def _read_remote_json(self, sandbox: SandboxController, remote_path: str) -> Any:
        result = await sandbox.run(
            f"if [ -f {shlex.quote(remote_path)} ]; then cat {shlex.quote(remote_path)}; fi"
        )
        text = (result.output or "").strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _capture_harness_ids(self, output: str) -> None:
        for line in (output or "").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "thread.started" and event.get("thread_id"):
                self._codex_thread_id = str(event["thread_id"])
            if (
                event.get("type") == "system"
                and event.get("subtype") == "init"
                and event.get("session_id")
            ):
                self._claude_session_id = str(event["session_id"])

    def _write_harness_output(
        self,
        current_date: date,
        command: str,
        result: SandboxCommandResult,
    ) -> None:
        if self.output_dir is None:
            return
        output_dir = self.output_dir / "harness_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        date_key = current_date.isoformat()
        (output_dir / f"{date_key}.stdout.log").write_text(result.output or "")
        (output_dir / f"{date_key}.meta.json").write_text(
            json.dumps(
                {
                    "command": command,
                    "exit_code": result.exit_code,
                    "truncated": result.truncated,
                },
                indent=2,
            )
        )

    def _remote_internal_path(self, *parts: str) -> str:
        suffix = "/".join(str(part).strip("/") for part in parts if str(part).strip("/"))
        return f"{self.internal_path}/{suffix}" if suffix else self.internal_path

    def _remote_host_runtime_dir(self, current_date: date) -> str:
        base = self.config.host_runtime_dir.rstrip("/")
        if not base:
            digest = hashlib.sha1(self.config.agent_root_path.encode()).hexdigest()[:12]
            base = f"/tmp/futuresim-hosted-{digest}"
        return f"{base}/{current_date.isoformat()}"

    def _signal_path(self, current_date: date) -> str:
        return self._remote_internal_path("signals", f"next_day_{current_date.isoformat()}.json")
