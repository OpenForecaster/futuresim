"""Prime Intellect Verifiers sandbox runner for Futuresim."""

from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Optional

try:
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    _IS_SOURCE_CHECKOUT = (_REPO_ROOT / "pyproject.toml").exists() and (_REPO_ROOT / "agents").is_dir()
    _old_path = list(sys.path)

    def _module_from_source_agents(module: Any) -> bool:
        if not _IS_SOURCE_CHECKOUT:
            return False
        module_file = getattr(module, "__file__", None)
        if not module_file:
            return False
        return Path(module_file).resolve().is_relative_to(_REPO_ROOT / "agents")

    _saved_agents_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "agents" or name.startswith("agents.")
        if _module_from_source_agents(module)
    }
    for name in list(_saved_agents_modules):
        sys.modules.pop(name, None)
    if _IS_SOURCE_CHECKOUT:
        sys.path = [
            item for item in sys.path
            if item and Path(item).resolve() != _REPO_ROOT
        ]

    import verifiers as vf
    from datasets import Dataset
    from verifiers.envs.sandbox_env import SandboxEnv as _SandboxEnv
except ImportError as exc:  # pragma: no cover - optional integration dependency
    _VERIFIERS_IMPORT_ERROR: Optional[BaseException] = exc
else:
    _VERIFIERS_IMPORT_ERROR = None
finally:
    if globals().get("_saved_agents_modules"):
        for name in [
            key for key in sys.modules
            if key == "agents" or key.startswith("agents.")
        ]:
            if name not in globals().get("_saved_agents_modules", {}):
                sys.modules.pop(name, None)
        sys.modules.update(globals().get("_saved_agents_modules", {}))
    sys.path = globals().get("_old_path", sys.path)

from integrations.adapter_runtime import (
    SandboxCommandResult,
    as_bool,
    coerce_command_result,
    default_articles_base,
)
from integrations.mcp_runner import (
    FuturesimMcpRunner,
    MinimalHarnessRunnerConfig,
)


RUN_PROMPT = (
    "Run the configured Futuresim MCP harness in the sandbox. The harness will "
    "use its own MCP forecast tools and the date-gated filesystem corpus."
)


def _default_task_rows(env_args: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(env_args.get("tasks"), list):
        return list(env_args["tasks"])
    if isinstance(env_args.get("task"), dict):
        return [dict(env_args["task"])]
    raw = os.environ.get("FSIM_VERIFIERS_TASKS") or os.environ.get("FSIM_HOSTED_TASKS")
    if raw:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else [parsed]

    config = MinimalHarnessRunnerConfig.from_mapping(env_args)
    if not config.futuresim.articles_base:
        config.futuresim.articles_base = default_articles_base()
    spec = config.to_task_spec()
    return [
        {
            "example_id": "futuresim-verifiers",
            "prompt": RUN_PROMPT,
            "answer": "",
            "futuresim": spec["futuresim"],
            "minimal_harness": {key: val for key, val in spec.items() if key != "futuresim"},
            "task_kind": "futuresim_mcp_runner",
        }
    ]


def _task_spec_from_state(state: Any) -> dict[str, Any]:
    task = dict(state.get("task") or state.get("input") or {})
    if isinstance(task.get("futuresim"), dict) or isinstance(task.get("minimal_harness"), dict):
        return task
    info = task.get("info")
    if isinstance(info, dict):
        spec: dict[str, Any] = {}
        if isinstance(info.get("futuresim"), dict):
            spec["futuresim"] = dict(info["futuresim"])
        if isinstance(info.get("minimal_harness"), dict):
            spec["minimal_harness"] = dict(info["minimal_harness"])
        if spec:
            return spec
    return task


def _resolve_network_access(sandbox_cfg: dict[str, Any]) -> bool:
    """Prime Sandboxes use network_access=False to block all outer egress."""
    if "network_access" in sandbox_cfg:
        return as_bool(sandbox_cfg["network_access"])
    if "block_network" in sandbox_cfg:
        return not as_bool(sandbox_cfg["block_network"])
    return True


def _set_request_network_access(request: Any, network_access: bool) -> Any:
    if hasattr(request, "model_copy"):
        return _assert_request_network_access(
            request.model_copy(update={"network_access": network_access}),
            network_access,
        )
    if hasattr(request, "copy"):
        return _assert_request_network_access(
            request.copy(update={"network_access": network_access}),
            network_access,
        )
    if not hasattr(request, "network_access"):
        raise RuntimeError(
            "Prime Sandbox request does not support network_access. "
            "Upgrade verifiers/prime-sandboxes before running Futuresim with sandboxed agents."
        )
    setattr(request, "network_access", network_access)
    return _assert_request_network_access(request, network_access)


def _assert_request_network_access(request: Any, network_access: bool) -> Any:
    payload = None
    if hasattr(request, "model_dump"):
        payload = request.model_dump()
    elif hasattr(request, "dict"):
        payload = request.dict()
    if isinstance(payload, dict):
        if payload.get("network_access") == network_access:
            return request
        raise RuntimeError(
            "Prime Sandbox request does not serialize network_access. "
            "Upgrade verifiers/prime-sandboxes before running Futuresim with sandboxed agents."
        )
    if getattr(request, "network_access", None) == network_access:
        return request
    raise RuntimeError(
        "Prime Sandbox request does not support network_access. "
        "Upgrade verifiers/prime-sandboxes before running Futuresim with sandboxed agents."
    )


if _VERIFIERS_IMPORT_ERROR is None:

    async def futuresim_reward(state: vf.State) -> float:
        """Final Futuresim reward emitted by the MCP runner."""
        return float(state.get("futuresim_reward", 0.0) or 0.0)


    class _VerifiersSandboxController:
        def __init__(self, env: Any, state: vf.State):
            self.env = env
            self.state = state

        async def upload_file(self, local_path: Path, remote_path: str) -> None:
            sandbox_id = self.state["sandbox_id"]
            parent = str(PurePosixPath(remote_path).parent)
            await self.env.sandbox_client.execute_command(
                sandbox_id,
                f"mkdir -p {shlex.quote(parent)}",
                timeout=self.env.timeout_per_command_seconds,
            )
            await self.env.sandbox_client.upload_file(sandbox_id, remote_path, str(local_path))

        async def run(
            self,
            command: str,
            *,
            timeout_seconds: Optional[int] = None,
        ) -> SandboxCommandResult:
            result = await self.env.sandbox_client.execute_command(
                self.state["sandbox_id"],
                command,
                timeout=timeout_seconds or self.env.timeout_per_command_seconds,
            )
            return coerce_command_result(result)


    class FuturesimVerifiersEnv(_SandboxEnv):
        """Futuresim as a Prime Intellect Verifiers MCP runner environment."""

        def __init__(self, env_args: Optional[dict[str, Any]] = None, **kwargs: Any):
            env_args = dict(env_args or {})
            task_rows = _default_task_rows(env_args)
            dataset = kwargs.pop("dataset", Dataset.from_list(task_rows))
            rubric = kwargs.pop("rubric", vf.Rubric(funcs=[futuresim_reward]))
            sandbox_cfg = {
                "name": "futuresim",
                "docker_image": "python:3.12-slim",
                "start_command": "tail -f /dev/null",
                "cpu_cores": 2,
                "memory_gb": 8,
                "disk_size_gb": 30,
                "gpu_count": 0,
                "timeout_minutes": 720,
                "timeout_per_command_seconds": 60,
                **dict(env_args.get("sandbox") or {}),
            }
            self._sandbox_network_access = _resolve_network_access(sandbox_cfg)

            super().__init__(
                dataset=dataset,
                eval_dataset=kwargs.pop("eval_dataset", None),
                rubric=rubric,
                sandbox_name=sandbox_cfg["name"],
                docker_image=sandbox_cfg["docker_image"],
                start_command=sandbox_cfg["start_command"],
                cpu_cores=int(sandbox_cfg["cpu_cores"]),
                memory_gb=int(sandbox_cfg["memory_gb"]),
                disk_size_gb=int(sandbox_cfg["disk_size_gb"]),
                gpu_count=int(sandbox_cfg["gpu_count"]),
                timeout_minutes=int(sandbox_cfg["timeout_minutes"]),
                timeout_per_command_seconds=int(sandbox_cfg["timeout_per_command_seconds"]),
                environment_vars=sandbox_cfg.get("environment_vars"),
                team_id=sandbox_cfg.get("team_id"),
                advanced_configs=sandbox_cfg.get("advanced_configs"),
                labels=sandbox_cfg.get("labels", ["futuresim", "mcp-runner"]),
                max_turns=int(env_args.get("max_turns", 10)),
                **kwargs,
            )
            self._runners: dict[str, FuturesimMcpRunner] = {}
            self.add_tool(self.run_minimal_harness, args_to_skip=["state"])

        def get_sandbox_request(self, state: vf.State) -> Any:
            request = super().get_sandbox_request(state)
            return _set_request_network_access(request, self._sandbox_network_access)

        async def setup_state(self, state: vf.State, **kwargs: Any) -> None:
            await super().setup_state(state, **kwargs)
            runner = FuturesimMcpRunner(
                MinimalHarnessRunnerConfig.from_mapping(_task_spec_from_state(state))
            )
            runner_id = str(state.get("trajectory_id") or state.get("example_id"))
            self._runners[runner_id] = runner
            state["futuresim_runner_id"] = runner_id
            state["futuresim_done"] = False
            state["futuresim_reward"] = 0.0
            state["working_dir"] = runner.workspace_path
            state["prompt"] = [{"role": "user", "content": RUN_PROMPT}]

        def update_tool_args(
            self,
            tool_name: str,
            tool_args: dict[str, Any],
            messages: Any,
            state: vf.State,
            **kwargs: Any,
        ) -> dict[str, Any]:
            updated = super().update_tool_args(tool_name, tool_args, messages, state, **kwargs)
            if tool_name == "run_minimal_harness":
                updated["state"] = state
            return updated

        async def _ensure_sandbox_ready(self, state: vf.State) -> None:
            sandbox_state = state["sandbox_state"]
            if not sandbox_state["ready"]:
                await self._wait_for_sandbox_ready(sandbox_state, state["sandbox_id"])

        def _runner_for_state(self, state: vf.State) -> FuturesimMcpRunner:
            runner_id = str(state.get("futuresim_runner_id", ""))
            runner = self._runners.get(runner_id)
            if runner is None:
                raise RuntimeError("Futuresim MCP runner is not initialized for this rollout.")
            return runner

        async def run_minimal_harness(self, state: vf.State) -> str:
            """Run the configured MinimalHarness-compatible CLI agent through MCP."""
            await self._ensure_sandbox_ready(state)
            runner_id = str(state.get("futuresim_runner_id", ""))
            runner = self._runner_for_state(state)
            try:
                reward = await runner.run_to_completion(_VerifiersSandboxController(self, state))
            finally:
                self._runners.pop(runner_id, None)
            state["futuresim_done"] = True
            state["futuresim_reward"] = reward
            state["reward"] = reward
            return f"Futuresim MCP harness completed.\nFinal reward: {reward:.6f}"

        @vf.stop
        async def futuresim_done(self, state: vf.State) -> bool:
            return bool(state.get("futuresim_done", False))

        @vf.stop
        async def no_tools_called(self, state: vf.State) -> bool:
            return False

        async def post_rollout(self, state: vf.State) -> None:
            runner_id = str(state.get("futuresim_runner_id", ""))
            runner = self._runners.pop(runner_id, None)
            if runner is not None:
                runner.close()


    def load_environment(env_args: Optional[dict[str, Any]] = None, **kwargs: Any) -> FuturesimVerifiersEnv:
        """Entry point used by Verifiers environment loading."""
        return FuturesimVerifiersEnv(env_args=env_args, **kwargs)

else:

    class FuturesimVerifiersEnv:  # pragma: no cover - optional dependency placeholder
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "Verifiers integration requires `verifiers`, `prime-sandboxes`, and `datasets`."
            ) from _VERIFIERS_IMPORT_ERROR


    def load_environment(env_args: Optional[dict[str, Any]] = None, **kwargs: Any) -> FuturesimVerifiersEnv:
        raise ImportError(
            "Verifiers integration requires `verifiers`, `prime-sandboxes`, and `datasets`."
        ) from _VERIFIERS_IMPORT_ERROR
