"""OpenReward/ORS sandbox runner for Futuresim."""

from __future__ import annotations

import json
import os
import shlex
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Optional

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

try:
    from openreward import AsyncOpenReward, SandboxBucketConfig, SandboxSettings
    from openreward.environments import Environment, JSONObject, TextBlock, ToolOutput, tool
except ImportError as exc:  # pragma: no cover - optional integration dependency
    _OPENREWARD_IMPORT_ERROR: Optional[BaseException] = exc
else:
    _OPENREWARD_IMPORT_ERROR = None


RUN_PROMPT = (
    "Run the configured Futuresim MCP harness in the sandbox. The harness will "
    "use its own MCP forecast tools and the date-gated filesystem corpus."
)


def _default_task_rows() -> list[dict[str, Any]]:
    raw = os.environ.get("FSIM_OPENREWARD_TASKS") or os.environ.get("FSIM_HOSTED_TASKS")
    if raw:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else [parsed]

    config = MinimalHarnessRunnerConfig()
    config.futuresim.articles_base = default_articles_base()
    spec = config.to_task_spec()
    return [
        {
            "example_id": "futuresim-openreward",
            "prompt": RUN_PROMPT,
            "answer": "",
            "futuresim": spec["futuresim"],
            "minimal_harness": {key: val for key, val in spec.items() if key != "futuresim"},
            "task_kind": "futuresim_mcp_runner",
        }
    ]


def _resolve_block_network(sandbox_cfg: dict[str, Any]) -> bool:
    """OpenReward uses block_network=True to block all outer egress."""
    if "block_network" in sandbox_cfg:
        return as_bool(sandbox_cfg["block_network"])
    if "network_access" in sandbox_cfg:
        return not as_bool(sandbox_cfg["network_access"])
    return False


if _OPENREWARD_IMPORT_ERROR is None:

    class _OpenRewardSandboxController:
        def __init__(self, env: "FuturesimOpenRewardEnv"):
            self.env = env

        async def upload_file(self, local_path: Path, remote_path: str) -> None:
            parent = str(PurePosixPath(remote_path).parent)
            await self.env.sandbox.run(f"mkdir -p {shlex.quote(parent)}")
            await self.env._upload_file(local_path, remote_path)

        async def run(
            self,
            command: str,
            *,
            timeout_seconds: Optional[int] = None,
        ) -> SandboxCommandResult:
            result = await self.env.sandbox.run(command)
            return coerce_command_result(result)


    class FuturesimOpenRewardEnv(Environment):
        """Futuresim as an OpenReward/ORS MCP runner environment."""

        def __init__(self, task_spec: JSONObject = {}, secrets: dict[str, str] = {}) -> None:
            super().__init__(task_spec, secrets)
            self.secrets = dict(secrets or {})
            self.task_spec = dict(task_spec or {})
            self.config = MinimalHarnessRunnerConfig.from_mapping(self.task_spec)
            if not self.config.futuresim.articles_base:
                self.config.futuresim.articles_base = default_articles_base()
            self.runner: Optional[FuturesimMcpRunner] = None
            self._sandbox_started = False
            self._chunk_dir = tempfile.TemporaryDirectory(prefix="futuresim-openreward-upload-")

            sandbox_cfg = dict(self.task_spec.get("openreward_sandbox") or {})
            self._max_upload_bytes = int(sandbox_cfg.get("max_upload_bytes", 9_000_000))
            api_key = (
                self.secrets.get("api_key")
                or self.secrets.get("OPENREWARD_API_KEY")
                or os.environ.get("OPENREWARD_API_KEY")
            )
            environment_name = (
                sandbox_cfg.get("environment")
                or self.secrets.get("environment")
                or os.environ.get("OPENREWARD_ENVIRONMENT")
            )
            if not api_key:
                raise ValueError("OpenReward sandbox mode requires OPENREWARD_API_KEY or secrets['api_key'].")
            if not environment_name:
                raise ValueError(
                    "OpenReward sandbox mode requires OPENREWARD_ENVIRONMENT, "
                    "secrets['environment'], or task_spec['openreward_sandbox']['environment']."
                )

            bucket_config = sandbox_cfg.get("bucket_config")
            self.sandbox_settings = SandboxSettings(
                environment=environment_name,
                image=sandbox_cfg.get("image", "generalreasoning/python-ds:3.12-tools"),
                machine_size=sandbox_cfg.get("machine_size", "2:8"),
                block_network=_resolve_block_network(sandbox_cfg),
                env=sandbox_cfg.get("env"),
                bucket_config=SandboxBucketConfig(**bucket_config) if bucket_config else None,
            )
            self.sandbox = AsyncOpenReward(api_key=api_key).sandbox(self.sandbox_settings)

        async def _ensure_sandbox_started(self) -> None:
            if not self._sandbox_started:
                await self.sandbox.start()
                self._sandbox_started = True

        async def _upload_file(self, local_path: Path, remote_path: str) -> None:
            if local_path.stat().st_size <= self._max_upload_bytes:
                await self.sandbox.upload(local_path, remote_path)
                return

            chunk_root = Path(self._chunk_dir.name) / PurePosixPath(remote_path).name
            chunk_root.mkdir(parents=True, exist_ok=True)
            for old_chunk in chunk_root.iterdir():
                old_chunk.unlink()

            remote_chunk_dir = f"{remote_path}.chunks"
            await self.sandbox.run(f"rm -rf {shlex.quote(remote_chunk_dir)} && mkdir -p {shlex.quote(remote_chunk_dir)}")
            with open(local_path, "rb") as src:
                index = 0
                while True:
                    data = src.read(self._max_upload_bytes)
                    if not data:
                        break
                    chunk_path = chunk_root / f"{index:06d}.part"
                    chunk_path.write_bytes(data)
                    await self.sandbox.upload(chunk_path, f"{remote_chunk_dir}/{chunk_path.name}")
                    index += 1
            await self.sandbox.run(
                f"cat {shlex.quote(remote_chunk_dir)}/*.part > {shlex.quote(remote_path)} "
                f"&& rm -rf {shlex.quote(remote_chunk_dir)}"
            )

        async def setup(self) -> None:
            await self._ensure_sandbox_started()

        async def teardown(self) -> None:
            if self.runner is not None:
                self.runner.close()
                self.runner = None
            self._chunk_dir.cleanup()
            if self._sandbox_started:
                await self.sandbox.stop()
                self._sandbox_started = False

        async def get_prompt(self) -> list[TextBlock]:
            return [TextBlock(text=RUN_PROMPT)]

        @tool
        async def run_minimal_harness(self) -> ToolOutput:
            """Run the configured MinimalHarness-compatible CLI agent through MCP."""
            await self._ensure_sandbox_started()
            self.runner = FuturesimMcpRunner(self.config)
            try:
                reward = await self.runner.run_to_completion(_OpenRewardSandboxController(self))
            finally:
                self.runner = None
            return ToolOutput(
                blocks=[TextBlock(text=f"Futuresim MCP harness completed.\nFinal reward: {reward:.6f}")],
                metadata={"runner": "minimal_harness_mcp"},
                reward=reward,
                finished=True,
            )

        @classmethod
        def list_tasks(cls, split: str) -> list[JSONObject]:
            if split not in {"train", "validation", "test"}:
                raise ValueError(f"Unknown split: {split}")
            return _default_task_rows()

        @classmethod
        def list_splits(cls) -> list[str]:
            return ["train", "validation", "test"]

else:

    class FuturesimOpenRewardEnv:  # pragma: no cover - optional dependency placeholder
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "OpenReward integration requires `openreward`."
            ) from _OPENREWARD_IMPORT_ERROR
