"""Run Futuresim on OpenReward/Firehorse with domain-scoped secrets."""

from __future__ import annotations

import argparse
import asyncio
import copy
import dataclasses
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

OPENROUTER_DOMAINS = ["openrouter.ai", "api.openrouter.ai"]
LOCAL_CLI_OPENAI_KEY_SENTINEL = "__FUTURESIM_LOCAL_CLI_AUTH__"


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    argv = list(sys.argv[1:] if argv is None else argv)

    parser = argparse.ArgumentParser(description="Run Futuresim via OpenReward Firehorse.")
    parser.add_argument("--env", default="ShashwatGoel/futuresim")
    parser.add_argument("--agent", default="codex")
    parser.add_argument("--model", default="openai/gpt-5.5")
    parser.add_argument("--effort", default="xhigh")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-tasks", type=int, default=1)
    parser.add_argument("--skip-tasks", type=int, default=0)
    parser.add_argument("--n-concurrent", type=int, default=1)
    parser.add_argument("--run-name")
    parser.add_argument("--output-dir")
    parser.add_argument("--variant")
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--provider-url")
    parser.add_argument("--toolset")
    parser.add_argument("--no-logging", action="store_true")
    parser.add_argument("--use-env-descriptions", action="store_true")
    parser.add_argument("--use-all-filesystem-tools", action="store_true")
    parser.add_argument(
        "--futuresim-days",
        type=int,
        help=(
            "Limit the Futuresim task to the first N simulation days. "
            "With start_date=2025-12-31 and lookback_days=7, N=2 runs "
            "2025-12-24 and 2025-12-25."
        ),
    )
    parser.add_argument("--futuresim-dataset")
    parser.add_argument("--futuresim-dataset-path")
    parser.add_argument("--futuresim-dataset-cache")
    parser.add_argument("--futuresim-split")
    parser.add_argument("--futuresim-start-date")
    parser.add_argument("--futuresim-end-date")
    parser.add_argument("--futuresim-resolution-start")
    parser.add_argument("--futuresim-resolution-end")
    parser.add_argument("--futuresim-lookback-days", type=int)
    parser.add_argument("--futuresim-output-base")
    parser.add_argument("--futuresim-agent-id")
    parser.add_argument("--futuresim-handholding-version")
    parser.add_argument("--futuresim-matching")
    parser.add_argument("--futuresim-matcher")
    parser.add_argument("--futuresim-matcher-cache")
    parser.add_argument(
        "--futuresim-mount-articles",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override openreward_sandbox.mount_articles in the fetched task spec.",
    )
    if argv and argv[0] in {"resume", "replay"}:
        if not os.environ.get("OPENREWARD_API_KEY"):
            raise SystemExit("OPENREWARD_API_KEY is required.")
        if argv[0] == "resume" and not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = LOCAL_CLI_OPENAI_KEY_SENTINEL
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        has_openrouter_secret = any(
            item.startswith("OPENROUTER_API_KEY=")
            for idx, item in enumerate(argv)
            if idx > 0 and argv[idx - 1] == "--secret"
        )
        if openrouter_key and not has_openrouter_secret:
            argv.extend(["--secret", f"OPENROUTER_API_KEY={openrouter_key}"])
        return _run_with_firehorse_shim(argv, resume_mode=True)

    args = parser.parse_args(argv)

    if not os.environ.get("OPENREWARD_API_KEY"):
        raise SystemExit("OPENREWARD_API_KEY is required.")

    local_cli_agents = {"codex", "claude-code", "claude_code"}
    if (
        args.agent in local_cli_agents
        and args.model.startswith("openai/")
        and not os.environ.get("OPENAI_API_KEY")
    ):
        # Firehorse's top-level provider preflight requires OPENAI_API_KEY for
        # all openai/* models, but local Codex/Claude Code CLIs may authenticate
        # through their own login state. Use a sentinel only to satisfy that
        # preflight, and strip it before spawning the CLI below.
        os.environ["OPENAI_API_KEY"] = LOCAL_CLI_OPENAI_KEY_SENTINEL

    secrets: dict[str, Any] = {}
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        secrets["OPENROUTER_API_KEY"] = (openrouter_key, OPENROUTER_DOMAINS)

    def run_firehorse() -> int:
        from firehorse.config import RunConfig
        from firehorse.orchestrator import run_evaluation

        config = RunConfig(
            env=args.env,
            agent=args.agent,
            model=args.model,
            variant=args.variant,
            n_concurrent=args.n_concurrent,
            split=args.split,
            max_tasks=args.max_tasks,
            skip_tasks=args.skip_tasks,
            run_name=args.run_name,
            max_turns=args.max_turns,
            provider_url=args.provider_url,
            secrets=secrets,
            output_dir=args.output_dir,
            effort=None if args.effort in ("", "none", "None") else args.effort,
            logging=not args.no_logging,
            use_builtin_descriptions=not args.use_env_descriptions,
            use_all_filesystem_tools=args.use_all_filesystem_tools,
            toolset=args.toolset,
        )
        summary = asyncio.run(_run_evaluation_with_futuresim_overrides(config, args))
        return 0 if summary.failed == 0 else 1

    return _run_with_firehorse_shim(run_firehorse)


async def _run_evaluation_with_futuresim_overrides(config: Any, args: argparse.Namespace) -> Any:
    clients: list[Any] = []
    from openreward.client import AsyncOpenReward

    original_client_init = AsyncOpenReward.__init__

    def client_init(self: Any, *init_args: Any, **init_kwargs: Any) -> None:
        original_client_init(self, *init_args, **init_kwargs)
        clients.append(self)

    AsyncOpenReward.__init__ = client_init
    overrides = _futuresim_override_requested(args)
    output_base = args.futuresim_output_base
    if not output_base and args.output_dir:
        output_base = os.path.abspath(args.output_dir)
        overrides = True
    try:
        if not overrides:
            from firehorse.orchestrator import run_evaluation

            return await run_evaluation(config)

        from firehorse.orchestrator import run_evaluation
        from openreward.api.environments.client import AsyncEnvironment

        original_get_task = AsyncEnvironment.get_task

        async def get_task_with_overrides(self: Any, split: str, index: int) -> Any:
            task = await original_get_task(self, split, index)
            patched = _patch_futuresim_task_spec(task.task_spec, args, output_base)
            if patched == task.task_spec:
                return task
            _print_futuresim_task_diff(task.task_spec, patched)
            return dataclasses.replace(task, task_spec=patched)

        AsyncEnvironment.get_task = get_task_with_overrides
        try:
            return await run_evaluation(config)
        finally:
            AsyncEnvironment.get_task = original_get_task
    finally:
        AsyncOpenReward.__init__ = original_client_init
        await _close_openreward_clients(clients)


def _futuresim_override_requested(args: argparse.Namespace) -> bool:
    names = [
        "futuresim_days",
        "futuresim_dataset",
        "futuresim_dataset_path",
        "futuresim_dataset_cache",
        "futuresim_split",
        "futuresim_start_date",
        "futuresim_end_date",
        "futuresim_resolution_start",
        "futuresim_resolution_end",
        "futuresim_lookback_days",
        "futuresim_output_base",
        "futuresim_agent_id",
        "futuresim_handholding_version",
        "futuresim_matching",
        "futuresim_matcher",
        "futuresim_matcher_cache",
        "futuresim_mount_articles",
    ]
    return any(getattr(args, name, None) is not None for name in names)


async def _close_openreward_clients(clients: list[Any]) -> None:
    for client in clients:
        rollout = getattr(client, "_rollout_api", None)
        if rollout is not None and hasattr(rollout, "close"):
            rollout.close()
        environments = getattr(client, "_environments_api", None)
        if environments is not None and hasattr(environments, "aclose"):
            await environments.aclose()


def _patch_futuresim_task_spec(
    task_spec: dict[str, Any],
    args: argparse.Namespace,
    output_base: str | None,
) -> dict[str, Any]:
    patched = copy.deepcopy(task_spec)
    futuresim = patched.setdefault("futuresim", {})
    sandbox = patched.setdefault("openreward_sandbox", {})

    field_args = {
        "dataset": args.futuresim_dataset,
        "dataset_path": args.futuresim_dataset_path,
        "dataset_cache": args.futuresim_dataset_cache,
        "split": args.futuresim_split,
        "start_date": args.futuresim_start_date,
        "end_date": args.futuresim_end_date,
        "resolution_start": args.futuresim_resolution_start,
        "resolution_end": args.futuresim_resolution_end,
        "lookback_days": args.futuresim_lookback_days,
        "agent_id": args.futuresim_agent_id,
        "handholding_version": args.futuresim_handholding_version,
        "matching": args.futuresim_matching,
        "matcher": args.futuresim_matcher,
    }
    for key, value in field_args.items():
        if value is not None:
            futuresim[key] = value
    if args.futuresim_matcher_cache is not None:
        futuresim["matcher_cache"] = {"path": args.futuresim_matcher_cache}
    if output_base:
        futuresim["output_base"] = output_base
    if args.futuresim_mount_articles is not None:
        sandbox["mount_articles"] = args.futuresim_mount_articles

    if args.futuresim_days is not None:
        if args.futuresim_days < 1:
            raise SystemExit("--futuresim-days must be >= 1")
        start = datetime.strptime(str(futuresim["start_date"]), "%Y-%m-%d").date()
        lookback_days = int(futuresim.get("lookback_days") or 0)
        first_sim_day = start - timedelta(days=max(0, lookback_days))
        futuresim["end_date"] = (
            first_sim_day + timedelta(days=args.futuresim_days - 1)
        ).isoformat()

    return patched


def _print_futuresim_task_diff(before: dict[str, Any], after: dict[str, Any]) -> None:
    before_fsim = before.get("futuresim") or {}
    after_fsim = after.get("futuresim") or {}
    before_sandbox = before.get("openreward_sandbox") or {}
    after_sandbox = after.get("openreward_sandbox") or {}
    interesting = [
        ("futuresim.split", before_fsim.get("split"), after_fsim.get("split")),
        ("futuresim.start_date", before_fsim.get("start_date"), after_fsim.get("start_date")),
        ("futuresim.end_date", before_fsim.get("end_date"), after_fsim.get("end_date")),
        ("futuresim.output_base", before_fsim.get("output_base"), after_fsim.get("output_base")),
        (
            "futuresim.handholding_version",
            before_fsim.get("handholding_version"),
            after_fsim.get("handholding_version"),
        ),
        ("futuresim.matcher", before_fsim.get("matcher"), after_fsim.get("matcher")),
        (
            "openreward_sandbox.mount_articles",
            before_sandbox.get("mount_articles"),
            after_sandbox.get("mount_articles"),
        ),
    ]
    changed = [f"{name}={new!r}" for name, old, new in interesting if old != new]
    if changed:
        print("Futuresim task overrides: " + ", ".join(changed), file=sys.stderr)


def _run_with_firehorse_shim(target: Any, *, resume_mode: bool = False) -> int:
    with tempfile.TemporaryDirectory(prefix="futuresim-openreward-") as shim_dir:
        shim = Path(shim_dir) / "sitecustomize.py"
        shim_code = (
            """
try:
    import aiohttp as _aiohttp

    _original_client_session_init = _aiohttp.ClientSession.__init__

    def _client_session_init(self, *args, **kwargs):
        kwargs.setdefault("trust_env", True)
        return _original_client_session_init(self, *args, **kwargs)

    if not getattr(_aiohttp.ClientSession.__init__, "_futuresim_trust_env_default", False):
        _client_session_init._futuresim_trust_env_default = True
        _aiohttp.ClientSession.__init__ = _client_session_init
except Exception:
    pass

try:
    from openreward.api.sandboxes import secrets as _secrets
    from openreward.api.sandboxes import client as _client
    from openreward.api.environments import client as _env_client

    _original = _secrets.build_secrets_header

    def _build_secrets_header(secrets):
        normalized = {
            key: tuple(value)
            if isinstance(value, list) and len(value) == 2 and isinstance(value[1], list)
            else value
            for key, value in secrets.items()
        }
        return _original(normalized)

    _secrets.build_secrets_header = _build_secrets_header
    _client.build_secrets_header = _build_secrets_header
    _env_client.build_secrets_header = _build_secrets_header
except Exception:
    pass
"""
        )
        shim.write_text(
            shim_code,
            encoding="utf-8",
        )
        os.environ["PYTHONPATH"] = (
            f"{shim_dir}{os.pathsep}{os.environ['PYTHONPATH']}"
            if os.environ.get("PYTHONPATH")
            else shim_dir
        )
        exec(compile(shim_code, str(shim), "exec"), {})

        original_create_subprocess_exec = asyncio.create_subprocess_exec

        async def create_subprocess_exec(*cmd, **kwargs):
            proc_env = kwargs.get("env")
            if (
                isinstance(proc_env, dict)
                and proc_env.get("OPENAI_API_KEY") == LOCAL_CLI_OPENAI_KEY_SENTINEL
            ):
                proc_env = dict(proc_env)
                proc_env.pop("OPENAI_API_KEY", None)
                kwargs["env"] = proc_env
            if len(cmd) >= 2 and str(cmd[1]) == "exec":
                mcp_env_flags = []
                for key in ("OPENREWARD_URL", "OPENREWARD_API_URL", "OPENREWARD_SESSION_URL"):
                    if os.environ.get(key):
                        mcp_env_flags.extend([
                            "-c",
                            f"mcp_servers.openreward.env.{key}={json.dumps(os.environ[key])}",
                        ])
                cmd = (
                    *cmd[:-1],
                    "-c",
                    f"mcp_servers.openreward.env.PYTHONPATH={json.dumps(os.environ['PYTHONPATH'])}",
                    *mcp_env_flags,
                    cmd[-1],
                )
            return await original_create_subprocess_exec(*cmd, **kwargs)

        asyncio.create_subprocess_exec = create_subprocess_exec

        if resume_mode:
            from firehorse.cli import main as firehorse_main

            return firehorse_main(target)
        return target()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
