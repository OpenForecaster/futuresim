"""Run Futuresim on OpenReward/Firehorse with domain-scoped secrets."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
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
        summary = asyncio.run(run_evaluation(config))
        return 0 if summary.failed == 0 else 1

    return _run_with_firehorse_shim(run_firehorse)


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
