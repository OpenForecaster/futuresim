"""Run Futuresim on OpenReward/Firehorse with domain-scoped secrets."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

OPENROUTER_DOMAINS = ["openrouter.ai", "api.openrouter.ai"]


def main(argv: list[str] | None = None) -> int:
    load_dotenv()

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
    args = parser.parse_args(argv)

    if not os.environ.get("OPENREWARD_API_KEY"):
        raise SystemExit("OPENREWARD_API_KEY is required.")

    secrets: dict[str, Any] = {}
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        secrets["OPENROUTER_API_KEY"] = (openrouter_key, OPENROUTER_DOMAINS)

    with tempfile.TemporaryDirectory(prefix="futuresim-openreward-") as shim_dir:
        shim = Path(shim_dir) / "sitecustomize.py"
        shim_code = (
            """
try:
    from openreward.api.sandboxes import secrets as _secrets
    from openreward.api.sandboxes import client as _client

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


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
