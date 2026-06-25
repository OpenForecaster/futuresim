"""Agent-facing scaffolding for the OpenReward Futuresim integration."""

from __future__ import annotations

import importlib
import os
from typing import Any, Callable

from pydantic import BaseModel

from futuresim_agents.minimalHarnessAgent.prompts.prompt import build_system_prompt
from integrations.adapter_runtime import (
    FuturesimAdapterConfig,
    FuturesimAdapterRuntime,
    parse_iso_date,
)

PromptBuilder = Callable[..., str]


def build_default_prompt(
    runtime: FuturesimAdapterRuntime,
    config: FuturesimAdapterConfig,
    *,
    mount_articles: bool,
) -> str:
    """Build the default OpenReward agent prompt.

    Custom prompt builders can use the same signature and be selected with
    task_spec["openreward_agent"]["prompt_builder"] = "module:function".
    """
    forecast_interface = runtime.forecast_interface()
    questions = forecast_interface.list_questions()
    resolved = getattr(forecast_interface, "resolved_questions", [])
    return build_system_prompt(
        workspace=runtime.workspace_path,
        current_date=runtime.env.current_date,
        start_date=runtime.env.start_date,
        end_date=parse_iso_date(config.end_date) or runtime.env.current_date,
        source_context=getattr(forecast_interface, "source_context", "") or "",
        source_name=getattr(forecast_interface, "source_name", "openforesight"),
        num_questions=len(questions) + len(resolved),
        num_active=len(questions),
        num_resolved=len(resolved),
        max_outcomes_per_question=config.max_outcomes_per_question,
        search_cutoff_days=config.article_search_cutoff_days,
        timegap_days=config.timegap_days,
        new_articles_count=None,
        last_active_date=getattr(forecast_interface, "last_active_date", None),
        next_active_date=getattr(forecast_interface, "next_active_date", None),
        handholding_version=config.handholding_version,
        prompt_mode=config.prompt_mode,
        article_files_available=mount_articles,
        tool_prefix="mcp__openreward__",
    )


def resolve_prompt_builder(task_spec: dict[str, Any]) -> PromptBuilder:
    agent_spec = task_spec.get("openreward_agent")
    dotted = ""
    if isinstance(agent_spec, dict):
        dotted = str(agent_spec.get("prompt_builder") or "").strip()
    dotted = dotted or os.environ.get("FSIM_OPENREWARD_PROMPT_BUILDER", "").strip()
    if not dotted:
        return build_default_prompt

    module_name, sep, attr = dotted.partition(":")
    if not sep:
        module_name, _, attr = dotted.rpartition(".")
    if not module_name or not attr:
        raise ValueError(
            "OpenReward prompt builder must be 'module:function' or 'module.function'."
        )
    obj = getattr(importlib.import_module(module_name), attr)
    if hasattr(obj, "build_prompt"):
        return obj.build_prompt
    return obj


def patch_openreward_cli_toolsets() -> None:
    """Patch OpenReward CLI toolsets at import time for known CLI quirks."""
    try:
        from openreward.toolsets import codex as codex_toolset
    except Exception:
        return

    class _CodexBashParams(BaseModel, extra="ignore"):
        command: str
        description: str = ""
        timeout: float | None = 30.0
        workdir: str | None = None

    # openreward==0.1.134 documents "Always set workdir" but rejects that
    # field. Codex also emits local shell metadata like max_output_tokens. Those
    # values are local-Codex concerns rather than guaranteed sandbox controls,
    # so the safest compatibility behavior is to accept and ignore extras.
    codex_toolset.BashParams = _CodexBashParams
