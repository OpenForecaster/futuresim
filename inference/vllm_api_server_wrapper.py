#!/usr/bin/env python
"""
Launch vLLM OpenAI API server with optional FSIM compatibility patches.

This wrapper avoids direct edits in site-packages by patching vLLM's Harmony
parser wiring at process startup when requested.
"""

from __future__ import annotations

import importlib
import inspect
import os
import runpy


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _patch_harmony_module(module_name: str) -> bool:
    """
    Patch vLLM Harmony parser construction to use non-strict mode.

    Returns True if patching was applied for the target module.
    """
    try:
        mod = importlib.import_module(module_name)
    except Exception:
        return False

    get_encoding = getattr(mod, "get_encoding", None)
    if not callable(get_encoding):
        return False

    try:
        from openai_harmony import Role, StreamableParser
    except Exception:
        return False

    try:
        sig = inspect.signature(StreamableParser)
    except Exception:
        return False
    if "strict" not in sig.parameters:
        return False

    def _fsim_non_strict_parser() -> StreamableParser:
        return StreamableParser(get_encoding(), role=Role.ASSISTANT, strict=False)

    setattr(mod, "get_streamable_parser_for_assistant", _fsim_non_strict_parser)
    return True


def _apply_compat_patches() -> None:
    # Keep off unless explicitly enabled by environment.
    if not _as_bool(os.getenv("FSIM_VLLM_HARMONY_NON_STRICT"), default=False):
        return

    patched = False
    for mod_name in (
        # vLLM modern path
        "vllm.entrypoints.openai.parser.harmony_utils",
        # vLLM older path
        "vllm.entrypoints.harmony_utils",
    ):
        patched = _patch_harmony_module(mod_name) or patched

    if patched:
        print(
            "[FSIM] Enabled non-strict Harmony parser patch (FSIM_VLLM_HARMONY_NON_STRICT=1).",
            flush=True,
        )
    else:
        print(
            "[FSIM] Harmony non-strict patch requested but no compatible vLLM/harmony module found.",
            flush=True,
        )


def main() -> None:
    _apply_compat_patches()
    # Execute the real API server module as __main__, preserving argv.
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
