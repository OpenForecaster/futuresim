from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping


def matcher_slug(matcher: str) -> str:
    raw = (matcher or "").strip()
    if not raw:
        return "default"
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw.replace("/", "_"))
    return slug[:200] if slug else "default"


def default_sim_matcher_cache_json(matcher: str, cache_dir: str | Path) -> Path:
    base = Path(str(cache_dir)).expanduser()
    return base.resolve() / f"{matcher_slug(matcher)}.json"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def resolve_sim_matcher_cache_path(
    *,
    output_dir: str | Path,
    matching: str,
    matcher: str,
    split: str,
    matcher_cache: Mapping[str, Any] | None = None,
) -> Path:
    """
    Resolve the matcher cache JSON path for a simulation run.

    Rules:
    - Default remains per-run ``<output_dir>/matcher_cache.json``.
    - If top-level YAML ``matcher_cache.enabled: false`` is set, force per-run cache.
    - If top-level YAML ``matcher_cache.path`` is set, use that exact JSON path.
    - Otherwise, when ``FSIM_SIM_MATCHER_CACHE_DIR`` is set:
      - auto-use it for ``split: test`` runs
      - or use it for any split when ``matcher_cache`` is present/enabled
    """
    local_path = (Path(output_dir).expanduser().resolve() / "matcher_cache.json")
    if str(matching or "").strip().lower() == "exact":
        return local_path

    if matcher_cache is not None and not isinstance(matcher_cache, Mapping):
        raise ValueError("Top-level `matcher_cache` must be a mapping when provided.")

    if matcher_cache is not None and not _as_bool(matcher_cache.get("enabled", True)):
        return local_path

    raw_path = ""
    if matcher_cache is not None:
        raw_path = str(matcher_cache.get("path", "") or "").strip()
    if raw_path:
        resolved = Path(os.path.expanduser(os.path.expandvars(raw_path))).resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved

    shared_root = (os.environ.get("FSIM_SIM_MATCHER_CACHE_DIR") or "").strip()
    explicit_opt_in = matcher_cache is not None
    auto_eval_opt_in = str(split or "").strip().lower() == "test"
    if not shared_root or not (explicit_opt_in or auto_eval_opt_in):
        return local_path

    resolved = default_sim_matcher_cache_json(matcher, shared_root)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
