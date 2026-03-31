"""
SkyRL launch helpers: matcher cache paths and core-dump cwd (stdlib only; safe for light importers).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from pathing import REPO_ROOT, load_repo_env


load_repo_env()


def _default_matcher_cache_dir() -> Path:
    skyrl_log_base = os.environ.get("FSIM_SKYRL_LOG_BASE")
    if skyrl_log_base:
        return Path(os.path.expanduser(os.path.expandvars(skyrl_log_base))).resolve() / "matcher_cache"
    return REPO_ROOT / "logs" / "skyrl" / "matcher_cache"


def _default_core_dump_dir() -> Path:
    return REPO_ROOT / "logs" / "core-dumps"


def _matcher_slug(matcher: str) -> str:
    raw = (matcher or "").strip()
    if not raw:
        return "default"
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw.replace("/", "_"))
    return slug[:200] if slug else "default"


def matcher_cache_dir() -> Path:
    raw = (os.environ.get("FSIM_SKYRL_MATCHER_CACHE_DIR") or str(_default_matcher_cache_dir())).strip()
    return Path(os.path.expanduser(os.path.expandvars(raw))).resolve()


def default_matcher_cache_json(matcher: str) -> Path:
    """When YAML ``matcher_cache.path`` is null: ``<matcher_cache_dir>/<matcher_slug>.json``."""
    return matcher_cache_dir() / f"{_matcher_slug(matcher)}.json"


def core_dump_dir() -> Path:
    raw = (os.environ.get("FSIM_CORE_DUMP_DIR") or str(_default_core_dump_dir())).strip()
    return Path(os.path.expanduser(os.path.expandvars(raw))).resolve()


def setup_core_dump_cwd() -> Path:
    """Create directory, ``chdir`` there, and normalize ``FSIM_CORE_DUMP_DIR`` in the environment."""
    path = core_dump_dir()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    os.environ["FSIM_CORE_DUMP_DIR"] = str(path)
    return path


def ray_worker_core_dump_setup_hook() -> None:
    """Ray ``runtime_env`` ``worker_process_setup_hook`` (FSDP / vLLM workers)."""
    setup_core_dump_cwd()
