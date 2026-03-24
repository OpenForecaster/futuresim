"""Shared OpenRouter matcher cache path for SkyRL (cross-run JSON on fast storage)."""

from __future__ import annotations

import os
from pathlib import Path

# Override on non-MPI machines via ``FSIM_MATCHER_CACHE_JSON``.
DEFAULT_MATCHER_CACHE_JSON = os.environ.get(
    "FSIM_MATCHER_CACHE_JSON",
    "/fast/sgoel/forecasting/skyrl/matcher_cache.json",
)


def resolve_matcher_cache_path(explicit: str | None) -> Path:
    """Return filesystem path; ``explicit`` wins when non-empty after strip."""
    raw = (explicit or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(DEFAULT_MATCHER_CACHE_JSON).expanduser().resolve()
