from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
UNRESOLVED_ENV_VAR_RE = re.compile(r"\$(?:\{([A-Z_][A-Z0-9_]*)\}|([A-Z_][A-Z0-9_]*))")


def _has_unresolved_env_var(value: str) -> bool:
    return bool(UNRESOLVED_ENV_VAR_RE.search(value))


def load_repo_env(repo_root: Path = REPO_ROOT) -> Path:
    repo_root = repo_root.resolve()
    os.environ.setdefault("FSIM_REPO_DIR", str(repo_root))
    env_path = repo_root / ".env"
    if not env_path.exists():
        return env_path

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        parsed = os.path.expanduser(os.path.expandvars(value.strip().strip('"').strip("'")))
        current = os.environ.get(key)
        if current is None or current == "" or _has_unresolved_env_var(current):
            os.environ[key] = parsed
    return env_path


def expand_env_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: expand_env_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env_tree(item) for item in value]
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    return value


def find_unresolved_env_vars(value: Any, path: str = "") -> list[str]:
    if isinstance(value, dict):
        matches: list[str] = []
        for key, item in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            matches.extend(find_unresolved_env_vars(item, child_path))
        return matches
    if isinstance(value, list):
        matches: list[str] = []
        for idx, item in enumerate(value):
            child_path = f"{path}[{idx}]" if path else f"[{idx}]"
            matches.extend(find_unresolved_env_vars(item, child_path))
        return matches
    if isinstance(value, str) and _has_unresolved_env_var(value):
        return [f"{path or '<root>'}: {value}"]
    return []


def raise_for_unresolved_env_vars(value: Any, context: str) -> None:
    unresolved = find_unresolved_env_vars(value)
    if not unresolved:
        return
    details = "\n".join(f"  - {item}" for item in unresolved)
    raise ValueError(
        f"Unresolved environment placeholders remain in {context}. "
        "Load the repo .env or pass concrete paths before running.\n"
        f"{details}"
    )
