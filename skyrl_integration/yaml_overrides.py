"""Apply ``key.path=value`` overrides to nested YAML-loaded dicts (SkyRL launchers)."""

from __future__ import annotations

import re
from typing import Any

import yaml


def parse_set_kv(s: str) -> tuple[str, Any]:
    if "=" not in s:
        raise ValueError(f"Invalid --set {s!r}. Expected key=value")
    key, raw = s.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Invalid --set {s!r}. Empty key")
    return key, yaml.safe_load(raw)


def _tokenize_path(path: str) -> list[Any]:
    tokens: list[Any] = []
    for part in path.split("."):
        part = part.strip()
        if not part:
            raise ValueError(f"Invalid --set path {path!r}: empty segment")
        for m in re.finditer(r"([^\[\]]+)|\[(\d+)\]", part):
            key, idx = m.group(1), m.group(2)
            if key is not None:
                tokens.append(key.strip())
            else:
                tokens.append(int(idx))
    return tokens


def _ensure_list_len(xs: list, n: int) -> None:
    while len(xs) <= n:
        xs.append(None)


def set_in_config(cfg: Any, path: str, value: Any) -> None:
    tokens = _tokenize_path(path)
    cur = cfg
    for i, tok in enumerate(tokens):
        is_last = i == len(tokens) - 1
        nxt = None if is_last else tokens[i + 1]

        if is_last:
            if isinstance(tok, int):
                if not isinstance(cur, list):
                    raise ValueError(f"Path {path!r} expects list at final container")
                _ensure_list_len(cur, tok)
                cur[tok] = value
                return
            if not isinstance(cur, dict):
                raise ValueError(f"Path {path!r} expects dict at final container")
            cur[tok] = value
            return

        if isinstance(tok, int):
            if not isinstance(cur, list):
                raise ValueError(f"Path {path!r} expects list container")
            _ensure_list_len(cur, tok)
            if cur[tok] is None:
                cur[tok] = [] if isinstance(nxt, int) else {}
            cur = cur[tok]
            continue

        if not isinstance(cur, dict):
            raise ValueError(f"Path {path!r} expects dict container")
        if tok not in cur or cur[tok] is None:
            cur[tok] = [] if isinstance(nxt, int) else {}
        cur = cur[tok]


def apply_set_overrides(cfg: dict[str, Any], set_values: list[str]) -> None:
    for item in set_values:
        key, value = parse_set_kv(item)
        set_in_config(cfg, key, value)
