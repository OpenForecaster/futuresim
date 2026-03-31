from __future__ import annotations

from pathlib import Path

from environment.matcher_cache import (
    default_sim_matcher_cache_json,
    matcher_slug,
    resolve_sim_matcher_cache_path,
)


def test_matcher_slug_replaces_unsafe_chars() -> None:
    assert matcher_slug("deepseek/deepseek-v3.2") == "deepseek_deepseek-v3.2"


def test_resolve_sim_matcher_cache_path_uses_shared_dir_for_test_split(
    monkeypatch,
    tmp_path: Path,
) -> None:
    shared_root = tmp_path / "shared"
    monkeypatch.setenv("FSIM_SIM_MATCHER_CACHE_DIR", str(shared_root))

    resolved = resolve_sim_matcher_cache_path(
        output_dir=tmp_path / "run",
        matching="openrouter",
        matcher="deepseek/deepseek-v3.2",
        split="test",
    )

    assert resolved == default_sim_matcher_cache_json("deepseek/deepseek-v3.2", shared_root)


def test_resolve_sim_matcher_cache_path_keeps_local_cache_for_non_test_without_opt_in(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FSIM_SIM_MATCHER_CACHE_DIR", str(tmp_path / "shared"))

    resolved = resolve_sim_matcher_cache_path(
        output_dir=tmp_path / "run",
        matching="openrouter",
        matcher="deepseek/deepseek-v3.2",
        split="train",
    )

    assert resolved == (tmp_path / "run" / "matcher_cache.json").resolve()


def test_resolve_sim_matcher_cache_path_honors_explicit_disable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FSIM_SIM_MATCHER_CACHE_DIR", str(tmp_path / "shared"))

    resolved = resolve_sim_matcher_cache_path(
        output_dir=tmp_path / "run",
        matching="openrouter",
        matcher="deepseek/deepseek-v3.2",
        split="test",
        matcher_cache={"enabled": False},
    )

    assert resolved == (tmp_path / "run" / "matcher_cache.json").resolve()


def test_resolve_sim_matcher_cache_path_honors_explicit_path(tmp_path: Path) -> None:
    explicit = tmp_path / "custom" / "matcher.json"

    resolved = resolve_sim_matcher_cache_path(
        output_dir=tmp_path / "run",
        matching="openrouter",
        matcher="deepseek/deepseek-v3.2",
        split="train",
        matcher_cache={"enabled": True, "path": str(explicit)},
    )

    assert resolved == explicit.resolve()
