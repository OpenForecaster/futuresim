import os
from pathlib import Path

import pytest

from pathing import load_repo_env, raise_for_unresolved_env_vars


def test_load_repo_env_overwrites_placeholder_env_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".env").write_text(
        "\n".join(
            [
                "FSIM_OUTPUT_BASE=/tmp/current_sim",
                "FSIM_SIM_LOG_BASE=/tmp/sim_logs",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("FSIM_OUTPUT_BASE", "${FSIM_OUTPUT_BASE}")
    monkeypatch.setenv("FSIM_SIM_LOG_BASE", "${FSIM_SIM_LOG_BASE}")

    load_repo_env(repo_root)

    assert os.environ["FSIM_OUTPUT_BASE"] == "/tmp/current_sim"
    assert os.environ["FSIM_SIM_LOG_BASE"] == "/tmp/sim_logs"


def test_raise_for_unresolved_env_vars_reports_paths() -> None:
    config = {
        "output_base": "${FSIM_OUTPUT_BASE}",
        "nested": [{"search_db": "${FSIM_SEARCH_DB}"}],
    }

    with pytest.raises(ValueError) as excinfo:
        raise_for_unresolved_env_vars(config, "test config")

    message = str(excinfo.value)
    assert "test config" in message
    assert "output_base: ${FSIM_OUTPUT_BASE}" in message
    assert "nested[0].search_db: ${FSIM_SEARCH_DB}" in message
