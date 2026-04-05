from __future__ import annotations

import json
from pathlib import Path

from skyrl_integration.train.iteration_logging import (
    RunArtifactLogger,
    parse_eval_step_from_dump_dir,
    resolve_run_root_from_log_path,
)


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class DummyTokenizer:
    def batch_decode(self, sequences, skip_special_tokens=False):
        assert skip_special_tokens is False
        return ["|".join(str(token) for token in sequence) for sequence in sequences]


def test_resolve_run_root_from_log_path_handles_infra_dir_and_file(tmp_path):
    run_root = tmp_path / "sim" / "run_01"
    infra_dir = run_root / "infra"
    infra_dir.mkdir(parents=True)
    infra_file = infra_dir / "infra-260401_120000.log"
    infra_file.write_text("", encoding="utf-8")

    assert resolve_run_root_from_log_path(infra_dir) == run_root
    assert resolve_run_root_from_log_path(infra_file) == run_root


def test_run_artifact_logger_merges_metrics_by_step(tmp_path):
    artifact_logger = RunArtifactLogger(tmp_path / "run")

    artifact_logger.log_metrics(3, {"reward/avg_raw_reward": 1.5})
    artifact_logger.log_metrics(3, {"timing/generate": 2.0})
    artifact_logger.log_metrics(4, {"reward/avg_raw_reward": 2.5})
    artifact_logger.finish()

    rows = _read_jsonl(tmp_path / "run" / "metrics.jsonl")
    assert len(rows) == 2
    assert rows[0]["step"] == 3
    assert rows[0]["reward/avg_raw_reward"] == 1.5
    assert rows[0]["timing/generate"] == 2.0
    assert rows[1]["step"] == 4
    assert rows[1]["reward/avg_raw_reward"] == 2.5


def test_write_train_rollout_summaries_uses_iteration_subdir(tmp_path):
    artifact_logger = RunArtifactLogger(tmp_path / "run")
    tokenizer = DummyTokenizer()
    generator_output = {
        "prompt_token_ids": [[1, 2], [3]],
        "response_ids": [[9, 8, 7], [6]],
        "rewards": [[0.0, 1.0, 0.0], 0.5],
        "stop_reasons": ["stop", "length"],
        "fsim_trajectory_ids": [None, None],
    }

    artifact_logger.append_train_rollout_summaries(
        tokenizer=tokenizer,
        step=12,
        generator_output=generator_output,
        uids=["uid_a", "uid_b"],
        env_extras=[{"question_id": "q1"}, {"question_id": "q2"}],
        env_classes=["env_a", "env_b"],
    )

    rows = _read_jsonl(tmp_path / "run" / "iteration_000012" / "train_rollout_summaries.jsonl")
    assert [row["uid"] for row in rows] == ["uid_a", "uid_b"]
    assert rows[0]["env_metadata"]["question_id"] == "q1"
    assert rows[0]["reward"] == 1.0
    assert rows[0]["input_prompt"] == "1|2"
    assert rows[0]["prompt_num_tokens"] == 2
    assert rows[0]["trajectory_text"] == "9|8|7"
    assert rows[0]["response_num_tokens"] == 3


def test_write_eval_rollout_summaries_and_parse_eval_step(tmp_path):
    artifact_logger = RunArtifactLogger(tmp_path / "run")
    tokenizer = DummyTokenizer()

    step = parse_eval_step_from_dump_dir("global_step_42_evals")
    assert step == 42
    assert parse_eval_step_from_dump_dir("eval_only") is None

    artifact_logger.write_eval_rollout_summaries(
        step=step,
        tokenizer=tokenizer,
        concat_generator_outputs={
            "prompt_token_ids": [[1]],
            "response_ids": [[2, 3]],
            "rewards": [0.75],
            "stop_reasons": ["stop"],
        },
        concat_data_sources=["test"],
        concat_all_envs=["openforesight"],
        concat_env_extras=[{"question_id": "q_eval"}],
        eval_metrics={"eval/all/avg_score": 0.75},
    )

    rows = _read_jsonl(tmp_path / "run" / "iteration_000042" / "eval_rollout_summaries.jsonl")
    assert rows[0]["data_source"] == "test"
    assert rows[0]["score"] == 0.75
    assert rows[0]["env_metadata"]["question_id"] == "q_eval"
    assert rows[0]["input_prompt"] == "1"
    assert rows[0]["trajectory_text"] == "2|3"
    metrics = json.loads((tmp_path / "run" / "iteration_000042" / "eval_metrics.json").read_text(encoding="utf-8"))
    assert metrics["eval/all/avg_score"] == 0.75
