import json
from datetime import date

from agents.utils.output_logger import AgentOutputLogger


def test_agent_output_logger_splits_daily_and_warmup_raw_logs_and_sorts_warmup(tmp_path):
    agent_dir = tmp_path / "agents" / "agent_a"
    logger = AgentOutputLogger("agent_a", str(agent_dir))

    logger.log_model_output(
        date(2025, 1, 1),
        [{"role": "user", "content": "daily prompt"}],
        "daily response",
        {"phase": "llm", "raw_stream": "daily"},
    )
    logger.log_model_output(
        date(2025, 1, 1),
        [{"role": "user", "content": "warmup q2"}],
        "resp q2",
        {"phase": "llm", "raw_stream": "warmup", "qid": "q2"},
    )
    logger.log_model_output(
        date(2025, 1, 1),
        [{"role": "user", "content": "warmup q1 first"}],
        "resp q1 first",
        {"phase": "llm", "raw_stream": "warmup", "qid": "q1"},
    )
    logger.log_model_output(
        date(2025, 1, 1),
        [{"role": "tool", "output": "warmup q1 second"}],
        "resp q1 second",
        {"phase": "submit", "raw_stream": "warmup", "qid": "q1"},
    )
    logger.flush_warmup_raw()
    logger.close()

    daily_rows = [json.loads(line) for line in (agent_dir / "model_raw_daily.jsonl").read_text(encoding="utf-8").splitlines() if line]
    warmup_rows = [json.loads(line) for line in (agent_dir / "model_raw_warmup.jsonl").read_text(encoding="utf-8").splitlines() if line]

    assert len(daily_rows) == 1
    assert daily_rows[0]["prompt"] == "daily prompt"
    assert daily_rows[0]["input_delta"] == [{"role": "user", "content": "daily prompt"}]

    assert [row["qid"] for row in warmup_rows] == ["q1", "q1", "q2"]
    assert warmup_rows[0]["prompt"] == "warmup q1 first"
    assert warmup_rows[1]["prompt"] == "warmup q1 second"
    assert warmup_rows[2]["prompt"] == "warmup q2"
