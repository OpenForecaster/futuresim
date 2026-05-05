import json
from datetime import date
from types import SimpleNamespace

import pytest

from agents.minimalHarnessAgent import agent as minimal_agent
from agents.minimalHarnessAgent.prompt import build_warmup_prompt
from agents.search_tools.base import BaseSearchTool, SearchResult


def _capture_codex_cmd(tmp_path, monkeypatch, *, resume: bool):
    internal_dir = tmp_path / ("agent_resume" if resume else "agent_fresh")
    workspace = internal_dir / "workspace"
    workspace.mkdir(parents=True)
    (internal_dir / "system_prompt.md").write_text("system prompt")

    cfg = minimal_agent.MinimalHarnessConfig(
        model="gpt-5.5",
        harness_backend="codex",
        codex_path="codex",
        reasoning_effort="none",
        codex_resume=resume,
        sandbox=False,
    )
    agent = minimal_agent.MinimalHarnessAgent(
        "agent_001",
        cfg,
        agent_dir=str(internal_dir),
    )
    if resume:
        agent._codex_thread_id = "thread_123"
    agent._build_mcp_invocation = lambda: ("python", ["mcp_server.py"])
    agent._maybe_sandbox = lambda cmd: cmd

    captured = {}

    class DummyProc:
        def poll(self):
            return 0

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        return DummyProc()

    monkeypatch.setattr(minimal_agent.subprocess, "Popen", fake_popen)

    agent._start_codex()

    agent._stdout_log.close()
    agent._stderr_log.close()
    return captured["cmd"]


def test_codex_launch_disables_native_web_search(tmp_path, monkeypatch):
    for resume in (False, True):
        cmd = _capture_codex_cmd(tmp_path, monkeypatch, resume=resume)
        assert 'web_search="disabled"' in cmd
        assert "--search" not in cmd


def test_codex_launch_configures_network_proxy_when_isolated(tmp_path, monkeypatch):
    internal_dir = tmp_path / "agent_netiso"
    workspace = internal_dir / "workspace"
    workspace.mkdir(parents=True)
    (internal_dir / "system_prompt.md").write_text("system prompt")

    cfg = minimal_agent.MinimalHarnessConfig(
        model="gpt-5.5",
        harness_backend="codex",
        codex_path="codex",
        reasoning_effort="low",
        sandbox=True,
        network_isolation=True,
    )
    agent = minimal_agent.MinimalHarnessAgent(
        "agent_001",
        cfg,
        agent_dir=str(internal_dir),
    )
    agent._build_mcp_invocation = lambda: ("python", ["mcp_server.py"])
    agent._maybe_sandbox = lambda cmd: cmd

    captured = {}

    class DummyProc:
        def poll(self):
            return 0

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        return DummyProc()

    monkeypatch.setattr(minimal_agent.subprocess, "Popen", fake_popen)

    agent._start_codex()

    agent._stdout_log.close()
    agent._stderr_log.close()
    assert 'network.proxy_url="http://127.0.0.1:9080"' in captured["cmd"]
    assert captured["env"]["ALL_PROXY"] == "http://127.0.0.1:9080"


def test_sandbox_mcp_invocation_uses_bound_bridge_wrapper(tmp_path):
    internal_dir = tmp_path / "agent_sandbox_socat"
    cfg = minimal_agent.MinimalHarnessConfig(
        harness_backend="codex",
        sandbox=True,
    )
    agent = minimal_agent.MinimalHarnessAgent(
        "agent_001",
        cfg,
        agent_dir=str(internal_dir),
    )
    agent._mcp_relay_sock = tmp_path / "mcp.sock"
    agent._mcp_bridge_wrapper = tmp_path / "connect_mcp.sh"

    cmd, args = agent._build_mcp_invocation()

    assert cmd == str(agent._mcp_bridge_wrapper)
    assert ".venv" not in cmd
    assert args == []


def test_system_prompt_uses_configured_cadence_and_outcome_limit(tmp_path):
    for prompt_mode in ("default", "no_memory"):
        internal_dir = tmp_path / f"agent_{prompt_mode}"
        internal_dir.mkdir()
        cfg = minimal_agent.MinimalHarnessConfig(
            start_date=date(2026, 1, 1),
            end_date=date(2026, 1, 31),
            prompt_mode=prompt_mode,
            max_outcomes_per_question=7,
            timegap_days=4,
        )
        agent = minimal_agent.MinimalHarnessAgent(
            "agent_001",
            cfg,
            agent_dir=str(internal_dir),
        )
        agent._current_date = date(2026, 1, 5)

        agent._write_system_prompt()

        prompt = (internal_dir / "system_prompt.md").read_text()
        assert "Maximum of 7 outcomes allowed per question." in prompt
        assert "Submit at most 7 outcomes per question." in prompt
        assert "update your predictions every 4 day(s)" in prompt


def test_codex_warmup_launch_uses_user_prompt_without_agents_md(tmp_path, monkeypatch):
    internal_dir = tmp_path / "agent_warmup"
    workspace = internal_dir / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "AGENTS.md").write_text("stale system prompt")

    cfg = minimal_agent.MinimalHarnessConfig(
        model="gpt-5.5",
        harness_backend="codex",
        codex_path="codex",
        reasoning_effort="low",
        prompt_mode="warmup",
        codex_resume=False,
        sandbox=False,
    )
    agent = minimal_agent.MinimalHarnessAgent(
        "agent_001",
        cfg,
        agent_dir=str(internal_dir),
    )
    agent._build_mcp_invocation = lambda: ("python", ["mcp_server.py"])
    agent._maybe_sandbox = lambda cmd: cmd
    agent._codex_initial_prompt_override = "single question prompt"

    captured = {}

    class DummyProc:
        def poll(self):
            return 0

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        return DummyProc()

    monkeypatch.setattr(minimal_agent.subprocess, "Popen", fake_popen)

    agent._start_codex()

    agent._stdout_log.close()
    agent._stderr_log.close()
    assert captured["cmd"][-1] == "single question prompt"
    assert "-C" in captured["cmd"]
    assert not (workspace / "AGENTS.md").exists()


def test_codex_warmup_rejects_resume_mode(tmp_path):
    cfg = minimal_agent.MinimalHarnessConfig(
        harness_backend="codex",
        prompt_mode="warmup",
        codex_resume=True,
    )
    with pytest.raises(ValueError, match="codex_resume=False"):
        minimal_agent.MinimalHarnessAgent("agent_001", cfg, agent_dir=str(tmp_path / "agent"))


def test_minimal_harness_state_can_carry_separate_search_date(tmp_path):
    internal_dir = tmp_path / "agent_state"
    internal_dir.mkdir()
    q = SimpleNamespace(
        qid="Q1",
        title="Will X happen?",
        background="Background",
        resolution_criteria="Criteria",
        answer_type="string",
        resolution_date=date(2026, 1, 10),
        options=None,
    )

    class DummyForecastInterface:
        resolution_events = []

        def list_questions(self):
            return [q]

        def get_market_csv_path(self):
            return None

        def get_agent_predictions(self, agent_id):
            return {}

    cfg = minimal_agent.MinimalHarnessConfig(
        harness_backend="codex",
        prompt_mode="warmup",
        codex_resume=False,
        max_outcomes_per_question=3,
    )
    agent = minimal_agent.MinimalHarnessAgent("agent_001", cfg, agent_dir=str(internal_dir))
    agent._write_state(
        DummyForecastInterface(),
        date(2026, 3, 28),
        questions_override=[q],
        search_current_date=date(2026, 1, 9),
    )

    state = json.loads((internal_dir / "state.json").read_text())
    assert state["current_date"] == "2026-03-28"
    assert state["search_current_date"] == "2026-01-09"
    assert state["max_outcomes_per_question"] == 3
    assert state["submit_ends_session"] is True
    assert state["next_day_returns_immediately"] is True
    assert [item["qid"] for item in state["questions"]] == ["Q1"]


def test_warmup_mcp_tools_are_search_and_submit_only(tmp_path):
    cfg = minimal_agent.MinimalHarnessConfig(
        harness_backend="codex",
        prompt_mode="warmup",
        codex_resume=False,
        sandbox=False,
    )
    agent = minimal_agent.MinimalHarnessAgent("agent_001", cfg, agent_dir=str(tmp_path / "agent"))

    assert agent._mcp_tool_filter_args() == ["--enabled-tools", "search_news,submit_forecasts"]


def test_codex_warmup_question_uses_isolated_runtime_and_preserves_logs(tmp_path, monkeypatch):
    internal_dir = tmp_path / "agent_warmup_isolated"
    internal_dir.mkdir()
    q = SimpleNamespace(
        qid="Q/1",
        title="Will X happen?",
        background="Background",
        resolution_criteria="Criteria",
        answer_type="binary",
        resolution_date=date(2026, 1, 10),
        options=["Yes", "No"],
    )

    class DummyForecastInterface:
        resolution_events = []
        source_context = ""
        source_name = "openforesight"

        def list_questions(self):
            return [q]

        def get_market_csv_path(self):
            return None

        def get_agent_predictions(self, agent_id):
            return {}

    def fake_start_codex(self):
        current_date = json.loads((self._internal_dir / "state.json").read_text())["current_date"]
        predictions = [{"question_id": "Q/1", "outcomes": {"Yes": 0.7, "No": 0.3}}]
        (self._internal_dir / "predictions").mkdir(parents=True, exist_ok=True)
        (self._internal_dir / "signals").mkdir(parents=True, exist_ok=True)
        (self._internal_dir / "predictions" / f"{current_date}.json").write_text(json.dumps(predictions))
        (self._internal_dir / "signals" / f"next_day_{current_date}.json").write_text(
            json.dumps({"predictions": predictions, "date": current_date})
        )
        (self._internal_dir / "codex_stdout.jsonl").write_text(
            json.dumps({"type": "thread.started", "thread_id": "thread_test"}) + "\n"
        )
        (self._internal_dir / "codex_stderr.log").write_text("stderr\n")

    monkeypatch.setattr(minimal_agent.MinimalHarnessAgent, "_start_codex", fake_start_codex)

    cfg = minimal_agent.MinimalHarnessConfig(
        harness_backend="codex",
        prompt_mode="warmup",
        codex_resume=False,
        resolution_guard=1,
        timeout_seconds=1,
        sandbox=False,
    )
    agent = minimal_agent.MinimalHarnessAgent("agent_001", cfg, agent_dir=str(internal_dir))

    qid, effective_current_date, predictions = agent._run_single_warmup_question(
        q,
        date(2026, 3, 28),
        DummyForecastInterface(),
    )

    safe_qid = agent._safe_qid(q.qid)
    log_dir = internal_dir / "warmup_logs" / safe_qid
    state = json.loads((log_dir / "state.json").read_text())

    assert qid == "Q/1"
    assert effective_current_date == date(2026, 1, 9)
    assert predictions == [{"question_id": "Q/1", "outcomes": {"Yes": 0.7, "No": 0.3}}]
    assert not (internal_dir / "warmup_runtime" / safe_qid).exists()
    assert state["current_date"] == "2026-03-28"
    assert state["search_current_date"] == "2026-01-09"
    assert [item["qid"] for item in state["questions"]] == ["Q/1"]
    assert (log_dir / "predictions.json").exists()
    assert (log_dir / "prompt.md").exists()
    assert (log_dir / "codex_stdout.jsonl").exists()

    written = minimal_agent.aggregate_codex_warmup_logs("agent_001", internal_dir)
    model_outputs = (internal_dir / "model_outputs.jsonl").read_text().splitlines()
    raw_warmup = (internal_dir / "model_raw_warmup.jsonl").read_text().splitlines()
    clean_record = json.loads(model_outputs[0])
    raw_record = json.loads(raw_warmup[0])

    assert written == 1
    assert len(model_outputs) == 1
    assert len(raw_warmup) == 1
    assert clean_record["qid"] == "Q/1"
    assert clean_record["metadata"]["predictions"] == predictions
    assert raw_record["qid"] == "Q/1"
    assert raw_record["response"][0]["type"] == "thread.started"
    assert not log_dir.exists()


def test_codex_warmup_parallel_submits_predictions_and_cleans_runtime(tmp_path, monkeypatch):
    q1 = SimpleNamespace(
        qid="Q1",
        title="Will X happen?",
        background="Background",
        resolution_criteria="Criteria",
        answer_type="binary",
        resolution_date=date(2026, 1, 10),
        options=["Yes", "No"],
    )
    q2 = SimpleNamespace(
        qid="Q2",
        title="Will Y happen?",
        background="Background",
        resolution_criteria="Criteria",
        answer_type="binary",
        resolution_date=date(2026, 1, 11),
        options=["Yes", "No"],
    )

    class DummyForecastInterface:
        def __init__(self):
            self.questions = {q.qid: q for q in (q1, q2)}
            self.submissions = []
            self.current_agent_id = None

        def submit_prediction(self, submission):
            self.submissions.append(submission)

    def fake_run_single(self, q, current_date, forecast_interface):
        effective_date = self._get_warmup_current_date(current_date, q)
        return str(q.qid), effective_date, [
            {"question_id": str(q.qid), "outcomes": {"Yes": 0.6, "No": 0.4}}
        ]

    monkeypatch.setattr(minimal_agent.MinimalHarnessAgent, "_run_single_warmup_question", fake_run_single)

    cfg = minimal_agent.MinimalHarnessConfig(
        harness_backend="codex",
        prompt_mode="warmup",
        codex_resume=False,
        resolution_guard=1,
        warmup_parallelism=2,
        sandbox=False,
    )
    agent = minimal_agent.MinimalHarnessAgent("agent_001", cfg, agent_dir=str(tmp_path / "agent"))
    forecast_interface = DummyForecastInterface()

    agent.warmup(forecast_interface, date(2026, 3, 28))

    submitted_qids = sorted(sub.question_id for sub in forecast_interface.submissions)
    index_records = [
        json.loads(line)
        for line in (tmp_path / "agent" / "warmup_logs" / "index.jsonl").read_text().splitlines()
    ]

    assert agent.warmed_up
    assert forecast_interface.current_agent_id == "agent_001"
    assert submitted_qids == ["Q1", "Q2"]
    assert not (tmp_path / "agent" / "warmup_runtime").exists()
    assert sorted(record["qid"] for record in index_records) == ["Q1", "Q2"]
    assert all(record["num_predictions"] == 1 for record in index_records)
    assert all(record["error"] is None for record in index_records)
    assert all("log_dir" not in record for record in index_records)
    assert all(record["raw_output_log"].endswith("model_raw_warmup.jsonl") for record in index_records)


def test_static_search_prompt_is_submit_only():
    q = SimpleNamespace(
        qid="Q1",
        title="Will X happen?",
        background="Background",
        resolution_criteria="Criteria",
        answer_type="binary",
        resolution_date=date(2026, 1, 10),
        options=["Yes", "No"],
    )

    prompt = build_warmup_prompt(
        current_date=date(2026, 1, 9),
        q=q,
        static_search_text="HEADLINE: Static article\nEvidence text.",
    )

    assert "STATIC RETRIEVED ARTICLES" in prompt
    assert "HEADLINE: Static article" in prompt
    assert "mcp__forecast__submit_forecasts" in prompt
    assert "mcp__forecast__search_news" not in prompt
    assert "mcp__forecast__next_day" not in prompt
    assert "harness will end this isolated question session" in prompt


def test_warmup_prompt_ends_after_submit_without_next_day():
    q = SimpleNamespace(
        qid="Q1",
        title="Will X happen?",
        background="Background",
        resolution_criteria="Criteria",
        answer_type="binary",
        resolution_date=date(2026, 1, 10),
        options=["Yes", "No"],
    )

    prompt = build_warmup_prompt(
        current_date=date(2026, 1, 9),
        q=q,
    )

    assert "mcp__forecast__search_news" in prompt
    assert "mcp__forecast__submit_forecasts" in prompt
    assert "mcp__forecast__next_day" not in prompt
    assert "harness will end this isolated question session" in prompt


def test_static_search_aggregate_writes_raw_to_warmup_log(tmp_path):
    agent_dir = tmp_path / "agent_static_raw"
    log_dir = agent_dir / "warmup_logs" / "Q1"
    log_dir.mkdir(parents=True)
    (log_dir / "state.json").write_text(json.dumps({
        "current_date": "2026-03-28",
        "search_current_date": "2026-01-09",
        "prompt_mode": "static_search",
        "questions": [{"qid": "Q1"}],
    }))
    (log_dir / "prompt.md").write_text("static prompt")
    (log_dir / "predictions.json").write_text(json.dumps([
        {"question_id": "Q1", "outcomes": {"Yes": 0.7, "No": 0.3}},
    ]))
    (log_dir / "codex_stdout.jsonl").write_text(
        json.dumps({"type": "thread.started", "thread_id": "thread_test"}) + "\n"
    )
    (log_dir / "codex_stderr.log").write_text("")

    written = minimal_agent.aggregate_codex_warmup_logs("agent_001", agent_dir)

    assert written == 1
    clean = json.loads((agent_dir / "model_outputs.jsonl").read_text().strip())
    raw_warmup = (agent_dir / "model_raw_warmup.jsonl").read_text().strip().splitlines()
    assert (agent_dir / "model_raw_daily.jsonl").read_text() == ""
    assert clean["metadata"]["prompt_mode"] == "static_search"
    assert clean["metadata"]["raw_stream"] == "warmup"
    assert len(raw_warmup) == 1


def test_codex_static_search_launch_filters_tools_and_disables_codex_tool_features(tmp_path, monkeypatch):
    internal_dir = tmp_path / "agent_static_search"
    workspace = internal_dir / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "AGENTS.md").write_text("stale system prompt")

    cfg = minimal_agent.MinimalHarnessConfig(
        model="gpt-5.5",
        harness_backend="codex",
        codex_path="codex",
        reasoning_effort="low",
        prompt_mode="static_search",
        codex_resume=False,
        sandbox=False,
    )
    agent = minimal_agent.MinimalHarnessAgent(
        "agent_001",
        cfg,
        agent_dir=str(internal_dir),
    )
    agent._maybe_sandbox = lambda cmd: cmd
    agent._codex_initial_prompt_override = "static prompt"

    captured = {}

    class DummyProc:
        def poll(self):
            return 0

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        return DummyProc()

    monkeypatch.setattr(minimal_agent.subprocess, "Popen", fake_popen)

    agent._start_codex()

    agent._stdout_log.close()
    agent._stderr_log.close()
    cmd = captured["cmd"]
    mcp_args_item = next(item for item in cmd if item.startswith("mcp_servers.forecast.args="))
    assert '"--enabled-tools", "submit_forecasts"' in mcp_args_item
    assert "--disable" in cmd
    assert "shell_tool" in cmd
    assert "tool_search" in cmd
    assert cmd[-1] == "static prompt"
    assert not (workspace / "AGENTS.md").exists()


def test_static_search_file_uses_title_query_and_effective_date(tmp_path):
    class DummySearchTool(BaseSearchTool):
        def __init__(self):
            self.calls = []

        @property
        def is_available(self):
            return True

        def search(self, query, max_results=10, max_date=None, search_type="hybrid", min_date=None):
            self.calls.append({
                "query": query,
                "max_results": max_results,
                "max_date": max_date,
                "search_type": search_type,
                "min_date": min_date,
            })
            return [SearchResult(
                article_id="a1",
                title="Hit",
                source="Example",
                date=date(2026, 1, 8),
                date_publish=date(2026, 1, 8),
                snippet="Relevant snippet",
                score=1.0,
                url="https://example.com/a1",
            )]

        def get_article(self, article_id):
            return None

    from agents.minimalHarnessAgent.static_search import ensure_static_search_file

    q = SimpleNamespace(
        qid="Q/1",
        title="Will X happen?",
        resolution_date=date(2026, 1, 10),
    )
    tool = DummySearchTool()
    path, text = ensure_static_search_file(
        output_dir=tmp_path,
        search_tool=tool,
        q=q,
        search_date=date(2026, 1, 9),
        max_results=5,
    )

    assert path.name == "Q_1.md"
    assert "Question Title: Will X happen?" in text
    assert "Resolution Date: 2026-01-10" in text
    assert "Search Date: 2026-01-09" in text
    assert "HEADLINE: Hit" in text
    assert tool.calls == [{
        "query": "Will X happen?",
        "max_results": 5,
        "max_date": date(2026, 1, 9),
        "search_type": "hybrid",
        "min_date": None,
    }]


def test_submit_forecasts_can_end_static_search_session(tmp_path, monkeypatch):
    from agents.minimalHarnessAgent import mcp_server

    monkeypatch.setattr(mcp_server, "_internal_dir", tmp_path)
    monkeypatch.setattr(mcp_server, "_today_predictions", [])
    monkeypatch.setattr(mcp_server, "_state", {
        "current_date": "2026-03-28",
        "max_outcomes_per_question": 5,
        "submit_ends_session": True,
    })

    result = mcp_server.submit_forecasts("Q1", {"Yes": 0.7, "No": 0.3})
    signal = json.loads((tmp_path / "signals" / "next_day_2026-03-28.json").read_text())
    preds = json.loads((tmp_path / "predictions" / "2026-03-28.json").read_text())

    assert "Session complete" in result
    assert signal["reason"] == "submit_ends_session"
    assert signal["predictions"] == preds
    assert preds == [{"question_id": "Q1", "outcomes": {"Yes": 0.7, "No": 0.3}}]


def test_codex_next_day_returns_immediately_without_continue_signal(tmp_path, monkeypatch):
    from agents.minimalHarnessAgent import mcp_server

    monkeypatch.setattr(mcp_server, "_internal_dir", tmp_path)
    monkeypatch.setattr(mcp_server, "_today_predictions", [
        {"question_id": "Q1", "outcomes": {"Yes": 0.7, "No": 0.3}},
    ])
    monkeypatch.setattr(mcp_server, "_state", {
        "current_date": "2026-03-28",
        "next_day_returns_immediately": True,
    })

    result = mcp_server.next_day()
    signal = json.loads((tmp_path / "signals" / "next_day_2026-03-28.json").read_text())

    assert "Day complete" in result
    assert signal == {
        "predictions": [{"question_id": "Q1", "outcomes": {"Yes": 0.7, "No": 0.3}}],
        "date": "2026-03-28",
    }


def test_codex_wait_for_next_day_drains_process_before_reading_predictions(tmp_path):
    internal_dir = tmp_path / "agent"
    (internal_dir / "signals").mkdir(parents=True)
    (internal_dir / "predictions").mkdir()
    (internal_dir / "signals" / "next_day_2026-03-28.json").write_text("{}")
    (internal_dir / "predictions" / "2026-03-28.json").write_text(json.dumps([
        {"qid": "Q1", "outcomes": {"Yes": 0.7, "No": 0.3}},
    ]))

    cfg = minimal_agent.MinimalHarnessConfig(
        harness_backend="codex",
        timeout_seconds=60,
    )
    agent = minimal_agent.MinimalHarnessAgent("agent_001", cfg, agent_dir=str(internal_dir))

    class DummyProc:
        def __init__(self):
            self.wait_timeout = None
            self.waited = False

        def poll(self):
            return None if not self.waited else 0

        def wait(self, timeout=None):
            self.wait_timeout = timeout
            self.waited = True
            return 0

    proc = DummyProc()
    agent._harness_proc = proc

    preds = agent._wait_for_next_day(date(2026, 3, 28))

    assert proc.waited is True
    assert proc.wait_timeout == 30.0
    assert preds == [{"qid": "Q1", "outcomes": {"Yes": 0.7, "No": 0.3}, "question_id": "Q1"}]
