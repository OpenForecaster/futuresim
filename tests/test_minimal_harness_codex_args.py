from datetime import date

from agents.minimalHarnessAgent import agent as minimal_agent


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
