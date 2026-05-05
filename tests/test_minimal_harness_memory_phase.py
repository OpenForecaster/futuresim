import json
import sys
import types
from datetime import date


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, default=str))


def _install_fake_mcp():
    class DummyFastMCP:
        def __init__(self, *args, **kwargs):
            pass

        def tool(self):
            def decorator(fn):
                return fn
            return decorator

        def run(self):
            return None

    mcp_mod = types.ModuleType("mcp")
    server_mod = types.ModuleType("mcp.server")
    fastmcp_mod = types.ModuleType("mcp.server.fastmcp")
    fastmcp_mod.FastMCP = DummyFastMCP
    sys.modules.setdefault("mcp", mcp_mod)
    sys.modules.setdefault("mcp.server", server_mod)
    sys.modules["mcp.server.fastmcp"] = fastmcp_mod


def test_active_memory2_next_day_enters_memory_phase_then_advances(tmp_path):
    _install_fake_mcp()
    import agents.minimalHarnessAgent.mcp_server as srv
    from agents.utils.memory import ActiveMemory

    workspace = tmp_path / "workspace"
    internal = tmp_path / "internal"
    workspace.mkdir()
    internal.mkdir()
    (internal / "signals").mkdir()

    state_day1 = {
        "current_date": "2025-05-01",
        "start_date": "2025-05-01",
        "end_date": "2025-05-03",
        "timeout_seconds": 2,
        "search_cutoff_days": 0,
        "questions": [{"qid": "Q1", "resolution_date": "2025-05-03"}],
        "resolution_events": [{"qid": "Q9", "title": "Resolved title", "ground_truth": "Truth"}],
        "agent_predictions": {},
        "total_predictions": 1,
    }
    _write_json(internal / "state.json", state_day1)

    srv._workspace = workspace
    srv._internal_dir = internal
    srv._state = state_day1
    srv._today_predictions = [{"question_id": "Q1", "outcomes": {"Yes": 0.6}}]
    srv._memory_tools_enabled = True
    srv._memory_phase_active = False
    srv._agent_id = "agent_001"
    srv._session_touched_qids = {"Q1"}
    srv._session_forecast_qids = {"Q1"}
    srv._active_memory = ActiveMemory("agent_001", memory_dir=str(workspace), max_entries=500)
    srv._active_memory.set_date(date(2025, 5, 1))
    srv._active_memory.mem_add("Q1", "Will X happen?", "Current reasoning", "test")
    srv._active_memory.add_entry(
        name="meta-test-pattern",
        description="Test pattern",
        content="Reusable lesson for future days.",
    )

    first = srv.next_day()

    assert "## MEMORY UPDATE" in first
    assert "Resolved title" in first
    assert "meta-test-pattern" in first
    assert "## YOUR CUMULATIVE PERFORMANCE TILL TODAY" in first
    assert "Questions you interacted with today: ['Q1']" in first
    assert not (internal / "signals" / "next_day_2025-05-01.json").exists()
    assert srv._memory_phase_active is True

    blocked = srv.submit_forecasts("Q1", {"Yes": 0.7})
    assert "memory-update phase" in blocked

    state_day2 = dict(state_day1)
    state_day2["current_date"] = "2025-05-02"
    state_day2["resolution_events"] = []
    _write_json(internal / "state.json", state_day2)
    _write_json(
        internal / "signals" / "continue_2025-05-01.json",
        {"status": "day_advanced", "date": "2025-05-01"},
    )

    second = srv.next_day()

    assert "Day advanced to 2025-05-02." in second
    assert (internal / "signals" / "next_day_2025-05-01.json").exists()
    assert (workspace / "memory" / "2025-05-01" / "mem.csv").exists()
    assert (workspace / "memory" / "2025-05-01" / "meta.yaml").exists()
    assert srv._memory_phase_active is False
