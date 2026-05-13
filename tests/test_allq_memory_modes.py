from datetime import date, timedelta
from pathlib import Path

import yaml

from agents.allQAgent import AllQAgent
from agents.basicAgent import AgentConfig
from agents.basicAgent.tools import build_action_tools


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs" / "memory"


class ForecastStub:
    source_name = "openforesight"
    source_context = ""
    num_agents = 1
    histories = {}
    questions = {"Q1": object()}
    last_active_date = date(2025, 1, 1)
    next_active_date = date(2025, 1, 11)

    def get_agent_predictions(self, agent_id):
        return {}


def _allq_agent(config: AgentConfig) -> AllQAgent:
    agent = AllQAgent(
        agent_id="allQ_deepseek-v3.2_001",
        inference_provider=object(),
        config=config,
        model_name="deepseek/deepseek-v3.2",
        start_date=date(2025, 1, 1),
    )
    agent._forecast_interface = ForecastStub()
    return agent


def _load_config(name: str) -> dict:
    with open(CONFIG_DIR / name, "r") as f:
        return yaml.safe_load(f)


def test_allq_no_memory_prompt_has_no_memory_tools_or_mem_df():
    agent = _allq_agent(AgentConfig(enable_memory=False, timegap_days=5))

    prompt = agent._build_instructions(date(2025, 1, 6))

    assert "every 5 days" in prompt
    assert "past predictions are the only information retained" in prompt
    assert "your memory (along with past predictions)" not in prompt
    assert "## YOUR MEMORY" not in prompt
    assert "memory operations" not in prompt
    assert "mem_df" not in prompt
    assert "mem_add" not in prompt
    assert "memory_retrieve" not in prompt


def test_allq_active_memory_prompt_keeps_mem_df_and_memory_tools(tmp_path):
    cfg = AgentConfig(
        enable_memory=True,
        memory_format="active",
        memory_dir=str(tmp_path),
        timegap_days=5,
    )
    agent = _allq_agent(cfg)
    agent._memory.set_date(date(2025, 1, 6))

    prompt = agent._build_instructions(date(2025, 1, 6))

    assert "your memory (along with past predictions)" in prompt
    assert "## YOUR MEMORY" in prompt
    assert "mem_df" in prompt
    assert "mem_add" in prompt
    assert "memory_retrieve" in prompt
    assert "memory operations" in prompt


def test_action_tools_follow_memory_mode():
    no_memory_names = {
        tool["function"]["name"]
        for tool in build_action_tools(
            enable_query=True,
            enable_search=True,
            max_outcomes_per_question=5,
            enable_memory=False,
            enable_mem_df=False,
        )
    }
    active_names = {
        tool["function"]["name"]
        for tool in build_action_tools(
            enable_query=True,
            enable_search=True,
            max_outcomes_per_question=5,
            enable_memory=True,
            enable_mem_df=True,
        )
    }

    assert {"query_df", "search_news", "submit_forecasts", "next_day"} <= no_memory_names
    assert "memory_retrieve" not in no_memory_names
    assert "mem_add" not in no_memory_names
    assert "memory_retrieve" in active_names
    assert "mem_add" in active_names


def test_deepseek_shared_warmup_and_restart_configs_are_paired():
    warmup = _load_config("aljazeera_2026Q1_deepseek_v3.2_shared_warmup.yaml")
    active = _load_config("aljazeera_2026Q1_deepseek_v3.2_active_restart.yaml")
    nomem = _load_config("aljazeera_2026Q1_deepseek_v3.2_nomem_restart.yaml")

    warmup_day = date.fromisoformat(warmup["start_date"])
    active_sim_start = (
        date.fromisoformat(active["start_date"])
        - timedelta(days=int(active["lookback_days"]))
    )

    assert warmup["end_date"] == warmup["start_date"]
    assert warmup_day == active_sim_start
    restart_days = {
        cfg.get("restart_from_day")
        for cfg in (active, nomem)
        if cfg.get("restart_from_day")
    }
    assert restart_days <= {"2025-12-25"}
    restart_paths = [
        cfg.get("restart_from")
        for cfg in (active, nomem)
        if cfg.get("restart_from")
    ]
    if len(restart_paths) == 2:
        assert restart_paths[0] == restart_paths[1]

    for cfg in (warmup, active, nomem):
        assert cfg["split"] == "aljazeera2026Q1"
        assert cfg["dataset_path"] == "${FSIM_DATASET_PATH}"
        assert cfg["dataset_cache"] == "${FSIM_DATASET_CACHE}"
        assert cfg["output_base"] == "${FSIM_OUTPUT_BASE}/final_runs_v37"
        assert cfg["search_db"] == "${FSIM_SEARCH_DB}"
        assert cfg["embedding_model"] == "${FSIM_EMBEDDING_MODEL}"
        assert cfg["articles_base"] == "${FSIM_NEWS_BASE}/deduped_articles/data"
        assert len(cfg["agents"]) == 1
        assert cfg["agents"][0]["model"] == "deepseek/deepseek-v3.2"
        assert cfg["defaults"]["scaffold"] == "allQ"
        assert cfg["defaults"]["reasoning"]["effort"] == "none"

    assert warmup["defaults"]["enable_memory"] is True
    assert warmup["defaults"]["memory_format"] == "active"
    assert active["defaults"]["enable_memory"] is True
    assert active["defaults"]["memory_format"] == "active"
    assert nomem["defaults"]["enable_memory"] is False
    assert "memory_format" not in nomem["defaults"]
