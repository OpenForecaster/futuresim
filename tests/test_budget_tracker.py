from agents.basicAgent import AgentConfig, BasicAgent
from agents.gptossAgent.tools import build_action_tools as build_gptoss_tools
from agents.qwenAgent import QwenBasicAgent
from agents.qwenAgent.tools import build_action_tools as build_qwen_tools
from agents.utils.budget import BudgetSettings, BudgetTracker


def test_action_budget_matches_legacy_final_turn_behavior():
    budget = BudgetTracker(BudgetSettings(max_actions=3))

    assert budget.should_force_submit() is False
    assert budget.is_exhausted() is False

    budget.consume_action()
    assert budget.actions_remaining == 2
    assert budget.should_force_submit() is False
    assert budget.is_exhausted() is False

    budget.consume_action()
    assert budget.actions_remaining == 1
    assert budget.should_force_submit() is True
    assert budget.is_exhausted() is False

    budget.consume_action()
    assert budget.actions_remaining == 0
    assert budget.should_force_submit() is False
    assert budget.is_exhausted() is True


def test_token_budget_uses_total_tokens_and_reserve_thresholds():
    budget = BudgetTracker(
        BudgetSettings(
            max_total_tokens=100,
            submit_reserve_tokens=10,
            force_submit_threshold_tokens=25,
        )
    )

    budget.record_usage({"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30})
    assert budget.total_tokens_used == 30
    assert budget.token_budget_remaining() == 70
    assert budget.should_force_submit() is False
    assert budget.is_exhausted() is False

    budget.record_usage({"prompt_tokens": 24, "completion_tokens": 22, "total_tokens": 46})
    assert budget.total_tokens_used == 76
    assert budget.token_budget_remaining() == 24
    assert budget.should_force_submit() is True
    assert budget.is_exhausted() is False

    budget.record_usage({"prompt_tokens": 8, "completion_tokens": 7, "total_tokens": 15})
    assert budget.total_tokens_used == 91
    assert budget.token_budget_remaining() == 9
    assert budget.should_force_submit() is False
    assert budget.is_exhausted() is True


def test_combined_budget_feedback_shows_both_constraints():
    budget = BudgetTracker(
        BudgetSettings(
            max_actions=2,
            max_total_tokens=100,
            submit_reserve_tokens=10,
            force_submit_threshold_tokens=25,
        )
    )
    budget.record_usage({"total_tokens": 80})
    budget.consume_action()

    text = budget.format_feedback("SEARCH RESULTS:\nexample")
    assert "Actions remaining: 1" in text
    assert "Token budget remaining: 20 (used 80 / 100)" in text
    assert "No more budget available" not in text
    assert budget.should_force_submit() is True


def test_budget_overview_mentions_dual_budget_rules():
    agent = BasicAgent(
        agent_id="test_agent",
        inference_provider=None,
        config=AgentConfig(
            max_actions=5,
            max_total_tokens=4096,
            submit_reserve_tokens=512,
            force_submit_threshold_tokens=1024,
        ),
    )

    overview = agent._build_budget_overview()
    assert "5 actions per day" in overview
    assert "total token budget of 4096 tokens" in overview
    assert "Keep at least 512 tokens in reserve" in overview
    assert "at or below 1024" in overview
    assert "both are enforced" in overview


def test_search_tool_descriptions_mention_chunk_limits():
    gptoss_search = next(
        tool
        for tool in build_gptoss_tools(
            enable_query=False,
            enable_search=True,
            max_outcomes_per_question=5,
            max_search_results=7,
            search_chunk_tokens=321,
        )
        if tool["name"] == "search_news"
    )
    qwen_search = next(
        tool
        for tool in build_qwen_tools(
            enable_query=False,
            enable_search=True,
            max_outcomes_per_question=5,
            max_search_results=7,
            search_chunk_tokens=321,
        )
        if tool["function"]["name"] == "search_news"
    )

    assert "up to 7 retrieved article chunks" in gptoss_search["description"]
    assert "roughly 321 tokens long" in gptoss_search["description"]
    assert "up to 7 retrieved article chunks" in qwen_search["function"]["description"]
    assert "roughly 321 tokens long" in qwen_search["function"]["description"]


def test_qwen_logging_payload_includes_full_messages_and_tools():
    payload = QwenBasicAgent._build_model_input_for_logging(
        messages=[
            {"role": "user", "content": "initial"},
            {"role": "tool", "name": "search_news", "tool_call_id": "call_1", "content": "SEARCH RESULTS"},
        ],
        tools=[{"function": {"name": "submit_forecasts"}}],
    )

    assert '"role": "tool"' in payload
    assert '"content": "SEARCH RESULTS"' in payload
    assert '"name": "submit_forecasts"' in payload
