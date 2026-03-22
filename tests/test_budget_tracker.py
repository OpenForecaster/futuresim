from datetime import date

from agents.basicAgent import AgentConfig, BasicAgent
from agents.gptossAgent.tools import build_action_tools as build_gptoss_tools
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


def test_token_budget_tracks_current_context_instead_of_cumulative_usage():
    budget = BudgetTracker(
        BudgetSettings(
            max_total_tokens=100,
            submit_reserve_tokens=10,
            force_submit_threshold_tokens=25,
        ),
        token_estimator=lambda item: len(item) if isinstance(item, str) else len(str(item)),
    )

    budget.bootstrap_context("seed")
    assert budget.current_context_tokens == 4
    assert budget.token_budget_remaining() == 96

    budget.record_usage({"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30})
    assert budget.current_context_tokens == 20
    assert budget.token_budget_remaining() == 80
    assert budget.should_force_submit() is False
    assert budget.is_exhausted() is False

    budget.record_appended_item("abc")
    assert budget.current_context_tokens == 23
    assert budget.token_budget_remaining() == 77

    budget.record_usage({"prompt_tokens": 24, "completion_tokens": 22, "total_tokens": 46})
    assert budget.current_context_tokens == 24
    assert budget.token_budget_remaining() == 76
    assert budget.should_force_submit() is False
    assert budget.is_exhausted() is False

    budget.record_appended_item("abcdefghijklmnop")
    assert budget.current_context_tokens == 40
    assert budget.token_budget_remaining() == 60

    budget.record_usage({"prompt_tokens": 76, "completion_tokens": 7, "total_tokens": 83})
    assert budget.current_context_tokens == 76
    assert budget.token_budget_remaining() == 24
    assert budget.should_force_submit() is True
    assert budget.is_exhausted() is False

    budget.record_appended_item("abcdefghijklmno")
    assert budget.current_context_tokens == 91
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
    budget.record_usage({"prompt_tokens": 80, "completion_tokens": 5, "total_tokens": 85})
    budget.consume_action()

    text = budget.format_feedback("SEARCH RESULTS:\nexample")
    assert "Actions remaining: 1" in text
    assert "Context tokens remaining: 20 (estimated current context 80 / 100)" in text
    assert "No more budget available" not in text
    assert budget.should_force_submit() is True


def test_budget_overview_mentions_dual_budget_rules():
    # Non-warmup without memory: no submit_reserve / force_submit language
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
    assert "context budget of 4096 tokens" in overview
    assert "for this session" in overview
    assert "tracks the current prompt length" in overview
    # Non-warmup days no longer mention submit_reserve / force_submit
    assert "Keep at least" not in overview
    assert "Force-submit" not in overview
    assert "session ends" in overview

    # Warmup still uses submit_reserve / force_submit semantics
    agent_with_warmup = BasicAgent(
        agent_id="test_agent",
        inference_provider=None,
        config=AgentConfig(
            max_actions=5,
            max_total_tokens=4096,
            warmup_max_total_tokens=2048,
            submit_reserve_tokens=512,
            force_submit_threshold_tokens=1024,
        ),
    )
    warmup_overview = agent_with_warmup._build_budget_overview(warmup=True)
    assert "Keep at least 512 tokens free" in warmup_overview
    assert "at or below 1024" in warmup_overview


def test_budget_overview_shows_memory_phase_for_structured_memory():
    from agents.utils.memory import StructuredMemory
    agent = BasicAgent(
        agent_id="test_agent",
        inference_provider=None,
        config=AgentConfig(
            max_total_tokens=100000,
            memory_update_max_total_tokens=30000,
        ),
    )
    # Simulate structured memory being enabled
    agent._memory = StructuredMemory.__new__(StructuredMemory)
    agent._memory._max_entries = 500
    agent._memory._entries = {}

    overview = agent._build_budget_overview()
    assert "memory update phase" in overview
    assert "~30000 tokens are reserved for memory" in overview
    assert "~70000 tokens have been used" in overview
    assert "Keep at least" not in overview


def test_memory_phase_threshold_triggers_correctly():
    budget = BudgetTracker(BudgetSettings(
        max_total_tokens=100000,
        submit_reserve_tokens=0,
        force_submit_threshold_tokens=0,
        memory_phase_threshold_tokens=30000,
    ))

    budget.record_usage({"prompt_tokens": 60000})
    assert budget.should_enter_memory_phase() is False
    assert budget.tokens_until_memory_phase() == 10000

    budget.record_usage({"prompt_tokens": 70000})
    assert budget.should_enter_memory_phase() is True
    assert budget.tokens_until_memory_phase() == 0

    # After entering memory phase, should_enter returns False
    budget.memory_phase = True
    assert budget.should_enter_memory_phase() is False


def test_status_text_shows_memory_phase_metric():
    budget = BudgetTracker(BudgetSettings(
        max_total_tokens=100000,
        submit_reserve_tokens=0,
        force_submit_threshold_tokens=0,
        memory_phase_threshold_tokens=30000,
    ))
    budget.record_usage({"prompt_tokens": 50000})

    text = budget.status_text()
    assert "Context tokens remaining: 50000" in text
    assert "Tokens remaining until memory phase: 20000" in text

    # In memory phase, the "until memory phase" line disappears
    budget.memory_phase = True
    text = budget.status_text()
    assert "Context tokens remaining: 50000" in text
    assert "Tokens remaining until memory phase" not in text


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


def test_qwen_tool_descriptions_preserve_basic_prompt_guidance():
    tools = build_qwen_tools(
        enable_query=True,
        enable_search=True,
        max_outcomes_per_question=5,
        max_search_results=7,
        search_chunk_tokens=321,
    )
    qwen_tools = {tool["function"]["name"]: tool["function"] for tool in tools}

    assert "plain .head() previews can be unreliable outside notebooks" in qwen_tools["query_df"]["description"]
    assert "import statements are unavailable" in qwen_tools["query_df"]["description"]
    assert "today's date" in qwen_tools["search_news"]["description"]
    assert "never placeholders like Unknown, TBD, Other, or N/A" in qwen_tools["submit_forecasts"]["description"]
    assert "done querying, searching, and submitting forecasts" in qwen_tools["next_day"]["description"]
    assert "wakeup" not in qwen_tools["next_day"]["description"].lower()


def test_native_tool_descriptions_share_submit_target_qid_wording():
    gptoss_tools = {
        tool["name"]: tool
        for tool in build_gptoss_tools(
            enable_query=True,
            enable_search=True,
            max_outcomes_per_question=5,
            max_search_results=7,
            search_chunk_tokens=321,
        )
    }
    qwen_tools = {
        tool["function"]["name"]: tool["function"]
        for tool in build_qwen_tools(
            enable_query=True,
            enable_search=True,
            max_outcomes_per_question=5,
            max_search_results=7,
            search_chunk_tokens=321,
        )
    }

    target_qid_phrase = "If the prompt specifies a target question ID"
    assert target_qid_phrase in gptoss_tools["submit_forecasts"]["description"]
    assert target_qid_phrase in qwen_tools["submit_forecasts"]["description"]
    assert "today's date" in gptoss_tools["search_news"]["description"]
    assert "today's date" in qwen_tools["search_news"]["description"]
    assert "session with no more actions" in gptoss_tools["next_day"]["description"]


def test_cadence_language_uses_updates_not_wakeups():
    agent = BasicAgent(
        agent_id="a1",
        inference_provider=object(),
        config=AgentConfig(enable_memory=False, timegap_days=5),
    )
    agent._forecast_interface = type(
        "ForecastStub",
        (),
        {
            "last_active_date": date(2025, 1, 1),
            "next_active_date": date(2025, 1, 11),
        },
    )()

    cadence = agent._build_cadence_section(date(2025, 1, 6))

    assert "make updates every 5 days" in cadence
    assert "Last update" in cadence
    assert "Current date" in cadence
    assert "Next scheduled update" in cadence
    assert "wakeup" not in cadence.lower()
