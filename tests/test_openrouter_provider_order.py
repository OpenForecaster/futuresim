"""Tests for the openrouter_provider_order config parameter."""

import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# 1. create_inference_provider: provider order specified
# ---------------------------------------------------------------------------

@patch("inference.openrouter.OpenRouterInference.__init__", return_value=None)
def test_create_inference_provider_with_provider_order(mock_init):
    """When openrouter_provider_order is given, OpenRouterInference receives a provider dict."""
    from scripts.test_basic_agent import create_inference_provider

    args = MagicMock()
    create_inference_provider(
        "openrouter", "deepseek/deepseek-v3.2", args,
        openrouter_provider_order=["Friendli", "Together"],
    )
    mock_init.assert_called_once_with(
        "deepseek/deepseek-v3.2",
        provider={"order": ["Friendli", "Together"], "allow_fallbacks": True},
    )


# ---------------------------------------------------------------------------
# 2. create_inference_provider: provider order not specified
# ---------------------------------------------------------------------------

@patch("inference.openrouter.OpenRouterInference.__init__", return_value=None)
def test_create_inference_provider_without_provider_order(mock_init):
    """When openrouter_provider_order is None, no provider kwarg is passed."""
    from scripts.test_basic_agent import create_inference_provider

    args = MagicMock()
    create_inference_provider("openrouter", "deepseek/deepseek-v3.2", args)
    mock_init.assert_called_once_with("deepseek/deepseek-v3.2")


@patch("inference.openrouter.OpenRouterInference.__init__", return_value=None)
def test_create_inference_provider_with_empty_provider_order(mock_init):
    """An empty list should behave the same as None — no provider kwarg."""
    from scripts.test_basic_agent import create_inference_provider

    args = MagicMock()
    create_inference_provider(
        "openrouter", "deepseek/deepseek-v3.2", args,
        openrouter_provider_order=[],
    )
    mock_init.assert_called_once_with("deepseek/deepseek-v3.2")


# ---------------------------------------------------------------------------
# 3. _build_payload: provider kwarg appears in the output payload
# ---------------------------------------------------------------------------

@patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
def test_build_payload_includes_provider():
    """Provider dict stored via kwargs should appear in the built payload."""
    from inference.openrouter import OpenRouterInference

    inf = OpenRouterInference(
        "deepseek/deepseek-v3.2",
        provider={"order": ["Friendli"], "allow_fallbacks": True},
    )
    payload = inf._build_payload(
        messages=[{"role": "user", "content": "hi"}],
        sampling_params={"temperature": 0.7},
    )
    assert payload["provider"] == {"order": ["Friendli"], "allow_fallbacks": True}


@patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
def test_build_payload_no_provider_when_omitted():
    """When no provider kwarg is given, the payload should not contain a provider key."""
    from inference.openrouter import OpenRouterInference

    inf = OpenRouterInference("deepseek/deepseek-v3.2")
    payload = inf._build_payload(
        messages=[{"role": "user", "content": "hi"}],
        sampling_params={"temperature": 0.7},
    )
    assert "provider" not in payload


# ---------------------------------------------------------------------------
# 4. Config parsing: openrouter_provider_order extraction logic
# ---------------------------------------------------------------------------
# create_agents_from_config uses: agent_def.get('openrouter_provider_order',
#   defaults.get('openrouter_provider_order', None))
# We test this pattern directly to avoid mocking the full agent creation stack.

def _extract_provider_order(agent_def, defaults):
    """Mirror the extraction logic used in create_agents_from_config."""
    return agent_def.get('openrouter_provider_order',
                         defaults.get('openrouter_provider_order', None))


def test_config_parsing_provider_order_from_defaults():
    """openrouter_provider_order in defaults flows through when agent doesn't override."""
    defaults = {"provider": "openrouter", "openrouter_provider_order": ["Friendli", "Together"]}
    agent_def = {"model": "deepseek/deepseek-v3.2"}
    assert _extract_provider_order(agent_def, defaults) == ["Friendli", "Together"]


def test_config_parsing_provider_order_per_agent_overrides_default():
    """Per-agent openrouter_provider_order overrides the defaults value."""
    defaults = {"provider": "openrouter", "openrouter_provider_order": ["Together"]}
    agent_def = {"model": "deepseek/deepseek-v3.2", "openrouter_provider_order": ["Friendli"]}
    assert _extract_provider_order(agent_def, defaults) == ["Friendli"]


def test_config_parsing_no_provider_order():
    """When openrouter_provider_order is absent from both, None is returned."""
    defaults = {"provider": "openrouter"}
    agent_def = {"model": "deepseek/deepseek-v3.2"}
    assert _extract_provider_order(agent_def, defaults) is None
