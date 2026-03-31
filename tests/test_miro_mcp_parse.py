"""Miro agent MCP-shaped tool extraction."""

import json

from agents.miroAgent.tools import chat_response_to_action, extract_mcp_tool_calls


def test_extract_mcp_search_news():
    text = """
Some reasoning
<use_mcp_tool>
<server_name>search_and_scrape_webpage</server_name>
<tool_name>search_news</tool_name>
<arguments>{"query": "test query", "from_date": null, "to_date": null}</arguments>
</use_mcp_tool>
"""
    calls = extract_mcp_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "search_news"
    assert calls[0]["arguments"]["query"] == "test query"


def test_extract_mcp_q_alias_normalized_to_query():
    text = """<use_mcp_tool>
<tool_name>search_news</tool_name>
<arguments>{"q": "hello world"}</arguments>
</use_mcp_tool>"""
    calls = extract_mcp_tool_calls(text)
    assert calls[0]["arguments"].get("query") == "hello world"


def test_extract_mcp_legacy_google_search_maps_to_search_news():
    text = """<use_mcp_tool>
<tool_name>google_search</tool_name>
<arguments>{"query": "x"}</arguments>
</use_mcp_tool>"""
    calls = extract_mcp_tool_calls(text)
    assert calls[0]["name"] == "search_news"


def test_chat_response_mcp_when_no_api_tool_calls():
    resp = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "<use_mcp_tool><tool_name>next_day</tool_name><arguments>{}</arguments></use_mcp_tool>",
                },
                "finish_reason": "stop",
            }
        ]
    }
    parsed, _, _ = chat_response_to_action(resp)
    assert parsed is not None
    assert parsed.action_type == "next"


def test_chat_response_prefers_mcp_text_over_api_tool_calls():
    resp = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "<use_mcp_tool><tool_name>next_day</tool_name><arguments>{}</arguments></use_mcp_tool>",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "search_news",
                                "arguments": json.dumps({"query": "api would win if chosen"}),
                            },
                        }
                    ],
                },
                "finish_reason": "stop",
            }
        ]
    }
    parsed, _, _ = chat_response_to_action(resp)
    assert parsed is not None
    assert parsed.action_type == "next"


def test_chat_response_falls_back_to_api_tool_calls_when_no_mcp():
    resp = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "I will search now.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "search_news",
                                "arguments": json.dumps({"query": "api path"}),
                            },
                        }
                    ],
                },
                "finish_reason": "stop",
            }
        ]
    }
    parsed, _, _ = chat_response_to_action(resp)
    assert parsed is not None
    assert parsed.action_type == "search"
    assert parsed.query == "api path"
