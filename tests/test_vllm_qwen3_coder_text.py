from skyrl_integration.vllm_qwen3_coder_text import extract_tool_calls_vllm_qwen3_coder


SEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "array"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_forecasts",
            "parameters": {
                "type": "object",
                "properties": {
                    "forecasts": {"type": "array"},
                },
            },
        },
    },
]


def test_extract_tool_calls_normalizes_set_arguments():
    text = """
<tool_call>
<function=search_news>
<parameter=query>{'fed', 'inflation'}</parameter>
</function>
</tool_call>
""".strip()

    calls = extract_tool_calls_vllm_qwen3_coder(text, tools=SEARCH_TOOLS)

    assert len(calls) == 1
    assert calls[0]["name"] == "search_news"
    assert calls[0]["arguments"]["query"] == ["fed", "inflation"]
    assert calls[0]["arguments_raw"] == '{"query": ["fed", "inflation"]}'


def test_extract_tool_calls_keeps_json_shaped_arguments():
    text = """
<tool_call>
<function=submit_forecasts>
<parameter=forecasts>[{"qid": "Q1", "outcomes": {"Yes": 0.7}}]</parameter>
</function>
</tool_call>
""".strip()

    calls = extract_tool_calls_vllm_qwen3_coder(text, tools=SEARCH_TOOLS)

    assert len(calls) == 1
    assert calls[0]["arguments"]["forecasts"][0]["qid"] == "Q1"
    assert calls[0]["arguments"]["forecasts"][0]["outcomes"]["Yes"] == 0.7
