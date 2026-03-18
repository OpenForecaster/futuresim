import json
from typing import Any, Dict, List, Optional, Tuple

from agents.gptossAgent.tools import response_to_action
from agents.utils.forecast_parser import ParsedAction


def _as_chat_function_tool(
    *,
    name: str,
    description: str,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def build_action_tools(
    *,
    enable_query: bool,
    enable_search: bool,
    max_outcomes_per_question: int,
    max_search_results: int = 5,
    search_chunk_tokens: Optional[int] = None,
) -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = []
    if search_chunk_tokens is None:
        search_results_description = (
            f"Search returns up to {max_search_results} retrieved article chunks."
        )
    else:
        search_results_description = (
            f"Search returns up to {max_search_results} retrieved article chunks, "
            f"each roughly {search_chunk_tokens} tokens long."
        )

    if enable_query:
        tools.append(
            _as_chat_function_tool(
                name="query_df",
                description=(
                    "Run Python code to inspect the questions DataFrame before forecasting. "
                    "Use print(...) to show results because plain .head() previews can be unreliable "
                    "outside notebooks. The sandbox predefines df, pd, today, date, datetime, "
                    "timedelta, and standard builtins; import statements are unavailable. "
                    "Example: print(df[df['is_resolved'] == False][['qid','title','answer_type']].head())"
                ),
                parameters={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": (
                                "Python code to execute in the provided sandbox. "
                                "You may write multi-step code, but you must use print(...) "
                                "for outputs."
                            ),
                        }
                    },
                    "required": ["code"],
                },
            )
        )

    if enable_search:
        tools.append(
            _as_chat_function_tool(
                name="search_news",
                description=(
                    "Search the news article database for evidence before submitting forecasts. "
                    "Use at most one search per turn. "
                    f"{search_results_description} "
                    "You may optionally pass YYYY-MM-DD date filters; to_date cannot be after "
                    "today's date. "
                    "Example: search for 'Fed rate cut 2026 inflation'."
                ),
                parameters={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "News search query string. Focus on concrete entities, events, "
                                "and evidence relevant to the target forecast."
                            ),
                        },
                        "from_date": {
                            "type": ["string", "null"],
                            "description": (
                                "Optional minimum date filter (YYYY-MM-DD)."
                            ),
                        },
                        "to_date": {
                            "type": ["string", "null"],
                            "description": (
                                "Optional maximum date filter (YYYY-MM-DD). If provided, it "
                                "cannot be after today's date."
                            ),
                        },
                    },
                    "required": ["query"],
                },
            )
        )

    tools.append(
        _as_chat_function_tool(
            name="submit_forecasts",
            description=(
                "Submit exactly one forecast for exactly one active question id. "
                "Each call must contain a single-item forecasts list for one qid only, and you "
                "may submit again later in the same session to update that qid. Use real predicted "
                "answers only; never placeholders like Unknown, TBD, Other, or N/A. Probabilities "
                "must sum to <= 1.0. If the prompt specifies a target question ID, you may submit "
                "only for that question. "
                "Example payload: "
                '{"forecasts":[{"qid":"Q123","outcomes":{"Candidate A":0.55,"Candidate B":0.35}}]}'
            ),
            parameters={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "forecasts": {
                        "type": "array",
                        "description": (
                            "Single-item list containing exactly one forecast for one qid."
                        ),
                        "minItems": 1,
                        "maxItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "qid": {
                                    "type": "string",
                                    "description": (
                                        "Question id for an active question. When the prompt "
                                        "specifies a target question ID, this must match it exactly."
                                    ),
                                },
                                "outcomes": {
                                    "type": "object",
                                    "description": (
                                        "Map outcome_name -> probability. Use concrete predicted "
                                        "answers only, not placeholders. "
                                        f"Max {max_outcomes_per_question} outcomes per question, "
                                        "with probabilities summing to <= 1.0."
                                    ),
                                    "additionalProperties": {"type": "number"},
                                },
                            },
                            "required": ["qid", "outcomes"],
                        },
                    }
                },
                "required": ["forecasts"],
            },
        )
    )

    tools.append(
        _as_chat_function_tool(
            name="next_day",
            description=(
                "End this session with no more actions. Call this only when you are done "
                "querying, searching, and submitting forecasts. Example payload: {}"
            ),
            parameters={
                "type": "object",
                "additionalProperties": False,
                "properties": {},
                "required": [],
            },
        )
    )

    return tools


def extract_assistant_text(chat_response_json: Dict[str, Any]) -> str:
    message = _extract_choice_message(chat_response_json)
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                chunks.append(text)
        return "".join(chunks)
    if content is None:
        return ""
    return str(content)


def extract_assistant_message(chat_response_json: Dict[str, Any]) -> Dict[str, Any]:
    message = _extract_choice_message(chat_response_json)
    out: Dict[str, Any] = {"role": "assistant"}
    if "content" in message:
        out["content"] = message.get("content")
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        out["tool_calls"] = tool_calls
    return out


def extract_tool_calls(chat_response_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    message = _extract_choice_message(chat_response_json)
    raw_calls = message.get("tool_calls")
    if not isinstance(raw_calls, list):
        return []

    calls: List[Dict[str, Any]] = []
    for call in raw_calls:
        if not isinstance(call, dict):
            continue
        if call.get("type") != "function":
            continue
        fn = call.get("function")
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        arguments_raw = fn.get("arguments")
        if not isinstance(name, str):
            continue
        if arguments_raw is None:
            arguments_raw = ""
        if not isinstance(arguments_raw, str):
            arguments_raw = str(arguments_raw)
        try:
            arguments = json.loads(arguments_raw) if arguments_raw else {}
        except Exception:
            arguments = {}
        call_id = call.get("id")
        calls.append(
            {
                "name": name,
                "arguments": arguments,
                "arguments_raw": arguments_raw,
                "call_id": call_id if isinstance(call_id, str) else None,
            }
        )
    return calls


def extract_finish_reason(chat_response_json: Dict[str, Any]) -> Optional[str]:
    choices = chat_response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    reason = first.get("finish_reason")
    return reason if isinstance(reason, str) else None


def chat_response_to_action(
    chat_response_json: Dict[str, Any],
) -> Tuple[Optional[ParsedAction], str, List[Dict[str, Any]]]:
    assistant_text = extract_assistant_text(chat_response_json)
    tool_calls = extract_tool_calls(chat_response_json)
    parsed, _, normalized_calls = response_to_action(
        {"output_text": assistant_text},
        tool_calls=tool_calls,
    )
    return parsed, assistant_text, normalized_calls


def _extract_choice_message(chat_response_json: Dict[str, Any]) -> Dict[str, Any]:
    choices = chat_response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    first = choices[0]
    if not isinstance(first, dict):
        return {}
    message = first.get("message")
    return message if isinstance(message, dict) else {}
