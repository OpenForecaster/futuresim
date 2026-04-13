"""Shared Chat Completions tool schemas and parsers for forecast agents."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from agents.utils.budget import BudgetTracker
from agents.utils.forecast_parser import ParsedAction

from .search import SearchHandler

import re

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _validate_date_arg(val: Any) -> Optional[str]:
    """Return a YYYY-MM-DD string if *val* looks like a valid date, else None.

    When the LLM puts non-date text (e.g. keywords) into from_date / to_date,
    this silently drops it to None so the search runs unfiltered rather than
    with a broken filter.
    """
    if not isinstance(val, str) or not val.strip():
        return None
    val = val.strip()
    if not _DATE_RE.match(val):
        return None
    from datetime import datetime as _datetime
    try:
        _datetime.strptime(val, "%Y-%m-%d")
    except ValueError:
        return None
    return val


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
    enable_memory: bool = False,
    enable_mem_df: bool = False,
) -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = []
    extra_search_info = "The search tool uses a hybrid approach to retrieve articles, combining both semantic similarity (through an embedding model) and keyword matching." 
    if search_chunk_tokens is None:
        search_results_description = (
            f"Search returns up to {max_search_results} retrieved article chunks. {extra_search_info}"
        )
    else:
        search_results_description = (
            f"Search returns up to {max_search_results} retrieved article chunks, "
            f"each roughly {search_chunk_tokens} tokens long. {extra_search_info}"
        )

    if enable_query:
        tools.append(
            _as_chat_function_tool(
                name="query_df",
                description=(
                    "Run Python code to inspect the questions DataFrame before forecasting. "
                    "Use print(...) to show results because plain .head() previews can be unreliable "
                    "outside notebooks. The sandbox predefines df, pd, today, date, datetime, "
                    "timedelta, and standard builtins. A small safe subset of stdlib imports "
                    "(for example datetime, json, math, re, ast) is allowed. "
                    "External file, network, process, and private-attribute access is blocked; "
                    "use only in-memory DataFrame/pandas operations. "
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
                                "Optional earliest date for articles (YYYY-MM-DD format ONLY, "
                                "e.g. '2024-06-01'). Must be a valid date, not keywords."
                            ),
                        },
                        "to_date": {
                            "type": ["string", "null"],
                            "description": (
                                "Optional latest date for articles (YYYY-MM-DD format ONLY, "
                                "e.g. '2025-03-15'). Must be a valid date, not keywords. "
                                "Cannot be after today's date."
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
                "only for that question. Example payload: "
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

    if enable_memory:
        tools.extend(_build_memory_tool_schemas())
    if enable_mem_df:
        tools.extend(_build_mem_df_tool_schemas())

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


def _build_memory_tool_schemas() -> List[Dict[str, Any]]:
    return [
        _as_chat_function_tool(
            name="memory_retrieve",
            description=(
                "Retrieve the full content of a memory entry by name. "
                "Use this to recall details before updating or re-using an entry."
            ),
            parameters={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The entry name (lowercase, hyphens, alphanumeric).",
                    },
                },
                "required": ["name"],
            },
        ),
        _as_chat_function_tool(
            name="memory_new",
            description=(
                "Create a new memory entry for cross-question insights, lessons, or patterns. "
                "Every entry must contain a reusable insight or reasoning chain, not logs."
            ),
            parameters={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "Short, descriptive entry name (lowercase, hyphens, alphanumeric). "
                            "Example: q247-lesson-ceremony-precedent"
                        ),
                    },
                    "description": {
                        "type": "string",
                        "description": "One-line summary shown in the index. Include when/why to use this entry.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full entry content with the insight, reasoning chain, or pattern.",
                    },
                },
                "required": ["name", "description", "content"],
            },
        ),
        _as_chat_function_tool(
            name="memory_update",
            description=(
                "Update an existing memory entry's description and/or content. "
                "Provide only the fields you want to change."
            ),
            parameters={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The entry name to update.",
                    },
                    "description": {
                        "type": "string",
                        "description": "New description (optional, omit to keep current).",
                    },
                    "content": {
                        "type": "string",
                        "description": "New content (optional, omit to keep current).",
                    },
                },
                "required": ["name"],
            },
        ),
        _as_chat_function_tool(
            name="memory_delete",
            description="Delete a memory entry by name. Use for stale or resolved entries.",
            parameters={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The entry name to delete.",
                    },
                },
                "required": ["name"],
            },
        ),
    ]


def _build_mem_df_tool_schemas() -> List[Dict[str, Any]]:
    return [
        _as_chat_function_tool(
            name="mem_add",
            description=(
                "Add or update a per-question memory entry in mem_df. "
                "If a row for this qid already exists, it will be replaced (upsert). "
                "Max 1000 characters for the memory field."
            ),
            parameters={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "qid": {
                        "type": "string",
                        "description": "Question ID (e.g. 'Q42').",
                    },
                    "question": {
                        "type": "string",
                        "description": "Short question title for context.",
                    },
                    "memory": {
                        "type": "string",
                        "description": "Your reasoning, key evidence, or prediction rationale for this question.",
                    },
                    "category": {
                        "type": "string",
                        "description": "Topic category (e.g. 'politics', 'sports', 'economics').",
                    },
                },
                "required": ["qid", "memory"],
            },
        ),
        _as_chat_function_tool(
            name="mem_update",
            description=(
                "Update the memory (and optionally category) for an existing mem_df entry."
            ),
            parameters={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "qid": {
                        "type": "string",
                        "description": "Question ID of the entry to update.",
                    },
                    "memory": {
                        "type": "string",
                        "description": "Updated reasoning or evidence.",
                    },
                    "category": {
                        "type": "string",
                        "description": "Updated category (optional, omit to keep current).",
                    },
                },
                "required": ["qid", "memory"],
            },
        ),
        _as_chat_function_tool(
            name="mem_delete",
            description="Delete a per-question memory entry from mem_df by question ID.",
            parameters={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "qid": {
                        "type": "string",
                        "description": "Question ID of the entry to delete.",
                    },
                },
                "required": ["qid"],
            },
        ),
    ]


def build_memory_phase_tools(
    *,
    enable_memory: bool = True,
    enable_mem_df: bool = False,
) -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = []
    if enable_memory:
        tools.extend(_build_memory_tool_schemas())
    if enable_mem_df:
        tools.extend(_build_mem_df_tool_schemas())
    tools.append(
        _as_chat_function_tool(
            name="next_day",
            description="End the memory update phase and finish the session.",
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


def tool_calls_to_parsed_action(
    tool_calls: Optional[List[Dict[str, Any]]],
    *,
    assistant_text: str = "",
) -> Tuple[Optional[ParsedAction], str, List[Dict[str, Any]]]:
    calls = tool_calls if tool_calls is not None else []
    if not calls:
        return None, assistant_text, calls

    call = calls[0]
    name = call.get("name")
    args = call.get("arguments")
    if args is None:
        args = {}

    if not isinstance(name, str) or not name:
        return (
            ParsedAction(
                action_type=None,
                code=None,
                forecasts=None,
                query=None,
                error=f"Malformed tool call: missing/invalid name ({name!r})",
            ),
            assistant_text,
            calls,
        )

    if not isinstance(args, dict):
        return (
            ParsedAction(
                action_type=None,
                code=None,
                forecasts=None,
                query=None,
                error=f"Malformed tool call args for {name!r}: expected object, got {type(args).__name__}",
            ),
            assistant_text,
            calls,
        )

    if name == "next_day":
        return (
            ParsedAction(action_type="next", code=None, forecasts=None, query=None, error=None),
            assistant_text,
            calls,
        )

    if name == "query_df":
        code = args.get("code")
        if not isinstance(code, str) or not code.strip():
            return (
                ParsedAction(
                    action_type="query",
                    code=None,
                    forecasts=None,
                    query=None,
                    error="Missing code",
                ),
                assistant_text,
                calls,
            )
        return (
            ParsedAction(action_type="query", code=code, forecasts=None, query=None, error=None),
            assistant_text,
            calls,
        )

    if name == "search_news":
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return (
                ParsedAction(
                    action_type="search",
                    code=None,
                    forecasts=None,
                    query=None,
                    error="Missing query",
                ),
                assistant_text,
                calls,
            )
        frm = _validate_date_arg(args.get("from_date"))
        to = _validate_date_arg(args.get("to_date"))
        return (
            ParsedAction(
                action_type="search",
                code=None,
                forecasts=None,
                query=query.strip(),
                search_from=frm,
                search_to=to,
                error=None,
            ),
            assistant_text,
            calls,
        )

    if name == "submit_forecasts":
        forecasts = args.get("forecasts")
        if not isinstance(forecasts, list) or not forecasts:
            return (
                ParsedAction(
                    action_type="submit",
                    code=None,
                    forecasts=None,
                    query=None,
                    error="Missing forecasts",
                ),
                assistant_text,
                calls,
            )
        out: List[Dict[str, Any]] = []
        for forecast in forecasts:
            if not isinstance(forecast, dict):
                continue
            qid = forecast.get("qid")
            outcomes = forecast.get("outcomes")
            if not isinstance(qid, str) or not isinstance(outcomes, dict):
                continue
            clean_outcomes: Dict[str, float] = {}
            for key, value in outcomes.items():
                if not isinstance(key, str):
                    continue
                try:
                    clean_outcomes[key] = float(value)
                except Exception:
                    continue
            out.append({"qid": qid, "outcomes": clean_outcomes})
        if not out:
            return (
                ParsedAction(
                    action_type="submit",
                    code=None,
                    forecasts=None,
                    query=None,
                    error="No valid forecasts",
                ),
                assistant_text,
                calls,
            )
        return (
            ParsedAction(action_type="submit", code=None, forecasts=out, query=None, error=None),
            assistant_text,
            calls,
        )

    return (
        ParsedAction(
            action_type=None,
            code=None,
            forecasts=None,
            query=None,
            error=f"Unknown tool call: {name!r}",
        ),
        assistant_text,
        calls,
    )


def single_call_to_parsed_action(
    call: Dict[str, Any],
) -> Tuple[Optional[ParsedAction], str]:
    """Parse a single tool call dict into a ParsedAction.

    Routes through ``_memory_tool_call_to_action`` first (matching the
    dispatch order in ``chat_response_to_action``) so that memory-phase
    tools are recognised correctly.
    """
    name = call.get("name", "")
    args = call.get("arguments") or {}
    if isinstance(args, dict):
        mem_parsed = _memory_tool_call_to_action(name, args)
        if mem_parsed is not None:
            return mem_parsed, ""
    parsed, text, _ = tool_calls_to_parsed_action([call])
    return parsed, text


def _memory_tool_call_to_action(name: str, args: Dict[str, Any]) -> Optional[ParsedAction]:
    if name == "memory_retrieve":
        entry_name = args.get("name")
        if not isinstance(entry_name, str) or not entry_name.strip():
            return ParsedAction(action_type="memory_retrieve", code=None, forecasts=None, query=None, error="Missing 'name' parameter")
        return ParsedAction(action_type="memory_retrieve", code=None, forecasts=None, query=None, memory_entry_name=entry_name.strip())

    if name == "memory_new":
        name_value = args.get("name")
        description = args.get("description")
        content = args.get("content")
        if not isinstance(name_value, str) or not name_value.strip():
            return ParsedAction(action_type="memory_new", code=None, forecasts=None, query=None, error="Missing 'name' parameter")
        if not isinstance(content, str) or not content.strip():
            return ParsedAction(action_type="memory_new", code=None, forecasts=None, query=None, error="Missing 'content' parameter")
        return ParsedAction(
            action_type="memory_new",
            code=None,
            forecasts=None,
            query=None,
            memory_new_data={
                "name": name_value.strip(),
                "description": (description.strip() if isinstance(description, str) and description.strip() else name_value.strip()),
                "content": content.strip(),
            },
        )

    if name == "memory_update":
        entry_name = args.get("name")
        if not isinstance(entry_name, str) or not entry_name.strip():
            return ParsedAction(action_type="memory_update", code=None, forecasts=None, query=None, error="Missing 'name' parameter")
        update_data: Dict[str, str] = {}
        for field in ("description", "content"):
            value = args.get(field)
            if isinstance(value, str) and value.strip():
                update_data[field] = value.strip()
        if not update_data:
            return ParsedAction(
                action_type="memory_update",
                code=None,
                forecasts=None,
                query=None,
                error="No fields to update (need description or content)",
            )
        return ParsedAction(
            action_type="memory_update",
            code=None,
            forecasts=None,
            query=None,
            memory_entry_name=entry_name.strip(),
            memory_update_data=update_data,
        )

    if name == "memory_delete":
        entry_name = args.get("name")
        if not isinstance(entry_name, str) or not entry_name.strip():
            return ParsedAction(action_type="memory_delete", code=None, forecasts=None, query=None, error="Missing 'name' parameter")
        return ParsedAction(action_type="memory_delete", code=None, forecasts=None, query=None, memory_entry_name=entry_name.strip())

    if name == "mem_add":
        qid = args.get("qid")
        memory = args.get("memory")
        if not isinstance(qid, str) or not qid.strip():
            return ParsedAction(action_type="mem_add", code=None, forecasts=None, query=None, error="Missing 'qid' parameter")
        if not isinstance(memory, str) or not memory.strip():
            return ParsedAction(action_type="mem_add", code=None, forecasts=None, query=None, error="Missing 'memory' parameter")
        question = args.get("question", "")
        category = args.get("category", "")
        return ParsedAction(
            action_type="mem_add",
            code=None,
            forecasts=None,
            query=None,
            mem_data={
                "qid": qid.strip(),
                "question": question.strip() if isinstance(question, str) else "",
                "memory": memory.strip(),
                "category": category.strip() if isinstance(category, str) else "",
            },
        )

    if name == "mem_update":
        qid = args.get("qid")
        memory = args.get("memory")
        if not isinstance(qid, str) or not qid.strip():
            return ParsedAction(action_type="mem_update", code=None, forecasts=None, query=None, error="Missing 'qid' parameter")
        if not isinstance(memory, str) or not memory.strip():
            return ParsedAction(action_type="mem_update", code=None, forecasts=None, query=None, error="Missing 'memory' parameter")
        category = args.get("category")
        return ParsedAction(
            action_type="mem_update",
            code=None,
            forecasts=None,
            query=None,
            mem_qid=qid.strip(),
            mem_data={
                "qid": qid.strip(),
                "memory": memory.strip(),
                "category": category.strip() if isinstance(category, str) and category.strip() else None,
            },
        )

    if name == "mem_delete":
        qid = args.get("qid")
        if not isinstance(qid, str) or not qid.strip():
            return ParsedAction(action_type="mem_delete", code=None, forecasts=None, query=None, error="Missing 'qid' parameter")
        return ParsedAction(action_type="mem_delete", code=None, forecasts=None, query=None, mem_qid=qid.strip())

    return None


def chat_response_to_action(
    chat_response_json: Dict[str, Any],
) -> Tuple[Optional[ParsedAction], str, List[Dict[str, Any]]]:
    assistant_text = extract_assistant_text(chat_response_json)
    tool_calls = extract_tool_calls(chat_response_json)

    if tool_calls:
        call = tool_calls[0]
        name = call.get("name", "")
        args = call.get("arguments") or {}
        mem_parsed = _memory_tool_call_to_action(name, args if isinstance(args, dict) else {})
        if mem_parsed is not None:
            return mem_parsed, assistant_text, tool_calls

    return tool_calls_to_parsed_action(tool_calls, assistant_text=assistant_text)


def final_submit_tool_instruction_text(
    target_qid: str,
    budget: Optional[BudgetTracker],
    *,
    forbid_query_df: bool = True,
) -> str:
    lines = ["FINAL ACTION: You MUST submit your best guess forecast now."]
    if budget is not None:
        status = budget.status_text()
        if status:
            lines.append(status)
    preamble = "\n".join(lines)
    forbid = (
        "Do NOT call query_df, search_news, or next_day.\n"
        if forbid_query_df
        else "Do NOT call search_news or next_day.\n"
    )
    return (
        f"{preamble}\n"
        f"Target question ID: {target_qid}\n"
        "Call exactly one tool: submit_forecasts.\n"
        f"{forbid}"
        "Submit only this question id with concrete outcomes and probabilities (sum <= 1.0)."
    )


@dataclass(frozen=True)
class NewsSearchStepResult:
    feedback: str
    phase: str
    successful_hit: bool


def optional_search_dates_from_parsed(parsed: ParsedAction) -> Tuple[Optional[date], Optional[date]]:
    min_date: Optional[date] = None
    max_date: Optional[date] = None
    if parsed.search_from:
        try:
            from datetime import datetime as _datetime

            min_date = _datetime.strptime(parsed.search_from, "%Y-%m-%d").date()
        except Exception:
            min_date = None
    if parsed.search_to:
        try:
            from datetime import datetime as _datetime

            max_date = _datetime.strptime(parsed.search_to, "%Y-%m-%d").date()
        except Exception:
            max_date = None
    return min_date, max_date


def execute_news_search(
    parsed: ParsedAction,
    search_handler: SearchHandler,
    *,
    max_results: int,
    search_type: str = "hybrid",
    min_date: Optional[date] = None,
    max_date: Optional[date] = None,
) -> NewsSearchStepResult:
    if not search_handler.is_available:
        return NewsSearchStepResult(
            feedback="SEARCH ERROR: Search is not available.",
            phase="search_unavailable",
            successful_hit=False,
        )
    if not parsed.query:
        return NewsSearchStepResult(
            feedback="SEARCH ERROR: Missing query.",
            phase="search_error",
            successful_hit=False,
        )

    result, error = search_handler.search(
        parsed.query,
        max_results=max_results,
        search_type=search_type,
        min_date=min_date,
        max_date=max_date,
    )
    if error:
        return NewsSearchStepResult(
            feedback=f"SEARCH ERROR: {error}",
            phase="search_error",
            successful_hit=False,
        )
    if not result or result.startswith("No articles found"):
        return NewsSearchStepResult(
            feedback=f"SEARCH RESULTS:\n{result}",
            phase="search",
            successful_hit=False,
        )
    return NewsSearchStepResult(
        feedback=f"SEARCH RESULTS:\n{result}",
        phase="search",
        successful_hit=True,
    )


def _extract_choice_message(chat_response_json: Dict[str, Any]) -> Dict[str, Any]:
    choices = chat_response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    first = choices[0]
    if not isinstance(first, dict):
        return {}
    message = first.get("message")
    return message if isinstance(message, dict) else {}
