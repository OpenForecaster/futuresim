import json
from typing import Any, Dict, List, Optional, Tuple

from agents.utils.forecast_parser import ParsedAction


# Responses API tool schema uses:
#   {"type":"function","name":...,"description":...,"parameters":{...},"strict":true}


def build_action_tools(
    *,
    enable_query: bool,
    enable_search: bool,
    max_outcomes_per_question: int,
) -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = []

    if enable_query:
        tools.append(
            {
                "type": "function",
                "name": "query_df",
                "description": (
                    "Run Python code to inspect the questions DataFrame. Use print(...) to show results. "
                    "Example: print(df[df['is_resolved'] == False][['qid','title']].head(10))"
                ),
                "strict": True,
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute in the provided sandbox. Must use print(...) for outputs.",
                        }
                    },
                    "required": ["code"],
                },
            }
        )

    if enable_search:
        tools.append(
            {
                "type": "function",
                "name": "search_news",
                "description": (
                    "Search the news article database for evidence. "
                    "Example: search for 'Fed rate cut 2026 inflation'."
                ),
                "strict": True,
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query string.",
                        },
                        "from_date": {
                            "type": ["string", "null"],
                            "description": "Optional minimum date (YYYY-MM-DD).",
                        },
                        "to_date": {
                            "type": ["string", "null"],
                            "description": "Optional maximum date (YYYY-MM-DD).",
                        },
                    },
                    "required": ["query"],
                },
            }
        )

    tools.append(
        {
            "type": "function",
            "name": "submit_forecasts",
            "description": (
                "Submit exactly one forecast for exactly one question id. "
                "Example payload: "
                '{"forecasts":[{"qid":"Q123","outcomes":{"Candidate A":0.55,"Candidate B":0.35}}]}'
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "forecasts": {
                        "type": "array",
                        "description": "Single-item list containing exactly one forecast.",
                        "minItems": 1,
                        "maxItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "qid": {"type": "string"},
                                "outcomes": {
                                    "type": "object",
                                    "description": f"Map outcome_name -> probability. Max {max_outcomes_per_question} outcomes per question.",
                                    "additionalProperties": {"type": "number"},
                                },
                            },
                            "required": ["qid", "outcomes"],
                        },
                    }
                },
                "required": ["forecasts"],
            },
        }
    )

    tools.append(
        {
            "type": "function",
            "name": "next_day",
            "description": "End the day (no more actions). Example payload: {}",
            "strict": True,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {},
                "required": [],
            },
        }
    )

    return tools


def build_memory_tools() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": "update_memory",
            "description": (
                "Replaces the agent memory with the provided text. This is the ONLY context you retain between days. "
                "Tomorrow you have search and the DataFrame, but your predictions for resolved questions are deleted on resolution. "
                "Store: (1) reasoning behind predictions and how you did on resolved questions, "
                "(2) performance/calibration patterns across resolutions, "
                "(3) non-obvious insights that search alone would not surface, "
                "(4) critical hard-to-find facts directly relevant to active questions. "
                "Do not store generic advice already in your instructions or easily searchable facts. "
                "Aim for under 2000 characters; drop stale entries."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "memory": {"type": "string"},
                },
                "required": ["memory"],
            },
        }
    ]


def _extract_output_text(response_json: Dict[str, Any]) -> str:
    # vLLM sometimes adds a convenience field.
    if isinstance(response_json.get("output_text"), str):
        return response_json["output_text"]
    chunks: List[str] = []
    for item in (response_json.get("output") or []):
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        for c in (item.get("content") or []):
            if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                t = c.get("text")
                if isinstance(t, str):
                    chunks.append(t)
    return "".join(chunks)


def extract_assistant_messages(response_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract assistant message items with Harmony-like metadata when available.

    Returns list of dicts:
      {"role":"assistant","channel":str|None,"recipient":str|None,"content_type":str|None,"content":str}
    """
    messages: List[Dict[str, Any]] = []
    for item in (response_json.get("output") or []):
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        recipient = item.get("recipient")
        recipient_str = recipient.strip() if isinstance(recipient, str) else None
        if isinstance(recipient_str, str) and recipient_str.startswith("functions."):
            # Tool calls should arrive as `function_call` output items.
            # Skip assistant message echoes that mimic "to=functions.*" headers.
            continue

        content_chunks: List[str] = []
        for c in (item.get("content") or []):
            if not isinstance(c, dict):
                continue
            if c.get("type") in ("output_text", "text"):
                t = c.get("text")
                if isinstance(t, str):
                    content_chunks.append(t)

        content = "".join(content_chunks)
        if not content:
            continue
        messages.append(
            {
                "role": "assistant",
                "channel": item.get("channel") if isinstance(item.get("channel"), str) else None,
                "recipient": recipient_str,
                "content_type": item.get("content_type") if isinstance(item.get("content_type"), str) else None,
                "content": content,
            }
        )
    return messages


def extract_output_items(response_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return raw output items from a Responses payload for replay into next turn.
    """
    out: List[Dict[str, Any]] = []
    for item in (response_json.get("output") or []):
        if isinstance(item, dict):
            out.append(item)
    return out


def extract_replay_items_for_tool_turn(response_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build a strict, parser-safe replay slice for the next tool turn.

    Keep only the item classes recommended for function-calling continuity:
    - reasoning
    - function_call
    - function_call_output

    We intentionally drop generic assistant message items because some vLLM builds
    are brittle when replaying Harmony headers (recipient/constrain) from prior output.
    """
    out: List[Dict[str, Any]] = []
    for item in (response_json.get("output") or []):
        if not isinstance(item, dict):
            continue
        t = item.get("type")
        if t == "reasoning":
            out.append(item)
            continue
        if t == "function_call":
            name = item.get("name")
            arguments = item.get("arguments")
            if not isinstance(name, str) or not isinstance(arguments, str):
                continue
            fc: Dict[str, Any] = {
                "type": "function_call",
                "name": name,
                "arguments": arguments,
            }
            call_id = item.get("call_id")
            if isinstance(call_id, str):
                fc["call_id"] = call_id
            out.append(fc)
            continue
        if t == "function_call_output":
            out_item = {"type": "function_call_output"}
            call_id = item.get("call_id")
            output = item.get("output")
            if isinstance(call_id, str):
                out_item["call_id"] = call_id
            if isinstance(output, str):
                out_item["output"] = output
            if "call_id" in out_item and "output" in out_item:
                out.append(out_item)
            continue
    return out


def extract_function_calls(response_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for item in (response_json.get("output") or []):
        if not isinstance(item, dict):
            continue
        if item.get("type") != "function_call":
            continue
        name = item.get("name")
        args_raw = item.get("arguments")
        if not isinstance(name, str) or not isinstance(args_raw, str):
            continue
        try:
            args = json.loads(args_raw) if args_raw else {}
        except Exception:
            args = {}
        call_id = item.get("call_id") or item.get("id")
        calls.append(
            {
                "name": name,
                "arguments": args,
                "arguments_raw": args_raw,
                "call_id": call_id if isinstance(call_id, str) else None,
            }
        )
    return calls


def extract_reasoning_text(response_json: Dict[str, Any]) -> str:
    """
    Extract reasoning text from Responses API output items, if present.

    vLLM/OpenAI-style reasoning items are typically:
      {"type":"reasoning","content":[{"type":"reasoning_text","text":"..."}], ...}
    """
    chunks: List[str] = []
    for item in (response_json.get("output") or []):
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "reasoning":
            for c in (item.get("content") or []):
                if not isinstance(c, dict):
                    continue
                if c.get("type") in ("reasoning_text", "text"):
                    t = c.get("text")
                    if isinstance(t, str) and t:
                        chunks.append(t)
        elif item_type == "message":
            # Some implementations expose analysis as assistant message channel.
            channel = item.get("channel")
            if channel != "analysis":
                continue
            for c in (item.get("content") or []):
                if not isinstance(c, dict):
                    continue
                if c.get("type") in ("output_text", "text", "reasoning_text"):
                    t = c.get("text")
                    if isinstance(t, str) and t:
                        chunks.append(t)
    return "\n".join(chunks)


def extract_reasoning_token_count(response_json: Dict[str, Any]) -> int:
    """
    Extract per-response reasoning token count from usage details when available.
    """
    usage = response_json.get("usage") or {}
    if not isinstance(usage, dict):
        return 0

    # Responses-style usage
    out_details = usage.get("output_tokens_details")
    if isinstance(out_details, dict):
        rt = out_details.get("reasoning_tokens")
        if isinstance(rt, int):
            return rt
        try:
            return int(rt)
        except Exception:
            pass

    # ChatCompletions-style usage
    comp_details = usage.get("completion_tokens_details")
    if isinstance(comp_details, dict):
        rt = comp_details.get("reasoning_tokens")
        if isinstance(rt, int):
            return rt
        try:
            return int(rt)
        except Exception:
            pass

    rt = usage.get("reasoning_tokens")
    if isinstance(rt, int):
        return rt
    try:
        return int(rt)
    except Exception:
        return 0


def response_to_action(
    response_json: Dict[str, Any],
    *,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Optional[ParsedAction], str, List[Dict[str, Any]]]:
    """
    Convert a Responses payload into (ParsedAction|None, assistant_text, tool_calls).

    We prefer the first function_call item as "the action" for this turn.
    """
    assistant_text = _extract_output_text(response_json) or ""
    if tool_calls is None:
        tool_calls = extract_function_calls(response_json)
    if not tool_calls:
        return None, assistant_text, tool_calls

    call = tool_calls[0]
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
            tool_calls,
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
            tool_calls,
        )

    if name == "next_day":
        return ParsedAction(action_type="next", code=None, forecasts=None, query=None, error=None), assistant_text, tool_calls

    if name == "query_df":
        code = args.get("code")
        if not isinstance(code, str) or not code.strip():
            return ParsedAction(action_type="query", code=None, forecasts=None, query=None, error="Missing code"), assistant_text, tool_calls
        return ParsedAction(action_type="query", code=code, forecasts=None, query=None, error=None), assistant_text, tool_calls

    if name == "search_news":
        q = args.get("query")
        if not isinstance(q, str) or not q.strip():
            return ParsedAction(action_type="search", code=None, forecasts=None, query=None, error="Missing query"), assistant_text, tool_calls
        frm = args.get("from_date")
        to = args.get("to_date")
        search_from = frm if isinstance(frm, str) and frm.strip() else None
        search_to = to if isinstance(to, str) and to.strip() else None
        return (
            ParsedAction(
                action_type="search",
                code=None,
                forecasts=None,
                query=q.strip(),
                search_from=search_from,
                search_to=search_to,
                error=None,
            ),
            assistant_text,
            tool_calls,
        )

    if name == "submit_forecasts":
        forecasts = args.get("forecasts")
        if not isinstance(forecasts, list) or not forecasts:
            return ParsedAction(action_type="submit", code=None, forecasts=None, query=None, error="Missing forecasts"), assistant_text, tool_calls
        out: List[Dict[str, Any]] = []
        for f in forecasts:
            if not isinstance(f, dict):
                continue
            qid = f.get("qid")
            outcomes = f.get("outcomes")
            if not isinstance(qid, str) or not isinstance(outcomes, dict):
                continue
            # Keep only numeric probabilities.
            clean_outcomes = {}
            for k, v in outcomes.items():
                if not isinstance(k, str):
                    continue
                try:
                    fv = float(v)
                except Exception:
                    continue
                clean_outcomes[k] = fv
            out.append({"qid": qid, "outcomes": clean_outcomes})
        if not out:
            return ParsedAction(action_type="submit", code=None, forecasts=None, query=None, error="No valid forecasts"), assistant_text, tool_calls
        return ParsedAction(action_type="submit", code=None, forecasts=out, query=None, error=None), assistant_text, tool_calls

    # Unknown tool name -> treat as invalid.
    return ParsedAction(action_type=None, code=None, forecasts=None, query=None, error=f"Unknown tool call: {name!r}"), assistant_text, tool_calls
