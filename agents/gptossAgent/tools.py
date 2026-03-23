import json
from typing import Any, Dict, List, Optional, Tuple


# Responses API tool schema uses:
#   {"type":"function","name":...,"description":...,"parameters":{...},"strict":true}


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
            {
                "type": "function",
                "name": "query_df",
                "description": (
                    "Run Python code to inspect the questions DataFrame before forecasting. "
                    "Use print(...) to show results because plain .head() previews can be unreliable "
                    "outside notebooks. The sandbox predefines df, pd, today, date, datetime, "
                    "timedelta, and standard builtins; import statements are unavailable. "
                    "Example: print(df[df['is_resolved'] == False][['qid','title','answer_type']].head())"
                ),
                "strict": True,
                "parameters": {
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
            }
        )

    if enable_search:
        tools.append(
            {
                "type": "function",
                "name": "search_news",
                "description": (
                    "Search the news article database for evidence before submitting forecasts. "
                    "Use at most one search per turn. "
                    f"{search_results_description} "
                    "You may optionally pass YYYY-MM-DD date filters; to_date cannot be after "
                    "today's date. "
                    "Example: search for 'Fed rate cut 2026 inflation'."
                ),
                "strict": True,
                "parameters": {
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
            }
        )

    tools.append(
        {
            "type": "function",
            "name": "submit_forecasts",
            "description": (
                "Submit exactly one forecast for exactly one active question id. "
                "Each call must contain a single-item forecasts list for one qid only, and you "
                "may submit again later in the same session to update that qid. Use real predicted "
                "answers only; never placeholders like Unknown, TBD, Other, or N/A. Probabilities "
                "must sum to <= 1.0. If the prompt specifies a target question ID, you may submit "
                "only for that question. "
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
        }
    )

    tools.append(
        {
            "type": "function",
            "name": "next_day",
            "description": (
                "End this session with no more actions. Call this only when you are done "
                "querying, searching, and submitting forecasts. Example payload: {}"
            ),
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
                "Replaces the agent memory with the provided text. This is the ONLY context you retain between sessions. "
                "In your next session you have search and the DataFrame, including your final predictions for resolved questions. "
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
