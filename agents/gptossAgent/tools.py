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
                "description": "Run Python code to inspect the questions DataFrame. Use print(...) to show results.",
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
                "description": "Search the news article database for evidence.",
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
            "description": "Submit forecasts for one or more questions.",
            "strict": True,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "forecasts": {
                        "type": "array",
                        "description": "A list of forecasts to submit.",
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
            "description": "End the day (no more actions).",
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
            "description": "Replace the agent memory with the provided text. Keep it concise.",
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
        calls.append({"name": name, "arguments": args})
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
        if item.get("type") != "reasoning":
            continue
        for c in (item.get("content") or []):
            if not isinstance(c, dict):
                continue
            if c.get("type") in ("reasoning_text", "text"):
                t = c.get("text")
                if isinstance(t, str) and t:
                    chunks.append(t)
    return "\n".join(chunks)


def response_to_action(
    response_json: Dict[str, Any],
) -> Tuple[Optional[ParsedAction], str, List[Dict[str, Any]]]:
    """
    Convert a Responses payload into (ParsedAction|None, assistant_text, tool_calls).

    We prefer the first function_call item as "the action" for this turn.
    """
    assistant_text = _extract_output_text(response_json) or ""
    tool_calls = extract_function_calls(response_json)
    if not tool_calls:
        return None, assistant_text, tool_calls

    call = tool_calls[0]
    name = call.get("name")
    args = call.get("arguments") or {}

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
