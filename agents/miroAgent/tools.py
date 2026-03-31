import json
import re
from typing import Any, Dict, List, Optional, Tuple

from agents.basicAgent.tools import (
    build_action_tools,
    extract_assistant_message,
    extract_assistant_text,
    extract_finish_reason,
    extract_tool_calls,
    tool_calls_to_parsed_action,
)
from agents.utils.forecast_parser import ParsedAction

# Blocks like <use_mcp_tool> ... </use_mcp_tool> (tolerate broken closings in long outputs)
_USE_MCP_TOOL_RE = re.compile(
    r"<use_mcp_tool\b[^>]*>([\s\S]*?)(?:</use_mcp_tool>|(?=<use_mcp_tool\b)|\Z)",
    flags=re.IGNORECASE,
)
_MCP_TOOL_NAME_RE = re.compile(
    r"<tool_name>\s*([^<]+?)\s*</tool_name>",
    flags=re.IGNORECASE | re.DOTALL,
)
_MCP_ARGUMENTS_RE = re.compile(
    r"<arguments>\s*([\s\S]*?)\s*</arguments>",
    flags=re.IGNORECASE,
)

_TOOL_NAME_ALIASES = {
    "next": "next_day",
    "next_day": "next_day",
    "nextday": "next_day",
    "query": "query_df",
    "query_df": "query_df",
    "querydf": "query_df",
    "search": "search_news",
    "search_news": "search_news",
    "searchnews": "search_news",
    "submit": "submit_forecasts",
    "submit_forecasts": "submit_forecasts",
    "submitforecasts": "submit_forecasts",
}

# Recover some upstream-MiroFlow-style names seen in model text (parse-only).
_MCP_LEGACY_TOOL_NAMES = {
    "google_search": "search_news",
    "run_python_code": "query_df",
    "run_command": "query_df",
}


def render_mcp_tool_instructions(tools: List[Dict[str, Any]]) -> str:
    """User-visible instructions aligned with MiroThinker HF model card (MCP + JSON Schema docs)."""
    names: List[str] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function")
        if isinstance(fn, dict):
            n = fn.get("name")
            if isinstance(n, str):
                names.append(n)
    name_list = ", ".join(names) if names else "(none)"
    schema_json = json.dumps(tools, ensure_ascii=True, indent=2)
    return (
        "## MCP TOOL INVOCATION\n"
        "You only have access to the tools described below. Use **at most one** tool per "
        "message; you will receive the tool result in the next user message.\n\n"
        "# Tool-use formatting\n"
        "Tool calls use XML-style tags. Wrap each call in `<use_mcp_tool></use_mcp_tool>`. "
        "Inside, include:\n"
        "- `<server_name>` — use `forecast_sim` for this environment.\n"
        "- `<tool_name>` — must be one of: "
        f"{name_list}.\n"
        "- `<arguments>` — a **single JSON object** with the tool's parameters (valid JSON; "
        "escape quotes inside strings).\n\n"
        "Example shape:\n"
        "<use_mcp_tool>\n"
        "<server_name>forecast_sim</server_name>\n"
        "<tool_name>search_news</tool_name>\n"
        "<arguments>\n"
        '{"query": "example query", "from_date": null, "to_date": null}\n'
        "</arguments>\n"
        "</use_mcp_tool>\n\n"
        "Important:\n"
        "- Place the `<use_mcp_tool>...</use_mcp_tool>` block **at the end** of your reply, "
        "**top-level** (not nested inside other tags).\n\n"
        "## Tool definitions (JSON Schema)\n"
        "Here are the functions in JSON Schema style (names, descriptions, parameters):\n\n"
        f"{schema_json}\n"
    )
def _normalize_mcp_tool_name(raw: str) -> Optional[str]:
    s = (raw or "").strip()
    if not s:
        return None
    key = s.lower().lstrip("/").replace("-", "_")
    if key in _MCP_LEGACY_TOOL_NAMES:
        return _MCP_LEGACY_TOOL_NAMES[key]
    return _TOOL_NAME_ALIASES.get(key)


def _normalize_search_arguments(arguments: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(arguments) if isinstance(arguments, dict) else {}
    if "query" not in out and isinstance(out.get("q"), str):
        out["query"] = out["q"]
    return out


def extract_mcp_tool_calls(output_text: str) -> List[Dict[str, Any]]:
    """Parse <use_mcp_tool> blocks into normalized {name, arguments, arguments_raw} entries."""
    text = output_text or ""
    if not text.strip():
        return []

    calls: List[Dict[str, Any]] = []
    for m in _USE_MCP_TOOL_RE.finditer(text):
        inner = m.group(1) or ""
        tn_m = _MCP_TOOL_NAME_RE.search(inner)
        if not tn_m:
            continue
        name = _normalize_mcp_tool_name(tn_m.group(1))
        if not name:
            continue

        arguments: Dict[str, Any] = {}
        arguments_raw = ""
        arg_m = _MCP_ARGUMENTS_RE.search(inner)
        if arg_m:
            raw = (arg_m.group(1) or "").strip()
            arguments_raw = raw
            if raw:
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict):
                        arguments = parsed
                except Exception:
                    for payload in _extract_json_payloads(raw):
                        if isinstance(payload, dict):
                            arguments = payload
                            break

        if name == "search_news":
            arguments = _normalize_search_arguments(arguments)

        calls.append(
            {
                "name": name,
                "arguments": arguments,
                "arguments_raw": arguments_raw or json.dumps(arguments, ensure_ascii=False),
                "call_id": None,
            }
        )

    return calls


def chat_response_to_action(
    chat_response_json: Dict[str, Any],
) -> Tuple[Optional[ParsedAction], str, List[Dict[str, Any]]]:
    """Prefer MCP text in assistant content; fall back to API tool_calls if absent."""
    assistant_text = extract_assistant_text(chat_response_json)
    tool_calls = extract_mcp_tool_calls(assistant_text)
    if not tool_calls:
        tool_calls = extract_tool_calls(chat_response_json)
    parsed, _, normalized_calls = tool_calls_to_parsed_action(
        tool_calls,
        assistant_text=assistant_text,
    )
    return parsed, assistant_text, normalized_calls


def _extract_json_payloads(text: str) -> List[Any]:
    payloads: List[Any] = []
    decoder = json.JSONDecoder()
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        try:
            payload, end = decoder.raw_decode(text[i:])
        except Exception:
            i += 1
            continue
        payloads.append(payload)
        i += max(1, end)
    return payloads
