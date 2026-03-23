"""
Offline extraction of Qwen3.5 tool calls from raw assistant text.

Aligned with vLLM's ``Qwen3CoderToolParser`` (``vllm/tool_parsers/qwen3coder_tool_parser.py``):
same regex boundaries and the same XML parse path as ``_parse_xml_function_call`` /
``_get_function_calls`` / ``_convert_param_value`` when a tools schema is supplied.

Used by SkyRL warmup env string ingress; vLLM eval uses structured ``tool_calls`` on the wire.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List, Optional

# --- Regexes copied from vLLM Qwen3CoderToolParser (v0.17.x) ---
_TOOL_CALL_REGEX = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", re.DOTALL)
_TOOL_CALL_FUNCTION_REGEX = re.compile(r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL)
_TOOL_CALL_PARAMETER_REGEX = re.compile(
    r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
    re.DOTALL,
)


def _get_arguments_properties(function_name: str, tools: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    if not tools:
        return {}
    for config in tools:
        if not isinstance(config, dict):
            continue
        if config.get("type") != "function":
            continue
        fn = config.get("function")
        if not isinstance(fn, dict):
            continue
        if fn.get("name") != function_name:
            continue
        params = fn.get("parameters")
        if not isinstance(params, dict):
            return {}
        props = params.get("properties")
        if isinstance(props, dict):
            return props
        return params
    return {}


def _convert_param_value(
    param_value: str,
    param_name: str,
    param_config: Dict[str, Any],
    func_name: str,
) -> Any:
    if param_value.lower() == "null":
        return None

    if param_name not in param_config:
        if param_config:
            # vLLM logs in this case; we keep behavior: raw string
            pass
        return param_value

    spec = param_config[param_name]
    if isinstance(spec, dict) and "type" in spec:
        param_type = str(spec["type"]).strip().lower()
    else:
        param_type = "string"

    if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
        return param_value
    if (
        param_type.startswith("int")
        or param_type.startswith("uint")
        or param_type.startswith("long")
        or param_type.startswith("short")
        or param_type.startswith("unsigned")
    ):
        try:
            return int(param_value)
        except (ValueError, TypeError):
            return param_value
    if param_type.startswith("num") or param_type.startswith("float"):
        try:
            float_param_value = float(param_value)
            return (
                float_param_value
                if float_param_value - int(float_param_value) != 0
                else int(float_param_value)
            )
        except (ValueError, TypeError):
            return param_value
    if param_type in ["boolean", "bool", "binary"]:
        pv = param_value.lower()
        if pv not in ["true", "false"]:
            return False
        return pv == "true"
    if (
        param_type in ["object", "array", "arr"]
        or param_type.startswith("dict")
        or param_type.startswith("list")
    ):
        try:
            return json.loads(param_value)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        try:
            return ast.literal_eval(param_value)
        except (ValueError, SyntaxError, TypeError):
            return param_value
    try:
        return ast.literal_eval(param_value)
    except (ValueError, SyntaxError, TypeError):
        return param_value


def _parse_xml_function_call(function_call_str: str, tools: Optional[List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """Mirror vLLM ``Qwen3CoderToolParser._parse_xml_function_call``."""
    try:
        end_index = function_call_str.index(">")
    except ValueError:
        return None
    function_name = function_call_str[:end_index].strip()
    if not function_name:
        return None
    param_config = _get_arguments_properties(function_name, tools)
    parameters = function_call_str[end_index + 1 :]
    param_dict: Dict[str, Any] = {}
    for match_text in _TOOL_CALL_PARAMETER_REGEX.findall(parameters):
        try:
            idx = match_text.index(">")
        except ValueError:
            continue
        param_name = match_text[:idx].strip()
        param_value = str(match_text[idx + 1 :])
        if param_value.startswith("\n"):
            param_value = param_value[1:]
        if param_value.endswith("\n"):
            param_value = param_value[:-1]
        if not param_name:
            continue
        param_dict[param_name] = _convert_param_value(param_value, param_name, param_config, function_name)

    return {"name": function_name, "arguments": param_dict}


def _get_function_call_strings(model_output: str) -> List[str]:
    """Mirror vLLM ``Qwen3CoderToolParser._get_function_calls`` return shape (inner XML blobs)."""
    matched_ranges = _TOOL_CALL_REGEX.findall(model_output)
    raw_tool_calls = [match[0] if match[0] else match[1] for match in matched_ranges]
    if len(raw_tool_calls) == 0:
        raw_tool_calls = [model_output]

    raw_function_calls: List[str] = []
    for tool_call in raw_tool_calls:
        raw_function_calls.extend(_TOOL_CALL_FUNCTION_REGEX.findall(tool_call))

    return [match[0] if match[0] else match[1] for match in raw_function_calls]


def extract_tool_calls_vllm_qwen3_coder(
    model_output: str,
    *,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Parse ``<tool_call>`` / ``<function=`` / ``<parameter=`` XML the same way vLLM does
    for ``--tool-call-parser qwen3_coder``.

    Returns OpenAI-style dicts: ``{"name", "arguments", "arguments_raw"}`` suitable for
    ``BasicAgent.tool_calls_to_parsed_action``.
    """
    if "<function=" not in (model_output or ""):
        return []

    calls: List[Dict[str, Any]] = []
    for function_call_str in _get_function_call_strings(model_output):
        parsed = _parse_xml_function_call(function_call_str, tools)
        if not parsed:
            continue
        name = parsed.get("name")
        args = parsed.get("arguments")
        if not isinstance(name, str) or not isinstance(args, dict):
            continue
        raw = json.dumps(args, ensure_ascii=False)
        calls.append(
            {
                "name": name,
                "arguments": args,
                "arguments_raw": raw,
            }
        )
    return calls
