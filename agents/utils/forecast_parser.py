"""
Forecast parsing utilities for typed action-based agent responses.

Supports three action types:
- <action type="query">: Execute Python code on DataFrame
- <action type="submit">: Submit forecast predictions
- <action type="next"/>: End the current day

Extracted from BasicAgent for reuse by other agents.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ParsedAction:
    """Result of parsing an action from agent response."""
    action_type: Optional[str]  # "query", "submit", "search", "memory_*", "mem_*", "next", or None
    code: Optional[str]  # For query: the Python code
    forecasts: Optional[List[Dict]]  # For submit: list of {qid, outcomes}
    query: Optional[str]  # For search: the search query
    search_from: Optional[str] = None  # For search: min date (YYYY-MM-DD)
    search_to: Optional[str] = None  # For search: max date (YYYY-MM-DD)
    memory_entry_name: Optional[str] = None  # For memory_retrieve/update/delete: entry name
    memory_add_data: Optional[Dict] = None  # For memory_add: {name, description, content}
    memory_update_data: Optional[Dict] = None  # For memory_update: partial fields
    mem_data: Optional[Dict] = None  # For mem_add/update: {qid, question, memory, category}
    mem_qid: Optional[str] = None  # For mem_update/delete: target qid
    error: Optional[str] = None  # Error message if parsing failed


def parse_answer_probability(response: str) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    """
    Parse an OpenForesight-style single answer response:
      <answer>...</answer>
      <probability>0.5</probability>

    Returns: (answer, prob, error)
    """
    if not response:
        return None, None, "Empty response"

    # Take the LAST occurrence if the model included multiple.
    ans_matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", response, flags=re.IGNORECASE | re.DOTALL)
    prob_matches = re.findall(r"<probability>\s*(.*?)\s*</probability>", response, flags=re.IGNORECASE | re.DOTALL)

    if not ans_matches or not prob_matches:
        return None, None, 'Missing <answer>...</answer> or <probability>...</probability> tags'

    answer = (ans_matches[-1] or "").strip()
    prob_raw = (prob_matches[-1] or "").strip()

    if not answer:
        return None, None, "Empty <answer> value"

    try:
        prob = float(prob_raw)
    except Exception:
        return None, None, f"Invalid probability: {prob_raw!r}"

    if not (0.0 <= prob <= 1.0):
        return None, None, f"Probability must be between 0 and 1, got {prob}"

    return answer, prob, None




def parse_action(response: str, max_outcomes: int = 5) -> ParsedAction:
    """
    Parse a typed action from agent response.
    
    Expected formats:
    
    <action type="query">
    ```python
    df[df['is_resolved'] == False].head()
    ```
    </action>
    
    <action type="submit">
    <forecast qid="Q123">
      <outcome name="Answer1" prob="0.6"/>
    </forecast>
    </action>
    
    <action type="next"/>
    
    Returns:
        ParsedAction with action_type, content, and optional error
    """
    # Check for self-closing <action type="next"/>
    next_pattern = r'<action\s+type=["\']next["\']\s*/>'
    if re.search(next_pattern, response, re.IGNORECASE):
        return ParsedAction(action_type="next", code=None, forecasts=None, 
                           query=None, error=None)

    
    # Look for <action type="...">...</action> (allowing additional attributes after type)
    action_pattern = r'<action\s+type=["\']([^"\']+)["\']([^>]*?)>(.*?)</action>'
    action_match = re.search(action_pattern, response, re.DOTALL | re.IGNORECASE)
    
    if not action_match:
        # Fallback: try to find unclosed action tag (model forgot </action>)
        # Match <action type="...">...until end of response or next tag
        unclosed_pattern = r'<action\s+type=["\']([^"\']+)["\']([^>]*)>(.*?)(?=<(?:action|reasoning|forecast)|$)'
        unclosed_match = re.search(unclosed_pattern, response, re.DOTALL | re.IGNORECASE)
        if unclosed_match:
            action_match = unclosed_match
    
    if not action_match:
        # Fallback: check for legacy formats
        return _parse_legacy_format(response, max_outcomes)
    
    action_type = action_match.group(1).lower().strip()
    extra_attrs = action_match.group(2)  # Additional attributes like from="..." to="..."
    content = action_match.group(3).strip()
    
    if action_type == "query":
        code = _extract_code(content)
        if code:
            return ParsedAction(action_type="query", code=code, forecasts=None, 
                               query=None, error=None)
        else:
            return ParsedAction(action_type="query", code=None, forecasts=None, 
                               query=None,
                               error="No Python code found in query action")
    
    elif action_type == "submit":
        forecasts, error = _parse_forecasts(content, max_outcomes)
        if error:
            return ParsedAction(action_type="submit", code=None, forecasts=None, 
                               query=None, error=error)
        return ParsedAction(action_type="submit", code=None, forecasts=forecasts, 
                           query=None, error=None)
    
    elif action_type == "search":
        # Extract optional from/to date range from extra attributes
        search_from = None
        search_to = None
        if extra_attrs:
            from_match = re.search(r'from=["\']([^"\']+)["\']', extra_attrs)
            to_match = re.search(r'to=["\']([^"\']+)["\']', extra_attrs)
            if from_match:
                search_from = from_match.group(1)
            if to_match:
                search_to = to_match.group(1)
        
        # Content is the search query
        search_query = content.strip()
        if search_query:
            return ParsedAction(action_type="search", code=None, forecasts=None,
                               query=search_query, search_from=search_from, 
                               search_to=search_to, error=None)
        else:
            return ParsedAction(action_type="search", code=None, forecasts=None,
                               query=None, 
                               error="No search query provided")
    
    elif action_type == "memory_retrieve":
        entry_name = content.strip()
        if entry_name:
            return ParsedAction(action_type="memory_retrieve", code=None, forecasts=None,
                               query=None, memory_entry_name=entry_name)
        return ParsedAction(action_type="memory_retrieve", code=None, forecasts=None,
                           query=None, error="No entry name provided for memory_retrieve")

    elif action_type == "memory_add":
        data = _parse_memory_entry_body(content, require_all=True)
        if data:
            return ParsedAction(action_type="memory_add", code=None, forecasts=None,
                               query=None, memory_add_data=data)
        return ParsedAction(action_type="memory_add", code=None, forecasts=None,
                           query=None, error="Could not parse memory_add fields (name, description, content required)")

    elif action_type == "memory_update":
        entry_name = _extract_name_attr(extra_attrs)
        if not entry_name:
            return ParsedAction(action_type="memory_update", code=None, forecasts=None,
                               query=None, error="memory_update requires name attribute: <action type=\"memory_update\" name=\"ENTRY_NAME\">")
        data = _parse_memory_entry_body(content, require_all=False)
        if data:
            return ParsedAction(action_type="memory_update", code=None, forecasts=None,
                               query=None, memory_entry_name=entry_name, memory_update_data=data)
        return ParsedAction(action_type="memory_update", code=None, forecasts=None,
                           query=None, memory_entry_name=entry_name,
                           error="memory_update requires at least one field (description or content)")

    elif action_type == "memory_delete":
        entry_name = content.strip()
        if entry_name:
            return ParsedAction(action_type="memory_delete", code=None, forecasts=None,
                               query=None, memory_entry_name=entry_name)
        return ParsedAction(action_type="memory_delete", code=None, forecasts=None,
                           query=None, error="No entry name provided for memory_delete")

    elif action_type == "mem_add":
        data = _parse_mem_body(content, require_question=True)
        if data:
            return ParsedAction(action_type="mem_add", code=None, forecasts=None,
                               query=None, mem_data=data)
        return ParsedAction(action_type="mem_add", code=None, forecasts=None,
                           query=None, error="Could not parse mem_add fields (qid and memory required)")

    elif action_type == "mem_update":
        qid = _extract_attr(extra_attrs, "qid")
        if not qid:
            return ParsedAction(action_type="mem_update", code=None, forecasts=None,
                               query=None, error='mem_update requires qid attribute: <action type="mem_update" qid="Q123">')
        data = _parse_mem_body(content, require_question=False)
        if data:
            data["qid"] = qid
            return ParsedAction(action_type="mem_update", code=None, forecasts=None,
                               query=None, mem_data=data, mem_qid=qid)
        return ParsedAction(action_type="mem_update", code=None, forecasts=None,
                           query=None, mem_qid=qid, error="mem_update requires at least a memory field")

    elif action_type == "mem_delete":
        qid = content.strip() or _extract_attr(extra_attrs, "qid")
        if qid:
            return ParsedAction(action_type="mem_delete", code=None, forecasts=None,
                               query=None, mem_qid=qid)
        return ParsedAction(action_type="mem_delete", code=None, forecasts=None,
                           query=None, error="No qid provided for mem_delete")

    elif action_type == "next":
        return ParsedAction(action_type="next", code=None, forecasts=None,
                           query=None, error=None)

    else:
        return ParsedAction(action_type=None, code=None, forecasts=None,
                           query=None,
                           error=f"Unknown action type: '{action_type}'. Valid types: query, submit, search, memory_retrieve, memory_add, memory_update, memory_delete, mem_add, mem_update, mem_delete, next")



def _parse_legacy_format(response: str, max_outcomes: int = 5) -> ParsedAction:
    """
    Handle legacy <action>...</action> format for backward compatibility.
    
    Detects intent from content: if has <submit>, parse forecasts; else try code.
    """
    # Check for <action>...</action> without type
    action_pattern = r'<action>(.*?)</action>'
    action_match = re.search(action_pattern, response, re.DOTALL | re.IGNORECASE)
    
    if not action_match:
        # Check if there's an opening tag without closing - give helpful error
        opening_only = re.search(r'<action\s+type=["\'][^"\']+["\']', response, re.IGNORECASE)
        if opening_only:
            return ParsedAction(action_type=None, code=None, forecasts=None, 
                               query=None,
                               error="Found <action> opening tag but no </action> closing tag. Make sure to close your action with </action>.")
        # No action found at all
        return ParsedAction(action_type=None, code=None, forecasts=None, 
                           query=None,
                           error="No valid <action type=\"...\">...</action> block found")

    
    content = action_match.group(1).strip()
    
    # Check if it's a submit action (has <submit> or <forecast> tags)
    if '<submit>' in content.lower() or '<forecast' in content.lower():
        forecasts, error = _parse_forecasts(content, max_outcomes)
        if error:
            return ParsedAction(action_type="submit", code=None, forecasts=None, 
                               query=None, error=error)
        return ParsedAction(action_type="submit", code=None, forecasts=forecasts, 
                           query=None, error=None)
    
    # Try to extract code
    code = _extract_code(content)
    if code:
        return ParsedAction(action_type="query", code=code, forecasts=None, 
                           query=None, error=None)
    
    return ParsedAction(action_type=None, code=None, forecasts=None,
                       query=None,
                       error="Could not determine action type from content")



def _extract_code(content: str) -> Optional[str]:
    """Extract Python code from content."""
    # Look for ```python blocks
    code_pattern = r'```python\s*(.*?)\s*```'
    code_matches = re.findall(code_pattern, content, re.DOTALL)
    if code_matches:
        return code_matches[-1].strip()
    
    # Fallback: if content looks like code (has df or pd references)
    content = content.strip()
    if content and ('df' in content or 'pd.' in content):
        return content
    
    return None


def _parse_forecasts(content: str, max_outcomes: int = 5) -> Tuple[List[Dict], Optional[str]]:
    """
    Parse forecast XML from content.
    
    Expected format:
    <forecast qid="QUESTION_ID">
      <outcome name="Answer1" prob="0.5"/>
      <outcome name="Answer2" prob="0.3"/>
    </forecast>
    
    Returns:
        (forecasts, error_message)
    """
    # Extract <submit>...</submit> block if present
    submit_pattern = r'<submit>(.*?)</submit>'
    submit_match = re.search(submit_pattern, content, re.DOTALL | re.IGNORECASE)
    if submit_match:
        content = submit_match.group(1)
    
    # Parse individual forecasts
    forecast_pattern = r'<forecast\s+qid=["\']([^"\']+)["\']>(.*?)</forecast>'
    forecast_matches = re.findall(forecast_pattern, content, re.DOTALL | re.IGNORECASE)
    
    if not forecast_matches:
        return [], "No valid <forecast qid=\"...\">...</forecast> blocks found"
    
    forecasts = []
    for qid, forecast_content in forecast_matches:
        # Parse outcomes
        outcome_pattern = r'<outcome\s+name=["\']([^"\']+)["\']\s+prob=["\']([^"\']+)["\']'
        outcome_matches = re.findall(outcome_pattern, forecast_content, re.IGNORECASE)
        
        if not outcome_matches:
            return [], f"No valid outcomes found for question {qid}"
        
        if len(outcome_matches) > max_outcomes:
            return [], f"Too many outcomes for {qid}: {len(outcome_matches)} > {max_outcomes}"
        
        outcomes = {}
        for name, prob_str in outcome_matches:
            try:
                prob = float(prob_str)
            except ValueError:
                return [], f"Invalid probability '{prob_str}' for outcome '{name}' in {qid}"
            
            if prob < 0 or prob > 1:
                return [], f"Probability {prob} out of range [0,1] for '{name}' in {qid}"
            
            outcomes[name] = prob
        
        # Check sum
        total = sum(outcomes.values())
        if total > 1.0 + 1e-6:
            return [], f"Probabilities sum to {total:.3f} > 1 for question {qid}"
        
        forecasts.append({'qid': qid, 'outcomes': outcomes})
    
    # Deduplicate by QID (keep last)
    unique_forecasts = {}
    for f in forecasts:
        unique_forecasts[f['qid']] = f
    
    return list(unique_forecasts.values()), None


def _extract_attr(extra_attrs: str, attr_name: str) -> Optional[str]:
    """Extract attr_name="..." from extra attributes on an action tag."""
    if not extra_attrs:
        return None
    match = re.search(rf'{attr_name}=["\']([^"\']+)["\']', extra_attrs)
    return match.group(1).strip() if match else None


def _extract_name_attr(extra_attrs: str) -> Optional[str]:
    """Extract name="..." from extra attributes on an action tag.

    Falls back to id="..." for backward compatibility with old model outputs.
    """
    return _extract_attr(extra_attrs, "name") or _extract_attr(extra_attrs, "id")


def _parse_memory_entry_body(body: str, require_all: bool = True) -> Optional[dict]:
    """
    Parse memory entry fields (name, description, content) from body text.

    Supports two formats:
    1. XML tags:    <name>...</name>  (closing tag optional — auto-closed at next opening tag or end)
    2. Plain-text:  name: ...

    XML is tried first. Small models sometimes omit closing tags, so the parser
    treats each opening tag as a delimiter: content runs until the next ``<field>``
    or end-of-string.

    When require_all=False (for updates), returns result if any field is present.
    """
    if not body:
        return None

    _FIELDS = ("name", "description", "content")
    result = {}

    # --- XML-style: split on opening tags, tolerant of missing closing tags ---
    # Build list of (field, start_of_value) from opening tags in order of appearance
    tag_positions = []
    for field in _FIELDS:
        m = re.search(rf'<{field}\s*>', body, re.IGNORECASE)
        if m:
            tag_positions.append((field, m.end()))
    tag_positions.sort(key=lambda x: x[1])

    if tag_positions:
        for i, (field, val_start) in enumerate(tag_positions):
            # Value ends at: explicit closing tag, next opening tag, or end of body
            if i + 1 < len(tag_positions):
                val_end = tag_positions[i + 1][1]
                # Back up to before the opening tag of the next field
                next_open = re.search(
                    rf'<{tag_positions[i + 1][0]}\s*>',
                    body[val_start:], re.IGNORECASE,
                )
                if next_open:
                    val_end = val_start + next_open.start()
            else:
                val_end = len(body)

            raw = body[val_start:val_end]
            # Strip explicit closing tag if present
            raw = re.sub(rf'</{field}\s*>\s*$', '', raw, flags=re.IGNORECASE)
            result[field] = raw.strip()

    # --- Fall back to plain-text key: value if XML didn't find all fields ---
    if not all(k in result for k in _FIELDS):
        # Extract content first (greedy: everything after "content:" to end of body)
        content_match = re.search(r'(?:^|\n)\s*content\s*:\s*(.*)', body, re.DOTALL | re.IGNORECASE)
        if content_match and "content" not in result:
            result["content"] = content_match.group(1).strip()
            body_before_content = body[:content_match.start()]
        else:
            body_before_content = body

        # Extract description (greedy until "content:" or end)
        if "description" not in result:
            desc_match = re.search(
                r'(?:^|\n)\s*description\s*:\s*(.*?)(?=\n\s*content\s*:|$)',
                body_before_content, re.DOTALL | re.IGNORECASE,
            )
            if desc_match:
                result["description"] = desc_match.group(1).strip()

        # Extract name (single-line)
        if "name" not in result:
            name_match = re.search(r'^\s*name\s*:\s*(.+)$', body_before_content, re.MULTILINE | re.IGNORECASE)
            if name_match:
                result["name"] = name_match.group(1).strip()

    if require_all:
        if not all(k in result for k in _FIELDS):
            return None
    else:
        if not result:
            return None

    return result


def extract_memory(response: str) -> Optional[str]:
    """
    Extract memory content from <memory></memory> tags.

    Returns None if no memory tags found, the content otherwise.
    """
    pattern = r'<memory>(.*?)</memory>'
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def extract_memory_ops(response: str) -> Tuple[List[dict], List[str]]:
    """
    Extract structured memory operations from <memory_add> and <memory_delete> tags.

    Returns:
        (adds, deletes) where adds is a list of dicts with keys
        {name, description, content} and deletes is a list of entry name strings.

    Backward compatible: old-style fields (type, qids) are folded into description.
    """
    adds = []
    for match in re.finditer(r'<memory_add>(.*?)</memory_add>', response, re.DOTALL | re.IGNORECASE):
        entry = _parse_memory_add_body(match.group(1).strip())
        if entry:
            adds.append(entry)

    deletes = []
    for match in re.finditer(r'<memory_delete>(.*?)</memory_delete>', response, re.DOTALL | re.IGNORECASE):
        entry_name = match.group(1).strip()
        if entry_name:
            deletes.append(entry_name)

    return adds, deletes


def _parse_memory_add_body(body: str) -> Optional[dict]:
    """
    Parse key: value format inside <memory_add> tags.

    New format:
        name: ...
        description: ...
        content: ...

    Backward compatible: old-style type/qids fields are folded into description.
    """
    if not body:
        return None

    # Try new 3-field format first
    entry = _parse_memory_entry_body(body, require_all=True)
    if entry:
        return entry

    # Backward compat: try old format with type/qids, convert to new format
    result = {}

    content_match = re.search(r'(?:^|\n)\s*content\s*:\s*(.*)', body, re.DOTALL | re.IGNORECASE)
    if content_match:
        result["content"] = content_match.group(1).strip()
        body_for_fields = body[:content_match.start()]
    else:
        body_for_fields = body

    for key in ["name", "type", "qids"]:
        pattern = rf'^\s*{key}\s*:\s*(.+)$'
        match = re.search(pattern, body_for_fields, re.MULTILINE | re.IGNORECASE)
        if match:
            result[key] = match.group(1).strip()

    if "name" not in result or "content" not in result:
        return None

    # Fold old-style type/qids into description
    desc_parts = []
    if result.get("type"):
        desc_parts.append(f"[{result['type']}]")
    if result.get("qids"):
        desc_parts.append(f"({result['qids']})")
    description = " ".join(desc_parts) if desc_parts else result.get("name", "")

    return {
        "name": result.get("name", ""),
        "description": description,
        "content": result.get("content", ""),
    }


def extract_mem_ops(response: str) -> Tuple[List[dict], List[dict], List[str]]:
    """
    Extract mem_df operations from <mem_add>/<memo_add>, <mem_update>/<memo_update>,
    <mem_delete>/<memo_delete> tags.

    Returns:
        (adds, updates, deletes) where:
        - adds: list of dicts with keys {qid, question, memory, category}
        - updates: list of dicts with keys {qid, memory, category}
        - deletes: list of qid strings
    """
    adds = []
    # Support both <mem_add> and <memo_add> (backward compat)
    for match in re.finditer(r'<mem(?:o)?_add>(.*?)</mem(?:o)?_add>', response, re.DOTALL | re.IGNORECASE):
        entry = _parse_mem_body(match.group(1).strip(), require_question=True)
        if entry:
            adds.append(entry)

    updates = []
    for match in re.finditer(r'<mem(?:o)?_update\s+qid\s*=\s*"([^"]*)">(.*?)</mem(?:o)?_update>', response, re.DOTALL | re.IGNORECASE):
        qid = match.group(1).strip()
        entry = _parse_mem_body(match.group(2).strip(), require_question=False)
        if entry and qid:
            entry["qid"] = qid
            updates.append(entry)

    deletes = []
    for match in re.finditer(r'<mem(?:o)?_delete\s+qid\s*=\s*"([^"]*)"\s*/?\s*>', response, re.IGNORECASE):
        qid = match.group(1).strip()
        if qid:
            deletes.append(qid)
    for match in re.finditer(r'<mem(?:o)?_delete>(.*?)</mem(?:o)?_delete>', response, re.DOTALL | re.IGNORECASE):
        qid = match.group(1).strip()
        if qid:
            deletes.append(qid)

    return adds, updates, deletes


# Backward compat alias
extract_memo_ops = extract_mem_ops


def _parse_mem_body(body: str, require_question: bool = True) -> Optional[dict]:
    """
    Parse key: value format inside <mem_add> or <mem_update> tags.

    Expected fields: qid, question, memory, category.
    The memory field captures everything after "memory:" until the next
    known field (category) or end of body.
    """
    if not body:
        return None

    result = {}

    # Try XML-style tags first: <qid>...</qid>, <memory>...</memory>, etc.
    _MEM_FIELDS = ("qid", "question", "memory", "category")
    tag_positions = []
    for field in _MEM_FIELDS:
        m = re.search(rf'<{field}\s*>', body, re.IGNORECASE)
        if m:
            tag_positions.append((field, m.end()))
    tag_positions.sort(key=lambda x: x[1])

    if tag_positions:
        for i, (field, val_start) in enumerate(tag_positions):
            if i + 1 < len(tag_positions):
                next_open = re.search(
                    rf'<{tag_positions[i + 1][0]}\s*>',
                    body[val_start:], re.IGNORECASE,
                )
                val_end = val_start + next_open.start() if next_open else tag_positions[i + 1][1]
            else:
                val_end = len(body)
            raw = body[val_start:val_end]
            raw = re.sub(rf'</{field}\s*>\s*$', '', raw, flags=re.IGNORECASE)
            result[field] = raw.strip()

    # Fall back to key: value format
    if "memory" not in result:
        memory_match = re.search(
            r'(?:^|\n)\s*memory\s*:\s*(.*?)(?=\n\s*(?:category)\s*:|$)',
            body, re.DOTALL | re.IGNORECASE,
        )
        if memory_match:
            result["memory"] = memory_match.group(1).strip()

    for key in ["qid", "question", "category"]:
        if key not in result:
            pattern = rf'^\s*{key}\s*:\s*(.+)$'
            match = re.search(pattern, body, re.MULTILINE | re.IGNORECASE)
            if match:
                result[key] = match.group(1).strip()

    # memory is required
    if "memory" not in result:
        return None

    # qid is required for adds
    if require_question and "qid" not in result:
        return None

    return {
        "qid": result.get("qid", ""),
        "question": result.get("question", ""),
        "memory": result.get("memory", ""),
        "category": result.get("category", ""),
    }


# Backward compat alias
_parse_memo_body = _parse_mem_body


# Legacy exports for backward compatibility (deprecated)
def extract_action_code(response: str) -> Optional[str]:
    """
    DEPRECATED: Use parse_action() instead.
    
    Extract Python code from <action> tag in agent response.
    """
    result = parse_action(response)
    if result.action_type == "query" and result.code:
        return result.code
    return None


def parse_forecasts(response: str, 
                    max_outcomes: int = 5) -> Tuple[List[Dict], Optional[str]]:
    """
    DEPRECATED: Use parse_action() instead.
    
    Parse XML forecasts from agent response.
    """
    result = parse_action(response, max_outcomes)
    if result.action_type == "submit":
        if result.error:
            return [], result.error
        return result.forecasts or [], None
    elif result.forecasts:
        return result.forecasts, None
    return [], result.error or "No forecasts found"
