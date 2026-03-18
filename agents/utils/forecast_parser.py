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
    action_type: Optional[str]  # "query", "submit", "search", "next", or None
    code: Optional[str]  # For query: the Python code
    forecasts: Optional[List[Dict]]  # For submit: list of {qid, outcomes}
    query: Optional[str]  # For search: the search query
    search_from: Optional[str] = None  # For search: min date (YYYY-MM-DD)
    search_to: Optional[str] = None  # For search: max date (YYYY-MM-DD)
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
    
    elif action_type == "next":
        return ParsedAction(action_type="next", code=None, forecasts=None, 
                           query=None, error=None)
    
    else:
        return ParsedAction(action_type=None, code=None, forecasts=None,
                           query=None,
                           error=f"Unknown action type: '{action_type}'. Valid types: query, submit, search, next")



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
        {name, type, qids, content} and deletes is a list of entry ID strings.
    """
    adds = []
    for match in re.finditer(r'<memory_add>(.*?)</memory_add>', response, re.DOTALL | re.IGNORECASE):
        entry = _parse_memory_add_body(match.group(1).strip())
        if entry:
            adds.append(entry)

    deletes = []
    for match in re.finditer(r'<memory_delete>(.*?)</memory_delete>', response, re.DOTALL | re.IGNORECASE):
        entry_id = match.group(1).strip()
        if entry_id:
            deletes.append(entry_id)

    return adds, deletes


def _parse_memory_add_body(body: str) -> Optional[dict]:
    """
    Parse key: value format inside <memory_add> tags.

    Expected format:
        name: ...
        type: reasoning|calibration|insight|fact
        qids: Q72, Q108
        content: ...

    The content field captures everything after "content:" to end of body.
    Single-line fields (name, type, qids) must NOT span multiple lines.
    """
    if not body:
        return None

    result = {}

    # Extract content first (greedy: everything after "content:" to end of body)
    content_match = re.search(r'(?:^|\n)\s*content\s*:\s*(.*)', body, re.DOTALL | re.IGNORECASE)
    if content_match:
        result["content"] = content_match.group(1).strip()
        # Remove content portion so single-line field regexes don't accidentally match into it
        body_for_fields = body[:content_match.start()]
    else:
        body_for_fields = body

    # Extract single-line fields from the portion BEFORE content:
    # These use re.MULTILINE (not DOTALL) so .+ stays within one line
    for key in ["name", "type", "qids"]:
        pattern = rf'^\s*{key}\s*:\s*(.+)$'
        match = re.search(pattern, body_for_fields, re.MULTILINE | re.IGNORECASE)
        if match:
            result[key] = match.group(1).strip()

    # name and content are required
    if "name" not in result or "content" not in result:
        return None

    # Validate qids: should only contain question IDs (Q followed by numbers, commas, spaces)
    # If it looks like prose, clear it
    qids_val = result.get("qids", "")
    if qids_val and len(qids_val) > 100:
        qids_val = ""
    result["qids"] = qids_val

    return {
        "name": result.get("name", ""),
        "type": result.get("type", "insight"),
        "qids": result.get("qids", ""),
        "content": result.get("content", ""),
    }


def extract_memo_ops(response: str) -> Tuple[List[dict], List[dict], List[str]]:
    """
    Extract memo_df operations from <memo_add>, <memo_update>, <memo_delete> tags.

    Returns:
        (adds, updates, deletes) where:
        - adds: list of dicts with keys {qid, question, memory, confidence, category}
        - updates: list of dicts with keys {qid, memory, confidence, category}
        - deletes: list of qid strings
    """
    adds = []
    for match in re.finditer(r'<memo_add>(.*?)</memo_add>', response, re.DOTALL | re.IGNORECASE):
        entry = _parse_memo_body(match.group(1).strip(), require_question=True)
        if entry:
            adds.append(entry)

    updates = []
    # <memo_update qid="...">...</memo_update>
    for match in re.finditer(r'<memo_update\s+qid\s*=\s*"([^"]*)">(.*?)</memo_update>', response, re.DOTALL | re.IGNORECASE):
        qid = match.group(1).strip()
        entry = _parse_memo_body(match.group(2).strip(), require_question=False)
        if entry and qid:
            entry["qid"] = qid
            updates.append(entry)

    deletes = []
    # <memo_delete qid="..."/> or <memo_delete>QID</memo_delete>
    for match in re.finditer(r'<memo_delete\s+qid\s*=\s*"([^"]*)"\s*/?\s*>', response, re.IGNORECASE):
        qid = match.group(1).strip()
        if qid:
            deletes.append(qid)
    for match in re.finditer(r'<memo_delete>(.*?)</memo_delete>', response, re.DOTALL | re.IGNORECASE):
        qid = match.group(1).strip()
        if qid:
            deletes.append(qid)

    return adds, updates, deletes


def _parse_memo_body(body: str, require_question: bool = True) -> Optional[dict]:
    """
    Parse key: value format inside <memo_add> or <memo_update> tags.

    Expected fields: qid, question, memory, confidence, category.
    The memory field captures everything after "memory:" until the next
    known field (confidence, category) or end of body.
    """
    if not body:
        return None

    result = {}

    # Extract memory: captures everything after "memory:" up to the next
    # known trailing field (confidence or category) or end of body.
    memory_match = re.search(
        r'(?:^|\n)\s*memory\s*:\s*(.*?)(?=\n\s*(?:confidence|category)\s*:|$)',
        body, re.DOTALL | re.IGNORECASE,
    )
    if memory_match:
        result["memory"] = memory_match.group(1).strip()

    # Extract single-line fields from the ENTIRE body (they can appear
    # before or after the memory block).
    for key in ["qid", "question", "confidence", "category"]:
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

    # Parse confidence as float
    conf = None
    if "confidence" in result:
        try:
            conf = float(result["confidence"])
        except (ValueError, TypeError):
            conf = None

    return {
        "qid": result.get("qid", ""),
        "question": result.get("question", ""),
        "memory": result.get("memory", ""),
        "confidence": conf,
        "category": result.get("category", ""),
    }


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
