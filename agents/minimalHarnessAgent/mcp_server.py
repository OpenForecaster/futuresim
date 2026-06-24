"""
MCP server for MinimalHarnessAgent backends.

Provides core forecast tools (search_news, submit_forecasts, next_day) plus
optional active_memory2 memory tools. Communicates with the driver via
file-based signals.

IMPORTANT: All heavy imports (lancedb, pandas, torch, etc.) are deferred to
first use. The MCP handshake must complete quickly or the CLI backend may give
up before tools are available.
"""

import argparse
import contextlib
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP


# ── helpers ────────────────────────────────────────────────────────────

def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, default=str)


def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    return datetime.strptime(s, "%Y-%m-%d").date()


def _current_submission_snapshot() -> list:
    by_qid = {}
    for pred in _today_predictions:
        qid = pred.get("question_id", pred.get("qid", ""))
        if qid:
            by_qid[str(qid)] = pred
    return list(by_qid.values())


# ── server ─────────────────────────────────────────────────────────────

mcp = FastMCP("futuresim")

# Global state — set in main(), lazily initialized on first tool call.
_state: dict = {}
_workspace: Path = Path(".")     # CLI workspace (market.csv, articles/, memory/)
_internal_dir: Path = Path(".")  # Driver coordination (state.json, signals/)
_search_handler = None       # Set by _ensure_search()
_search_tool = None           # Set by _ensure_search()
_today_predictions: list = []
_search_cutoff_days: int = 0
_search_initialized: bool = False
_cli_args = None              # Parsed CLI args, used by _ensure_search()
# Handholding version controls how much "stay-on-it" guidance next_day appends.
# - "v1": minimal (no extra reminders)
# - "v2": adds "questions still active, revise stale forecasts" message
# - "v3": v2 + "questions resolving tomorrow" reminder
# Default "v1" matches state before commit dadfc2f.
_handholding_version: str = "v1"
_HANDHOLDING_VERSIONS = ("v1", "v2", "v3")

# active_memory2 mode: server-side memory store. Lazily instantiated on first
# memory tool call or next_day, then persisted read-only for that session date.
_memory_tools_enabled: bool = False
_agent_id: str = "agent"
_active_memory = None  # type: ignore[assignment]
_memory_phase_active: bool = False
_session_touched_qids: set[str] = set()
_session_forecast_qids: set[str] = set()

# Cumulative scoring tracker (mirrors BasicAgent's FeedbackHandler).
_cumulative = {
    "brier_sum": 0.0,
    "tw_peer_sum": 0.0,
    "accuracy_count": 0,
    "resolved_count": 0,
    "processed_qids": set(),
}


def _reload_state() -> None:
    """Re-read state.json from internal dir and update search handler date."""
    global _state, _today_predictions, _session_touched_qids, _session_forecast_qids
    _state = _read_json(_internal_dir / "state.json")
    _today_predictions = []
    _session_touched_qids = set()
    _session_forecast_qids = set()
    if _search_handler is not None:
        search_date = _state.get("search_current_date") or _state["current_date"]
        _search_handler.set_date(_parse_date(search_date))


def _build_new_articles_message(previous_date: Optional[date], current_date: Optional[date]) -> str:
    """Match BasicAgent's article-count wording when search can provide a count."""
    if previous_date is None or current_date is None:
        return (
            "New articles are available in articles/ or via the search_news MCP tool."
        )

    search_date = _parse_date(_state.get("search_current_date")) or current_date
    max_date = search_date - timedelta(days=_search_cutoff_days)
    if search_date < current_date or previous_date > max_date:
        return (
            f"No new articles are available beyond the current search date "
            f"{max_date.isoformat()}; search_news and articles/ remain capped there."
        )

    with contextlib.redirect_stdout(sys.stderr):
        _ensure_search()
    if _search_handler is not None and _search_handler.is_available:
        with contextlib.redirect_stdout(sys.stderr):
            count = _search_handler.count_articles(
                min_date=previous_date,
                max_date=max_date,
            )
        if count is not None:
            return (
                f"{count:,} new articles have been published since your last update "
                "and are available via the search_news tool. New date "
                "directories are also now present in articles/."
            )

    return (
        f"New articles are available for {current_date.isoformat()} in articles/ "
        "or via the search_news tool."
    )


def _ensure_search() -> None:
    """Lazily initialize the search backend on first use.

    This keeps the heavy imports (lancedb, sentence-transformers, etc.) out of
    the startup path so the MCP handshake completes quickly.
    """
    global _search_initialized, _search_handler, _search_tool

    if _search_initialized:
        return
    _search_initialized = True

    args = _cli_args
    if args is None:
        return

    # Add repo root to path (may not have been done if --repo-root was given).
    if args.repo_root and args.repo_root not in sys.path:
        sys.path.insert(0, args.repo_root)

    from futuresim_agents.search_tools.handler import SearchHandler

    search_backend = (
        args.search_backend
        or os.environ.get("FSIM_SEARCH_BACKEND", "")
        or _state.get("search_backend", "")
    ).strip().lower()

    search_db = args.search_db or os.environ.get("FSIM_SEARCH_DB", _state.get("search_db", ""))
    if not search_backend:
        search_backend = "lancedb" if search_db else ""

    if search_backend == "openreward":
        from futuresim_agents.search_tools.openreward import OpenRewardSearchTool

        _search_tool = OpenRewardSearchTool.from_env(
            search_url=(
                args.openreward_search_url
                or os.environ.get("FSIM_OPENREWARD_SEARCH_URL", "")
                or _state.get("openreward_search_url", "")
                or "https://search.openreward.ai/search"
            ),
            fetch_url=(
                args.openreward_fetch_url
                or os.environ.get("FSIM_OPENREWARD_FETCH_URL", "")
                or _state.get("openreward_fetch_url", "")
                or "https://search.openreward.ai/fetch"
            ),
        )
    else:
        embedding_model_path = args.embedding_model or os.environ.get("FSIM_EMBEDDING_MODEL", _state.get("embedding_model", ""))
        if not search_db or not os.path.exists(search_db):
            return

        from futuresim_agents.search_tools.lancedb.store import LanceDBSearchTool

        embedding_model = None
        if args.embedding_server_url:
            embedding_model = _VLLMEmbeddingClient(args.embedding_server_url, embedding_model_path)

        _search_tool = LanceDBSearchTool(
            search_db,
            embedding_model=embedding_model,
            model_path=embedding_model_path if not embedding_model else None,
        )
    _search_handler = SearchHandler(
        search_tool=_search_tool,
        search_cutoff_days=_search_cutoff_days,
    )
    search_date = _state.get("search_current_date") or _state["current_date"]
    _search_handler.set_date(_parse_date(search_date))


# ── embedding client ───────────────────────────────────────────────────

class _EmbeddingOutput:
    """Mimics vLLM EmbeddingRequestOutput for LanceDBSearchTool compatibility."""
    def __init__(self, embedding):
        self.outputs = type("obj", (), {"embedding": embedding})()


class _VLLMEmbeddingClient:
    """Lightweight client that connects to an already-running vLLM embedding server."""

    def __init__(self, server_url: str, model_name: str = ""):
        self._url = server_url.rstrip("/")
        if self._url.endswith("/v1"):
            self._url = self._url[:-3].rstrip("/")
        self._model = model_name

    def embed(self, texts: list, use_tqdm: bool = False) -> list:
        import urllib.request
        import json as _json
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        payload = _json.dumps({"input": texts, "model": self._model}).encode()
        req = urllib.request.Request(
            f"{self._url}/v1/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with opener.open(req, timeout=30) as resp:
            data = _json.loads(resp.read())
        return [_EmbeddingOutput(item["embedding"]) for item in data["data"]]


# ── active_memory2 helpers ─────────────────────────────────────────────

def _ensure_active_memory():
    """Lazy-init the server-side ActiveMemory and load the prior day's snapshot.

    Called from every memory tool and from next_day before saving. Returns the
    instance, or None when memory tools aren't enabled.
    """
    global _active_memory
    if not _memory_tools_enabled:
        return None
    if _active_memory is not None:
        return _active_memory

    from futuresim_agents.utils.memory import ActiveMemory

    _active_memory = ActiveMemory(_agent_id, memory_dir=str(_workspace), max_entries=500)
    cur = _parse_date(_state.get("current_date"))
    if cur is not None:
        _active_memory.set_date(cur)
    return _active_memory


def _save_active_memory_for_today() -> None:
    """Persist mem.csv + meta.yaml for the current sim date and chmod read-only.

    No-op when memory tools are disabled.
    Files are written to workspace/memory/{current_date}/{mem.csv,meta.yaml};
    after writing, both files are chmod'd to 0o444 and the date directory to
    0o555 so the harness physically can't overwrite or delete the snapshot.
    """
    am = _ensure_active_memory()
    if am is None:
        return
    cur = _parse_date(_state.get("current_date"))
    if cur is None:
        return
    am.save(cur)
    date_dir = Path(_workspace) / "memory" / str(cur)
    if not date_dir.exists():
        return
    for fname in ("mem.csv", "meta.yaml"):
        f = date_dir / fname
        if f.exists():
            try:
                os.chmod(f, 0o444)
            except OSError:
                pass
    try:
        os.chmod(date_dir, 0o555)
    except OSError:
        pass


def _build_feedback_recap(previous_label: str, active_count: int, *, mutate: bool) -> str:
    """Render AllQ-style resolved-question and cumulative-performance recap."""
    events = _state.get("resolution_events", [])
    new_events = [
        ev for ev in events
        if ev.get("qid") not in _cumulative["processed_qids"]
    ]
    if not new_events:
        return ""

    lines = [f"## RESULTS SINCE YOUR LAST SESSION ({previous_label})", ""]
    brier_sum = _cumulative["brier_sum"]
    tw_peer_sum = _cumulative["tw_peer_sum"]
    accuracy_count = _cumulative["accuracy_count"]
    resolved_count = _cumulative["resolved_count"]

    for ev in new_events:
        qid = ev.get("qid", ev.get("question_id", "?"))
        title = ev.get("title", "")
        gt = ev.get("ground_truth", "?")
        agents_data = ev.get("agents", {})
        my_stats = None
        for _, stats in agents_data.items():
            if isinstance(stats, dict):
                my_stats = stats
                break

        if my_stats:
            brier = float(my_stats.get("brier", 0))
            tw_peer = float(my_stats.get("tw_peer", 0))
            best_outcome = my_stats.get("best_outcome", "?")
            best_prob = float(my_stats.get("best_prob", 0))
            is_accurate = my_stats.get("is_accurate", False)
            brier_sum += brier
            tw_peer_sum += tw_peer
            resolved_count += 1
            if is_accurate:
                accuracy_count += 1

            agent_preds = _state.get("agent_predictions", {})
            pred_dist = agent_preds.get(str(qid), {})
            if pred_dist:
                dist_items = sorted(pred_dist.items(), key=lambda kv: -kv[1])
                dist_str = ", ".join(f"{o}: {p:.2f}" for o, p in dist_items)
                dist_str = "{" + dist_str + "}"
            else:
                dist_str = f"{{{best_outcome}: {best_prob:.2f}}}"

            lines.append(
                f"- \"{title}\"\n"
                f"  Your prediction distribution: {dist_str} | Truth: {gt}\n"
                f"  Brier: {brier:+.2f} | TW-Score: {tw_peer:+.2f}"
            )
        else:
            resolved_count += 1
            lines.append(f"- \"{title}\" → {gt}")

        if mutate:
            _cumulative["processed_qids"].add(qid)

    denom = resolved_count + active_count
    total_preds = _state.get("total_predictions", 0)
    if denom > 0:
        avg_brier = brier_sum / denom
        accuracy = accuracy_count / denom * 100
        lines.extend([
            "",
            "## YOUR CUMULATIVE PERFORMANCE TILL TODAY",
            f"- Total Predictions: {total_preds} ({resolved_count} resolved, {active_count} active)",
            f"- accuracy: {accuracy:.1f}% | brier skill score: {avg_brier:.3f} | time weighted score: {tw_peer_sum:.2f}",
            "  accuracy = fraction of ALL questions (resolved + active) where your top outcome matched the truth (0 credit for questions you did not predict or that have not resolved); "
            "brier skill score = mean brier skill score across ALL questions (0 for questions you did not predict or that have not resolved); "
            "time weighted score = sum of brier skill scores across all resolved questions, across all days you held your respective predictions",
        ])

    if mutate:
        _cumulative["brier_sum"] = brier_sum
        _cumulative["tw_peer_sum"] = tw_peer_sum
        _cumulative["accuracy_count"] = accuracy_count
        _cumulative["resolved_count"] = resolved_count

    return "\n".join(lines)


# ── tools ──────────────────────────────────────────────────────────────

@mcp.tool()
def search_news(
    query: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> str:
    """Search news articles (semantic + keyword hybrid). Results are date-capped to the current simulation date.

    Args:
        query: Search query string.
        from_date: Optional earliest article date (YYYY-MM-DD).
        to_date: Optional latest article date (YYYY-MM-DD). Capped to sim date.
    """
    with contextlib.redirect_stdout(sys.stderr):
        _ensure_search()

    if _search_handler is None or not _search_handler.is_available:
        return "Search is not available for this simulation run."

    search_type = _state.get("search_type", "hybrid")
    with contextlib.redirect_stdout(sys.stderr):
        result_text, err = _search_handler.search(
            query,
            max_results=5,
            search_type=search_type,
            min_date=_parse_date(from_date),
            max_date=_parse_date(to_date),
        )
    if err:
        return f"Search error: {err}"
    return result_text or "No articles found matching your query."


@mcp.tool()
def submit_forecasts(question_id: str, outcomes: dict[str, float]) -> str:
    """Submit a probabilistic prediction for a forecasting question.

    Args:
        question_id: The qid of the question to predict on.
        outcomes: A dict mapping outcome strings to probabilities, e.g. {"Yes": 0.7, "No": 0.3}. Must sum to <= 1.0.
    """
    if _memory_tools_enabled and _memory_phase_active:
        return (
            "Error: you are in the memory-update phase. Do not submit more forecasts now; "
            "update memory and call next_day() again to advance."
        )

    question_id = str(question_id).strip()
    active_qids = {
        str(q.get("qid"))
        for q in _state.get("questions", [])
        if isinstance(q, dict) and q.get("qid") is not None
    }
    if active_qids and question_id not in active_qids:
        return f"Error: question_id {question_id!r} is not an active question in market.csv."

    if not isinstance(outcomes, dict) or not outcomes:
        return "Error: outcomes must be a non-empty object mapping outcome names to probabilities."

    try:
        max_outcomes = int(_state.get("max_outcomes_per_question", 5) or 5)
    except (TypeError, ValueError):
        max_outcomes = 5
    if len(outcomes) > max_outcomes:
        return f"Error: {len(outcomes)} outcomes exceeds maximum of {max_outcomes} per question."

    normalized_outcomes = {}
    for outcome, prob in outcomes.items():
        try:
            value = float(prob)
        except (TypeError, ValueError):
            return f"Error: probability for {outcome!r} must be numeric."
        if value < 0 or value > 1:
            return f"Error: probability {value} for '{outcome}' is out of [0, 1] range."
        normalized_outcomes[str(outcome)] = value

    total = sum(normalized_outcomes.values())
    if total > 1.0 + 1e-6:
        return f"Error: probabilities sum to {total:.4f} which exceeds 1.0."

    pred = {"question_id": question_id, "outcomes": normalized_outcomes}
    _today_predictions.append(pred)
    _session_touched_qids.add(question_id)
    _session_forecast_qids.add(question_id)

    if _state.get("submit_ends_session"):
        cur_date = _state["current_date"]
        signal_dir = _internal_dir / "signals"
        signal_dir.mkdir(parents=True, exist_ok=True)
        _write_json(signal_dir / f"next_day_{cur_date}.json", {
            "predictions": _current_submission_snapshot(),
            "date": cur_date,
            "reason": "submit_ends_session",
        })
        return f"Prediction recorded for question {question_id}: {normalized_outcomes}. Session complete."

    return f"Prediction recorded for question {question_id}: {normalized_outcomes}"


@mcp.tool()
def next_day() -> str:
    """Signal that you are done with the current day. Blocks until the simulation advances.

    Returns context for the new day: new date, resolution events, and whether the simulation is complete.
    You may call this after submitting forecasts, or with zero forecasts if you choose not to update today.
    """
    global _memory_phase_active
    cur_date = _state["current_date"]

    if _memory_tools_enabled and not _memory_phase_active:
        from futuresim_agents.minimalHarnessAgent.prompts.prompt_active_memory2 import (
            build_memory_update_prompt,
        )

        am = _ensure_active_memory()
        _memory_phase_active = True
        if am is None:
            return (
                "Entered memory-update phase. Use memory tools to update notes, then "
                "call next_day() again to advance."
            )
        return build_memory_update_prompt(
            current_date=_parse_date(cur_date) or date.today(),
            mem_summary=am.mem_summary(expanded_qids=_session_touched_qids),
            meta_index=am.get_index() or "",
            mem_count=am.mem_count,
            meta_count=am.entry_count,
            max_meta_entries=am._max_entries,
            resolution_recap=_build_feedback_recap(cur_date, len(_state.get("questions", [])), mutate=False),
            touched_qids=sorted(_session_touched_qids),
        )

    # active_memory2: persist the day's mem.csv + meta.yaml (and chmod read-only)
    # before the driver advances. No-op when memory tools aren't enabled.
    _save_active_memory_for_today()

    signal_dir = _internal_dir / "signals"
    signal_dir.mkdir(parents=True, exist_ok=True)
    _write_json(signal_dir / f"next_day_{cur_date}.json", {
        "predictions": _current_submission_snapshot(),
        "date": cur_date,
    })

    if _state.get("next_day_returns_immediately"):
        return "Day complete. The simulator will advance after this session exits."

    # Poll for continue signal, consuming it so the next next_day call blocks.
    poll_interval = 0.5
    timeout = float(_state.get("timeout_seconds", 3600))
    start = time.time()
    while True:
        for p in signal_dir.glob("continue_*.json"):
            cont = _read_json(p)
            p.unlink(missing_ok=True)
            _reload_state()
            _memory_phase_active = False
            status = cont.get("status", "day_advanced")
            new_date = _state.get("current_date", "unknown")
            new_date_obj = _parse_date(new_date)

            response_parts = [
                f"Day advanced to {new_date}.",
                _build_new_articles_message(_parse_date(cur_date), new_date_obj),
            ]

            # Handholding: v2/v3 append a "still-active" reminder; v3 also
            # appends an "imminent (resolves-tomorrow)" reminder.
            if _handholding_version in ("v2", "v3"):
                active_count = len(_state.get("questions", []))
                response_parts.append(
                    f"{active_count} question(s) are still active. Re-read market.csv, "
                    "scan today's news, and resubmit any forecast where new evidence "
                    "has shifted your view before calling next_day again. A forecast "
                    "is never \"done\" while its question is still active."
                )
            if _handholding_version == "v3":
                tomorrow_iso = (
                    (new_date_obj + timedelta(days=1)).isoformat()
                    if new_date_obj else None
                )
                imminent_qids = [
                    q.get("qid") for q in _state.get("questions", [])
                    if tomorrow_iso and q.get("resolution_date") == tomorrow_iso
                ]
                imminent_msg = (
                    f"**IMPORTANT**: {len(imminent_qids)} question(s) resolve tomorrow ({tomorrow_iso}): "
                    f"{imminent_qids}. Make sure your prediction on each is up-to-date before calling next_day — "
                    "stale forecasts might hurt your performance."
                    if imminent_qids else
                    f"No questions resolve tomorrow ({tomorrow_iso}), but still scan for ones "
                    "resolving soon (check the resolution_date column in market.csv)."
                )
                response_parts.append(imminent_msg)

            feedback_recap = _build_feedback_recap(f"{cur_date} -> {new_date}", len(_state.get("questions", [])), mutate=True)
            if feedback_recap:
                response_parts.append("\n" + feedback_recap)

            if status == "simulation_complete":
                response_parts.append("\nSimulation is complete. No more days to process.")

            return "\n".join(response_parts)

        if time.time() - start > timeout:
            return "Error: timed out waiting for simulation to advance."
        time.sleep(poll_interval)


# ── active_memory2 memory tools ────────────────────────────────────────

def _register_memory_tools() -> None:
    """Register the 7 ActiveMemory MCP tools. Called from main() only when
    --enable-memory-tools is set, so the tool list stays minimal otherwise.

    Mirrors the chat-tool surface that AllQAgent uses with ActiveMemory:
      mem_add / mem_update / mem_delete           — per-question mem_df rows
      memory_retrieve / memory_new / memory_update / memory_delete — meta-insights
    All mutations stay in-memory until next_day, when the day's snapshot is
    written to workspace/memory/{date}/{mem.csv,meta.yaml} and chmod'd 0o444.
    """

    @mcp.tool()
    def mem_add(qid: str, question: str, memory: str, category: str = "") -> str:
        """Add or upsert a per-question memory entry in mem_df.

        Args:
            qid: Question ID.
            question: Short question title for context.
            memory: Reasoning, evidence, or prediction rationale (truncated to 1000 chars).
            category: Optional topic category (e.g. 'politics', 'sports').
        """
        am = _ensure_active_memory()
        if am is None:
            return "Error: memory tools are not enabled for this run."
        am.mem_add(qid=qid, question=question, memory=memory, category=category or "")
        _session_touched_qids.add(str(qid).strip())
        return f"mem_df row {qid!r}: added/updated."

    @mcp.tool()
    def mem_update(qid: str, memory: str, category: str = "") -> str:
        """Update an existing mem_df entry's memory (and optionally category).

        If the qid doesn't exist yet, this falls back to an add with an empty question.
        """
        am = _ensure_active_memory()
        if am is None:
            return "Error: memory tools are not enabled for this run."
        am.mem_update(qid=qid, memory=memory, category=(category or None))
        _session_touched_qids.add(str(qid).strip())
        return f"mem_df row {qid!r}: updated."

    @mcp.tool()
    def mem_delete(qid: str) -> str:
        """Delete a per-question mem_df entry by qid."""
        am = _ensure_active_memory()
        if am is None:
            return "Error: memory tools are not enabled for this run."
        _session_touched_qids.add(str(qid).strip())
        if am.mem_delete(qid):
            return f"mem_df row {qid!r}: deleted."
        return f"mem_df row {qid!r}: not found."

    @mcp.tool()
    def memory_retrieve(name: str) -> str:
        """Fetch the full content of a meta-insight entry by name."""
        am = _ensure_active_memory()
        if am is None:
            return "Error: memory tools are not enabled for this run."
        result = am.retrieve(name)
        if result is None:
            return f"Meta-insight {name!r}: not found."
        return result

    @mcp.tool()
    def memory_new(name: str, description: str, content: str) -> str:
        """Create a new meta-insight entry (cross-question lessons / patterns).

        Args:
            name: Short, lowercase-hyphenated name (primary key).
            description: One-line summary shown in the index.
            content: Full insight, reasoning chain, or pattern.
        """
        am = _ensure_active_memory()
        if am is None:
            return "Error: memory tools are not enabled for this run."
        try:
            stored = am.add_entry(name=name, description=description, content=content)
            return f"Meta-insight {stored!r}: created."
        except ValueError as e:
            return f"Error: {e}"

    @mcp.tool()
    def memory_update(name: str, description: str = "", content: str = "") -> str:
        """Update an existing meta-insight entry. Provide only the fields you want to change."""
        am = _ensure_active_memory()
        if am is None:
            return "Error: memory tools are not enabled for this run."
        kwargs = {}
        if description:
            kwargs["description"] = description
        if content:
            kwargs["content"] = content
        if not kwargs:
            return "Error: provide at least one of description or content."
        if not am.update_entry(name, **kwargs):
            return f"Meta-insight {name!r}: not found."
        return f"Meta-insight {name!r}: updated."

    @mcp.tool()
    def memory_delete(name: str) -> str:
        """Delete a meta-insight entry by name."""
        am = _ensure_active_memory()
        if am is None:
            return "Error: memory tools are not enabled for this run."
        if not am.delete_entry(name):
            return f"Meta-insight {name!r}: not found."
        return f"Meta-insight {name!r}: deleted."


# ── main ───────────────────────────────────────────────────────────────

def main():
    global _workspace, _internal_dir, _search_cutoff_days, _cli_args, _handholding_version
    global _memory_tools_enabled, _agent_id

    parser = argparse.ArgumentParser(description="Forecast-sim MCP server")
    parser.add_argument("--config-file", default="",
                        help="JSON file containing MCP server CLI options.")
    parser.add_argument("--workspace", default="", help="Path to CLI workspace")
    parser.add_argument("--internal-dir", default="", help="Path to internal dir (state.json, signals/)")
    parser.add_argument("--repo-root", default="", help="Repo root to add to sys.path")
    parser.add_argument("--search-backend", default="", help="Search backend: lancedb or openreward")
    parser.add_argument("--search-db", default="", help="Path to LanceDB search index")
    parser.add_argument("--embedding-model", default="", help="Path to embedding model")
    parser.add_argument("--openreward-search-url", default="", help="OpenReward search endpoint override")
    parser.add_argument("--openreward-fetch-url", default="", help="OpenReward fetch endpoint override")
    parser.add_argument("--embedding-server-url", default="",
                        help="URL of an already-running vLLM embedding server (e.g. http://127.0.0.1:8001)")
    parser.add_argument(
        "--handholding-version",
        choices=_HANDHOLDING_VERSIONS,
        default="v1",
        help="How much guidance next_day appends: v1 (minimal), v2 (adds still-active nudge), v3 (v2 + imminent reminder).",
    )
    parser.add_argument("--enabled-tools", default="",
                        help="Comma-separated MCP tools to expose. Empty exposes all tools.")
    parser.add_argument("--agent-id", default="agent",
                        help="Agent ID, used as the ActiveMemory owner label when --enable-memory-tools is set.")
    parser.add_argument("--enable-memory-tools", action="store_true",
                        help="Register MCP memory tools (mem_add/update/delete + memory_retrieve/new/update/delete) "
                             "for prompt_mode='active_memory2'. Persists workspace/memory/{date}/{mem.csv,meta.yaml} "
                             "on next_day and chmods the snapshot read-only.")
    args = parser.parse_args()
    if args.config_file:
        config_path = Path(args.config_file)
        try:
            config_data = json.loads(config_path.read_text())
        except Exception as exc:
            parser.error(f"could not read --config-file {config_path}: {exc}")
        for key, value in config_data.items():
            if hasattr(args, key):
                setattr(args, key, value)
    if not args.workspace:
        parser.error("--workspace is required unless supplied by --config-file")
    _cli_args = args
    _handholding_version = args.handholding_version
    _memory_tools_enabled = args.enable_memory_tools
    _agent_id = args.agent_id

    # Only set repo-root on sys.path here (lightweight).
    # All heavy imports (lancedb etc.) are deferred to _ensure_search().
    if args.repo_root and args.repo_root not in sys.path:
        sys.path.insert(0, args.repo_root)

    _workspace = Path(args.workspace).resolve()
    # Internal dir for state.json and signals — defaults to workspace for backward compat.
    _internal_dir = Path(args.internal_dir).resolve() if args.internal_dir else _workspace
    _internal_dir.mkdir(parents=True, exist_ok=True)

    # Load state (lightweight — just reads a JSON file).
    _reload_state()
    _search_cutoff_days = _state.get("search_cutoff_days", 0)

    # Register the optional memory tools BEFORE mcp.run() so they appear in the
    # initial tool-list handshake.
    if _memory_tools_enabled:
        _register_memory_tools()

    enabled_tools = {name.strip() for name in args.enabled_tools.split(",") if name.strip()}
    if enabled_tools:
        for tool_name in list(mcp._tool_manager._tools.keys()):
            if tool_name not in enabled_tools:
                mcp.remove_tool(tool_name)

    # Start the MCP server IMMEDIATELY — heavy init happens on first tool call.
    mcp.run()


if __name__ == "__main__":
    main()
