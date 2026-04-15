"""
MCP server for Claude Code forecasting agent.

Provides 4 tools: search_news, read_article, submit_forecast, next_day.
Communicates with the driver (ClaudeCodeAgent) via file-based signals.

IMPORTANT: All heavy imports (lancedb, pandas, torch, etc.) are deferred to
first use.  The MCP handshake must complete in <10s or Claude Code will give
up and the tools won't be available.
"""

import argparse
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


# ── server ─────────────────────────────────────────────────────────────

mcp = FastMCP("forecast-sim")

# Global state — set in main(), lazily initialized on first tool call.
_state: dict = {}
_workspace: Path = Path(".")     # CC's workspace (market.csv, articles/, memory/)
_internal_dir: Path = Path(".")  # Driver coordination (state.json, signals/)
_search_handler = None       # Set by _ensure_search()
_search_tool = None           # Set by _ensure_search()
_today_predictions: list = []
_search_cutoff_days: int = 0
_search_initialized: bool = False
_cli_args = None              # Parsed CLI args, used by _ensure_search()

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
    global _state, _today_predictions
    _state = _read_json(_internal_dir / "state.json")
    _today_predictions = []
    if _search_handler is not None:
        _search_handler.set_date(_parse_date(_state["current_date"]))


def _build_new_articles_message(previous_date: Optional[date], current_date: Optional[date]) -> str:
    """Match BasicAgent's article-count wording when search can provide a count."""
    if previous_date is None or current_date is None:
        return (
            "New articles may be available in articles/ or via the search_news MCP tool."
        )

    _ensure_search()
    if _search_handler is not None and _search_handler.is_available:
        max_date = current_date - timedelta(days=_search_cutoff_days)
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
        f"New articles may be available for {current_date.isoformat()} in articles/ "
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

    search_db = args.search_db or os.environ.get("FSIM_SEARCH_DB", _state.get("search_db", ""))
    embedding_model_path = args.embedding_model or os.environ.get("FSIM_EMBEDDING_MODEL", _state.get("embedding_model", ""))

    if not search_db or not os.path.exists(search_db):
        return

    from agents.search_tools.lancedb.store import LanceDBSearchTool
    from agents.basicAgent.search import SearchHandler

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
    _search_handler.set_date(_parse_date(_state["current_date"]))


# ── embedding client ───────────────────────────────────────────────────

class _EmbeddingOutput:
    """Mimics vLLM EmbeddingRequestOutput for LanceDBSearchTool compatibility."""
    def __init__(self, embedding):
        self.outputs = type("obj", (), {"embedding": embedding})()


class _VLLMEmbeddingClient:
    """Lightweight client that connects to an already-running vLLM embedding server."""

    def __init__(self, server_url: str, model_name: str = ""):
        self._url = server_url.rstrip("/")
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


# ── tools ──────────────────────────────────────────────────────────────

@mcp.tool()
def search_news(
    query: str,
    max_results: int = 5,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> str:
    """Search news articles (semantic + keyword hybrid). Results are date-capped to the current simulation date.

    Args:
        query: Search query string.
        max_results: Maximum number of article chunks to return (default 5, max 10).
        from_date: Optional earliest article date (YYYY-MM-DD).
        to_date: Optional latest article date (YYYY-MM-DD). Capped to sim date.
    """
    _ensure_search()

    if _search_handler is None or not _search_handler.is_available:
        return "Search is not available for this simulation run."

    max_results = min(max_results, 10)
    search_type = _state.get("search_type", "hybrid")
    result_text, err = _search_handler.search(
        query,
        max_results=max_results,
        search_type=search_type,
        min_date=_parse_date(from_date),
        max_date=_parse_date(to_date),
    )
    if err:
        return f"Search error: {err}"
    return result_text or "No articles found matching your query."


@mcp.tool()
def submit_forecast(question_id: str, outcomes: dict[str, float]) -> str:
    """Submit a probabilistic prediction for a forecasting question.

    Args:
        question_id: The qid of the question to predict on.
        outcomes: A dict mapping outcome strings to probabilities, e.g. {"Yes": 0.7, "No": 0.3}. Must sum to <= 1.0.
    """
    max_outcomes = _state.get("max_outcomes_per_question", 5)
    if len(outcomes) > max_outcomes:
        return f"Error: {len(outcomes)} outcomes exceeds maximum of {max_outcomes} per question."
    total = sum(outcomes.values())
    if total > 1.0 + 1e-6:
        return f"Error: probabilities sum to {total:.4f} which exceeds 1.0."
    for outcome, prob in outcomes.items():
        if prob < 0 or prob > 1:
            return f"Error: probability {prob} for '{outcome}' is out of [0, 1] range."

    pred = {"question_id": question_id, "outcomes": outcomes}
    _today_predictions.append(pred)

    # Merge with any predictions already in the file (e.g. written via Bash).
    # Later submissions for the same qid overwrite earlier ones.
    cur_date = _state["current_date"]
    pred_path = _internal_dir / "predictions" / f"{cur_date}.json"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if pred_path.exists():
        try:
            existing = json.loads(pred_path.read_text())
        except (json.JSONDecodeError, OSError):
            existing = []
    # Build merged list: existing + _today_predictions, last write wins per qid.
    by_qid = {}
    for p in existing:
        qid_key = p.get("question_id", p.get("qid", ""))
        if qid_key:
            by_qid[qid_key] = p
    for p in _today_predictions:
        qid_key = p.get("question_id", p.get("qid", ""))
        if qid_key:
            by_qid[qid_key] = p
    merged = list(by_qid.values())
    with open(pred_path, "w") as f:
        json.dump(merged, f, default=str)

    return f"Prediction recorded for question {question_id}: {outcomes}"


@mcp.tool()
def next_day() -> str:
    """Signal that you are done with the current day. Blocks until the simulation advances.

    Returns context for the new day: new date, resolution events, and whether the simulation is complete.
    Call this only after submitting predictions for all active questions.
    """
    cur_date = _state["current_date"]

    signal_dir = _internal_dir / "signals"
    signal_dir.mkdir(parents=True, exist_ok=True)
    _write_json(signal_dir / f"next_day_{cur_date}.json", {
        "predictions": _today_predictions,
        "date": cur_date,
    })

    # Poll for continue signal, consuming it so the next next_day call blocks.
    poll_interval = 0.5
    timeout = float(_state.get("timeout_seconds", 3600))
    start = time.time()
    while True:
        for p in signal_dir.glob("continue_*.json"):
            cont = _read_json(p)
            p.unlink(missing_ok=True)
            _reload_state()
            status = cont.get("status", "day_advanced")
            new_date = _state.get("current_date", "unknown")
            new_date_obj = _parse_date(new_date)

            response_parts = [
                f"Day advanced to {new_date}.",
                _build_new_articles_message(_parse_date(cur_date), new_date_obj),
            ]

            # --- Per-question resolution feedback with scores ---
            events = _state.get("resolution_events", [])
            new_events = [
                ev for ev in events
                if ev.get("qid") not in _cumulative["processed_qids"]
            ]
            if new_events:
                response_parts.append(
                    f"\n## RESULTS SINCE YOUR LAST SESSION ({cur_date} -> {new_date})\n"
                )
                for ev in new_events:
                    qid = ev.get("qid", ev.get("question_id", "?"))
                    title = ev.get("title", "")
                    gt = ev.get("ground_truth", "?")
                    _cumulative["processed_qids"].add(qid)

                    # Extract per-agent scoring from the event.
                    agents_data = ev.get("agents", {})
                    my_stats = None
                    # Find this agent's stats (there should be only one agent).
                    for aid, stats in agents_data.items():
                        if isinstance(stats, dict):
                            my_stats = stats
                            break

                    if my_stats:
                        brier = float(my_stats.get("brier", 0))
                        tw_peer = float(my_stats.get("tw_peer", 0))
                        best_outcome = my_stats.get("best_outcome", "?")
                        best_prob = float(my_stats.get("best_prob", 0))
                        is_accurate = my_stats.get("is_accurate", False)
                        _cumulative["brier_sum"] += brier
                        _cumulative["tw_peer_sum"] += tw_peer
                        _cumulative["resolved_count"] += 1
                        if is_accurate:
                            _cumulative["accuracy_count"] += 1

                        # Show full prediction distribution (matching BasicAgent).
                        agent_preds = _state.get("agent_predictions", {})
                        pred_dist = agent_preds.get(str(qid), {})
                        if pred_dist:
                            dist_items = sorted(pred_dist.items(), key=lambda kv: -kv[1])
                            dist_str = ", ".join(f"{o}: {p:.2f}" for o, p in dist_items)
                            dist_str = "{" + dist_str + "}"
                        else:
                            dist_str = f"{{{best_outcome}: {best_prob:.2f}}}"

                        verdict = "✓ CORRECT" if is_accurate else "✗ WRONG"
                        response_parts.append(
                            f"- \"{title}\"\n"
                            f"  Your prediction distribution: {dist_str} | Truth: {gt}\n"
                            f"  Brier: {brier:+.2f} | TW-Peer: {tw_peer:+.2f}"
                        )
                    else:
                        response_parts.append(f"- \"{title}\" → {gt}")

                # --- Cumulative performance summary ---
                rc = _cumulative["resolved_count"]
                total_preds = _state.get("total_predictions", 0)
                if rc > 0:
                    avg_brier = _cumulative["brier_sum"] / rc
                    accuracy = _cumulative["accuracy_count"] / rc * 100
                    response_parts.append(
                        f"\n## YOUR CUMULATIVE PERFORMANCE TILL TODAY\n"
                        f"- Total Predictions: {total_preds} ({rc} resolved)\n"
                        f"- accuracy: {accuracy:.1f}% | brier skill score: {avg_brier:.3f} | "
                        f"time weighted score: {_cumulative['tw_peer_sum']:.2f}\n"
                        f"  accuracy = fraction of resolved questions where your top outcome "
                        f"matched the truth; brier skill score = mean brier skill score across resolved questions; "
                        f"time weighted score = sum of brier skill scores across all resolved questions, across all days you held your respective predictions"
                    )

            if status == "simulation_complete":
                response_parts.append("\nSimulation is complete. No more days to process.")

            return "\n".join(response_parts)

        if time.time() - start > timeout:
            return "Error: timed out waiting for simulation to advance."
        time.sleep(poll_interval)


# ── main ───────────────────────────────────────────────────────────────

def main():
    global _workspace, _internal_dir, _search_cutoff_days, _cli_args

    parser = argparse.ArgumentParser(description="Forecast-sim MCP server")
    parser.add_argument("--workspace", required=True, help="Path to CC workspace (predictions/)")
    parser.add_argument("--internal-dir", default="", help="Path to internal dir (state.json, signals/)")
    parser.add_argument("--repo-root", default="", help="Repo root to add to sys.path")
    parser.add_argument("--search-db", default="", help="Path to LanceDB search index")
    parser.add_argument("--embedding-model", default="", help="Path to embedding model")
    parser.add_argument("--embedding-server-url", default="",
                        help="URL of an already-running vLLM embedding server (e.g. http://127.0.0.1:8001)")
    args = parser.parse_args()
    _cli_args = args

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

    # Start the MCP server IMMEDIATELY — heavy init happens on first tool call.
    mcp.run()


if __name__ == "__main__":
    main()
