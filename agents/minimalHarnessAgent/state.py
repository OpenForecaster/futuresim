"""State and data-visibility helpers for MinimalHarnessAgent."""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class StateHelpers:
    """Workspace, state.json, market.csv, memory, and article staging helpers."""

    def _apply_bootstrap(self, forecast_interface: Any, current_date: date) -> None:
        """Seed Day-1 from fixedWarmup memory/predictions.

        Copies bootstrap_dir/{mem.csv,meta.yaml} to workspace/memory/<prev>/,
        then injects prediction.json into env.histories with day=<prev> so
        Day 1 market.csv has my_prediction.
        """
        from environment.scoring.base import DailyPrediction

        bdir = Path(self.config.bootstrap_dir)
        if not bdir.is_dir():
            raise FileNotFoundError(f"bootstrap_dir not found: {bdir}")

        prev_date = current_date - timedelta(days=1)
        prev_iso = prev_date.isoformat()

        mem_dst = self.workspace / "memory" / prev_iso
        mem_dst.mkdir(parents=True, exist_ok=True)
        for fname in ("mem.csv", "meta.yaml"):
            src = bdir / fname
            if src.exists():
                shutil.copy2(src, mem_dst / fname)

        pred_src = bdir / "prediction.json"

        injected = skipped = 0
        if pred_src.exists():
            preds = json.loads(pred_src.read_text())
            histories = forecast_interface.histories
            lock = forecast_interface._histories_lock
            with lock:
                for p in preds:
                    qid = str(p.get("question_id"))
                    outcomes = p.get("outcomes")
                    if not qid or not outcomes:
                        skipped += 1
                        continue
                    history = histories.get(qid)
                    if history is None:
                        skipped += 1
                        continue
                    history.add_prediction(DailyPrediction(
                        agent_id=self.agent_id,
                        question_id=qid,
                        day=prev_date,
                        outcomes=dict(outcomes),
                    ))
                    injected += 1
            for p in preds:
                qid = str(p.get("question_id"))
                outcomes = p.get("outcomes")
                if qid and outcomes and histories.get(qid) is not None:
                    forecast_interface.logger.log_prediction(
                        prev_date, self.agent_id, qid, dict(outcomes)
                    )

        print(
            f"[{self.agent_id}] Bootstrap from {bdir}: "
            f"copied memory/{prev_iso}; "
            f"injected {injected} predictions ({skipped} skipped)."
        )

    @staticmethod
    def _to_iso(d) -> str:
        """Convert a date, datetime, or string to ISO date string."""
        if d is None:
            return ""
        if hasattr(d, "isoformat"):
            return d.isoformat()
        return str(d)

    def _write_state(
        self,
        forecast_interface: Any,
        current_date: date,
        questions_override: Optional[List[Any]] = None,
        search_current_date: Optional[date] = None,
    ) -> None:
        """Serialize simulation state for the MCP server."""
        questions = []
        visible_questions = (
            questions_override
            if questions_override is not None
            else forecast_interface.list_questions()
        )
        for q in visible_questions:
            questions.append({
                "qid": q.qid,
                "title": q.title,
                "background": q.background,
                "resolution_criteria": q.resolution_criteria,
                "answer_type": q.answer_type,
                "resolution_date": str(q.resolution_date) if q.resolution_date else None,
                "options": q.options,
            })

        # Include agent's prediction distributions so the MCP server can show
        # the full distribution in resolution feedback (matching BasicAgent).
        agent_predictions = {}
        get_preds = getattr(forecast_interface, "get_agent_predictions", None)
        if callable(get_preds):
            try:
                raw = get_preds(self.agent_id) or {}
                for qid, record in raw.items():
                    if isinstance(record, dict) and record.get("outcomes"):
                        agent_predictions[qid] = record["outcomes"]
            except Exception:
                pass

        # Count total active predictions.
        total_predictions = len(agent_predictions)

        state = {
            "current_date": current_date.isoformat(),
            "search_current_date": (
                self._to_iso(search_current_date)
                if search_current_date is not None
                else current_date.isoformat()
            ),
            "start_date": self._to_iso(self.config.start_date or current_date),
            "end_date": self._to_iso(self.config.end_date or current_date),
            "market_csv": forecast_interface.get_market_csv_path(),
            "search_backend": self.config.search_backend,
            "search_db": self.config.search_db,
            "embedding_model": self.config.embedding_model,
            "search_type": self.config.search_type,
            "search_cutoff_days": self.config.search_cutoff_days,
            "openreward_search_url": self.config.openreward_search_url,
            "openreward_fetch_url": self.config.openreward_fetch_url,
            "timeout_seconds": self.config.timeout_seconds,
            "questions": questions,
            "resolution_events": forecast_interface.resolution_events,
            "agent_predictions": agent_predictions,
            "total_predictions": total_predictions,
            "prompt_mode": self.config.prompt_mode,
            "submit_ends_session": self.config.prompt_mode in {"warmup", "static_search"},
            "next_day_returns_immediately": self._next_day_returns_immediately(),
            "max_outcomes_per_question": int(self.config.max_outcomes_per_question),
        }
        state_path = self._internal_dir / "state.json"
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def _protect_market_csv(self, forecast_interface: Any) -> None:
        """Write a per-agent workspace market.csv and protect the shared one.

        MarketWriter's shared csv lacks `my_prediction` / `my_prediction_date`;
        stamp them from this agent's history (like DfInterface in-memory for
        other scaffolds), write the workspace copy, and chmod the shared csv
        read-only as an accidental-write guard.
        """
        csv_path = forecast_interface.get_market_csv_path()
        if not (csv_path and os.path.exists(csv_path)):
            return
        self._market_csv_path = csv_path
        os.chmod(csv_path, 0o444)

        if (
            self.config.prompt_mode == "no_memory"
            and self.config.bootstrap_dir
            and self._bootstrap_applied
            and not getattr(self, "_bootstrap_market_csv_used", False)
        ):
            bootstrap_market_csv = Path(self.config.bootstrap_dir) / "market.csv"
            if bootstrap_market_csv.exists():
                workspace_csv = self.workspace / "market.csv"
                if workspace_csv.is_symlink() or workspace_csv.exists():
                    workspace_csv.unlink()
                shutil.copy2(bootstrap_market_csv, workspace_csv)
                self._bootstrap_market_csv_used = True
                return

        df = pd.read_csv(csv_path, dtype={"qid": str})
        get_preds = getattr(forecast_interface, "get_agent_predictions", None)
        agent_preds = get_preds(self.agent_id) if callable(get_preds) else {}
        agent_preds = agent_preds or {}

        df["my_prediction"] = df["qid"].apply(
            lambda qid: json.dumps(agent_preds[qid]["outcomes"])
            if qid in agent_preds and agent_preds[qid].get("outcomes") else None
        )
        df["my_prediction_date"] = df["qid"].apply(
            lambda qid: str(agent_preds[qid]["date"])
            if qid in agent_preds and agent_preds[qid].get("date") else None
        )

        workspace_csv = self.workspace / "market.csv"
        if workspace_csv.is_symlink() or workspace_csv.exists():
            workspace_csv.unlink()
        df.to_csv(workspace_csv, index=False)

    def _unprotect_market_csv(self) -> None:
        """Restore write permission so the env can overwrite on the next day."""
        csv_path = getattr(self, "_market_csv_path", None)
        if csv_path and os.path.exists(csv_path):
            os.chmod(csv_path, 0o644)

    def _find_latest_memory_date_before(self, current_date: date) -> Optional[date]:
        """Return the latest persisted ActiveMemory snapshot before current_date."""
        memory_root = self.workspace / "memory"
        if not memory_root.exists():
            return None

        latest: Optional[date] = None
        for entry in memory_root.iterdir():
            if not entry.is_dir():
                continue
            try:
                entry_date = date.fromisoformat(entry.name)
            except ValueError:
                continue
            if entry_date >= current_date:
                continue
            if not ((entry / "mem.csv").exists() or (entry / "meta.yaml").exists()):
                continue
            if latest is None or entry_date > latest:
                latest = entry_date
        return latest

    def _count_new_articles(
        self,
        last_active_date: Optional[date],
        current_date: Optional[date],
    ) -> Optional[int]:
        """Match BasicAgent's article-count window when search supports it."""
        if (
            self.search_tool is None
            or not getattr(self.search_tool, "is_available", False)
            or last_active_date is None
            or current_date is None
        ):
            return None
        effective_current_date = current_date
        if self.config.freeze_search_after_start and self.config.start_date is not None:
            effective_current_date = min(current_date, self.config.start_date)
        max_date = effective_current_date - timedelta(days=self.config.search_cutoff_days)
        if last_active_date > max_date:
            return 0
        try:
            return self.search_tool.count_articles(
                min_date=last_active_date,
                max_date=max_date,
            )
        except Exception:
            return None

    def _detect_embedding_server(self) -> str:
        """Locate the driver-owned vLLM embedding server.

        Prefer $FSIM_EMBEDDING_URL (set after vLLMInference is warm) so parallel
        same-host sims do not grab a sibling server; otherwise scan
        127.0.0.1:8001-8009 for /v1/models 200 for legacy/external launchers.
        127.0.0.1 avoids proxies that intercept localhost. Return URL or "".
        """
        import urllib.request
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        env_url = os.environ.get("FSIM_EMBEDDING_URL", "").strip()
        if env_url:
            try:
                req = urllib.request.Request(f"{env_url}/v1/models", method="GET")
                with opener.open(req, timeout=2) as resp:
                    if resp.status == 200:
                        logger.info("Using embedding server from FSIM_EMBEDDING_URL: %s", env_url)
                        return env_url
            except Exception as e:
                logger.warning(
                    "FSIM_EMBEDDING_URL=%s set but unreachable (%s); falling back to port scan",
                    env_url, e)

        for port in range(8001, 8010):
            url = f"http://127.0.0.1:{port}"
            try:
                req = urllib.request.Request(f"{url}/v1/models", method="GET")
                with opener.open(req, timeout=2) as resp:
                    if resp.status == 200:
                        logger.info("Found running embedding server at %s (scan)", url)
                        return url
            except Exception:
                continue
        return ""
