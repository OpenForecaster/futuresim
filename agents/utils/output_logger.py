"""
Agent-side logging for model transcripts and raw turn deltas.

The simulation environment owns shared simulation logs (predictions, resolutions,
metrics). Each agent owns its own model-output logs.
"""

from __future__ import annotations

import atexit
import json
from datetime import date
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple


class AgentOutputLogger:
    """Persist per-agent cleaned and raw model outputs."""

    def __init__(self, agent_id: str, output_dir: Optional[str], *, append: bool = False):
        self.agent_id = agent_id
        self.output_dir = Path(output_dir) if output_dir else None
        self.append = bool(append)
        self._lock = Lock()
        self._files_opened = False
        self._model_outputs_file = None
        self._raw_daily_file = None
        self._raw_warmup_file = None
        self._warmup_raw_buffer: List[Tuple[str, int, Dict[str, Any]]] = []
        self._warmup_raw_seq = 0

        if self.output_dir is not None:
            atexit.register(self.close)

    def _ensure_files_open(self) -> bool:
        if self.output_dir is None:
            return False
        if self._files_opened:
            return True

        self.output_dir.mkdir(parents=True, exist_ok=True)
        mode = "a" if self.append else "w"
        self._model_outputs_file = open(self.output_dir / "model_outputs.jsonl", mode)
        self._raw_daily_file = open(self.output_dir / "model_raw_daily.jsonl", mode)
        self._raw_warmup_file = open(self.output_dir / "model_raw_warmup.jsonl", mode)
        self._files_opened = True
        # If the logger ever re-opens later in the same process, do not truncate again.
        self.append = True
        return True

    @staticmethod
    def _render_prompt_text(prompt: Any) -> str:
        if prompt is None:
            return ""
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, list):
            parts = [AgentOutputLogger._render_prompt_text(item) for item in prompt]
            return "\n\n".join(part for part in parts if part)
        if isinstance(prompt, dict):
            if isinstance(prompt.get("content"), str) and prompt.get("content").strip():
                return prompt["content"]
            if isinstance(prompt.get("output"), str) and prompt.get("output").strip():
                return prompt["output"]
        return json.dumps(prompt, ensure_ascii=False)

    def log_model_output(
        self,
        sim_date: date,
        prompt: Any,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._ensure_files_open():
            return

        metadata = metadata or {}
        raw_stream = str(metadata.get("raw_stream", "daily") or "daily").strip().lower()
        if raw_stream not in {"daily", "warmup"}:
            raw_stream = "daily"

        qid = metadata.get("qid")
        raw_input_delta = metadata.get("_logger_raw_input_delta", prompt)
        raw_response = metadata.get("_logger_raw_response", response)
        raw_metadata = metadata.get("_logger_raw_metadata")
        clean_metadata = {
            k: v for k, v in metadata.items()
            if not k.startswith("_logger_")
        }
        if raw_metadata is None:
            raw_metadata = clean_metadata.copy()
        prompt_text = self._render_prompt_text(prompt)
        raw_record = {
            "sim_date": str(sim_date),
            "agent_id": self.agent_id,
            "qid": qid,
            "prompt": prompt_text,
            "input_delta": raw_input_delta,
            "response": raw_response,
            "metadata": raw_metadata,
        }

        clean_response = response
        reasoning = clean_metadata.get("reasoning")
        if reasoning and "<reasoning>" not in (clean_response or ""):
            clean_response = f"<reasoning>{reasoning}</reasoning>\n{clean_response}"

        clean_record = {
            "sim_date": str(sim_date),
            "agent_id": self.agent_id,
            "qid": qid,
            "response": clean_response,
            "metadata": clean_metadata,
        }

        with self._lock:
            self._model_outputs_file.write(json.dumps(clean_record) + "\n")
            self._model_outputs_file.flush()

            if raw_stream == "warmup":
                self._warmup_raw_seq += 1
                qid_key = "" if qid is None else str(qid)
                self._warmup_raw_buffer.append((qid_key, self._warmup_raw_seq, raw_record))
                return

            self._raw_daily_file.write(json.dumps(raw_record) + "\n")
            self._raw_daily_file.flush()

    def flush_warmup_raw(self) -> None:
        if not self._ensure_files_open():
            return

        with self._lock:
            buffered = self._warmup_raw_buffer
            if not buffered:
                return
            for _, _, raw_record in sorted(buffered, key=lambda item: (item[0], item[1])):
                self._raw_warmup_file.write(json.dumps(raw_record) + "\n")
            self._raw_warmup_file.flush()
            self._warmup_raw_buffer = []

    def close(self) -> None:
        if self.output_dir is None:
            return
        with self._lock:
            if not self._files_opened:
                return
            if self._warmup_raw_buffer:
                for _, _, raw_record in sorted(self._warmup_raw_buffer, key=lambda item: (item[0], item[1])):
                    self._raw_warmup_file.write(json.dumps(raw_record) + "\n")
                self._raw_warmup_file.flush()
                self._warmup_raw_buffer = []
            self._model_outputs_file.close()
            self._raw_daily_file.close()
            self._raw_warmup_file.close()
            self._files_opened = False
