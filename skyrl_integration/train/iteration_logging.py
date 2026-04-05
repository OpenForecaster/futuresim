"""Tinker-style run/iteration logging helpers for forecast-sim SkyRL runs."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional

from loguru import logger

_ITERATION_RE = re.compile(r"global_step_(\d+)_evals$")


def _coerce_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = resolved.resolve()
    return resolved


def resolve_run_root_from_log_path(log_path: str | Path) -> Path:
    """Return the run root, treating ``.../infra`` as a child log directory."""
    path = _coerce_path(log_path)
    if path.is_file() or path.suffix == ".log":
        path = path.parent
    return path.parent if path.name == "infra" else path


def iteration_output_dir(run_root: Path, step: int | None) -> Path:
    if step is None:
        return run_root / "eval_only"
    return run_root / f"iteration_{int(step):06d}"


def parse_eval_step_from_dump_dir(dump_dir_path: str | Path) -> int | None:
    name = Path(dump_dir_path).name
    match = _ITERATION_RE.match(name)
    if match:
        return int(match.group(1))
    return None


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return str(value)


def _scalar_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    scalars: Dict[str, Any] = {}
    for key, value in data.items():
        safe = _json_safe(value)
        if isinstance(safe, (str, int, float, bool)) or safe is None:
            scalars[str(key)] = safe
    return scalars


def _sequence_reward_total(reward: Any) -> float | None:
    safe = _json_safe(reward)
    if isinstance(safe, (int, float)):
        return float(safe)
    if isinstance(safe, list):
        total = 0.0
        for item in safe:
            if isinstance(item, (int, float)):
                total += float(item)
        return total
    return None


def _truncate_string(value: str, limit: int = 240) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "...<truncated>"


def _compact_env_value(value: Any, *, depth: int = 0) -> Any:
    safe = _json_safe(value)
    if isinstance(safe, str):
        return _truncate_string(safe)
    if isinstance(safe, (int, float, bool)) or safe is None:
        return safe
    if depth >= 2:
        return None
    if isinstance(safe, dict):
        compact: Dict[str, Any] = {}
        for key, item in safe.items():
            if str(key) in {"prompt", "messages", "chat_history", "conversation", "input_prompt", "trajectory_text"}:
                continue
            compact_item = _compact_env_value(item, depth=depth + 1)
            if compact_item is not None:
                compact[str(key)] = compact_item
        return compact or None
    if isinstance(safe, list):
        if len(safe) > 8:
            return None
        compact_list = [_compact_env_value(item, depth=depth + 1) for item in safe]
        compact_list = [item for item in compact_list if item is not None]
        return compact_list or None
    return None


def _compact_env_payload(env_extra: Any) -> Any:
    return _compact_env_value(env_extra, depth=0)


def _decode_token_sequences(tokenizer: Any, sequences: List[List[int]]) -> List[str | None]:
    if tokenizer is None:
        return [None] * len(sequences)
    if hasattr(tokenizer, "batch_decode"):
        return list(tokenizer.batch_decode(sequences, skip_special_tokens=False))
    return [tokenizer.decode(sequence, skip_special_tokens=False) for sequence in sequences]


class RunArtifactLogger:
    """Writes top-level metrics and per-iteration rollout files under a SkyRL run root."""

    def __init__(self, run_root: str | Path):
        self.run_root = _coerce_path(run_root)
        self.run_root.mkdir(parents=True, exist_ok=True)
        self._metrics_path = self.run_root / "metrics.jsonl"
        self._lock = Lock()
        self._pending_step: int | None = None
        self._pending_metrics: Dict[str, Any] = {}
        self._config_written = False

    def _flush_pending_locked(self) -> None:
        if not self._pending_metrics:
            return
        record = {
            "step": self._pending_step,
            "logged_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            **self._pending_metrics,
        }
        with self._metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        self._pending_step = None
        self._pending_metrics = {}

    def write_config(self, config: Any) -> None:
        if config is None:
            return
        with self._lock:
            if self._config_written:
                return
            try:
                from skyrl.train.config import get_config_as_dict

                payload = _json_safe(get_config_as_dict(config))
            except Exception:
                payload = _json_safe(config)
            (self.run_root / "config.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            self._config_written = True

    def log_metrics(self, step: int | None, data: Dict[str, Any], *, commit: bool = False) -> None:
        metrics = _scalar_metrics(data)
        if not metrics:
            return
        with self._lock:
            if self._pending_metrics and step != self._pending_step:
                self._flush_pending_locked()
            self._pending_step = step if step is None else int(step)
            self._pending_metrics.update(metrics)
            if commit:
                self._flush_pending_locked()

    def finish(self) -> None:
        with self._lock:
            self._flush_pending_locked()

    def append_train_rollout_summaries(
        self,
        *,
        tokenizer: Any,
        step: int | None,
        generator_output: Dict[str, Any],
        uids: List[str],
        env_extras: Optional[List[Dict[str, Any]]] = None,
        env_classes: Optional[List[str]] = None,
    ) -> None:
        responses = generator_output.get("response_ids") or []
        prompts = generator_output.get("prompt_token_ids") or []
        rewards = generator_output.get("rewards") or []
        if not responses or not prompts:
            return

        iter_dir = iteration_output_dir(self.run_root, step)
        iter_dir.mkdir(parents=True, exist_ok=True)
        out_path = iter_dir / "train_rollout_summaries.jsonl"

        stop_reasons = generator_output.get("stop_reasons") or [None] * len(responses)
        input_prompts = _decode_token_sequences(tokenizer, prompts)
        trajectory_texts = _decode_token_sequences(tokenizer, responses)
        trajectory_ids = (
            generator_output.get("fsim_trajectory_ids")
            or generator_output.get("trajectory_ids")
            or [None] * len(responses)
        )
        is_last_step = generator_output.get("is_last_step") or [None] * len(responses)
        env_extras = env_extras or [None] * len(responses)
        env_classes = env_classes or [None] * len(responses)

        with out_path.open("a", encoding="utf-8") as handle:
            for idx, response_ids in enumerate(responses):
                reward = rewards[idx] if idx < len(rewards) else None
                trajectory_id = trajectory_ids[idx] if idx < len(trajectory_ids) else None
                record = {
                    "step": step,
                    "uid": uids[idx] if idx < len(uids) else None,
                    "env_class": env_classes[idx] if idx < len(env_classes) else None,
                    "env_metadata": _compact_env_payload(env_extras[idx] if idx < len(env_extras) else None),
                    "trajectory_id": (
                        trajectory_id.to_string()
                        if hasattr(trajectory_id, "to_string")
                        else _json_safe(trajectory_id)
                    ),
                    "repetition_id": getattr(trajectory_id, "repetition_id", None),
                    "is_last_step": is_last_step[idx] if idx < len(is_last_step) else None,
                    "input_prompt": input_prompts[idx] if idx < len(input_prompts) else None,
                    "prompt_num_tokens": len(prompts[idx]),
                    "trajectory_text": trajectory_texts[idx] if idx < len(trajectory_texts) else None,
                    "response_num_tokens": len(response_ids),
                    "reward": _sequence_reward_total(reward),
                    "stop_reason": stop_reasons[idx] if idx < len(stop_reasons) else None,
                }
                handle.write(json.dumps(_json_safe(record), ensure_ascii=False, sort_keys=True) + "\n")

    def write_eval_rollout_summaries(
        self,
        *,
        step: int | None,
        tokenizer: Any,
        concat_generator_outputs: Dict[str, Any],
        concat_data_sources: Iterable[str | None],
        concat_all_envs: Iterable[str],
        concat_env_extras: Iterable[Dict[str, Any]],
        eval_metrics: Dict[str, Any],
    ) -> None:
        iter_dir = iteration_output_dir(self.run_root, step)
        iter_dir.mkdir(parents=True, exist_ok=True)

        prompts = concat_generator_outputs.get("prompt_token_ids") or []
        responses = concat_generator_outputs.get("response_ids") or []
        rewards = concat_generator_outputs.get("rewards") or []
        stop_reasons = concat_generator_outputs.get("stop_reasons") or [None] * len(responses)
        input_prompts = _decode_token_sequences(tokenizer, prompts)
        trajectory_texts = _decode_token_sequences(tokenizer, responses)
        data_sources = list(concat_data_sources)
        env_classes = list(concat_all_envs)
        env_extras = list(concat_env_extras)

        with (iter_dir / "eval_rollout_summaries.jsonl").open("w", encoding="utf-8") as handle:
            for idx, response_ids in enumerate(responses):
                reward = rewards[idx] if idx < len(rewards) else None
                record = {
                    "step": step,
                    "data_source": data_sources[idx] if idx < len(data_sources) else None,
                    "env_class": env_classes[idx] if idx < len(env_classes) else None,
                    "env_metadata": _compact_env_payload(env_extras[idx] if idx < len(env_extras) else None),
                    "input_prompt": input_prompts[idx] if idx < len(input_prompts) else None,
                    "prompt_num_tokens": len(prompts[idx]),
                    "trajectory_text": trajectory_texts[idx] if idx < len(trajectory_texts) else None,
                    "response_num_tokens": len(response_ids),
                    "score": _sequence_reward_total(reward),
                    "stop_reason": stop_reasons[idx] if idx < len(stop_reasons) else None,
                }
                handle.write(json.dumps(_json_safe(record), ensure_ascii=False, sort_keys=True) + "\n")

        (iter_dir / "eval_metrics.json").write_text(
            json.dumps(_json_safe(eval_metrics), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


_LOGGER_CACHE: Dict[str, RunArtifactLogger] = {}
_CACHE_LOCK = Lock()


def get_run_artifact_logger(
    *,
    config: Any = None,
    log_path: str | Path | None = None,
) -> RunArtifactLogger | None:
    effective_log_path = log_path
    if effective_log_path is None and config is not None:
        trainer_cfg = getattr(config, "trainer", None)
        effective_log_path = getattr(trainer_cfg, "log_path", None) if trainer_cfg is not None else None
    if effective_log_path is None:
        effective_log_path = os.environ.get("SIM_OUTPUT_DIR") or os.environ.get("SKYRL_LOG_FILE")
    if not effective_log_path:
        return None

    run_root = resolve_run_root_from_log_path(effective_log_path)
    cache_key = str(run_root)
    with _CACHE_LOCK:
        artifact_logger = _LOGGER_CACHE.get(cache_key)
        if artifact_logger is None:
            artifact_logger = RunArtifactLogger(run_root)
            _LOGGER_CACHE[cache_key] = artifact_logger
            logger.info(f"[forecast-sim] SkyRL run artifacts will be written under {run_root}")
        return artifact_logger
