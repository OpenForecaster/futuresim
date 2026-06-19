"""Shared environment/session helpers for evaluation-platform adapters."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field, fields
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

from futuresim_agents.search_tools.hybrid import HybridSearchConfig
from environment.matcher_cache import resolve_sim_matcher_cache_path
from environment.env import SimulationEnvironment
from environment.interfaces import PredictionSubmission


def parse_iso_date(value: str | date | None) -> Optional[date]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


@dataclass
class FuturesimAdapterConfig:
    """Serializable Futuresim task configuration shared by all adapters."""

    dataset: str = "openforesight"
    dataset_path: str = "nikhilchandak/OpenForesight"
    dataset_cache: str = ""
    split: str = "aljazeera2026Q1"
    start_date: str = "2025-12-31"
    end_date: str = "2026-03-28"
    resolution_start: str = ""
    resolution_end: str = ""
    lookback_days: int = 7
    timegap_days: int = 1
    min_forecasters: int = 0
    resolved_only: bool = False
    max_outcomes_per_question: int = 5
    articles_base: str = ""
    article_stage_mode: str = "copy"
    article_search_cutoff_days: int = 0
    article_freeze_after_start: bool = False
    output_base: str = ""
    resume_dir: str = ""
    agent_id: str = "agent"
    sandbox_workspace: str = "/workspace"
    workspace_articles_subdir: str = "articles"
    prompt_mode: str = "default"
    handholding_version: str = "v1"
    enable_hybrid_search: bool = False
    hybrid_search: HybridSearchConfig = field(default_factory=HybridSearchConfig)
    matching: str = "openrouter"
    matcher: str = "deepseek/deepseek-v3.2"
    matcher_cache: dict[str, Any] | None = None
    matcher_max_concurrency: int = 300
    matcher_api_key_env: str = "OPENROUTER_API_KEY"
    matcher_openrouter_provider_order: list[str] | None = None
    matcher_openrouter_provider: dict[str, Any] | None = None
    matcher_vllm_max_model_len: int = 32768
    matcher_vllm_gpu_mem: float = 0.3
    matcher_vllm_max_num_seqs: int = 8
    matcher_vllm_startup_timeout: float = 300.0
    matcher_vllm_request_timeout: float = 120.0

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "FuturesimAdapterConfig":
        value = dict(value or {})
        if isinstance(value.get("futuresim"), dict):
            value = dict(value["futuresim"])
        if "matcher_cache_path" in value:
            value["matcher_cache"] = {"path": value.pop("matcher_cache_path")}
        if "output_dir" in value and "output_base" not in value:
            value["output_base"] = value.pop("output_dir")
        if "search_cutoff_days" in value and "article_search_cutoff_days" not in value:
            value["article_search_cutoff_days"] = value["search_cutoff_days"]
        if "freeze_search_after_start" in value and "article_freeze_after_start" not in value:
            value["article_freeze_after_start"] = value["freeze_search_after_start"]
        hybrid_value = value.pop("hybrid_search", None)
        allowed = {item.name for item in fields(cls)}
        cfg = cls(**{key: val for key, val in value.items() if key in allowed})
        if isinstance(hybrid_value, dict):
            cfg.hybrid_search = HybridSearchConfig(**hybrid_value)
        else:
            hybrid_keys = {
                "search_db",
                "embedding_model",
                "embedding_server_url",
                "search_type",
                "search_cutoff_days",
                "max_results",
            }
            hybrid_kwargs = {key: value[key] for key in hybrid_keys if key in value}
            if hybrid_kwargs:
                cfg.hybrid_search = HybridSearchConfig(**hybrid_kwargs)
        if "enable_hybrid_search" not in value and cfg.hybrid_search.search_db:
            cfg.enable_hybrid_search = True
        return cfg

    def to_task_spec(self) -> dict[str, Any]:
        data = asdict(self)
        data["hybrid_search"] = asdict(self.hybrid_search)
        return data


@dataclass(frozen=True)
class SandboxUpload:
    """One file that should be copied into an agent sandbox."""

    local_path: Path
    remote_path: str


@dataclass(frozen=True)
class SandboxCommandResult:
    output: str
    exit_code: int = 0
    truncated: bool = False


@dataclass(frozen=True)
class AdapterDayResult:
    done: bool
    reward: float = 0.0


def build_simulation_environment(
    config: FuturesimAdapterConfig,
    *,
    output_dir: str | None = None,
) -> SimulationEnvironment:
    out = output_dir or config.output_base or tempfile.mkdtemp(prefix="futuresim-adapter-")
    start_date = parse_iso_date(config.start_date)
    if start_date is not None and config.lookback_days:
        from datetime import timedelta

        start_date = start_date - timedelta(days=max(0, int(config.lookback_days)))

    matcher_provider = build_matcher_provider(config)
    matcher_cache_path = None
    if matcher_provider is not None:
        matcher_cache_path = str(
            resolve_sim_matcher_cache_path(
                output_dir=out,
                matching=config.matching,
                matcher=config.matcher,
                split=config.split,
                matcher_cache=config.matcher_cache,
            )
        )

    env = SimulationEnvironment(
        dataset=config.dataset,
        dataset_path=config.dataset_path,
        dataset_cache=config.dataset_cache or None,
        start_date=start_date,
        end_date=parse_iso_date(config.end_date),
        output_dir=out,
        resume_dir=config.resume_dir or None,
        resolution_start=parse_iso_date(config.resolution_start),
        resolution_end=parse_iso_date(config.resolution_end),
        parallel=False,
        split=config.split,
        timegap_days=config.timegap_days,
        min_forecasters=config.min_forecasters,
        resolved_only=config.resolved_only,
        inference_provider=matcher_provider,
        matcher_cache_path=matcher_cache_path,
        matcher_max_concurrency=config.matcher_max_concurrency,
        max_outcomes_per_question=config.max_outcomes_per_question,
        articles_base=config.articles_base,
        article_stage_mode=config.article_stage_mode,  # type: ignore[arg-type]
        article_search_cutoff_days=config.article_search_cutoff_days,
        article_freeze_after_start=config.article_freeze_after_start,
    )
    env._futuresim_adapter_matcher_provider = matcher_provider  # type: ignore[attr-defined]
    env.register_agent_id(config.agent_id)
    return env


def build_matcher_provider(config: FuturesimAdapterConfig):
    """Construct the optional semantic answer matcher provider for adapter runs."""

    mode = str(config.matching or "exact").strip().lower()
    if mode in {"", "none", "exact"}:
        return None

    if mode == "openrouter":
        from inference.openrouter import OpenRouterInference

        kwargs: dict[str, Any] = {}
        if config.matcher_api_key_env:
            api_key = os.environ.get(config.matcher_api_key_env)
            if not api_key:
                raise ValueError(
                    f"Futuresim matcher_api_key_env={config.matcher_api_key_env!r} "
                    "is set, but that environment variable is empty."
                )
            kwargs["api_key"] = api_key
        provider = _build_openrouter_provider_config(config)
        if provider:
            kwargs["provider"] = provider
        return OpenRouterInference(config.matcher, **kwargs)

    if mode == "vllm":
        from inference.vllm import VLLMInference

        return VLLMInference(
            config.matcher,
            max_model_len=int(config.matcher_vllm_max_model_len),
            gpu_memory_utilization=float(config.matcher_vllm_gpu_mem),
            max_num_seqs=int(config.matcher_vllm_max_num_seqs),
            startup_timeout=float(config.matcher_vllm_startup_timeout),
            timeout=float(config.matcher_vllm_request_timeout),
        )

    raise ValueError(f"Unknown Futuresim matching mode: {config.matching!r}")


def _build_openrouter_provider_config(config: FuturesimAdapterConfig) -> dict[str, Any]:
    provider: dict[str, Any] = {}
    if isinstance(config.matcher_openrouter_provider, Mapping):
        provider.update(dict(config.matcher_openrouter_provider))
    if config.matcher_openrouter_provider_order:
        provider["order"] = list(config.matcher_openrouter_provider_order)
        provider.setdefault("allow_fallbacks", True)
    return provider


class FuturesimAdapterRuntime:
    """Small adapter-facing wrapper around ``SimulationEnvironment``."""

    def __init__(
        self,
        config: FuturesimAdapterConfig,
        *,
        output_dir: str | None = None,
        workspace_path: str | None = None,
    ):
        self.config = config
        self.agent_id = config.agent_id
        self.workspace_path = (workspace_path or config.sandbox_workspace).rstrip("/") or "/workspace"
        self.env = build_simulation_environment(config, output_dir=output_dir)
        self.active_questions = []
        self._article_stage = tempfile.TemporaryDirectory(prefix="futuresim-articles-")
        self._last_article_upload_date: Optional[date] = None

    @property
    def articles_remote_dir(self) -> str:
        return self.remote_path(self.config.workspace_articles_subdir)

    @property
    def market_remote_path(self) -> str:
        return self.remote_path("market.csv")

    def remote_path(self, *parts: str | Path) -> str:
        suffix = "/".join(str(part).strip("/") for part in parts if str(part).strip("/"))
        return f"{self.workspace_path}/{suffix}" if suffix else self.workspace_path

    def begin_day(self) -> list[Any]:
        self.active_questions = self.env.begin_day()
        return list(self.active_questions)

    def forecast_interface(self):
        return self.env.make_forecast_interface(
            self.agent_id,
            active_questions=self.active_questions,
        )

    def write_agent_market_csv(self, path: str | Path) -> None:
        forecast_interface = self.forecast_interface()
        src = forecast_interface.get_market_csv_path()
        if not src:
            raise RuntimeError("Simulation did not produce a market.csv path.")
        df = pd.read_csv(src, dtype={"qid": str})
        agent_preds = forecast_interface.get_agent_predictions(self.agent_id) or {}
        df["my_prediction"] = df["qid"].apply(
            lambda qid: json.dumps(agent_preds[qid]["outcomes"])
            if qid in agent_preds and agent_preds[qid].get("outcomes") else None
        )
        df["my_prediction_date"] = df["qid"].apply(
            lambda qid: str(agent_preds[qid]["date"])
            if qid in agent_preds and agent_preds[qid].get("date") else None
        )
        df.to_csv(path, index=False)

    def submit_predictions(self, predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        accepted: list[dict[str, Any]] = []
        forecast_interface = self.forecast_interface()
        for pred in predictions:
            question_id = pred.get("question_id", pred.get("qid"))
            outcomes = pred.get("outcomes")
            if not question_id or not isinstance(outcomes, dict):
                continue
            forecast_interface.submit_prediction(
                PredictionSubmission(question_id=str(question_id), outcomes=dict(outcomes))
            )
            accepted.append({"question_id": str(question_id), "outcomes": dict(outcomes)})
        return accepted

    def finish_day(self) -> AdapterDayResult:
        self.env.end_day(self.active_questions)
        self.active_questions = []
        reward = self.reward()
        self.env.advance_day()
        if self.env.is_complete():
            return AdapterDayResult(done=True, reward=reward)
        return AdapterDayResult(done=False, reward=0.0)

    def prepare_article_uploads(self) -> tuple[list[SandboxUpload], Optional[date]]:
        corpus = self.env.article_corpus
        if corpus is None or not corpus.is_available:
            return [], None

        local_articles_dir = Path(self._article_stage.name) / self.config.workspace_articles_subdir
        marker = corpus.stage_local(
            local_articles_dir,
            self.env.current_date,
            mode="copy",
            start_date=self.env.start_date,
            search_cutoff_days=self.config.article_search_cutoff_days,
            freeze_after_start=self.config.article_freeze_after_start,
            since_date=self._last_article_upload_date,
        )
        if marker is None:
            return [], None

        uploads: list[SandboxUpload] = []
        for item in corpus.visible_files(
            self.env.current_date,
            start_date=self.env.start_date,
            search_cutoff_days=self.config.article_search_cutoff_days,
            freeze_after_start=self.config.article_freeze_after_start,
            since_date=self._last_article_upload_date,
        ):
            local_path = local_articles_dir / item.relative_path
            if local_path.exists():
                uploads.append(
                    SandboxUpload(
                        local_path=local_path,
                        remote_path=self.remote_path(
                            self.config.workspace_articles_subdir,
                            item.relative_path,
                        ),
                    )
                )
        return uploads, marker

    def commit_article_uploads(self, marker: Optional[date]) -> None:
        if marker is not None:
            self._last_article_upload_date = marker

    def reward(self) -> float:
        for row in self.env.metrics():
            if row.get("agent_id") == self.agent_id:
                return float(row.get("avg_brier", 0.0))
        return 0.0

    def close(self) -> None:
        try:
            self.env.logger.close()
        except Exception:
            pass
        provider = getattr(self.env, "_futuresim_adapter_matcher_provider", None)
        close = getattr(provider, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        self._article_stage.cleanup()


def default_articles_base() -> str:
    return os.environ.get("FSIM_ARTICLES_BASE", "")


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def coerce_command_result(value: Any) -> SandboxCommandResult:
    if isinstance(value, SandboxCommandResult):
        return value
    if isinstance(value, tuple) and len(value) >= 2:
        return SandboxCommandResult(
            output=str(value[0] or ""),
            exit_code=int(value[1] or 0),
            truncated=bool(getattr(value, "truncated", False)),
        )
    if isinstance(value, dict):
        stdout = value.get("stdout", value.get("output", value.get("text", "")))
        stderr = value.get("stderr", "")
        output = "\n".join(str(part) for part in (stdout, stderr) if part)
        exit_code = value.get(
            "exit_code",
            value.get("return_code", value.get("returncode", value.get("code", 0))),
        )
        if value.get("timed_out") and not exit_code:
            exit_code = 124
        return SandboxCommandResult(
            output=str(output or ""),
            exit_code=int(exit_code or 0),
            truncated=bool(value.get("truncated", False)),
        )

    output = getattr(value, "stdout", None)
    stderr = getattr(value, "stderr", None)
    if output is None:
        output = getattr(value, "output", None)
    if output is None:
        output = getattr(value, "text", None)
    if output is None:
        output = str(value or "")
    if stderr:
        output = "\n".join(str(part) for part in (output, stderr) if part)
    exit_code = getattr(value, "exit_code", None)
    if exit_code is None:
        exit_code = getattr(value, "return_code", None)
    if exit_code is None:
        exit_code = getattr(value, "returncode", 0)
    if getattr(value, "timed_out", False) and not exit_code:
        exit_code = 124
    return SandboxCommandResult(
        output=str(output or ""),
        exit_code=int(exit_code or 0),
        truncated=bool(getattr(value, "truncated", False)),
    )
