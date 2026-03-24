"""Qwen-style OpenForesight warmup environment for SkyRL."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from agents.basicAgent.agent import BasicAgent
from agents.basicAgent.search import SearchHandler
from agents.qwenAgent.agent import (
    QwenBasicAgent,
    qwen_execute_news_search,
    qwen_final_submit_instruction_text,
    qwen_optional_search_dates_from_parsed,
    qwen_parse_warmup_submit_outcomes,
)
from agents.qwenAgent.tools import build_action_tools
from agents.search_tools.lancedb import LanceDBSearchTool
from agents.utils.budget import BudgetSettings, BudgetTracker, estimate_budget_tokens
from environment.ansmatching import AnswerMatcher
from environment.forecast_metrics import (
    accuracy_rank_bonus,
    episodes_from_fs_metric_dicts,
    forecast_scalar_metrics,
    rollup_openforesight_eval_metrics,
)
from environment.scoring import BrierScorer
from environment.scoring.base import DailyPrediction
from inference.vllm import VLLMInference
from inference.openrouter import OpenRouterInference
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from skyrl_integration.vllm_qwen3_coder_text import extract_tool_calls_vllm_qwen3_coder

_SEARCH_TOOL_CACHE: Dict[Tuple[str, str, float, str], LanceDBSearchTool] = {}
_SEARCH_TOOL_LOCK = Lock()
_MATCHER_CACHE: Dict[tuple[str, str], AnswerMatcher] = {}
_MATCHER_LOCK = Lock()


def _to_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()

    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _to_str_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _get_config_value(env_config: Any, extras: Dict[str, Any], key: str, default: Any = None) -> Any:
    if hasattr(env_config, key):
        return getattr(env_config, key)
    if isinstance(env_config, dict) and key in env_config:
        return env_config[key]
    getter = getattr(env_config, "get", None)
    if callable(getter):
        value = getter(key, default)
        if value is not default:
            return value
    return extras.get(key, default)


def _extract_tool_calls_from_text(
    output_text: str,
    *,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> list[Dict[str, Any]]:
    """Parse assistant text into tool calls using vLLM ``qwen3_coder`` XML only."""
    return extract_tool_calls_vllm_qwen3_coder(output_text or "", tools=tools)


def _assistant_intended_search_news(text: str) -> bool:
    """True when assistant text looks like a ``search_news`` tool call (XML), for parse-failure tracking."""
    s = text or ""
    if "search_news" not in s:
        return False
    return "<tool_call" in s


def _get_or_create_search_tool(
    db_path: str,
    model_path: str,
    embedding_gpu_mem: float,
    aux_cuda_visible_devices: str,
) -> Optional[LanceDBSearchTool]:
    if not db_path:
        return None

    cache_key = (
        db_path,
        model_path or "",
        float(embedding_gpu_mem),
        aux_cuda_visible_devices or "",
    )
    with _SEARCH_TOOL_LOCK:
        tool = _SEARCH_TOOL_CACHE.get(cache_key)
        if tool is None:
            embedding_model = None
            if model_path:
                embedding_model = VLLMInference(
                    model_path,
                    gpu_memory_utilization=float(embedding_gpu_mem),
                    enable_prefix_caching=False,
                    cuda_visible_devices=aux_cuda_visible_devices or None,
                )
            tool = LanceDBSearchTool(db_path=db_path, embedding_model=embedding_model)
            _SEARCH_TOOL_CACHE[cache_key] = tool
    return tool


def _get_or_create_matcher(
    matching: str,
    matcher_model: str,
    *,
    cache_path: Optional[str] = None,
) -> AnswerMatcher:
    if matching != "openrouter":
        raise ValueError("internal: _get_or_create_matcher expects matching='openrouter'")
    if not matcher_model:
        raise ValueError("matcher model is required when matching='openrouter'")

    cache_key = (matching, matcher_model, str(cache_path or ""))
    with _MATCHER_LOCK:
        matcher = _MATCHER_CACHE.get(cache_key)
        if matcher is None:
            matcher = AnswerMatcher(
                OpenRouterInference(matcher_model),
                cache_path=cache_path,
            )
            _MATCHER_CACHE[cache_key] = matcher
    return matcher


class OpenForesightSearchWarmupEnv(BaseTextEnv):
    """Single-question Qwen3.5-style warmup for SkyRL (search + submit + next_day).

    Parity target: Qwen eval message shapes from ``QwenBasicAgent`` (no ``query_df``).
    When ``BudgetTracker`` is configured, remaining budget is surfaced the same way as
    eval (status lines on force-submit and formatted feedback).
    RL-only concerns stay here: ``BaseTextEnvStepOutput``, reward = Brier skill + ``acc_bonus_coef``
    × rank-based ``accuracy_rank_bonus``, format/skip penalties, and turning raw
    assistant text into tool calls via vLLM ``qwen3_coder`` XML only (eval uses structured
    ``tool_calls`` on the API).

    Eval rollups use ``environment.forecast_metrics`` (same definitions as
    ``SimulationEnvironment`` / ``daily_metrics.csv``).
    """

    def __init__(self, env_config: Dict[str, Any] = None, extras: Dict[str, Any] = None):
        super().__init__()
        env_config = env_config or {}
        extras = extras or {}

        reward_spec = extras.get("reward_spec", {}) or {}
        if "ground_truth" not in reward_spec:
            raise ValueError("reward_spec.ground_truth is required")

        self.question_id = str(extras.get("question_id", ""))
        self.question_title = str(extras.get("question_title", ""))
        self.ground_truth = str(reward_spec.get("ground_truth", "")).strip()

        raw_max_turns = extras.get("max_turns", 10)
        self.max_turns = None if raw_max_turns is None else int(raw_max_turns)
        if self.max_turns is not None and self.max_turns <= 0:
            self.max_turns = None
        self.search_db = str(extras.get("search_db", ""))
        self.embedding_model = str(extras.get("embedding_model", "")).strip()
        self.embedding_gpu_mem = float(extras.get("embedding_gpu_mem", 0.3))
        self.aux_cuda_visible_devices = str(extras.get("aux_cuda_visible_devices", "")).strip()
        self.search_topk = int(extras.get("search_topk", 5))
        self.search_type = str(extras.get("search_type", "hybrid")).strip().lower() or "hybrid"
        self.search_cutoff_days = int(extras.get("search_cutoff_days", 0))
        self.search_min_days = int(extras.get("search_min_days", 0))
        self.search_max_date = _to_date(extras.get("search_max_date"))
        # Single terminal penalty for any failed episode: bad tool/format, max_turns, budget exhausted,
        # next_day without submit, etc. (default -1).
        self.invalid_format_reward = float(extras.get("invalid_format_reward", -1.0))
        self.acc_bonus_coef = float(extras.get("acc_bonus_coef", 1.0))
        self.max_outcomes_per_question = int(_get_config_value(env_config, extras, "max_outcomes_per_question", 5))
        self.max_search_results = int(_get_config_value(env_config, extras, "max_search_results", self.search_topk))
        self.allow_substring_match = _to_str_bool(
            _get_config_value(env_config, extras, "allow_substring_match", True),
            True,
        )
        self.matching = str(_get_config_value(env_config, extras, "matching", "openrouter")).strip().lower()
        self.matcher_model = str(_get_config_value(env_config, extras, "matcher", "")).strip()
        self.matcher_cache_path = str(extras.get("matcher_cache_path", "") or "").strip() or None
        self.warmup_max_actions = _get_config_value(env_config, extras, "warmup_max_actions", None)
        self.warmup_max_total_tokens = _get_config_value(env_config, extras, "warmup_max_total_tokens", None)
        self.warmup_submit_reserve_tokens = int(
            _get_config_value(env_config, extras, "warmup_submit_reserve_tokens", 8192)
        )
        self.warmup_force_submit_threshold_tokens = int(
            _get_config_value(env_config, extras, "warmup_force_submit_threshold_tokens", 16384)
        )
        self.budget_model_path = str(_get_config_value(env_config, extras, "budget_model_path", "")).strip()

        sim_date = _to_date(extras.get("sim_date"))
        resolution_date = _to_date(extras.get("resolution_date"))
        self.current_date = sim_date or resolution_date
        if self.current_date is None:
            self.current_date = date.today()

        self.tool = _get_or_create_search_tool(
            self.search_db,
            self.embedding_model,
            self.embedding_gpu_mem,
            self.aux_cuda_visible_devices,
        )
        self.search_handler = SearchHandler(
            search_tool=self.tool,
            search_cutoff_days=self.search_cutoff_days,
            max_search_date=self.search_max_date,
        )
        self.search_handler.set_date(self.current_date)
        self.search_available = self.search_handler.is_available

        if self.matching != "openrouter":
            raise ValueError(
                "OpenForesightSearchWarmupEnv requires matching='openrouter' so Brier / top-1 use the "
                f"LLM matcher (got matching={self.matching!r}). Set matching: openrouter in the SkyRL YAML."
            )
        if not self.matcher_model:
            raise ValueError("OpenForesightSearchWarmupEnv requires a non-empty matcher model when matching='openrouter'.")
        self.matcher = _get_or_create_matcher(
            self.matching,
            self.matcher_model,
            cache_path=self.matcher_cache_path,
        )
        self._matcher_timing_before = self._get_matcher_timing_snapshot()
        self.scorer = BrierScorer()

        self.chat_history: ConversationType = []
        self.search_calls = 0
        self.successful_searches = 0
        self.search_parse_failed_turns: List[int] = []
        self.final_forecast: Optional[Dict[str, float]] = None
        self.final_truth_probability: float = 0.0
        self.final_reward: Optional[float] = None
        self.last_acc_bonus: float = 0.0
        self.is_correct: Optional[bool] = None
        self.matcher_metrics = self._zero_matcher_metrics()
        self.force_submit_prompt_injected = False
        self.budget: Optional[BudgetTracker] = None
        # Align SkyRL ``max_turns`` with action budget when ``warmup_max_actions`` is unset.
        if self.warmup_max_actions is None and self.max_turns is not None:
            self.warmup_max_actions = int(self.max_turns)
        if self.warmup_max_actions is not None:
            self.warmup_max_actions = int(self.warmup_max_actions)
        if self.warmup_max_total_tokens is not None:
            self.warmup_max_total_tokens = int(self.warmup_max_total_tokens)
        if self.warmup_max_actions is not None or self.warmup_max_total_tokens is not None:
            self.budget = BudgetTracker(
                BudgetSettings(
                    max_actions=self.warmup_max_actions,
                    max_total_tokens=self.warmup_max_total_tokens,
                    submit_reserve_tokens=self.warmup_submit_reserve_tokens,
                    force_submit_threshold_tokens=self.warmup_force_submit_threshold_tokens,
                ),
                token_estimator=lambda payload: estimate_budget_tokens(payload, model_name=self.budget_model_path),
            )

    @staticmethod
    def _zero_matcher_metrics() -> Dict[str, float]:
        return {
            "matcher_count": 0,
            "matcher_total_seconds": 0.0,
            "matcher_total_cost": 0.0,
        }

    def _get_matcher_timing_snapshot(self) -> Dict[str, float]:
        if self.matcher is None:
            return self._zero_matcher_metrics()
        if hasattr(self.matcher, "get_timing_snapshot"):
            snapshot = self.matcher.get_timing_snapshot() or {}
            return {
                "matcher_count": int(snapshot.get("matcher_count", 0)),
                "matcher_total_seconds": float(snapshot.get("matcher_total_seconds", 0.0)),
                "matcher_total_cost": float(snapshot.get("matcher_total_cost", 0.0)),
            }
        return self._zero_matcher_metrics()

    def _update_matcher_metrics(self) -> None:
        before = self._matcher_timing_before
        after = self._get_matcher_timing_snapshot()
        self.matcher_metrics = {
            "matcher_count": max(0, int(after["matcher_count"]) - int(before["matcher_count"])),
            "matcher_total_seconds": max(
                0.0,
                float(after["matcher_total_seconds"]) - float(before["matcher_total_seconds"]),
            ),
            "matcher_total_cost": max(
                0.0,
                float(after["matcher_total_cost"]) - float(before["matcher_total_cost"]),
            ),
        }

    def _user_message(self, content: str) -> Dict[str, str]:
        return {"role": "user", "content": content}

    def _budget_metadata(self) -> Dict[str, Any]:
        if self.budget is None:
            return {}
        return self.budget.metadata()

    def _build_budget_payload(self) -> Dict[str, Any]:
        tool_schemas = build_action_tools(
            enable_query=False,
            enable_search=self.search_available,
            max_outcomes_per_question=self.max_outcomes_per_question,
            max_search_results=self.max_search_results,
            search_chunk_tokens=self.search_handler.chunk_tokens,
        )
        return {"messages": self.chat_history, "tools": tool_schemas}

    def _tool_schemas_for_parse(self) -> List[Dict[str, Any]]:
        """Same OpenAI tool schemas as eval; used to mirror vLLM param typing (``qwen3_coder``)."""
        return build_action_tools(
            enable_query=False,
            enable_search=self.search_available,
            max_outcomes_per_question=self.max_outcomes_per_question,
            max_search_results=self.max_search_results,
            search_chunk_tokens=self.search_handler.chunk_tokens,
        )

    def _build_final_submit_instruction(self) -> str:
        return qwen_final_submit_instruction_text(
            self.question_id,
            self.budget,
            forbid_query_df=False,
        )

    def init(self, prompt: ConversationType):
        self.chat_history = [dict(message) for message in prompt]
        if self.budget is not None:
            self.budget.bootstrap_context(self._build_budget_payload())
        return prompt, {}

    def _terminal_metadata_core(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question_title": self.question_title,
            "turns": self.turns,
            "search_calls": self.search_calls,
            "successful_searches": self.successful_searches,
            "matching": self.matching,
            "matcher": self.matcher_model,
            **self.matcher_metrics,
            **self._budget_metadata(),
        }

    def _nonterminal_step_metadata(self, phase: str) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question_title": self.question_title,
            "turn": self.turns,
            "phase": phase,
            "search_calls": self.search_calls,
            "matching": self.matching,
            **self._budget_metadata(),
        }

    def _finish_budget_exhausted(self, reason: str) -> BaseTextEnvStepOutput:
        self.final_forecast = None
        self.final_truth_probability = 0.0
        self.last_acc_bonus = 0.0
        self.final_reward = self.invalid_format_reward
        self.is_correct = False
        self._update_matcher_metrics()
        return BaseTextEnvStepOutput(
            observations=[],
            reward=self.invalid_format_reward,
            done=True,
            metadata={
                **self._terminal_metadata_core(),
                "is_correct": self.is_correct,
                "final_forecast": self.final_forecast,
                "final_truth_probability": self.final_truth_probability,
                "ground_truth": self.ground_truth,
                "reward": self.final_reward,
                "parse_error": reason,
            },
        )

    def _finish_invalid(self, parse_error: str) -> BaseTextEnvStepOutput:
        self.final_forecast = None
        self.final_truth_probability = 0.0
        self.last_acc_bonus = 0.0
        self.final_reward = self.invalid_format_reward
        self.is_correct = False
        self._update_matcher_metrics()
        return BaseTextEnvStepOutput(
            observations=[],
            reward=self.invalid_format_reward,
            done=True,
            metadata={
                **self._terminal_metadata_core(),
                "is_correct": self.is_correct,
                "final_forecast": self.final_forecast,
                "final_truth_probability": self.final_truth_probability,
                "ground_truth": self.ground_truth,
                "parse_error": parse_error,
            },
        )

    def _continue_with_feedback(
        self,
        *,
        content: str,
        feedback_kind: str,
        phase: str,
    ) -> BaseTextEnvStepOutput:
        if self.max_turns is not None and self.turns >= self.max_turns:
            return self._finish_invalid("max_turns exceeded")

        formatted = content
        if self.budget is not None:
            formatted = self.budget.format_feedback(content, include_exhaustion_warning=False)

        if feedback_kind == "invalid":
            message = self._user_message(formatted)
            self.chat_history.append(message)
        elif feedback_kind == "search":
            message = QwenBasicAgent._append_tool_output_message(
                self.chat_history,
                tool_call=None,
                tool_name="search_news",
                output=formatted,
            )
        elif feedback_kind == "submit_error":
            message = QwenBasicAgent._append_tool_output_message(
                self.chat_history,
                tool_call=None,
                tool_name="submit_forecasts",
                output=formatted,
            )
        else:
            raise ValueError(f"Unknown feedback_kind: {feedback_kind!r}")

        if self.budget is not None:
            self.budget.record_appended_item(message)

        observations: ConversationType = [message]
        if self.budget is not None and self.budget.should_force_submit() and not self.force_submit_prompt_injected:
            final_instruction = self._user_message(self._build_final_submit_instruction())
            self.chat_history.append(final_instruction)
            self.budget.record_appended_item(final_instruction)
            observations.append(final_instruction)
            self.force_submit_prompt_injected = True

        if self.budget is not None and self.budget.is_exhausted():
            return self._finish_budget_exhausted("budget exhausted before submitting a forecast")

        return BaseTextEnvStepOutput(
            observations=observations,
            reward=0.0,
            done=False,
            metadata=self._nonterminal_step_metadata(phase),
        )

    def _run_search(self, parsed) -> BaseTextEnvStepOutput:
        self.search_calls += 1
        min_date, max_date = qwen_optional_search_dates_from_parsed(parsed)
        if min_date is None and self.search_min_days > 0:
            min_date = self.current_date - timedelta(days=self.search_min_days)
        effect = qwen_execute_news_search(
            parsed,
            self.search_handler,
            max_results=self.max_search_results,
            search_type=self.search_type,
            min_date=min_date,
            max_date=max_date,
        )
        if effect.successful_hit:
            self.successful_searches += 1
        return self._continue_with_feedback(
            content=effect.feedback,
            feedback_kind="search",
            phase=effect.phase,
        )

    def _finish_with_submit(self, outcomes: Dict[str, float]) -> BaseTextEnvStepOutput:
        prediction = DailyPrediction(
            agent_id="skyrl_policy",
            question_id=self.question_id,
            day=self.current_date,
            outcomes=outcomes,
        )
        brier_skill, top1_correct, truth_prob = forecast_scalar_metrics(
            prediction,
            self.ground_truth,
            matcher=self.matcher,
            scorer=self.scorer,
            question_id=self.question_id,
            question_title=self.question_title,
        )
        acc_raw = accuracy_rank_bonus(
            prediction,
            self.ground_truth,
            matcher=self.matcher,
            question_id=self.question_id,
            question_title=self.question_title,
        )
        acc_term = float(self.acc_bonus_coef) * float(acc_raw)
        total_reward = float(brier_skill) + acc_term

        self._update_matcher_metrics()
        self.final_forecast = dict(outcomes)
        self.final_truth_probability = float(truth_prob)
        self.last_acc_bonus = float(acc_term)
        self.final_reward = float(total_reward)
        self.is_correct = bool(top1_correct)

        return BaseTextEnvStepOutput(
            observations=[],
            reward=float(total_reward),
            done=True,
            metadata={
                **self._terminal_metadata_core(),
                "is_correct": self.is_correct,
                "final_forecast": self.final_forecast,
                "final_truth_probability": self.final_truth_probability,
                "ground_truth": self.ground_truth,
                "brier_skill": float(brier_skill),
                "acc_bonus": float(acc_term),
                "reward": self.final_reward,
                "parse_error": None,
            },
        )

    def _run_submit(self, parsed) -> BaseTextEnvStepOutput:
        outcomes, detail = qwen_parse_warmup_submit_outcomes(parsed, self.question_id)
        if outcomes is None:
            return self._continue_with_feedback(
                content=f"SUBMIT ERROR: {detail}",
                feedback_kind="submit_error",
                phase="submit_error",
            )
        return self._finish_with_submit(outcomes)

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        assistant_message = {"role": "assistant", "content": action or ""}
        self.chat_history.append(assistant_message)
        if self.budget is not None:
            self.budget.consume_action()
            self.budget.record_appended_item(assistant_message)

        tool_calls = _extract_tool_calls_from_text(
            action or "",
            tools=self._tool_schemas_for_parse(),
        )
        parsed, _, tool_calls = BasicAgent.tool_calls_to_parsed_action(
            tool_calls,
            assistant_text=action or "",
        )
        intended_search = _assistant_intended_search_news(action or "")
        search_reached = (
            len(tool_calls) == 1
            and parsed is not None
            and getattr(parsed, "action_type", None) == "search"
        )
        if intended_search and not search_reached:
            self.search_parse_failed_turns.append(self.turns)

        if len(tool_calls) > 1:
            return self._continue_with_feedback(
                content="Only one tool call is allowed per turn.",
                feedback_kind="invalid",
                phase="invalid_action",
            )

        if parsed is None:
            return self._continue_with_feedback(
                content="No tool call detected. You MUST call exactly one tool each turn.",
                feedback_kind="invalid",
                phase="invalid_action",
            )

        if parsed.action_type == "search":
            return self._run_search(parsed)

        if parsed.action_type == "submit":
            return self._run_submit(parsed)

        if parsed.action_type == "next":
            self.final_forecast = None
            self.final_truth_probability = 0.0
            self.last_acc_bonus = 0.0
            self.final_reward = self.invalid_format_reward
            self.is_correct = False
            self._update_matcher_metrics()
            return BaseTextEnvStepOutput(
                observations=[],
                reward=self.invalid_format_reward,
                done=True,
                metadata={
                    **self._terminal_metadata_core(),
                    "is_correct": self.is_correct,
                    "final_forecast": self.final_forecast,
                    "final_truth_probability": self.final_truth_probability,
                    "ground_truth": self.ground_truth,
                    "reward": self.final_reward,
                    "parse_error": "next_day called before submitting a forecast",
                },
            )

        error = parsed.error or "Invalid action."
        return self._continue_with_feedback(
            content=f"Invalid action: {error}",
            feedback_kind="invalid",
            phase="invalid_action",
        )

    def get_metrics(self) -> Dict[str, Any]:
        valid_submit = bool(self.final_forecast and len(self.final_forecast) > 0)
        if valid_submit:
            prediction = DailyPrediction(
                agent_id="skyrl_policy",
                question_id=self.question_id,
                day=self.current_date,
                outcomes=dict(self.final_forecast),
            )
            brier_f, top1_b, truth_f = forecast_scalar_metrics(
                prediction,
                self.ground_truth,
                matcher=self.matcher,
                scorer=self.scorer,
                question_id=self.question_id,
                question_title=self.question_title,
            )
            acc_raw = accuracy_rank_bonus(
                prediction,
                self.ground_truth,
                matcher=self.matcher,
                question_id=self.question_id,
                question_title=self.question_title,
            )
            acc_term = float(self.acc_bonus_coef) * float(acc_raw)
            reward_f, top1, truth_f = float(brier_f), 1.0 if top1_b else 0.0, float(truth_f)
        else:
            brier_f, acc_term, reward_f, top1, truth_f = 0.0, 0.0, 0.0, 0.0, 0.0
        return {
            # Consumed by aggregate_metrics (stripped before default matcher aggregation).
            "fs_valid_submit": 1.0 if valid_submit else 0.0,
            "fs_brier_skill": float(brier_f) if valid_submit else 0.0,
            "fs_acc_bonus": float(acc_term) if valid_submit else 0.0,
            "fs_top1_correct": top1,
            "fs_truth_prob": truth_f,
            "search_parse_failed_count": float(len(self.search_parse_failed_turns)),
            "search_calls": self.search_calls,
            "successful_searches": self.successful_searches,
            "turns": self.turns,
            "matching": self.matching,
            "is_correct": self.is_correct,
            "final_truth_probability": self.final_truth_probability,
            "final_reward": self.final_reward,
            **self.matcher_metrics,
            **self._budget_metadata(),
            "search_parse_failed_turns": list(self.search_parse_failed_turns),
        }

    @staticmethod
    def aggregate_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Roll up via ``environment.forecast_metrics`` plus default matcher averages."""
        from skyrl_gym.metrics import default_aggregate_metrics

        if not metrics:
            return {}

        out: Dict[str, Any] = dict(rollup_openforesight_eval_metrics(episodes_from_fs_metric_dicts(metrics)))

        _omit_from_default = {"is_correct", "final_truth_probability", "final_reward", "search_parse_failed_turns"}
        stripped: List[Dict[str, Any]] = []
        for m in metrics:
            stripped.append(
                {
                    k: v
                    for k, v in m.items()
                    if not str(k).startswith("fs_") and k not in _omit_from_default
                }
            )

        dm = default_aggregate_metrics(stripped)
        out.update(dm)
        if "search_parse_failed_count" in dm:
            out["avg_search_parse_failures"] = float(dm["search_parse_failed_count"])
        return out
