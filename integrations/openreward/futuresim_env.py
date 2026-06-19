"""OpenReward-native Futuresim environment."""

from __future__ import annotations

import json
import os
import shlex
import tempfile
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any, Optional

from pydantic import BaseModel, Field

from futuresim_agents.minimalHarnessAgent.prompts.prompt import build_system_prompt
from futuresim_agents.search_tools.handler import SearchHandler
from futuresim_agents.search_tools.openreward import OpenRewardSearchTool
from integrations.adapter_runtime import (
    FuturesimAdapterConfig,
    FuturesimAdapterRuntime,
    as_bool,
    parse_iso_date,
)

try:
    from openreward import AsyncOpenReward, SandboxBucketConfig, SandboxSettings
    from openreward.environments import (
        Environment,
        JSONObject,
        TextBlock,
        ToolOutput,
        tool,
    )
except ImportError as exc:  # pragma: no cover - optional integration dependency
    _OPENREWARD_IMPORT_ERROR: Optional[BaseException] = exc
else:
    _OPENREWARD_IMPORT_ERROR = None


class SearchNewsParams(BaseModel):
    query: str = Field(..., description="News search query.")
    from_date: Optional[str] = Field(None, description="Optional earliest article date, YYYY-MM-DD.")
    to_date: Optional[str] = Field(None, description="Optional latest article date, YYYY-MM-DD.")


class SubmitForecastsParams(BaseModel):
    question_id: str = Field(..., description="Question id from market.csv.")
    outcomes: dict[str, float] = Field(
        ...,
        description='Outcome probabilities, e.g. {"Yes": 0.7, "No": 0.3}.',
    )


class NextDayParams(BaseModel):
    pass


def _default_task_rows() -> list[dict[str, Any]]:
    raw = os.environ.get("FSIM_OPENREWARD_TASKS") or os.environ.get("FSIM_HOSTED_TASKS")
    if raw:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else [parsed]

    sandbox_environment = (
        os.environ.get("OPENREWARD_ENVIRONMENT")
        or os.environ.get("OPENREWARD_ENVIRONMENT_NAME")
        or os.environ.get("FSIM_OPENREWARD_SANDBOX_ENV")
        or "ShashwatGoel/futuresim"
    )
    sandbox = {
        "image": "generalreasoning/python-ds:3.12-tools",
        "machine_size": "2:8",
        "block_network": True,
        "mount_articles": False,
    }
    if sandbox_environment:
        sandbox["environment"] = sandbox_environment

    return [
        {
            "example_id": "futuresim-openreward",
            "futuresim": FuturesimAdapterConfig().to_task_spec(),
            "openreward_sandbox": sandbox,
        }
    ]


if _OPENREWARD_IMPORT_ERROR is None:

    class FuturesimOpenRewardEnv(Environment):
        """Futuresim as an OpenReward harness-toolset environment."""

        def __init__(
            self,
            task_spec: Optional[JSONObject] = None,
            secrets: Optional[dict[str, str]] = None,
        ) -> None:
            super().__init__(task_spec, secrets)
            self.secrets = dict(secrets or {})
            self.task_spec = dict(task_spec or {})
            if self.secrets.get("OPENROUTER_API_KEY") and not os.environ.get("OPENROUTER_API_KEY"):
                os.environ["OPENROUTER_API_KEY"] = self.secrets["OPENROUTER_API_KEY"]
            self.config = FuturesimAdapterConfig.from_mapping(self.task_spec)
            api_key = (
                self.secrets.get("OPENREWARD_API_KEY")
                or self.secrets.get("api_key")
                or os.environ.get("OPENREWARD_API_KEY")
            )
            sandbox_cfg = dict(self.task_spec.get("openreward_sandbox") or {})
            environment_name = (
                sandbox_cfg.get("environment")
                or self.secrets.get("environment")
                or os.environ.get("OPENREWARD_ENVIRONMENT")
                or os.environ.get("OPENREWARD_ENVIRONMENT_NAME")
            )
            if not api_key:
                raise ValueError("OpenReward integration requires OPENREWARD_API_KEY or secrets['api_key'].")
            if not environment_name:
                raise ValueError(
                    "OpenReward integration requires OPENREWARD_ENVIRONMENT, "
                    "OPENREWARD_ENVIRONMENT_NAME, secrets['environment'], or "
                    "task_spec['openreward_sandbox']['environment']."
                )

            self.runtime = FuturesimAdapterRuntime(self.config)
            self.search_handler = SearchHandler(
                OpenRewardSearchTool.from_env(api_key=api_key),
                search_cutoff_days=self.config.article_search_cutoff_days,
            )
            self._sandbox_started = False
            self._day_started = False
            self.mount_articles = as_bool(sandbox_cfg.get("mount_articles", False))
            self._feedback_seen_qids: set[str] = set()
            self._feedback_brier_sum = 0.0
            self._feedback_tw_sum = 0.0
            self._feedback_accuracy_count = 0
            self._feedback_resolved_count = 0
            self._today_predictions: list[dict[str, Any]] = []

            bucket_config = sandbox_cfg.get("bucket_config")
            self.sandbox = AsyncOpenReward(api_key=api_key).sandbox(
                SandboxSettings(
                    environment=environment_name,
                    image=sandbox_cfg.get("image", "generalreasoning/python-ds:3.12-tools"),
                    machine_size=sandbox_cfg.get("machine_size", "2:8"),
                    block_network=as_bool(sandbox_cfg.get("block_network", True)),
                    env=sandbox_cfg.get("env"),
                    bucket_config=SandboxBucketConfig(**bucket_config) if bucket_config else None,
                    sidecars=sandbox_cfg.get("sidecars"),
                    host_aliases=sandbox_cfg.get("host_aliases"),
                )
            )

        async def setup(self) -> None:
            await self._ensure_sandbox_started()
            if not self._day_started:
                await self._start_day()

        async def teardown(self) -> None:
            self.runtime.close()
            if self._sandbox_started:
                await self.sandbox.stop()
                self._sandbox_started = False

        async def get_prompt(self) -> list[TextBlock]:
            await self._ensure_sandbox_started()
            if not self._day_started:
                await self._start_day()
            forecast_interface = self.runtime.forecast_interface()
            questions = forecast_interface.list_questions()
            resolved = getattr(forecast_interface, "resolved_questions", [])
            prompt = build_system_prompt(
                workspace=self.runtime.workspace_path,
                current_date=self.runtime.env.current_date,
                start_date=self.runtime.env.start_date,
                end_date=parse_iso_date(self.config.end_date) or self.runtime.env.current_date,
                source_context=getattr(forecast_interface, "source_context", "") or "",
                source_name=getattr(forecast_interface, "source_name", "openforesight"),
                num_questions=len(questions) + len(resolved),
                num_active=len(questions),
                num_resolved=len(resolved),
                max_outcomes_per_question=self.config.max_outcomes_per_question,
                search_cutoff_days=self.config.article_search_cutoff_days,
                timegap_days=self.config.timegap_days,
                new_articles_count=None,
                last_active_date=getattr(forecast_interface, "last_active_date", None),
                next_active_date=getattr(forecast_interface, "next_active_date", None),
                handholding_version=self.config.handholding_version,
                prompt_mode=self.config.prompt_mode,
                article_files_available=self.mount_articles,
                tool_prefix="mcp__openreward__",
            )
            return [TextBlock(text=prompt)]

        @tool
        async def search_news(self, params: SearchNewsParams) -> ToolOutput:
            """Search date-gated news evidence for the current simulation day."""
            self.search_handler.set_date(self.runtime.env.current_date)
            try:
                min_date = parse_iso_date(params.from_date)
                max_date = parse_iso_date(params.to_date)
            except ValueError as exc:
                return self._text(f"Search error: {exc}")
            text, err = self.search_handler.search(
                params.query,
                max_results=5,
                search_type="hybrid",
                min_date=min_date,
                max_date=max_date,
            )
            return ToolOutput(blocks=[TextBlock(text=f"Search error: {err}" if err else text)])

        @tool
        async def submit_forecasts(self, params: SubmitForecastsParams) -> ToolOutput:
            """Submit one probabilistic forecast for an active question."""
            question_id = params.question_id.strip()
            active_qids = {str(q.qid) for q in self.runtime.active_questions}
            if question_id not in active_qids:
                return self._text(f"Error: question_id {question_id!r} is not active in market.csv.")
            if not params.outcomes:
                return self._text("Error: outcomes must be a non-empty object.")
            if len(params.outcomes) > self.config.max_outcomes_per_question:
                return self._text(
                    f"Error: {len(params.outcomes)} outcomes exceeds maximum "
                    f"{self.config.max_outcomes_per_question}."
                )

            outcomes = {str(outcome): float(prob) for outcome, prob in params.outcomes.items()}
            total = sum(outcomes.values())
            if total > 1.0 + 1e-6:
                return self._text(f"Error: probabilities sum to {total:.4f}, which exceeds 1.0.")
            bad = [(outcome, prob) for outcome, prob in outcomes.items() if prob < 0 or prob > 1]
            if bad:
                outcome, prob = bad[0]
                return self._text(f"Error: probability {prob} for {outcome!r} is outside [0, 1].")

            self.runtime.submit_predictions([{"question_id": question_id, "outcomes": outcomes}])
            self._today_predictions.append({"question_id": question_id, "outcomes": outcomes})
            return self._text(f"Prediction recorded for question {question_id}: {outcomes}")

        @tool
        async def next_day(self, params: NextDayParams) -> ToolOutput:
            """Advance the simulation after the agent is done with the current day."""
            previous_date = self.runtime.env.current_date
            result = self.runtime.finish_day()
            await self._upload_prediction_snapshot(previous_date)
            self._day_started = False
            if result.done:
                feedback = self._feedback_recap(
                    previous_date.isoformat(),
                    active_count=0,
                    mutate=True,
                )
                text = "\n\n".join(
                    part for part in (
                        feedback,
                        "Simulation is complete. No more days to process.",
                        f"Final reward: {result.reward:.6f}",
                    )
                    if part
                )
                return ToolOutput(
                    blocks=[TextBlock(text=text)],
                    metadata={"reward": result.reward},
                    reward=result.reward,
                    finished=True,
                )

            await self._start_day()
            new_date = self.runtime.env.current_date
            parts = [
                f"Day advanced to {new_date.isoformat()}.",
                self._new_articles_message(previous_date, new_date),
            ]
            feedback = self._feedback_recap(
                f"{previous_date.isoformat()} -> {new_date.isoformat()}",
                active_count=len(self.runtime.active_questions),
                mutate=True,
            )
            if feedback:
                parts.append("\n" + feedback)
            return self._text("\n".join(parts))

        async def _start_day(self) -> None:
            if self._day_started:
                return
            self.runtime.begin_day()
            self._today_predictions = []
            self.search_handler.set_date(self.runtime.env.current_date)
            await self._upload_market_csv()
            if self.mount_articles:
                await self._upload_articles()
            self._day_started = True

        async def _ensure_sandbox_started(self) -> None:
            if not self._sandbox_started:
                await self.sandbox.start()
                self._sandbox_started = True

        async def _upload_market_csv(self) -> None:
            remote_path = self.runtime.market_remote_path
            with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
                market_path = Path(tmp.name)
            self.runtime.write_agent_market_csv(market_path)
            await self.sandbox.run(
                f"mkdir -p {shlex.quote(str(PurePosixPath(remote_path).parent))} "
                f"{shlex.quote(self.runtime.remote_path('memory'))} "
                f"{shlex.quote(self.runtime.remote_path('predictions'))} "
                f"&& rm -f {shlex.quote(remote_path)}"
            )
            try:
                await self.sandbox.upload(market_path, remote_path)
                await self.sandbox.run(f"chmod 444 {shlex.quote(remote_path)}")
            finally:
                market_path.unlink(missing_ok=True)

        async def _upload_prediction_snapshot(self, sim_date: date) -> None:
            remote_path = self.runtime.remote_path("predictions", f"{sim_date.isoformat()}.json")
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
                local_path = Path(tmp.name)
                json.dump(self._today_predictions, tmp)
            await self.sandbox.run(f"mkdir -p {shlex.quote(str(PurePosixPath(remote_path).parent))}")
            try:
                await self.sandbox.upload(local_path, remote_path)
                await self.sandbox.run(f"chmod 444 {shlex.quote(remote_path)}")
            finally:
                local_path.unlink(missing_ok=True)

        async def _upload_articles(self) -> None:
            uploads, marker = self.runtime.prepare_article_uploads()
            for upload in uploads:
                await self.sandbox.run(f"mkdir -p {shlex.quote(str(PurePosixPath(upload.remote_path).parent))}")
                await self.sandbox.upload(upload.local_path, upload.remote_path)
            self.runtime.commit_article_uploads(marker)

        def _new_articles_message(self, previous_date: Optional[date], current_date: Optional[date]) -> str:
            if previous_date is None or current_date is None:
                return "New articles are available via the search_news tool."
            if not self.mount_articles:
                return "New articles have been published since your last update and are available via the search_news tool."

            corpus = self.runtime.env.article_corpus
            if corpus is not None and corpus.is_available:
                files = corpus.visible_files(
                    current_date,
                    start_date=self.runtime.env.start_date,
                    search_cutoff_days=self.config.article_search_cutoff_days,
                    freeze_after_start=self.config.article_freeze_after_start,
                    since_date=previous_date,
                )
                count = 0
                for item in files:
                    with open(item.source_path) as f:
                        count += sum(1 for _ in f)
                if count:
                    return (
                        f"{count:,} new articles have been published since your last update "
                        "and are available via the search_news tool. New date "
                        "directories are also now present in articles/."
                    )
            return (
                f"New articles are available for {current_date.isoformat()} in articles/ "
                "or via the search_news tool."
            )

        def _feedback_recap(self, previous_label: str, active_count: int, *, mutate: bool) -> str:
            events = [
                ev for ev in self.runtime.env.resolution_events
                if ev.get("qid") not in self._feedback_seen_qids
            ]
            if not events:
                return ""

            lines = [f"## RESULTS SINCE YOUR LAST SESSION ({previous_label})", ""]
            brier_sum = self._feedback_brier_sum
            tw_sum = self._feedback_tw_sum
            accuracy_count = self._feedback_accuracy_count
            resolved_count = self._feedback_resolved_count

            for ev in events:
                qid = str(ev.get("qid", ev.get("question_id", "?")))
                title = ev.get("title", "")
                truth = ev.get("ground_truth", "?")
                stats = (ev.get("agents") or {}).get(self.runtime.agent_id)
                if stats:
                    brier = float(stats.get("brier") or 0.0)
                    tw_peer = float(stats.get("tw_peer") or 0.0)
                    best_outcome = stats.get("best_outcome", "?")
                    best_prob = float(stats.get("best_prob") or 0.0)
                    pred = (
                        self.runtime.env.resolved_agent_predictions
                        .get(qid, {})
                        .get(self.runtime.agent_id, {})
                        .get("outcomes", {})
                    )
                    if pred:
                        dist = "{" + ", ".join(
                            f"{outcome}: {float(prob):.2f}"
                            for outcome, prob in sorted(pred.items(), key=lambda item: -float(item[1]))
                        ) + "}"
                    else:
                        dist = f"{{{best_outcome}: {best_prob:.2f}}}"
                    brier_sum += brier
                    tw_sum += tw_peer
                    resolved_count += 1
                    if stats.get("is_accurate"):
                        accuracy_count += 1
                    lines.append(
                        f"- \"{title}\"\n"
                        f"  Your prediction distribution: {dist} | Truth: {truth}\n"
                        f"  Brier: {brier:+.2f} | TW-Score: {tw_peer:+.2f}"
                    )
                else:
                    resolved_count += 1
                    lines.append(f"- \"{title}\" → {truth}")
                if mutate:
                    self._feedback_seen_qids.add(qid)

            denom = resolved_count + active_count
            if denom > 0:
                avg_brier = brier_sum / denom
                accuracy = accuracy_count / denom * 100
                total_preds = self.runtime.env.agent_questions.get(self.runtime.agent_id, 0)
                lines.extend([
                    "",
                    "## YOUR CUMULATIVE PERFORMANCE TILL TODAY",
                    f"- Total Predictions: {total_preds} ({resolved_count} resolved, {active_count} active)",
                    f"- accuracy: {accuracy:.1f}% | brier skill score: {avg_brier:.3f} | time weighted score: {tw_sum:.2f}",
                    "  accuracy = fraction of ALL questions (resolved + active) where your top outcome matched the truth (0 credit for questions you did not predict or that have not resolved); "
                    "brier skill score = mean brier skill score across ALL questions (0 for questions you did not predict or that have not resolved); "
                    "time weighted score = sum of brier skill scores across all resolved questions, across all days you held your respective predictions",
                ])

            if mutate:
                self._feedback_brier_sum = brier_sum
                self._feedback_tw_sum = tw_sum
                self._feedback_accuracy_count = accuracy_count
                self._feedback_resolved_count = resolved_count
            return "\n".join(lines)

        def _text(self, text: str) -> ToolOutput:
            return ToolOutput(blocks=[TextBlock(text=text)])

        @classmethod
        def list_tasks(cls, split: str) -> list[JSONObject]:
            if split not in {"train", "validation", "test"}:
                raise ValueError(f"Unknown split: {split}")
            return _default_task_rows()

        @classmethod
        def list_splits(cls) -> list[str]:
            return ["train", "validation", "test"]

else:

    class FuturesimOpenRewardEnv:  # pragma: no cover - optional dependency placeholder
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("OpenReward integration requires `openreward`.") from _OPENREWARD_IMPORT_ERROR
