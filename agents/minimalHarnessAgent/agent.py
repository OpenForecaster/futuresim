"""
MinimalHarnessAgent — forecasting agent that delegates to a single persistent
harness CLI session (claude_code, opencode, or codex).

The first call to act() spawns the harness + its MCP server.  Subsequent calls
signal the MCP server (via a file-based continue signal) so the already-running
harness session can resume with the next simulation day.
"""

import hashlib
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from agents.base import BaseAgent
from agents.minimalHarnessAgent.prompt import build_system_prompt
from agents.utils.output_logger import AgentOutputLogger
from environment.interfaces import PredictionSubmission

logger = logging.getLogger(__name__)


def _resolve_socat_cmd() -> str:
    return shutil.which("socat") or ("/usr/bin/socat" if os.path.exists("/usr/bin/socat") else "socat")


def _resolve_sandbox_socat_cmd() -> str:
    for candidate in ("/usr/bin/socat", "/bin/socat"):
        if os.path.exists(candidate):
            return candidate
    return "socat"


def _read_json_file(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def _read_text_file(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text()
    except OSError:
        return ""


def _read_jsonl_file(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    if not path.exists():
        return events
    try:
        with open(path) as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    events.append({
                        "type": "parse_error",
                        "line_no": line_no,
                        "text": line,
                    })
    except OSError:
        return events
    return events


def _render_codex_response(events: List[Dict[str, Any]], predictions: List[Dict[str, Any]]) -> str:
    messages: List[str] = []
    for ev in events:
        item = ev.get("item") if isinstance(ev, dict) else None
        if not isinstance(item, dict):
            continue
        if ev.get("type") == "item.completed" and item.get("type") == "agent_message":
            text = item.get("text")
            if text:
                messages.append(str(text))

    if predictions:
        messages.append("Submitted predictions: " + json.dumps(predictions, sort_keys=True))
    return "\n\n".join(messages)


def _rewrite_warmup_index_for_aggregated_logs(agent_dir: Path) -> None:
    index_path = agent_dir / "warmup_logs" / "index.jsonl"
    if not index_path.exists():
        return
    records = _read_jsonl_file(index_path)
    with open(index_path, "w") as f:
        for record in records:
            record.pop("log_dir", None)
            record["processed_output_log"] = str(agent_dir / "model_outputs.jsonl")
            record["raw_output_log"] = str(agent_dir / "model_raw_warmup.jsonl")
            f.write(json.dumps(record) + "\n")


def aggregate_codex_warmup_logs(
    agent_id: str,
    agent_dir: Any,
    *,
    remove_per_question_logs: bool = True,
    append: bool = False,
) -> int:
    """Aggregate per-question Codex warmup transcripts into standard agent logs."""
    agent_dir = Path(agent_dir)
    warmup_logs_dir = agent_dir / "warmup_logs"
    if not warmup_logs_dir.exists():
        return 0

    log_dirs = sorted(p for p in warmup_logs_dir.iterdir() if p.is_dir())
    if not log_dirs:
        _rewrite_warmup_index_for_aggregated_logs(agent_dir)
        return 0

    output_logger = AgentOutputLogger(agent_id, str(agent_dir), append=append)
    written = 0
    try:
        for log_dir in log_dirs:
            state = _read_json_file(log_dir / "state.json", {}) or {}
            questions = state.get("questions") or []
            qid = str((questions[0] or {}).get("qid") if questions else log_dir.name)
            sim_date = state.get("current_date", "")
            prompt = _read_text_file(log_dir / "prompt.md")
            predictions = _read_json_file(log_dir / "predictions.json", []) or []
            events = _read_jsonl_file(log_dir / "codex_stdout.jsonl")
            stderr_text = _read_text_file(log_dir / "codex_stderr.log")
            prompt_mode = state.get("prompt_mode", "warmup")
            raw_stream = "warmup"
            thread_id = next(
                (
                    ev.get("thread_id")
                    for ev in events
                    if ev.get("type") == "thread.started" and ev.get("thread_id")
                ),
                None,
            )
            tool_calls = [
                ev.get("item", {})
                for ev in events
                if isinstance(ev.get("item"), dict)
                and ev.get("item", {}).get("type") == "mcp_tool_call"
                and ev.get("type") == "item.completed"
            ]
            response = _render_codex_response(events, predictions)
            metadata = {
                "backend": "codex",
                "harness_backend": "codex",
                "prompt_mode": prompt_mode,
                "raw_stream": raw_stream,
                "qid": qid,
                "current_date": sim_date,
                "search_current_date": state.get("search_current_date", sim_date),
                "thread_id": thread_id,
                "predictions": predictions,
                "codex_event_count": len(events),
                "codex_tool_call_count": len(tool_calls),
                "codex_submit_call_count": sum(
                    1 for item in tool_calls if item.get("tool") == "submit_forecasts"
                ),
                "_logger_raw_input_delta": prompt,
                "_logger_raw_response": events,
                "_logger_raw_metadata": {
                    "backend": "codex",
                    "harness_backend": "codex",
                    "prompt_mode": prompt_mode,
                    "raw_stream": raw_stream,
                    "qid": qid,
                    "current_date": sim_date,
                    "search_current_date": state.get("search_current_date", sim_date),
                    "thread_id": thread_id,
                    "predictions": predictions,
                    "codex_stderr": stderr_text,
                    "mcp_relay_log": _read_text_file(log_dir / "mcp_relay.log"),
                    "state": state,
                },
            }
            output_logger.log_model_output(sim_date, prompt, response, metadata)
            written += 1
            if remove_per_question_logs:
                shutil.rmtree(log_dir, ignore_errors=True)
        output_logger.flush_warmup_raw()
    finally:
        output_logger.close()

    _rewrite_warmup_index_for_aggregated_logs(agent_dir)
    return written


@dataclass
class MinimalHarnessConfig:
    model: str = "claude-opus-4-6"
    timeout_seconds: int = 7200
    max_budget_usd: Optional[float] = None  # None = unconstrained
    search_db: str = ""
    embedding_model: str = ""
    search_type: str = "hybrid"
    search_cutoff_days: int = 0
    freeze_search_after_start: bool = False
    # Warmup-only per-question current/search date:
    # effective_date = question.resolution_date - resolution_guard days.
    resolution_guard: Optional[int] = None
    articles_base: str = ""
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    claude_code_path: str = "claude"
    extra_flags: List[str] = field(default_factory=list)
    # "claude_code" (default), "opencode", or "codex".  When "opencode", `model`
    # should be a provider-qualified id like "deepseek/deepseek-v3.2" and the
    # call is routed through OpenRouter configured in the workspace-local
    # opencode.json.  When "codex", `model` is an OpenAI model id like "gpt-5.4"
    # and auth is taken from the user's ~/.codex login (CODEX_HOME).
    harness_backend: str = "claude_code"
    opencode_path: str = "opencode"
    codex_path: str = "codex"
    # Codex reasoning effort: "none" | "minimal" | "low" | "medium" | "high" |
    # "xhigh".  "xhigh" is the max level Codex accepts for gpt-5.x.
    reasoning_effort: str = "xhigh"
    # When true and harness_backend == "codex", resume codex's thread on each
    # subsequent simulation day via `codex exec resume <thread_id>` instead of
    # spawning a fresh `codex exec`. The model keeps yesterday's reasoning in
    # context (auto-compacted by codex when needed). Day 0 still spawns
    # normally; the thread_id is captured from the first session's stdout.
    codex_resume: bool = False
    # When true and harness_backend == "claude_code", relaunch claude with
    # `--resume <session_id>` instead of a fresh `-p "Begin forecasting..."`.
    # session_id is captured from the first system/init line of an existing
    # claude_code_stdout.jsonl (e.g., from a `--resume <output_dir>` env-level
    # restart). When no prior stdout/session_id is found, falls back to a fresh
    # spawn — so this flag is safe to leave on for new runs.
    claude_code_resume: bool = False
    # Optional: opencode reads OPENROUTER_API_KEY from env.  When set here, we
    # pass it into the subprocess environment instead of inheriting.
    openrouter_api_key: str = ""
    # Optional: route Claude Code through a non-Anthropic backend (e.g. z.ai for
    # GLM) by setting ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN in the subprocess
    # env. Empty => default Anthropic endpoint.
    anthropic_base_url: str = ""
    anthropic_auth_token: str = ""
    # Prompt/session mode for harness backends.
    # - "default": existing behavior (AGENTS.md + 2-line "Begin forecasting..." per day,
    #   or codex_resume thread continuation).
    # - "active_memory": AllQAgent-style fresh-context-per-day. No AGENTS.md; the
    #   full daily prompt (mirroring BasicAgent._build_instructions with surgical
    #   swaps for codex) is sent as the user prompt to a fresh `codex exec`. Codex
    #   is told to read prior memory/{prev}/{mem.csv,meta.yaml} and write
    #   memory/{today}/{mem.csv,meta.yaml} before next_day. Requires
    #   harness_backend="codex" and codex_resume=False.
    # - "warmup": AllQAgent-style fresh-context-per-question for the explicit
    #   pre-simulation warmup hook. No AGENTS.md, no codex_resume, one Codex
    #   session per question.
    # - "static_search": same per-question Codex warmup shape, but the driver
    #   injects one precomputed title-search result and exposes only submit.
    # - "active_memory2": same fresh-context-per-day flow as "active_memory", but
    #   memory is mutated through MCP tools (mem_add/mem_update/mem_delete +
    #   memory_retrieve/memory_new/memory_update/memory_delete) instead of native
    #   filesystem reads/writes. The MCP server owns an ActiveMemory instance,
    #   loads the prior day's memory at startup, and persists today's mem.csv +
    #   meta.yaml on next_day before the harness exits.
    prompt_mode: str = "default"
    # Handholding "stay-on-it" guidance level (orthogonal to prompt_mode):
    # - "v1": minimal (state before commit dadfc2f)
    # - "v2": adds "never done while active" nudge (state after dadfc2f)
    # - "v3": v2 + "questions resolving tomorrow" reminders (state after 16b1b7a)
    # Only consumed by the default prompt_mode build (prompt.build_system_prompt)
    # and by the MCP server's next_day response. The active_memory daily prompt
    # is intentionally decoupled from this flag.
    # When None (the default), resolved per-model in __init__: deepseek/qwen/glm
    # families default to "v3" (they need the extra nudges to keep submitting),
    # everything else defaults to "v1".
    handholding_version: Optional[str] = None
    # Max outcomes per question — used by the prompt and MCP submit validation.
    max_outcomes_per_question: int = 5
    # Days between successive agent updates — passed into the cadence section
    # (used by active_memory daily prompt).
    timegap_days: int = 1
    # Number of concurrent fresh Codex sessions during prompt_mode="warmup".
    # Default stays conservative; configs can opt into more parallelism.
    warmup_parallelism: int = 1
    # Optional qid filter for targeted warmup reruns.
    warmup_qids: List[str] = field(default_factory=list)
    # Static-search ablation cache. Each file contains one title-query result
    # at the per-question effective search date.
    static_search_dir: str = ""
    static_search_max_results: int = 5
    # When true, wrap the harness subprocess in bwrap so the filesystem is
    # restricted to just {workspace, _internal_dir, repo_root+venv, search_db,
    # harness runtime} plus /usr,/etc,/lib*. articles_base, dataset path,
    # sibling sim dirs, and /home (outside the harness' own state dirs) stay
    # invisible. Network is kept via --share-net so Anthropic/OpenRouter +
    # vLLM embedding server remain reachable. Articles are staged via hardlink
    # instead of symlink (required: workspace and articles_base must be on the
    # same filesystem). Supported for both claude_code and opencode backends.
    sandbox: bool = False
    # Procfs strategy for the bwrap sandbox.
    # - "new": current strict behavior; create a private PID namespace and a
    #   fresh procfs. This is preferred where the host allows procfs mounts.
    # - "host_ro": keep filesystem isolation but bind the host/job /proc
    #   read-only. This is less process-isolated, but works on Condor nodes
    #   that reject `bwrap --proc`.
    sandbox_proc_mode: str = "new"
    # When true (requires sandbox=True), replace --share-net with --unshare-net
    # and route outbound traffic through a host-side allowlist proxy bound to a
    # unix socket. SDK clients honouring HTTPS_PROXY (Anthropic, OpenAI,
    # OpenRouter, DeepSeek, z.ai, ...) reach allowed APIs; everything else is
    # rejected by the proxy with HTTP 403. The vLLM embedding server (which the
    # MCP client reaches with HTTPS_PROXY bypassed) is exposed via a separate
    # raw TCP forward unix socket.
    network_isolation: bool = False
    # Allowlisted upstream targets ("host[:port]", glob ok in host). When empty,
    # falls back to DEFAULT_EGRESS_ALLOWLIST.
    egress_allowlist: List[str] = field(default_factory=list)
    # Path to a fixedWarmup directory containing {mem.csv, meta.yaml,
    # prediction.json}. When set, on the first act() the agent copies the
    # memory files into workspace/memory/<prev_date>/, the prediction file into
    # predictions/<prev_date>.json, and pushes the predictions into the env's
    # histories with day=<prev_date> so the workspace market.csv shows
    # my_prediction filled. <prev_date> = first sim day - 1. Used by the
    # active_memory mode to seed Day 1 from a prior agent's warmup output.
    bootstrap_dir: str = ""


# Default allowlist covers the LLM provider endpoints used (or likely to be
# used) across current and near-future configs. Wildcards follow fnmatch
# shell-glob semantics: "*.foo.com" matches any single- or multi-label
# subdomain of foo.com. Entries are pinned to :443 (HTTPS) since SDK clients
# all speak TLS. Extend or replace per-config via egress_allowlist.
DEFAULT_EGRESS_ALLOWLIST = [
    # Anthropic / Claude Code (api, console, statsig telemetry, ...).
    "*.anthropic.com:443",
    # Claude Code OAuth: platform.claude.com is the token-refresh endpoint;
    # without it long-lived persistent sessions fail with 401 once the access
    # token expires (~7-8h after spawn).
    "*.claude.com:443",
    # OpenAI / Codex (api, auth, platform; chatgpt.com is used by codex
    # login-token refresh).
    "*.openai.com:443",
    "chatgpt.com:443",
    "*.chatgpt.com:443",
    # Azure OpenAI (regional resource and Cognitive Services hosts).
    "*.openai.azure.com:443",
    "*.cognitiveservices.azure.com:443",
    # OpenRouter (multi-provider gateway).
    "openrouter.ai:443",
    "*.openrouter.ai:443",
    # DeepSeek (direct API).
    "api.deepseek.com:443",
    # Zhipu / GLM (z.ai is the new domain; open.bigmodel.cn is the legacy one).
    "api.z.ai:443",
    "*.z.ai:443",
    "open.bigmodel.cn:443",
    "*.bigmodel.cn:443",
    # Moonshot / Kimi (CN + intl endpoints).
    "api.moonshot.cn:443",
    "api.moonshot.ai:443",
    "*.moonshot.cn:443",
    "*.moonshot.ai:443",
    # Alibaba / Qwen (DashScope CN + intl, plus newer qwen.ai).
    "dashscope.aliyuncs.com:443",
    "dashscope-intl.aliyuncs.com:443",
    "*.qwen.ai:443",
    # Google Gemini direct + Vertex AI (regional aiplatform hosts + token
    # exchange endpoint for service-account auth).
    "generativelanguage.googleapis.com:443",
    "aiplatform.googleapis.com:443",
    "*-aiplatform.googleapis.com:443",
    "oauth2.googleapis.com:443",
    # Mistral (chat + Codestral).
    "api.mistral.ai:443",
    "codestral.mistral.ai:443",
    # xAI / Grok.
    "api.x.ai:443",
    # Cohere.
    "api.cohere.com:443",
    "api.cohere.ai:443",
    # Together.ai (legacy .xyz + current .ai).
    "api.together.xyz:443",
    "api.together.ai:443",
    # Fireworks.
    "api.fireworks.ai:443",
    # Groq.
    "api.groq.com:443",
    # Perplexity.
    "api.perplexity.ai:443",
    # 01.ai / Yi.
    "api.lingyiwanwu.com:443",
    # MiniMax (CN + intl).
    "api.minimaxi.chat:443",
    "api.minimax.chat:443",
    "api.minimax.io:443",
    # ByteDance / Doubao via Volcano Engine ("ark").
    "*.volces.com:443",
    # StepFun.
    "api.stepfun.com:443",
    # SiliconFlow (CN proxy widely used for OSS models).
    "api.siliconflow.cn:443",
    "api.siliconflow.com:443",
    # Baidu ERNIE / Qianfan.
    "qianfan.baidubce.com:443",
    "aip.baidubce.com:443",
    # Tencent Hunyuan.
    "hunyuan.tencentcloudapi.com:443",
    # Replicate (multi-model gateway).
    "api.replicate.com:443",
    # HuggingFace (inference endpoints + model download CDN).
    "huggingface.co:443",
    "*.huggingface.co:443",
]


class MinimalHarnessAgent(BaseAgent):
    """Forecasting agent backed by a single persistent harness session
    (claude_code, opencode, or codex)."""

    def __init__(
        self,
        agent_id: str,
        config: MinimalHarnessConfig,
        search_tool: Any = None,
        agent_dir: str = "",
        articles_base: str = "",
    ):
        super().__init__(agent_id)
        self.config = config
        self.search_tool = search_tool
        # agent_dir is the top-level agent directory.
        # _internal_dir: driver/MCP coordination (signals, mcp_config, logs) — CC doesn't see this.
        # workspace: CC's working directory (--add-dir) — only what CC needs.
        self._internal_dir = Path(agent_dir)
        self.workspace = self._internal_dir / "workspace"
        self.articles_base = Path(articles_base or config.articles_base)
        self._harness_proc: Optional[subprocess.Popen] = None
        self._started = False
        self._last_symlink_date: Optional[date] = None
        self._current_date: Optional[date] = None
        self._session_id: Optional[str] = None  # opencode session id, captured from stdout
        self._resume_attempts: Dict[date, int] = {}
        self._codex_thread_id: Optional[str] = None  # codex thread_id, captured from Day 0 stdout
        self._claude_code_session_id: Optional[str] = None  # claude session_id, captured from Day 0 stdout
        self._bootstrap_applied: bool = False

        # Egress proxy state (network_isolation=True). Started lazily on first act().
        self._egress_proc: Optional[subprocess.Popen] = None
        self._egress_dir: Optional[Path] = None
        self._egress_proxy_sock: Optional[Path] = None
        self._egress_embed_sock: Optional[Path] = None
        self._egress_embed_port: Optional[int] = None
        # In-sandbox loopback port the bridge sidecar listens on for the proxy.
        self._egress_proxy_port = 9080

        # MCP relay state (sandbox=True). Started lazily on first act() — keeps
        # the MCP server (and thus search_db / lancedb / embedding_model) on
        # the host so the sandboxed harness can't bypass date-capping by
        # querying the index directly. The harness reaches MCP via a socat
        # stdio bridge to a per-agent unix socket.
        self._mcp_relay_proc: Optional[subprocess.Popen] = None
        self._mcp_relay_dir: Optional[Path] = None
        self._mcp_relay_sock: Optional[Path] = None
        self._mcp_bridge_wrapper: Optional[Path] = None

        if self.config.network_isolation and not self.config.sandbox:
            raise ValueError("network_isolation=True requires sandbox=True.")
        if self.config.sandbox_proc_mode not in {"new", "host_ro"}:
            raise ValueError(
                "sandbox_proc_mode must be 'new' or 'host_ro' "
                f"(got {self.config.sandbox_proc_mode!r})."
            )

        # Per-day codex initial prompt (set by _build_active_memory_prompt when
        # config.prompt_mode == "active_memory"). Read by _start_codex.
        self._codex_initial_prompt_override: Optional[str] = None

        # active_memory / active_memory2 mode wiring.
        if self.config.prompt_mode in ("active_memory", "active_memory2"):
            if self.config.harness_backend not in ("codex", "claude_code", "opencode"):
                raise ValueError(
                    f"prompt_mode={self.config.prompt_mode!r} requires harness_backend in "
                    "{'codex', 'claude_code', 'opencode'} "
                    f"(got {self.config.harness_backend!r})."
                )
            if self.config.harness_backend == "codex" and self.config.codex_resume:
                raise ValueError(
                    f"prompt_mode={self.config.prompt_mode!r} requires codex_resume=False; "
                    "this mode delivers AllQ-style fresh-context-per-day with "
                    "memory files, not a persistent codex thread."
                )
            if self.config.harness_backend == "claude_code" and self.config.claude_code_resume:
                raise ValueError(
                    f"prompt_mode={self.config.prompt_mode!r} requires claude_code_resume=False; "
                    "this mode delivers fresh-context-per-day with memory files, "
                    "not a persistent claude session."
                )
            # Stateful feedback aggregator (shared across days, dedups on qid).
            from agents.basicAgent.feedback import FeedbackHandler
            self._feedback_handler: Optional[FeedbackHandler] = FeedbackHandler(self.agent_id)
        elif self.config.prompt_mode in {"warmup", "static_search"}:
            if self.config.harness_backend != "codex":
                raise ValueError(
                    f"prompt_mode={self.config.prompt_mode!r} currently requires harness_backend='codex' "
                    f"(got {self.config.harness_backend!r})."
                )
            if self.config.codex_resume:
                raise ValueError(
                    f"prompt_mode={self.config.prompt_mode!r} requires codex_resume=False; each question "
                    "gets a fresh Codex session."
                )
            self._feedback_handler = None
        elif self.config.prompt_mode not in ("default", "no_memory"):
            raise ValueError(
                f"Unknown prompt_mode={self.config.prompt_mode!r}; "
                "expected 'default', 'no_memory', 'active_memory', 'warmup', 'static_search', or 'active_memory2'."
            )
        else:
            self._feedback_handler = None
        self.warmed_up = False

        if self.config.handholding_version is None:
            # Per-model default. deepseek/qwen/glm need the v3 nudges (TW-update
            # reminder + imminent-resolution callout) to keep submitting; without
            # them they drift into "no new evidence, skip" silence (e.g. glm-5.1
            # only submitted on 33/95 days under v1). Other models keep v1.
            model_lc = (self.config.model or "").lower()
            if any(fam in model_lc for fam in ("deepseek", "qwen", "glm")):
                self.config.handholding_version = "v3"
            else:
                self.config.handholding_version = "v1"
        if self.config.handholding_version not in ("v1", "v2", "v3"):
            raise ValueError(
                f"Unknown handholding_version={self.config.handholding_version!r}; "
                "expected one of 'v1', 'v2', 'v3'."
            )
        # Active memory's prompt builder hardcodes the maximal-handholding
        # shared sections (TW nudge + imminent reminder). Force the field to
        # v3 so the runtime stays consistent everywhere it's read (mcp server,
        # logs, build_system_prompt fallbacks, future audits).
        if (
            self.config.prompt_mode in ("active_memory", "active_memory2")
            and self.config.handholding_version != "v3"
        ):
            print(
                f"[minimalHarness] prompt_mode={self.config.prompt_mode!r} coerces "
                f"handholding_version {self.config.handholding_version!r} -> 'v3'."
            )
            self.config.handholding_version = "v3"

    # ── BaseAgent contract ─────────────────────────────────────────────

    def act(
        self,
        doc_interface: Any,
        forecast_interface: Any,
        current_date: date,
    ) -> List[Dict[str, Any]]:
        self._current_date = current_date

        if (
            self.config.prompt_mode in {"warmup", "static_search"}
            and self.warmed_up
            and self.config.start_date == current_date
        ):
            print(f"[{self.agent_id}] Skipping standard act() on Day 0 (Warmup already completed).")
            forecast_interface.next_day()
            return []

        # Internal dirs (driver/MCP coordination — harness doesn't see these).
        for d in ("signals",):
            (self._internal_dir / d).mkdir(parents=True, exist_ok=True)

        # Internal dirs (driver/MCP coordination — harness doesn't see these).
        (self._internal_dir / "predictions").mkdir(parents=True, exist_ok=True)

        # Workspace dirs (harness' working directory — only what the harness needs).
        for d in ("memory", "articles"):
            (self.workspace / d).mkdir(parents=True, exist_ok=True)

        # Expose past predictions/ as a symlink inside the workspace so the
        # harness can always recall what it submitted on previous days
        # (the in-CSV `my_prediction` column only carries the latest
        # forecast per qid). _internal_dir is bind-mounted into the sandbox
        # at the same absolute path, so the symlink resolves there too.
        pred_link = self.workspace / "predictions"
        if not pred_link.exists():
            pred_link.symlink_to(self._internal_dir / "predictions")

        # 0. Bootstrap from fixedWarmup only on the true first sim day.
        # Env-level resumes may start mid-run with copied memory already present.
        if (
            self.config.bootstrap_dir
            and not self._bootstrap_applied
            and (self.config.start_date is None or current_date == self.config.start_date)
        ):
            self._apply_bootstrap(forecast_interface, current_date)
            self._bootstrap_applied = True

        # 1. Write state.json for MCP server to read.
        search_current_date = (
            self.config.start_date
            if self.config.freeze_search_after_start
            else None
        )
        self._write_state(
            forecast_interface,
            current_date,
            search_current_date=search_current_date,
        )

        # 2. chmod market.csv read-only so the harness can't corrupt it.
        self._protect_market_csv(forecast_interface)

        # 3. Update article symlinks up to the same date visible to search.
        self._update_article_symlinks(search_current_date or current_date)

        # 3b. If network_isolation is on, ensure the host-side egress proxy is
        # running before we spawn (or re-spawn) the harness. Idempotent.
        if self.config.network_isolation:
            self._start_egress_proxy()

        # 3c. If sandboxed, start the host-side MCP relay so search_db and
        # the embedding model stay out of the sandbox. The harness reaches
        # MCP only via the bind-mounted unix socket. Idempotent.
        if self.config.sandbox:
            self._start_mcp_relay()

        # codex has no persistent session: each `codex exec` runs once and
        # exits. So we re-spawn codex (and its child MCP server) every day.
        # claude_code and opencode normally use a single long-lived process
        # whose MCP server polls for a per-day continue signal — but in
        # active_memory{,2} and no_memory modes they also re-spawn per day so
        # each day starts with a fresh conversation (active_memory bridges
        # state through structured memory files; active_memory2 bridges via
        # MCP memory tools backed by the same files; no_memory has no bridge,
        # mirroring codex no_memory's per-day fresh-thread semantics).
        codex_per_day = self.config.harness_backend == "codex"
        fresh_session_per_day = (
            self.config.harness_backend in ("claude_code", "opencode")
            and self.config.prompt_mode in ("active_memory", "active_memory2", "no_memory")
        )
        respawn_per_day = codex_per_day or fresh_session_per_day

        if not self._started or respawn_per_day:
            if respawn_per_day and self._started:
                # Reap the previous day's harness if still alive (should have
                # exited after writing next_day, but be defensive).
                if self._harness_proc and self._harness_proc.poll() is None:
                    self._harness_proc.terminate()
                    try:
                        self._harness_proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self._harness_proc.kill()
                # Clear stale next_day signals so _wait_for_next_day blocks
                # until the freshly-spawned harness writes a new one.
                signal_dir = self._internal_dir / "signals"
                signal_dir.mkdir(parents=True, exist_ok=True)
                for old in signal_dir.glob("next_day_*.json"):
                    old.unlink(missing_ok=True)

            if self.config.prompt_mode in ("active_memory", "active_memory2"):
                # active_memory{,2} mode replaces AGENTS.md + 2-line per-day prompt
                # with a full daily user prompt (mirroring AllQAgent). Skip the
                # system_prompt.md write entirely.
                self._build_active_memory_prompt(forecast_interface, current_date)
            else:
                self._write_system_prompt(forecast_interface)
            if self.config.harness_backend == "opencode":
                self._write_opencode_config()
                self._start_opencode()
            elif self.config.harness_backend == "codex":
                self._start_codex()
            else:
                self._write_mcp_config()
                self._start_claude_code()
            self._started = True
        else:
            # Subsequent days for persistent backends — unblock the MCP
            # server's next_day poll.
            self._write_continue_signal(current_date)

        # 4. Wait for the harness to call next_day (MCP server writes signal).
        predictions = self._wait_for_next_day(current_date)

        # codex_resume mode: capture thread_id from Day 0 stdout so
        # subsequent days can run `codex exec resume <thread_id>`.
        if (
            self.config.harness_backend == "codex"
            and self.config.codex_resume
            and self._codex_thread_id is None
        ):
            self._find_codex_thread_id()

        # 5. Submit predictions to the environment.
        actions: List[Dict[str, Any]] = []
        for pred in predictions:
            try:
                forecast_interface.submit_prediction(PredictionSubmission(
                    question_id=pred["question_id"],
                    outcomes=pred["outcomes"],
                ))
                actions.append(pred)
            except Exception as e:
                logger.warning("Failed to submit prediction %s: %s", pred, e)

        # 6. Restore market.csv write permission so the env can overwrite next day.
        self._unprotect_market_csv()

        # 7. Signal day complete.
        forecast_interface.next_day()
        return actions

    # ── bootstrap ──────────────────────────────────────────────────────

    def _apply_bootstrap(self, forecast_interface: Any, current_date: date) -> None:
        """Seed Day-1 state from a fixedWarmup directory.

        Copies bootstrap_dir/{mem.csv, meta.yaml} into workspace/memory/<prev>/
        and bootstrap_dir/prediction.json into predictions/<prev>.json, then
        injects each prediction into env.histories with day=<prev> so the
        workspace market.csv shows my_prediction filled on Day 1.
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

        pred_dst = self._internal_dir / "predictions"
        pred_dst.mkdir(parents=True, exist_ok=True)
        pred_src = bdir / "prediction.json"
        if pred_src.exists():
            shutil.copy2(pred_src, pred_dst / f"{prev_iso}.json")

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
            f"copied memory/{prev_iso}, predictions/{prev_iso}.json; "
            f"injected {injected} predictions ({skipped} skipped)."
        )

    # ── state / config writers ─────────────────────────────────────────

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
            "search_db": self.config.search_db,
            "embedding_model": self.config.embedding_model,
            "search_type": self.config.search_type,
            "search_cutoff_days": self.config.search_cutoff_days,
            "timeout_seconds": self.config.timeout_seconds,
            "questions": questions,
            "resolution_events": forecast_interface.resolution_events,
            "agent_predictions": agent_predictions,
            "total_predictions": total_predictions,
            "prompt_mode": self.config.prompt_mode,
            "submit_ends_session": self.config.prompt_mode in {"warmup", "static_search"},
            "next_day_returns_immediately": self.config.harness_backend == "codex",
            "max_outcomes_per_question": int(self.config.max_outcomes_per_question),
        }
        state_path = self._internal_dir / "state.json"
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def _get_warmup_current_date(self, current_date: date, q: Any) -> date:
        resolution_guard = self.config.resolution_guard
        resolution_date = getattr(q, "resolution_date", None)
        if resolution_guard is None or resolution_date is None:
            return current_date
        return resolution_date - timedelta(days=resolution_guard)

    def _build_warmup_prompt(
        self,
        forecast_interface: Any,
        q: Any,
        effective_current_date: date,
    ) -> str:
        from agents.minimalHarnessAgent.prompt import build_warmup_prompt

        source_context = getattr(forecast_interface, "source_context", "") or ""
        source_name = getattr(forecast_interface, "source_name", "openforesight")
        static_search_text = None
        if self.config.prompt_mode == "static_search":
            static_search_text = self._get_static_search_text(q, effective_current_date)
        prompt = build_warmup_prompt(
            current_date=effective_current_date,
            q=q,
            source_context=source_context,
            source_name=source_name,
            max_outcomes_per_question=self.config.max_outcomes_per_question,
            search_cutoff_days=self.config.search_cutoff_days,
            static_search_text=static_search_text,
        )
        safe_qid = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(q.qid))
        prompt_path = self._internal_dir / "warmup_prompts" / f"{safe_qid}.md"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(prompt_path, "w") as f:
            f.write(prompt)
        return prompt

    def _terminate_harness_process(self) -> None:
        if self._harness_proc and self._harness_proc.poll() is None:
            self._harness_proc.terminate()
            try:
                self._harness_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._harness_proc.kill()
        self._harness_proc = None
        for f in (getattr(self, "_stdout_log", None), getattr(self, "_stderr_log", None)):
            if f and not f.closed:
                f.close()

    def _warmup_runtime_dir(self, safe_qid: str) -> Path:
        return self._internal_dir / "warmup_runtime" / safe_qid

    def _warmup_log_dir(self, safe_qid: str) -> Path:
        return self._internal_dir / "warmup_logs" / safe_qid

    @staticmethod
    def _safe_qid(qid: Any) -> str:
        return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(qid))

    def _static_search_root(self) -> Path:
        if self.config.static_search_dir:
            return Path(os.path.expanduser(os.path.expandvars(self.config.static_search_dir)))
        return self._internal_dir / "static_search"

    def _ensure_static_search_file(self, q: Any, effective_current_date: date) -> Tuple[Path, str]:
        from agents.minimalHarnessAgent.static_search import ensure_static_search_file

        return ensure_static_search_file(
            output_dir=self._static_search_root(),
            search_tool=self.search_tool,
            q=q,
            search_date=effective_current_date,
            search_type=self.config.search_type,
            search_cutoff_days=self.config.search_cutoff_days,
            max_results=int(self.config.static_search_max_results or 5),
        )

    def _get_static_search_text(self, q: Any, effective_current_date: date) -> str:
        _, text = self._ensure_static_search_file(q, effective_current_date)
        return text

    def _prepare_static_search_files(self, questions: List[Any], current_date: date) -> None:
        root = self._static_search_root()
        root.mkdir(parents=True, exist_ok=True)
        index_path = root / "index.jsonl"
        index_path.unlink(missing_ok=True)

        total = len(questions)
        print(f"[{self.agent_id}] Preparing static title-search cache in {root}...")
        with open(index_path, "a") as index_f:
            for i, q in enumerate(questions, start=1):
                effective_current_date = self._get_warmup_current_date(current_date, q)
                path, _ = self._ensure_static_search_file(q, effective_current_date)
                index_f.write(json.dumps({
                    "qid": str(q.qid),
                    "title": str(q.title),
                    "resolution_date": self._to_iso(getattr(q, "resolution_date", "")),
                    "search_date": effective_current_date.isoformat(),
                    "query": str(q.title),
                    "path": str(path),
                }) + "\n")
                if i % 10 == 0 or i == total:
                    print(f"[{self.agent_id}] Static Search Progress: {i}/{total}", flush=True)

    def _copy_warmup_runtime_logs(
        self,
        runtime_dir: Path,
        log_dir: Path,
        qid: str,
        current_date: date,
    ) -> None:
        """Preserve per-question raw logs, then allow the runtime dir to be deleted."""
        log_dir.mkdir(parents=True, exist_ok=True)
        copy_map = {
            runtime_dir / "codex_stdout.jsonl": log_dir / "codex_stdout.jsonl",
            runtime_dir / "codex_stderr.log": log_dir / "codex_stderr.log",
            runtime_dir / "mcp_relay.log": log_dir / "mcp_relay.log",
            runtime_dir / "state.json": log_dir / "state.json",
            runtime_dir / "predictions" / f"{current_date.isoformat()}.json": log_dir / "predictions.json",
            runtime_dir / "warmup_prompts" / f"{self._safe_qid(qid)}.md": log_dir / "prompt.md",
        }
        for src, dst in copy_map.items():
            if src.exists():
                shutil.copy2(src, dst)

    def _stop_warmup_worker(self, worker: "MinimalHarnessAgent") -> None:
        worker._terminate_harness_process()
        worker._stop_egress_proxy()
        worker._stop_mcp_relay()
        worker._started = False

    def _run_single_warmup_question(
        self,
        q: Any,
        current_date: date,
        forecast_interface: Any,
    ) -> Tuple[str, date, List[Dict[str, Any]]]:
        qid = str(q.qid)
        safe_qid = self._safe_qid(qid)
        runtime_dir = self._warmup_runtime_dir(safe_qid)
        log_dir = self._warmup_log_dir(safe_qid)

        if runtime_dir.exists():
            shutil.rmtree(runtime_dir)
        if log_dir.exists():
            shutil.rmtree(log_dir)
        runtime_dir.mkdir(parents=True, exist_ok=True)

        worker_config = replace(self.config, warmup_parallelism=1)
        worker = MinimalHarnessAgent(
            agent_id=self.agent_id,
            config=worker_config,
            search_tool=self.search_tool,
            agent_dir=str(runtime_dir),
            articles_base=str(self.articles_base),
        )

        effective_current_date = self._get_warmup_current_date(current_date, q)
        try:
            for d in ("signals", "predictions"):
                (worker._internal_dir / d).mkdir(parents=True, exist_ok=True)
            for d in ("memory", "articles"):
                (worker.workspace / d).mkdir(parents=True, exist_ok=True)

            if worker.config.network_isolation:
                worker._start_egress_proxy()
            if worker.config.sandbox:
                worker._start_mcp_relay()

            worker._current_date = current_date
            worker._write_state(
                forecast_interface,
                current_date,
                questions_override=[q],
                search_current_date=effective_current_date,
            )
            worker._codex_initial_prompt_override = worker._build_warmup_prompt(
                forecast_interface,
                q,
                effective_current_date,
            )

            worker._start_codex()
            worker._started = True
            predictions = worker._wait_for_next_day(current_date)
            predictions = [
                pred for pred in predictions
                if str(pred.get("question_id", pred.get("qid", ""))) == qid
            ]
            return qid, effective_current_date, predictions
        finally:
            self._stop_warmup_worker(worker)
            self._copy_warmup_runtime_logs(runtime_dir, log_dir, qid, current_date)
            shutil.rmtree(runtime_dir, ignore_errors=True)

    def _append_warmup_log_index(
        self,
        qid: str,
        effective_current_date: date,
        predictions: List[Dict[str, Any]],
        error: Optional[str] = None,
    ) -> None:
        index_path = self._internal_dir / "warmup_logs" / "index.jsonl"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "qid": qid,
            "search_current_date": effective_current_date.isoformat(),
            "num_predictions": len(predictions),
            "error": error,
            "processed_output_log": str(self._internal_dir / "model_outputs.jsonl"),
            "raw_output_log": str(self._internal_dir / "model_raw_warmup.jsonl"),
        }
        with open(index_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def warmup(self, forecast_interface: Any, current_date: date) -> None:
        """Run AllQ-style per-question warmup with one fresh Codex session per question."""
        if self.config.prompt_mode not in {"warmup", "static_search"}:
            return

        mode_label = self.config.prompt_mode.upper()
        print(f"[{self.agent_id}] Starting MinimalHarness {mode_label} phase on {current_date}")

        forecast_interface.current_agent_id = self.agent_id

        questions = list(forecast_interface.questions.values())
        questions.sort(key=lambda q: (self._get_warmup_current_date(current_date, q), str(q.qid)))
        if self.config.warmup_qids:
            requested_qids = {str(qid) for qid in self.config.warmup_qids}
            questions = [q for q in questions if str(q.qid) in requested_qids]
            missing_qids = requested_qids - {str(q.qid) for q in questions}
            if missing_qids:
                raise ValueError(
                    f"warmup_qids not found in active questions: {sorted(missing_qids)}"
                )
        total = len(questions)
        if self.config.prompt_mode == "static_search":
            self._prepare_static_search_files(questions, current_date)

        max_workers = max(1, min(int(self.config.warmup_parallelism or 1), total or 1))
        print(f"[{self.agent_id}] Parallelizing MinimalHarness warmup with {max_workers} worker(s)...")

        runtime_root = self._internal_dir / "warmup_runtime"
        runtime_root.mkdir(parents=True, exist_ok=True)
        index_path = self._internal_dir / "warmup_logs" / "index.jsonl"
        index_path.unlink(missing_ok=True)

        completed = 0
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_q = {
                    executor.submit(
                        self._run_single_warmup_question,
                        q,
                        current_date,
                        forecast_interface,
                    ): q
                    for q in questions
                }
                for future in as_completed(future_to_q):
                    q = future_to_q[future]
                    qid = str(q.qid)
                    effective_current_date = self._get_warmup_current_date(current_date, q)
                    error = None
                    try:
                        _, effective_current_date, predictions = future.result()
                    except Exception as e:
                        logger.warning("Warmup question %s failed: %s", qid, e)
                        predictions = []
                        error = str(e)

                    submitted = False
                    for pred in predictions:
                        try:
                            forecast_interface.submit_prediction(PredictionSubmission(
                                question_id=pred["question_id"],
                                outcomes=pred["outcomes"],
                            ))
                            submitted = True
                        except Exception as e:
                            logger.warning("Failed to submit warmup prediction %s: %s", pred, e)

                    self._append_warmup_log_index(qid, effective_current_date, predictions, error=error)
                    if not submitted:
                        logger.warning("No warmup prediction submitted for qid %s", qid)

                    completed += 1
                    if completed % 10 == 0 or completed == total:
                        print(f"[{self.agent_id}] Warmup Progress: {completed}/{total}", flush=True)
        finally:
            shutil.rmtree(runtime_root, ignore_errors=True)

        aggregated = aggregate_codex_warmup_logs(
            self.agent_id,
            self._internal_dir,
            remove_per_question_logs=True,
        )
        if aggregated:
            print(f"[{self.agent_id}] Aggregated {aggregated} Codex warmup transcript(s).", flush=True)

        self._codex_initial_prompt_override = None
        self.warmed_up = True
        print(f"[{self.agent_id}] MinimalHarness {self.config.prompt_mode} complete.")

    def _protect_market_csv(self, forecast_interface: Any) -> None:
        """Materialize a per-agent market.csv in the workspace.

        The env's shared market.csv (MarketWriter) has no per-agent columns,
        but our prompt advertises `my_prediction` / `my_prediction_date`. We
        read the shared csv, stamp those columns from this agent's history
        (mirroring what DfInterface does in-memory for other scaffolds), and
        write a per-agent copy into the workspace. The shared file is also
        chmod'd read-only as a sanity lock against accidental writes.
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

    def _write_mcp_config(self) -> None:
        """Write the MCP config JSON that Claude Code will use to spawn the server.

        We pass the repo root via the server's own --repo-root flag rather than
        using the 'env' key in the MCP config, because Claude Code replaces
        (rather than merges) the process environment when 'env' is set.

        With sandbox=True, _build_mcp_invocation returns a socat bridge so
        the MCP server stays on the host (see _start_mcp_relay).
        """
        cmd, args = self._build_mcp_invocation()
        config = {
            "mcpServers": {
                "forecast": {
                    "command": cmd,
                    "args": args,
                }
            }
        }
        config_path = self._internal_dir / "mcp_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    def _write_system_prompt(self, forecast_interface: Any = None) -> None:
        source_context = ""
        source_name = "openforesight"
        num_questions = 0
        num_active = 0
        num_resolved = 0
        new_articles_count = None
        last_active_date = None
        next_active_date = None
        if forecast_interface is not None:
            source_context = getattr(forecast_interface, "source_context", "")
            source_name = getattr(forecast_interface, "source_name", "openforesight")
            questions = forecast_interface.list_questions()
            num_active = len(questions)
            num_resolved = len(getattr(forecast_interface, "resolved_questions", []))
            num_questions = num_active + num_resolved
            last_active_date = getattr(forecast_interface, "last_active_date", None)
            next_active_date = getattr(forecast_interface, "next_active_date", None)
            prompt_date = self._current_date or self.config.start_date
            new_articles_count = self._count_new_articles(last_active_date, prompt_date)
        if self.config.prompt_mode == "no_memory":
            from agents.minimalHarnessAgent.prompt_no_memory import (
                build_system_prompt as _build_system_prompt,
            )
        else:
            _build_system_prompt = build_system_prompt
        prompt = _build_system_prompt(
            workspace=str(self.workspace),
            current_date=self._current_date or self.config.start_date,
            start_date=self.config.start_date or self._current_date,
            end_date=self.config.end_date or self._current_date,
            source_context=source_context,
            source_name=source_name,
            num_questions=num_questions,
            num_active=num_active,
            num_resolved=num_resolved,
            max_outcomes_per_question=self.config.max_outcomes_per_question,
            search_cutoff_days=self.config.search_cutoff_days,
            timegap_days=self.config.timegap_days,
            new_articles_count=new_articles_count,
            last_active_date=last_active_date,
            next_active_date=next_active_date,
            handholding_version=self.config.handholding_version,
        )
        prompt_path = self._internal_dir / "system_prompt.md"
        with open(prompt_path, "w") as f:
            f.write(prompt)

    def _build_active_memory_prompt(
        self, forecast_interface: Any, current_date: date
    ) -> None:
        """Compose the per-day active_memory{,2} user prompt and stash it for _start_codex.

        Mirrors BasicAgent / AllQAgent daily prompt construction:
          - feedback aggregated via FeedbackHandler (cumulative across days),
          - prior day's meta-insight index loaded from disk via ActiveMemory,
          - AllQ "already-predicted" reminder with current predicted/active counts.

        prompt_mode="active_memory2" routes to prompt_active_memory2.build_daily_prompt,
        which advertises the MCP memory tools instead of native filesystem reads/writes.
        """
        if self.config.prompt_mode == "active_memory2":
            from agents.minimalHarnessAgent.prompt_active_memory2 import build_daily_prompt
        else:
            from agents.minimalHarnessAgent.prompt_active_memory import build_daily_prompt
        from agents.utils.memory import ActiveMemory

        source_context = getattr(forecast_interface, "source_context", "") or ""
        source_name = getattr(forecast_interface, "source_name", "openforesight")
        questions = forecast_interface.list_questions()
        num_active = len(questions)
        num_resolved = len(getattr(forecast_interface, "resolved_questions", []))
        num_questions = num_active + num_resolved
        last_active_date = getattr(forecast_interface, "last_active_date", None)
        if last_active_date is None:
            last_active_date = self._find_latest_memory_date_before(current_date)
        next_active_date = getattr(forecast_interface, "next_active_date", None)
        new_articles_count = self._count_new_articles(last_active_date, current_date)

        # Feedback (consumes resolution_events from the env; dedups on qid).
        assert self._feedback_handler is not None
        feedback_data = self._feedback_handler.generate_feedback(
            forecast_interface, current_date, inference_provider=None
        )
        feedback_text = self._feedback_handler.format_feedback(
            feedback_data, show_tw_peer=False
        )

        # Prior-day meta-insight index from memory/{prev}/meta.yaml on disk.
        meta_index = ""
        am = ActiveMemory(self.agent_id, memory_dir=str(self.workspace))
        am.set_date(current_date)
        meta_index = am.get_index() or ""

        # Predicted / active counts for the AllQ reminder.
        active_qids = {q.qid for q in questions}
        predicted_count = 0
        get_preds = getattr(forecast_interface, "get_agent_predictions", None)
        if callable(get_preds):
            preds = get_preds(self.agent_id) or {}
            predicted_count = sum(1 for qid in preds.keys() if qid in active_qids)

        # Questions resolving tomorrow — surfaced inline in the cadence section.
        tomorrow = current_date + timedelta(days=1)
        imminent_qids = [
            q.qid for q in questions
            if getattr(q, "resolution_date", None) == tomorrow
        ]

        prompt = build_daily_prompt(
            current_date=current_date,
            start_date=self.config.start_date or current_date,
            end_date=self.config.end_date or current_date,
            last_active_date=last_active_date,
            next_active_date=next_active_date,
            source_context=source_context,
            source_name=source_name,
            feedback_text=feedback_text,
            meta_index=meta_index,
            num_questions=num_questions,
            num_active=num_active,
            num_resolved=num_resolved,
            predicted_count=predicted_count,
            active_count=num_active,
            max_outcomes_per_question=self.config.max_outcomes_per_question,
            search_cutoff_days=self.config.search_cutoff_days,
            timegap_days=self.config.timegap_days,
            new_articles_count=new_articles_count,
            imminent_qids=imminent_qids,
        )
        # Persist for debugging / offline review.
        prompt_path = (
            self._internal_dir
            / f"{self.config.prompt_mode}_prompt_{current_date.isoformat()}.md"
        )
        with open(prompt_path, "w") as f:
            f.write(prompt)
        self._codex_initial_prompt_override = prompt

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
        """Locate the vLLM embedding server started by the driver.

        Prefers $FSIM_EMBEDDING_URL (set by test_basic_agent.py after the
        driver-owned vLLMInference is warm) so parallel sims with multiple
        vLLMs on the same host don't grab a sibling sim's server. Falls back
        to scanning 127.0.0.1:8001-8009 for a /v1/models 200 if the env var
        is unset (legacy callers / external launchers).
        Uses 127.0.0.1 to bypass any HTTP proxy that intercepts 'localhost'.
        Returns the server URL or empty string."""
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

    # ── article staging ────────────────────────────────────────────────

    def _update_article_symlinks(self, current_date: date) -> None:
        """Incrementally expose per-day articles.jsonl files up to current_date.

        When sandbox=False, uses symlinks (fast, but requires articles_base
        to be accessible in the process's filesystem view).
        When sandbox=True, uses hardlinks so the files appear as real entries
        inside the workspace without needing articles_base to be bound — this
        keeps articles_base invisible from inside the sandbox, enforcing the
        date-gate via the driver instead of by filesystem ACLs.

        Only articles.jsonl is exposed; the sibling parquet and headlines JSON
        files are intentionally omitted (parquet isn't greppable and the
        headlines JSON duplicates info available in articles.jsonl)."""
        if not self.articles_base or not self.articles_base.exists():
            return

        articles_dir = self.workspace / "articles"
        if self._last_symlink_date:
            start = self._last_symlink_date + timedelta(days=1)
        elif self.config.start_date is not None:
            start = min(self.config.start_date, current_date)
        else:
            start = current_date - timedelta(days=30)

        d = start
        while d <= current_date:
            src = self.articles_base / f"{d.year}" / f"{d.month:02d}" / f"{d.day:02d}" / "articles.jsonl"
            if src.exists():
                dst = articles_dir / f"{d.year}" / f"{d.month:02d}" / f"{d.day:02d}" / "articles.jsonl"
                dst.parent.mkdir(parents=True, exist_ok=True)
                if self.config.freeze_search_after_start:
                    cutoff = current_date.isoformat()
                    tmp = dst.with_suffix(".jsonl.tmp")
                    with open(src) as inp, open(tmp, "w") as out:
                        for line in inp:
                            article = json.loads(line)
                            article_date = str(article.get("date") or "")[:10]
                            publish_date = str(article.get("date_publish") or "")[:10]
                            if article_date and article_date > cutoff:
                                continue
                            if publish_date and publish_date > cutoff:
                                continue
                            out.write(line)
                    if dst.exists() or dst.is_symlink():
                        dst.unlink()
                    tmp.replace(dst)
                elif not dst.exists():
                    if self.config.sandbox:
                        try:
                            os.link(src, dst)
                        except OSError as e:
                            if e.errno == 18:  # EXDEV
                                raise RuntimeError(
                                    f"sandbox=True requires workspace and articles_base on the same filesystem "
                                    f"(hardlink failed: {src} -> {dst}). Move one of them or disable sandbox."
                                ) from e
                            raise
                    else:
                        dst.symlink_to(src)
            d += timedelta(days=1)

        self._last_symlink_date = current_date

    # ── Claude Code subprocess ─────────────────────────────────────────

    def _start_claude_code(self) -> None:
        # Resume mode: if claude_code_resume is set and a previous session_id
        # is recoverable from claude_code_stdout.jsonl, reattach to that
        # conversation via `claude --resume <id>`. Falls back to a fresh spawn
        # when no prior session is found (safe default for new runs).
        resume_session_id: Optional[str] = None
        if self.config.claude_code_resume:
            resume_session_id = self._find_claude_code_session_id()
            if resume_session_id:
                self._claude_code_session_id = resume_session_id

        if resume_session_id:
            day_iso = (
                self._current_date.isoformat()
                if self._current_date else ""
            )
            initial_prompt = (
                f"Simulation resumed at {day_iso}. The previous session was "
                "interrupted; re-read state.json and market.csv for the "
                "current day's questions, resolutions, and your existing "
                "predictions, then continue research, update predictions "
                "where new info changes your view, and call next_day."
            )
        elif (
            self.config.prompt_mode in ("active_memory", "active_memory2")
            and self._codex_initial_prompt_override
        ):
            initial_prompt = self._codex_initial_prompt_override
        else:
            initial_prompt = (
                "Begin forecasting. Read market.csv to see your questions, "
                "then research and submit predictions."
            )
        # When sandboxed, pass the realpath of the claude binary as argv[0] so
        # the subprocess can exec it without needing the ~/.local/bin/claude
        # symlink to be preserved inside the sandbox.
        claude_bin = (
            os.path.realpath(self.config.claude_code_path)
            if self.config.sandbox
            else self.config.claude_code_path
        )
        cmd = [
            claude_bin,
        ]
        if resume_session_id:
            cmd.extend(["--resume", resume_session_id])
        if self.config.prompt_mode == "no_memory":
            disallowed_tools = "WebSearch,WebFetch,Write,Edit,MultiEdit,NotebookEdit"
        else:
            disallowed_tools = "WebSearch,WebFetch"

        cmd.extend([
            "-p", initial_prompt,
            "--verbose",
            "--effort", "max",
            "--model", self.config.model,
            "--output-format", "stream-json",
            "--dangerously-skip-permissions",
            "--mcp-config", str(self._internal_dir / "mcp_config.json"),
            "--strict-mcp-config",
            "--disallowedTools",
            disallowed_tools,
        ])
        # active_memory{,2} mode delivers the full daily instructions via -p, so
        # skip the standalone system prompt file (which isn't written in this
        # mode anyway).
        if self.config.prompt_mode not in ("active_memory", "active_memory2"):
            cmd.extend([
                "--system-prompt-file", str(self._internal_dir / "system_prompt.md"),
            ])
        cmd.extend([
            "--add-dir", str(self.workspace),
        ])

        if self.config.max_budget_usd is not None:
            cmd.extend(["--max-budget-usd", str(self.config.max_budget_usd)])

        cmd.extend(self.config.extra_flags)

        # Append mode so a resume relaunch doesn't truncate the prior stdout
        # (which is what we read session_id from on subsequent restarts).
        log_stdout = open(self._internal_dir / "claude_code_stdout.jsonl", "a")
        log_stderr = open(self._internal_dir / "claude_code_stderr.log", "a")

        env = os.environ.copy()
        if self.config.anthropic_base_url:
            env["ANTHROPIC_BASE_URL"] = self.config.anthropic_base_url
        if self.config.anthropic_auth_token:
            env["ANTHROPIC_AUTH_TOKEN"] = self.config.anthropic_auth_token
        if self.config.network_isolation:
            env.update(self._egress_env_overrides())

        launch_cmd = self._maybe_sandbox(cmd)
        # When sandboxed, bwrap sets cwd via --chdir; passing cwd= to Popen
        # would chdir in the host namespace before exec and can fail if the
        # path is visible only through symlinks that bwrap reshapes.
        popen_cwd = None if self.config.sandbox else str(self.workspace)

        logger.info("Starting Claude Code%s%s: %s",
                    " [sandboxed]" if self.config.sandbox else "",
                    " [net-isolated]" if self.config.network_isolation else "",
                    " ".join(launch_cmd))
        self._harness_proc = subprocess.Popen(
            launch_cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_stdout,
            stderr=log_stderr,
            cwd=popen_cwd,
            env=env,
        )
        self._stdout_log = log_stdout
        self._stderr_log = log_stderr

    # ── opencode backend ───────────────────────────────────────────────

    def _write_opencode_config(self) -> None:
        """Write opencode.json into the workspace.

        Mirrors the Claude Code harness as closely as opencode permits:
        - MCP server key is `mcp__forecast_` so opencode's `<key>_<tool>`
          naming produces `mcp__forecast__search_news` etc., matching CC.
        - The forecasting system prompt is inlined into the custom agent
          `forecast` so the model receives the same instructions as CC.
          (opencode's built-in system prompt always prepends; this is an
          opencode limitation — agent.prompt augments rather than replaces.)
        - Web tools and opencode-only meta tools are disabled so the tool
          surface matches CC's forecasting scaffold as closely as possible.
        """
        mcp_cmd, mcp_args = self._build_mcp_invocation()

        # active_memory{,2} mode delivers the full daily instructions as the user
        # prompt each day (mirroring claude_code/codex). opencode requires a
        # non-empty agent.prompt, so use a minimal one-liner.
        if self.config.prompt_mode in ("active_memory", "active_memory2"):
            system_prompt = "You are a forecasting agent. Follow the per-day instructions in the user message."
        else:
            system_prompt = (self._internal_dir / "system_prompt.md").read_text()

        config = {
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                "openrouter": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "OpenRouter",
                    "options": {"baseURL": "https://openrouter.ai/api/v1"},
                    "models": {self.config.model: {"name": self.config.model}},
                }
            },
            "mcp": {
                "mcp__forecast_": {
                    "type": "local",
                    "command": [mcp_cmd] + mcp_args,
                    "enabled": True,
                }
            },
            # Disable opencode-only meta tools that CC's forecasting harness
            # doesn't expose, plus the web tools CC disallows explicitly.
            "tools": {
                "webfetch": False,
                "websearch": False,
                "task": False,
                "todowrite": False,
                "todoread": False,
                "question": False,
                "patch": False,
            },
            "agent": {
                "forecast": {
                    "prompt": system_prompt,
                    "mode": "primary",
                }
            },
        }
        with open(self.workspace / "opencode.json", "w") as f:
            json.dump(config, f, indent=2)

    def _opencode_variant_flags(self) -> List[str]:
        # Default opencode reasoning effort to high; callers can override by
        # passing --variant in extra_flags.
        if any(f == "--variant" for f in self.config.extra_flags):
            return []
        return ["--variant", "high"]

    # ── egress proxy (network_isolation) ───────────────────────────────

    def _start_egress_proxy(self) -> None:
        """Start the host-side allowlist proxy daemon on a per-agent unix socket.

        Idempotent — second call is a no-op if the daemon is already running.
        Sockets live on local scratch (Condor scratch / /tmp) since unix-domain
        sockets aren't reliable on shared FS (Lustre/NFS), and are bind-mounted
        into the sandbox at the same realpath.
        """
        if self._egress_proc is not None and self._egress_proc.poll() is None:
            return
        scratch = os.path.realpath(os.environ.get("_CONDOR_SCRATCH_DIR") or tempfile.gettempdir())
        # Include a short hash of the run output dir so parallel sims with the
        # same agent_id (same model name, different sim_name/timestamp) don't
        # collide on the same unix-socket path.
        run_tag = hashlib.sha1(
            str(Path(self._internal_dir).resolve()).encode()
        ).hexdigest()[:8]
        d = Path(scratch) / f"egress_{self.agent_id}_{run_tag}"
        d.mkdir(parents=True, exist_ok=True)
        os.chmod(d, 0o700)
        self._egress_dir = d
        self._egress_proxy_sock = d / "proxy.sock"

        # Detect a running embedding server so we can raw-forward it. Detection
        # runs on the host (where loopback works); the resulting URL is
        # rewritten so the in-sandbox bridge port matches.
        host_embed_url = self._detect_embedding_server()
        raw_forwards: List[str] = []
        if host_embed_url:
            self._egress_embed_port = int(host_embed_url.rsplit(":", 1)[1])
            self._egress_embed_sock = d / "embed.sock"
            raw_forwards.append(f"{self._egress_embed_sock}=127.0.0.1:{self._egress_embed_port}")

        allow = list(self.config.egress_allowlist) or list(DEFAULT_EGRESS_ALLOWLIST)

        proxy_script = str(Path(__file__).resolve().parent / "egress_proxy.py")
        repo_root = str(Path(__file__).resolve().parents[2])
        venv_python = os.path.join(repo_root, ".venv", "bin", "python")
        python_cmd = venv_python if os.path.exists(venv_python) else sys.executable

        cmd = [python_cmd, proxy_script,
               "--proxy-socket", str(self._egress_proxy_sock)]
        for a in allow:
            cmd.extend(["--allow", a])
        for rf in raw_forwards:
            cmd.extend(["--raw-forward", rf])

        # ready_fd: the proxy writes "ready\n" once both servers are listening.
        r_fd, w_fd = os.pipe()
        cmd.extend(["--ready-fd", str(w_fd)])

        log_path = self._internal_dir / "egress_proxy.log"
        log_f = open(log_path, "a")
        logger.info("Starting egress proxy: %s", " ".join(cmd))
        self._egress_proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_f,
            stderr=log_f,
            pass_fds=(w_fd,),
        )
        os.close(w_fd)
        # Block until the proxy reports ready (or dies / 5s timeout).
        ready = b""
        try:
            os.set_blocking(r_fd, False)
            deadline = time.time() + 5.0
            while time.time() < deadline:
                try:
                    chunk = os.read(r_fd, 16)
                    if chunk:
                        ready += chunk
                        if b"ready" in ready:
                            break
                except BlockingIOError:
                    pass
                if self._egress_proc.poll() is not None:
                    break
                time.sleep(0.05)
        finally:
            os.close(r_fd)
        if b"ready" not in ready:
            raise RuntimeError(
                f"egress proxy failed to become ready (rc={self._egress_proc.poll()}); "
                f"see {log_path}"
            )

    def _stop_egress_proxy(self) -> None:
        if self._egress_proc and self._egress_proc.poll() is None:
            self._egress_proc.terminate()
            try:
                self._egress_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._egress_proc.kill()
        self._egress_proc = None

    # ── MCP relay (sandbox=True) ───────────────────────────────────────

    def _mcp_tool_filter_args(self) -> List[str]:
        if self.config.prompt_mode == "static_search":
            return ["--enabled-tools", "submit_forecasts"]
        if self.config.prompt_mode == "warmup":
            return ["--enabled-tools", "search_news,submit_forecasts"]
        return []

    def _build_mcp_invocation(self) -> Tuple[str, List[str]]:
        """Return (command, args) for the harness' MCP config.

        sandbox=False: the harness spawns mcp_server.py as a child process
        directly. The MCP server runs alongside the harness with full repo
        access — this is the pre-refactor behavior.

        sandbox=True: the MCP server runs on the host (started via
        _start_mcp_relay) and the harness reaches it through a per-agent
        unix socket. The harness invokes `socat - UNIX-CONNECT:<sock>` as
        its MCP child; socat bridges Claude Code's stdio to the host's
        socat, which forks a fresh mcp_server.py per connection. Keeping
        the MCP server (and therefore search_db, lancedb, and the
        embedding model) on the host prevents the sandboxed harness from
        bypassing date-capping by querying the index directly.
        """
        if self.config.sandbox:
            if self._mcp_relay_sock is None:
                raise RuntimeError(
                    "MCP relay socket not initialized; "
                    "_start_mcp_relay must run before _build_mcp_invocation"
                )
            if self._mcp_bridge_wrapper is not None:
                return (str(self._mcp_bridge_wrapper), [])
            return (_resolve_sandbox_socat_cmd(), ["-", f"UNIX-CONNECT:{self._mcp_relay_sock}"])

        repo_root = str(Path(__file__).resolve().parents[2])
        mcp_server_script = str(Path(__file__).resolve().parent / "mcp_server.py")
        venv_python = os.path.join(repo_root, ".venv", "bin", "python")
        python_cmd = venv_python if os.path.exists(venv_python) else "python"
        embed_server_url = self._detect_embedding_server()

        mcp_args = [
            mcp_server_script,
            "--workspace", str(self.workspace),
            "--internal-dir", str(self._internal_dir),
            "--repo-root", repo_root,
            "--search-db", self.config.search_db or "",
            "--embedding-model", self.config.embedding_model or "",
            "--handholding-version", self.config.handholding_version,
            "--agent-id", self.agent_id,
        ]
        mcp_args.extend(self._mcp_tool_filter_args())
        if embed_server_url:
            mcp_args.extend(["--embedding-server-url", embed_server_url])
        if self.config.prompt_mode == "active_memory2":
            mcp_args.append("--enable-memory-tools")
        return (python_cmd, mcp_args)

    def _start_mcp_relay(self) -> None:
        """Spawn host-side MCP server bridged via a unix socket. Idempotent.

        Only used when sandbox=True. The host listener is `socat
        UNIX-LISTEN:<sock>,fork EXEC:<wrapper.sh>`; each new connection
        from inside the sandbox forks a fresh mcp_server.py wired through
        the wrapper script's stdio. The wrapper exists so paths with
        spaces don't trip socat's whitespace-split EXEC parser.
        """
        if self._mcp_relay_proc is not None and self._mcp_relay_proc.poll() is None:
            return

        scratch = os.path.realpath(os.environ.get("_CONDOR_SCRATCH_DIR") or tempfile.gettempdir())
        run_tag = hashlib.sha1(
            str(Path(self._internal_dir).resolve()).encode()
        ).hexdigest()[:8]
        d = Path(scratch) / f"mcp_{self.agent_id}_{run_tag}"
        d.mkdir(parents=True, exist_ok=True)
        os.chmod(d, 0o700)
        self._mcp_relay_dir = d
        self._mcp_relay_sock = d / "mcp.sock"

        # Stale socket from a prior run will block bind; remove it.
        if self._mcp_relay_sock.exists() or self._mcp_relay_sock.is_symlink():
            self._mcp_relay_sock.unlink()

        repo_root = str(Path(__file__).resolve().parents[2])
        mcp_server_script = str(Path(__file__).resolve().parent / "mcp_server.py")
        venv_python = os.path.join(repo_root, ".venv", "bin", "python")
        python_cmd = venv_python if os.path.exists(venv_python) else sys.executable

        embed_server_url = self._detect_embedding_server()

        mcp_args = [
            mcp_server_script,
            "--workspace", str(self.workspace),
            "--internal-dir", str(self._internal_dir),
            "--repo-root", repo_root,
            "--search-db", self.config.search_db or "",
            "--embedding-model", self.config.embedding_model or "",
            "--handholding-version", self.config.handholding_version,
            "--agent-id", self.agent_id,
        ]
        mcp_args.extend(self._mcp_tool_filter_args())
        if embed_server_url:
            mcp_args.extend(["--embedding-server-url", embed_server_url])
        if self.config.prompt_mode == "active_memory2":
            mcp_args.append("--enable-memory-tools")

        wrapper = d / "launch_mcp.sh"
        wrapper.write_text(
            "#!/bin/sh\nexec " + shlex.join([python_cmd, *mcp_args]) + "\n"
        )
        wrapper.chmod(0o755)

        bridge_wrapper = d / "connect_mcp.sh"
        bridge_wrapper.write_text(
            "#!/bin/sh\n"
            f"sock={shlex.quote(str(self._mcp_relay_sock))}\n"
            'if [ ! -S "$sock" ]; then\n'
            '  echo "MCP relay socket missing: $sock" >&2\n'
            "  exit 127\n"
            "fi\n"
            "for socat_cmd in /usr/bin/socat /bin/socat socat; do\n"
            '  if command -v "$socat_cmd" >/dev/null 2>&1; then\n'
            '    exec "$socat_cmd" - "UNIX-CONNECT:$sock"\n'
            "  fi\n"
            "done\n"
            'echo "socat not found in sandbox" >&2\n'
            "exit 127\n"
        )
        bridge_wrapper.chmod(0o755)
        self._mcp_bridge_wrapper = bridge_wrapper

        # NOTE: do NOT add `stderr` to the EXEC option — that dup2's the
        # subprocess's stderr onto stdout, which corrupts the JSON-RPC
        # stream the harness reads back over the unix socket. Leaving it
        # off keeps mcp_server.py stderr on socat's parent stderr (which
        # we redirect into mcp_relay.log below).
        cmd = [
            _resolve_socat_cmd(),
            f"UNIX-LISTEN:{self._mcp_relay_sock},fork",
            f"EXEC:{wrapper}",
        ]
        log_path = self._internal_dir / "mcp_relay.log"
        log_f = open(log_path, "a")
        logger.info("Starting MCP relay: %s", " ".join(cmd))
        self._mcp_relay_proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_f,
            stderr=log_f,
        )

        # Wait for socat to bind. socat creates the socket synchronously
        # before accepting; 5s is generous.
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if self._mcp_relay_sock.exists():
                return
            if self._mcp_relay_proc.poll() is not None:
                raise RuntimeError(
                    f"MCP relay died before listening (rc={self._mcp_relay_proc.returncode}); "
                    f"see {log_path}"
                )
            time.sleep(0.05)
        raise RuntimeError(
            f"MCP relay socket {self._mcp_relay_sock} did not appear within 5s; see {log_path}"
        )

    def _stop_mcp_relay(self) -> None:
        if self._mcp_relay_proc and self._mcp_relay_proc.poll() is None:
            self._mcp_relay_proc.terminate()
            try:
                self._mcp_relay_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._mcp_relay_proc.kill()
        self._mcp_relay_proc = None
        if self._mcp_relay_sock and self._mcp_relay_sock.exists():
            try:
                self._mcp_relay_sock.unlink()
            except OSError:
                pass
        self._mcp_bridge_wrapper = None

    def _wrap_for_egress_bridge(self, cmd: List[str]) -> List[str]:
        """Prefix `cmd` with a /bin/sh wrapper that starts socat sidecars
        (TCP-listen on sandbox loopback → unix-connect to bind-mounted host
        socket) and then exec's the harness. Only applied when
        network_isolation=True."""
        proxy_sock = str(self._egress_proxy_sock)
        proxy_port = self._egress_proxy_port
        socat_cmd = shlex.quote(_resolve_sandbox_socat_cmd())
        bridge = (
            f"{socat_cmd} TCP-LISTEN:{proxy_port},bind=127.0.0.1,fork,reuseaddr "
            f"UNIX-CONNECT:{proxy_sock} 2>/dev/null & "
        )
        if self._egress_embed_sock and self._egress_embed_port:
            bridge += (
                f"{socat_cmd} TCP-LISTEN:{self._egress_embed_port},bind=127.0.0.1,"
                f"fork,reuseaddr UNIX-CONNECT:{self._egress_embed_sock} 2>/dev/null & "
            )
        # Brief grace for socat to bind its listening sockets before the
        # harness fires its first request. socat binds in <50ms typically;
        # 0.3s is conservative.
        wrapper = bridge + 'sleep 0.3; exec "$@"'
        return ["/bin/sh", "-c", wrapper, "egress-bridge", *cmd]

    def _egress_env_overrides(self) -> Dict[str, str]:
        """Env vars to set in the harness subprocess so SDK clients route
        through the in-sandbox bridge to the host proxy."""
        proxy_url = f"http://127.0.0.1:{self._egress_proxy_port}"
        # NO_PROXY=127.0.0.1,localhost so the embedding client (which already
        # bypasses HTTPS_PROXY explicitly) and any other loopback-to-loopback
        # call stays local. The loopback embedding server reaches the host via
        # the raw-forward bridge on the same port.
        return {
            "HTTPS_PROXY": proxy_url,
            "https_proxy": proxy_url,
            "HTTP_PROXY": proxy_url,
            "http_proxy": proxy_url,
            "ALL_PROXY": proxy_url,
            "all_proxy": proxy_url,
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }

    def _build_bwrap_cmd(self, inner_cmd: List[str]) -> List[str]:
        """Wrap `inner_cmd` in bwrap for filesystem isolation.

        The sandbox hides everything by default and only exposes the
        absolute minimum the harness binary needs to run:
          - workspace (rw) — harness cwd; articles/, memory/, market.csv,
            plus any files the model writes
          - _internal_dir (rw) — MCP signals/, predictions/, state.json,
            system_prompt.md
          - mcp_relay_dir (rw) — per-agent unix socket for the host-side
            MCP server (the only path through which retrieval is reachable)
          - harness binary install tree (ro), harness state dirs under ~ (rw):
              codex:       ~/.codex (auth, sessions, cache)
              claude_code: ~/.claude, ~/.claude.json, ~/.cache/claude*
              opencode:    ~/.cache|.local|.config/opencode plus an
                           XDG_DATA_HOME scratch for its SQLite session DB
          - /usr, /etc, /lib, /lib64, /bin, /sbin (ro) — OS tools/libs
            (provides /usr/bin/python3, socat, sh, curl, etc.)
          - /proc (fresh namespace by default, or host read-only with
            sandbox_proc_mode="host_ro"), /dev (stripped), /tmp (tmpfs)

        NOT bound (stays invisible to the model):
          - the forecast-sim repo itself — the harness binary is
            self-contained and the MCP server runs host-side; binding the
            repo would expose analysis/reports/ ground-truth tables,
            prior-run logs under logs/sims/, and other sim state
          - .venv / sys.base_prefix — no Python is invoked inside the
            sandbox; system /usr/bin/python3 is sufficient when needed
          - search_db / lancedb / embedding_model — retrieval lives on the
            host behind the MCP relay, so the harness can only search via
            `mcp__forecast__search_news`, where date-capping is enforced
          - articles_base — date-gating is enforced by hardlinks we placed in
            workspace/articles; articles_base itself is unreachable
          - FSIM_DATASET_PATH — HF dataset with resolution labels
          - sibling sim dirs, daily_metrics.csv, test_daily_metrics.csv,
            actions.jsonl, matcher.jsonl in parent output dir
          - /home outside explicitly bound harness subdirs

        Every path we care about is bound at its realpath, and top-level
        symlink aliases (/fast, /is/cluster/fast) are recreated via
        --symlink so code using aliased paths still resolves.

        By default, network is kept via --share-net so OpenRouter + the local
        vLLM embedding server at 127.0.0.1:800x remain reachable. When
        network_isolation=True, --share-net is replaced with --unshare-net
        and outbound traffic is funnelled through the host-side allowlist
        proxy (see _start_egress_proxy + _wrap_for_egress_bridge).
        """
        workspace_real = os.path.realpath(str(self.workspace))
        internal_real = os.path.realpath(str(self._internal_dir))

        args = [
            "bwrap",
            "--unshare-net" if self.config.network_isolation else "--share-net",
            "--die-with-parent",
            "--unshare-ipc",
            "--unshare-uts",
            "--dev", "/dev",
            "--tmpfs", "/tmp",
            "--ro-bind", "/usr", "/usr",
            "--ro-bind", "/etc", "/etc",
            "--ro-bind-try", "/lib", "/lib",
            "--ro-bind-try", "/lib64", "/lib64",
            "--ro-bind-try", "/bin", "/bin",
            "--ro-bind-try", "/sbin", "/sbin",
        ]
        if self.config.sandbox_proc_mode == "new":
            args.extend(["--unshare-pid", "--proc", "/proc"])
        else:
            # Some Condor execution environments forbid mounting a new procfs.
            # This preserves the filesystem sandbox while exposing process
            # metadata read-only instead of creating a private PID namespace.
            args.extend(["--ro-bind", "/proc", "/proc"])

        # Recreate top-level path aliases BEFORE any binds whose DEST would
        # implicitly create the alias dir as tmpfs. Each alias only resolves
        # to paths we separately bind below, so this leaks no data.
        # (e.g. on this cluster /home -> /lustre/home, /fast -> /lustre/fast/fast.)
        for alias in ("/home", "/fast", "/is/cluster/fast"):
            if os.path.islink(alias):
                target = os.readlink(alias)
                if not os.path.isabs(target):
                    target = os.path.normpath(os.path.join(os.path.dirname(alias), target))
                # Strip any trailing slash that readlink may return.
                target = target.rstrip("/") or "/"
                args.extend(["--symlink", target, alias])

        # Workspace + internal dir (rw), at realpath so DEST dirs don't clash
        # with the alias symlinks above.
        args.extend(["--bind", workspace_real, workspace_real])
        args.extend(["--bind", internal_real, internal_real])

        # Overlay workspace/articles as read-only. Articles are hardlinked from
        # articles_base — since hardlinks share permissions with the source,
        # a write through the hardlink would corrupt articles_base. The ro
        # overlay blocks writes regardless. Driver writes on the host side are
        # unaffected (they go straight to the filesystem, not through bwrap).
        articles_dst = os.path.join(workspace_real, "articles")
        if os.path.exists(articles_dst):
            args.extend(["--ro-bind", articles_dst, articles_dst])

        # The forecast-sim repo is intentionally NOT bound. Earlier the
        # MCP server ran inside the sandbox and needed agents/, environment/,
        # and the repo's .venv. The MCP server now runs host-side over the
        # unix-socket relay (see _start_mcp_relay), and the harness binaries
        # (codex/claude/opencode) are self-contained — none of them import
        # repo Python. Binding the repo would re-expose analysis/reports/
        # ground-truth tables, sibling-run logs under logs/sims/, prior-run
        # state.json/predictions/, and the YAML configs themselves, all of
        # which would let the model recover answers without any retrieval.

        # LanceDB search_db and the embedding_model are intentionally NOT
        # bound when sandbox=True. They live on the host alongside the
        # MCP relay (see _start_mcp_relay); the harness reaches search via
        # the MCP `search_news` tool only, where date-capping is enforced.
        # Binding the index here would let the harness open lancedb
        # directly and bypass the date filter.

        # Harness binary install tree (ro).
        if self.config.harness_backend == "claude_code":
            # claude lives at <install>/versions/<v>, where <install> is
            # ~/.local/share/claude. Bind the install dir; _start_claude_code
            # passes the realpath as argv[0] so the bin symlink isn't needed.
            bin_real = os.path.realpath(self.config.claude_code_path)
            if os.path.exists(bin_real):
                install = os.path.dirname(os.path.dirname(bin_real))
                args.extend(["--ro-bind", install, install])
            home_subs = (".claude", ".claude.json",
                         ".cache/claude", ".cache/claude-cli-nodejs")
        elif self.config.harness_backend == "codex":
            # codex lives at <install>/versions/<v>/codex (e.g.
            # ~/.local/share/codex/versions/0.125.0/codex). Bind the install
            # dir RO. Auth/session/cache live in ~/.codex (RW).
            bin_real = os.path.realpath(self.config.codex_path)
            if os.path.exists(bin_real):
                install = os.path.dirname(os.path.dirname(bin_real))
                args.extend(["--ro-bind", install, install])
            home_subs = (".codex",)
        else:
            bin_real = os.path.realpath(self.config.opencode_path)
            if os.path.exists(bin_real):
                oc_bin_dir = os.path.dirname(bin_real)
                oc_install = os.path.dirname(oc_bin_dir)
                if os.path.basename(oc_install) == ".opencode":
                    args.extend(["--ro-bind", oc_install, oc_install])
                else:
                    args.extend(["--ro-bind", oc_bin_dir, oc_bin_dir])
            home_subs = (".cache/opencode", ".local/share/opencode", ".config/opencode")

        # Harness runtime state in user home (reads+writes).
        home = os.path.expanduser("~")
        for sub in home_subs:
            p = os.path.realpath(os.path.join(home, sub))
            if os.path.exists(p):
                args.extend(["--bind-try", p, p])

        # Per-agent XDG_DATA_HOME scratch for opencode's SQLite session DB.
        data_home = getattr(self, "_opencode_data_home", None)
        if data_home:
            dh = os.path.realpath(str(data_home))
            if os.path.exists(dh):
                args.extend(["--bind", dh, dh])

        # network_isolation: bind-mount the per-agent egress socket dir at the
        # same realpath so the in-sandbox bridge can UNIX-CONNECT to it.
        if self.config.network_isolation and self._egress_dir:
            ed = os.path.realpath(str(self._egress_dir))
            args.extend(["--bind", ed, ed])

        # sandbox: bind-mount the per-agent MCP relay socket dir so the
        # harness' `socat - UNIX-CONNECT:<sock>` MCP child can reach the
        # host-side MCP server.
        if self._mcp_relay_dir:
            md = os.path.realpath(str(self._mcp_relay_dir))
            if os.path.exists(md):
                args.extend(["--bind", md, md])

        # chdir inside the sandbox to the workspace realpath (matches the bind).
        args.extend(["--chdir", workspace_real])
        args.append("--")
        # When network_isolation is on, prefix the harness with a tiny shell
        # that brings up TCP→Unix-socket bridges before exec'ing the harness.
        if self.config.network_isolation:
            inner_cmd = self._wrap_for_egress_bridge(list(inner_cmd))
        args.extend(inner_cmd)
        return args

    def _maybe_sandbox(self, cmd: List[str]) -> List[str]:
        """Return cmd wrapped in bwrap when sandbox is enabled, else unchanged."""
        if self.config.sandbox:
            return self._build_bwrap_cmd(cmd)
        return cmd

    def _start_opencode(self) -> None:
        if (
            self.config.prompt_mode in ("active_memory", "active_memory2")
            and self._codex_initial_prompt_override
        ):
            initial_prompt = self._codex_initial_prompt_override
        else:
            initial_prompt = (
                "Begin forecasting. Read market.csv to see your questions, "
                "then research and submit predictions."
            )
        model_id = f"openrouter/{self.config.model}"
        cmd = [
            self.config.opencode_path,
            "run",
            "--model", model_id,
            "--agent", "forecast",
            "--dangerously-skip-permissions",
            "--format", "json",
            "--print-logs",
            "--log-level", "INFO",
            initial_prompt,
        ]
        cmd.extend(self._opencode_variant_flags())
        cmd.extend(self.config.extra_flags)

        env = os.environ.copy()
        if self.config.openrouter_api_key:
            env["OPENROUTER_API_KEY"] = self.config.openrouter_api_key
        # Isolate opencode's SQLite session DB per agent. Must live on local
        # disk — shared FS (Lustre/NFS) breaks SQLite WAL ("disk I/O error").
        scratch = os.path.realpath(os.environ.get("_CONDOR_SCRATCH_DIR") or tempfile.gettempdir())
        run_tag = hashlib.sha1(
            str(Path(self._internal_dir).resolve()).encode()
        ).hexdigest()[:8]
        data_home = Path(scratch) / f"opencode_{self.agent_id}_{run_tag}"
        data_home.mkdir(parents=True, exist_ok=True)
        env["XDG_DATA_HOME"] = str(data_home)
        self._opencode_data_home = data_home
        if self.config.network_isolation:
            env.update(self._egress_env_overrides())

        # Append mode so per-day respawn (active_memory) doesn't truncate
        # prior days' logs; harmless in default single-process mode.
        log_stdout = open(self._internal_dir / "opencode_stdout.jsonl", "a")
        log_stderr = open(self._internal_dir / "opencode_stderr.log", "a")

        launch_cmd = self._maybe_sandbox(cmd)
        # When sandboxed, bwrap sets cwd via --chdir; passing cwd= to Popen
        # would chdir in the host namespace before exec and can fail if the
        # path is visible only through symlinks that bwrap reshapes.
        popen_cwd = None if self.config.sandbox else str(self.workspace)

        logger.info("Starting opencode%s%s: %s",
                    " [sandboxed]" if self.config.sandbox else "",
                    " [net-isolated]" if self.config.network_isolation else "",
                    " ".join(launch_cmd))
        self._harness_proc = subprocess.Popen(
            launch_cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_stdout,
            stderr=log_stderr,
            cwd=popen_cwd,
            env=env,
        )
        self._stdout_log = log_stdout
        self._stderr_log = log_stderr

    # ── codex backend ──────────────────────────────────────────────────

    def _start_codex(self) -> None:
        """Spawn a `codex exec` process for the current sim day.

        Codex doesn't have a persistent `--continue` mode like Claude Code:
        `codex exec` runs once and exits.  Two strategies are supported:

        - Default (codex_resume=False): act() calls _start_codex each day
          with a freshly-rewritten state.json; every day starts a NEW codex
          thread that picks up workspace state from disk only.

        - codex_resume=True: Day 0 starts a fresh thread (captured into
          self._codex_thread_id), and subsequent days run
          `codex exec resume <thread_id>` so the model retains yesterday's
          reasoning in context (codex auto-compacts when needed).

        System prompt: codex auto-loads `AGENTS.md` from cwd, so we copy
        system_prompt.md → workspace/AGENTS.md.
        MCP server: configured via `-c mcp_servers.forecast.command/args`.
        With sandbox=True the command becomes a socat bridge to the
        host-side MCP relay (see _build_mcp_invocation).
        """
        mcp_cmd, mcp_args = self._build_mcp_invocation()

        # These modes deliver the full task as the user prompt instead of
        # AGENTS.md; clear any stale copy from a prior config so Codex can't
        # pick it up.
        if self.config.prompt_mode in {"active_memory", "warmup", "static_search", "active_memory2"}:
            stale_agents_md = self.workspace / "AGENTS.md"
            if stale_agents_md.exists():
                stale_agents_md.unlink()
        else:
            system_prompt = (self._internal_dir / "system_prompt.md").read_text()
            (self.workspace / "AGENTS.md").write_text(system_prompt)

        # When sandboxed, pass the realpath of the codex binary as argv[0] so
        # the subprocess can exec it without needing the ~/.local/bin/codex
        # symlink to be preserved inside the sandbox.
        codex_bin = (
            os.path.realpath(self.config.codex_path)
            if self.config.sandbox
            else self.config.codex_path
        )

        # `codex exec` accepts -C (cwd); `codex exec resume` does NOT — it
        # inherits cwd from the spawning process instead. Keep -C in the
        # fresh-spawn branch only; resume relies on Popen cwd / bwrap --chdir.
        common_args = [
            "-m", self.config.model,
            "-c", f'model_reasoning_effort="{self.config.reasoning_effort}"',
            # Codex exposes native web_search by default in current CLIs. Forecast
            # sims must use the date-capped MCP search_news tool / staged articles
            # only, so remove native web search from every codex exec/resume call.
            "-c", 'web_search="disabled"',
            "-c", f'mcp_servers.forecast.command="{mcp_cmd}"',
            "-c", f"mcp_servers.forecast.args={json.dumps(mcp_args)}",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "--json",
        ]
        if self.config.network_isolation:
            proxy_url = f"http://127.0.0.1:{self._egress_proxy_port}"
            common_args.extend(["-c", f'network.proxy_url="{proxy_url}"'])
        if self.config.prompt_mode == "static_search":
            common_args.extend([
                "--disable", "shell_tool",
                "--disable", "shell_snapshot",
                "--disable", "apply_patch_freeform",
                "--disable", "browser_use",
                "--disable", "computer_use",
                "--disable", "tool_search",
                "--disable", "tool_suggest",
                "--disable", "image_generation",
                "--disable", "multi_agent",
            ])

        if self.config.codex_resume and self._codex_thread_id is None:
            self._find_codex_thread_id()
        use_resume = (
            self.config.codex_resume
            and self._codex_thread_id is not None
        )

        if use_resume:
            day_iso = (
                self._current_date.isoformat()
                if self._current_date else ""
            )
            if self.config.prompt_mode in {"warmup", "static_search"} and self._codex_initial_prompt_override:
                resume_prompt = (
                    "The previous per-question warmup session was interrupted. "
                    "Continue the same target question from the current state, "
                    "submit the forecast if needed.\n\n"
                    f"{self._codex_initial_prompt_override}"
                )
            else:
                resume_prompt = (
                    f"Simulation advanced to {day_iso}. Re-read state.json and "
                    "market.csv for the new day's questions, resolution events, "
                    "and your existing predictions; update predictions where "
                    "new info changes your view, then call next_day."
                )
            cmd = [
                codex_bin, "exec", "resume", self._codex_thread_id,
                *common_args, resume_prompt,
            ]
        else:
            if (
                self.config.prompt_mode in {"active_memory", "warmup", "static_search", "active_memory2"}
                and self._codex_initial_prompt_override
            ):
                initial_prompt = self._codex_initial_prompt_override
            else:
                initial_prompt = (
                    "Begin forecasting. Read market.csv to see your questions, "
                    "then research and submit predictions."
                )
            cmd = [
                codex_bin, "exec",
                *common_args,
                "-C", str(self.workspace),
                initial_prompt,
            ]
        cmd.extend(self.config.extra_flags)

        # Append mode so multi-day re-spawns don't truncate prior days' logs.
        log_stdout = open(self._internal_dir / "codex_stdout.jsonl", "a")
        log_stderr = open(self._internal_dir / "codex_stderr.log", "a")

        env = os.environ.copy()
        if self.config.network_isolation:
            env.update(self._egress_env_overrides())

        launch_cmd = self._maybe_sandbox(cmd)
        # When sandboxed, bwrap sets cwd via --chdir; passing cwd= to Popen
        # would chdir in the host namespace before exec.
        popen_cwd = None if self.config.sandbox else str(self.workspace)

        logger.info("Starting codex%s%s: %s",
                    " [sandboxed]" if self.config.sandbox else "",
                    " [net-isolated]" if self.config.network_isolation else "",
                    " ".join(launch_cmd))
        self._harness_proc = subprocess.Popen(
            launch_cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_stdout,
            stderr=log_stderr,
            cwd=popen_cwd,
            env=env,
        )
        self._stdout_log = log_stdout
        self._stderr_log = log_stderr

    def _find_session_id(self) -> Optional[str]:
        """Scan opencode_stdout.jsonl for the first sessionID field.

        opencode emits the sessionID on its first event after launch; once found
        we cache it so resumes use `opencode run --session <id>`.
        """
        if self._session_id:
            return self._session_id
        path = self._internal_dir / "opencode_stdout.jsonl"
        if not path.exists():
            return None
        with open(path) as f:
            for line in f:
                try:
                    sid = json.loads(line).get("sessionID")
                except Exception:
                    continue
                if sid:
                    self._session_id = sid
                    return sid
        return None

    def _find_claude_code_session_id(self) -> Optional[str]:
        """Scan claude_code_stdout.jsonl for the first system/init session_id.

        claude emits {"type":"system","subtype":"init","session_id":"<uuid>",...}
        as its first event; we cache it so claude_code_resume mode can pass
        that uuid to `claude --resume <session_id>` on relaunch.
        """
        if self._claude_code_session_id:
            return self._claude_code_session_id
        path = self._internal_dir / "claude_code_stdout.jsonl"
        if not path.exists():
            return None
        with open(path) as f:
            for line in f:
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if ev.get("type") == "system" and ev.get("subtype") == "init":
                    sid = ev.get("session_id")
                    if sid:
                        self._claude_code_session_id = sid
                        return sid
        return None

    def _find_codex_thread_id(self) -> Optional[str]:
        """Scan codex_stdout.jsonl for the first thread.started event.

        codex emits {"type":"thread.started","thread_id":"<uuid>"} as its
        first event; we cache it so codex_resume mode can pass that uuid to
        `codex exec resume <thread_id>` on subsequent days.
        """
        if self._codex_thread_id:
            return self._codex_thread_id
        path = self._internal_dir / "codex_stdout.jsonl"
        if not path.exists():
            return None
        latest_tid = None
        with open(path) as f:
            for line in f:
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if ev.get("type") == "thread.started":
                    tid = ev.get("thread_id")
                    if tid:
                        latest_tid = tid
        if latest_tid:
            self._codex_thread_id = latest_tid
            return latest_tid
        return None

    def _resume_opencode(self, current_date: date) -> bool:
        """Relaunch opencode resuming the same session.

        Returns True if relaunch succeeded (new process running), False on
        unrecoverable error (no session id / exec failure).
        """
        session_id = self._find_session_id()
        if not session_id:
            logger.error("opencode died and no session_id found — cannot resume")
            return False

        model_id = f"openrouter/{self.config.model}"
        resume_prompt = "Please continue from where you left."
        cmd = [
            self.config.opencode_path,
            "run",
            "--session", session_id,
            "--model", model_id,
            "--agent", "forecast",
            "--dangerously-skip-permissions",
            "--format", "json",
            "--print-logs",
            "--log-level", "INFO",
            resume_prompt,
        ]
        cmd.extend(self._opencode_variant_flags())
        cmd.extend(self.config.extra_flags)

        env = os.environ.copy()
        if self.config.openrouter_api_key:
            env["OPENROUTER_API_KEY"] = self.config.openrouter_api_key
        env["XDG_DATA_HOME"] = str(self._opencode_data_home)
        if self.config.network_isolation:
            env.update(self._egress_env_overrides())

        try:
            self._stdout_log.close()
            self._stderr_log.close()
        except Exception:
            pass
        log_stdout = open(self._internal_dir / "opencode_stdout.jsonl", "a")
        log_stderr = open(self._internal_dir / "opencode_stderr.log", "a")

        launch_cmd = self._maybe_sandbox(cmd)
        popen_cwd = None if self.config.sandbox else str(self.workspace)

        logger.warning(
            "opencode exited (rc=%s) — resuming session %s for %s%s",
            self._harness_proc.returncode if self._harness_proc else "?",
            session_id, current_date,
            " [sandboxed]" if self.config.sandbox else "",
        )
        self._harness_proc = subprocess.Popen(
            launch_cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_stdout,
            stderr=log_stderr,
            cwd=popen_cwd,
            env=env,
        )
        self._stdout_log = log_stdout
        self._stderr_log = log_stderr
        return True

    def _resume_claude_code(self, current_date: date) -> bool:
        """Relaunch claude with `--resume <session_id>` after early exit.

        Reuses _start_claude_code's resume branch by temporarily forcing
        config.claude_code_resume=True. Returns False if no prior session_id
        is recoverable from claude_code_stdout.jsonl.
        """
        if not self._find_claude_code_session_id():
            logger.error("claude_code died and no session_id found — cannot resume")
            return False
        rc = self._harness_proc.returncode if self._harness_proc else "?"
        prev = self.config.claude_code_resume
        self.config.claude_code_resume = True
        try:
            self._start_claude_code()
        finally:
            self.config.claude_code_resume = prev
        logger.warning(
            "claude_code exited (rc=%s) — resumed session %s for %s",
            rc, self._claude_code_session_id, current_date,
        )
        return True

    def _resume_codex(self, current_date: date) -> bool:
        """Relaunch codex via `exec resume <thread_id>` after early exit.

        Reuses _start_codex's resume branch by ensuring _codex_thread_id is
        populated and temporarily forcing config.codex_resume=True. Returns
        False if no prior thread_id is recoverable from codex_stdout.jsonl.
        """
        if self._codex_thread_id is None:
            self._find_codex_thread_id()
        if self._codex_thread_id is None:
            logger.error("codex died and no thread_id found — cannot resume")
            return False
        rc = self._harness_proc.returncode if self._harness_proc else "?"
        prev = self.config.codex_resume
        self.config.codex_resume = True
        try:
            self._start_codex()
        finally:
            self.config.codex_resume = prev
        logger.warning(
            "codex exited (rc=%s) — resumed thread %s for %s",
            rc, self._codex_thread_id, current_date,
        )
        return True

    # ── signal coordination ────────────────────────────────────────────

    def _wait_for_next_day(self, current_date: date) -> List[Dict[str, Any]]:
        """Poll for the MCP server's next_day signal, then return predictions.

        Reads predictions from the signal file (authoritative snapshot from
        _today_predictions at the moment next_day was called), falling back to
        the predictions/{date}.json file if the signal doesn't contain them.
        """
        signal_path = self._internal_dir / "signals" / f"next_day_{current_date.isoformat()}.json"
        poll_interval = 1.0
        timeout = self.config.timeout_seconds
        start = time.time()

        MAX_RESUMES_PER_DAY = 3

        while not signal_path.exists():
            # Check if harness process died.
            if self._harness_proc and self._harness_proc.poll() is not None:
                backend = self.config.harness_backend
                attempts = self._resume_attempts.get(current_date, 0)
                resumed = False
                if attempts < MAX_RESUMES_PER_DAY:
                    if backend == "opencode":
                        resumed = self._resume_opencode(current_date)
                    elif backend == "claude_code":
                        resumed = self._resume_claude_code(current_date)
                    elif backend == "codex":
                        resumed = self._resume_codex(current_date)
                if resumed:
                    self._resume_attempts[current_date] = attempts + 1
                    continue
                logger.error(
                    "%s exited with code %d before signaling next_day "
                    "(resume attempts=%d)",
                    backend, self._harness_proc.returncode, attempts,
                )
                return self._read_predictions(current_date)
            if time.time() - start > timeout:
                logger.error("Timeout waiting for next_day signal on %s", current_date)
                return self._read_predictions(current_date)
            time.sleep(poll_interval)

        # Read predictions from the predictions file — this is the durable record
        # of all submit_forecasts MCP calls. Each call writes to this file immediately,
        # so it survives MCP server restarts (unlike the in-memory _today_predictions).
        if (
            self.config.harness_backend == "codex"
            and self._harness_proc
            and self._harness_proc.poll() is None
        ):
            drain_timeout = min(30.0, max(0.0, timeout - (time.time() - start)))
            try:
                self._harness_proc.wait(timeout=drain_timeout)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "codex did not exit within %.1fs after next_day signal on %s; "
                    "reading durable predictions anyway",
                    drain_timeout,
                    current_date,
                )
        return self._read_predictions(current_date)

    def _read_predictions(self, current_date: date) -> List[Dict[str, Any]]:
        pred_path = self._internal_dir / "predictions" / f"{current_date.isoformat()}.json"
        if not pred_path.exists():
            logger.warning("No predictions file for %s", current_date)
            return []
        with open(pred_path) as f:
            preds = json.load(f)
        # Normalize: CC may use "question_id" or "qid" depending on whether
        # predictions were submitted via MCP or written directly via Bash.
        for p in preds:
            if "qid" in p and "question_id" not in p:
                p["question_id"] = p["qid"]
        return preds

    def _write_continue_signal(self, current_date: date) -> None:
        """Tell the MCP server's next_day poll that a new day is ready."""
        signal_dir = self._internal_dir / "signals"
        # Clean up old signals to avoid confusion.
        for old in signal_dir.glob("continue_*.json"):
            old.unlink(missing_ok=True)
        for old in signal_dir.glob("next_day_*.json"):
            old.unlink(missing_ok=True)

        signal_path = signal_dir / f"continue_{current_date.isoformat()}.json"
        with open(signal_path, "w") as f:
            json.dump({"status": "day_advanced", "date": current_date.isoformat()}, f)

    def signal_complete(self) -> None:
        """Write a simulation_complete continue signal so the MCP server's
        next_day poll unblocks and tells the harness the sim is over."""
        if not self._started:
            return
        signal_dir = self._internal_dir / "signals"
        signal_dir.mkdir(parents=True, exist_ok=True)
        # Clean old signals.
        for old in signal_dir.glob("continue_*.json"):
            old.unlink(missing_ok=True)
        for old in signal_dir.glob("next_day_*.json"):
            old.unlink(missing_ok=True)
        # Write a terminal state.json so the MCP server can read it.
        end_state = {
            "current_date": self._to_iso(self._current_date or date.today()),
            "start_date": self._to_iso(self.config.start_date or date.today()),
            "end_date": self._to_iso(self.config.end_date or date.today()),
            "questions": [],
            "resolution_events": [],
        }
        with open(self._internal_dir / "state.json", "w") as f:
            json.dump(end_state, f, default=str)
        cur = self._to_iso(self._current_date or date.today())
        signal_path = signal_dir / f"continue_{cur}.json"
        with open(signal_path, "w") as f:
            json.dump({"status": "simulation_complete", "date": cur}, f)

    # ── cleanup ────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Signal simulation complete and terminate the harness process."""
        self.signal_complete()
        if self._harness_proc and self._harness_proc.poll() is None:
            logger.info(
                "Terminating %s process (pid=%d)",
                self.config.harness_backend, self._harness_proc.pid,
            )
            self._harness_proc.terminate()
            try:
                self._harness_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._harness_proc.kill()

        self._stop_egress_proxy()
        self._stop_mcp_relay()

        for f in (getattr(self, "_stdout_log", None), getattr(self, "_stderr_log", None)):
            if f and not f.closed:
                f.close()

    def __del__(self):
        self.cleanup()
