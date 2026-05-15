"""Configuration for MinimalHarnessAgent backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional


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
    # When true, wrap the harness subprocess in bwrap. The CLI sees workspace,
    # _internal_dir, harness runtime state, MCP/egress sockets, and OS dirs;
    # the repo, raw search indices, datasets, sibling sim dirs, and most of
    # /home stay invisible. Network is shared unless network_isolation=True.
    # Articles are staged via hardlink instead of symlink (required: workspace
    # and articles_base must be on the same filesystem).
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
