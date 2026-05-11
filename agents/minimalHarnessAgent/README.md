# MinimalHarnessAgent

`minimalHarnessAgent` lets forecast-sim run an external coding-agent CLI as a
forecasting agent while forecast-sim keeps ownership of dates, questions,
market snapshots, scoring, article visibility, search date caps, and prediction
submission.

`MinimalHarnessAgent(...)` is still the compatibility entrypoint. Direct
construction dispatches to the backend subclass selected by
`MinimalHarnessConfig.harness_backend`.

## Runtime Shape

On each `act()` call, the shared driver:

1. Updates the per-agent workspace and internal state.
2. Stages only date-visible articles into `workspace/articles/`.
3. Writes `state.json` for the MCP server.
4. Starts the host MCP relay and optional sandbox/egress proxy when configured.
5. Starts or resumes the selected CLI backend.
6. Waits for `submit_forecasts` / `next_day` signals written by the MCP server.
7. Returns forecast-sim `PredictionSubmission` objects to the environment.

The CLI sees the workspace. Driver/MCP coordination files live in the internal
run directory and are not meant to be edited by the model.

## File Map

- `__init__.py`: exports the public classes for this package.
- `agent.py`: shared `MinimalHarnessAgent` driver. Owns lifecycle, workspace
  setup, backend dispatch, BaseAgent integration, shared process launching/log
  plumbing, signal waiting, cleanup, and backend hook definitions.
- `config.py`: `MinimalHarnessConfig` plus `DEFAULT_EGRESS_ALLOWLIST`.
  Contains backend selection, prompt mode, search/date controls, resume flags,
  sandbox flags, network isolation, warmup/static-search knobs, and bootstrap
  config.
- `state.py`: `StateHelpers`. Handles `state.json`, per-agent `market.csv`,
  bootstrap seeding, active-memory snapshot dates, embedding-server detection,
  and date-gated article staging.
- `mcp_helpers.py`: `McpHelpers`. Builds MCP command lines/config, filters enabled MCP
  tools by prompt mode, and manages the host-side MCP relay used by sandboxed
  harnesses.
- `mcp_server.py`: FastMCP server exposed to the CLI. Provides forecast tools
  such as `search_news`, `submit_forecasts`, `next_day`, and active-memory2
  memory tools. Heavy imports are lazy so the MCP handshake stays fast.
- `sandbox.py`: `SandboxHelpers`. Builds the `bwrap` filesystem sandbox and
  optional network-isolated egress bridge. Keeps repo data, datasets, raw search
  indices, sibling runs, and non-harness home dirs hidden from the model.
- `egress_proxy.py`: host-side allowlist proxy for `network_isolation=True`.
  Handles HTTP CONNECT/proxy traffic and raw-forwards the local embedding server.
- `claude_code_agent.py`: `ClaudeCodeAgent`, the Claude Code CLI backend.
- `codex_agent.py`: `CodexAgent`, the Codex CLI backend. Also owns Codex warmup
  and static-search session aggregation helpers.
- `opencode_agent.py`: `OpenCodeAgent`, the OpenCode CLI backend.
- `prompts/`: prompt builders, grouped away from harness runtime code.
- `prompts/prompt.py`: default system prompt builder and Codex warmup prompt
  builder.
- `prompts/prompt_no_memory.py`: system prompt builder for
  `prompt_mode="no_memory"`.
- `prompts/prompt_active_memory.py`: daily fresh-context prompt builder for
  `prompt_mode="active_memory"`.
- `prompts/prompt_active_memory2.py`: daily prompt and memory-update prompts
  for `prompt_mode="active_memory2"`.
- `static_search.py`: static-search cache helpers for the Codex warmup ablation.
- `README.md`: this folder guide.

Manual local debugging harness: `tests/manual/minimal_harness_harness.py`.
It is intentionally outside this package and is not part of normal pytest
collection.

## Backends

Set `harness_backend` in config:

```yaml
defaults:
  harness_backend: "codex"        # "claude_code", "opencode", or "codex"
  prompt_mode: "default"          # see prompt modes below
```

Backend subclasses own CLI-specific behavior:

- command construction and environment variables
- launch/resume/session-id handling
- backend-specific stdout/stderr log names
- install-tree and home-state sandbox binds
- whether the backend respawns each day

Shared forecast-sim semantics should stay in `agent.py` or the helper modules,
not in one backend file, unless the behavior is truly backend-specific.

## Prompt Modes

- `default`: persistent-ish CLI workflow with the shared system prompt and MCP
  tools. Resume behavior depends on backend resume flags.
- `no_memory`: no persistent memory prompt variant. Useful for clean ablations.
- `active_memory`: fresh CLI session per day. The prompt asks the model to read
  prior memory files and write today's memory files before `next_day`.
- `active_memory2`: fresh CLI session per day, but memory is managed through MCP
  tools. The MCP server loads prior memory and saves today's memory on
  `next_day`.
- `warmup`: Codex-only, fresh session per question during explicit warmup.
- `static_search`: Codex-only warmup shape with one precomputed title-search
  result and only forecast submission enabled.

`active_memory` and `active_memory2` support Codex, Claude Code, and OpenCode.
`warmup` and `static_search` currently require Codex.

## Sandbox And Data Visibility

With `sandbox=True`, the CLI runs under `bwrap`. The model sees only the
workspace, internal coordination dir, harness runtime state, required OS files,
and the MCP/egress socket dirs. The forecast-sim repo, datasets, raw search
indices, sibling simulation outputs, and most of `/home` remain hidden.

Search runs through MCP. In sandboxed mode the MCP server stays host-side, so
the CLI cannot open LanceDB/search files directly or bypass the date cap.
Articles are exposed incrementally in `workspace/articles/`; sandboxed runs use
hardlinks plus a read-only bind so `articles_base` stays hidden and source files
cannot be modified through the sandbox.

With `network_isolation=True`, `bwrap` uses a private network namespace and the
CLI reaches only configured egress targets through the host allowlist proxy.
The local embedding server is exposed separately through a raw socket forward.

## Adding Or Changing Code

Use the smallest module that owns the behavior:

- shared simulation lifecycle: `agent.py`
- state/data visibility: `state.py`
- MCP command/relay behavior: `mcp_helpers.py`
- filesystem/network isolation: `sandbox.py`
- backend launch/resume quirks: the backend file
- prompt wording: the matching `prompts/prompt*.py` file
- config surface: `config.py`

Backend files should implement the existing hook surface instead of adding
backend branches to the shared driver:

- `_prepare_harness_launch`
- `_start_harness`
- `_resume_harness`
- `_respawns_each_day`
- `_next_day_returns_immediately`
- `_after_next_day_signal`
- `_add_sandbox_harness_install_binds`
- `_sandbox_harness_home_subdirs`

After edits, at minimum run:

```bash
python -m py_compile agents/minimalHarnessAgent/*.py
pytest tests/test_minimal_harness_codex_args.py tests/test_sandbox_egress.py tests/test_raw_logging.py tests/test_resume_state.py tests/test_matcher_timing.py -q
```
