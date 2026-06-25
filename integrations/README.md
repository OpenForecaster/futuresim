# Futuresim Integrations

Futuresim can run as a hosted evaluation environment on:

- OpenReward / ORS
- Prime Intellect Verifiers

The OpenReward integration exposes Futuresim through OpenReward/Firehorse
sessions. For the `codex` and `claude-code` agents,
Firehorse launches the user's local Codex or Claude Code CLI on the driver
machine, then connects that CLI to the OpenReward environment through the
OpenReward harness toolset. The CLI does not run inside the OpenReward sandbox.
The Verifiers integration targets strict MinimalHarness-compatible CLI
reproduction through MCP.

Useful links:

- Blogpost: [openforecaster.github.io/futuresim](https://openforecaster.github.io/futuresim/)
- Paper: [arxiv.org/abs/2605.15188](https://arxiv.org/abs/2605.15188)
- Questions: [nikhilchandak/OpenForesight](https://huggingface.co/datasets/nikhilchandak/OpenForesight)
- Article corpus: [shash42/forecast-news](https://huggingface.co/datasets/shash42/forecast-news)
- LanceDB hybrid index: [shash42/forecast-news-embeddings](https://huggingface.co/datasets/shash42/forecast-news-embeddings)
- Embedding model: [Qwen/Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B)
- OpenReward docs: [docs.openreward.ai](https://docs.openreward.ai/)
- Firehorse quickstart: [OpenReward harness quickstart](https://docs.openreward.ai/harnesses/quickstart)
- Harness toolsets: [OpenReward harness toolsets](https://docs.openreward.ai/harnesses/harness-toolsets)
- Prime Verifiers: [Verifiers overview](https://docs.primeintellect.ai/verifiers/overview)
- Prime Sandboxes: [sandbox overview](https://docs.primeintellect.ai/sandboxes/overview)

## What You Need

For any run, provide:

- A Futuresim dataset. The default is
  [nikhilchandak/OpenForesight](https://huggingface.co/datasets/nikhilchandak/OpenForesight)
  with split `aljazeera2026Q1`.
- Your own model/API credentials. Futuresim does not include maintainer keys.

For OpenReward runs, provide:

- `OPENREWARD_API_KEY` for sessions, sandbox creation, and hosted search.
- Firehorse on the driver machine.
- Local Codex or Claude Code auth for the `codex` or `claude-code` Firehorse
  agents.
- `OPENROUTER_API_KEY` when using OpenRouter for answer matching or API-direct
  model runs.

For Verifiers or local MinimalHarness-style runs, provide an article corpus
directory if you want filesystem news access. It must use:

```text
articles_base/
  YYYY/
    MM/
      DD/
        articles.jsonl
```

- A sandbox image with the selected CLI installed, for example `codex` or
  `claude`, if reproducing a CLI-agent run. The hosted runner can bootstrap the
  Futuresim harness code into the sandbox, but baking it into the image is
  faster and more stable for long runs.

For hybrid LanceDB search, also provide:

- `search_db`: path to the prebuilt
  [LanceDB index](https://huggingface.co/datasets/shash42/forecast-news-embeddings).
- `embedding_model`: path/name for
  [Qwen/Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B), or
  an embedding server.
- GPU or embedding-server access if your hybrid setup requires it.

Hybrid LanceDB search is opt-in because it needs extra artifacts and, depending
on your setup, GPU or embedding-server access. For faithful MinimalHarness
paper reproduction, use hybrid LanceDB search and answer matching. The
OpenReward-native path instead uses OpenReward hosted hybrid search by default.

## OpenReward / ORS

The OpenReward integration has three moving parts:

1. An OpenReward ORS environment server running `FuturesimOpenRewardEnv`.
2. A Firehorse driver process on the user's machine.
3. An agent. For `codex`/`claude-code`, Firehorse launches the local CLI and
   authenticates with the user's existing CLI credentials. For API-key agents
   such as `react`, Firehorse talks directly to the model provider and can
   attach an OpenReward harness toolset such as `claude-code`.

The model sees an OpenReward MCP server named `openreward`. That server exposes
OpenReward's harness-toolset tools, including sandbox filesystem/shell tools,
plus Futuresim task tools:

- `market.csv` in the sandbox workspace for current questions and prior
  predictions.
- `search_news(query, from_date?, to_date?)`, backed by OpenReward hosted
  search/fetch through the OpenReward SDK.
- `submit_forecasts(question_id, outcomes)` for one forecast update.
- `next_day()` to score the current simulation day and advance the environment.

The sandbox should run with network disabled. Model calls happen outside the
sandbox through the user's local CLI/provider auth, and search happens through
the environment's OpenReward SDK client, not arbitrary sandbox networking.

Agent-specific code is intentionally separate from the OpenReward environment:

- `integrations/openreward/futuresim_env.py` owns Futuresim state, sandbox file
  staging, and environment tools.
- `integrations/openreward/agent.py` owns the default agent-facing prompt
  scaffold and toolset compatibility shims.
- `integrations/openreward/firehorse_run.py` owns Firehorse launch-time model
  and local CLI wiring.

To customize prompts without editing the environment, point the task spec at an
importable prompt builder:

```json
{
  "openreward_agent": {
    "prompt_builder": "my_package.prompts:build_prompt"
  }
}
```

The function is called as `build_prompt(runtime, config, mount_articles=...)`
and must return the prompt string. The module must be importable in the ORS
environment process.

```bash
pip install --no-compile -e ".[openreward]" firehorse-cli
```

`--no-compile` is optional, but it avoids slow bytecode writes on shared
filesystems.

Required credentials depend on the run:

```bash
# Always required for OpenReward sessions and sandbox creation.
export OPENREWARD_API_KEY=...

# Required when Futuresim answer matching uses OpenRouter.
export OPENROUTER_API_KEY=...

# Required only for API-key agents/models, not for Codex ChatGPT-login runs.
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
```

For Codex ChatGPT-login runs, install and authenticate the local Codex CLI in
the normal way:

```bash
codex login
```

For Claude Code runs, install and authenticate the local Claude Code CLI in the
normal way. The integration needs those local CLIs only because that is the
supported access path for Codex/Claude Code agents.

Run a hosted OpenReward environment with Firehorse. Use
`futuresim-openreward-firehorse` instead of bare `firehorse` when Futuresim
answer matching uses OpenRouter; the wrapper passes `OPENROUTER_API_KEY` as a
domain-scoped OpenReward secret. The closest reproduction of Futuresim results
uses a CLI agent such as `codex` or `claude-code`, launched with the user's
local CLI auth.

Choose the Firehorse harness with `--agent`:

- `codex`: launches the local Codex CLI and connects it to OpenReward tools.
- `claude-code`: launches the local Claude Code CLI and connects it to
  OpenReward tools.
- `react`: calls an API model directly. Use `--toolset claude-code` to give it
  OpenReward's Claude Code-style sandbox/file tools.

Closest reproduction with Codex CLI:

```bash
futuresim-openreward-firehorse \
  --env <namespace>/futuresim \
  --agent codex \
  --model openai/gpt-5.5 \
  --effort xhigh \
  --split test \
  --max-tasks 1 \
  --output-dir runs/openreward/codex-gpt55-xhigh \
  --task-spec configs/openreward/aljazeera2026Q1_v1_day0_day1.yaml
```

The bundled OpenReward task specs include day 0 + day 1 and 7-day smoke
windows. With the default `start_date=2025-12-31` and `lookback_days=7`, the
2-day specs run `2025-12-24` and `2025-12-25`; the 7-day specs run through
`2025-12-30`. Use `configs/openreward/aljazeera2026Q1_v1_7day.yaml` or
`configs/openreward/aljazeera2026Q1_v3_7day.yaml` for the 7-day versions.

Claude Code CLI example:

```bash
futuresim-openreward-firehorse \
  --env <namespace>/futuresim \
  --agent claude-code \
  --model anthropic/claude-sonnet-4-6 \
  --effort high \
  --split test \
  --max-tasks 1 \
  --output-dir runs/openreward/claude-sonnet \
  --task-spec configs/openreward/aljazeera2026Q1_v1_day0_day1.yaml
```

OpenRouter API model example:

```bash
futuresim-openreward-firehorse \
  --env <namespace>/futuresim \
  --agent react \
  --toolset claude-code \
  --model openrouter/deepseek/deepseek-v4-pro \
  --split test \
  --max-tasks 1 \
  --output-dir runs/openreward/deepseek-v4-pro-react \
  --task-spec configs/openreward/aljazeera2026Q1_v3_day0_day1.yaml
```

The API-direct path is useful for smoke tests and model comparisons when users
do not want to install local Codex or Claude Code. It does not reproduce the
original paper runs: the model sees an OpenReward toolset that resembles a CLI
tool surface, but it is not the actual Codex or Claude Code CLI harness,
session behavior, system prompting, or tool formatting used by the original
local runs.

`--task-spec` accepts a JSON or YAML OpenReward task-spec overlay. Copy one of
the files under `configs/openreward/` to change the dataset, date window,
answer matcher, handholding version, article mounting, or sandbox settings.
When `--output-dir` is set and the task spec leaves `futuresim.output_base`
blank, the wrapper also sets `output_base` to the same directory.

### Differences From Local Futuresim Runs

The OpenReward-native path shares Futuresim's simulation core, scoring, answer
matching, `market.csv` format, and output files, but differs from original
local Futuresim runs in the following ways:

- **Tool server:** local runs expose Futuresim's `forecast` MCP server, usually
  as `mcp__forecast__...`. OpenReward exposes an `openreward` MCP server, so the
  model sees tools as `mcp__openreward__search_news`,
  `mcp__openreward__submit_forecasts`, and `mcp__openreward__next_day`, plus
  the OpenReward harness-toolset filesystem/shell tools.
- **Search backend:** original paper-style local runs used the local LanceDB
  hybrid index and, when configured, a browsable `articles/` tree. The
  OpenReward-native default uses OpenReward hosted search through the SDK. The
  `search_news` tool keeps Futuresim's combined search shape for the model, but
  under the hood it calls OpenReward `Backsearch` for up to 5 hits and
  `Backfetch` for each hit, then formats the fetched articles as search
  results. Each fetched article body is capped at 5,000 characters before
  formatting.
- **Sandbox resources:** the default OpenReward sandbox is CPU-only. Search uses
  OpenReward hosted search and answer matching uses OpenRouter by default, so
  no sandbox GPU is needed.
- **Article files:** `mount_articles` defaults to `false`; the sandbox usually
  has no grep-able `articles/` directory. Set `mount_articles: true` and provide
  `articles_base` only when the ORS server can read the dated article tree and
  you want filesystem article browsing.
- **Forecast submission timing:** local MCP buffers submissions until
  `next_day`. OpenReward records each `submit_forecasts` call immediately, but
  the sandbox `market.csv` is refreshed only after `next_day`. Normal
  model-visible file behavior is therefore the same; interrupted partial
  episodes can differ.
- **Auth and execution:** for `codex` and `claude-code`, Firehorse launches the
  user's local CLI with their local auth. API-direct `react` runs use provider
  keys and an OpenReward harness toolset instead; they are not faithful
  reproductions of local CLI-agent runs.

For the closest reproduction of Futuresim results, match the
dataset, split, date window, answer matcher, model, reasoning effort, and output
directory layout. If the historical run used LanceDB/articles, the
OpenReward-native hosted-search path should be treated as a different retrieval
condition unless you add a matching hosted hybrid-search mode.

### Task Spec

Pass task rows through the OpenReward deployment's task source. For local ORS
development, `FuturesimOpenRewardEnv` also accepts `FSIM_OPENREWARD_TASKS` as a
JSON object or list of objects.

Closest-reproduction task spec:

```json
{
  "example_id": "futuresim-openreward-smoke",
  "futuresim": {
    "dataset": "openforesight",
    "dataset_path": "nikhilchandak/OpenForesight",
    "split": "aljazeera2026Q1",
    "start_date": "2025-12-31",
    "end_date": "2025-12-24",
    "resolution_start": "2025-12-31",
    "resolution_end": "2026-03-28",
    "lookback_days": 7,
    "timegap_days": 1,
    "min_forecasters": 0,
    "resolved_only": false,
    "max_outcomes_per_question": 5,
    "output_base": "/path/writable/by/the/ors-server",
    "agent_id": "minimalHarness_gpt-55_001",
    "matching": "openrouter",
    "matcher": "deepseek/deepseek-v3.2",
    "matcher_max_concurrency": 32,
    "matcher_api_key_env": "OPENROUTER_API_KEY"
  },
  "openreward_sandbox": {
    "environment": "<namespace>/futuresim",
    "image": "generalreasoning/python-ds:3.12-tools",
    "machine_size": "2:8",
    "block_network": true,
    "mount_articles": false
  }
}
```

Paths and fields new users usually need to change:

- `dataset_path`: use a Hugging Face dataset id such as
  `nikhilchandak/OpenForesight`, or use `dataset: custom` with a custom dataset
  file as shown below.
- `split`: a split present in `dataset_path`. The public dataset includes
  `aljazeera2026Q1`.
- `output_base`: where Futuresim writes `actions.jsonl`, `daily_metrics.csv`,
  `test_daily_metrics.csv`, `matcher.jsonl`, and `market.csv`. This path is on
  the machine/container running the ORS environment server, not necessarily on
  the Firehorse driver machine.
- `matcher_cache`: optional answer-matcher cache controls. By default,
  OpenReward runs write `matcher_cache.json` under `output_base`; when
  `FSIM_SIM_MATCHER_CACHE_DIR` is set and `split: test`, they use the shared
  `<matcher_slug>.json` cache there.
- `matcher_max_concurrency`: maximum concurrent answer-matcher requests.
  OpenReward defaults to `32`; raise it only if your model provider and network
  path can handle higher fan-out reliably.
- `articles_base`: optional dated article tree. Leave it unset when using the
  OpenReward search tool without filesystem article mounting.
- `openreward_sandbox.environment`: the OpenReward environment used to create
  the agent sandbox.
- `openreward_sandbox.image` and `machine_size`: change only if the default
  CPU Python tools image is insufficient. The default OpenReward sandbox does
  not need a GPU because search uses OpenReward hosted search and answer
  matching uses OpenRouter by default.

### Loading Alternate Datasets

OpenReward task specs pass their dataset settings directly into Futuresim.
`dataset_path` is resolved by the ORS environment server, not by the Firehorse
driver and not by the agent sandbox.

Use the public OpenForesight Hugging Face dataset:

```json
{
  "futuresim": {
    "dataset": "openforesight",
    "dataset_path": "nikhilchandak/OpenForesight",
    "split": "aljazeera2026Q1"
  }
}
```

Use a custom CSV, JSONL, JSON, or Parquet dataset:

```json
{
  "futuresim": {
    "dataset": "custom",
    "dataset_path": "/path/readable/by/ors/questions.jsonl",
    "split": "test"
  }
}
```

A custom directory may contain split files such as `test.jsonl`, `test.csv`,
`test.parquet`, or `test-*.parquet`. Required columns are:

| Required field | Accepted aliases |
| --- | --- |
| `qid` | `question_id`, `id` |
| `title` | `question_title`, `question` |
| `resolution_date` | `close_time`, `resolve_time` |
| `ground_truth_answer` | `ground_truth`, `answer`, `resolution`, `resolved_to` |

Optional columns are `background`, `resolution_criteria`, `answer_type`,
`options`, `source_split`, `prompt`, and `source`. `options` should be a JSON
list for multiple-choice questions. `ground_truth_answer` is required because
Futuresim scores the run after questions resolve.

`futuresim-openreward-firehorse --output-dir` controls local Firehorse artifacts
such as `run_result.json`, `trial_*.jsonl`, and `driver.log`. The Futuresim
wrapper also sends that path as `futuresim.output_base` so Futuresim actions and
metrics land with the trajectory logs when the environment server can write the
same path. Put a non-empty `futuresim.output_base` in the task spec when the
hosted server needs a different writable path.

By default this integration does not provide a local article corpus, LanceDB
index, embedding model, or grep-able `articles/` directory in the sandbox. Do
not prompt the model to use those resources unless you explicitly enable
article mounting in the task spec.

### Local ORS Development

Local ORS is the easiest way to reproduce Futuresim's raw output folder layout,
because the environment server writes files directly to your local
`output_base`.

Create `task.json` from the task spec above, then:

```bash
export OPENREWARD_API_KEY=...
export OPENROUTER_API_KEY=...
export OPENREWARD_API_URL=https://api.openreward.ai
export NO_PROXY=127.0.0.1,localhost,::1
export no_proxy=127.0.0.1,localhost,::1
export FSIM_OPENREWARD_TASKS="$(jq -c '[.]' task.json)"

# Shell 1: start the local ORS server.
# Do not set OPENREWARD_SESSION_URL in this process; sandbox creation must
# still use the hosted OpenReward session service.
unset OPENREWARD_SESSION_URL
futuresim-openreward-server --host 127.0.0.1 --port 8080
```

In another shell:

```bash
export OPENREWARD_API_KEY=...
export OPENROUTER_API_KEY=...
export OPENREWARD_API_URL=https://api.openreward.ai
export OPENREWARD_SESSION_URL=http://127.0.0.1:8080
export NO_PROXY=127.0.0.1,localhost,::1
export no_proxy=127.0.0.1,localhost,::1

futuresim-openreward-firehorse \
  --env <namespace>/futuresim \
  --agent codex \
  --model openai/gpt-5.5 \
  --effort xhigh \
  --split test \
  --max-tasks 1 \
  --run-name futuresim-openreward-smoke \
  --output-dir /same/path/as/task-json-output-base
```

If your cluster requires HTTP proxies, make sure `NO_PROXY` includes
`127.0.0.1` and `localhost`; otherwise the Firehorse MCP bridge may try to reach
the local ORS server through the proxy.

### Hosted ORS Deployment

To deploy on OpenReward, create a standard ORS environment and link this
repository. The hosted deployment must point at a commit that includes
`integrations/openreward/*` and the SDK-backed `search_news` implementation.

```bash
export OPENREWARD_API_KEY=...

orwd create futuresim \
  --namespace <your-openreward-namespace> \
  --description "Futuresim forecasting simulation environment"

orwd link <your-openreward-namespace>/futuresim OpenForecaster/futuresim \
  --cpu-memory 2:4 \
  --concurrency 20 \
  --max-scale 2
```

## Verifiers / Prime

### Current Hosted Limitation

Strict hosted Codex/Claude MinimalHarness reproduction is currently blocked on
default hosted sandboxes. Futuresim needs an inner `bubblewrap` sandbox or
equivalent custom URL/network blocklisting, so the agent shell cannot use
arbitrary web access while still reaching its model provider.
[Prime Sandboxes](https://docs.primeintellect.ai/sandboxes/overview) document
disabling network access, but the hosted Verifiers path does not currently
expose the custom URL allowlist/blocklist surface Futuresim needs. The
OpenReward-native toolset integration avoids this by disabling sandbox network
and making model/search calls outside the sandbox.

### Build The Sandbox Image

Clone Futuresim and enter the repo:

```bash
git clone https://github.com/OpenForecaster/futuresim.git
cd futuresim
```

For Verifiers or local MinimalHarness-style runs, the sandbox image is where the
CLI agent runs. Start from:

```bash
docker build -f integrations/sandbox.Dockerfile -t futuresim-sandbox:latest .
```

If you want Codex or Claude Code reproduction, extend that image with the
corresponding CLI and make sure the platform can inject the user's private
credentials.

The default sandbox expects `bubblewrap` and `socat`. These keep the CLI agent
from reading raw LanceDB/search artifacts or using arbitrary network access.
The platform sandbox must also allow bubblewrap to create an inner filesystem
and network sandbox.

### Common Task Config

Use this shape for Verifiers or local MinimalHarness-style smoke tests. For the
closest reproduction of Futuresim results, keep the same model/reasoning effort
and add the OpenRouter matcher plus LanceDB hybrid search settings below.

```json
{
  "futuresim": {
    "dataset": "openforesight",
    "dataset_path": "nikhilchandak/OpenForesight",
    "split": "aljazeera2026Q1",
    "start_date": "2025-12-31",
    "end_date": "2026-03-28",
    "resolution_start": "2025-12-31",
    "resolution_end": "2026-03-28",
    "lookback_days": 7,
    "timegap_days": 1,
    "articles_base": "/path/to/articles",
    "matching": "exact"
  },
  "minimal_harness": {
    "harness_backend": "codex",
    "model": "gpt-5.5",
    "reasoning_effort": "xhigh",
    "codex_resume": true,
    "agent_filesystem_sandbox": true,
    "network_isolation": true
  }
}
```

This exact match, no embeddings based search config is intentionally
lightweight. It is not a faithful reproduction of the original paper
experiments.

The recommended evaluation setup uses answer matching with OpenRouter:

```json
{
  "futuresim": {
    "matching": "openrouter",
    "matcher": "deepseek/deepseek-v3.2",
    "matcher_api_key_env": "OPENROUTER_API_KEY"
  }
}
```

Set `OPENROUTER_API_KEY` as a platform secret or environment variable.

### Hybrid Search

For faithful reproduction of the original paper experiments, also enable the
LanceDB hybrid MCP search tool:

```json
{
  "futuresim": {
    "enable_hybrid_search": true,
    "hybrid_search": {
      "search_db": "/path/to/lancedb",
      "embedding_model": "/path/to/embedding-model",
      "search_type": "hybrid",
      "max_results": 5
    }
  },
  "minimal_harness": {
    "agent_filesystem_sandbox": true
  }
}
```

Keep `agent_filesystem_sandbox: true` for public or reproducibility runs. With a
supported platform sandbox, the MCP server can access LanceDB while the agent
shell cannot directly read the raw index or embedding-model files.

### Install And Load

From a cloned checkout, install with the Verifiers extra:

```bash
pip install -e ".[verifiers]"
```

If installing from a published package instead:

```bash
pip install "futuresim[verifiers]"
```

For hybrid search support:

```bash
pip install -e ".[verifiers,hybrid-search]"
```

Load the environment:

```python
from verifiers import load_environment

env = load_environment(
    "futuresim",
    env_args={
        "futuresim": {
            "articles_base": "/path/to/articles",
            "matching": "exact"
        },
        "minimal_harness": {
            "harness_backend": "codex",
            "model": "gpt-5.5",
            "reasoning_effort": "xhigh",
            "codex_resume": True,
            "agent_filesystem_sandbox": True,
            "network_isolation": True
        },
        "sandbox": {
            "docker_image": "your-registry/futuresim-sandbox:latest",
            "network_access": True,
            "timeout_minutes": 720,
            "timeout_per_command_seconds": 86400
        }
    }
)
```

This example uses the lightweight exact match and no embeddings search quick
setup. Add LLM answer matching and the LanceDB search tool when reproducing the
paper-style results.

`network_access: True` is for the outer platform sandbox so the CLI can reach
its model provider. Strict CLI-agent reproduction still requires platform
support for the inner `bubblewrap` sandbox or equivalent custom URL/network
blocklisting. As of June 24, 2026,
[Prime Sandboxes](https://docs.primeintellect.ai/sandboxes/overview) document
disabling network access, but the hosted Verifiers path does not expose the
custom URL allowlist/blocklist support Futuresim needs, so this path fails fast
for strict reproduction.

## Credentials

Supply credentials through platform secrets, environment variables, or private
credential mounts. Do not put keys in task specs or Dockerfiles.

Common credentials:

| Credential | Needed for |
| --- | --- |
| `OPENROUTER_API_KEY` | OpenRouter answer matcher or OpenRouter-backed agents |
| `OPENREWARD_API_KEY` | OpenReward sessions, sandbox creation, and hosted search |
| `OPENREWARD_API_URL` | Optional API endpoint override; keep hosted API when using local ORS |
| `OPENREWARD_SESSION_URL` | Optional session endpoint override; set only in the Firehorse driver/MCP bridge for local ORS |
| `OPENREWARD_ENVIRONMENT` | Default sandbox environment name when omitted from the task spec |
| `FSIM_OPENREWARD_TASKS` | Local ORS task-spec override, JSON object or list |
| Codex CLI login or `CODEX_HOME` | Codex CLI reproduction |
| Anthropic / Claude Code credentials | Claude Code reproduction |

Each user must provide their own CLI/provider credentials.

## Reproducing A CLI-Agent Run

For faithful Codex-style reproduction:

- Use the same model, reasoning effort, date range, split, and answer matcher.
- Enable the hybrid LanceDB search tool with the same search artifacts.
- Use the same date-gated article corpus.
- Set `codex_resume: true` if reproducing a single resumed Codex thread.
- Keep `agent_filesystem_sandbox: true`.
- Keep `network_isolation: true`.
- Inject the user's own Codex credentials into the sandbox image or runtime.

Exact matching without hybrid search is useful for checking that the integration
boots, but it is not the original paper experiment configuration.

Agents can submit zero forecasts on a day. Forecasts are only counted when the
agent calls MCP `submit_forecasts` and then `next_day`.

## Resume

To resume environment state from a previous run:

```json
{
  "futuresim": {
    "resume_dir": "/path/to/previous/output"
  },
  "minimal_harness": {
    "codex_resume": true,
    "codex_thread_id": "previous-thread-id"
  }
}
```

For Claude Code, use `claude_code_resume` and `claude_session_id`.
