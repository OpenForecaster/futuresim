# Futuresim Integrations

Futuresim can run as a hosted evaluation environment on:

- Prime Intellect Verifiers
- OpenReward / ORS

Both integrations run the Futuresim environment plus an optional
MinimalHarness-compatible CLI agent such as Codex or Claude Code. The agent sees
a date-gated filesystem workspace and talks to Futuresim through MCP.

Useful links:

- Blogpost: [openforecaster.github.io/futuresim](https://openforecaster.github.io/futuresim/)
- Paper: [arxiv.org/abs/2605.15188](https://arxiv.org/abs/2605.15188)
- Questions: [nikhilchandak/OpenForesight](https://huggingface.co/datasets/nikhilchandak/OpenForesight)
- Article corpus: [shash42/forecast-news](https://huggingface.co/datasets/shash42/forecast-news)
- LanceDB hybrid index: [shash42/forecast-news-embeddings](https://huggingface.co/datasets/shash42/forecast-news-embeddings)
- Embedding model: [Qwen/Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B)

## What You Need

For any run, provide:

- A Futuresim dataset. The default is
  [nikhilchandak/OpenForesight](https://huggingface.co/datasets/nikhilchandak/OpenForesight)
  with split `aljazeera2026Q1`.
- An article corpus directory, if you want filesystem news access. It must use:

```text
articles_base/
  YYYY/
    MM/
      DD/
        articles.jsonl
```

- Your own model/API credentials. Futuresim does not include maintainer keys.
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

Hybrid search is opt-in because it needs extra artifacts and, depending on your
setup, GPU or embedding-server access. For faithful reproduction of the original
paper experiments, use hybrid LanceDB search and answer matching. The default
exact/no-hybrid setup is only meant for quick setup checks.

## Current Hosted Limitation

Strict hosted Codex/Claude MinimalHarness reproduction is currently blocked on
default Prime/Verifiers and OpenReward sandboxes. Futuresim needs an inner
`bubblewrap` sandbox, or equivalent custom URL/network blocklisting, so the
agent shell cannot use arbitrary web access while still reaching its model
provider. Current hosted sandboxes do not yet expose that capability, so the
integration fails fast instead of running an unfaithful evaluation.

## Build The Sandbox Image

Clone Futuresim and enter the repo:

```bash
git clone https://github.com/OpenForecaster/futuresim.git
cd futuresim
```

The sandbox image is where the agent runs. Start from:

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

## Common Task Config

Use this shape in either Verifiers or OpenReward task specs for a quick smoke
test:

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

This exact match, no embeddings based search config is intentionally lightweight. It is not a faithful reproduction of the original paper experiments.

The actual recommended evaluation setup is to use answer matching with OpenRouter:

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

## Hybrid Search

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

## Verifiers / Prime

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

This example uses the lightweight exact match and no embeddings search quick setup. We recommend adding LLM answer matching and the LanceDB search tool for reproducing our results.

`network_access: True` is for the outer platform sandbox so the CLI can reach
its model provider. Strict CLI-agent reproduction still requires platform
support for the inner `bubblewrap` sandbox or equivalent custom URL/network
blocklisting; until then this path fails fast.

## OpenReward / ORS

From a cloned checkout, install with the OpenReward extra:

```bash
pip install -e ".[openreward]"
```

If installing from a published package instead:

```bash
pip install "futuresim[openreward]"
```

For hybrid search support:

```bash
pip install -e ".[openreward,hybrid-search]"
```

Build the ORS server image:

```bash
docker build -t futuresim-openreward-server:latest .
```

Run the ORS server locally:

```bash
docker run --rm -p 8080:8080 \
  -e OPENREWARD_API_KEY="$OPENREWARD_API_KEY" \
  -e OPENREWARD_ENVIRONMENT="$OPENREWARD_ENVIRONMENT" \
  futuresim-openreward-server:latest
```

OpenReward task spec example:

```json
{
  "futuresim": {
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
  },
  "openreward_sandbox": {
    "environment": "your-openreward-sandbox-environment",
    "image": "your-registry/futuresim-sandbox:latest",
    "block_network": false
  }
}
```

This example uses the lightweight exact match and no embeddings search quick setup. We recommend adding LLM answer matching and the LanceDB search tool for reproducing our results.

`block_network: false` is for the outer OpenReward sandbox so the CLI can reach
its model provider. OpenReward's current `block_network: true` blocks too much
for Codex/Claude reproduction, while `block_network: false` still needs the
inner `bubblewrap` sandbox or equivalent custom URL/network blocklisting. Until
that is supported, strict CLI-agent reproduction fails fast.

To deploy on OpenReward, create a standard ORS environment and link this GitHub
repository:

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

Do not enable Harbor mode for this integration. Futuresim ships a custom ORS
server in `server.py`; Harbor is for repositories made of Harbor task
directories where OpenReward generates the server.

## Credentials

Supply credentials through platform secrets, environment variables, or private
credential mounts. Do not put keys in task specs or Dockerfiles.

Common credentials:

| Credential | Needed for |
| --- | --- |
| `OPENROUTER_API_KEY` | OpenRouter answer matcher or OpenRouter-backed agents |
| `OPENREWARD_API_KEY` | OpenReward sandbox creation |
| `OPENREWARD_ENVIRONMENT` | OpenReward sandbox environment name |
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
