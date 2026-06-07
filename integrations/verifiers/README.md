# Futuresim Verifiers Environment

Futuresim is a Prime Intellect Verifiers environment for evaluating forecasting
agents in a date-gated news simulation. The environment advances a simulated
forecasting market, exposes only information available at each date, records
forecasts, and scores them after resolution.

Useful links:

- Hub install: `prime env install shash42/futuresim`
- Blogpost: [openforecaster.github.io/futuresim](https://openforecaster.github.io/futuresim/)
- Paper: [arxiv.org/abs/2605.15188](https://arxiv.org/abs/2605.15188)
- Questions: [nikhilchandak/OpenForesight](https://huggingface.co/datasets/nikhilchandak/OpenForesight)
- Article corpus: [shash42/forecast-news](https://huggingface.co/datasets/shash42/forecast-news)
- LanceDB hybrid index: [shash42/forecast-news-embeddings](https://huggingface.co/datasets/shash42/forecast-news-embeddings)
- Embedding model: [Qwen/Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B)

## What It Provides

- A Verifiers `load_environment("futuresim")` entrypoint.
- A date-gated filesystem article workspace for CLI agents.
- Optional MinimalHarness-compatible Codex and Claude Code runner support.
- Optional LanceDB hybrid MCP search, with raw search artifacts kept outside the
  agent shell when `agent_filesystem_sandbox` is enabled.
- Exact matching for quick smoke tests, and OpenRouter/vLLM answer matching for
  faithful reproduction.

The OpenReward/ORS integration is also available in the GitHub repository, but
this package page focuses on the Verifiers environment.

## Current Hosted Limitation

Strict hosted Codex/Claude MinimalHarness reproduction is currently blocked on
default Prime/Verifiers sandboxes. Futuresim needs an inner `bubblewrap`
sandbox, or equivalent custom URL/network blocklisting, so the agent shell
cannot use arbitrary web access while still reaching its model provider.
Current hosted sandboxes do not yet expose that capability, so the integration
fails fast instead of running an unfaithful evaluation.

## Quick Start

Install from Prime:

```bash
prime env install shash42/futuresim
```

Load locally:

```python
from verifiers import load_environment

env = load_environment(
    "futuresim",
    env_args={
        "futuresim": {
            "articles_base": "/path/to/articles",
            "split": "aljazeera2026Q1",
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

This exact/no-hybrid configuration is only a smoke test. To reproduce the paper
experiments, use answer matching and the LanceDB hybrid search tool.

## Required Artifacts

Download the filesystem article corpus:

```bash
hf download shash42/forecast-news \
  --repo-type dataset \
  --local-dir /path/to/articles \
  --include '2025/12/**' \
  --include '2026/**'
```

The article tree must be shaped as:

```text
articles_base/
  YYYY/
    MM/
      DD/
        articles.jsonl
```

For hybrid search, also provide:

```bash
hf download shash42/forecast-news-embeddings \
  --repo-type dataset \
  --local-dir /path/to/lancedb

hf download Qwen/Qwen3-Embedding-8B \
  --local-dir /path/to/embedding-model
```

Hybrid search can also use an already-running OpenAI-compatible embedding
server via `embedding_server_url`.

## Faithful Reproduction

Recommended Futuresim settings:

```json
{
  "futuresim": {
    "dataset_path": "nikhilchandak/OpenForesight",
    "split": "aljazeera2026Q1",
    "start_date": "2025-12-31",
    "end_date": "2026-03-28",
    "resolution_start": "2025-12-31",
    "resolution_end": "2026-03-28",
    "articles_base": "/path/to/articles",
    "matching": "openrouter",
    "matcher": "deepseek/deepseek-v3.2",
    "matcher_api_key_env": "OPENROUTER_API_KEY",
    "enable_hybrid_search": true,
    "hybrid_search": {
      "search_db": "/path/to/lancedb",
      "embedding_model": "/path/to/embedding-model",
      "search_type": "hybrid",
      "max_results": 5
    }
  }
}
```

Set `OPENROUTER_API_KEY` as a Prime secret or environment variable. CLI-agent
reproductions also require each user to provide their own Codex or Claude Code
credentials; Futuresim does not ship maintainer keys.

## Sandbox Notes

Use `agent_filesystem_sandbox: true` and `network_isolation: true` for public
or reproducibility runs. The outer Prime sandbox may need `network_access: true`
so the CLI can reach its model provider. With a supported platform sandbox,
Futuresim's inner runner blocks general agent egress and routes allowed provider
traffic through its proxy.

Strict MinimalHarness CLI reproduction requires a sandbox backend that can run
`bubblewrap` inside the sandbox, or equivalent custom URL/network blocklisting.
Default hosted sandboxes currently fail that preflight, so this path stops with
a clear error rather than running without the intended filesystem/network
boundary.

The agent may submit zero forecasts on a day. Forecasts count only after the
agent calls MCP `submit_forecasts` and then `next_day`.
