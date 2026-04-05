---
name: news-pipeline-search
description: Use when building or updating the news corpus, embeddings, LanceDB indices, or search-enabled forecasting runs in forecast-sim.
---

# News Pipeline And Search

Use this skill for `data/news/`, search DB maintenance, and search-enabled simulation runs.

## Workflow

1. Decide whether the task is pipeline ingestion, embedding/index maintenance, or agent-side search configuration.
2. If editing cluster wrappers or paths, also use `collaboration-paths`.
3. Read [references/news-pipeline.md](references/news-pipeline.md) for the staged build workflow.
4. Read [references/search-agent.md](references/search-agent.md) for agent configuration, search modes, and action syntax.
5. Rebuild only the stages you actually need. Do not rerun embeddings or full index builds casually.
6. Validate with a small search-enabled simulation or a targeted index build command.

## Repo Conventions

- `data/news/scripts/setup_news_pipeline.sh` prepares the patched `news-please` setup and related Python deps.
- Stage 1 builds the LanceDB table plus scalar date index.
- Stage 2 builds FTS and can also build the IVF-PQ vector index used to speed up semantic and hybrid search.
- Reusing a prebuilt LanceDB table and rebuilding only Stage 2 locally is often the simplest collaborator path.
- Keep repo wrapper defaults aligned unless the user explicitly wants to change retrieval behavior.

## Ask The User When

- A rebuild would consume substantial GPU time, storage, or cluster quota.
- A change would alter retrieval behavior for ongoing experiments.
- You need to choose between MPI-specific wrappers and a more portable direct-script path.
