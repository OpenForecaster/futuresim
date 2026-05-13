---
name: agent-scaffolds
description: Use when choosing, wiring, or modifying forecast-sim agent scaffolds, including Basic, AllQ, AllQD, Qwen wrappers, and MinimalHarness.
---

# Agent Scaffolds

Use this skill for scaffold selection, scaffold routing in `scripts/test_basic_agent.py`, and model-specific prompt or tool-loop variants.

## Workflow

1. Start with [references/scaffold-map.md](references/scaffold-map.md) for the explicit scaffold-to-class mapping.
2. Read [references/model-specific-notes.md](references/model-specific-notes.md) for GPT-OSS and Qwen-specific behavior and launch knobs.
3. Keep scaffold routing explicit. Do not infer scaffold class from the model name.
4. If the change affects prompt semantics, scoring text, or loop behavior, also use `forecast-architecture`.
5. Validate with a small run that clearly exercises the chosen scaffold.

## Repo Conventions

- Shared config fields should stay reusable across scaffolds when possible.
- **Base scaffolds (`basic` / `allQ` / `allqd`) now require chat tools.** For Qwen3 on vLLM, enable tools and use `hermes`; for Qwen3.5, enable tools and use `qwen3_coder`. `qwenbasic` / `qwenallq` remain thin Qwen-named wrappers. Details: [references/model-specific-notes.md](references/model-specific-notes.md).
- Qwen-named scaffolds are thin compatibility wrappers over the shared chat-tools loop and should preserve comparable simulation semantics.
- Warmup and memory knobs are scaffold-level controls, not model-name heuristics.
- MinimalHarness `prompt_mode: "no_memory"` configs should keep `handholding_version: "v1"` unless the user explicitly asks otherwise. For Claude Code no-memory runs, allow Bash/Read for read-only inspection, but keep native Write/Edit-style tools disallowed so durable state changes go through forecast MCP submissions.

## Ask The User When

- A scaffold choice affects benchmark comparability.
- You are introducing a new scaffold rather than extending an existing one.
- A model-specific optimization would make comparisons with the base scaffold ambiguous.
