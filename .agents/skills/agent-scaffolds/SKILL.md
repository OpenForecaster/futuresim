---
name: agent-scaffolds
description: Use when choosing, wiring, or modifying forecast-sim agent scaffolds, including Basic, AllQ, AllQD, Qwen-native, and GPT-OSS Harmony variants.
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
- **`qwenbasic` / `qwenallq` are for Qwen3.5 + vLLM native tools only.** Qwen3 should use `basic` / `allQ` / `allqd` with `vllm_enable_tools: false` (and `hermes` if you enable tools). Details: [references/model-specific-notes.md](references/model-specific-notes.md).
- Native Qwen and GPT-OSS scaffolds may change tool-call formatting, but should preserve comparable simulation semantics.
- Warmup and memory knobs are scaffold-level controls, not model-name heuristics.

## Ask The User When

- A scaffold choice affects benchmark comparability.
- You are introducing a new scaffold rather than extending an existing one.
- A model-specific optimization would make comparisons with the base scaffold ambiguous.
