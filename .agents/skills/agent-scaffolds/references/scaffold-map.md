# Scaffold Map

`scripts/test_basic_agent.py` routes scaffold names explicitly.

| Scaffold | Class | Behavior |
| --- | --- | --- |
| `basic` | `BasicAgent` | standard daily loop with DataFrame query path and optional memory |
| `allQ` / `allq` | `AllQAgent` | Day 0 all-question warmup, then normal daily loop |
| `allqd` | `AllQDailyAgent` | per-question focused loop every day, no DataFrame query, no end-of-day memory update |
| `og` | `OgAgent` | AllQ-style warmup with the upstream prompt variant |
| `qwenbasic` | `QwenBasicAgent` | basic semantics with vLLM native tools — **use for Qwen3.5**, not Qwen3 (see `model-specific-notes.md`) |
| `qwenallq` | `QwenAllQAgent` | AllQ semantics with native Qwen tool loop — **use for Qwen3.5**, not Qwen3 |
| `mirobasic` | `MiroBasicAgent` | basic semantics with native MiroThinker tool prompting |
| `miroallq` | `MiroAllQAgent` | AllQ semantics with native MiroThinker warmup loop |
| `gptossbasic` | `GPTOSSBasicAgent` | basic semantics with Harmony-format interactions |
| `gptossallq` | `GPTOSSAllQAgent` | AllQ semantics with Harmony-format warmup loop |

## Shared Knobs That Often Matter

- `warmup_max_actions`
- `warmup_max_total_tokens`
- `warmup_parallelism`
- `max_actions`
- `max_total_tokens`
- `submit_reserve_tokens`
- `force_submit_threshold_tokens`
- `max_tokens`
- `enable_memory`

Model choice alone does not change scaffold behavior. That separation is intentional.
