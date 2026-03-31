# Model-Specific Notes

## GPT-OSS

- Use scaffold `gptossbasic` or `gptossallq` explicitly.
- GPT-OSS scaffolds use Harmony-format interactions.
- Common launch knobs:
  - `defaults.gptoss_reasoning_effort=medium|high`
  - `agent_max_model_len=131072`

Example:

```bash
python mpi_scripts/run_sim/submit_sim.py \
  --config configs/allq_sim_oss20b.yaml \
  --runs 1 \
  --set sim_name=allq_sim_oss20b_128k_a10_50_med \
  --set defaults.warmup_max_actions=10 \
  --set defaults.max_actions=50 \
  --set defaults.gptoss_reasoning_effort=medium \
  --set agent_max_model_len=131072
```

## Qwen

### Qwen3 vs Qwen3.5 (which scaffold)

- **Qwen3:** Base scaffolds are now tool-only too. For Qwen3 on vLLM, use **`basic` / `allQ` / `allqd`** with **`vllm_enable_tools: true`** and set **`vllm_tool_call_parser: hermes`** explicitly. `qwenbasic` / `qwenallq` are still the Qwen-native compatibility wrappers, but the repo no longer keeps a separate XML/text protocol for base scaffolds.
- **Qwen3.5:** Use **`basic` / `allQ` / `allqd`** or the thin **`qwenbasic` / `qwenallq`** wrappers with **`vllm_enable_tools: true`** and **`vllm_tool_call_parser: qwen3_coder`**. See `configs/qwen3.5/`.

### Qwen3.5 tuning and loop notes

- Use scaffold `qwenbasic` or `qwenallq` explicitly for 3.5 only if you want the thin Qwen-named wrappers; the base scaffolds now share the same chat-tools loop.
- Qwen scaffolds keep the base simulation semantics and now mostly act as compatibility shims over the base tool loop.
- `qwenallq` mirrors AllQ warmup semantics with the native Qwen loop.
- For local Qwen3.5 runs in this repo, the current full-context tuning is:
  - `max_model_len=262144`
  - `agent_max_model_len=262144`
  - `defaults.temperature=0.7` as the repo's forecasting-oriented compromise between Qwen's general "thinking mode" defaults and lower-temperature precise-task settings
  - `defaults.top_p=0.95`
  - `defaults.top_k=20`
  - `defaults.max_tokens=8192` because it is a per-call output cap, not an input+output cap
- Historical hidden thinking is intentionally not replayed across turns. The Qwen loop feeds back only the assistant's final visible content and tool calls, not `reasoning_content`, which matches the Qwen3.5 model-card guidance that conversation history should exclude past thinking content.
- When token budgeting is enabled, `max_total_tokens` tracks current prompt occupancy/headroom, not cumulative token spend.
- Current token-budget tuning for Qwen full-context runs:
  - `submit_reserve_tokens=8192`
  - `force_submit_threshold_tokens=16384`
- Distinction between the two token-budget thresholds:
  - `force_submit_threshold_tokens` is the soft threshold. At or below it, budget-aware per-question loops inject a strict final-submit instruction and stop exposing non-submit actions/tools.
  - `submit_reserve_tokens` is the hard floor for the forecast loop. Once remaining context drops below it, the loop stops before another forecast-turn LLM call.
- The gap between them is intentional runway for the final submit attempt, because the forced-submit prompt, assistant response, and tool transcript can still grow the prompt.
- Reserve exhaustion does not currently suppress post-loop memory calls by itself; it only ends the action loop.

## MiroThinker

- Use scaffold `mirobasic` or `miroallq` explicitly.
- MiroThinker scaffolds keep the base simulation semantics but rely on the checkpoint's native chat template, including `<tool_call>...</tool_call>` / `<tool_response>...</tool_response>` formatting and visible `<think>` replay.
- Current native tuning from the Hugging Face card for MiroThinker-1.7-mini:
  - `max_model_len=262144`
  - `agent_max_model_len=262144`
  - `defaults.temperature=1.0`
  - `defaults.top_p=0.95`
  - `defaults.repetition_penalty=1.05`
  - `defaults.max_tokens=16384`
- The recommended upstream context-retention profile is `keep5`, which maps here to `tool_result_keep_last=5`.
- We still keep forecast-sim's one-tool-per-turn loop semantics and submit-budget guardrails for comparability with the native Qwen/GPT-OSS scaffolds.
