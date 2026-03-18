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

- Use scaffold `qwenbasic` or `qwenallq` explicitly.
- Qwen scaffolds keep the base simulation semantics but use Qwen-native tool-calling and prompt formatting.
- `qwenallq` mirrors AllQ warmup semantics with the native Qwen loop.
- For local Qwen3.5 runs in this repo, the current full-context tuning is:
  - `max_model_len=262144`
  - `agent_max_model_len=262144`
  - `defaults.max_tokens=4096` because it is a per-call output cap, not an input+output cap
- When token budgeting is enabled, `max_total_tokens` tracks current prompt occupancy/headroom, not cumulative token spend.
- Current token-budget tuning for Qwen full-context runs:
  - `submit_reserve_tokens=8192`
  - `force_submit_threshold_tokens=16384`
