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
