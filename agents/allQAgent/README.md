# AllQ Agent

AllQ uses the BasicAgent chat-tools loop over all active questions, with an initial day-zero warmup pass. Token-budget config fields apply to this Basic/AllQ loop; MinimalHarness does not use them.

## Token Budgets

- `max_total_tokens` tracks current prompt occupancy/headroom, not cumulative token spend.
- `warmup_max_total_tokens` overrides `max_total_tokens` during warmup.
- `force_submit_threshold_tokens` is the soft landing threshold: when remaining context is at or below it, the loop switches into final-submit mode.
- `submit_reserve_tokens` is the hard floor: when remaining context drops below it, the loop stops taking more forecast actions.
- `warmup_force_submit_threshold_tokens` and `warmup_submit_reserve_tokens` override those thresholds during warmup.
- `memory_update_max_total_tokens` caps the end-of-day memory update mini-loop.
- Keep `force_submit_threshold_tokens >= submit_reserve_tokens`; the config validator enforces the same relationship for warmup overrides.
- Reserve exhaustion stops the forecast/action loop only. End-of-session memory update is still attempted unless the scaffold hit a provider context-limit error.
