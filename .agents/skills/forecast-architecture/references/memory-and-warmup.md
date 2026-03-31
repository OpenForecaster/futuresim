# Memory And Warmup

## Memory

- Persistent memory is controlled by `AgentConfig.enable_memory`.
- Per-session files live at `agents/<agent_id>/memory/YYYY-MM-DD.txt`, keyed by wakeup date.
- At the start of each wakeup, `BasicAgent`-style scaffolds load the latest memory snapshot strictly before the current date.
- Wakeup-to-wakeup transcript/session carryover is scaffold-owned; the environment only schedules wakeups and exposes the current market / scoring context.
- If `enable_memory=false`, the BasicAgent end-of-session memory update is skipped.
- Token-budget reserve exhaustion is separate from memory-update behavior. `submit_reserve_tokens` stops the forecast/action loop, but end-of-session memory update is still attempted afterward in the Basic and GPT-OSS paths.
- In the Qwen paths, memory update is skipped only when the provider already returned a real context-limit error; going below `submit_reserve_tokens` by itself does not skip memory update.

## AllQ

`allQ` runs a Day 0 warmup across every active question before the normal session loop.

- warmup is per-question
- DataFrame query is disabled during warmup
- persistent memory is not used during warmup
- `warmup_max_actions` and related warmup budget settings apply
- `warmup_parallelism` controls fan-out
- warmup prompts stay focused on the current question; cadence/wakeup reminders belong in the normal session prompt, not the warmup prompt
- The per-question warmup memo call is also post-loop work. Reserve exhaustion ends the per-question action loop, but does not by itself block the follow-up memo request.

After warmup, the standard `act()` loop on that same start date is skipped for `allQ`, so the next normal session is the next wakeup date.

## AllQD

`allqd` runs the same per-question focused loop every wakeup.

- no DataFrame query path
- no BasicAgent end-of-session memory update
- per-question budget comes from the warmup budget fields

## Resume And Restart Implications

- Warmup predictions are restored from `actions.jsonl`.
- Restart-from-day is the normal way to preserve an expensive Day 0 and rerun later days.
- Validate changes by checking `actions.jsonl`, `daily_metrics.csv`, `test_daily_metrics.csv` when relevant, and per-agent output logs after a short run.
