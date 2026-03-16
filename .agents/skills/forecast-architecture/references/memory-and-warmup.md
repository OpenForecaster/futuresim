# Memory And Warmup

## Memory

- Persistent memory is controlled by `AgentConfig.enable_memory`.
- Per-day files live at `agents/<agent_id>/memory/YYYY-MM-DD.txt`.
- At the start of each day, the agent loads the latest memory snapshot strictly before the current date.
- If `enable_memory=false`, the BasicAgent end-of-day memory update is skipped.

## AllQ

`allQ` runs a Day 0 warmup across every active question before the normal daily loop.

- warmup is per-question
- DataFrame query is disabled during warmup
- persistent memory is not used during warmup
- `warmup_max_actions` and related warmup budget settings apply
- `warmup_parallelism` controls fan-out

Day 1 onward returns to the normal daily loop with awareness that initial predictions already exist.

## AllQD

`allqd` runs the same per-question focused loop every day.

- no DataFrame query path
- no BasicAgent end-of-day memory update
- per-question budget comes from the warmup budget fields

## Resume And Restart Implications

- Warmup predictions are restored from `actions.jsonl`.
- Restart-from-day is the normal way to preserve an expensive Day 0 and rerun later days.
- Validate changes by checking `actions.jsonl`, `daily_metrics.csv`, and per-agent output logs after a short run.
