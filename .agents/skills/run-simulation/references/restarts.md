# Resume And Restart

## Resume

Use `--resume <output_dir>` to continue an interrupted run in place.

- State is rebuilt from `actions.jsonl`.
- The simulation fast-forwards to the last recorded day.
- Day 0 warmup is skipped because those predictions are already present in the restored history.

## Restart From A Specific Day

Use `--restart_from <old_dir> --restart_from_day YYYY-MM-DD` to create a new run directory that preserves everything before the restart day and replays the rest.

Current helper behavior:

- copies `actions.jsonl` entries before the restart day
- copies per-session memory snapshots before the restart day
- copies timing stats before the restart day
- copies `daily_metrics.csv` rows before the restart day
- copies `test_daily_metrics.csv` rows before the restart day
- copies `matcher_cache.json` when present

This is the normal way to preserve an expensive AllQ Day 0 warmup and rerun later days.

Keep `timegap_days` aligned with the original run when resuming or restarting. Resume state advances by that cadence, so changing it mid-run changes which wakeup date comes next.

## Shared Config Rule

Do not commit concrete `restart_from` run directories in shared configs. Pass them with `--set restart_from=...` or keep them local.

## Good Sanity Checks

- Confirm the restart day is the first day you actually want to recompute.
- Inspect the new output dir before launching the rerun.
- Spot-check that `actions.jsonl` still contains the expected warmup predictions.
