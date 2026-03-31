---
name: run-simulation
description: Use when running, resuming, restarting, dry-running, or submitting forecast-sim simulations; choosing configs or output locations; or checking how results and logs are laid out.
---

# Run Simulation

Use this skill for `scripts/test_basic_agent.py`, `mpi_scripts/run_sim/submit_sim.py`, and shared simulation configs.

## Workflow

1. Load repo env from `.env` and prefer `FSIM_*` paths. If you need to edit path-sensitive launch or config files, also use `collaboration-paths`.
2. Decide whether the task is a local run, HTCondor submit, `--resume`, or `--restart_from`.
3. Read [references/commands.md](references/commands.md) for the standard commands and env-backed locations.
4. Read [references/restarts.md](references/restarts.md) when the task touches warmup reuse, truncation, or state restoration.
5. Use the `agent-scaffolds` skill when the task involves choosing or changing scaffold behavior.
6. Validate with the smallest run that exercises the changed path: local smoke test, `--dry-run`, or one cluster run. Inspect `daily_metrics.csv` and `test_daily_metrics.csv` when split-specific metrics matter.

## Repo Conventions

- Shared configs are YAML-first. Use `--set key=value` for one-off overrides instead of editing committed configs for a single experiment.
- Scaffold selection is explicit. Model names do not auto-switch scaffold classes.
- `scripts/test_basic_agent.py` writes run outputs under `FSIM_OUTPUT_BASE`.
- `mpi_scripts/run_sim/submit_sim.py` writes cluster logs under `FSIM_SIM_LOG_BASE`.
- Sim answer matching still falls back to per-run `matcher_cache.json`, but if `FSIM_SIM_MATCHER_CACHE_DIR` is set then `split: "test"` runs automatically reuse a shared `<matcher_slug>.json` and only merge back at run end. Non-test runs can opt in with top-level YAML `matcher_cache: {enabled: true, path: null}`.
- Resume mode skips Day 0 warmup because predictions are restored from `actions.jsonl`.
- `daily_metrics.csv` is session-based: one row per wakeup date. `test_daily_metrics.csv` mirrors it for test-only metrics.

## Ask The User When

- A config change would bake in a personal `restart_from` directory or cluster-specific path.
- There are multiple plausible scaffolds and the tradeoff is about methodology rather than code mechanics.
- A large cluster run is about to replace a cheap smoke test.
