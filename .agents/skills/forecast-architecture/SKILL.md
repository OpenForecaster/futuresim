---
name: forecast-architecture
description: Use when changing agent behavior, prompts, scoring, memory, warmup loops, or the environment-agent interaction model in forecast-sim.
---

# Forecast Architecture

Use this skill for edits in `agents/`, `environment/`, prompt construction, and simulation semantics.

## Workflow

1. Start with [references/loop-and-state.md](references/loop-and-state.md) for the current run lifecycle and agent-visible state.
2. Read [references/scoring.md](references/scoring.md) before changing scoring, metrics, or benchmark-facing prompt text.
3. Read [references/memory-and-warmup.md](references/memory-and-warmup.md) before touching `allQ`, `allqd`, memory persistence, or restart behavior.
4. Keep the core invariants stable unless the user explicitly wants a benchmark-methodology change.
5. Validate with a short simulation and inspect `actions.jsonl`, `daily_metrics.csv`, `test_daily_metrics.csv` when relevant, and agent logs.

## Core Invariants

- Predictions append to per-question history; they do not overwrite earlier forecasts.
- Agents in the same wakeup session should see the same frozen aggregate state for fairness.
- The environment owns wakeup scheduling, market snapshots, and scoring; scaffold-specific session/transcript policy lives in `agents/`.
- Persistent memory is the durable cross-run context for BasicAgent-style scaffolds, but live wakeup-session carryover is scaffold-owned rather than an environment invariant.
- `timegap_days` turns the daily loop into wakeup sessions; prompt text, memory carryover, and active-question scoring should respect that session cadence.
- `allQ` and `allqd` use per-question loops without the DataFrame query path.
- Scaffold-specific prompt formats can differ, but simulation semantics should stay comparable unless the user wants a methodological shift.

## Ask The User When

- A change alters the benchmark's fairness assumptions, coverage incentives, or scoring semantics.
- A prompt refactor would intentionally change what information agents can access.
- You are choosing between two architectures that trade off comparability versus raw performance.
