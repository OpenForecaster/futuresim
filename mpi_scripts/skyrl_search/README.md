# SkyRL Warmup Search Training

Operational guidance moved to `.agents/skills/skyrl-training/` (and the mirrored `.codex/skills/skyrl-training/` copy).

Use that skill for submit commands, local launch, runtime wrapper notes, and submodule maintenance.

**HTCondor / Ray:** see `references/runtime-notes.md` → **Ray + HTCondor** (short `RAY_TMPDIR` under Condor scratch, `ulimit -n`, optional `SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1` for debugging).

If `training.logger=wandb`, export `WANDB_API_KEY` in your local `.env` before submitting.
