---
name: collaboration-paths
description: Use when editing shared configs, cluster launchers, path defaults, or restart workflows in this repo. Preserve multi-user and multi-cluster compatibility, avoid committing personal absolute paths, and prefer concise env-driven or repo-relative fixes.
---

# Collaboration Paths

Use this skill for changes in `configs/`, `mpi_scripts/`, `aisa_scripts/`, `syntheticQA/`, `data/news/`, and other path-sensitive code.

## Workflow

1. Check the touched area for hardcoded paths with `rg '/home/|/fast/|/lustre/|/is/cluster/fast/|/mnt/nfs/'`.
2. For repo files, derive paths from `__file__` or `$BASH_SOURCE`.
3. For external storage, prefer the shared `FSIM_*` env vars from `.env` and the tracked `.env.*.example` files.
4. Keep shared configs generic. Do not commit personal `restart_from` values or one-off run directories.
5. If adding a new env var, keep it generic and reusable. Reuse an existing `FSIM_*` variable when possible.
6. Keep the implementation small. Prefer a tiny shared helper or direct path derivation over a new abstraction layer.

## Repo Conventions

- `pathing.py` provides repo `.env` loading and recursive `${FSIM_*}` expansion for Python configs.
- Shared config defaults should favor `${FSIM_*}` placeholders.
- Local examples live in `.env.example`, `.env.mpi.example`, and `.env.aisa.example`.
- MPI collaborators can often share `/fast/shash42` and `/fast/nchandak` directly once permissions are set. AISA is the main boundary where env-driven paths matter most.
- For restart configs, prefer `--set restart_from=...` at submit time.

## Ask The User When

- A path decision would favor one cluster over another.
- A new env var could be either generic or experiment-specific.
- A change would move outputs between repo-local storage and shared cluster storage.
