# Forecast-Sim Agent Guide

This repo is shared across multiple users and clusters. Optimize for collaboration first.

- Do not reward hack by making dummy files or something. When in doubt, please just ask the user, unless the user explicitly told you to keep going until something is fixed, in which case please take rational decisions and raise anything suspicious you might have done to the user later.
- When a design choice has real tradeoffs, ask the user for a preference with concise evidence from the repo instead of hedging or silently picking a path.
- Explain changes clearly and concisely so collaborators can learn the codebase quickly.

- Prefer minimal, clean, easy-to-understand implementations. We will add complexity only as we need it. 
- Avoid unnecessary helper layers, `try/except`, or `if/else` bloat when a simpler structure works.
- When classes inherit from higher up classes, sometimes it might be best to make changes to the highest level of class (when it makes sense) and just propagate it to lower ones. In other words, I always want the most modular change possible instead of code bloat.
- Do not add unnecessary function indirection, preferring to write the logic where its used directly unless its really being reused somewhere else.

- Create a new skill only when it captures an important recurring pattern or repo-specific workflow that future coding agents really need to know. If the guidance is minor, one-off, or obvious from the code, do not make a skill for it.
- When an important breaking change makes an existing skill outdated, update that skill in the same coding-agent session so the repo guidance stays current.

- Do not commit user-specific absolute paths when a repo-relative path or `FSIM_*` env var can be used.
- For files inside the repo, derive paths from `__file__` or `$BASH_SOURCE` instead of `/home/<user>/forecast-sim`.
- For external data, models, logs, caches, and cluster storage, use the shared `FSIM_*` variables from `.env`.
- On MPI, `/fast/shash42` and `/fast/nchandak` can often be shared by permissions, so do not over-abstract those paths if a simple shared path already works. Treat AISA as the main cross-cluster compatibility boundary.
- Do not commit concrete `restart_from` run directories in shared configs. Pass them with `--set restart_from=...` or keep them local.
- Preserve multi-cluster behavior when editing `configs/`, `mpi_scripts/`, `aisa_scripts/`, `syntheticQA/`, `data/news/`, or other path-sensitive code.
- Before editing path-sensitive code, use the repo skill at `.agents/skills/collaboration-paths/`.