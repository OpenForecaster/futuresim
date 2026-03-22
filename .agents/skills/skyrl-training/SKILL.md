---
name: skyrl-training
description: Use when preparing OpenForesight SkyRL warmup-search data, launching local or HTCondor SkyRL jobs, or maintaining the repo's SkyRL submodule integration.
---

# SkyRL Training

Use this skill for `scripts/run_skyrl_openforesight_search.py`, `mpi_scripts/skyrl_search/`, `skyrl_integration/`, and `third_party/SkyRL`.

## Workflow

1. If the task touches cluster paths, caches, or wrappers, also use `collaboration-paths`.
2. Read [references/setup-and-launch.md](references/setup-and-launch.md) for environment setup, data prep, and launch commands.
3. Read [references/runtime-notes.md](references/runtime-notes.md) when touching wrapper env, compatibility shims, or job logs.
4. Read [references/submodule-maintenance.md](references/submodule-maintenance.md) before bumping or editing `third_party/SkyRL`.
5. Validate with `--dry-run`, one submitted run, or a local launch using the tracked SkyRL config.

## Repo Conventions

- SkyRL dependencies are typically installed into the **repo `.venv`** (`uv sync` at root, then `uv sync --active` under `third_party/SkyRL`); HTCondor wrappers try `.venv` first.
- SkyRL lives as a submodule in `third_party/SkyRL`.
- Repo-specific integration code lives in `skyrl_integration/`.
- `scripts/run_skyrl_openforesight_search.py` can auto-build training data if it is missing.
- HTCondor submission goes through `mpi_scripts/skyrl_search/submit_skyrl_search_train.py`.
- Keep the Qwen boundary clean: prompt text and tool schemas should come from `agents/qwenAgent`, while SkyRL-only parsing/bridging should stay in `skyrl_integration/`.
- OpenForesight warmup env (`skyrl_integration/envs/openforesight_search_warmup_env.py`) should **call** `QwenBasicAgent._append_tool_output_message` / shared submit helpers for message parity; raw assistant completions are parsed with **`skyrl_integration/vllm_qwen3_coder_text.py`** (vLLM `qwen3_coder` XML only — no JSON-in-`<tool_call>` fallback).
- Training launcher defaults: tokenizer **HF** `chat_template` when `training.chat_template_path` is unset; Hermes Jinja is **opt-in**; `training.enable_thinking` defaults **true** (override in YAML for disable-thinking evals).

## Ask The User When

- A change would repin the SkyRL submodule or alter training semantics.
- You need to choose between local debugging and cluster execution.
- A data rebuild would be large enough to deserve confirmation.
