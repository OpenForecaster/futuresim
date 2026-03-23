# SkyRL Docs

Operational guidance moved to `.agents/skills/skyrl-training/`.

Use that skill for environment setup, OpenForesight warmup-search data prep, local and HTCondor launch, runtime wrapper notes, and SkyRL submodule maintenance.

Boundary rule for the Qwen warmup integration:
- keep model-facing prompt text and tool schemas on the native `agents/qwenAgent` path
- keep SkyRL-only adaptation in `skyrl_integration/`
- env-side, parse raw assistant strings with **`skyrl_integration/vllm_qwen3_coder_text.py`**
  (same XML shape as vLLM `--tool-call-parser qwen3_coder`), then `BasicAgent.tool_calls_to_parsed_action`
- message/tool feedback should follow **`QwenBasicAgent._append_tool_output_message`** (see warmup env)

See `.agents/skills/skyrl-training/` for launch, runtime notes, and submodule maintenance.
