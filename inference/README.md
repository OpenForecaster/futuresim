# Inference (`inference/`)

Local **vLLM** OpenAI-compatible servers (`inference/vllm.py`) and related helpers. The matcher and embedding paths also use this stack when configured for vLLM.

## vLLM subprocess and `PYTHONPATH`

Worker processes prepend the **repo root** to `PYTHONPATH` so imports resolve (`inference/vllm_api_server_wrapper.py`, etc.). That same path lets Python load the repo’s root **`sitecustomize.py`** in the worker.

## Qwen3.5 MoE + `transformers` (RoPE)

Some `transformers` builds crash while loading Qwen3.5 MoE checkpoints with:

`TypeError: unsupported operand type(s) for |: 'list' and 'set'`

in RoPE validation, when `ignore_keys_at_rope_validation` is a JSON **list** but the code uses set union (`|`). Root **`sitecustomize.py`** patches `RotaryEmbeddingConfigMixin.convert_rope_params_to_dict` to coerce non-sets to `set(...)` before the original runs (see `_patch_transformers_rope_ignore_keys_to_set`).

**Long-term:** pin or upgrade **`transformers`** once upstream fixes this, then **remove** that shim from `sitecustomize.py`.
