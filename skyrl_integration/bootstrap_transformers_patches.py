"""Transformers hooks used before vLLM loads Qwen3.5 (driver, Ray workers, vLLM subprocesses).

Ray/vLLM engine processes often do not execute repo-root ``sitecustomize.py``. SkyRL's
``vllm_engine`` imports this module before ``import vllm`` so the same patches apply everywhere.
"""

from __future__ import annotations


def apply_transformers_runtime_patches() -> None:
    """Idempotent: safe to call from sitecustomize and from SkyRL vLLM engine."""
    try:
        from transformers.modeling_rope_utils import RotaryEmbeddingConfigMixin
    except Exception:
        return

    if getattr(RotaryEmbeddingConfigMixin, "_forecast_sim_rope_ignore_keys_patch", False):
        return

    _original = RotaryEmbeddingConfigMixin.convert_rope_params_to_dict

    def _patched_convert_rope_params_to_dict(self, ignore_keys_at_rope_validation=None, **kwargs):
        if isinstance(ignore_keys_at_rope_validation, (list, tuple)):
            ignore_keys_at_rope_validation = set(ignore_keys_at_rope_validation)
        return _original(self, ignore_keys_at_rope_validation=ignore_keys_at_rope_validation, **kwargs)

    RotaryEmbeddingConfigMixin.convert_rope_params_to_dict = _patched_convert_rope_params_to_dict
    RotaryEmbeddingConfigMixin._forecast_sim_rope_ignore_keys_patch = True
