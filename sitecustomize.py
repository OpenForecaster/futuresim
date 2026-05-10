"""Runtime compatibility shims loaded automatically by Python.

These patches cover small Qwen/Transformers issues that affect the core
forecast-sim inference stack. Keep this file narrow: release branches should not
carry training-framework-specific shims here.
"""

from __future__ import annotations


def _extract_input_ids(value):
    if isinstance(value, dict):
        return value.get("input_ids", value)
    get_fn = getattr(value, "get", None)
    if callable(get_fn):
        try:
            maybe_ids = get_fn("input_ids")
        except Exception:
            maybe_ids = None
        if maybe_ids is not None:
            return maybe_ids
    return value


def _patch_qwen_chat_template_tokenize_output() -> None:
    """Normalize Qwen chat-template tokenize output to token-id lists."""

    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    except Exception:
        return

    if getattr(PreTrainedTokenizerBase, "_forecast_sim_chat_template_patch", False):
        return

    original_apply_chat_template = PreTrainedTokenizerBase.apply_chat_template

    def _patched_apply_chat_template(self, *args, **kwargs):
        output = original_apply_chat_template(self, *args, **kwargs)
        tokenize = kwargs.get("tokenize", False)
        return_dict = kwargs.get("return_dict", False)
        return_tensors = kwargs.get("return_tensors", None)

        if not tokenize or return_dict or return_tensors is not None:
            return output

        return _extract_input_ids(output)

    PreTrainedTokenizerBase.apply_chat_template = _patched_apply_chat_template
    PreTrainedTokenizerBase._forecast_sim_chat_template_patch = True


_patch_qwen_chat_template_tokenize_output()


def _patch_transformers_rope_ignore_keys_to_set() -> None:
    """
    Qwen3.5 MoE HF configs may pass `ignore_keys_at_rope_validation` as a list.

    Some Transformers versions then use set union (`|`) on that value and crash.
    Coerce non-set iterables to `set(...)` before the original method runs.
    """

    try:
        from transformers.modeling_rope_utils import RotaryEmbeddingConfigMixin
    except Exception:
        return

    if getattr(RotaryEmbeddingConfigMixin, "_forecast_sim_rope_ignore_keys_patch", False):
        return

    original_convert = RotaryEmbeddingConfigMixin.convert_rope_params_to_dict

    def _patched(self, ignore_keys_at_rope_validation=None, **kwargs):
        if ignore_keys_at_rope_validation is not None and not isinstance(ignore_keys_at_rope_validation, set):
            ignore_keys_at_rope_validation = set(ignore_keys_at_rope_validation)
        return original_convert(self, ignore_keys_at_rope_validation=ignore_keys_at_rope_validation, **kwargs)

    RotaryEmbeddingConfigMixin.convert_rope_params_to_dict = _patched
    RotaryEmbeddingConfigMixin._forecast_sim_rope_ignore_keys_patch = True


_patch_transformers_rope_ignore_keys_to_set()
