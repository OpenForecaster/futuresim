"""Runtime compatibility shims loaded automatically by Python (repo `PYTHONPATH`).

Targets **NovaSky SkyRL v0.1.0** (`skyrl-v0.1.0`). We do not keep compatibility with older
SkyRL versions; delete or adjust these shims when bumping the submodule.

What is still needed vs upstream:

- **`init_custom_process_group` / `torch.__version__` string compare** — SkyRL still uses
  ``str(torch.__version__) >= "2.6"`` to pick ``backend_options`` vs ``pg_options``, which
  breaks for **2.10+** (lexicographic ``"2.10" < "2.6"``). We patch using
  ``inspect.signature(_new_process_group_helper)`` instead.

- **FSDP wrap policy for Qwen3.5 text checkpoints** — `get_fsdp_wrap_policy` still raises if
  `_no_split_modules` lists vision layers that are not present on the loaded text module. We
  filter to classes that actually exist and keep the FSDP2 apply path aligned.

- **Qwen3.5 HF → vLLM weight names** — Policy FSDP state dicts still expose `model.*` while
  vLLM’s Qwen3.5 stack expects `language_model.*`. We remap names in `FSDPWeightExtractor.extract_weights`
  only (v0.1.0 dropped `get_weight_metadata` on this class).

- **`apply_chat_template` tokenize output** — Some Qwen tokenizers return `BatchEncoding` when
  `tokenize=True` without `return_dict`; `skyrl_gym_generator` still calls it that way. We
  normalize to raw token-id lists on `PreTrainedTokenizerBase`.

- **`ignore_keys_at_rope_validation` list vs set** — vLLM’s `Qwen3_5TextConfig` passes a
  **list** into `PretrainedConfig` / `RotaryEmbeddingConfigMixin.convert_rope_params_to_dict`,
  but current **Transformers** unions with ``{"partial_rotary_factor"}`` using set logic
  (`|=`). We coerce list/tuple → **set** on that mixin method so vLLM engine init does not
  require editing site-packages vLLM.

Removed as redundant with v0.1.0 + the tokenizer shim:

- **InferenceEngineClient prompt_token_ids normalization** — v0.1.0 uses `return_dict=True`
  for the prompts-only path; multi-turn ids come from the same tokenizer `apply_chat_template`
  calls fixed by the patch above, so monkey-patching the client is unnecessary.
"""

from __future__ import annotations

import inspect


def _is_qwen35_config(config) -> bool:
    model_type = str(getattr(config, "model_type", "")).strip().lower()
    return model_type.startswith("qwen3_5")


def _remap_qwen35_weight_name(name: str) -> str:
    """Map HF trainer names to vLLM Qwen3.5 conditional-generation names."""
    if not isinstance(name, str):
        return name
    if name.startswith("language_model.") or name.startswith("visual."):
        return name
    if name == "model" or name.startswith("model."):
        return f"language_model.{name}"
    if name == "lm_head" or name.startswith("lm_head."):
        return f"language_model.{name}"
    return name


def _patch_transformers_rope_ignore_keys_iterable_compat() -> None:
    """Delegate to ``skyrl_integration.bootstrap_transformers_patches`` (single implementation)."""

    try:
        from skyrl_integration.bootstrap_transformers_patches import apply_transformers_runtime_patches
    except Exception:
        return

    apply_transformers_runtime_patches()


_patch_transformers_rope_ignore_keys_iterable_compat()


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


def _patch_skyrl_fsdp_wrap_policy_for_qwen35() -> None:
    """
    Make SkyRL FSDP wrap-policy robust to missing multimodal layer classes.

    Qwen3.5 text checkpoints can advertise vision block names in `_no_split_modules`
    even though those classes do not exist in the instantiated text model.
    """

    try:
        import torch.nn as nn
        from transformers.trainer_pt_utils import get_module_class_from_name
        from skyrl.backends.skyrl_train.distributed import fsdp_utils
        from skyrl.backends.skyrl_train.distributed import fsdp_strategy
    except Exception:
        return

    if getattr(fsdp_utils, "_forecast_sim_fsdp_wrap_patch", False):
        return

    original_get_fsdp_wrap_policy = fsdp_utils.get_fsdp_wrap_policy

    def _patched_get_fsdp_wrap_policy(module, config=None, is_lora=False):
        try:
            return original_get_fsdp_wrap_policy(module=module, config=config, is_lora=is_lora)
        except Exception as exc:
            if "Could not find the transformer layer class to wrap in the model." not in str(exc):
                raise

        if config is not None and hasattr(config, "get"):
            layer_names = config.get("transformer_layer_cls_to_wrap", None)
        else:
            layer_names = None

        if not layer_names:
            layer_names = getattr(module, "_no_split_modules", None)
        if not layer_names:
            return None

        if isinstance(layer_names, str):
            layer_names = [layer_names]

        valid_layer_names = [
            layer_name for layer_name in layer_names if get_module_class_from_name(module, layer_name) is not None
        ]
        if not valid_layer_names:
            return None

        if config is not None and hasattr(config, "items"):
            filtered_config = dict(config)
        else:
            filtered_config = {}
        filtered_config["transformer_layer_cls_to_wrap"] = valid_layer_names

        return original_get_fsdp_wrap_policy(module=module, config=filtered_config, is_lora=is_lora)

    def _normalize_layer_names(layer_names) -> list[str]:
        if layer_names is None:
            return []
        if isinstance(layer_names, (str, bytes)):
            raw_names = [layer_names]
        elif isinstance(layer_names, (list, tuple, set)):
            raw_names = list(layer_names)
        else:
            raw_names = [layer_names]

        normalized: list[str] = []
        for value in raw_names:
            if value is None:
                continue
            if isinstance(value, str):
                name = value
            elif hasattr(value, "__name__"):
                name = str(value.__name__)
            else:
                name = str(value)
            if name and name not in normalized:
                normalized.append(name)
        return normalized

    def _patched_apply_fsdp2(model, fsdp_kwargs, config):
        wrap_policy = getattr(config, "wrap_policy", {}) or {}
        configured_layer_names = (
            wrap_policy.get("transformer_layer_cls_to_wrap", None) if hasattr(wrap_policy, "get") else None
        )

        layer_names = _normalize_layer_names(configured_layer_names)
        if not layer_names:
            layer_names = _normalize_layer_names(getattr(model, "_no_split_modules", None))
        if not layer_names:
            layer_names = sorted(
                {module.__class__.__name__ for module in model.modules() if "DecoderLayer" in module.__class__.__name__}
            )

        modules = []
        for _, module in model.named_modules():
            if module.__class__.__name__ in layer_names or (
                isinstance(module, nn.Embedding) and not model.config.tie_word_embeddings
            ):
                modules.append(module)

        for module in modules:
            fsdp_utils.fully_shard(module, **fsdp_kwargs)
        fsdp_utils.fully_shard(model, **fsdp_kwargs)

    fsdp_utils.get_fsdp_wrap_policy = _patched_get_fsdp_wrap_policy
    fsdp_strategy.get_fsdp_wrap_policy = _patched_get_fsdp_wrap_policy
    fsdp_utils.apply_fsdp2 = _patched_apply_fsdp2
    fsdp_strategy.apply_fsdp2 = _patched_apply_fsdp2
    fsdp_utils._forecast_sim_fsdp_wrap_patch = True


_patch_skyrl_fsdp_wrap_policy_for_qwen35()


def _patch_skyrl_custom_process_group_for_new_torch() -> None:
    """
    Patch SkyRL custom process-group helper: choose ``backend_options`` vs ``pg_options``
    from the real ``_new_process_group_helper`` signature, not string torch version compare.
    """

    try:
        from torch.distributed.distributed_c10d import Backend, PrefixStore, _world, default_pg_timeout, rendezvous
        from skyrl.backends.skyrl_train.distributed import utils as dist_utils
    except Exception:
        return

    if getattr(dist_utils, "_forecast_sim_custom_pg_patch", False):
        return

    helper_params = inspect.signature(dist_utils._new_process_group_helper).parameters
    if "backend_options" in helper_params:
        options_kwarg = "backend_options"
    elif "pg_options" in helper_params:
        options_kwarg = "pg_options"
    else:
        options_kwarg = None

    def _patched_init_custom_process_group(
        backend=None,
        init_method=None,
        timeout=None,
        world_size=-1,
        rank=-1,
        store=None,
        group_name=None,
        pg_options=None,
    ):
        assert (store is None) or (init_method is None), "Cannot specify both init_method and store."

        if store is not None:
            assert world_size > 0, "world_size must be positive if using store"
            assert rank >= 0, "rank must be non-negative if using store"
        elif init_method is None:
            init_method = "env://"

        backend_obj = Backend(backend) if backend else Backend("undefined")
        timeout = timeout or default_pg_timeout

        if store is None:
            rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
            store, rank, world_size = next(rendezvous_iterator)
            store.set_timeout(timeout)
            store = PrefixStore(group_name, store)

        helper_kwargs = {
            "group_name": group_name,
            "timeout": timeout,
        }
        if options_kwarg is not None:
            helper_kwargs[options_kwarg] = pg_options

        pg, _ = dist_utils._new_process_group_helper(
            world_size,
            rank,
            [],
            backend_obj,
            store,
            **helper_kwargs,
        )
        _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
        return pg

    dist_utils.init_custom_process_group = _patched_init_custom_process_group
    dist_utils._forecast_sim_custom_pg_patch = True


_patch_skyrl_custom_process_group_for_new_torch()


def _patch_skyrl_qwen35_weight_name_mapping() -> None:
    """
    Patch FSDP weight extraction names for Qwen3.5 -> vLLM sync.

    SkyRL policy FSDP weights expose names like `model.*` while vLLM's
    Qwen3_5ForConditionalGeneration expects `language_model.*`.
    """

    try:
        from skyrl.backends.skyrl_train.workers.fsdp import fsdp_worker
    except Exception:
        return

    if getattr(fsdp_worker, "_forecast_sim_qwen35_name_patch", False):
        return

    original_extract_weights = fsdp_worker.FSDPWeightExtractor.extract_weights

    def _is_qwen35(extractor) -> bool:
        model = getattr(extractor, "model", None)
        config = getattr(model, "config", None)
        if config is None:
            wrapped = getattr(model, "_fsdp_wrapped_module", None)
            config = getattr(wrapped, "config", None)
        return _is_qwen35_config(config)

    def _patched_extract_weights(self, dtype):
        if not _is_qwen35(self):
            yield from original_extract_weights(self, dtype)
            return

        for chunk in original_extract_weights(self, dtype):
            chunk.names = [_remap_qwen35_weight_name(name) for name in chunk.names]
            yield chunk

    fsdp_worker.FSDPWeightExtractor.extract_weights = _patched_extract_weights
    fsdp_worker._forecast_sim_qwen35_name_patch = True


_patch_skyrl_qwen35_weight_name_mapping()


def _patch_qwen_chat_template_tokenize_output() -> None:
    """
    Normalize Qwen chat-template tokenize output to token-id lists.

    Some tokenizer implementations return BatchEncoding for tokenize=True even
    when return_dict is not requested; SkyRL gym generator assumes plain token-id lists.
    """

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

        normalized = _extract_input_ids(output)
        return normalized

    PreTrainedTokenizerBase.apply_chat_template = _patched_apply_chat_template
    PreTrainedTokenizerBase._forecast_sim_chat_template_patch = True


_patch_qwen_chat_template_tokenize_output()
