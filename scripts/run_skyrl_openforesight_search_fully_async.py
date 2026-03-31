#!/usr/bin/env python3
"""Prepare data + launch fully async SkyRL GRPO warmup search training from a YAML config."""

from __future__ import annotations

import argparse
import json
import math
import netrc
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.basicAgent.tools import build_action_tools
from pathing import expand_env_tree, load_repo_env, raise_for_unresolved_env_vars
from skyrl_integration.data import (
    prepare_openforesight_search_dataset,
    read_search_chunk_tokens,
)
from skyrl_integration.envs import OPENFORESIGHT_SEARCH_WARMUP_ENV_ID
from skyrl_integration.matcher_cache import default_matcher_cache_json, setup_core_dump_cwd

load_repo_env(REPO_ROOT)


def _release_parent_resources_before_skyrl_child() -> None:
    """
    Best-effort trim before ``execve`` into SkyRL. Parquet prep runs in a subprocess so the driver
    should not hold GPU memory from embeddings/matcher.
    """

    import gc

    gc.collect()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def _resolve_matcher_cache_settings(config: Dict[str, Any]) -> str:
    """Return matcher-cache path for env / parquet prep, or ``""`` when disabled."""
    if "matcher_cache" not in config:
        return ""
    mc = config.get("matcher_cache") or {}
    if not bool(mc.get("enabled", True)):
        return ""
    matcher = str(config.get("matcher", "")).strip()
    if not matcher:
        return ""
    raw_path = mc.get("path")
    if raw_path:
        path = Path(str(raw_path).strip()).expanduser()
    else:
        path = default_matcher_cache_json(matcher)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return str(path.resolve())


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping")
    return data


def _apply_reserved_aux_cuda_visible_devices(config: Dict[str, Any]) -> Dict[str, Any]:
    reserved_aux = os.environ.get("FSIM_RESERVED_AUX_CUDA_VISIBLE_DEVICES", "").strip()
    if not reserved_aux:
        return config
    search_cfg = config.setdefault("search", {})
    if not isinstance(search_cfg, dict):
        raise ValueError("Top-level `search` config must be a mapping")
    search_cfg["aux_cuda_visible_devices"] = reserved_aux
    print(
        "[forecast-sim] Overriding search.aux_cuda_visible_devices from "
        f"FSIM_RESERVED_AUX_CUDA_VISIBLE_DEVICES={reserved_aux}",
        flush=True,
    )
    return config


def _build_runtime_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONFAULTHANDLER", "1")
    repo_paths = [
        str(REPO_ROOT),
        str(REPO_ROOT / "third_party" / "SkyRL"),
        str(REPO_ROOT / "third_party" / "SkyRL" / "skyrl-gym"),
    ]
    existing = [part for part in env.get("PYTHONPATH", "").split(os.pathsep) if part]
    merged: list[str] = []
    for part in [*repo_paths, *existing]:
        if part and part not in merged:
            merged.append(part)
    env["PYTHONPATH"] = os.pathsep.join(merged)
    return env


def _hydra_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return json.dumps(str(value))


def _require(mapping: Dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing required config key: {key}")
    return mapping[key]


def _with_run_name(path: str, run_name: str) -> str:
    if "{run_name}" in path:
        return path.format(run_name=run_name)
    return path


def _prep_paths_json_out() -> Path:
    """Temp file for subprocess parquet prep to write train/val paths (local scratch preferred)."""
    for key in ("_CONDOR_SCRATCH_DIR", "CONDOR_SCRATCH_DIR", "TMPDIR"):
        base = os.environ.get(key, "").strip()
        if base and os.path.isdir(base):
            return Path(base) / f"fsim_skyrl_prep_paths_{os.getpid()}.json"
    return Path("/tmp") / f"fsim_skyrl_prep_paths_{os.getpid()}.json"


def _local_scratch_roots() -> list[Path]:
    roots: list[Path] = []
    for key in ("FSIM_SKYRL_STAGE_SCRATCH", "_CONDOR_SCRATCH_DIR", "CONDOR_SCRATCH_DIR", "TMPDIR"):
        base = os.environ.get(key, "").strip()
        if not base:
            continue
        path = Path(base).expanduser()
        if path.is_dir():
            roots.append(path.resolve())
    return roots


def _stage_training_model_to_local_scratch(config: Dict[str, Any]) -> str | None:
    """
    Copy the shared Qwen checkpoint into node-local scratch once per job and override
    ``training.model_path`` to the local copy.

    Async 1-4-3 launches bring up 7 model consumers (4 vLLM engines + 3 FSDP workers).
    Loading all of them from ``/fast`` in parallel is currently the dominant startup bottleneck.
    """

    raw_toggle = os.environ.get("FSIM_STAGE_SKYRL_MODEL_TO_LOCAL", "1").strip().lower()
    if raw_toggle in {"0", "false", "no"}:
        return None

    training_cfg = config.get("training", {}) or {}
    raw_model_path = str(training_cfg.get("model_path", "")).strip()
    if not raw_model_path:
        return None

    source = Path(raw_model_path).expanduser().resolve()
    if not source.exists():
        return None

    scratch_roots = _local_scratch_roots()
    if not scratch_roots:
        return None

    stage_root = scratch_roots[0] / "fsim_staged_models"
    stage_root.mkdir(parents=True, exist_ok=True)
    dest = stage_root / source.name

    if not dest.exists():
        print(f"[forecast-sim] Staging training model to local scratch: {source} -> {dest}", flush=True)
        if source.is_dir():
            subprocess.run(["cp", "-a", str(source), str(dest)], check=True)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
    else:
        print(f"[forecast-sim] Reusing staged training model from local scratch: {dest}", flush=True)

    training_cfg["model_path"] = str(dest)
    return str(dest)


def _per_run_prepared_root(training: Dict[str, Any], run_name: str) -> Path:
    """Directory for this run's ``train.parquet`` / ``validation.parquet`` (removed after the job).

    Prefer **local scratch** (HTCondor ``_CONDOR_SCRATCH_DIR`` / ``TMPDIR``) so Ray/HF do not read
    short-lived parquets from Lustre under ``.../infra/...`` (matches older evals that used a
    stable ``/fast/...`` cache; scratch avoids cross-stack locking/path issues on some nodes).
    """
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(run_name).strip()).strip("_")[:120] or "run"
    for key in ("FSIM_SKYRL_PREP_SCRATCH", "_CONDOR_SCRATCH_DIR", "CONDOR_SCRATCH_DIR", "TMPDIR"):
        base = os.environ.get(key, "").strip()
        if not base:
            continue
        try:
            root = Path(base).resolve() / f"skyrl_prepared_{slug}"
            root.mkdir(parents=True, exist_ok=True)
            pq = (root / "prepared_parquet").resolve()
            pq.mkdir(parents=True, exist_ok=True)
            print(f"[forecast-sim] Using ephemeral prepared_parquet on {key}={base} -> {pq}", flush=True)
            return pq
        except OSError:
            continue

    raw = str(training.get("log_path", "/tmp/skyrl-logs")).strip() or "/tmp/skyrl-logs"
    expanded = Path(_with_run_name(raw, run_name))
    if "{run_name}" not in raw:
        expanded = expanded / run_name
    root = (expanded / "prepared_parquet").resolve()
    root.mkdir(parents=True, exist_ok=True)
    print(f"[forecast-sim] Using ephemeral prepared_parquet under log_path tree -> {root}", flush=True)
    return root


def _normalize_mode(config: Dict[str, Any]) -> str:
    mode = str(config.get("mode", "train")).strip().lower() or "train"
    if mode not in {"train", "eval"}:
        raise ValueError(f"Unsupported SkyRL mode: {mode}")
    return mode


def _effective_max_turns(raw_value: Any) -> int:
    if raw_value is None:
        return 0
    value = int(raw_value)
    return 0 if value <= 0 else value


def _ray_visible_gpu_budget(config: Dict[str, Any]) -> int:
    resources = config.get("resources", {}) or {}
    configured_gpus = int(resources.get("gpus", 4))
    env_budget = str(os.environ.get("FSIM_RAY_NUM_GPUS", "")).strip()
    if env_budget:
        try:
            parsed = int(env_budget)
        except ValueError:
            parsed = configured_gpus
        else:
            if parsed > 0:
                return parsed
    return configured_gpus


def _resolve_training_paths(config: Dict[str, Any]) -> tuple[str, str, str]:
    training = config.get("training", {}) or {}
    run_name = str(_require(training, "run_name"))
    ckpt_path = _with_run_name(str(_require(training, "ckpt_path")), run_name)
    export_path = _with_run_name(str(_require(training, "export_path")), run_name)
    return run_name, ckpt_path, export_path


def _build_overrides(config: Dict[str, Any], train_path: str, val_path: str) -> List[str]:
    resources = config.get("resources", {}) or {}
    training = config.get("training", {}) or {}
    placement_cfg = training.get("placement", {}) or {}
    fully_async_cfg = training.get("fully_async", {}) or {}
    search_cfg = config.get("search", {}) or {}
    agent_cfg = config.get("agent", {}) or {}
    mode = _normalize_mode(config)
    run_name, ckpt_path, export_path = _resolve_training_paths(config)

    gpus = int(resources.get("gpus", 4))
    ray_visible_gpus = _ray_visible_gpu_budget(config)
    num_engines = int(training.get("inference_num_engines", gpus))
    inference_tensor_parallel_size = int(training.get("inference_tensor_parallel_size", 1))
    colocate_all = bool(placement_cfg.get("colocate_all", False))
    colocate_policy_ref = bool(placement_cfg.get("colocate_policy_ref", True))
    policy_gpus = int(placement_cfg.get("policy_num_gpus_per_node", max(1, gpus - num_engines)))
    ref_gpus = int(placement_cfg.get("ref_num_gpus_per_node", policy_gpus))
    if not fully_async_cfg:
        raise ValueError("Fully async launcher requires `training.fully_async` in the YAML config.")
    if colocate_all:
        raise ValueError("Fully async launcher requires `training.placement.colocate_all=false`.")
    if bool(training.get("batched", False)):
        raise ValueError("Fully async launcher requires `training.batched=false`.")
    inference_gpu_budget = num_engines * inference_tensor_parallel_size
    training_gpu_budget = policy_gpus if colocate_policy_ref else (policy_gpus + ref_gpus)
    if bool(training.get("run_engines_locally", True)) and training_gpu_budget + inference_gpu_budget != ray_visible_gpus:
        raise ValueError(
            "Fully async GPU split must match the Ray-visible GPU budget: "
            f"training={training_gpu_budget} + inference={inference_gpu_budget} != total={ray_visible_gpus} "
            f"(resources.gpus={gpus})."
        )
    search_db = str(_require(search_cfg, "search_db"))
    max_outcomes_per_question = int(
        agent_cfg.get("max_outcomes_per_question", training.get("max_outcomes_per_question", 5))
    )
    max_search_results = int(
        agent_cfg.get("max_search_results", search_cfg.get("max_search_results", search_cfg.get("search_topk", 5)))
    )
    search_chunk_tokens = read_search_chunk_tokens(search_db)
    tool_schemas = build_action_tools(
        enable_query=False,
        enable_search=True,
        max_outcomes_per_question=max_outcomes_per_question,
        max_search_results=max_search_results,
        search_chunk_tokens=search_chunk_tokens,
    )
    chat_template_path = training.get("chat_template_path")
    if chat_template_path:
        template_path = str(Path(chat_template_path).expanduser())
    else:
        template_path = None

    stop_tokens = training.get("stop_tokens", ["</tool_call>"])
    eval_stop_tokens = training.get("eval_stop_tokens", stop_tokens)
    configured_train_batch_size = int(training.get("train_batch_size", 128))
    configured_policy_mini_batch_size = int(training.get("policy_mini_batch_size", 64))
    configured_critic_mini_batch_size = int(training.get("critic_mini_batch_size", 64))
    configured_micro_forward_batch_size = int(training.get("micro_forward_batch_size_per_gpu", 2))
    configured_micro_train_batch_size = int(training.get("micro_train_batch_size_per_gpu", 2))
    shared_sequence_parallel_size = int(training.get("sequence_parallel_size", 1))
    policy_sequence_parallel_size = int(training.get("policy_sequence_parallel_size", shared_sequence_parallel_size))
    critic_sequence_parallel_size = int(training.get("critic_sequence_parallel_size", shared_sequence_parallel_size))
    ref_sequence_parallel_size = int(training.get("ref_sequence_parallel_size", shared_sequence_parallel_size))
    configured_eval_batch_size = int(training.get("eval_batch_size", 128))
    n_samples_per_prompt = int(training.get("n_samples_per_prompt", 4))
    effective_max_turns = _effective_max_turns(training.get("max_turns", 10))
    eval_n_samples_per_prompt = int(training.get("eval_n_samples_per_prompt", training.get("n_samples_per_prompt", 1)))
    # Eval runs no policy update; skip ref model + KL (see trainer.build_models use_ref_model gate).
    use_kl_loss = bool(training.get("use_kl_loss", True)) if mode != "eval" else False
    epochs = 0 if mode == "eval" else int(training.get("epochs", 1))
    eval_before_train = True if mode == "eval" else bool(training.get("eval_before_train", False))
    eval_interval = max(1, int(training.get("eval_interval", 50))) if mode == "eval" else int(training.get("eval_interval", 50))
    if mode == "eval":
        train_rows = len(pd.read_parquet(train_path, columns=["prompt"]))
        val_rows = len(pd.read_parquet(val_path, columns=["prompt"]))
        if train_rows < 1:
            raise ValueError(
                f"Eval mode needs at least 1 train row in {train_path}. "
                "SkyRL still constructs a train DataLoader (drop_last=True); zero rows ⇒ zero batches "
                "and RayPPOTrainer crashes before eval runs, even with trainer.epochs=0."
            )
        if val_rows < 1:
            raise ValueError(f"Eval mode needs at least 1 validation prompt, got 0 in {val_path}")
        # No training iterations when mode=eval (we pass trainer.epochs=0). Eval runs via
        # eval_before_train only. SkyRL still builds a train loader; drop_last=True requires
        # train_batch_size <= train_rows and train_batch_size * n_samples_per_prompt >= policy DP
        # (single-node: gpus), same as validate_batch_sizes.
        min_train_batch_size = max(1, math.ceil(policy_gpus / max(1, n_samples_per_prompt)))
        if train_rows < min_train_batch_size:
            raise ValueError(
                "Eval mode: train row count must be at least "
                f"{min_train_batch_size} for {policy_gpus} policy GPUs and n_samples_per_prompt={n_samples_per_prompt} "
                f"(so train_batch_size * n_samples_per_prompt >= {policy_gpus}); got {train_rows} in {train_path}. "
                "Raise n_samples_per_prompt or add train prompts."
            )
        train_batch_size = min(configured_train_batch_size, train_rows)
        train_batch_size = max(min_train_batch_size, train_batch_size)
        train_batch_size = min(train_batch_size, train_rows)
        policy_mini_batch_size = train_batch_size
        critic_mini_batch_size = train_batch_size
        micro_forward_batch_size = 1
        micro_train_batch_size = 1
        eval_batch_size = min(configured_eval_batch_size, val_rows)
        # SkyRL always runs a *final* save_checkpoints + save_hf_model after the train loop when
        # these intervals are >0, even if epochs=0 (eval-only). That export hits FSDP Ray actors
        # after long vLLM eval and often dies (OOM / NCCL / dead worker). Eval has nothing to train
        # or checkpoint — skip both.
        ckpt_interval = 0
        hf_save_interval = 0
    else:
        train_batch_size = configured_train_batch_size
        policy_mini_batch_size = configured_policy_mini_batch_size
        critic_mini_batch_size = configured_critic_mini_batch_size
        micro_forward_batch_size = configured_micro_forward_batch_size
        micro_train_batch_size = configured_micro_train_batch_size
        eval_batch_size = configured_eval_batch_size
        ckpt_interval = int(training.get("ckpt_interval", 20))
        hf_save_interval = int(training.get("hf_save_interval", 200))

    # Rollout (train) sampling — keep eval_* in YAML sync by defaulting eval from these values.
    rollout_sampling = {
        "max_generate_length": int(training.get("max_generate_length", 768)),
        "temperature": training.get("temperature", 1.0),
        "top_p": training.get("top_p", 1.0),
        "top_k": training.get("top_k", -1),
        "logprobs": training.get("sampling_logprobs", 1),
        "repetition_penalty": training.get("repetition_penalty", 1.0),
        "min_p": training.get("min_p", 0.0),
    }

    def _eval_sampling_field(eval_key: str, rollout_key: str):
        """Use training[eval_key] when set to a non-null value; else mirror rollout sampling."""
        if eval_key in training and training.get(eval_key) is not None:
            return training.get(eval_key)
        return rollout_sampling[rollout_key]

    eval_sampling = {
        "max_generate_length": int(_eval_sampling_field("eval_max_generate_length", "max_generate_length")),
        "temperature": _eval_sampling_field("eval_temperature", "temperature"),
        "top_p": _eval_sampling_field("eval_top_p", "top_p"),
        "top_k": _eval_sampling_field("eval_top_k", "top_k"),
        "logprobs": _eval_sampling_field("eval_sampling_logprobs", "logprobs"),
        "repetition_penalty": _eval_sampling_field("eval_repetition_penalty", "repetition_penalty"),
        "min_p": _eval_sampling_field("eval_min_p", "min_p"),
    }
    extra_overrides = config.get("hydra_overrides", []) or []
    explicit_engine_max_model_len = any(
        str(item).lstrip("+").startswith("generator.inference_engine.engine_init_kwargs.max_model_len=")
        for item in extra_overrides
    )
    inferred_engine_max_model_len = training.get("inference_max_model_len")
    if inferred_engine_max_model_len is None:
        inferred_engine_max_model_len = int(training.get("max_input_length", 8192)) + max(
            int(rollout_sampling["max_generate_length"]),
            int(eval_sampling["max_generate_length"]),
        )
    inferred_engine_max_model_len = int(inferred_engine_max_model_len)

    overrides: List[str] = [
        f"data.train_data={_hydra_value([train_path])}",
        f"data.val_data={_hydra_value([val_path])}",
        "trainer.algorithm.advantage_estimator=grpo",
        f"trainer.policy.model.path={_hydra_value(_require(training, 'model_path'))}",
        f"trainer.placement.colocate_all={_hydra_value(colocate_all)}",
        f"trainer.placement.colocate_policy_ref={_hydra_value(colocate_policy_ref)}",
        f"trainer.strategy={_hydra_value(training.get('strategy', 'fsdp2'))}",
        f"trainer.sequence_parallel_backend={_hydra_value(training.get('sequence_parallel_backend', 'ulysses'))}",
        "trainer.policy.fsdp_config.cpu_offload=false",
        f"trainer.ref.fsdp_config.cpu_offload={_hydra_value(training.get('ref_fsdp_cpu_offload', True))}",
        f"trainer.policy.sequence_parallel_size={policy_sequence_parallel_size}",
        f"trainer.critic.sequence_parallel_size={critic_sequence_parallel_size}",
        f"trainer.ref.sequence_parallel_size={ref_sequence_parallel_size}",
        f"trainer.placement.policy_num_gpus_per_node={policy_gpus}",
        f"trainer.placement.ref_num_gpus_per_node={ref_gpus}",
        f"generator.inference_engine.num_engines={num_engines}",
        f"generator.inference_engine.tensor_parallel_size={inference_tensor_parallel_size}",
        f"generator.inference_engine.backend={_hydra_value(training.get('inference_backend', 'vllm'))}",
        f"generator.inference_engine.run_engines_locally={_hydra_value(training.get('run_engines_locally', True))}",
        f"generator.inference_engine.weight_sync_backend={_hydra_value(training.get('weight_sync_backend', 'nccl'))}",
        f"generator.inference_engine.gpu_memory_utilization={_hydra_value(training.get('gpu_memory_utilization', 0.6))}",
        f"generator.inference_engine.async_engine={_hydra_value(training.get('async_engine', True))}",
        f"generator.inference_engine.enforce_eager={_hydra_value(training.get('enforce_eager', True))}",
        f"trainer.epochs={epochs}",
        f"trainer.update_epochs_per_batch={int(training.get('update_epochs_per_batch', 1))}",
        f"trainer.train_batch_size={train_batch_size}",
        f"trainer.policy_mini_batch_size={policy_mini_batch_size}",
        f"trainer.critic_mini_batch_size={critic_mini_batch_size}",
        f"trainer.micro_forward_batch_size_per_gpu={micro_forward_batch_size}",
        f"trainer.micro_train_batch_size_per_gpu={micro_train_batch_size}",
        f"trainer.max_prompt_length={int(training.get('max_prompt_length', 4096))}",
        f"trainer.flash_attn={_hydra_value(training.get('flash_attn', False))}",
        f"trainer.gradient_checkpointing={_hydra_value(training.get('gradient_checkpointing', True))}",
        f"trainer.gradient_checkpointing_use_reentrant={_hydra_value(training.get('gradient_checkpointing_use_reentrant', False))}",
        f"trainer.use_sample_packing={_hydra_value(training.get('use_sample_packing', False))}",
        f"generator.max_input_length={int(training.get('max_input_length', 8192))}",
        f"generator.sampling_params.max_generate_length={rollout_sampling['max_generate_length']}",
        f"generator.sampling_params.temperature={_hydra_value(rollout_sampling['temperature'])}",
        f"generator.sampling_params.top_p={_hydra_value(rollout_sampling['top_p'])}",
        f"generator.sampling_params.top_k={_hydra_value(rollout_sampling['top_k'])}",
        f"generator.sampling_params.min_p={_hydra_value(rollout_sampling['min_p'])}",
        f"generator.sampling_params.repetition_penalty={_hydra_value(rollout_sampling['repetition_penalty'])}",
        f"generator.sampling_params.logprobs={_hydra_value(rollout_sampling['logprobs'])}",
        f"generator.sampling_params.stop={_hydra_value(stop_tokens)}",
        f"trainer.eval_batch_size={eval_batch_size}",
        f"trainer.eval_before_train={_hydra_value(eval_before_train)}",
        f"trainer.eval_interval={eval_interval}",
        f"trainer.dump_eval_results={_hydra_value(training.get('dump_eval_results', True))}",
        f"generator.eval_sampling_params.max_generate_length={eval_sampling['max_generate_length']}",
        f"generator.eval_sampling_params.temperature={_hydra_value(eval_sampling['temperature'])}",
        f"generator.eval_sampling_params.top_p={_hydra_value(eval_sampling['top_p'])}",
        f"generator.eval_sampling_params.top_k={_hydra_value(eval_sampling['top_k'])}",
        f"generator.eval_sampling_params.min_p={_hydra_value(eval_sampling['min_p'])}",
        f"generator.eval_sampling_params.repetition_penalty={_hydra_value(eval_sampling['repetition_penalty'])}",
        f"generator.eval_sampling_params.logprobs={_hydra_value(eval_sampling['logprobs'])}",
        f"generator.eval_sampling_params.stop={_hydra_value(eval_stop_tokens)}",
        f"generator.eval_n_samples_per_prompt={eval_n_samples_per_prompt}",
        f"trainer.policy.optimizer_config.lr={_hydra_value(training.get('learning_rate', 1e-6))}",
        f"trainer.policy.optimizer_config.max_grad_norm={_hydra_value(training.get('max_grad_norm', 1.0))}",
        f"trainer.policy.optimizer_config.num_warmup_steps={int(training.get('num_warmup_steps', 0))}",
        f"trainer.algorithm.use_kl_loss={_hydra_value(use_kl_loss)}",
        f"trainer.algorithm.kl_loss_coef={_hydra_value(training.get('kl_loss_coef', 0.001))}",
        f"trainer.algorithm.grpo_norm_by_std={_hydra_value(training.get('grpo_norm_by_std', True))}",
        f"trainer.algorithm.zero_variance_filter={_hydra_value(training.get('zero_variance_filter', False))}",
        f"trainer.algorithm.use_entropy_loss={_hydra_value(training.get('use_entropy_loss', False))}",
        f"trainer.algorithm.off_policy_correction.tis_ratio_type={_hydra_value(training.get('tis_ratio_type', 'token'))}",
        f"trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high={_hydra_value(training.get('token_tis_ratio_clip_high', 2.0))}",
        f"trainer.fully_async.max_staleness_steps={int(fully_async_cfg.get('max_staleness_steps', 4))}",
        f"trainer.fully_async.num_parallel_generation_workers={int(fully_async_cfg.get('num_parallel_generation_workers', 192))}",
        f"generator.batched={_hydra_value(training.get('batched', False))}",
        f"generator.use_conversation_multi_turn={_hydra_value(training.get('use_conversation_multi_turn', True))}",
        f"generator.append_eos_token_after_stop_str_in_multi_turn={_hydra_value(training.get('append_eos_token_after_stop_str_in_multi_turn', True))}",
        f"generator.n_samples_per_prompt={n_samples_per_prompt}",
        f"generator.max_turns={effective_max_turns}",
    ]
    # When unset: do not override ``generator.chat_template`` (same as historical eval_r00 / eval_bs160:
    # Hydra keeps SkyRL defaults → HF tokenizer ``chat_template``).
    if template_path:
        overrides.extend(
            [
                "generator.chat_template.source=file",
                f"generator.chat_template.name_or_path={_hydra_value(template_path)}",
            ]
        )
    overrides.extend(
        [
            f"generator.chat_template_kwargs.tools={_hydra_value(tool_schemas)}",
            f"generator.chat_template_kwargs.enable_thinking={_hydra_value(training.get('enable_thinking', True))}",
            f"environment.env_class={_hydra_value(OPENFORESIGHT_SEARCH_WARMUP_ENV_ID)}",
            f"environment.skyrl_gym.max_env_workers={int(training.get('max_env_workers', 16))}",
            f"trainer.logger={_hydra_value(training.get('logger', 'console'))}",
            f"trainer.project_name={_hydra_value(training.get('project_name', 'forecast-sim-skyrl'))}",
            f"trainer.run_name={_hydra_value(run_name)}",
            f"trainer.ckpt_path={_hydra_value(ckpt_path)}",
            f"trainer.export_path={_hydra_value(export_path)}",
            f"trainer.log_path={_hydra_value(_with_run_name(str(training.get('log_path', '/tmp/skyrl-logs')), run_name))}",
            f"trainer.ckpt_interval={ckpt_interval}",
            f"trainer.hf_save_interval={hf_save_interval}",
            f"trainer.max_ckpts_to_keep={int(training.get('max_ckpts_to_keep', 5))}",
        ]
    )

    if "inference_max_num_batched_tokens" in training:
        overrides.append(
            f"generator.inference_engine.max_num_batched_tokens={int(training['inference_max_num_batched_tokens'])}"
        )
    if "inference_max_num_seqs" in training:
        overrides.append(f"generator.inference_engine.max_num_seqs={int(training['inference_max_num_seqs'])}")
    if not explicit_engine_max_model_len:
        overrides.append(f"generator.inference_engine.engine_init_kwargs.max_model_len={inferred_engine_max_model_len}")

    for item in extra_overrides:
        overrides.append(str(item))

    return overrides


def _prepare_openforesight_parquets(config: Dict[str, Any], *, run_name: str) -> Tuple[str, str, Path]:
    """Build fresh ``train.parquet`` / ``validation.parquet`` under the per-run log tree."""
    data_cfg = config.get("data", {}) or {}
    if "output_dir" in data_cfg:
        raise ValueError(
            "Obsolete key `data.output_dir` — prepared data now lives next to `training.log_path` "
            "(see `prepared_parquet/` under each run) and is rebuilt every launch."
        )
    search_cfg = config.get("search", {}) or {}
    agent_cfg = config.get("agent", {}) or {}
    training_cfg = config.get("training", {}) or {}
    matching = str(config.get("matching", "exact")).strip().lower()
    matcher = str(config.get("matcher", "")).strip()
    allow_substring_match = bool(search_cfg.get("allow_substring_match", True))
    embedding_model = str(search_cfg.get("embedding_model", "")).strip()
    embedding_gpu_mem = float(search_cfg.get("embedding_gpu_mem", 0.3))
    embedding_max_num_seqs = int(search_cfg.get("embedding_max_num_seqs", 16))
    aux_cuda_visible_devices = str(search_cfg.get("aux_cuda_visible_devices", "") or "").strip()
    search_type = str(search_cfg.get("search_type", "hybrid")).strip().lower()
    search_topk = int(agent_cfg.get("max_search_results", search_cfg.get("max_search_results", search_cfg.get("search_topk", 5))))
    search_cutoff_days = int(search_cfg.get("search_cutoff_days", 0))
    search_min_days = int(search_cfg.get("search_min_days", 0))
    search_db = str(_require(search_cfg, "search_db"))
    train_split = str(data_cfg.get("train_split", "train"))
    val_split = str(data_cfg.get("val_split", "validation"))
    lookback_days = int(data_cfg.get("lookback_days", 7))
    max_train_questions = data_cfg.get("max_train_questions")
    max_val_questions = data_cfg.get("max_val_questions")
    resolution_start = data_cfg.get("resolution_start")
    resolution_end = data_cfg.get("resolution_end")
    max_outcomes_per_question = int(
        agent_cfg.get("max_outcomes_per_question", training_cfg.get("max_outcomes_per_question", 5))
    )
    warmup_max_actions = agent_cfg.get("warmup_max_actions", None)
    warmup_max_total_tokens = agent_cfg.get("warmup_max_total_tokens", None)
    warmup_submit_reserve_tokens = int(agent_cfg.get("warmup_submit_reserve_tokens", 8192))
    warmup_force_submit_threshold_tokens = int(agent_cfg.get("warmup_force_submit_threshold_tokens", 16384))
    budget_model_path = str(training_cfg.get("model_path", "")).strip()
    matcher_cache_path = _resolve_matcher_cache_settings(config)

    prepared_root = _per_run_prepared_root(training_cfg, run_name)
    result = prepare_openforesight_search_dataset(
        dataset_path=str(_require(data_cfg, "dataset_path")),
        prepared_data_dir=str(prepared_root),
        search_db=search_db,
        embedding_model=embedding_model,
        embedding_gpu_mem=embedding_gpu_mem,
        embedding_max_num_seqs=embedding_max_num_seqs,
        aux_cuda_visible_devices=aux_cuda_visible_devices,
        train_split=train_split,
        val_split=val_split,
        lookback_days=lookback_days,
        global_sim_date=data_cfg.get("global_sim_date"),
        search_type=search_type,
        search_topk=search_topk,
        search_cutoff_days=search_cutoff_days,
        search_min_days=search_min_days,
        search_max_date=search_cfg.get("search_max_date"),
        resolution_start=resolution_start,
        resolution_end=resolution_end,
        max_train_questions=max_train_questions,
        max_val_questions=max_val_questions,
        seed=int(data_cfg.get("seed", 42)),
        max_snippet_chars=int(search_cfg.get("max_snippet_chars", 1200)),
        allow_substring_match=allow_substring_match,
        matching=matching,
        matcher=matcher,
        warmup_max_actions=warmup_max_actions,
        warmup_max_total_tokens=warmup_max_total_tokens,
        warmup_submit_reserve_tokens=warmup_submit_reserve_tokens,
        warmup_force_submit_threshold_tokens=warmup_force_submit_threshold_tokens,
        budget_model_path=budget_model_path,
        max_outcomes_per_question=max_outcomes_per_question,
        matcher_cache_path=matcher_cache_path,
    )
    print(f"Prepared train data: {result.train.rows} rows -> {result.train.path}")
    print(f"Prepared val data:   {result.validation.rows} rows -> {result.validation.path}")

    train_path = prepared_root / "train.parquet"
    val_path = prepared_root / "validation.parquet"
    return str(train_path), str(val_path), prepared_root


def _write_eval_prompt_preview(config: Dict[str, Any], val_path: str) -> Path:
    from transformers import AutoTokenizer

    training = config.get("training", {}) or {}
    search_cfg = config.get("search", {}) or {}
    agent_cfg = config.get("agent", {}) or {}
    _, _, export_path = _resolve_training_paths(config)
    preview_dir = Path(export_path) / "dumped_evals" / "global_step_0_evals"
    preview_dir.mkdir(parents=True, exist_ok=True)

    sample = pd.read_parquet(val_path, columns=["prompt"]).head(1)
    if sample.empty:
        raise ValueError(f"No validation prompts found in {val_path}")

    prompt_messages = sample.iloc[0]["prompt"]
    search_db = str(_require(search_cfg, "search_db"))
    max_outcomes_per_question = int(
        agent_cfg.get("max_outcomes_per_question", training.get("max_outcomes_per_question", 5))
    )
    max_search_results = int(
        agent_cfg.get("max_search_results", search_cfg.get("max_search_results", search_cfg.get("search_topk", 5)))
    )
    tool_schemas = build_action_tools(
        enable_query=False,
        enable_search=True,
        max_outcomes_per_question=max_outcomes_per_question,
        max_search_results=max_search_results,
        search_chunk_tokens=read_search_chunk_tokens(search_db),
    )

    template_path = REPO_ROOT / "skyrl_integration" / "templates" / "qwen3_tools_without_thinking.jinja2"
    tokenizer = AutoTokenizer.from_pretrained(str(_require(training, "model_path")), trust_remote_code=True)
    rendered = tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        tokenize=False,
        chat_template=template_path.read_text(),
        tools=tool_schemas,
    )

    preview_path = preview_dir / "initial_prompt_rendered.txt"
    preview_path.write_text(rendered)
    return preview_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to training YAML config")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    core_dump_cwd = setup_core_dump_cwd()
    config = expand_env_tree(_read_yaml(config_path))
    config = _apply_reserved_aux_cuda_visible_devices(config)
    raise_for_unresolved_env_vars(config, f"SkyRL training config {config_path}")
    mode = _normalize_mode(config)
    training = config.get("training", {}) or {}
    staged_model_path = None if args.dry_run else _stage_training_model_to_local_scratch(config)
    if staged_model_path:
        training = config.get("training", {}) or {}
    run_name = str(_require(training, "run_name"))
    if str(training.get("logger", "console")).strip().lower() == "wandb" and not os.environ.get("WANDB_API_KEY"):
        try:
            auth = netrc.netrc().authenticators("api.wandb.ai")
        except (FileNotFoundError, netrc.NetrcParseError):
            auth = None
        if auth and auth[2]:
            os.environ["WANDB_API_KEY"] = auth[2]
    if str(training.get("logger", "console")).strip().lower() == "wandb" and not os.environ.get("WANDB_API_KEY"):
        raise ValueError(
            "training.logger=wandb requires WANDB_API_KEY in the environment. "
            "Put it in the repo .env or export it before launching SkyRL."
        )

    prepared_root: Path | None = None
    try:
        if args.dry_run:
            prepared_root = _per_run_prepared_root(training, run_name)
            train_path = str(prepared_root / "train.parquet")
            val_path = str(prepared_root / "validation.parquet")
            print("dry-run: skipping parquet build; printed paths are where data would be written.")
        else:
            # Subprocess: embedding/matcher can use GPU; child exit releases CUDA before SkyRL (same-PID exec alone does not).
            paths_out = _prep_paths_json_out()
            prep_cmd = [
                sys.executable,
                "-u",
                str(REPO_ROOT / "scripts" / "skyrl_prepare_openforesight_parquets.py"),
                "--config",
                str(config_path),
                "--run-name",
                run_name,
                "--paths-out",
                str(paths_out),
            ]
            subprocess.run(
                prep_cmd,
                check=True,
                cwd=str(core_dump_cwd),
                env=os.environ.copy(),
                stderr=subprocess.STDOUT,
            )
            payload = json.loads(paths_out.read_text(encoding="utf-8"))
            train_path = str(payload["train"])
            val_path = str(payload["val"])
            prepared_root = Path(str(payload["prepared_root"]))
            try:
                paths_out.unlink(missing_ok=True)
            except OSError:
                pass

        overrides = _build_overrides(config=config, train_path=train_path, val_path=val_path)
        argv_skyrl = [
            sys.executable,
            "-u",
            "-m",
            "skyrl_integration.train.main_openforesight_search_fully_async",
            *overrides,
        ]

        print("Launching SkyRL with command:")
        print(" ".join(json.dumps(part) if " " in part else part for part in argv_skyrl))

        if args.dry_run:
            if prepared_root is not None and prepared_root.exists():
                shutil.rmtree(prepared_root, ignore_errors=True)
            return 0

        if mode == "eval":
            preview_path = _write_eval_prompt_preview(config=config, val_path=val_path)
            print(f"Wrote rendered eval prompt preview: {preview_path}")

        # Replace this process with SkyRL (`execve`) so we never `fork` a fat parquet parent into Ray.
        # `subprocess.run` / default multiprocessing fork inherit COW mappings and can break `ray.init()`.
        env = _build_runtime_env()
        _release_parent_resources_before_skyrl_child()
        os.environ.clear()
        os.environ.update(env)
        os.chdir(str(core_dump_cwd))
        try:
            os.execve(sys.executable, argv_skyrl, env)
        except OSError as exc:
            print(f"os.execve failed: {exc}", file=sys.stderr)
            raise
        return 0
    except BaseException:
        if prepared_root is not None and prepared_root.exists():
            shutil.rmtree(prepared_root, ignore_errors=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
