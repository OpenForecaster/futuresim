#!/usr/bin/env python3
"""Prepare data + launch SkyRL GRPO warmup search training from a YAML config."""

from __future__ import annotations

import argparse
import json
import math
import netrc
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.qwenAgent.tools import build_action_tools
from pathing import expand_env_tree, load_repo_env, raise_for_unresolved_env_vars
from skyrl_integration.data import (
    prepare_openforesight_search_dataset,
    read_search_chunk_tokens,
)
from skyrl_integration.constants import OPENFORESIGHT_SEARCH_WARMUP_ENV_ID

load_repo_env(REPO_ROOT)


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping")
    return data


def _build_runtime_env() -> Dict[str, str]:
    env = os.environ.copy()
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


def _resolve_prepared_data_dir(data_cfg: Dict[str, Any]) -> Path:
    """
    Shared directory for SkyRL warmup `train.parquet` / `validation.parquet` (one cache, many runs).

    Per-run SkyRL logs, Ray, and WandB live under `training.log_path` / FSIM_SKYRL_LOG_BASE, not here.
    """
    if "output_dir" in data_cfg and "prepared_data_dir" not in data_cfg:
        raise ValueError(
            "SkyRL configs renamed `data.output_dir` to `data.prepared_data_dir` "
            "(shared prepared parquet cache). "
            "Use `training.log_path` / FSIM_SKYRL_LOG_BASE for per-run logs and infra."
        )
    raw = data_cfg.get("prepared_data_dir") or os.environ.get("FSIM_SKYRL_PREPARED_DATA_DIR")
    if not raw:
        raise KeyError(
            "Set `data.prepared_data_dir` in the YAML (e.g. ${FSIM_SKYRL_PREPARED_DATA_DIR}) "
            "or export FSIM_SKYRL_PREPARED_DATA_DIR to a single shared directory."
        )
    return Path(str(raw).strip())


def _with_run_name(path: str, run_name: str) -> str:
    if "{run_name}" in path:
        return path.format(run_name=run_name)
    return path


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


def _resolve_training_paths(config: Dict[str, Any]) -> tuple[str, str, str]:
    training = config.get("training", {}) or {}
    run_name = str(_require(training, "run_name"))
    ckpt_path = _with_run_name(str(_require(training, "ckpt_path")), run_name)
    export_path = _with_run_name(str(_require(training, "export_path")), run_name)
    return run_name, ckpt_path, export_path


def _build_overrides(config: Dict[str, Any], train_path: str, val_path: str) -> List[str]:
    resources = config.get("resources", {}) or {}
    training = config.get("training", {}) or {}
    search_cfg = config.get("search", {}) or {}
    agent_cfg = config.get("agent", {}) or {}
    mode = _normalize_mode(config)
    run_name, ckpt_path, export_path = _resolve_training_paths(config)

    gpus = int(resources.get("gpus", 4))
    num_engines = int(training.get("inference_num_engines", gpus))
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
        min_eval_train_batch_size = max(1, math.ceil(gpus / max(1, n_samples_per_prompt)))
        if train_rows < min_eval_train_batch_size:
            raise ValueError(
                "Eval mode needs at least "
                f"{min_eval_train_batch_size} train prompts for {gpus} GPUs and "
                f"n_samples_per_prompt={n_samples_per_prompt}, got {train_rows} in {train_path}"
            )
        if val_rows < 1:
            raise ValueError(f"Eval mode needs at least 1 validation prompt, got 0 in {val_path}")
        train_batch_size = min(configured_train_batch_size, train_rows)
        train_batch_size = max(min_eval_train_batch_size, train_batch_size)
        policy_mini_batch_size = train_batch_size
        critic_mini_batch_size = train_batch_size
        micro_forward_batch_size = 1
        micro_train_batch_size = 1
        eval_batch_size = min(configured_eval_batch_size, val_rows)
    else:
        train_batch_size = configured_train_batch_size
        policy_mini_batch_size = configured_policy_mini_batch_size
        critic_mini_batch_size = configured_critic_mini_batch_size
        micro_forward_batch_size = configured_micro_forward_batch_size
        micro_train_batch_size = configured_micro_train_batch_size
        eval_batch_size = configured_eval_batch_size

    overrides: List[str] = [
        f"data.train_data={_hydra_value([train_path])}",
        f"data.val_data={_hydra_value([val_path])}",
        "trainer.algorithm.advantage_estimator=grpo",
        f"trainer.policy.model.path={_hydra_value(_require(training, 'model_path'))}",
        "trainer.placement.colocate_all=true",
        f"trainer.strategy={_hydra_value(training.get('strategy', 'fsdp2'))}",
        "trainer.policy.fsdp_config.cpu_offload=false",
        f"trainer.ref.fsdp_config.cpu_offload={_hydra_value(training.get('ref_fsdp_cpu_offload', True))}",
        f"trainer.placement.policy_num_gpus_per_node={gpus}",
        f"trainer.placement.ref_num_gpus_per_node={gpus}",
        f"generator.inference_engine.num_engines={num_engines}",
        f"generator.inference_engine.tensor_parallel_size={int(training.get('inference_tensor_parallel_size', 1))}",
        f"generator.inference_engine.backend={_hydra_value(training.get('inference_backend', 'vllm'))}",
        f"generator.inference_engine.run_engines_locally={_hydra_value(training.get('run_engines_locally', True))}",
        f"generator.inference_engine.weight_sync_backend={_hydra_value(training.get('weight_sync_backend', 'nccl'))}",
        f"generator.inference_engine.gpu_memory_utilization={_hydra_value(training.get('gpu_memory_utilization', 0.6))}",
        f"generator.inference_engine.async_engine={_hydra_value(training.get('async_engine', True))}",
        f"trainer.epochs={epochs}",
        f"trainer.update_epochs_per_batch={int(training.get('update_epochs_per_batch', 1))}",
        f"trainer.train_batch_size={train_batch_size}",
        f"trainer.policy_mini_batch_size={policy_mini_batch_size}",
        f"trainer.critic_mini_batch_size={critic_mini_batch_size}",
        f"trainer.micro_forward_batch_size_per_gpu={micro_forward_batch_size}",
        f"trainer.micro_train_batch_size_per_gpu={micro_train_batch_size}",
        f"trainer.max_prompt_length={int(training.get('max_prompt_length', 4096))}",
        f"trainer.flash_attn={_hydra_value(training.get('flash_attn', False))}",
        f"trainer.use_sample_packing={_hydra_value(training.get('use_sample_packing', False))}",
        f"generator.max_input_length={int(training.get('max_input_length', 8192))}",
        f"generator.sampling_params.max_generate_length={int(training.get('max_generate_length', 768))}",
        f"generator.sampling_params.temperature={_hydra_value(training.get('temperature', 1.0))}",
        f"generator.sampling_params.top_p={_hydra_value(training.get('top_p', 1.0))}",
        f"generator.sampling_params.top_k={_hydra_value(training.get('top_k', -1))}",
        f"generator.sampling_params.logprobs={_hydra_value(training.get('sampling_logprobs', 1))}",
        f"generator.sampling_params.stop={_hydra_value(stop_tokens)}",
        f"trainer.eval_batch_size={eval_batch_size}",
        f"trainer.eval_before_train={_hydra_value(eval_before_train)}",
        f"trainer.eval_interval={eval_interval}",
        f"trainer.dump_eval_results={_hydra_value(training.get('dump_eval_results', True))}",
        f"generator.eval_sampling_params.temperature={_hydra_value(training.get('eval_temperature', 0.0))}",
        f"generator.eval_sampling_params.max_generate_length={int(training.get('eval_max_generate_length', training.get('max_generate_length', 768)))}",
        f"generator.eval_sampling_params.top_p={_hydra_value(training.get('eval_top_p', training.get('top_p', 1.0)))}",
        f"generator.eval_sampling_params.top_k={_hydra_value(training.get('eval_top_k', training.get('top_k', -1)))}",
        f"generator.eval_sampling_params.logprobs={_hydra_value(training.get('eval_sampling_logprobs', training.get('sampling_logprobs', 1)))}",
        f"generator.eval_sampling_params.stop={_hydra_value(eval_stop_tokens)}",
        f"generator.eval_n_samples_per_prompt={eval_n_samples_per_prompt}",
        f"trainer.policy.optimizer_config.lr={_hydra_value(training.get('learning_rate', 1e-6))}",
        f"trainer.algorithm.use_kl_loss={_hydra_value(use_kl_loss)}",
        f"trainer.algorithm.kl_loss_coef={_hydra_value(training.get('kl_loss_coef', 0.001))}",
        f"trainer.algorithm.off_policy_correction.tis_ratio_type={_hydra_value(training.get('tis_ratio_type', 'token'))}",
        f"trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high={_hydra_value(training.get('token_tis_ratio_clip_high', 2.0))}",
        f"generator.batched={_hydra_value(training.get('batched', False))}",
        f"generator.use_conversation_multi_turn={_hydra_value(training.get('use_conversation_multi_turn', True))}",
        f"generator.append_eos_token_after_stop_str_in_multi_turn={_hydra_value(training.get('append_eos_token_after_stop_str_in_multi_turn', True))}",
        f"generator.n_samples_per_prompt={n_samples_per_prompt}",
        f"generator.max_turns={effective_max_turns}",
        "generator.chat_template.source=file",
        f"generator.chat_template.name_or_path={_hydra_value(template_path)}",
        f"generator.chat_template_kwargs.tools={_hydra_value(tool_schemas)}",
        f"generator.chat_template_kwargs.enable_thinking={_hydra_value(training.get('enable_thinking', True))}",
        f"environment.env_class={_hydra_value(OPENFORESIGHT_SEARCH_WARMUP_ENV_ID)}",
        f"environment.skyrl_gym.max_env_workers={int(training.get('max_env_workers', 16))}",
        f"trainer.logger={_hydra_value(training.get('logger', 'console'))}",
        f"trainer.project_name={_hydra_value(training.get('project_name', 'forecast-sim-skyrl'))}",
        f"trainer.run_name={_hydra_value(run_name)}",
        f"trainer.ckpt_path={_hydra_value(ckpt_path)}",
        f"trainer.export_path={_hydra_value(export_path)}",
        f"trainer.log_path={_hydra_value(training.get('log_path', '/tmp/skyrl-logs'))}",
        f"trainer.ckpt_interval={int(training.get('ckpt_interval', 20))}",
        f"trainer.hf_save_interval={int(training.get('hf_save_interval', 200))}",
        f"trainer.max_ckpts_to_keep={int(training.get('max_ckpts_to_keep', 5))}",
    ]

    extra_overrides = config.get("hydra_overrides", []) or []
    for item in extra_overrides:
        overrides.append(str(item))

    return overrides


def _dataset_matches_runtime_config(
    path: Path,
    *,
    source_split: str,
    matching: str,
    matcher: str,
    allow_substring_match: bool,
    embedding_model: str,
    embedding_gpu_mem: float,
    aux_cuda_visible_devices: str,
    search_type: str,
    search_topk: int,
    search_cutoff_days: int,
    search_min_days: int,
    search_max_date: str,
    search_db: str,
    max_outcomes_per_question: int,
    warmup_max_actions: Any,
    warmup_max_total_tokens: Any,
    warmup_submit_reserve_tokens: int,
    warmup_force_submit_threshold_tokens: int,
    budget_model_path: str,
    lookback_days: int,
    global_sim_date: str,
) -> bool:
    if not path.exists():
        return False

    try:
        sample = pd.read_parquet(
            path,
            columns=[
                "matching",
                "matcher",
                "allow_substring_match",
                "embedding_model",
                "embedding_gpu_mem",
                "aux_cuda_visible_devices",
                "search_type",
                "search_topk",
                "search_cutoff_days",
                "search_min_days",
                "search_max_date",
                "search_db",
                "max_outcomes_per_question",
                "warmup_max_actions",
                "warmup_max_total_tokens",
                "warmup_submit_reserve_tokens",
                "warmup_force_submit_threshold_tokens",
                "budget_model_path",
                "lookback_days",
                "global_sim_date",
                "source_split",
            ],
        ).head(1)
    except Exception:
        return False

    if sample.empty:
        return False

    row = sample.iloc[0]
    row_matching = str(row.get("matching", "exact")).strip().lower()
    row_matcher = str(row.get("matcher", "")).strip()
    row_allow_substring = bool(row.get("allow_substring_match", True))
    row_embedding_model = str(row.get("embedding_model", "")).strip()
    row_embedding_gpu_mem = float(row.get("embedding_gpu_mem", 0.3))
    row_aux_cuda_visible_devices = str(row.get("aux_cuda_visible_devices", "")).strip()
    row_search_type = str(row.get("search_type", "hybrid")).strip().lower()
    row_search_topk = int(row.get("search_topk", 5))
    row_search_cutoff_days = int(row.get("search_cutoff_days", 0))
    row_search_min_days = int(row.get("search_min_days", 0))
    row_search_max_date = str(row.get("search_max_date", "") or "").strip()
    row_search_db = str(row.get("search_db", "")).strip()
    row_max_outcomes = int(row.get("max_outcomes_per_question", 5))
    row_warmup_max_actions = row.get("warmup_max_actions", None)
    row_warmup_max_total_tokens = row.get("warmup_max_total_tokens", None)
    row_warmup_max_actions = None if pd.isna(row_warmup_max_actions) else int(row_warmup_max_actions)
    row_warmup_max_total_tokens = None if pd.isna(row_warmup_max_total_tokens) else int(row_warmup_max_total_tokens)
    row_warmup_submit_reserve_tokens = int(row.get("warmup_submit_reserve_tokens", 8192))
    row_warmup_force_submit_threshold_tokens = int(row.get("warmup_force_submit_threshold_tokens", 16384))
    row_budget_model_path = str(row.get("budget_model_path", "")).strip()
    row_lookback_days = int(row.get("lookback_days", 7))
    row_global_sim_date = str(row.get("global_sim_date", "") or "").strip()
    row_source_split = str(row.get("source_split", "")).strip()
    return (
        row_source_split == source_split
        and row_matching == matching
        and row_matcher == matcher
        and row_allow_substring == allow_substring_match
        and row_embedding_model == embedding_model
        and abs(row_embedding_gpu_mem - float(embedding_gpu_mem)) < 1e-9
        and row_aux_cuda_visible_devices == aux_cuda_visible_devices
        and row_search_type == search_type
        and row_search_topk == search_topk
        and row_search_cutoff_days == search_cutoff_days
        and row_search_min_days == search_min_days
        and row_search_max_date == search_max_date
        and row_search_db == search_db
        and row_max_outcomes == max_outcomes_per_question
        and row_warmup_max_actions == warmup_max_actions
        and row_warmup_max_total_tokens == warmup_max_total_tokens
        and row_warmup_submit_reserve_tokens == warmup_submit_reserve_tokens
        and row_warmup_force_submit_threshold_tokens == warmup_force_submit_threshold_tokens
        and row_budget_model_path == budget_model_path
        and row_lookback_days == lookback_days
        and row_global_sim_date == global_sim_date
    )


def _prepare_data_if_needed(config: Dict[str, Any], force: bool) -> tuple[str, str]:
    data_cfg = config.get("data", {}) or {}
    search_cfg = config.get("search", {}) or {}
    agent_cfg = config.get("agent", {}) or {}
    training_cfg = config.get("training", {}) or {}
    matching = str(config.get("matching", "exact")).strip().lower()
    matcher = str(config.get("matcher", "")).strip()
    allow_substring_match = bool(search_cfg.get("allow_substring_match", True))
    embedding_model = str(search_cfg.get("embedding_model", "")).strip()
    embedding_gpu_mem = float(search_cfg.get("embedding_gpu_mem", 0.3))
    aux_cuda_visible_devices = str(search_cfg.get("aux_cuda_visible_devices", "") or "").strip()
    search_type = str(search_cfg.get("search_type", "hybrid")).strip().lower()
    search_topk = int(agent_cfg.get("max_search_results", search_cfg.get("max_search_results", search_cfg.get("search_topk", 5))))
    search_cutoff_days = int(search_cfg.get("search_cutoff_days", 0))
    search_min_days = int(search_cfg.get("search_min_days", 0))
    search_max_date = str(search_cfg.get("search_max_date") or "").strip()
    search_db = str(_require(search_cfg, "search_db"))
    train_split = str(data_cfg.get("train_split", "train"))
    val_split = str(data_cfg.get("val_split", "validation"))
    lookback_days = int(data_cfg.get("lookback_days", 7))
    global_sim_date = str(data_cfg.get("global_sim_date") or "").strip()
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

    prepared_root = _resolve_prepared_data_dir(data_cfg)
    train_path = prepared_root / "train.parquet"
    val_path = prepared_root / "validation.parquet"
    subset_requested = any(
        value is not None
        for value in (
            max_train_questions,
            max_val_questions,
            resolution_start,
            resolution_end,
        )
    )
    needs_refresh = (
        force
        or subset_requested
        or not _dataset_matches_runtime_config(
            train_path,
            source_split=train_split,
            matching=matching,
            matcher=matcher,
            allow_substring_match=allow_substring_match,
            embedding_model=embedding_model,
            embedding_gpu_mem=embedding_gpu_mem,
            aux_cuda_visible_devices=aux_cuda_visible_devices,
            search_type=search_type,
            search_topk=search_topk,
            search_cutoff_days=search_cutoff_days,
            search_min_days=search_min_days,
            search_max_date=search_max_date,
            search_db=search_db,
            max_outcomes_per_question=max_outcomes_per_question,
            warmup_max_actions=warmup_max_actions,
            warmup_max_total_tokens=warmup_max_total_tokens,
            warmup_submit_reserve_tokens=warmup_submit_reserve_tokens,
            warmup_force_submit_threshold_tokens=warmup_force_submit_threshold_tokens,
            budget_model_path=budget_model_path,
            lookback_days=lookback_days,
            global_sim_date=global_sim_date,
        )
        or not _dataset_matches_runtime_config(
            val_path,
            source_split=val_split,
            matching=matching,
            matcher=matcher,
            allow_substring_match=allow_substring_match,
            embedding_model=embedding_model,
            embedding_gpu_mem=embedding_gpu_mem,
            aux_cuda_visible_devices=aux_cuda_visible_devices,
            search_type=search_type,
            search_topk=search_topk,
            search_cutoff_days=search_cutoff_days,
            search_min_days=search_min_days,
            search_max_date=search_max_date,
            search_db=search_db,
            max_outcomes_per_question=max_outcomes_per_question,
            warmup_max_actions=warmup_max_actions,
            warmup_max_total_tokens=warmup_max_total_tokens,
            warmup_submit_reserve_tokens=warmup_submit_reserve_tokens,
            warmup_force_submit_threshold_tokens=warmup_force_submit_threshold_tokens,
            budget_model_path=budget_model_path,
            lookback_days=lookback_days,
            global_sim_date=global_sim_date,
        )
    )

    if needs_refresh:
        result = prepare_openforesight_search_dataset(
            dataset_path=str(_require(data_cfg, "dataset_path")),
            prepared_data_dir=str(prepared_root),
            search_db=search_db,
            embedding_model=embedding_model,
            embedding_gpu_mem=embedding_gpu_mem,
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
        )
        print(f"Prepared train data: {result.train.rows} rows -> {result.train.path}")
        print(f"Prepared val data:   {result.validation.rows} rows -> {result.validation.path}")

    return str(train_path), str(val_path)


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
    parser.add_argument("--force_rebuild_data", action="store_true")
    parser.add_argument("--skip_data_prep", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = expand_env_tree(_read_yaml(config_path))
    raise_for_unresolved_env_vars(config, f"SkyRL training config {config_path}")
    mode = _normalize_mode(config)
    training = config.get("training", {}) or {}
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

    if args.skip_data_prep:
        prepared_root = _resolve_prepared_data_dir(config.get("data", {}) or {})
        train_path = str(prepared_root / "train.parquet")
        val_path = str(prepared_root / "validation.parquet")
    else:
        train_path, val_path = _prepare_data_if_needed(config=config, force=args.force_rebuild_data)

    overrides = _build_overrides(config=config, train_path=train_path, val_path=val_path)
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "skyrl_integration.train.main_openforesight_search",
        *overrides,
    ]

    print("Launching SkyRL with command:")
    print(" ".join(json.dumps(part) if " " in part else part for part in cmd))

    if args.dry_run:
        return 0

    if mode == "eval":
        preview_path = _write_eval_prompt_preview(config=config, val_path=val_path)
        print(f"Wrote rendered eval prompt preview: {preview_path}")

    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=_build_runtime_env())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
