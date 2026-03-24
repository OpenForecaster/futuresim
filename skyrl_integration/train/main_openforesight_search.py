"""SkyRL entrypoint for OpenForesight warmup-style search training."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure repo root is importable and forecast-sim `sitecustomize.py` shims run in this process.
# Ray workers default to not inheriting PYTHONPATH unless we forward it in runtime_env (see below).
_repo_root = Path(__file__).resolve().parents[2]
_repo_root_str = str(_repo_root)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)
try:
    import sitecustomize  # noqa: F401
except ImportError:
    pass

import json
import socket
import traceback
from datetime import datetime
from typing import Any, Dict

import ray
from loguru import logger
from skyrl.train.config import SkyRLTrainConfig
from skyrl_gym.envs import register

from skyrl_integration.constants import OPENFORESIGHT_SEARCH_WARMUP_ENV_ID


def _coerce_eval_float(value: Any) -> float | None:
    """Normalize numpy / bool / int scalars for W&B + JSON (SkyRL often uses ``np.float64``)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            return None
    return None


def _json_default(obj: Any):
    if hasattr(obj, "item"):
        try:
            return float(obj.item())
        except Exception:
            pass
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _flatten_forecast_eval_metrics(eval_metrics: Dict[str, Any]) -> None:
    """Mirror ``eval/all/environment/*`` as ``eval/forecast/*`` for W&B / trackers."""
    prefix = "eval/all/environment/"
    extra: Dict[str, float] = {}
    for k, v in eval_metrics.items():
        if not k.startswith(prefix):
            continue
        fv = _coerce_eval_float(v)
        if fv is None:
            continue
        extra["eval/forecast/" + k[len(prefix) :]] = fv
    eval_metrics.update(extra)


def _emit_forecast_eval_artifacts(
    eval_metrics: Dict[str, Any],
    *,
    export_path: str | Path,
    global_step: int | None,
) -> None:
    """Print forecast rollups to stdout and write JSON next to SkyRL's eval dump."""
    env_block = {k: v for k, v in eval_metrics.items() if k.startswith("eval/all/environment/")}
    if not env_block:
        logger.warning(
            "[forecast-sim] No eval/all/environment/* in eval_metrics "
            "(OpenForesightSearchWarmupEnv rollout_metrics missing?)."
        )
        return

    step_label = "none" if global_step is None else str(int(global_step))
    dump_dir = (
        Path(export_path).resolve()
        / "dumped_evals"
        / (f"global_step_{int(global_step)}_evals" if global_step is not None else "eval_only")
    )
    dump_dir.mkdir(parents=True, exist_ok=True)
    out_path = dump_dir / f"forecast_sim_eval_metrics_step_{step_label}.json"
    serializable: Dict[str, Any] = {}
    for k, v in env_block.items():
        fv = _coerce_eval_float(v)
        serializable[k] = fv if fv is not None else v
    out_path.write_text(
        json.dumps({"global_step": global_step, "metrics": serializable}, indent=2, sort_keys=True, default=_json_default)
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "",
        "=" * 72,
        "[forecast-sim] OpenForesight eval — forecast_metrics (daily_metrics-aligned)",
        f"  step={step_label}  wrote {out_path}",
        "-" * 72,
    ]
    for k in sorted(env_block.keys()):
        disp = _coerce_eval_float(env_block[k])
        lines.append(f"  {k}: {env_block[k] if disp is None else disp}")
    lines.append("=" * 72 + "\n")
    print("\n".join(lines), file=sys.stdout, flush=True)


def _patch_concatenate_generator_outputs_for_forecast_env_metrics() -> None:
    """
    SkyRL's ``concatenate_generator_outputs`` recomputes ``rollout_metrics`` using only
    token/reward stats, which **drops** per-env aggregates (``environment/*``) produced by
    ``get_rollout_metrics(..., env_metrics, env_classes)``. Restore merged ``environment/*``
    keys so ``evaluate()`` exposes ``eval/all/environment/*`` for logging + W&B.
    """

    import skyrl.train.evaluate as ev
    import skyrl.train.generators.utils as gen_utils
    import skyrl.train.utils.trainer_utils as tu

    if getattr(gen_utils, "_forecast_sim_concat_patch", False):
        return

    _orig = gen_utils.concatenate_generator_outputs

    def _env_slice(rollout_metrics: Dict[str, Any] | None) -> Dict[str, Any]:
        if not rollout_metrics:
            return {}
        return {k: v for k, v in rollout_metrics.items() if str(k).startswith("environment/")}

    def _f(x: Any, default: float = 0.0) -> float:
        v = _coerce_eval_float(x)
        return default if v is None else v

    def _merge_env_rollout_metrics(batch_rollouts: list[Dict[str, Any]]) -> Dict[str, float]:
        chunks = [_env_slice(rm) for rm in batch_rollouts]
        chunks = [c for c in chunks if c]
        if not chunks:
            return {}
        if len(chunks) == 1:
            return {k: _f(v, 0.0) for k, v in chunks[0].items()}

        def pick(chunk: Dict[str, Any], name: str) -> float:
            return _f(chunk.get("environment/" + name), 0.0)

        total_n = sum(pick(c, "total_episodes") for c in chunks)
        valid_n = sum(pick(c, "valid_submits") for c in chunks)
        out: Dict[str, float] = {}

        if total_n > 0.0:
            acc = sum(pick(c, "accuracy") * pick(c, "total_episodes") for c in chunks) / total_n
            exp = sum(pick(c, "exp_acc") * pick(c, "total_episodes") for c in chunks) / total_n
            out["environment/accuracy"] = float(acc)
            out["environment/exp_acc"] = float(exp)
            out["environment/format_failures"] = float(sum(pick(c, "format_failures") for c in chunks))
            out["environment/total_episodes"] = float(total_n)
            out["environment/valid_submits"] = float(valid_n)

        if valid_n > 0.0:
            brier = sum(pick(c, "avg_brier") * pick(c, "valid_submits") for c in chunks) / valid_n
            out["environment/avg_brier"] = float(brier)
        else:
            out["environment/avg_brier"] = 0.0

        skip = {
            "accuracy",
            "exp_acc",
            "format_failures",
            "total_episodes",
            "valid_submits",
            "avg_brier",
        }
        suffixes: set[str] = set()
        for c in chunks:
            for k in c:
                suffixes.add(str(k).split("environment/", 1)[-1])
        for suf in sorted(suffixes):
            if suf in skip:
                continue
            if total_n > 0.0:
                num = sum(pick(c, suf) * pick(c, "total_episodes") for c in chunks)
                out["environment/" + suf] = float(num / total_n)
        return out

    def _patched(generator_outputs: list[Any]) -> Any:
        result = _orig(generator_outputs)
        per_batch = [go.get("rollout_metrics") or {} for go in generator_outputs]
        merged_env = _merge_env_rollout_metrics(per_batch)
        if not merged_env:
            return result
        new_rm = dict(result.get("rollout_metrics") or {})
        new_rm.update(merged_env)
        result["rollout_metrics"] = new_rm
        return result

    gen_utils.concatenate_generator_outputs = _patched  # type: ignore[assignment]
    ev.concatenate_generator_outputs = _patched  # type: ignore[assignment]
    tu.concatenate_generator_outputs = _patched  # type: ignore[assignment]
    try:
        import skyrl.train.fully_async_trainer as fat

        fat.concatenate_generator_outputs = _patched  # type: ignore[assignment]
    except ImportError:
        pass

    gen_utils._forecast_sim_concat_patch = True


def _patch_skyrl_evaluate_for_forecast_metrics() -> None:
    """Wrap SkyRL ``evaluate`` once per process; no third_party edits."""
    import skyrl.train.evaluate as ev

    if getattr(ev, "_forecast_sim_eval_metrics_sink", False):
        return

    _orig = ev.evaluate
    _orig_sw = ev.evaluate_step_wise

    def _wrap_eval(orig):
        async def _wrapped(*args: Any, **kwargs: Any):
            out = await orig(*args, **kwargs)
            cfg = kwargs.get("cfg") if "cfg" in kwargs else args[2]
            global_step = kwargs.get("global_step") if "global_step" in kwargs else args[3]
            _flatten_forecast_eval_metrics(out)
            _emit_forecast_eval_artifacts(out, export_path=cfg.trainer.export_path, global_step=global_step)
            return out

        return _wrapped

    ev.evaluate = _wrap_eval(_orig)  # type: ignore[assignment]
    ev.evaluate_step_wise = _wrap_eval(_orig_sw)  # type: ignore[assignment]
    ev._forecast_sim_eval_metrics_sink = True


def _log_step(message: str) -> None:
    print(f"[forecast-sim skyrl] {message}", file=sys.stderr, flush=True)


def _set_sim_output_dir(cfg: SkyRLTrainConfig) -> None:
    trainer_cfg = getattr(cfg, "trainer", None)
    log_path = getattr(trainer_cfg, "log_path", None) if trainer_cfg is not None else None
    if log_path:
        os.environ.setdefault("SIM_OUTPUT_DIR", str(log_path))


def _ensure_vllm_spawn_multiproc() -> None:
    """
    vLLM launches extra engine-core subprocesses during generation.

    In this SkyRL topology, those subprocesses are created from Ray actors after CUDA
    is already initialized, so `fork` is unsafe. Use vLLM's supported env knob to
    force `spawn` before any inference engine setup.
    """

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def _patch_skyrl_rendezvous_for_new_ray() -> None:
    """
    Patch SkyRL's rendezvous helper for Ray versions without
    ray.experimental.collective.util.get_address_and_port.
    """

    from ray.util.placement_group import PlacementGroupSchedulingStrategy
    from skyrl.backends.skyrl_train.inference_engines import utils as engine_utils

    if getattr(engine_utils, "_forecast_sim_ray_collective_patch", False):
        return

    def _get_rendezvous_addr_port(placement_group, pg_index: int) -> tuple[str, int]:
        @ray.remote(num_cpus=0, num_gpus=0)
        def _get_addr_port() -> tuple[str, int]:
            address = ray.util.get_node_ip_address()
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("", 0))
                port = int(sock.getsockname()[1])
            return address, port

        master_sched = PlacementGroupSchedulingStrategy(
            placement_group=placement_group,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=pg_index,
        )
        return ray.get(_get_addr_port.options(scheduling_strategy=master_sched).remote())

    engine_utils.get_rendezvous_addr_port = _get_rendezvous_addr_port
    engine_utils._forecast_sim_ray_collective_patch = True


def _patch_skyrl_peer_access_supported_for_gpu_driver() -> None:
    """
    SkyRL's stock ``peer_access_supported`` spins up a temporary Ray cluster on CPU-only
    drivers to run a 2-GPU P2P probe. HTCondor/MPI training always runs this entrypoint on
    a GPU execute node with CUDA visible on the driver process, so we drop that path: local
    ``run_p2p_access_check`` only, otherwise treat as no peer access (SkyRL sets NCCL
    P2P/SHM disables — same conservative outcome as a failed remote probe).
    """

    import torch
    from skyrl.train.utils import utils as train_utils

    if getattr(train_utils, "_forecast_sim_peer_access_patch", False):
        return

    _run_p2p = train_utils.run_p2p_access_check

    def _peer_access_supported(max_num_gpus_per_node: int) -> bool:
        if max_num_gpus_per_node <= 1:
            return False
        if torch.cuda.is_available():
            return bool(_run_p2p())
        logger.info(
            "[forecast-sim] CUDA not visible on SkyRL driver; skipping P2P probe "
            "(run on a GPU execute node with GPUs visible to the training process)."
        )
        return False

    train_utils.peer_access_supported = _peer_access_supported
    train_utils._forecast_sim_peer_access_patch = True


def _patch_skyrl_compat() -> None:
    """Rendezvous patch + GPU-driver-only peer access (no temporary Ray for CPU heads)."""
    _patch_skyrl_rendezvous_for_new_ray()
    _patch_skyrl_peer_access_supported_for_gpu_driver()


def _initialize_ray_with_memory(cfg: SkyRLTrainConfig) -> None:
    """
    Initialize Ray like SkyRL normally does, but with explicit memory sizing.

    This keeps the fix at the launch boundary instead of monkey-patching `ray.init`
    process-wide.
    """

    from skyrl.backends.skyrl_train.utils.ppo_utils import sync_registries
    from skyrl.train.utils.utils import (
        SKYRL_DUMP_INFRA_LOG_TO_STDOUT,
        prepare_runtime_environment,
    )

    verbose_logging = SKYRL_DUMP_INFRA_LOG_TO_STDOUT
    if not verbose_logging:
        os.environ["RAY_BACKEND_LOG_LEVEL"] = "fatal"

    env_vars = prepare_runtime_environment(cfg)

    # SkyRL only exports PYTHONPATH when SKYRL_PYTHONPATH_EXPORT is set. HTCondor jobs may also
    # run with a minimal inherited env. Always push a canonical PYTHONPATH + FSIM_REPO_DIR so Ray
    # workers (and any child processes that inherit PYTHONPATH) can import repo code and run
    # `sitecustomize.py` shims when appropriate.
    repo_root = _repo_root_str
    env_vars.setdefault("FSIM_REPO_DIR", os.environ.get("FSIM_REPO_DIR", "").strip() or repo_root)
    _skyrl_root = os.path.join(repo_root, "third_party", "SkyRL")
    _gym_root = os.path.join(_skyrl_root, "skyrl-gym")
    _driver_pp = os.environ.get("PYTHONPATH", "").strip()
    _pp_parts = [repo_root, _skyrl_root, _gym_root]
    if _driver_pp:
        _pp_parts.append(_driver_pp)
    env_vars["PYTHONPATH"] = os.pathsep.join([p for p in _pp_parts if p])

    if not verbose_logging:
        log_path = Path(cfg.trainer.log_path).resolve()
        log_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        log_file = str(log_path / f"infra-{timestamp}.log")
        os.environ["SKYRL_LOG_FILE"] = log_file
        env_vars["SKYRL_LOG_FILE"] = log_file
    else:
        log_file = ""

    # Registry sync cloudpickles every policy-loss / advantage-estimator into the object store.
    # 256MiB is too small in practice (driver/worker put failures or OOM with no Python traceback).
    _obj_store = int(os.environ.get("FSIM_RAY_OBJECT_STORE_BYTES", str(2 * 1024**3)))
    _ray_heap = int(os.environ.get("FSIM_RAY_HEAP_MEMORY_BYTES", str(4 * 1024**3)))
    _log_step(f"ray.init: object_store_memory={_obj_store} _memory={_ray_heap}")

    ray.init(
        runtime_env={"env_vars": env_vars},
        log_to_driver=True,
        include_dashboard=False,
        object_store_memory=_obj_store,
        _memory=_ray_heap,
    )
    _log_step("ray.init returned")

    # Avoid Loguru -> SKYRL_LOG_FILE here: on Lustre/NFS the file sink can block the driver before
    # ``sync_registries()``, which looks like a hang right after "Started a local Ray instance".
    if not verbose_logging:
        _log_step(f"Infrastructure logs will be written to: {log_file}")

    _log_step("sync_registries: starting (PolicyLossRegistry + AdvantageEstimatorRegistry)")
    try:
        sync_registries()
    except BaseException:
        _log_step("sync_registries: FAILED")
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
    _log_step("sync_registries: done")


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    try:
        _log_step("skyrl_entrypoint: starting")
        _set_sim_output_dir(cfg)
        _ensure_vllm_spawn_multiproc()
        _patch_skyrl_compat()
        from skyrl.train.entrypoints.main_base import BasePPOExp

        register(
            id=OPENFORESIGHT_SEARCH_WARMUP_ENV_ID,
            entry_point="skyrl_integration.envs.openforesight_search_warmup_env:OpenForesightSearchWarmupEnv",
        )
        _log_step("skyrl_entrypoint: env registered")

        _patch_concatenate_generator_outputs_for_forecast_env_metrics()
        _patch_skyrl_evaluate_for_forecast_metrics()

        exp = BasePPOExp(cfg)
        _log_step("skyrl_entrypoint: experiment constructed")
        exp.run()
        _log_step("skyrl_entrypoint: experiment finished")
    except BaseException:
        _log_step("skyrl_entrypoint: unhandled exception")
        traceback.print_exc(file=sys.stderr)
        raise


def main() -> None:
    try:
        _log_step("main: starting")
        _patch_skyrl_compat()
        from skyrl.train.entrypoints.main_base import validate_cfg

        cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
        _log_step("main: config loaded from CLI overrides")
        validate_cfg(cfg)
        _log_step("main: config validated")
        _set_sim_output_dir(cfg)

        _initialize_ray_with_memory(cfg)
        _log_step("main: ray initialized")
        ray.get(skyrl_entrypoint.remote(cfg))
        _log_step("main: skyrl_entrypoint completed")
    except BaseException:
        _log_step("main: unhandled exception")
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
