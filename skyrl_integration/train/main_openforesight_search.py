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
    from skyrl_integration.bootstrap_transformers_patches import apply_transformers_runtime_patches

    apply_transformers_runtime_patches()
except Exception:
    pass
try:
    import sitecustomize  # noqa: F401
except ImportError:
    pass

import socket
import traceback
from datetime import datetime

import ray
from skyrl.train.config import SkyRLTrainConfig
from skyrl_gym.envs import register

from skyrl_integration.constants import OPENFORESIGHT_SEARCH_WARMUP_ENV_ID


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


def _patch_skyrl_peer_access_probe_for_ray_memory() -> None:
    """
    Make SkyRL's CPU-side peer-access probe robust to Ray memory auto-detection failures.

    On some MPI execute nodes, Ray's temporary `ray.init()` inside `peer_access_supported()`
    misestimates available memory and raises before the actual training cluster starts.
    If the probe cannot be completed cleanly, fall back to "peer access unsupported" so
    SkyRL disables NCCL P2P/SHM rather than crashing the whole launch.
    """

    import torch
    from ray.util.placement_group import PlacementGroupSchedulingStrategy, placement_group
    from skyrl.train.utils import utils as train_utils

    if getattr(train_utils, "_forecast_sim_peer_access_patch", False):
        return

    def _patched_peer_access_supported(max_num_gpus_per_node: int):
        if max_num_gpus_per_node <= 1:
            return False

        if torch.cuda.is_available():
            return train_utils.run_p2p_access_check()

        started_ray = False
        try:
            if not ray.is_initialized():
                ray.init(
                    include_dashboard=False,
                    log_to_driver=False,
                    object_store_memory=256 * 1024**2,
                    _memory=2 * 1024**3,
                )
                started_ray = True

            pg = placement_group([{"CPU": 1, "GPU": 2}], strategy="PACK")
            train_utils.get_ray_pg_ready_with_timeout(pg, timeout=train_utils.SKYRL_RAY_PG_TIMEOUT_IN_S)
            result = ray.get(
                ray.remote(num_gpus=2, scheduling_strategy=PlacementGroupSchedulingStrategy(pg))(
                    train_utils.run_p2p_access_check
                ).remote()
            )
            return bool(result)
        except Exception as exc:
            _log_step(f"peer_access_supported probe failed; disabling NCCL P2P/SHM. Cause: {exc}")
            return False
        finally:
            if started_ray and ray.is_initialized():
                ray.shutdown()

    train_utils.peer_access_supported = _patched_peer_access_supported
    train_utils._forecast_sim_peer_access_patch = True


def _initialize_ray_with_memory(cfg: SkyRLTrainConfig) -> None:
    """
    Initialize Ray like SkyRL normally does, but with explicit memory sizing.

    This keeps the fix at the launch boundary instead of monkey-patching `ray.init`
    process-wide.
    """

    from loguru import logger
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
    # workers and vLLM subprocesses can load `skyrl_integration/bootstrap_transformers_patches.py`
    # and repo `sitecustomize.py` shims.
    _repo_root = str(Path(__file__).resolve().parents[2])
    env_vars.setdefault("FSIM_REPO_DIR", os.environ.get("FSIM_REPO_DIR", "").strip() or _repo_root)
    _skyrl_root = os.path.join(_repo_root, "third_party", "SkyRL")
    _gym_root = os.path.join(_skyrl_root, "skyrl-gym")
    _driver_pp = os.environ.get("PYTHONPATH", "").strip()
    _pp_parts = [_repo_root, _skyrl_root, _gym_root]
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

    ray.init(
        runtime_env={"env_vars": env_vars},
        log_to_driver=True,
        include_dashboard=False,
        object_store_memory=256 * 1024**2,
        _memory=2 * 1024**3,
    )

    if not verbose_logging:
        logger.info(f"Infrastructure logs will be written to: {log_file}")

    sync_registries()


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    try:
        _log_step("skyrl_entrypoint: starting")
        _set_sim_output_dir(cfg)
        _ensure_vllm_spawn_multiproc()
        _patch_skyrl_rendezvous_for_new_ray()
        _patch_skyrl_peer_access_probe_for_ray_memory()
        from skyrl.train.entrypoints.main_base import BasePPOExp

        register(
            id=OPENFORESIGHT_SEARCH_WARMUP_ENV_ID,
            entry_point="skyrl_integration.envs.openforesight_search_warmup_env:OpenForesightSearchWarmupEnv",
        )
        _log_step("skyrl_entrypoint: env registered")

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
        _patch_skyrl_rendezvous_for_new_ray()
        _patch_skyrl_peer_access_probe_for_ray_memory()
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
