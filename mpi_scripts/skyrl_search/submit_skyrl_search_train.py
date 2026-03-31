#!/usr/bin/env python3
"""Submit SkyRL OpenForesight warmup-search training jobs to HTCondor."""

from __future__ import annotations

import argparse
import copy
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pathing import expand_env_tree, load_repo_env, raise_for_unresolved_env_vars

load_repo_env(REPO_ROOT)
DEFAULT_SKYRL_LOG_BASE = Path(os.getenv("FSIM_SKYRL_LOG_BASE", str(REPO_ROOT / "logs" / "skyrl")))


def _parse_set_kv(s: str) -> tuple[str, Any]:
    if "=" not in s:
        raise ValueError(f"Invalid --set {s!r}. Expected key=value")
    key, raw = s.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Invalid --set {s!r}. Empty key")
    return key, yaml.safe_load(raw)


def _tokenize_set_path(path: str) -> list[Any]:
    tokens: list[Any] = []
    for part in path.split("."):
        part = part.strip()
        if not part:
            raise ValueError(f"Invalid --set path {path!r}: empty segment")
        for m in re.finditer(r"([^\[\]]+)|\[(\d+)\]", part):
            key, idx = m.group(1), m.group(2)
            if key is not None:
                tokens.append(key.strip())
            else:
                tokens.append(int(idx))
    return tokens


def _ensure_list_len(xs: list, n: int) -> None:
    while len(xs) <= n:
        xs.append(None)


def _set_in_config(cfg: Any, path: str, value: Any) -> None:
    tokens = _tokenize_set_path(path)
    cur = cfg
    for i, tok in enumerate(tokens):
        is_last = i == len(tokens) - 1
        nxt = None if is_last else tokens[i + 1]

        if is_last:
            if isinstance(tok, int):
                if not isinstance(cur, list):
                    raise ValueError(f"Path {path!r} expects list at final container")
                _ensure_list_len(cur, tok)
                cur[tok] = value
                return
            if not isinstance(cur, dict):
                raise ValueError(f"Path {path!r} expects dict at final container")
            cur[tok] = value
            return

        if isinstance(tok, int):
            if not isinstance(cur, list):
                raise ValueError(f"Path {path!r} expects list container")
            _ensure_list_len(cur, tok)
            if cur[tok] is None:
                cur[tok] = [] if isinstance(nxt, int) else {}
            cur = cur[tok]
            continue

        if not isinstance(cur, dict):
            raise ValueError(f"Path {path!r} expects dict container")
        if tok not in cur or cur[tok] is None:
            cur[tok] = [] if isinstance(nxt, int) else {}
        cur = cur[tok]


def apply_set_overrides(cfg: dict[str, Any], set_values: list[str]) -> None:
    for item in set_values:
        key, value = _parse_set_kv(item)
        _set_in_config(cfg, key, value)


def _with_run_name(path: str, run_name: str) -> str:
    if "{run_name}" in path:
        return path.format(run_name=run_name)
    return path


def _submit_job(
    *,
    config: dict,
    run_id: int,
    gpus: int,
    full_node_gpus: int | None,
    full_node_mode: str | None,
    cpus: int,
    memory_gb: int,
    disk_gb: int,
    bid: int,
    requirements: str,
    dry_run: bool,
) -> int:
    script_dir = Path(__file__).parent
    run_base = Path(str(config.get("run_base", DEFAULT_SKYRL_LOG_BASE))).expanduser()

    base_sim_name = str(config.get("sim_name", "skyrl_search_train"))
    unique_name = f"{base_sim_name}_r{run_id:02d}"

    run_dir = (run_base / base_sim_name / unique_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = copy.deepcopy(config)
    training = run_cfg.setdefault("training", {})
    training["run_name"] = unique_name

    ckpt_path = training.get("ckpt_path")
    if ckpt_path:
        training["ckpt_path"] = _with_run_name(str(ckpt_path), unique_name)
    else:
        training["ckpt_path"] = str(run_dir / "ckpts")

    export_path = training.get("export_path")
    if export_path:
        training["export_path"] = _with_run_name(str(export_path), unique_name)
    else:
        training["export_path"] = str(run_dir / "exports")

    log_path = training.get("log_path")
    if log_path:
        training["log_path"] = _with_run_name(str(log_path), unique_name)
    else:
        training["log_path"] = str(run_dir / "infra")

    cfg_path = (run_dir / "config.yaml").resolve()
    cfg_path.write_text(yaml.safe_dump(run_cfg, sort_keys=False))

    run_script = str(config.get("run_script", "run_skyrl_search_train.sh"))
    executable = str(script_dir / run_script)

    env_map = {
        "PYTHONUNBUFFERED": "1",
        "FSIM_RAY_NUM_CPUS": str(cpus),
        "FSIM_RAY_NUM_GPUS": str(gpus),
    }
    extra_env = config.get("job_env", {}) or {}
    if not isinstance(extra_env, dict):
        raise ValueError("Top-level job_env must be a mapping when provided")
    for raw_key, raw_value in extra_env.items():
        key = str(raw_key).strip()
        if not key:
            raise ValueError("job_env keys must be non-empty strings")
        env_map[key] = "" if raw_value is None else str(raw_value)
    env_parts = [f"{key}={value}" for key, value in env_map.items()]
    environment = '"' + " ".join(env_parts).replace('"', '\\"') + '"'

    effective_requirements = requirements
    if full_node_gpus is not None and full_node_mode == "root_slot":
        # Match only the root partitionable slot for an otherwise-empty full machine.
        effective_requirements = (
            f"({requirements}) && regexp(\"^slot1@\", TARGET.Name) && TARGET.TotalGpus >= {full_node_gpus}"
        )

    job_settings = {
        "executable": executable,
        "arguments": str(cfg_path),
        "output": str(run_dir / "$(ClusterId).out"),
        "error": str(run_dir / "$(ClusterId).err"),
        "log": str(run_dir / "$(ClusterId).log"),
        "jobprio": str(bid - 1000),
        "requirements": effective_requirements,
        "environment": environment,
    }
    if full_node_gpus is None:
        job_settings["request_cpus"] = str(cpus)
        job_settings["request_memory"] = f"{memory_gb}GB"
        job_settings["request_disk"] = f"{disk_gb}GB"
        job_settings["request_gpus"] = str(gpus)
    elif full_node_mode == "root_slot":
        job_settings["request_cpus"] = str(cpus)
        job_settings["request_memory"] = f"{memory_gb}GB"
        job_settings["request_disk"] = f"{disk_gb}GB"
        job_settings["request_gpus"] = str(full_node_gpus)

    full_node_template = None
    if full_node_gpus is not None and full_node_mode != "root_slot":
        full_node_template = "FullNode" if full_node_gpus == 8 else f"FullNode({full_node_gpus})"

    if dry_run:
        print(f"Dry run: would submit {unique_name}")
        print(f"  run_dir: {run_dir}")
        print(f"  executable: {executable}")
        print(f"  arguments: {cfg_path}")
        print(f"  effective requirements: {effective_requirements}")
        if full_node_template is not None:
            print(f"  use template : {full_node_template}")
        return -1

    try:
        import htcondor2 as htcondor  # type: ignore
    except ImportError:
        try:
            import htcondor  # type: ignore
        except ImportError:
            htcondor = None

    if htcondor is not None and full_node_template is None:
        job = htcondor.Submit(job_settings)
        schedd = htcondor.Schedd()
        result = schedd.submit(job, count=1)
        return int(result.cluster())

    with tempfile.NamedTemporaryFile("w", suffix=".sub", delete=False) as tf:
        sub_path = Path(tf.name)
        for k, v in job_settings.items():
            tf.write(f"{k} = {v}\n")
        if full_node_template is not None:
            # HTCondor templates are submit-file syntax only; requirements must appear first.
            tf.write(f"use template : {full_node_template}\n")
        tf.write("queue 1\n")

    try:
        proc = subprocess.run(
            ["condor_submit_bid", str(bid), str(sub_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        match = re.search(r"cluster\s+(\d+)", output, flags=re.IGNORECASE)
        if not match:
            raise RuntimeError(f"Unable to parse cluster id from condor output:\n{output}")
        return int(match.group(1))
    finally:
        sub_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to job/training YAML config")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        help="Override YAML values (repeatable). Example: --set resources.gpus=8",
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        base_config = yaml.safe_load(f)

    if not isinstance(base_config, dict):
        raise ValueError("Top-level YAML config must be a mapping")

    apply_set_overrides(base_config, args.set_values)
    base_config = expand_env_tree(base_config)
    raise_for_unresolved_env_vars(base_config, f"submit config {args.config}")

    resources = base_config.get("resources", {}) or {}
    gpus = int(resources.get("gpus", 4))
    full_node = bool(resources.get("full_node", False))
    full_node_gpus = int(resources.get("full_node_gpus", gpus)) if full_node else None
    full_node_mode = str(resources.get("full_node_mode", "template")) if full_node else None
    cpus = int(resources.get("cpus", 48))
    memory_gb = int(resources.get("memory_gb", 384))
    disk_gb = int(resources.get("disk_gb", 200))
    bid = int(resources.get("bid", 55))
    requirements = str(
        resources.get(
            "requirements",
            'TARGET.CUDACapability >= 9.0 && TARGET.CUDAGlobalMemoryMb > 70000 && TARGET.Machine =!= "i103.internal.cluster.is.localnet"',
        )
    )

    print(f"Submitting {args.runs} SkyRL training job(s)")
    print(f"  Config: {args.config}")
    resource_summary = f"gpus={gpus} cpus={cpus} mem={memory_gb}GB disk={disk_gb}GB bid={bid}"
    if full_node_gpus is not None:
        resource_summary += f" full_node={full_node_gpus} mode={full_node_mode}"
    print(f"  Resources: {resource_summary}")
    print(f"  Requirements: {requirements}")

    cluster_ids: list[int] = []
    for i in range(args.runs):
        cluster_id = _submit_job(
            config=base_config,
            run_id=i,
            gpus=gpus,
            full_node_gpus=full_node_gpus,
            full_node_mode=full_node_mode,
            cpus=cpus,
            memory_gb=memory_gb,
            disk_gb=disk_gb,
            bid=bid,
            requirements=requirements,
            dry_run=args.dry_run,
        )
        cluster_ids.append(cluster_id)
        print(f"  submitted run {i}: cluster {cluster_id}")

    sim_name = str(base_config.get("sim_name", "skyrl_search_train"))
    log_root = Path(str(base_config.get("run_base", DEFAULT_SKYRL_LOG_BASE))) / sim_name
    print(f"All done. Logs: {log_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
