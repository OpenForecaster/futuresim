#!/usr/bin/env python3
"""
Submit simulation job(s) to SLURM using AISA path conventions.

Usage:
    # Basic:
    python aisa_scripts/run_sim/submit_sim_aisa.py --config configs/allq_sim_ds_aisa.yaml --runs 1

    # Override config keys at submit time (repeatable):
    python aisa_scripts/run_sim/submit_sim_aisa.py --config configs/allq_sim_ds_aisa.yaml \
        --set sim_name=allq_tmp \
        --set defaults.temperature=0.2 \
        --set resources.cpus=32
"""

import argparse
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any


def _parse_set_kv(s: str) -> tuple[str, Any]:
    if "=" not in s:
        raise ValueError(f"Invalid --set {s!r}. Expected key=value.")
    key, raw = s.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Invalid --set {s!r}. Empty key.")
    import yaml
    val = yaml.safe_load(raw)
    return key, val


def _tokenize_path(path: str) -> list[Any]:
    tokens: list[Any] = []
    for part in path.split("."):
        part = part.strip()
        if not part:
            raise ValueError(f"Invalid --set path {path!r}: empty segment")
        for m in re.finditer(r"([^\[\]]+)|\[(\d+)\]", part):
            key, idx = m.group(1), m.group(2)
            if key is not None:
                k = key.strip()
                if not k:
                    raise ValueError(f"Invalid --set path {path!r}: empty key in segment {part!r}")
                tokens.append(k)
            else:
                tokens.append(int(idx))
    return tokens


def _ensure_list_len(xs: list, n: int) -> None:
    while len(xs) <= n:
        xs.append(None)


def _set_in_config(cfg: Any, path: str, value: Any) -> None:
    tokens = _tokenize_path(path)
    cur = cfg
    for i, tok in enumerate(tokens):
        is_last = i == len(tokens) - 1
        nxt = None if is_last else tokens[i + 1]

        if is_last:
            if isinstance(tok, int):
                if not isinstance(cur, list):
                    raise ValueError(f"Path {path!r} expects list at final container, got {type(cur).__name__}")
                _ensure_list_len(cur, tok)
                cur[tok] = value
                return
            if not isinstance(cur, dict):
                raise ValueError(f"Path {path!r} expects dict at final container, got {type(cur).__name__}")
            cur[tok] = value
            return

        if isinstance(tok, int):
            if not isinstance(cur, list):
                raise ValueError(f"Path {path!r} expects list container, got {type(cur).__name__}")
            _ensure_list_len(cur, tok)
            if cur[tok] is None:
                cur[tok] = [] if isinstance(nxt, int) else {}
            cur = cur[tok]
            continue

        if not isinstance(cur, dict):
            raise ValueError(f"Path {path!r} expects dict container, got {type(cur).__name__}")
        if tok not in cur or cur[tok] is None:
            cur[tok] = [] if isinstance(nxt, int) else {}
        cur = cur[tok]


def _parse_sbatch_output(stdout: str) -> str:
    out = (stdout or "").strip()
    if not out:
        return ""
    # --parsable usually returns just jobid or jobid;cluster.
    first = out.splitlines()[0].strip()
    return first.split(";")[0].strip()


def submit_sim_job(
    config: dict,
    run_id: int = 0,
    gpus: int = 1,
    cpus: int = 16,
    memory_gb: int = 80,
    disk_gb: int = 50,
    partition: str | None = None,
    account: str | None = None,
    qos: str | None = None,
    time_limit: str | None = None,
    constraint: str | None = None,
    sbatch_args: list[str] | None = None,
    dry_run: bool = False,
) -> str:
    """Submit a single simulation job to SLURM."""
    if not shutil_which("sbatch"):
        raise RuntimeError("sbatch not found in PATH. SLURM client is required.")

    script_dir = Path(__file__).parent
    base_sim_name = config.get("sim_name", "sim_run")
    unique_name = f"{base_sim_name}_r{run_id:02d}"

    resume_path = config.get("resume")
    if resume_path:
        run_dir = Path(resume_path)
    else:
        log_base = Path(config.get("_repo_dir", Path.cwd())) / "logs" / "sims"
        run_dir = log_base / base_sim_name / unique_name
        run_dir.mkdir(parents=True, exist_ok=True)

    run_config = config.copy()
    if not resume_path:
        run_config["sim_name"] = unique_name

    config_path = run_dir / "config.yaml"
    import yaml
    with open(config_path, "w") as f:
        yaml.dump(run_config, f)

    executable = str(script_dir / "run_sim_aisa.sh")
    sbatch_cmd = [
        "sbatch",
        "--parsable",
        "--job-name",
        unique_name,
        "--output",
        str(run_dir / "%j.out"),
        "--error",
        str(run_dir / "%j.err"),
        "--cpus-per-task",
        str(cpus),
        "--mem",
        f"{memory_gb}G",
    ]

    if gpus > 0:
        sbatch_cmd += ["--gres", f"gpu:{gpus}"]
    if partition:
        sbatch_cmd += ["--partition", str(partition)]
    if account:
        sbatch_cmd += ["--account", str(account)]
    if qos:
        sbatch_cmd += ["--qos", str(qos)]
    if time_limit:
        sbatch_cmd += ["--time", str(time_limit)]
    if constraint:
        sbatch_cmd += ["--constraint", str(constraint)]
    if sbatch_args:
        sbatch_cmd.extend(sbatch_args)

    sbatch_cmd += [executable, str(config_path)]

    if dry_run:
        print("Dry run: job would be submitted with command:")
        print("  " + " ".join(shlex.quote(x) for x in sbatch_cmd))
        return "DRY_RUN"

    proc = subprocess.run(sbatch_cmd, check=True, capture_output=True, text=True)
    job_id = _parse_sbatch_output(proc.stdout)
    if not job_id:
        raise RuntimeError(f"Failed to parse SLURM job id. sbatch output:\n{proc.stdout}\n{proc.stderr}")
    return job_id


def shutil_which(binary: str) -> str | None:
    from shutil import which
    return which(binary)


def _coerce_sbatch_args(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw if str(x).strip()]
    if isinstance(raw, str):
        return shlex.split(raw)
    return [str(raw)]


def main():
    parser = argparse.ArgumentParser(description="Submit simulation job(s) to SLURM using a config file")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        help="Override config values (repeatable). Example: --set defaults.temperature=0.2",
    )
    parser.add_argument("--runs", type=int, default=1, help="Number of runs (for variance)")
    parser.add_argument("--dry-run", action="store_true", help="Generate config and print sbatch command")
    parser.add_argument("--resume", help="Directory of a previous run to resume")
    parser.add_argument("--rescore", action="store_true", help="Recalculate metrics from history before resuming")
    parser.add_argument("--restart_from", help="Directory of a previous run to restart from")
    parser.add_argument("--restart_from_day", help="Day to restart from (YYYY-MM-DD)")
    # SLURM resource overrides (take precedence over config)
    parser.add_argument("--gpus", type=int, default=None, help="Number of GPUs")
    parser.add_argument("--cpus", type=int, default=None, help="Number of CPUs")
    parser.add_argument("--memory", type=int, default=None, help="Memory in GB")
    parser.add_argument("--time", default=None, help="SLURM time limit (e.g. 2-00:00:00)")
    parser.add_argument("--partition", default=None, help="SLURM partition")
    parser.add_argument("--constraint", default=None, help="SLURM constraint")
    args = parser.parse_args()

    import yaml
    with open(args.config, "r") as f:
        base_config = yaml.safe_load(f)

    if args.resume:
        base_config["resume"] = args.resume
    if args.rescore:
        base_config["rescore"] = True
    if args.restart_from:
        base_config["restart_from"] = args.restart_from
    if args.restart_from_day:
        base_config["restart_from_day"] = args.restart_from_day
    for s in args.set_values:
        k, v = _parse_set_kv(s)
        _set_in_config(base_config, k, v)

    print(f"Submitting {args.runs} simulation job(s) to SLURM...")
    print(f"  Config: {args.config}")
    print(f"  Dataset: {base_config.get('dataset', 'unknown')}")

    resources = base_config.get("resources", {}) or {}
    gpus = args.gpus if args.gpus is not None else int(resources.get("gpus", base_config.get("gpus", 1)))
    cpus = args.cpus if args.cpus is not None else int(resources.get("cpus", base_config.get("cpus", 16)))
    memory_gb = args.memory if args.memory is not None else int(resources.get("memory_gb", base_config.get("memory_gb", 80)))
    disk_gb = int(resources.get("disk_gb", base_config.get("disk_gb", 50)))

    partition = args.partition or resources.get("partition", base_config.get("partition"))
    account = resources.get("account", base_config.get("account"))
    qos = resources.get("qos", base_config.get("qos"))
    time_limit = args.time or resources.get("time", base_config.get("time"))
    constraint = args.constraint or resources.get("constraint", base_config.get("constraint"))
    sbatch_args = _coerce_sbatch_args(resources.get("sbatch_args", base_config.get("sbatch_args")))

    if resources.get("requirements") or base_config.get("requirements"):
        print("  Note: HTCondor 'requirements' is ignored by SLURM submitter.")
    if resources.get("bid") or base_config.get("bid"):
        print("  Note: HTCondor 'bid' is ignored by SLURM submitter.")

    print(f"  Resources: gpus={gpus} cpus={cpus} mem={memory_gb}GB")
    if disk_gb:
        print(f"  Note: HTCondor disk_gb={disk_gb} is ignored by this SLURM submitter.")
    if partition:
        print(f"  Partition: {partition}")
    if account:
        print(f"  Account: {account}")
    if qos:
        print(f"  QOS: {qos}")
    if time_limit:
        print(f"  Time: {time_limit}")
    if constraint:
        print(f"  Constraint: {constraint}")
    if sbatch_args:
        print(f"  Extra sbatch args: {' '.join(sbatch_args)}")

    job_ids: list[str] = []
    for i in range(args.runs):
        job_id = submit_sim_job(
            config=base_config,
            run_id=i,
            gpus=gpus,
            cpus=cpus,
            memory_gb=memory_gb,
            disk_gb=disk_gb,
            partition=partition,
            account=account,
            qos=qos,
            time_limit=time_limit,
            constraint=constraint,
            sbatch_args=sbatch_args,
            dry_run=args.dry_run,
        )
        job_ids.append(job_id)
        print(f"  Submitted run {i}: job {job_id}")

    print(f"\nAll jobs submitted! Job IDs: {job_ids}")
    sim_name = base_config.get("sim_name", "sim_run")
    logs_root = Path.cwd() / "logs" / "sims"
    print(f"Logs: {logs_root / sim_name}/")


if __name__ == "__main__":
    main()
