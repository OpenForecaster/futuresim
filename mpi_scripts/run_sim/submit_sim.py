#!/usr/bin/env python3
"""
Submit simulation job(s) to HTCondor.

Usage:
    # Basic:
    python submit_sim.py --config configs/allq_sim_ds.yaml --runs 1

    # Override config keys at submit time (repeatable):
    python submit_sim.py --config configs/allq_sim_ds.yaml \\
        --set sim_name=allq_tmp \\
        --set defaults.temperature=0.2 \\
        --set resources.cpus=32
"""

import argparse
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def _parse_set_kv(s: str) -> tuple[str, Any]:
    if "=" not in s:
        raise ValueError(f"Invalid --set {s!r}. Expected key=value.")
    key, raw = s.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Invalid --set {s!r}. Empty key.")
    # Parse value as YAML to support bool/int/float/null and quoted strings.
    import yaml
    val = yaml.safe_load(raw)
    return key, val


def _tokenize_path(path: str) -> list[Any]:
    # Supports dot paths + list indices: agents[0].model
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

        # Not last: ensure next container exists and has correct type.
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


def submit_sim_job(
    config: dict,
    run_id: int = 0,
    gpus: int = 1,
    cpus: int = 16,
    memory_gb: int = 80,
    disk_gb: int = 50,
    bid: int = 25,
    requirements: str = None,
    dry_run: bool = False,
) -> int:
    """Submit a single simulation job."""
    htcondor = None
    try:
        import htcondor2 as _htcondor
        htcondor = _htcondor
    except ImportError:
        try:
            import htcondor as _htcondor
            htcondor = _htcondor
        except ImportError:
            htcondor = None
    script_dir = Path(__file__).parent
    
    # 1. Determine sim_name
    base_sim_name = config.get("sim_name", "sim_run")
    
    # Unique sim name for this specific run instance
    unique_name = f"{base_sim_name}_r{run_id:02d}"
    
    # 2. Setup directories
    resume_path = config.get("resume")
    if resume_path:
        run_dir = Path(resume_path)
    else:
        # Logs go to /fast/sgoel/logs/forecasting-sim/sims/<base_name>/<unique_name>
        log_base = Path("/fast/nchandak/logs/forecasting-sim/sims")
        run_dir = log_base / base_sim_name / unique_name
        run_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Prepare run specific config
    run_config = config.copy()
    if not resume_path:
        run_config["sim_name"] = unique_name
    
    # Save run config
    config_path = run_dir / "config.yaml"
    import yaml
    with open(config_path, "w") as f:
        yaml.dump(run_config, f)
        
    if resume_path:
        print(f"Resuming in directory: {run_dir}")
    else:
        print(f"Generated run config: {config_path}")

    # 4. Prepare submission
    run_script = config.get("run_script", "run_sim.sh")
    executable = str(script_dir / run_script)
    log_prefix = str(run_dir / "job")
    
    job_settings = {
        "executable": executable,
        "arguments": str(config_path),
        "output": str(run_dir / "$(ClusterId).out"),
        "error": str(run_dir / "$(ClusterId).err"), 
        "log": str(run_dir / "$(ClusterId).log"),
        "request_cpus": str(cpus),
        "request_memory": f"{memory_gb}GB",
        "request_disk": f"{disk_gb}GB",
        "request_gpus": str(gpus),
        "jobprio": str(bid - 1000),
        "requirements": requirements or "TARGET.CUDACapability >= 8.0 && TARGET.CUDAGlobalMemoryMb > 70000",
        "environment": "PYTHONUNBUFFERED=1",
    }
    
    if dry_run:
        print(f"Dry run: Job would be submitted to {run_dir}")
        print(f"  Executable: {executable}")
        print(f"  Arguments: {config_path}")
        return -1

    if htcondor is not None:
        job = htcondor.Submit(job_settings)
        schedd = htcondor.Schedd()
        result = schedd.submit(job, count=1)
        return result.cluster()

    # Fallback path when python HTCondor bindings are unavailable.
    with tempfile.NamedTemporaryFile("w", suffix=".sub", delete=False) as tf:
        sub_path = Path(tf.name)
        for k, v in job_settings.items():
            tf.write(f"{k} = {v}\n")
        tf.write("queue 1\n")

    try:
        proc = subprocess.run(
            ["condor_submit_bid", str(bid), str(sub_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        m = re.search(r"cluster\s+(\d+)", out, flags=re.IGNORECASE)
        if not m:
            raise RuntimeError(f"Failed to parse cluster id from condor_submit output:\n{out}")
        return int(m.group(1))
    finally:
        try:
            sub_path.unlink(missing_ok=True)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Submit simulation job to HTCondor using a config file")
    
    # Configuration input
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        help="Override config values (repeatable). Example: --set defaults.temperature=0.2",
    )
    
    # Submission overrides / controls
    parser.add_argument("--runs", type=int, default=1, help="Number of runs (for variance)")
    parser.add_argument("--dry-run", action="store_true", help="Generate config but do not submit job")
    
    parser.add_argument("--resume", help="Directory of a previous run to resume")
    parser.add_argument("--rescore", action="store_true", help="Recalculate metrics from history before resuming")
    parser.add_argument("--restart_from", help="Directory of a previous run to restart from")
    parser.add_argument("--restart_from_day", help="Day to restart from (YYYY-MM-DD)")

    args = parser.parse_args()
    
    # Load input config
    import yaml
    with open(args.config, "r") as f:
        base_config = yaml.safe_load(f)

    # Apply CLI overrides
    if args.resume:
        base_config["resume"] = args.resume
    if args.rescore:
        base_config["rescore"] = True
    if args.restart_from:
        base_config["restart_from"] = args.restart_from
    if args.restart_from_day:
        base_config["restart_from_day"] = args.restart_from_day
    if args.set_values:
        for s in args.set_values:
            k, v = _parse_set_kv(s)
            _set_in_config(base_config, k, v)
        
    print(f"Submitting {args.runs} simulation job(s)...")
    print(f"  Config: {args.config}")
    print(f"  Dataset: {base_config.get('dataset', 'unknown')}")

    # Infer resources from config unless explicitly overridden by CLI flags.
    # Preferred format:
    #   resources:
    #     gpus: 1
    #     cpus: 16
    #     memory_gb: 80
    #     disk_gb: 50
    #     bid: 25
    #     requirements: "..."
    resources = base_config.get("resources", {}) or {}
    inferred_gpus = resources.get("gpus", base_config.get("gpus", 1))
    inferred_cpus = resources.get("cpus", base_config.get("cpus", 16))
    inferred_memory = resources.get("memory_gb", base_config.get("memory_gb", 80))
    inferred_disk = resources.get("disk_gb", base_config.get("disk_gb", 50))
    inferred_bid = resources.get("bid", base_config.get("bid", 25))
    inferred_requirements = resources.get("requirements", base_config.get("requirements"))
    
    print(f"  Resources: gpus={inferred_gpus} cpus={inferred_cpus} mem={inferred_memory}GB disk={inferred_disk}GB bid={inferred_bid}")
    if inferred_requirements:
        print(f"  Requirements: {inferred_requirements}")
    
    cluster_ids = []
    for i in range(args.runs):
        cluster_id = submit_sim_job(
            config=base_config,
            run_id=i,
            gpus=inferred_gpus,
            cpus=inferred_cpus,
            memory_gb=inferred_memory,
            disk_gb=inferred_disk,
            bid=inferred_bid,
            requirements=inferred_requirements,
            dry_run=args.dry_run,
        )
        cluster_ids.append(cluster_id)
        print(f"  Submitted run {i}: cluster {cluster_id}")
    
    print(f"\nAll jobs submitted! Cluster IDs: {cluster_ids}")
    
    sim_name = base_config.get("sim_name", "sim_run")
    log_base = Path("/fast/nchandak/logs/forecasting-sim/sims")
    print(f"Logs: {log_base / sim_name}/")


if __name__ == "__main__":
    main()
