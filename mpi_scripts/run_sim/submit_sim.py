#!/usr/bin/env python3
"""
Submit simulation job(s) to HTCondor.

Usage:
    # Single agent, single run:
    python submit_sim.py --model xiaomi/mimo-v2-flash:free --name single_test

    # Single agent, 5 runs for variance testing:
    python submit_sim.py --model xiaomi/mimo-v2-flash:free --name variance_test --runs 5

    # 4-agent selfplay, single run:
    python submit_sim.py --model xiaomi/mimo-v2-flash:free --agents 4 --name selfplay_test

    # With search enabled:
    python submit_sim.py --model xiaomi/mimo-v2-flash:free --name search_test --search
"""

import argparse
from pathlib import Path

import htcondor2 as htcondor


def submit_sim_job(
    config: dict,
    sim_name_override: str = None,
    run_id: int = 0,
    gpus: int = 1,
    cpus: int = 16,
    memory_gb: int = 80,
    disk_gb: int = 50,
    bid: int = 25,
    dry_run: bool = False,
) -> int:
    """Submit a single simulation job."""
    script_dir = Path(__file__).parent
    
    # 1. Determine sim_name
    # If passed in args, use that. Else get from config. Else default.
    base_sim_name = sim_name_override or config.get("sim_name", "sim_run")
    
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
    executable = str(script_dir / "run_sim.sh")
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
        "requirements": "TARGET.CUDACapability >= 8.0 && TARGET.CUDAGlobalMemoryMb > 70000",
        "environment": "PYTHONUNBUFFERED=1",
    }
    
    job = htcondor.Submit(job_settings)
    
    if dry_run:
        print(f"Dry run: Job would be submitted to {run_dir}")
        print(f"  Executable: {executable}")
        print(f"  Arguments: {config_path}")
        return -1
        
    schedd = htcondor.Schedd()
    result = schedd.submit(job, count=1)
    
    return result.cluster()


def main():
    parser = argparse.ArgumentParser(description="Submit simulation job to HTCondor using a config file")
    
    # Configuration input
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    
    # Submission overrides / controls
    parser.add_argument("--name", help="Override simulation name prefix")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs (for variance)")
    parser.add_argument("--dry-run", action="store_true", help="Generate config but do not submit job")
    
    # Resources
    # Resources
    parser.add_argument("--gpus", type=int, default=1, help="GPUs per job")
    parser.add_argument("--memory", type=int, default=80, help="Memory in GB")
    parser.add_argument("--bid", type=int, default=25, help="HTCondor bid")
    
    parser.add_argument("--resume", help="Directory of a previous run to resume")
    parser.add_argument("--rescore", action="store_true", help="Recalculate metrics from history before resuming")
    parser.add_argument("--dataset", help="Override dataset in config")

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
    if args.dataset:
        base_config["dataset"] = args.dataset
        
    print(f"Submitting {args.runs} simulation job(s)...")
    print(f"  Config: {args.config}")
    if args.name:
        print(f"  Name override: {args.name}")
    print(f"  Dataset: {base_config.get('dataset', 'unknown')}")
    
    cluster_ids = []
    for i in range(args.runs):
        cluster_id = submit_sim_job(
            config=base_config,
            sim_name_override=args.name,
            run_id=i,
            gpus=args.gpus,
            memory_gb=args.memory,
            bid=args.bid,
            dry_run=args.dry_run,
            # Pass hardware defaults if not in config? 
            # Actually hardcoded defaults in function signature are clearer for this script's usage
            disk_gb=50, 
            cpus=16,
        )
        cluster_ids.append(cluster_id)
        print(f"  Submitted run {i}: cluster {cluster_id}")
    
    print(f"\nAll jobs submitted! Cluster IDs: {cluster_ids}")
    
    # Determine where logs went for helpful message
    sim_name = args.name or base_config.get("sim_name", "sim_run")
    print(f"Logs: /fast/nchandak/logs/forecasting-sim/sims/{sim_name}/")


if __name__ == "__main__":
    main()
