#!/usr/bin/env python3
"""
HTCondor job launcher for CPU-only jobs on MPI-IS cluster.

Usage:
    python submit_job.py --cpus 32 --memory 256 --bid 500
"""

import argparse
from pathlib import Path

import htcondor


def submit_cpu_job(
    executable: str,
    log_dir: str,
    cpus: int = 8,
    memory_gb: int = 32,
    disk_gb: int = 32,
    bid: int = 15,
    notify_email: str = None,
) -> int:
    """
    Submit a CPU-only job to HTCondor.
    
    Args:
        executable: Path to the shell script to run
        log_dir: Directory for job logs
        cpus: Number of CPU cores
        memory_gb: Memory in GB
        disk_gb: Disk space in GB
        bid: Job priority bid (higher = more priority, costs more)
        notify_email: Email for error notifications (optional)
    
    Returns:
        Cluster ID of the submitted job
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    log_prefix = str(log_path / "$(Cluster).$(Process)")
    
    job_settings = {
        "executable": executable,
        "output": f"{log_prefix}.out",
        "error": f"{log_prefix}.err",
        "log": f"{log_prefix}.log",
        "request_cpus": str(cpus),
        "request_memory": f"{memory_gb}GB",
        "request_disk": f"{disk_gb}GB",
        "request_gpus": "0",
        "jobprio": str(bid - 1000),
    }
    
    if notify_email:
        job_settings["notify_user"] = notify_email
        job_settings["notification"] = "error"
    
    job = htcondor.Submit(job_settings)
    schedd = htcondor.Schedd()
    result = schedd.submit(job)
    
    return result.cluster()


def main():
    parser = argparse.ArgumentParser(description="Submit CPU job to HTCondor")
    parser.add_argument("--cpus", type=int, default=32, help="Number of CPUs")
    parser.add_argument("--memory", type=int, default=256, help="Memory in GB")
    parser.add_argument("--disk", type=int, default=64, help="Disk in GB")
    parser.add_argument("--bid", type=int, default=15, help="Job priority bid")
    parser.add_argument("--email", type=str, default=None, help="Notification email")
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    executable = str(script_dir / "run_conversion.sh")
    log_dir = "/is/cluster/fast/sgoel/logs/forecasting-sim/news_conversion"
    
    cluster_id = submit_cpu_job(
        executable=executable,
        log_dir=log_dir,
        cpus=args.cpus,
        memory_gb=args.memory,
        disk_gb=args.disk,
        bid=args.bid,
        notify_email=args.email,
    )
    
    print(f"Submitted job with cluster ID: {cluster_id}")
    print(f"Logs will be at: {log_dir}/{cluster_id}.*")


if __name__ == "__main__":
    main()
