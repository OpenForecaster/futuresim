#!/usr/bin/env python3
"""
HTCondor job launcher for GPU embedding jobs on MPI-IS cluster.

Usage:
    # Single GPU job:
    python submit_job.py --gpus 1 --memory 64 --bid 25

    # Array job with 8 parallel workers:
    python submit_job.py --gpus 1 --memory 64 --bid 25 --num_workers 8
"""

import argparse
from pathlib import Path

import htcondor2 as htcondor


def submit_gpu_job(
    executable: str,
    log_dir: str,
    gpus: int = 1,
    cpus: int = 16,
    memory_gb: int = 64,
    disk_gb: int = 100,
    bid: int = 25,
    num_workers: int = 1,
    notify_email: str = None,
) -> int:
    """
    Submit GPU job(s) to HTCondor.
    
    Args:
        executable: Path to the shell script to run
        log_dir: Directory for job logs
        gpus: Number of GPUs per job
        cpus: Number of CPU cores per job
        memory_gb: Memory in GB per job
        disk_gb: Disk space in GB per job
        bid: Job priority bid (higher = more priority, costs more)
        num_workers: Number of parallel workers (array job)
        notify_email: Email for error notifications (optional)
    
    Returns:
        Cluster ID of the submitted job
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    log_prefix = str(log_path / "$(Cluster).$(Process)")
    
    job_settings = {
        "executable": executable,
        "arguments": f"$(Process) {num_workers}",
        "output": f"{log_prefix}.out",
        "error": f"{log_prefix}.err",
        "log": str(log_path / "$(Cluster).log"),  # Shared log file
        "request_cpus": str(cpus),
        "request_memory": f"{memory_gb}GB",
        "request_disk": f"{disk_gb}GB",
        "request_gpus": str(gpus),
        "jobprio": str(bid - 1000),
        # GPU requirements (A100/H100)
        "requirements": "TARGET.CUDACapability >= 8.0",
    }
    
    if notify_email:
        job_settings["notify_user"] = notify_email
        job_settings["notification"] = "error"
    
    job = htcondor.Submit(job_settings)
    schedd = htcondor.Schedd()
    result = schedd.submit(job, count=num_workers)
    
    return result.cluster()


def main():
    parser = argparse.ArgumentParser(description="Submit GPU embedding job to HTCondor")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs per job")
    parser.add_argument("--cpus", type=int, default=16, help="Number of CPUs per job")
    parser.add_argument("--memory", type=int, default=64, help="Memory in GB per job")
    parser.add_argument("--disk", type=int, default=100, help="Disk in GB per job")
    parser.add_argument("--bid", type=int, default=25, help="Job priority bid")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--email", type=str, default=None, help="Notification email")
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    executable = str(script_dir / "run_embed.sh")
    log_dir = "/is/cluster/fast/sgoel/logs/forecasting-sim/embed"
    
    cluster_id = submit_gpu_job(
        executable=executable,
        log_dir=log_dir,
        gpus=args.gpus,
        cpus=args.cpus,
        memory_gb=args.memory,
        disk_gb=args.disk,
        bid=args.bid,
        num_workers=args.num_workers,
        notify_email=args.email,
    )
    
    print(f"Submitted {args.num_workers} GPU job(s) with cluster ID: {cluster_id}")
    print(f"Logs: {log_dir}/{cluster_id}/")


if __name__ == "__main__":
    main()
