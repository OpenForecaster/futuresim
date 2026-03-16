#!/usr/bin/env python3
"""
Launch HTCondor job for deduplication.

Usage:
    python launch_dedup.py --jsonl_path /path/to/jsonl
"""

import os
import argparse
from pathlib import Path
import htcondor2 as htcondor

JOB_BID = 15
REPO_ROOT = Path(__file__).resolve().parents[3]
NEWS_LOG_BASE = Path(os.getenv("FSIM_NEWS_LOG_BASE", str(REPO_ROOT / "logs" / "news")))

def launch_dedup_job(
        jsonl_path: str,
        num_workers: int = 16,
        job_memory: int = None,
        job_cpus: int = 48,
        job_bid: int = JOB_BID,
):
    """Launch HTCondor job to deduplicate JSONL files."""
    
    LOG_PATH = str(NEWS_LOG_BASE / "dedup")
    
    log_dir = Path(LOG_PATH)
    os.makedirs(log_dir, exist_ok=True)
    
    cluster_job_log = str(log_dir / "$(Cluster).$(Process)")
    
    executable = str(REPO_ROOT / "data" / "news" / "condor" / "run_dedup.sh")
    
    if job_memory is None:
        job_memory = job_cpus * 16
    
    args_str = f"--jsonl_path {jsonl_path} --num_workers {num_workers}"

    job_settings = {
        "executable": executable,
        "arguments": args_str,
        "output": f"{cluster_job_log}.out",
        "error": f"{cluster_job_log}.err",
        "log": f"{cluster_job_log}.log",
        
        "request_cpus": str(job_cpus),
        "request_memory": f"{job_memory}GB",
        "request_disk": "100GB",
        
        "jobprio": str(job_bid - 1000),
        "notify_user": "shashwat.goel@tuebingen.mpg.de",
        "notification": "error",
    }

    job_description = htcondor.Submit(job_settings)
    schedd = htcondor.Schedd()
    submit_result = schedd.submit(job_description)

    print(f"Launched deduplication job: cluster-ID={submit_result.cluster()}")
    print(f"  JSONL path: {jsonl_path}")
    print(f"  Workers: {num_workers}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch deduplication job")
    
    parser.add_argument('--jsonl_path', type=str, required=True,
                       help="Directory containing JSONL files")
    parser.add_argument('--num_workers', type=int, default=16,
                       help="Number of parallel workers")
    parser.add_argument('--job_memory', type=int, default=None,
                       help="Job memory in GB")
    parser.add_argument('--job_cpus', type=int, default=48,
                       help="Number of CPUs to request")
    parser.add_argument('--bid', type=int, default=JOB_BID,
                       help="HTCondor job bid")
    
    args = parser.parse_args()
    
    launch_dedup_job(
        jsonl_path=args.jsonl_path,
        num_workers=args.num_workers,
        job_memory=args.job_memory,
        job_cpus=args.job_cpus,
        job_bid=args.bid,
    )
