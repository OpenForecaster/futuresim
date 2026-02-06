#!/usr/bin/env python3
"""
Launch HTCondor job for JSONL conversion.

Usage:
    python launch_jsonl_conversion.py --json_dir /path/to/articles --output_dir /path/to/jsonl
"""

import os
import argparse
from pathlib import Path
import htcondor

JOB_BID = 15

def launch_jsonl_conversion_job(
        json_dir: str,
        output_dir: str = None,
        workers: int = None,
        verify: float = 0.1,
        delete: bool = False,
        job_memory: int = None,
        job_cpus: int = 48,
        job_bid: int = JOB_BID,
):
    """Launch HTCondor job to convert JSON files to JSONL."""
    
    LOG_PATH = "/fast/sgoel/logs/forecasting-sim/news/jsonl_conversion"
    
    log_dir = Path(LOG_PATH)
    os.makedirs(log_dir, exist_ok=True)
    
    cluster_job_log = str(log_dir / "$(Cluster).$(Process)")
    
    executable = '/home/sgoel/forecast-sim/data/news/condor/run_jsonl_conversion.sh'
    
    if workers is None:
        workers = job_cpus
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(json_dir), "jsonl")
    
    if job_memory is None:
        job_memory = job_cpus * 16
    
    # Build arguments
    args_str = f"{json_dir} {output_dir} --workers {workers} --verify {verify}"
    if delete:
        args_str += " --delete"

    job_settings = {
        "executable": executable,
        "arguments": args_str,
        "output": f"{cluster_job_log}.out",
        "error": f"{cluster_job_log}.err",
        "log": f"{cluster_job_log}.log",
        
        "request_cpus": str(job_cpus),
        "request_memory": f"{job_memory}GB",
        "request_disk": f"{job_memory * 2}GB",
        
        "jobprio": str(job_bid - 1000),
        "notify_user": "shashwat.goel@tuebingen.mpg.de",
        "notification": "error",
    }

    job_description = htcondor.Submit(job_settings)
    schedd = htcondor.Schedd()
    submit_result = schedd.submit(job_description)

    print(f"Launched JSONL conversion job: cluster-ID={submit_result.cluster()}")
    print(f"  Source: {json_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Workers: {workers}, Verify: {verify*100}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch JSONL conversion job")
    
    parser.add_argument('--json_dir', type=str, required=True,
                       help="Directory containing JSON files to convert")
    parser.add_argument('--output_dir', type=str, default=None,
                       help="Output directory for JSONL files")
    parser.add_argument('--workers', type=int, default=None,
                       help="Number of parallel workers")
    parser.add_argument('--verify', type=float, default=0.1,
                       help="Fraction of documents to verify")
    parser.add_argument('--delete', action='store_true',
                       help="Delete JSON files after conversion")
    parser.add_argument('--job_memory', type=int, default=None,
                       help="Job memory in GB")
    parser.add_argument('--job_cpus', type=int, default=48,
                       help="Number of CPUs to request")
    parser.add_argument('--bid', type=int, default=JOB_BID,
                       help="HTCondor job bid")
    
    args = parser.parse_args()
    
    launch_jsonl_conversion_job(
        json_dir=args.json_dir,
        output_dir=args.output_dir,
        workers=args.workers,
        verify=args.verify,
        delete=args.delete,
        job_memory=args.job_memory,
        job_cpus=args.job_cpus,
        job_bid=args.bid,
    )
