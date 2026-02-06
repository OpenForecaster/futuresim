#!/usr/bin/env python3
"""
Launch HTCondor job for Parquet conversion.

Usage:
    python launch_parquet_conversion.py \
        --input-dirs /path/to/deduped1 /path/to/deduped2 \
        --output-dir /path/to/output
"""

import os
import argparse
from pathlib import Path
import htcondor

JOB_BID = 15

def launch_parquet_conversion_job(
        input_dirs: list,
        output_dir: str,
        workers: int = 32,
        batch_size: int = 128,
        job_memory: int = None,
        job_cpus: int = 48,
        job_bid: int = JOB_BID,
):
    """Launch HTCondor job to convert JSONL to Parquet."""
    
    LOG_PATH = "/fast/sgoel/logs/forecasting-sim/news/parquet_conversion"
    
    log_dir = Path(LOG_PATH)
    os.makedirs(log_dir, exist_ok=True)
    
    cluster_job_log = str(log_dir / "$(Cluster).$(Process)")
    
    executable = '/home/sgoel/forecast-sim/data/news/condor/run_parquet_conversion.sh'
    
    if job_memory is None:
        job_memory = job_cpus * 16
    
    # Build arguments
    input_dirs_str = " ".join(input_dirs)
    args_str = f"--input-dirs {input_dirs_str} --output-dir {output_dir} --workers {workers} --batch-size {batch_size}"

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

    print(f"Launched Parquet conversion job: cluster-ID={submit_result.cluster()}")
    print(f"  Input dirs: {input_dirs}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Parquet conversion job")
    
    parser.add_argument('--input-dirs', nargs='+', required=True,
                       help="Input directories with deduped JSONL files")
    parser.add_argument('--output-dir', type=str, required=True,
                       help="Output directory for Parquet files")
    parser.add_argument('--workers', type=int, default=32,
                       help="Number of parallel workers")
    parser.add_argument('--batch-size', type=int, default=128,
                       help="Batch size for processing")
    parser.add_argument('--job_memory', type=int, default=None,
                       help="Job memory in GB")
    parser.add_argument('--job_cpus', type=int, default=48,
                       help="Number of CPUs to request")
    parser.add_argument('--bid', type=int, default=JOB_BID,
                       help="HTCondor job bid")
    
    args = parser.parse_args()
    
    launch_parquet_conversion_job(
        input_dirs=args.input_dirs,
        output_dir=args.output_dir,
        workers=args.workers,
        batch_size=args.batch_size,
        job_memory=args.job_memory,
        job_cpus=args.job_cpus,
        job_bid=args.bid,
    )
