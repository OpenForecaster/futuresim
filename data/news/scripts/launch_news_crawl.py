#!/usr/bin/env python3
"""
Launch HTCondor job for CCNews article extraction.

This is a long-running job that downloads and extracts articles from CommonCrawl News.
It can take days to weeks depending on WARC file availability.

Usage:
    # Download Aug 2025 to Jan 2026:
    python launch_news_crawl.py --start-date 2025-08-01 --end-date 2026-01-31 --bid 15
    
    # Download just October 2025:
    python launch_news_crawl.py --start-date 2025-10-01 --end-date 2025-10-31
"""

import os
import argparse
from pathlib import Path
from datetime import datetime

import htcondor2 as htcondor

JOB_BID = 15


def launch_news_crawl_job(
        start_date: str,
        end_date: str = None,
        job_memory: int = 32,
        job_cpus: int = 4,
        job_bid: int = JOB_BID,
):
    """Launch HTCondor job to download news from CommonCrawl."""
    
    LOG_PATH = "/fast/sgoel/logs/forecasting-sim/news/crawl"
    
    log_dir = Path(LOG_PATH)
    os.makedirs(log_dir, exist_ok=True)
    
    cluster_job_log = str(log_dir / "$(Cluster).$(Process)")
    
    executable = '/home/sgoel/forecast-sim/data/news/condor/run_news_crawl.sh'
    
    # Pass dates as arguments to the shell script
    args_str = f"{start_date}"
    if end_date:
        args_str += f" {end_date}"

        job_settings = {
            "executable": executable,
            "arguments": args_str,
            "output": f"{cluster_job_log}.out",
            "error": f"{cluster_job_log}.err",
            "log": f"{cluster_job_log}.log",
        
            "request_cpus": str(job_cpus),
            "request_memory": f"{job_memory}GB",
        "request_disk": "500GB",  # Need disk for WARCs
        
            "jobprio": str(job_bid - 1000),
            "notify_user": "shashwat.goel@tuebingen.mpg.de",
            "notification": "error",
        }

        job_description = htcondor.Submit(job_settings)
        schedd = htcondor.Schedd()
        submit_result = schedd.submit(job_description)

    print(f"Launched CCNews crawl job: cluster-ID={submit_result.cluster()}")
    print(f"  Date range: {start_date} to {end_date or 'now'}")
    print(f"  This is a LONG-RUNNING job (days/weeks)")
    print(f"  Monitor with: condor_q {submit_result.cluster()}")
    print(f"  Logs: {LOG_PATH}/{submit_result.cluster()}.out")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch CCNews crawl job")
    
    parser.add_argument('--start-date', type=str, required=True,
                       help="Start date (YYYY-MM-DD), e.g., 2025-08-01")
    parser.add_argument('--end-date', type=str, default=None,
                       help="End date (YYYY-MM-DD), e.g., 2026-01-31. If not set, downloads up to now.")
    parser.add_argument('--job_memory', type=int, default=32,
                       help="Job memory in GB")
    parser.add_argument('--job_cpus', type=int, default=4,
                       help="Number of CPUs to request")
    parser.add_argument('--bid', type=int, default=JOB_BID,
                       help="HTCondor job bid")
    
    args = parser.parse_args()
    
    # Validate dates
    try:
        datetime.strptime(args.start_date, "%Y-%m-%d")
        if args.end_date:
            datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError as e:
        print(f"Error: Invalid date format. Use YYYY-MM-DD. {e}")
        exit(1)
    
    launch_news_crawl_job(
        start_date=args.start_date,
        end_date=args.end_date,
        job_memory=args.job_memory,
        job_cpus=args.job_cpus,
        job_bid=args.bid,
    )
