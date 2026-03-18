#!/usr/bin/env python3
"""
Generic HTCondor job submission script for syntheticQA.

Submits any Python script in the syntheticQA directory to the cluster with
configurable GPU/memory resources. Everything after "--" (or any unknown args)
is forwarded to the script.

Usage:
    # Submit generate_qa.py on 8 H100 GPUs
    python syntheticQA/submit_job.py --script generate_qa.py --gpu_type h100 --num_gpus 8 -- --num_q 10 --num_article 100

    # Submit eval_qa.py on 1 GPU
    python syntheticQA/submit_job.py --script eval_qa.py --num_gpus 1 -- --model /fast/nchandak/models/Qwen3-8B --questions /path/to/folder

    # Submit eval_with_answer.py on 8 GPUs with dry run
    python syntheticQA/submit_job.py --script eval_with_answer.py --num_gpus 8 --dry_run -- --model /fast/nchandak/models/gpt-oss-20b --questions /path

    # Submit prepare_distillation.py (no GPU needed)
    python syntheticQA/submit_job.py --script prepare_distillation.py --num_gpus 0 -- --input /path/to/folder
"""

import argparse
import os
import sys
from pathlib import Path

import htcondor2 as htcondor

# Defaults
DEFAULT_JOB_BID = 50
DEFAULT_MEMORY_PER_GPU = 64  # GB
DEFAULT_CPUS_PER_GPU = 8
DEFAULT_LOG_PATH = os.getenv(
    "FSIM_SYNTHETICQA_LOG_DIR",
    str(Path(__file__).resolve().parents[1] / "logs" / "syntheticqa"),
)

# GPU type configurations
GPU_CONFIGS = {
    "a100": {
        "min_memory": 45000,   # MB
        "max_memory": 89999,   # MB
        "cuda_capability": 8.0,
    },
    "h100": {
        "min_memory": 45000,   # MB
        "max_memory": 89999,   # MB
        "cuda_capability": 9.0,
    },
    "b100": {
        "min_memory": 90000,   # MB (90GB+)
        "max_memory": None,
        "cuda_capability": 9.0,
    },
}


def build_requirements(gpu_type: str) -> str:
    """Build HTCondor requirements string for GPU type."""
    config = GPU_CONFIGS[gpu_type]
    parts = []
    parts.append(f"(CUDACapability >= {config['cuda_capability']})")
    if config["min_memory"]:
        parts.append(f"(TARGET.CUDAGlobalMemoryMb >= {config['min_memory']})")
    if config["max_memory"]:
        parts.append(f"(TARGET.CUDAGlobalMemoryMb < {config['max_memory']})")
    return " && ".join(parts)


def submit_job(
    script: str,
    script_args: list,
    gpu_type: str,
    num_gpus: int,
    job_bid: int,
    memory_per_gpu: int,
    cpus_per_gpu: int,
    log_path: str,
    dry_run: bool,
):
    """Submit a Python script as an HTCondor job."""
    script_dir = Path(__file__).parent.resolve()
    executable = str(script_dir / "run_job.sh")

    # Logs directory
    log_dir = Path(log_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_prefix = str(log_dir / "$(Cluster).$(Process)")

    # Build the arguments string: script_name arg1 arg2 ...
    args_string = script
    if script_args:
        args_string += " " + " ".join(script_args)

    total_memory = memory_per_gpu * max(num_gpus, 1)
    total_cpus = max(cpus_per_gpu * max(num_gpus, 1), 32)

    job_settings = {
        "executable": executable,
        "arguments": args_string,
        "output": f"{log_prefix}.out",
        "error": f"{log_prefix}.err",
        "log": f"{log_prefix}.log",
        "request_cpus": str(total_cpus),
        "request_memory": f"{total_memory}GB",
        "request_disk": f"{total_memory}GB",
        "jobprio": str(job_bid - 1000),
        "notify_user": "nikhil.chandak@tuebingen.mpg.de",
        "notification": "error",
        "environment": "PYTHONUNBUFFERED=1",
    }

    if num_gpus > 0:
        job_settings["request_gpus"] = str(num_gpus)
        job_settings["requirements"] = build_requirements(gpu_type)

    print(f"Job submission:")
    print(f"  Script: {script}")
    print(f"  Script args: {' '.join(script_args) if script_args else '(none)'}")
    print(f"  GPU type: {gpu_type}, Num GPUs: {num_gpus}")
    print(f"  CPUs: {total_cpus}, Memory: {total_memory}GB")
    print()
    print("HTCondor settings:")
    for k, v in job_settings.items():
        print(f"  {k}: {v}")

    if dry_run:
        print("\n[Dry run] Job NOT submitted.")
        return -1

    job = htcondor.Submit(job_settings)
    schedd = htcondor.Schedd()
    result = schedd.submit(job, count=1)
    cluster_id = result.cluster()

    print(f"\nSubmitted! Cluster ID: {cluster_id}")
    print(f"Logs: {log_path}/{cluster_id}.0.{{out,err,log}}")
    return cluster_id


def main():
    parser = argparse.ArgumentParser(
        description="Submit syntheticQA jobs to HTCondor. "
                    "Everything after '--' is forwarded to the script.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate QA on 8 H100 GPUs
    python syntheticQA/submit_job.py --script generate_qa.py --gpu_type h100 --num_gpus 8 -- --num_q 10

    # Eval on 1 GPU
    python syntheticQA/submit_job.py --script eval_qa.py --num_gpus 1 -- --questions /path/to/folder

    # Eval with answer on 8 B100 GPUs
    python syntheticQA/submit_job.py --script eval_with_answer.py --gpu_type b100 --num_gpus 8 -- --questions /path

    # Prepare distillation (CPU only)
    python syntheticQA/submit_job.py --script prepare_distillation.py --num_gpus 0 -- --input /path
        """,
    )

    parser.add_argument(
        "--script", type=str, required=True,
        help="Python script to run (e.g., generate_qa.py, eval_qa.py)",
    )
    parser.add_argument(
        "--gpu_type", type=str, choices=["a100", "h100", "b100"], default="h100",
        help="GPU type: a100 (40-80GB), h100 (<90GB), b100 (>=90GB) (default: h100)",
    )
    parser.add_argument(
        "--num_gpus", type=int, default=8,
        help="Number of GPUs to request (default: 8, use 0 for CPU-only jobs)",
    )
    parser.add_argument(
        "--job_bid", type=int, default=DEFAULT_JOB_BID,
        help=f"Job priority bid (default: {DEFAULT_JOB_BID})",
    )
    parser.add_argument(
        "--memory_per_gpu", type=int, default=DEFAULT_MEMORY_PER_GPU,
        help=f"Memory per GPU in GB (default: {DEFAULT_MEMORY_PER_GPU})",
    )
    parser.add_argument(
        "--cpus_per_gpu", type=int, default=DEFAULT_CPUS_PER_GPU,
        help=f"CPUs per GPU (default: {DEFAULT_CPUS_PER_GPU})",
    )
    parser.add_argument(
        "--log_path", type=str, default=DEFAULT_LOG_PATH,
        help=f"Log directory path (default: {DEFAULT_LOG_PATH})",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print job settings without submitting",
    )

    args, script_args = parser.parse_known_args()

    # Remove leading "--" separator if present
    if script_args and script_args[0] == "--":
        script_args = script_args[1:]

    # Validate script exists
    script_dir = Path(__file__).parent
    script_path = script_dir / args.script
    if not script_path.exists():
        print(f"Error: Script not found: {script_path}")
        sys.exit(1)

    submit_job(
        script=args.script,
        script_args=script_args,
        gpu_type=args.gpu_type,
        num_gpus=args.num_gpus,
        job_bid=args.job_bid,
        memory_per_gpu=args.memory_per_gpu,
        cpus_per_gpu=args.cpus_per_gpu,
        log_path=args.log_path,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
