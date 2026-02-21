#!/usr/bin/env python3
"""
SLURM job launcher for GPU embedding jobs (AISA path conventions).

Usage:
    # Single worker:
    python aisa_scripts/embed/submit_job_aisa.py --gpus 1 --memory 64 --num_workers 1

    # Array job with 8 parallel workers:
    python aisa_scripts/embed/submit_job_aisa.py --gpus 1 --memory 64 --num_workers 8
"""

import argparse
import subprocess
from pathlib import Path
from shutil import which


def submit_gpu_job(
    executable: str,
    log_dir: str,
    gpus: int = 1,
    cpus: int = 16,
    memory_gb: int = 64,
    disk_gb: int = 100,
    num_workers: int = 1,
    gpu_type: str | None = None,
    job_name: str = "embed_aisa",
    mail_user: str | None = None,
    mail_type: str | None = None,
    output_pattern: str | None = None,
    error_pattern: str | None = None,
    partition: str | None = None,
    account: str | None = None,
    qos: str | None = None,
    time_limit: str | None = None,
    constraint: str | None = None,
    dry_run: bool = False,
) -> str:
    """Submit GPU embedding job(s) to SLURM. Returns job ID."""
    if which("sbatch") is None:
        raise RuntimeError("sbatch not found in PATH. SLURM client is required.")

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    repo_root = str(Path(__file__).resolve().parents[2])
    if output_pattern is None:
        output_pattern = str(log_path / ("%A_%a.out" if num_workers > 1 else "%j.out"))
    if error_pattern is None:
        error_pattern = str(log_path / ("%A_%a.err" if num_workers > 1 else "%j.err"))

    cmd = [
        "sbatch",
        "--parsable",
        "--job-name",
        job_name,
        "--output",
        output_pattern,
        "--error",
        error_pattern,
        "--cpus-per-task",
        str(cpus),
        "--mem",
        f"{memory_gb}G",
        "--chdir",
        repo_root,
        "--export",
        f"ALL,AISA_REPO_DIR={repo_root}",
    ]

    if gpus > 0:
        gres = f"gpu:{gpus}"
        if gpu_type:
            gres = f"gpu:{gpu_type}:{gpus}"
        cmd += ["--gres", gres]
    if mail_user:
        cmd += ["--mail-user", mail_user]
    if mail_type:
        cmd += ["--mail-type", mail_type]
    if partition:
        cmd += ["--partition", partition]
    if account:
        cmd += ["--account", account]
    if qos:
        cmd += ["--qos", qos]
    if time_limit:
        cmd += ["--time", time_limit]
    if constraint:
        cmd += ["--constraint", constraint]

    if num_workers > 1:
        cmd += ["--array", f"0-{num_workers - 1}"]

    # Pass num_workers; worker id comes from SLURM_ARRAY_TASK_ID.
    cmd += [executable, str(num_workers)]

    if dry_run:
        print("Dry run: would submit command:")
        print("  " + " ".join(cmd))
        return "DRY_RUN"

    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        stdout = (e.stdout or "").strip()
        stderr = (e.stderr or "").strip()
        raise RuntimeError(
            "sbatch submission failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout: {stdout or '<empty>'}\n"
            f"stderr: {stderr or '<empty>'}"
        ) from e
    out = (proc.stdout or "").strip()
    job_id = out.splitlines()[0].split(";")[0].strip() if out else ""
    if not job_id:
        raise RuntimeError(f"Failed to parse SLURM job id. sbatch output:\n{proc.stdout}\n{proc.stderr}")
    return job_id


def main():
    parser = argparse.ArgumentParser(description="Submit GPU embedding job to SLURM")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs per job")
    parser.add_argument("--cpus", type=int, default=16, help="Number of CPUs per job")
    parser.add_argument("--memory", type=int, default=64, help="Memory in GB per job")
    parser.add_argument("--disk", type=int, default=100, help="Tmp disk in GB per job")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--gpu-type", type=str, default=None, help="GPU type for --gres (e.g. A100)")
    parser.add_argument("--job-name", type=str, default="embed_aisa", help="SLURM job name")
    parser.add_argument("--mail-user", type=str, default=None, help="Email for SLURM notifications")
    parser.add_argument("--mail-type", type=str, default=None, help="SLURM mail types (e.g. END,FAIL)")
    parser.add_argument(
        "--output-pattern",
        type=str,
        default=None,
        help="Custom sbatch --output pattern (default: %j.out or %A_%a.out in log dir)",
    )
    parser.add_argument(
        "--error-pattern",
        type=str,
        default=None,
        help="Custom sbatch --error pattern (default: %j.err or %A_%a.err in log dir)",
    )
    parser.add_argument("--partition", type=str, default=None, help="SLURM partition")
    parser.add_argument("--account", type=str, default=None, help="SLURM account")
    parser.add_argument("--qos", type=str, default=None, help="SLURM QoS")
    parser.add_argument("--time", type=str, default=None, help="SLURM time limit (e.g. 12:00:00)")
    parser.add_argument("--constraint", type=str, default=None, help="SLURM node constraint")
    parser.add_argument("--dry-run", action="store_true", help="Print sbatch command without submitting")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    executable = str(script_dir / "run_embed_aisa.sh")
    log_dir = str(Path(__file__).resolve().parent.parent.parent / "logs" / "embed")

    job_id = submit_gpu_job(
        executable=executable,
        log_dir=log_dir,
        gpus=args.gpus,
        cpus=args.cpus,
        memory_gb=args.memory,
        disk_gb=args.disk,
        num_workers=args.num_workers,
        gpu_type=args.gpu_type,
        job_name=args.job_name,
        mail_user=args.mail_user,
        mail_type=args.mail_type,
        output_pattern=args.output_pattern,
        error_pattern=args.error_pattern,
        partition=args.partition,
        account=args.account,
        qos=args.qos,
        time_limit=args.time,
        constraint=args.constraint,
        dry_run=args.dry_run,
    )

    if args.disk:
        print(f"Note: --disk={args.disk} is ignored by this SLURM submitter.")
    print(f"Submitted {args.num_workers} worker job(s) with SLURM job ID: {job_id}")
    print(f"Logs: {log_dir}/{job_id}_*.out")


if __name__ == "__main__":
    main()
