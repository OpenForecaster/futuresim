#!/usr/bin/env python3
"""Submit the HF upload job to HTCondor."""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit HF upload-large-folder as an HTCondor job")
    parser.add_argument("--repo_id", default="shash42/forecast-news")
    parser.add_argument(
        "--local_path",
        default="/lustre/fast/fast/sgoel/forecasting/news/deduped_articles/data",
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--include", default="**/*.parquet")
    parser.add_argument("--exclude", default="")
    parser.add_argument("--progress_secs", type=int, default=60)
    parser.add_argument(
        "--stage_root",
        default="",
        help=(
            "Root directory for per-job staging copies. "
            "If unset, run_hf_upload.sh defaults to $_CONDOR_SCRATCH_DIR/hf_upload_staging "
            "inside the execute node."
        ),
    )
    parser.add_argument(
        "--stage_path",
        default="",
        help="Optional explicit staging directory path.",
    )
    parser.add_argument(
        "--no_verify_remote",
        action="store_true",
        help="Skip post-upload remote file verification.",
    )
    parser.add_argument(
        "--keep_stage",
        action="store_true",
        help="Keep staged copy after successful upload (default deletes it).",
    )
    parser.add_argument("--cpus", type=int, default=16)
    parser.add_argument("--memory", type=int, default=64, help="GB")
    parser.add_argument("--disk", type=int, default=200, help="GB")
    parser.add_argument("--bid", type=int, default=15)
    parser.add_argument(
        "--log_dir",
        default="/is/cluster/fast/sgoel/logs/forecasting-sim/hf_upload",
    )
    args = parser.parse_args()

    if args.cpus < args.num_workers:
        print(f"Raising CPUs from {args.cpus} to {args.num_workers} to match --num_workers.")
        args.cpus = args.num_workers

    if " " in args.repo_id or " " in args.local_path:
        raise ValueError("repo_id/local_path cannot contain spaces in this submit helper.")

    script_path = "/home/sgoel/forecast-sim/mpi_scripts/hf_upload/run_hf_upload.sh"
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    env_parts = [
        f"HF_UPLOAD_NUM_WORKERS={args.num_workers}",
        f"HF_UPLOAD_INCLUDE_GLOB={args.include}",
        f"HF_UPLOAD_PROGRESS_SECS={args.progress_secs}",
        f"HF_UPLOAD_VERIFY_REMOTE={0 if args.no_verify_remote else 1}",
        f"HF_UPLOAD_DELETE_STAGE_ON_SUCCESS={0 if args.keep_stage else 1}",
    ]
    if args.stage_root:
        env_parts.append(f"HF_UPLOAD_STAGE_ROOT={args.stage_root}")
    if args.stage_path:
        env_parts.append(f"HF_UPLOAD_STAGE_PATH={args.stage_path}")
    if args.exclude:
        env_parts.append(f"HF_UPLOAD_EXCLUDE_GLOB={args.exclude}")

    env = " ".join(env_parts).replace("\\", "\\\\").replace('"', '\\"')

    sub = f"""executable = {script_path}
arguments = {args.repo_id} {args.local_path}
output = {log_dir}/$(Cluster).$(Process).out
error = {log_dir}/$(Cluster).$(Process).err
log = {log_dir}/$(Cluster).$(Process).log
request_cpus = {args.cpus}
request_memory = {args.memory}GB
request_disk = {args.disk}GB
request_gpus = 0
jobprio = {args.bid - 1000}
getenv = true
environment = "{env}"
queue 1
"""

    with tempfile.NamedTemporaryFile("w", suffix=".sub", delete=False) as tf:
        tf.write(sub)
        sub_path = Path(tf.name)

    proc = subprocess.run(
        ["condor_submit_bid", str(args.bid), str(sub_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    combined = (out + "\n" + err).strip()

    if proc.returncode != 0:
        print("condor_submit_bid failed.")
        if out:
            print("stdout:")
            print(out)
        if err:
            print("stderr:")
            print(err)
        print(f"submit file kept for debugging: {sub_path}")
        sys.exit(proc.returncode)

    match = re.search(r"cluster\s+(\d+)", combined, flags=re.IGNORECASE)
    if match:
        cluster_id = match.group(1)
        print(f"Submitted cluster: {cluster_id}")
        print(f"Logs: {args.log_dir}/{cluster_id}.0.out")
    else:
        print(combined)

    sub_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
