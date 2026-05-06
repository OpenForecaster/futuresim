#!/usr/bin/env python3
"""Monitor forecast-sim HTCondor jobs and resume interrupted runs.

This is intentionally conservative:
- It starts by tracking jobs that are currently running/idle under the owner.
- It ignores jobs that were already held or removed before the monitor started.
- If a tracked job stops before the simulation is complete, it first submits a
  cluster resume. Local tmux fallback is only used when cluster submission fails
  and nvidia-smi shows a genuinely idle GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - repo venv should have pyyaml
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[1]
STATUS = {
    1: "IDLE",
    2: "RUNNING",
    3: "REMOVED",
    4: "COMPLETED",
    5: "HELD",
    6: "TRANSFERRING_OUTPUT",
    7: "SUSPENDED",
}
EXCLUDE_ARGS_RE = None


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def run(cmd: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def log_line(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(f"{now()} {message}\n")


def load_state(path: Path) -> dict[str, Any]:
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return {"created_at": now(), "tracked": {}, "completed": {}, "events": []}


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    tmp.replace(path)


def condor_q(owner: str) -> dict[str, dict[str, Any]]:
    proc = run(["condor_q", owner, "-json"], timeout=90)
    if proc.returncode != 0 or not proc.stdout.strip():
        return {}
    jobs = json.loads(proc.stdout)
    out = {}
    for job in jobs:
        cid = job.get("ClusterId")
        if cid is not None:
            out[str(cid)] = job
    return out


def condor_history(job_id: str) -> dict[str, Any]:
    proc = run(["condor_history", str(job_id), "-json", "-limit", "1"], timeout=90)
    if proc.returncode != 0 or not proc.stdout.strip():
        return {}
    jobs = json.loads(proc.stdout)
    return jobs[0] if jobs else {}


def is_forecast_sim_job(job: dict[str, Any]) -> bool:
    cmd = str(job.get("Cmd") or "")
    args = str(job.get("Args") or "")
    return (
        cmd.endswith("run_sim.sh")
        and args.endswith("config.yaml")
        and ("forecast-sim" in cmd or "forecast-sim" in args or "forecasting/current_sim" in args)
    )


def status_name(value: Any) -> str:
    try:
        return STATUS.get(int(value), str(value))
    except Exception:
        return str(value)


def read_text(path: Path, max_bytes: int = 2_000_000) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    if len(data) > max_bytes:
        data = data[-max_bytes:]
    return data.decode(errors="replace")


def log_dir_for_config(config_path: str) -> Path:
    return Path(config_path).expanduser().resolve().parent


def stdout_paths(config_path: str, job_id: str) -> list[Path]:
    d = log_dir_for_config(config_path)
    paths = []
    preferred = d / f"{job_id}.out"
    if preferred.exists():
        paths.append(preferred)
    paths.extend(sorted(d.glob("*.out"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True))
    dedup = []
    seen = set()
    for p in paths:
        if p not in seen:
            dedup.append(p)
            seen.add(p)
    return dedup


def find_output_dir(config_path: str, job_id: str) -> str:
    for out_path in stdout_paths(config_path, job_id):
        text = read_text(out_path)
        matches = re.findall(
            r"(?:Output directory|Restart output directory|Resuming in directory):\s*(\S+)",
            text,
        )
        if matches:
            return matches[-1]

    parent = log_dir_for_config(config_path)
    if (parent / "daily_metrics.csv").exists() or (parent / "actions.jsonl").exists():
        return str(parent)
    return ""


def config_end_date(config_path: str) -> str:
    if yaml is None:
        return ""
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return ""
    return str(cfg.get("end_date") or "")


def last_metric_date(output_dir: str) -> str:
    metrics = Path(output_dir) / "daily_metrics.csv"
    if not metrics.exists() or metrics.stat().st_size == 0:
        return ""
    lines = [line.strip() for line in read_text(metrics).splitlines() if line.strip()]
    if len(lines) <= 1:
        return ""
    return lines[-1].split(",", 1)[0]


def simulation_complete(config_path: str, job_id: str, output_dir: str) -> bool:
    stdout = "\n".join(read_text(p) for p in stdout_paths(config_path, job_id))
    if "Simulation complete!" in stdout:
        return True

    end_date = config_end_date(config_path)
    last_date = last_metric_date(output_dir) if output_dir else ""
    return bool(end_date and last_date and last_date >= end_date)


def parse_cluster_ids(text: str) -> list[str]:
    m = re.search(r"Cluster IDs:\s*\[([^\]]*)\]", text)
    if m:
        return re.findall(r"\d+", m.group(1))
    return re.findall(r"cluster\s+(\d+)", text, flags=re.IGNORECASE)


def submit_cluster_resume(config_path: str, output_dir: str) -> tuple[bool, list[str], str]:
    cmd = [
        sys.executable,
        "mpi_scripts/run_sim/submit_sim.py",
        "--config",
        config_path,
        "--resume",
        output_dir,
    ]
    proc = run(cmd, timeout=180)
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    ids = parse_cluster_ids(combined)
    return proc.returncode == 0 and bool(ids), ids, combined[-4000:]


def free_gpu_index() -> str:
    proc = run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout=15,
    )
    if proc.returncode != 0:
        return ""
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx, mem_used, util = parts[0], int(parts[1]), int(parts[2])
        except ValueError:
            continue
        if mem_used < 1000 and util <= 5:
            return idx
    return ""


def launch_local_tmux(config_path: str, output_dir: str, old_job: str, log_path: Path) -> tuple[bool, str]:
    gpu = free_gpu_index()
    if not gpu:
        return False, "no free local GPU found"

    session = f"fsim_resume_{old_job}_{datetime.now().strftime('%m%d_%H%M%S')}"
    local_log = log_path.parent / f"{session}.log"
    inner = (
        f"cd {shlex.quote(str(REPO_ROOT))} && "
        "source .venv/bin/activate && "
        f"CUDA_VISIBLE_DEVICES={shlex.quote(gpu)} "
        f"python -u scripts/test_basic_agent.py --config {shlex.quote(config_path)} "
        f"--resume {shlex.quote(output_dir)} "
        f"> {shlex.quote(str(local_log))} 2>&1"
    )
    proc = run(["tmux", "new-session", "-d", "-s", session, "bash", "-lc", inner], timeout=30)
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout or "tmux launch failed").strip()
    return True, session


def track_active_jobs(state: dict[str, Any], jobs: dict[str, dict[str, Any]], log_path: Path) -> None:
    for jid, job in jobs.items():
        if not is_forecast_sim_job(job):
            continue
        if EXCLUDE_ARGS_RE and EXCLUDE_ARGS_RE.search(str(job.get("Args") or "")):
            continue
        if int(job.get("JobStatus", 0) or 0) not in (1, 2, 6, 7):
            continue
        if jid in state["completed"]:
            continue
        if jid not in state["tracked"]:
            state["tracked"][jid] = {
                "config_path": str(job.get("Args") or ""),
                "first_seen": now(),
                "last_seen": now(),
                "last_status": status_name(job.get("JobStatus")),
                "output_dir": "",
                "relaunch_of": "",
                "handled_stop": False,
            }
            log_line(log_path, f"TRACK job={jid} status={status_name(job.get('JobStatus'))} config={job.get('Args')}")


def handle_stopped_job(
    job_id: str,
    rec: dict[str, Any],
    reason: str,
    log_path: Path,
    state: dict[str, Any],
) -> None:
    if rec.get("handled_stop"):
        return

    config_path = rec.get("config_path") or ""
    output_dir = rec.get("output_dir") or find_output_dir(config_path, job_id)
    rec["output_dir"] = output_dir
    rec["stop_reason"] = reason

    if output_dir and simulation_complete(config_path, job_id, output_dir):
        state["completed"][job_id] = {**rec, "completed_at": now()}
        rec["handled_stop"] = True
        log_line(log_path, f"COMPLETE job={job_id} reason={reason} output={output_dir}")
        return

    if not config_path or not output_dir:
        rec["handled_stop"] = True
        log_line(log_path, f"STOPPED_NO_RESUME job={job_id} reason={reason} config={config_path} output={output_dir}")
        return

    ok, new_ids, cluster_output = submit_cluster_resume(config_path, output_dir)
    if ok:
        rec["handled_stop"] = True
        rec["resumed_at"] = now()
        rec["resume_method"] = "cluster"
        rec["resume_job_ids"] = new_ids
        for new_id in new_ids:
            state["tracked"][new_id] = {
                "config_path": config_path,
                "first_seen": now(),
                "last_seen": now(),
                "last_status": "SUBMITTED",
                "output_dir": output_dir,
                "relaunch_of": job_id,
                "handled_stop": False,
            }
        log_line(log_path, f"RESUME_CLUSTER old_job={job_id} new_jobs={new_ids} output={output_dir}")
        return

    local_ok, local_msg = launch_local_tmux(config_path, output_dir, job_id, log_path)
    rec["handled_stop"] = True
    rec["resumed_at"] = now()
    rec["cluster_resume_error"] = cluster_output
    if local_ok:
        rec["resume_method"] = "local_tmux"
        rec["local_tmux_session"] = local_msg
        log_line(log_path, f"RESUME_LOCAL old_job={job_id} session={local_msg} output={output_dir}")
    else:
        rec["resume_method"] = "failed"
        rec["local_resume_error"] = local_msg
        log_line(log_path, f"RESUME_FAILED old_job={job_id} output={output_dir} local={local_msg}")


def check_once(args: argparse.Namespace, state: dict[str, Any]) -> None:
    log_path = Path(args.log)
    jobs = condor_q(args.owner)
    if EXCLUDE_ARGS_RE:
        jobs = {
            jid: job
            for jid, job in jobs.items()
            if not EXCLUDE_ARGS_RE.search(str(job.get("Args") or ""))
        }
    track_active_jobs(state, jobs, log_path)

    active_summary = []
    for job_id, rec in list(state["tracked"].items()):
        if job_id in state["completed"]:
            continue
        if rec.get("handled_stop"):
            continue
        job = jobs.get(job_id)
        if job:
            status = int(job.get("JobStatus", 0) or 0)
            rec["last_seen"] = now()
            rec["last_status"] = status_name(status)
            rec["remote_host"] = str(job.get("RemoteHost") or "")
            if not rec.get("output_dir"):
                rec["output_dir"] = find_output_dir(rec.get("config_path", ""), job_id)
            active_summary.append(f"{job_id}:{status_name(status)}")
            if status == 5:
                reason = str(job.get("HoldReason") or "held")
                log_line(log_path, f"HELD job={job_id} reason={reason}")
                handle_stopped_job(job_id, rec, f"held: {reason}", log_path, state)
            elif status in (3, 4):
                handle_stopped_job(job_id, rec, f"condor_status={status_name(status)}", log_path, state)
            continue

        hist = condor_history(job_id)
        if hist:
            status = status_name(hist.get("JobStatus") or hist.get("LastJobStatus") or "missing")
            exit_status = hist.get("ExitStatus")
            exit_signal = hist.get("ExitBySignal")
            remove_reason = hist.get("RemoveReason") or hist.get("HoldReason") or ""
            reason = f"missing/history status={status} exit={exit_status} signal={exit_signal} reason={remove_reason}"
        else:
            reason = "missing/no_history"
        log_line(log_path, f"STOP_DETECTED job={job_id} {reason}")
        handle_stopped_job(job_id, rec, reason, log_path, state)

    log_line(log_path, "CHECK " + (" ".join(active_summary) if active_summary else "no-active-tracked-jobs"))
    save_state(Path(args.state), state)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", default=os.environ.get("USER", "nchandak"))
    parser.add_argument("--interval", type=int, default=900)
    parser.add_argument("--state", default=str(REPO_ROOT / "logs/monitors/condor_run_monitor_state.json"))
    parser.add_argument("--log", default=str(REPO_ROOT / "logs/monitors/condor_run_monitor.log"))
    parser.add_argument("--exclude-args-regex", default="")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    global EXCLUDE_ARGS_RE
    EXCLUDE_ARGS_RE = re.compile(args.exclude_args_regex) if args.exclude_args_regex else None

    state_path = Path(args.state)
    state = load_state(state_path)
    log_line(Path(args.log), f"START owner={args.owner} interval={args.interval}s state={state_path}")
    while True:
        try:
            check_once(args, state)
        except Exception as exc:
            log_line(Path(args.log), f"ERROR {type(exc).__name__}: {exc}")
            save_state(state_path, state)
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
