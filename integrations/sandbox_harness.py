"""Nested CLI sandbox launcher used by hosted Futuresim adapters."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from futuresim_agents.minimalHarnessAgent.config import DEFAULT_EGRESS_ALLOWLIST
from futuresim_agents.minimalHarnessAgent.sandbox_common import (
    egress_bridge_command,
    proxy_env,
    real_path,
    resolve_socat_cmd,
    terminate_process,
    write_mcp_connector,
)


def _start_mcp_relay(
    spec: dict[str, Any],
    runtime_dir: Path,
    private_dir: Path,
) -> subprocess.Popen[Any]:
    sock = runtime_dir / "mcp.sock"
    sock.unlink(missing_ok=True)

    mcp_config = private_dir / "mcp_server_config.json"
    mcp_config.write_text(json.dumps(spec["mcp_server_config"], indent=2))

    launcher = private_dir / "launch_mcp.sh"
    mcp_entry_args = list(spec.get("mcp_entry_args") or ["-m", "futuresim_agents.minimalHarnessAgent.mcp_server"])
    launcher.write_text(
        "#!/bin/sh\nexec "
        + shlex.join([spec["mcp_command"], *mcp_entry_args, "--config-file", str(mcp_config)])
        + "\n"
    )
    launcher.chmod(0o755)

    connector = runtime_dir / "connect_mcp.sh"
    write_mcp_connector(connector, sock)

    log = open(private_dir / "mcp_relay.log", "a")
    proc = subprocess.Popen(
        [resolve_socat_cmd(), f"UNIX-LISTEN:{sock},fork", f"EXEC:{launcher}"],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=log,
    )
    deadline = time.time() + 5
    while time.time() < deadline:
        if sock.exists():
            return proc
        if proc.poll() is not None:
            raise RuntimeError(f"MCP relay exited before listening; see {private_dir / 'mcp_relay.log'}")
        time.sleep(0.05)
    raise RuntimeError(f"MCP relay socket did not appear: {sock}")


def _start_egress_proxy(
    spec: dict[str, Any],
    runtime_dir: Path,
    private_dir: Path,
) -> subprocess.Popen[Any]:
    proxy_sock = runtime_dir / "egress_proxy.sock"
    proxy_sock.unlink(missing_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "futuresim_agents.minimalHarnessAgent.egress_proxy",
        "--proxy-socket",
        str(proxy_sock),
    ]
    allowlist = list(spec.get("egress_allowlist") or DEFAULT_EGRESS_ALLOWLIST)
    for item in allowlist:
        cmd.extend(["--allow", str(item)])

    read_fd, write_fd = os.pipe()
    cmd.extend(["--ready-fd", str(write_fd)])
    log = open(private_dir / "egress_proxy.log", "a")
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=log,
        pass_fds=(write_fd,),
    )
    os.close(write_fd)
    try:
        os.set_blocking(read_fd, False)
        deadline = time.time() + 5
        ready = b""
        while time.time() < deadline:
            try:
                chunk = os.read(read_fd, 16)
            except BlockingIOError:
                chunk = b""
            if chunk:
                ready += chunk
                if b"ready" in ready:
                    return proc
            if proc.poll() is not None:
                break
            time.sleep(0.05)
    finally:
        os.close(read_fd)
    raise RuntimeError(f"Egress proxy failed to become ready; see {private_dir / 'egress_proxy.log'}")


def _codex_cmd(spec: dict[str, Any], runtime_dir: Path) -> list[str]:
    mcp_command = str(runtime_dir / "connect_mcp.sh")
    proxy_url = proxy_env(int(spec.get("egress_proxy_port") or 18765))["HTTPS_PROXY"]
    common = [
        "-m",
        str(spec["model"]),
        "-c",
        f'model_reasoning_effort="{spec.get("reasoning_effort", "xhigh")}"',
        "-c",
        'web_search="disabled"',
        "-c",
        f'mcp_servers.forecast.command="{mcp_command}"',
        "-c",
        "mcp_servers.forecast.args=[]",
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",
        "--json",
    ]
    if spec.get("network_isolation"):
        common.extend(
            [
                "-c",
                f'network.proxy_url="{proxy_url}"',
            ]
        )

    codex_path = real_path(str(spec.get("codex_path") or "codex"))
    prompt = str(spec.get("prompt") or "")
    thread_id = str(spec.get("codex_thread_id") or "")
    if spec.get("codex_resume") and thread_id:
        cmd = [codex_path, "exec", "resume", thread_id, *common, prompt]
    else:
        cmd = [codex_path, "exec", *common, "-C", str(spec["workspace"]), prompt]
    cmd.extend(str(x) for x in (spec.get("extra_flags") or []))
    return cmd


def _claude_cmd(spec: dict[str, Any], runtime_dir: Path) -> list[str]:
    mcp_config = runtime_dir / "mcp_config.json"
    mcp_config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "forecast": {
                        "command": str(runtime_dir / "connect_mcp.sh"),
                        "args": [],
                    }
                }
            },
            indent=2,
        )
    )
    cmd = [real_path(str(spec.get("claude_code_path") or "claude"))]
    session_id = str(spec.get("claude_session_id") or "")
    if spec.get("claude_code_resume") and session_id:
        cmd.extend(["--resume", session_id])
    disallowed = (
        "WebSearch,WebFetch,Write,Edit,MultiEdit,NotebookEdit"
        if spec.get("prompt_mode") == "no_memory"
        else "WebSearch,WebFetch"
    )
    cmd.extend(
        [
            "-p",
            str(spec.get("prompt") or ""),
            "--verbose",
            "--effort",
            "max",
            "--model",
            str(spec["model"]),
            "--output-format",
            "stream-json",
            "--dangerously-skip-permissions",
            "--mcp-config",
            str(mcp_config),
            "--strict-mcp-config",
            "--disallowedTools",
            disallowed,
            "--system-prompt-file",
            str(Path(spec["internal_dir"]) / "system_prompt.md"),
            "--add-dir",
            str(spec["workspace"]),
        ]
    )
    if spec.get("max_budget_usd") is not None:
        cmd.extend(["--max-budget-usd", str(spec["max_budget_usd"])])
    cmd.extend(str(x) for x in (spec.get("extra_flags") or []))
    return cmd


def _bind_if_exists(args: list[str], flag: str, path: str, dest: str | None = None) -> None:
    real = real_path(path)
    if os.path.exists(real):
        args.extend([flag, real, dest or real])


def _add_home_binds(args: list[str], spec: dict[str, Any]) -> None:
    backend = spec.get("backend")
    if backend == "codex":
        codex_home = os.environ.get("CODEX_HOME")
        if codex_home:
            _bind_if_exists(args, "--bind", codex_home)
        else:
            _bind_if_exists(args, "--bind-try", "~/.codex")
        codex_bin = real_path(str(spec.get("codex_path") or "codex"))
        if os.path.exists(codex_bin):
            install = Path(codex_bin).parent.parent
            _bind_if_exists(args, "--ro-bind", str(install))
    elif backend == "claude_code":
        _bind_if_exists(args, "--bind-try", "~/.claude")
        _bind_if_exists(args, "--bind-try", "~/.claude.json")
        cache = Path(real_path("~/.cache"))
        if cache.exists():
            for item in cache.glob("claude*"):
                _bind_if_exists(args, "--bind-try", str(item))
        claude_bin = real_path(str(spec.get("claude_code_path") or "claude"))
        if os.path.exists(claude_bin):
            install = Path(claude_bin).parent.parent
            _bind_if_exists(args, "--ro-bind", str(install))


def _bwrap_cmd(spec: dict[str, Any], inner: list[str], runtime_dir: Path) -> list[str]:
    workspace = real_path(str(spec["workspace"]))
    internal = real_path(str(spec["internal_dir"]))
    runtime = real_path(str(runtime_dir))

    args = [
        "bwrap",
        "--unshare-net" if spec.get("network_isolation") else "--share-net",
        "--die-with-parent",
        "--unshare-ipc",
        "--unshare-uts",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind",
        "/etc",
        "/etc",
        "--ro-bind-try",
        "/lib",
        "/lib",
        "--ro-bind-try",
        "/lib64",
        "/lib64",
        "--ro-bind-try",
        "/bin",
        "/bin",
        "--ro-bind-try",
        "/sbin",
        "/sbin",
    ]
    proc_mode = spec.get("sandbox_proc_mode", "none")
    if proc_mode == "new":
        args.extend(["--unshare-pid", "--proc", "/proc"])
    elif proc_mode == "host_ro":
        args.extend(["--ro-bind", "/proc", "/proc"])
    else:
        args.extend(["--dir", "/proc", "--dir", "/proc/self", "--symlink", real_path(inner[0]), "/proc/self/exe"])

    for alias in ("/home", "/fast"):
        if os.path.islink(alias):
            target = os.readlink(alias)
            if not os.path.isabs(target):
                target = os.path.normpath(os.path.join(os.path.dirname(alias), target))
            args.extend(["--symlink", target.rstrip("/") or "/", alias])

    args.extend(["--bind", workspace, workspace])
    args.extend(["--bind", internal, internal])
    args.extend(["--bind", runtime, runtime])
    articles = os.path.join(workspace, "articles")
    if os.path.exists(articles):
        args.extend(["--ro-bind", articles, articles])

    _add_home_binds(args, spec)

    args.extend(["--chdir", workspace, "--"])
    if spec.get("network_isolation"):
        inner = egress_bridge_command(
            inner,
            proxy_sock=runtime_dir / "egress_proxy.sock",
            proxy_port=int(spec.get("egress_proxy_port") or 18765),
            socat_cmd=resolve_socat_cmd(),
        )
    args.extend(inner)
    return args


def _run_streaming(cmd: list[str], env: dict[str, str]) -> int:
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    assert proc.stdout is not None
    for chunk in iter(lambda: proc.stdout.readline(), b""):
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
    return int(proc.wait())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()

    spec_path = Path(args.spec)
    spec = json.loads(spec_path.read_text())
    runtime_dir = Path(spec["host_runtime_dir"])
    private_dir = Path(str(runtime_dir) + ".private")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(runtime_dir, 0o700)
    os.chmod(private_dir, 0o700)
    try:
        spec_path.unlink(missing_ok=True)
    except OSError:
        pass

    relay_proc: subprocess.Popen[Any] | None = None
    egress_proc: subprocess.Popen[Any] | None = None
    try:
        relay_proc = _start_mcp_relay(spec, runtime_dir, private_dir)
        if spec.get("network_isolation"):
            egress_proc = _start_egress_proxy(spec, runtime_dir, private_dir)
        if spec.get("backend") == "codex":
            inner = _codex_cmd(spec, runtime_dir)
        elif spec.get("backend") == "claude_code":
            inner = _claude_cmd(spec, runtime_dir)
        else:
            raise ValueError(f"Unsupported hosted harness backend: {spec.get('backend')!r}")
        cmd = _bwrap_cmd(spec, inner, runtime_dir)
        env = os.environ.copy()
        if spec.get("network_isolation"):
            env.update(proxy_env(int(spec.get("egress_proxy_port") or 18765)))

        def _handle_signal(signum: int, _frame: Any) -> None:
            terminate_process(relay_proc)
            terminate_process(egress_proc)
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)
        raise SystemExit(_run_streaming(cmd, env))
    finally:
        terminate_process(relay_proc)
        terminate_process(egress_proc)
        shutil.rmtree(private_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
