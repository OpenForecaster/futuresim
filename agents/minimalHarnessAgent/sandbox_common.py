"""Shared process and sandbox plumbing for MinimalHarness runners."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, Sequence

LOOPBACK_HOST = "127.0.0.1"
LOCAL_NO_PROXY = "127.0.0.1,localhost"


def real_path(path: str | os.PathLike[str]) -> str:
    return os.path.realpath(os.path.expanduser(str(path)))


def resolve_socat_cmd(*, prefer_path: bool = False) -> str:
    if prefer_path:
        found = shutil.which("socat")
        if found:
            return found
    for candidate in ("/usr/bin/socat", "/bin/socat"):
        if os.path.exists(candidate):
            return candidate
    return "socat"


def terminate_process(proc: subprocess.Popen[Any] | None, *, timeout: float = 5.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()


def proxy_env(port: int) -> dict[str, str]:
    proxy_url = f"http://{LOOPBACK_HOST}:{int(port)}"
    return {
        "HTTPS_PROXY": proxy_url,
        "https_proxy": proxy_url,
        "HTTP_PROXY": proxy_url,
        "http_proxy": proxy_url,
        "ALL_PROXY": proxy_url,
        "all_proxy": proxy_url,
        "NO_PROXY": LOCAL_NO_PROXY,
        "no_proxy": LOCAL_NO_PROXY,
    }


def egress_bridge_command(
    inner_cmd: Sequence[str],
    *,
    proxy_sock: str | os.PathLike[str],
    proxy_port: int,
    raw_bridges: Iterable[tuple[str | os.PathLike[str], int]] = (),
    socat_cmd: str | None = None,
) -> list[str]:
    socat = shlex.quote(socat_cmd or resolve_socat_cmd())
    bridge = (
        f"{socat} TCP-LISTEN:{int(proxy_port)},bind={LOOPBACK_HOST},fork,reuseaddr "
        f"UNIX-CONNECT:{shlex.quote(str(proxy_sock))} 2>/dev/null & "
    )
    for raw_sock, raw_port in raw_bridges:
        bridge += (
            f"{socat} TCP-LISTEN:{int(raw_port)},bind={LOOPBACK_HOST},fork,reuseaddr "
            f"UNIX-CONNECT:{shlex.quote(str(raw_sock))} 2>/dev/null & "
        )
    bridge += 'sleep 0.3; exec "$@"'
    return ["/bin/sh", "-c", bridge, "egress-bridge", *inner_cmd]


def write_mcp_connector(path: Path, sock: str | os.PathLike[str]) -> None:
    path.write_text(
        "#!/bin/sh\n"
        f"sock={shlex.quote(str(sock))}\n"
        'if [ ! -S "$sock" ]; then\n'
        '  echo "MCP relay socket missing: $sock" >&2\n'
        "  exit 127\n"
        "fi\n"
        "for socat_cmd in /usr/bin/socat /bin/socat socat; do\n"
        '  if command -v "$socat_cmd" >/dev/null 2>&1; then\n'
        '    exec "$socat_cmd" - "UNIX-CONNECT:$sock"\n'
        "  fi\n"
        "done\n"
        'echo "socat not found in sandbox" >&2\n'
        "exit 127\n"
    )
    path.chmod(0o755)
