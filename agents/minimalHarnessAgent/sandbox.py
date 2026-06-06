"""Filesystem sandbox and network egress helpers for MinimalHarnessAgent."""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List

from futuresim_agents.minimalHarnessAgent.config import DEFAULT_EGRESS_ALLOWLIST
from futuresim_agents.minimalHarnessAgent.sandbox_common import (
    LOOPBACK_HOST,
    egress_bridge_command,
    proxy_env,
    real_path,
    resolve_socat_cmd,
    terminate_process,
)

logger = logging.getLogger(__name__)


def _resolve_sandbox_socat_cmd() -> str:
    return resolve_socat_cmd()


class SandboxHelpers:
    """bwrap sandbox and optional allowlisted network egress support."""

    def _start_egress_proxy(self) -> None:
        """Start the idempotent host allowlist proxy on a per-agent unix socket.

        The socket dir uses local scratch (_CONDOR_SCRATCH_DIR or /tmp), because
        unix sockets are unreliable on shared FS (Lustre/NFS); bwrap binds that
        same realpath inside the sandbox.
        """
        if self._egress_proc is not None and self._egress_proc.poll() is None:
            return
        scratch = os.path.realpath(os.environ.get("_CONDOR_SCRATCH_DIR") or tempfile.gettempdir())
        # Hash the run output dir so parallel sims with the same agent_id but
        # different sim_name/timestamp get distinct unix-socket paths.
        run_tag = hashlib.sha1(
            str(Path(self._internal_dir).resolve()).encode()
        ).hexdigest()[:8]
        d = Path(scratch) / f"egress_{self.agent_id}_{run_tag}"
        d.mkdir(parents=True, exist_ok=True)
        os.chmod(d, 0o700)
        self._egress_dir = d
        self._egress_proxy_sock = d / "proxy.sock"

        # Detect the host loopback embedding server and raw-forward it through
        # the same port inside the sandbox.
        host_embed_url = self._detect_embedding_server()
        raw_forwards: List[str] = []
        if host_embed_url:
            self._egress_embed_port = int(host_embed_url.rsplit(":", 1)[1])
            self._egress_embed_sock = d / "embed.sock"
            raw_forwards.append(f"{self._egress_embed_sock}={LOOPBACK_HOST}:{self._egress_embed_port}")

        allow = list(self.config.egress_allowlist) or list(DEFAULT_EGRESS_ALLOWLIST)

        proxy_script = str(Path(__file__).resolve().parent / "egress_proxy.py")
        repo_root = str(Path(__file__).resolve().parents[2])
        venv_python = os.path.join(repo_root, ".venv", "bin", "python")
        python_cmd = venv_python if os.path.exists(venv_python) else sys.executable

        cmd = [python_cmd, proxy_script,
               "--proxy-socket", str(self._egress_proxy_sock)]
        for a in allow:
            cmd.extend(["--allow", a])
        for rf in raw_forwards:
            cmd.extend(["--raw-forward", rf])

        # The proxy writes "ready\n" after both servers are listening.
        r_fd, w_fd = os.pipe()
        cmd.extend(["--ready-fd", str(w_fd)])

        log_path = self._internal_dir / "egress_proxy.log"
        log_f = open(log_path, "a")
        logger.info("Starting egress proxy: %s", " ".join(cmd))
        self._egress_proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_f,
            stderr=log_f,
            pass_fds=(w_fd,),
        )
        os.close(w_fd)
        # Block until ready, process death, or a 5s timeout.
        ready = b""
        try:
            os.set_blocking(r_fd, False)
            deadline = time.time() + 5.0
            while time.time() < deadline:
                try:
                    chunk = os.read(r_fd, 16)
                    if chunk:
                        ready += chunk
                        if b"ready" in ready:
                            break
                except BlockingIOError:
                    pass
                if self._egress_proc.poll() is not None:
                    break
                time.sleep(0.05)
        finally:
            os.close(r_fd)
        if b"ready" not in ready:
            raise RuntimeError(
                f"egress proxy failed to become ready (rc={self._egress_proc.poll()}); "
                f"see {log_path}"
            )

    def _stop_egress_proxy(self) -> None:
        terminate_process(self._egress_proc)
        self._egress_proc = None

    def _wrap_for_egress_bridge(self, cmd: List[str]) -> List[str]:
        """Start socat TCP-loopback -> host-unix bridges, then exec the harness.

        Only used with network_isolation=True.
        """
        raw_bridges: list[tuple[Path, int]] = []
        if self._egress_embed_sock and self._egress_embed_port:
            raw_bridges.append((self._egress_embed_sock, int(self._egress_embed_port)))
        return egress_bridge_command(
            cmd,
            proxy_sock=str(self._egress_proxy_sock),
            proxy_port=int(self._egress_proxy_port),
            raw_bridges=raw_bridges,
            socat_cmd=_resolve_sandbox_socat_cmd(),
        )

    def _egress_env_overrides(self) -> Dict[str, str]:
        """Proxy env vars that route SDK clients through the host proxy bridge."""
        # Keep loopback local: the embedding client already bypasses HTTPS_PROXY,
        # and the embedding server reaches the host via the same-port raw bridge.
        return proxy_env(int(self._egress_proxy_port))

    def _build_bwrap_cmd(self, inner_cmd: List[str]) -> List[str]:
        """Wrap `inner_cmd` in bwrap, exposing only the harness needs.

        Exposed: workspace rw (cwd, articles/, memory/, market.csv, model
        outputs); _internal_dir rw (MCP signals, state.json,
        system_prompt.md); mcp_relay_dir rw (per-agent host MCP/retrieval
        socket); harness install ro plus backend state dirs rw (codex
        CODEX_HOME when set, else ~/.codex; Claude ~/.claude, ~/.claude.json, ~/.cache/claude*; opencode
        ~/.cache|.local|.config/opencode plus XDG_DATA_HOME SQLite scratch);
        OS dirs ro (/usr, /etc, /lib*, /bin, /sbin); stripped /dev; tmpfs /tmp;
        and /proc private by default or host_ro via sandbox_proc_mode.

        Hidden: the repo (MCP is host-side and harnesses are self-contained;
        binding it exposes analysis/reports, ground-truth tables, logs/sims,
        prior state and YAML configs/answers); .venv/sys.base_prefix
        (no repo Python runs inside; /usr/bin/python3 is enough); search_db,
        lancedb, and embedding_model (only host MCP `search_news` is reachable,
        with date-capping); articles_base (only date-gated workspace hardlinks
        are visible); FSIM_DATASET_PATH (HF labels); sibling sim dirs,
        daily_metrics.csv, test_daily_metrics.csv, actions.jsonl, matcher.jsonl
        in the parent output dir; and /home outside explicit harness dirs.

        Bind realpaths and recreate /home when it is a symlink so aliased paths
        resolve only to separately bound targets. Network is shared by default
        for OpenRouter and local vLLM; network_isolation uses a private net plus
        the host allowlist proxy/bridge.
        """
        workspace_real = real_path(self.workspace)
        internal_real = real_path(self._internal_dir)

        args = [
            "bwrap",
            "--unshare-net" if self.config.network_isolation else "--share-net",
            "--die-with-parent",
            "--unshare-ipc",
            "--unshare-uts",
            "--dev", "/dev",
            "--tmpfs", "/tmp",
            "--ro-bind", "/usr", "/usr",
            "--ro-bind", "/etc", "/etc",
            "--ro-bind-try", "/lib", "/lib",
            "--ro-bind-try", "/lib64", "/lib64",
            "--ro-bind-try", "/bin", "/bin",
            "--ro-bind-try", "/sbin", "/sbin",
        ]
        if self.config.sandbox_proc_mode == "new":
            args.extend(["--unshare-pid", "--proc", "/proc"])
        else:
            args.extend(["--ro-bind", "/proc", "/proc"])

        # Recreate aliases before binds create tmpfs DEST dirs.
        for alias in ("/home", "/fast"):
            if os.path.islink(alias):
                target = os.readlink(alias)
                if not os.path.isabs(target):
                    target = os.path.normpath(os.path.join(os.path.dirname(alias), target))
                # Normalize readlink's occasional trailing slash.
                target = target.rstrip("/") or "/"
                args.extend(["--symlink", target, alias])

        # Bind workspace and internal dir rw at realpaths to avoid alias clashes.
        args.extend(["--bind", workspace_real, workspace_real])
        args.extend(["--bind", internal_real, internal_real])

        # Articles are hardlinked from articles_base; ro-bind prevents sandbox
        # writes from corrupting sources while host-side driver writes bypass bwrap.
        articles_dst = os.path.join(workspace_real, "articles")
        if os.path.exists(articles_dst):
            args.extend(["--ro-bind", articles_dst, articles_dst])

        # Repo, search_db, and embedding model stay hidden; host-side MCP is the
        # only retrieval path, so date-capping cannot be bypassed by direct reads.

        # Backend owns install tree (ro) and runtime-state (rw) bind choices.
        self._add_sandbox_harness_install_binds(args)

        home = os.path.expanduser("~")
        for sub in self._sandbox_harness_home_subdirs():
            p = real_path(os.path.join(home, sub))
            if os.path.exists(p):
                args.extend(["--bind-try", p, p])

        # Per-agent XDG_DATA_HOME scratch for opencode's SQLite session DB.
        data_home = getattr(self, "_opencode_data_home", None)
        if data_home:
            dh = real_path(data_home)
            if os.path.exists(dh):
                args.extend(["--bind", dh, dh])

        # Bind per-agent egress socket dir at realpath for sandbox UNIX-CONNECT.
        if self.config.network_isolation and self._egress_dir:
            ed = real_path(self._egress_dir)
            args.extend(["--bind", ed, ed])

        # Bind MCP relay socket dir so `socat - UNIX-CONNECT:<sock>` reaches host MCP.
        if self._mcp_relay_dir:
            md = real_path(self._mcp_relay_dir)
            if os.path.exists(md):
                args.extend(["--bind", md, md])

        # chdir to the workspace realpath, matching the bind.
        args.extend(["--chdir", workspace_real])
        args.append("--")
        # With network isolation, start TCP->Unix bridges before exec'ing.
        if self.config.network_isolation:
            inner_cmd = self._wrap_for_egress_bridge(list(inner_cmd))
        args.extend(inner_cmd)
        return args

    def _maybe_sandbox(self, cmd: List[str]) -> List[str]:
        """Return cmd wrapped in bwrap when sandbox is enabled, else unchanged."""
        if self.config.sandbox:
            return self._build_bwrap_cmd(cmd)
        return cmd
