"""
Cheap end-to-end check of the minimalHarness sandbox isolation pipeline.

Stands up two loopback HTTP servers (one allowed, one denied), starts the
egress proxy and the MCP relay via MinimalHarnessAgent, runs probes inside a
bwrap sandbox built from the agent's own _build_bwrap_cmd, and asserts:

  Network isolation (network_isolation=True):
    - allow:    curl through the proxy to the allowed loopback target -> 200
    - deny:     curl through the proxy to the denied loopback target  -> 403
    - isolated: curl bypassing the proxy fails (sandbox netns has no network)

  Search-DB isolation (sandbox=True):
    - search_db_hidden_fs: a host path configured as search_db is NOT visible
                           from inside the sandbox (`test -e` returns nonzero).
    - search_db_hidden_py: Python from inside the sandbox confirms the path
                           does not exist.
    - mcp_socket_visible:  the per-agent MCP relay unix socket IS visible at
                           the same realpath inside the sandbox.

No model API keys, no internet, no LLM costs.

Usage:
  python scripts/test_sandbox_egress.py
"""
import http.server
import os
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agents.minimalHarnessAgent.agent import MinimalHarnessAgent, MinimalHarnessConfig


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"hello\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a, **kw):
        return  # silence


class _ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def _serve(port=0):
    """Start a loopback HTTP server. Pass port=0 to let the OS pick a free port.
    Returns (httpd, port)."""
    httpd = _ReusableTCPServer(("127.0.0.1", port), _Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, httpd.server_address[1]


def _run_inside(agent, curl_args, label, use_proxy=True):
    """Run curl inside the bwrap sandbox.

    use_proxy=True forces the proxy with curl -x (overriding NO_PROXY).
    use_proxy=False uses --noproxy '*' to verify direct connections fail.
    """
    proxy_flag = (["-x", f"http://127.0.0.1:{agent._egress_proxy_port}"]
                  if use_proxy else ["--noproxy", "*"])
    cmd = agent._build_bwrap_cmd([
        "curl", "-sS", "-o", "/dev/null",
        "-w", "%{http_code}",
        "--max-time", "5",
        *proxy_flag,
        *curl_args,
    ])
    env = os.environ.copy()
    env.update(agent._egress_env_overrides())
    if use_proxy:
        # The default NO_PROXY=127.0.0.1 (production setting for the embedding
        # raw-forward path) makes curl skip the proxy for loopback targets.
        # Clear it for this test since our "allowed/denied" upstreams are
        # 127.0.0.1 stand-ins for real provider hosts.
        env["NO_PROXY"] = ""
        env["no_proxy"] = ""
    print(f"[{label}] launching: bwrap ... {' '.join(proxy_flag + list(curl_args))}")
    res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=30)
    print(f"[{label}] rc={res.returncode} stdout={res.stdout!r} "
          f"stderr_tail={res.stderr.strip()[-180:]!r}")
    return res


def _run_cmd_inside(agent, inner_cmd, label):
    """Run an arbitrary command inside the bwrap sandbox and return CompletedProcess."""
    cmd = agent._build_bwrap_cmd(list(inner_cmd))
    print(f"[{label}] launching: bwrap ... {' '.join(inner_cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    print(f"[{label}] rc={res.returncode} stdout={res.stdout!r} "
          f"stderr_tail={res.stderr.strip()[-180:]!r}")
    return res


def main():
    work = Path(tempfile.mkdtemp(prefix="egress_test_"))
    print(f"work dir: {work}")

    # Plant a marker file outside any path the sandbox should expose. Use it
    # as the agent's search_db so we can verify the bwrap mount layout
    # actually hides it from inside.
    fake_search_db = Path(tempfile.mkdtemp(prefix="fake_search_db_"))
    (fake_search_db / "MARKER").write_text("should-not-be-readable\n")
    print(f"fake search_db: {fake_search_db}")

    s1, allowed_port = _serve()
    s2, denied_port = _serve()
    print(f"allowed loopback: 127.0.0.1:{allowed_port}; "
          f"denied loopback: 127.0.0.1:{denied_port}")
    time.sleep(0.1)

    cfg = MinimalHarnessConfig(
        model="ignored",
        sandbox=True,
        network_isolation=True,
        egress_allowlist=[f"127.0.0.1:{allowed_port}"],
        search_db=str(fake_search_db),
        harness_backend="claude_code",
        claude_code_path="claude",  # may not resolve; binding skipped silently
    )
    agent = MinimalHarnessAgent(
        agent_id="net_iso_test",
        config=cfg,
        search_tool=None,
        agent_dir=str(work),
        articles_base="",
    )
    agent.workspace.mkdir(parents=True, exist_ok=True)
    agent._start_egress_proxy()
    agent._start_mcp_relay()

    results = []
    try:
        allow = _run_inside(
            agent, [f"http://127.0.0.1:{allowed_port}/"], "allow")
        results.append(("allow", allow.stdout.strip() == "200",
                        f"http_code={allow.stdout!r}, rc={allow.returncode}"))

        deny = _run_inside(
            agent, [f"http://127.0.0.1:{denied_port}/"], "deny")
        results.append(("deny", deny.stdout.strip() == "403",
                        f"http_code={deny.stdout!r}, rc={deny.returncode}"))

        direct = _run_inside(
            agent,
            [f"http://127.0.0.1:{allowed_port}/"],
            "isolated",
            use_proxy=False,
        )
        # No upstream reachable in sandbox netns -> curl exits nonzero.
        results.append(("isolated",
                        direct.returncode != 0 and direct.stdout.strip() != "200",
                        f"rc={direct.returncode}, stdout={direct.stdout!r}"))

        # search_db must NOT be visible inside the sandbox. The host MCP
        # relay holds the lancedb handle; the harness only reaches search
        # via the date-capped search_news MCP tool.
        marker = str(fake_search_db / "MARKER")
        fs_check = _run_cmd_inside(agent, ["test", "-e", marker],
                                   "search_db_hidden_fs")
        results.append(("search_db_hidden_fs", fs_check.returncode != 0,
                        f"test -e {marker} rc={fs_check.returncode}"))

        py_check = _run_cmd_inside(
            agent,
            ["python3", "-c", f"import os,sys; sys.exit(0 if not os.path.exists({marker!r}) else 1)"],
            "search_db_hidden_py",
        )
        results.append(("search_db_hidden_py", py_check.returncode == 0,
                        f"python exists check rc={py_check.returncode}"))

        # Positive check: the MCP relay socket SHOULD be visible at its
        # host realpath so `socat - UNIX-CONNECT:<sock>` works in-sandbox.
        sock = str(agent._mcp_relay_sock)
        sock_check = _run_cmd_inside(agent, ["test", "-S", sock],
                                     "mcp_socket_visible")
        results.append(("mcp_socket_visible", sock_check.returncode == 0,
                        f"test -S {sock} rc={sock_check.returncode}"))
    finally:
        agent._stop_egress_proxy()
        agent._stop_mcp_relay()
        s1.shutdown(); s2.shutdown()
        shutil.rmtree(fake_search_db, ignore_errors=True)

    print("\n=== RESULTS ===")
    width = max(len(n) for n, _, _ in results)
    for name, ok, detail in results:
        marker = "PASS" if ok else "FAIL"
        print(f"  [{marker}] {name.ljust(width)}  {detail}")
    sys.exit(0 if all(ok for _, ok, _ in results) else 1)


if __name__ == "__main__":
    main()
