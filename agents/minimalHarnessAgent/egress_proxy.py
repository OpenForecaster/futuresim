"""
Egress allowlist proxy for the minimalHarness sandbox.

Provides two flavours of unix-socket-bound listeners:

  --proxy-socket: HTTP CONNECT + HTTP forward proxy. Validates the upstream
    target host:port against an allowlist before opening upstream. Used by
    SDK clients that honour HTTPS_PROXY (Anthropic, OpenAI, OpenRouter,
    DeepSeek, z.ai, etc.) and by curl/wget inside the sandbox.

  --raw-forward UNIX=HOST:PORT: raw TCP splice from a unix socket to a fixed
    host:port on the host. Used by the MCP server's _VLLMEmbeddingClient
    which explicitly bypasses HTTPS_PROXY (urllib ProxyHandler({})) and so
    needs a direct path to the host's loopback embedding server.

Designed to be paired with `bwrap --unshare-net` plus a small TCP→Unix
bridge inside the sandbox: SDK clients dial 127.0.0.1:PORT (the sandbox's
own loopback), the bridge forwards into the bind-mounted unix socket, and
this proxy enforces the allowlist on the host side. The sandbox never gets
direct network access, and DNS is resolved on the host (where it works).
"""
import argparse
import asyncio
import fnmatch
import logging
import os
import signal
from typing import List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger("egress_proxy")

HEADER_TIMEOUT_S = 30
UPSTREAM_TIMEOUT_S = 15
BUF = 65536
MAX_HEADER_BYTES = 1 << 20  # 1 MiB


def _parse_allow(spec: str) -> Tuple[str, int]:
    """'host:port' or 'host' (default 443)."""
    if ":" in spec:
        host, port = spec.rsplit(":", 1)
        return host.lower(), int(port)
    return spec.lower(), 443


def _allowed(host: str, port: int, allowlist: List[Tuple[str, int]]) -> bool:
    h = host.lower()
    for pat, ap in allowlist:
        if ap == port and fnmatch.fnmatchcase(h, pat):
            return True
    return False


def _is_loopback_host(host: str) -> bool:
    h = host.lower()
    return (h in ("localhost", "ip6-localhost", "ip6-loopback")
            or h == "::1"
            or h.startswith("127."))


async def _open_upstream(host: str, port: int,
                         upstream: Optional[Tuple[str, int]]):
    """Open a TCP stream to (host, port). If `upstream` is set, dial the upstream
    HTTP proxy and issue a CONNECT host:port; the returned reader/writer is the
    tunneled stream (TLS bytes flow through unchanged). Loopback targets always
    bypass the upstream proxy (the upstream cannot reach our loopback)."""
    if upstream is None or _is_loopback_host(host):
        return await asyncio.open_connection(
            host, port, happy_eyeballs_delay=0.25)
    uhost, uport = upstream
    ureader, uwriter = await asyncio.open_connection(
        uhost, uport, happy_eyeballs_delay=0.25)
    req = (f"CONNECT {host}:{port} HTTP/1.1\r\n"
           f"Host: {host}:{port}\r\n"
           f"Proxy-Connection: keep-alive\r\n\r\n").encode("latin-1")
    uwriter.write(req)
    await uwriter.drain()
    status_line = await ureader.readline()
    if not status_line.startswith(b"HTTP/"):
        raise OSError(f"upstream proxy bad status: {status_line!r}")
    parts = status_line.split(b" ", 2)
    if len(parts) < 2 or parts[1] != b"200":
        # drain any remaining headers for cleanliness
        while True:
            line = await ureader.readline()
            if line in (b"\r\n", b""):
                break
        raise OSError(f"upstream proxy refused CONNECT: {status_line.strip()!r}")
    while True:
        line = await ureader.readline()
        if line in (b"\r\n", b""):
            break
    return ureader, uwriter


async def _splice(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    try:
        while True:
            data = await reader.read(BUF)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def _proxy_handle(reader, writer, allowlist, upstream):
    target_label = "?"
    try:
        head = bytearray()
        while b"\r\n\r\n" not in head:
            chunk = await asyncio.wait_for(reader.read(BUF), HEADER_TIMEOUT_S)
            if not chunk:
                return
            head.extend(chunk)
            if len(head) > MAX_HEADER_BYTES:
                writer.write(b"HTTP/1.1 431 Request Header Fields Too Large\r\n\r\n")
                await writer.drain()
                return
        head_end = head.index(b"\r\n\r\n") + 4
        head_bytes = bytes(head[:head_end])
        already_buffered = bytes(head[head_end:])

        request_line, _, header_block = head_bytes.partition(b"\r\n")
        try:
            method, target, _ver = request_line.decode("latin-1").split(" ", 2)
        except ValueError:
            writer.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
            await writer.drain()
            return

        if method.upper() == "CONNECT":
            host, _, port_s = target.partition(":")
            port = int(port_s) if port_s else 443
            target_label = f"{host}:{port}"
            if not _allowed(host, port, allowlist):
                logger.warning("DENY CONNECT %s", target_label)
                writer.write(b"HTTP/1.1 403 Forbidden\r\nProxy-Agent: egress-proxy\r\nContent-Length: 8\r\n\r\nblocked\n")
                await writer.drain()
                return
            try:
                ureader, uwriter = await asyncio.wait_for(
                    _open_upstream(host, port, upstream), UPSTREAM_TIMEOUT_S)
            except (OSError, asyncio.TimeoutError) as e:
                logger.warning("UPSTREAM-FAIL CONNECT %s: %s", target_label, e)
                writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
                await writer.drain()
                return
            logger.info("ALLOW CONNECT %s", target_label)
            writer.write(b"HTTP/1.1 200 Connection Established\r\nProxy-Agent: egress-proxy\r\n\r\n")
            await writer.drain()
            if already_buffered:
                uwriter.write(already_buffered)
                await uwriter.drain()
            await asyncio.gather(_splice(reader, uwriter), _splice(ureader, writer))
            return

        # HTTP forward proxy: target must be absolute URI.
        if not target.lower().startswith("http://"):
            writer.write(b"HTTP/1.1 400 Bad Request\r\nProxy-Agent: egress-proxy\r\n\r\nrelative URI not allowed\n")
            await writer.drain()
            return
        rest = target[7:]
        slash = rest.find("/")
        host_port = rest if slash == -1 else rest[:slash]
        path = "/" if slash == -1 else rest[slash:]
        if ":" in host_port:
            host, port_s = host_port.rsplit(":", 1)
            port = int(port_s)
        else:
            host, port = host_port, 80
        target_label = f"{host}:{port}"
        if not _allowed(host, port, allowlist):
            logger.warning("DENY %s %s", method, target_label)
            writer.write(b"HTTP/1.1 403 Forbidden\r\nProxy-Agent: egress-proxy\r\nContent-Length: 8\r\n\r\nblocked\n")
            await writer.drain()
            return
        try:
            ureader, uwriter = await asyncio.wait_for(
                _open_upstream(host, port, upstream), UPSTREAM_TIMEOUT_S)
        except (OSError, asyncio.TimeoutError) as e:
            logger.warning("UPSTREAM-FAIL %s %s: %s", method, target_label, e)
            writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            await writer.drain()
            return
        logger.info("ALLOW %s %s%s", method, target_label, path)
        new_request = f"{method} {path} HTTP/1.1\r\n".encode("latin-1") + header_block + b"\r\n\r\n"
        uwriter.write(new_request + already_buffered)
        await uwriter.drain()
        await asyncio.gather(_splice(reader, uwriter), _splice(ureader, writer))
    except (asyncio.TimeoutError, ConnectionError):
        pass
    except Exception:
        logger.exception("proxy handler error (%s)", target_label)
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def _raw_handle(reader, writer, host, port):
    try:
        ureader, uwriter = await asyncio.wait_for(
            asyncio.open_connection(host, port), UPSTREAM_TIMEOUT_S)
    except (OSError, asyncio.TimeoutError) as e:
        logger.warning("RAW UPSTREAM-FAIL %s:%d: %s", host, port, e)
        writer.close()
        return
    logger.info("RAW connect -> %s:%d", host, port)
    await asyncio.gather(_splice(reader, uwriter), _splice(ureader, writer))
    try:
        writer.close()
    except Exception:
        pass


def _parse_upstream(spec: Optional[str]) -> Optional[Tuple[str, int]]:
    if not spec:
        return None
    if "://" not in spec:
        spec = "http://" + spec
    u = urlparse(spec)
    if u.scheme not in ("http", "https"):
        raise ValueError(f"unsupported upstream proxy scheme: {u.scheme}")
    if u.scheme == "https":
        raise ValueError("https upstream proxy not supported (TLS-to-proxy)")
    if not u.hostname:
        raise ValueError(f"bad upstream proxy: {spec!r}")
    return u.hostname, (u.port or 8080)


async def run(args):
    allowlist = [_parse_allow(x) for x in (args.allow or [])]
    upstream_spec = args.upstream_proxy or os.environ.get("EGRESS_UPSTREAM_PROXY") \
        or os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") \
        or os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    upstream = _parse_upstream(upstream_spec)
    servers = []

    paths = [args.proxy_socket] + [r.split("=", 1)[0] for r in (args.raw_forward or [])]
    for p in paths:
        if p and os.path.exists(p) and not os.path.isdir(p):
            os.unlink(p)

    proxy_server = await asyncio.start_unix_server(
        lambda r, w: _proxy_handle(r, w, allowlist, upstream),
        path=args.proxy_socket)
    os.chmod(args.proxy_socket, 0o660)
    logger.info("proxy socket: %s; upstream=%s; allow=%s",
                args.proxy_socket,
                f"{upstream[0]}:{upstream[1]}" if upstream else "DIRECT",
                [f"{h}:{p}" for h, p in allowlist])
    servers.append(proxy_server)

    for spec in args.raw_forward or []:
        sock_path, target = spec.split("=", 1)
        thost, tport_s = target.rsplit(":", 1)
        tport = int(tport_s)
        srv = await asyncio.start_unix_server(
            (lambda h, p: lambda r, w: _raw_handle(r, w, h, p))(thost, tport),
            path=sock_path)
        os.chmod(sock_path, 0o660)
        logger.info("raw forward: %s -> %s:%d", sock_path, thost, tport)
        servers.append(srv)

    stop = asyncio.get_event_loop().create_future()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            asyncio.get_event_loop().add_signal_handler(sig, stop.set_result, None)
        except NotImplementedError:
            pass
    if args.ready_fd is not None:
        os.write(args.ready_fd, b"ready\n")
        os.close(args.ready_fd)
    try:
        await stop
    finally:
        for srv in servers:
            srv.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy-socket", required=True)
    parser.add_argument("--allow", action="append", default=[],
                        help="Allowed host[:port] (repeatable). Glob ok in host.")
    parser.add_argument("--raw-forward", action="append", default=[],
                        help="UNIX_SOCKET=HOST:PORT (repeatable). Raw TCP splice.")
    parser.add_argument("--ready-fd", type=int, default=None,
                        help="If set, write 'ready\\n' and close once listening.")
    parser.add_argument("--upstream-proxy", default=None,
                        help="Optional upstream HTTP proxy (host:port or http://host:port). "
                             "Overrides EGRESS_UPSTREAM_PROXY / HTTPS_PROXY / HTTP_PROXY env. "
                             "Use to chain through cluster proxies that gate outbound traffic.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s egress_proxy: %(message)s",
        datefmt="%H:%M:%S",
    )
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
