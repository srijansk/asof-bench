"""The egress proxy: the one route out of a system's environment (spec section 8).

A system runs on an internal network that has no route out; this proxy is the only container that
joins both that network and one with a route out. It opens a tunnel for ``CONNECT host:port
HTTP/1.x`` and for nothing else: the host is compared in lower case and the port exactly with the
allowlist, a host that is an IP address is refused even when it is the address an allowed name
resolves to, and any other method, including an absolute-form ``GET http://...``, is refused
before any connection is made. A tunnel carries bytes in both directions until either side closes
it or neither side has sent anything for the idle timeout.

Every request is logged as one JSON line on stdout, with the keys ``ts``, ``client``, ``method``,
``target``, ``allowed`` and ``reason``; nothing that passes through a tunnel is read or logged.

Configuration: ``EGRESS_ALLOW``, a comma-separated list of ``host:port`` entries naming hosts by
DNS name (default ``api.anthropic.com:443``). The proxy listens on 0.0.0.0:3128. It uses only the
standard library.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import re
import signal
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus

DEFAULT_ALLOW = "api.anthropic.com:443"
LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 3128
# The request line and header fields are read up to this many bytes, and refused beyond it.
HEAD_LIMIT = 16 * 1024
# Seconds after which a connection is closed: when its request is not complete this long after it
# opened, when the upstream connection is not made this long after it began, and when neither
# side of a tunnel has sent anything for this long.
IDLE_TIMEOUT = 300.0
CHUNK = 64 * 1024

ESTABLISHED = b"HTTP/1.1 200 Connection established\r\n\r\n"
# RFC 9110: a method is a token, and the request line is method, target and version, separated by
# single spaces. The target is taken as visible ASCII only, so that no other alphabet's digits or
# letters can stand for a port or a host.
REQUEST_LINE = re.compile(r"(?P<method>[!#$%&'*+.^_`|~0-9A-Za-z-]+) (?P<target>[\x21-\x7e]+) "
                          r"HTTP/1\.[0-9]")
HEADER_FIELD = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+:[\x20-\x7e\t]*")
PORT = re.compile(r"[0-9]{1,5}")
DNS_LABEL = re.compile(r"[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?")


@dataclass(frozen=True)
class Decision:
    """What the proxy does with one request: ``status`` 200 opens a tunnel to ``host`` and
    ``port``; any other status refuses the request with that status."""

    status: int
    reason: str
    method: str | None = None
    target: str | None = None
    host: str | None = None
    port: int | None = None


def split_authority(target: str) -> tuple[str, int]:
    """The host, in lower case, and the port of ``host:port``; ValueError if it is not one."""
    host, sep, port = target.rpartition(":")
    if not sep or not host or not PORT.fullmatch(port) or not 1 <= int(port) <= 65535:
        raise ValueError(f"{target!r} is not host:port with a port from 1 to 65535")
    return host.lower(), int(port)


def is_ip_literal(host: str) -> bool:
    """Whether ``host`` is an IPv4 or IPv6 address, bracketed or not."""
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def parse_allow(text: str) -> frozenset[str]:
    """The allowlist that ``text``, comma-separated ``host:port`` entries, names, each as
    ``host:port`` with the host in lower case. ValueError for an entry that is not host:port, a
    host that is not a DNS name (an IP address, or a name whose last label has no letter, which a
    resolver may read as an address), or a list with no entry."""
    entries = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        host, port = split_authority(item)
        labels = host.split(".")
        if (is_ip_literal(host) or not all(DNS_LABEL.fullmatch(label) for label in labels)
                or not re.search("[a-z]", labels[-1])):
            raise ValueError(f"EGRESS_ALLOW entry {item!r} does not name a host by its DNS name")
        entries.add(f"{host}:{port}")
    if not entries:
        raise ValueError("EGRESS_ALLOW names no host:port")
    return frozenset(entries)


def allowlist(environ: Mapping[str, str] | None = None) -> frozenset[str]:
    """The allowlist from ``EGRESS_ALLOW`` (default ``api.anthropic.com:443``)."""
    environ = os.environ if environ is None else environ
    return parse_allow(environ.get("EGRESS_ALLOW", DEFAULT_ALLOW))


def decide(head: bytes, allow: frozenset[str]) -> Decision:
    """The decision for a request whose request line and header fields are ``head``, up to and
    including the blank line that ends them. Nothing here touches the network."""
    lines = head.decode("latin-1").split("\r\n")
    match = REQUEST_LINE.fullmatch(lines[0])
    if match is None:
        return Decision(400, "malformed request line")
    method, target = match["method"], match["target"]
    if any(not HEADER_FIELD.fullmatch(line) for line in lines[1:-2]):
        return Decision(400, "malformed header field", method, target)
    if method != "CONNECT":
        return Decision(405, "method not allowed: only CONNECT opens a tunnel", method, target)
    try:
        host, port = split_authority(target)
    except ValueError:
        return Decision(400, "malformed request line: the target is not host:port", method,
                        target)
    if is_ip_literal(host):
        return Decision(403, "IP literal: the allowlist names hosts, not addresses", method,
                        target)
    if f"{host}:{port}" not in allow:
        return Decision(403, "not in the allowlist", method, target)
    return Decision(200, "allowlisted", method, target, host, port)


def log(client: str, method: str | None, target: str | None, allowed: bool, reason: str) -> None:
    """One JSON line on stdout for one request."""
    ts = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    record = {"ts": ts, "client": client, "method": method, "target": target,
              "allowed": allowed, "reason": reason}
    print(json.dumps(record), file=sys.stdout, flush=True)


def response(status: int) -> bytes:
    allow = "Allow: CONNECT\r\n" if status == 405 else ""
    return (f"HTTP/1.1 {status} {HTTPStatus(status).phrase}\r\n{allow}Content-Length: 0\r\n"
            "Connection: close\r\n\r\n").encode("ascii")


def peer(writer: asyncio.StreamWriter) -> str:
    address = writer.get_extra_info("peername")
    if not address:
        return "unknown"
    host, port = address[0], address[1]
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


async def serve(allow: frozenset[str], host: str = LISTEN_HOST, port: int = LISTEN_PORT, *,
                idle_timeout: float = IDLE_TIMEOUT) -> asyncio.Server:
    """Starts the proxy on ``host`` and ``port`` (0 picks a free port) and returns its server."""

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await _handle(reader, writer, allow, idle_timeout)

    return await asyncio.start_server(handle, host, port, limit=HEAD_LIMIT)


async def _refuse(writer: asyncio.StreamWriter, client: str, decision: Decision,
                  allowed: bool = False) -> None:
    log(client, decision.method, decision.target, allowed, decision.reason)
    writer.write(response(decision.status))
    await writer.drain()


async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
                  allow: frozenset[str], idle_timeout: float) -> None:
    client = peer(writer)
    upstream: asyncio.StreamWriter | None = None
    try:
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), idle_timeout)
        except asyncio.IncompleteReadError as error:
            if error.partial:  # a connection closed before it sent a byte made no request
                await _refuse(writer, client, Decision(400, "incomplete request"))
            return
        except asyncio.LimitOverrunError:
            await _refuse(writer, client, Decision(431, f"request head over {HEAD_LIMIT} bytes"))
            return
        except TimeoutError:
            await _refuse(writer, client, Decision(408, "no complete request before the timeout"))
            return
        decision = decide(head, allow)
        if decision.status != 200:
            await _refuse(writer, client, decision)
            return
        try:
            upstream_reader, upstream = await asyncio.wait_for(
                asyncio.open_connection(decision.host, decision.port), idle_timeout)
        except TimeoutError:
            await _refuse(writer, client, Decision(504, "upstream connection timed out",
                                                   decision.method, decision.target), True)
            return
        except OSError as error:
            why = error.strerror or type(error).__name__
            await _refuse(writer, client, Decision(502, f"upstream unreachable: {why}",
                                                   decision.method, decision.target), True)
            return
        log(client, decision.method, decision.target, True, decision.reason)
        writer.write(ESTABLISHED)
        await writer.drain()
        await _relay(reader, writer, upstream_reader, upstream, idle_timeout)
    except OSError:
        pass  # a side that went away ends the connection; there is nothing left to answer
    finally:
        writer.close()
        if upstream is not None:
            upstream.close()


async def _relay(client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter,
                 upstream_reader: asyncio.StreamReader, upstream_writer: asyncio.StreamWriter,
                 idle_timeout: float) -> None:
    """Copies bytes both ways until either side closes or the tunnel is idle for
    ``idle_timeout`` seconds."""
    loop = asyncio.get_running_loop()
    last = [loop.time()]

    async def pump(source: asyncio.StreamReader, sink: asyncio.StreamWriter) -> None:
        while data := await source.read(CHUNK):
            last[0] = loop.time()
            sink.write(data)
            await sink.drain()

    async def watchdog() -> None:
        while (remaining := last[0] + idle_timeout - loop.time()) > 0:
            await asyncio.sleep(remaining)

    tasks = [asyncio.create_task(pump(client_reader, upstream_writer)),
             asyncio.create_task(pump(upstream_reader, client_writer)),
             asyncio.create_task(watchdog())]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _run(allow: frozenset[str]) -> None:
    server = await serve(allow)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stop.set)
    print(f"egress: listening on {LISTEN_HOST}:{LISTEN_PORT}, allowing "
          f"{', '.join(sorted(allow))}", file=sys.stderr, flush=True)
    await stop.wait()
    server.close()


def main() -> int:
    try:
        allow = allowlist()
    except ValueError as error:
        print(f"egress: {error}", file=sys.stderr)
        return 2
    asyncio.run(_run(allow))
    return 0


if __name__ == "__main__":
    sys.exit(main())
