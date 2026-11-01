"""The egress proxy, the one route out of a system's environment (spec section 8): it opens a
tunnel only for ``CONNECT host:port`` to a name and port on its allowlist, and logs one JSON line
per request. The tests run it in-process on an ephemeral port, with ``EGRESS_ALLOW`` naming a
plain TCP echo server on 127.0.0.1 as ``localhost:<port>``, which stands in for the provider's
API."""

from __future__ import annotations

import asyncio
import json
import socket
import socketserver
import threading
import time
from collections.abc import Iterator
from contextlib import closing, contextmanager

import pytest
from env.egress import proxy

TIMEOUT = 5.0
LOG_KEYS = {"ts", "client", "method", "target", "allowed", "reason"}


class Echo(socketserver.ThreadingTCPServer):
    """A TCP echo server on 127.0.0.1 that counts the connections it accepts."""

    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), EchoHandler)
        self.connections = 0

    @property
    def port(self) -> int:
        return self.server_address[1]

    def verify_request(self, request, client_address) -> bool:
        self.connections += 1
        return True


class EchoHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        while data := self.request.recv(65536):
            self.request.sendall(data)


@pytest.fixture
def upstream() -> Iterator[Echo]:
    server = Echo()
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05},
                              daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


@contextmanager
def running(allow: frozenset[str], **options) -> Iterator[int]:
    """The proxy serving on an ephemeral port of 127.0.0.1 in a thread of its own; yields the
    port."""
    loop = asyncio.new_event_loop()
    server = loop.run_until_complete(proxy.serve(allow, "127.0.0.1", 0, **options))
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        yield server.sockets[0].getsockname()[1]
    finally:
        asyncio.run_coroutine_threadsafe(_shut(server), loop).result(TIMEOUT)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(TIMEOUT)
        loop.close()


async def _shut(server: asyncio.Server) -> None:
    server.close()
    tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


@pytest.fixture
def egress(upstream: Echo, monkeypatch: pytest.MonkeyPatch) -> Iterator[int]:
    monkeypatch.setenv("EGRESS_ALLOW", f"localhost:{upstream.port}")
    with running(proxy.allowlist()) as port:
        yield port


def send(port: int, data: bytes) -> socket.socket:
    sock = socket.create_connection(("127.0.0.1", port), timeout=TIMEOUT)
    sock.sendall(data)
    return sock


def connect(port: int, target: str) -> socket.socket:
    return send(port, f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n".encode("ascii"))


def head(sock: socket.socket) -> bytes:
    """The response's status line and header fields, read up to the blank line that ends them."""
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(1)
        if not chunk:
            break
        data += chunk
    return data


def status(sock: socket.socket) -> int:
    return int(head(sock).split(b" ", 2)[1])


def closed(sock: socket.socket) -> bool:
    return sock.recv(1) == b""


def round_trip(sock: socket.socket, data: bytes) -> bytes:
    sock.sendall(data)
    echoed = b""
    while len(echoed) < len(data):
        chunk = sock.recv(65536)
        if not chunk:
            break
        echoed += chunk
    return echoed


def test_allows_exact_host_port(egress, upstream):
    with closing(connect(egress, f"localhost:{upstream.port}")) as sock:
        assert head(sock) == b"HTTP/1.1 200 Connection established\r\n\r\n"
        payload = bytes(range(256)) * 64
        assert round_trip(sock, payload) == payload
    assert upstream.connections == 1


def test_host_compare_is_case_insensitive(egress, upstream):
    with closing(connect(egress, f"LOCALHOST:{upstream.port}")) as sock:
        assert status(sock) == 200
        assert round_trip(sock, b"ping") == b"ping"
    assert upstream.connections == 1


def test_denies_other_host(egress, upstream):
    with closing(connect(egress, "example.invalid:443")) as sock:
        assert status(sock) == 403
        assert closed(sock)
    assert upstream.connections == 0


def test_denies_other_port(egress, upstream):
    for port in (80, 443, upstream.port + 1):
        with closing(connect(egress, f"localhost:{port}")) as sock:
            assert status(sock) == 403, port
            assert closed(sock)
    assert upstream.connections == 0


def test_denies_ip_literal(egress, upstream):
    """127.0.0.1 is the very upstream that localhost names, and is still refused: the allowlist
    names hosts, never addresses."""
    for target in (f"127.0.0.1:{upstream.port}", f"[::1]:{upstream.port}", f"::1:{upstream.port}"):
        with closing(connect(egress, target)) as sock:
            assert status(sock) == 403, target
            assert closed(sock)
    assert upstream.connections == 0


def test_refuses_absolute_form(egress, upstream):
    url = f"http://localhost:{upstream.port}/"
    request = f"GET {url} HTTP/1.1\r\nHost: localhost:{upstream.port}\r\n\r\n".encode("ascii")
    with closing(send(egress, request)) as sock:
        assert status(sock) == 405
        assert closed(sock)
    assert upstream.connections == 0


@pytest.mark.parametrize("line", [
    b"",
    b"CONNECT",
    b"CONNECT localhost:443",
    b"CONNECT localhost HTTP/1.1",
    b"CONNECT localhost: HTTP/1.1",
    b"CONNECT :443 HTTP/1.1",
    b"CONNECT localhost:https HTTP/1.1",
    b"CONNECT localhost:0 HTTP/1.1",
    b"CONNECT localhost:65536 HTTP/1.1",
    b"CONNECT localhost:443 HTTP/2.0",
    b"CONNECT  localhost:443 HTTP/1.1",
    b"CONNECT localhost:443 HTTP/1.1 extra",
    "CONNECT löcalhost:443 HTTP/1.1".encode(),
])
def test_rejects_malformed_request_line(egress, upstream, line):
    with closing(send(egress, line + b"\r\n\r\n")) as sock:
        assert status(sock) == 400
        assert closed(sock)
    assert upstream.connections == 0


def test_rejects_a_head_over_the_cap(egress, upstream):
    """The request line and header fields are read up to 16 KiB and no further."""
    padding = b"x" * (16 * 1024)
    with closing(send(egress, b"CONNECT localhost:443 HTTP/1.1\r\nX-Padding: " + padding)) as sock:
        sock.shutdown(socket.SHUT_WR)
        assert status(sock) == 431
    assert upstream.connections == 0


def test_closes_an_idle_tunnel(upstream):
    """Traffic in either direction keeps a tunnel open past the idle timeout; silence closes it."""
    with running(frozenset({f"localhost:{upstream.port}"}), idle_timeout=0.6) as port:
        with closing(connect(port, f"localhost:{upstream.port}")) as sock:
            assert status(sock) == 200
            for _ in range(4):
                time.sleep(0.2)
                assert round_trip(sock, b"still here") == b"still here"
            started = time.monotonic()
            assert closed(sock)
            assert 0.3 < time.monotonic() - started < TIMEOUT


def test_logs_one_json_line_per_request(egress, upstream, capsys):
    allowed = f"localhost:{upstream.port}"
    cases = [
        (f"CONNECT {allowed} HTTP/1.1\r\n\r\n", True),
        (f"CONNECT {allowed.upper()} HTTP/1.1\r\n\r\n", True),
        ("CONNECT example.invalid:443 HTTP/1.1\r\n\r\n", False),
        ("CONNECT localhost:80 HTTP/1.1\r\n\r\n", False),
        (f"CONNECT 127.0.0.1:{upstream.port} HTTP/1.1\r\n\r\n", False),
        (f"GET http://{allowed}/ HTTP/1.1\r\nHost: {allowed}\r\n\r\n", False),
        ("CONNECT localhost HTTP/1.1\r\n\r\n", False),
    ]
    capsys.readouterr()
    for request, expected in cases:
        with closing(send(egress, request.encode("ascii"))) as sock:
            head(sock)
        lines = capsys.readouterr().out.splitlines()
        assert len(lines) == 1, request
        record = json.loads(lines[0])
        assert set(record) == LOG_KEYS
        assert record["allowed"] is expected, request
        assert record["client"].startswith("127.0.0.1:")
        assert isinstance(record["ts"], str) and isinstance(record["reason"], str)


def test_the_allowlist_comes_from_egress_allow(monkeypatch):
    monkeypatch.delenv("EGRESS_ALLOW", raising=False)
    assert proxy.allowlist() == frozenset({"api.anthropic.com:443"})
    monkeypatch.setenv("EGRESS_ALLOW", " API.Anthropic.com:443 , localhost:8443 ")
    assert proxy.allowlist() == frozenset({"api.anthropic.com:443", "localhost:8443"})
    for bad in ("", "api.anthropic.com", "api.anthropic.com:0", "1.1.1.1:443", "[::1]:443",
                "127.1:443", "api.anthropic.com:443/x"):
        monkeypatch.setenv("EGRESS_ALLOW", bad)
        with pytest.raises(ValueError):
            proxy.allowlist()
