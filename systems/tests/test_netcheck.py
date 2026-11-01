"""The network check (env/netcheck.py), which runs inside a system's container and checks the
environment that section 8 of the registration requires. These tests cover its pure parts, and
the checks that make connections against stand-in servers on 127.0.0.1; the check against the
real proxy and the real API is made by hand, as systems/README.md describes."""

from __future__ import annotations

import ipaddress
import json
import socket
import socketserver
import threading
from collections.abc import Iterator
from contextlib import closing, contextmanager

import pytest
from env import netcheck

TIMEOUT = 5.0
# /proc/mounts as a container sees it with /input, /output and a secret file mounted.
MOUNTS = """\
overlay / overlay rw,relatime,lowerdir=/l/a:/l/b,upperdir=/u/diff,workdir=/u/work 0 0
proc /proc proc rw,nosuid,nodev,noexec,relatime 0 0
tmpfs /dev tmpfs rw,nosuid,size=65536k,mode=755 0 0
devpts /dev/pts devpts rw,nosuid,noexec,relatime,gid=5,mode=620,ptmxmode=666 0 0
sysfs /sys sysfs ro,nosuid,nodev,noexec,relatime 0 0
cgroup /sys/fs/cgroup cgroup2 ro,nosuid,nodev,noexec,relatime 0 0
mqueue /dev/mqueue mqueue rw,nosuid,nodev,noexec,relatime 0 0
shm /dev/shm tmpfs rw,nosuid,nodev,noexec,relatime,size=65536k 0 0
/run/host_mark/private /input fakeowner ro,nosuid,nodev,relatime,fakeowner 0 0
/run/host_mark/private /output fakeowner rw,nosuid,nodev,relatime,fakeowner 0 0
/dev/vda1 /etc/resolv.conf ext4 rw,relatime,discard 0 0
/dev/vda1 /etc/hostname ext4 rw,relatime,discard 0 0
/dev/vda1 /etc/hosts ext4 rw,relatime,discard 0 0
/run/host_mark/private /run/secrets/api_key fakeowner ro,nosuid,nodev,relatime,fakeowner 0 0
proc /proc/bus proc ro,nosuid,nodev,noexec,relatime 0 0
tmpfs /proc/kcore tmpfs rw,nosuid,size=65536k,mode=755 0 0
tmpfs /sys/firmware tmpfs ro,relatime,size=4k,nr_inodes=1 0 0
"""
ALLOWED = ["/input", "/output", "/run/secrets"]


def test_parse_targets():
    assert netcheck.parse_targets("github.com:443,huggingface.co:443") == [
        ("github.com", 443), ("huggingface.co", 443)]
    assert netcheck.parse_targets(" embedder:8080 , neo4j:7687 ,") == [("embedder", 8080),
                                                                       ("neo4j", 7687)]
    assert netcheck.parse_targets("140.82.112.3:443,[2606:4700::1111]:443") == [
        ("140.82.112.3", 443), ("2606:4700::1111", 443)]
    assert netcheck.parse_targets("") == []
    for bad in ("github.com", "github.com:", ":443", "github.com:0", "github.com:65536",
                "github.com:https", "::1:443", "github.com:443/x"):
        with pytest.raises(ValueError):
            netcheck.parse_targets(bad)


def test_report_fails_when_any_check_fails():
    passing = [netcheck.Check("deny", "github.com:443", True, "CONNECT answered 403"),
               netcheck.Check("direct", "1.1.1.1:443", True, "no connection: timed out")]
    failing = [*passing, netcheck.Check("service", "embedder:8080", False, "no connection")]
    good, bad = netcheck.report(passing), netcheck.report(failing)
    assert good["ok"] is True and netcheck.exit_code(good) == 0
    assert bad["ok"] is False and netcheck.exit_code(bad) == 1
    assert [check["ok"] for check in bad["checks"]] == [True, True, False]
    assert bad["checks"][2] == {"kind": "service", "target": "embedder:8080", "ok": False,
                                "detail": "no connection"}
    assert netcheck.report([])["ok"] is False  # no check is no evidence


def test_mount_check_flags_unexpected_mount():
    assert netcheck.check_mounts(MOUNTS, ALLOWED).ok
    extra = MOUNTS + "/dev/sda1 /corpora/peps ext4 ro,relatime 0 0\n"
    check = netcheck.check_mounts(extra, ALLOWED)
    assert not check.ok
    assert "/corpora/peps" in check.detail
    beside = netcheck.check_mounts(MOUNTS + "/dev/sda1 /inputs ext4 rw,relatime 0 0\n", ALLOWED)
    assert not beside.ok and "/inputs" in beside.detail  # beside /input, not beneath it
    secret = netcheck.check_mounts(MOUNTS, ["/input", "/output"])
    assert not secret.ok and "/run/secrets/api_key" in secret.detail
    escaped = netcheck.check_mounts(MOUNTS + "/dev/sda1 /my\\040data ext4 rw 0 0\n", ALLOWED)
    assert not escaped.ok and "/my data" in escaped.detail
    hidden = netcheck.check_mounts(MOUNTS + "/dev/sda1 /dev/corpora ext4 ro 0 0\n", ALLOWED)
    assert not hidden.ok and "/dev/corpora" in hidden.detail  # a host's files, beneath /dev
    assert not netcheck.check_mounts(MOUNTS + "garbage\n", ALLOWED).ok


class StandIn(socketserver.ThreadingTCPServer):
    """A stand-in for the proxy on 127.0.0.1, which answers every request with one status."""

    daemon_threads = True

    def __init__(self, status: int) -> None:
        super().__init__(("127.0.0.1", 0), StandInHandler)
        self.status = status


class StandInHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        data = b""
        while b"\r\n\r\n" not in data and (chunk := self.request.recv(1024)):
            data += chunk
        self.request.sendall(f"HTTP/1.1 {self.server.status} Stand-in\r\n\r\n".encode("ascii"))
        while self.request.recv(1024):
            pass


@contextmanager
def stand_in(status: int) -> Iterator[tuple[str, int]]:
    server = StandIn(status)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05},
                              daemon=True)
    thread.start()
    try:
        yield server.server_address[:2]
    finally:
        server.shutdown()
        server.server_close()


@contextmanager
def listening() -> Iterator[tuple[str, int]]:
    with closing(socket.create_server(("127.0.0.1", 0))) as server:
        yield server.getsockname()[:2]


def closed_port() -> tuple[str, int]:
    with listening() as address:
        pass
    return address


REAL_GETADDRINFO = socket.getaddrinfo


def resolver(names: dict[str, str]):
    """A stand-in for ``socket.getaddrinfo`` as a container on an internal network sees it: each
    name of ``names`` resolves to its address, every other name raises gaierror, as Docker's DNS
    does for a name it does not forward, and an address is looked up as usual (no DNS is asked)."""

    def getaddrinfo(host, port, *args, **kwargs):
        if host in names:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (names[host], port or 0))]
        try:
            ipaddress.ip_address(host)
        except ValueError:
            raise socket.gaierror(socket.EAI_AGAIN,
                                  "Temporary failure in name resolution") from None
        return REAL_GETADDRINFO(host, port, *args, **kwargs)

    return getaddrinfo


def test_the_dns_check_fails_a_host_that_resolves_and_passes_one_that_does_not(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", resolver({"github.com": "140.82.112.3",
                                                         "embedder": "172.22.0.3"}))
    leaked = netcheck.check_dns("github.com", must_resolve=False)
    assert not leaked.ok and "140.82.112.3" in leaked.detail  # a DNS route out
    held = netcheck.check_dns("pypi.org", must_resolve=False)
    assert held.ok and "Temporary failure in name resolution" in held.detail
    assert netcheck.check_dns("embedder", must_resolve=True).ok  # a service the system runs
    missing = netcheck.check_dns("neo4j", must_resolve=True)
    assert not missing.ok and "does not resolve" in missing.detail


def test_the_dns_checks_name_each_host_once_and_skip_addresses(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", resolver({"embedder": "172.22.0.3"}))
    checks = netcheck.dns_checks(
        [("api.example.invalid", 443), ("github.com", 443), ("GitHub.com", 80),
         ("160.79.104.10", 443), ("2606:4700::1111", 443)],
        [("embedder", 8080), ("embedder", 8081), ("172.22.0.9", 7687)])
    assert [(check.kind, check.target, check.ok) for check in checks] == [
        ("dns", "api.example.invalid", True), ("dns", "github.com", True),
        ("dns", "embedder", True)]


def test_the_checks_that_connect():
    with stand_in(403) as proxy:
        assert netcheck.check_deny(proxy, ("github.com", 443), TIMEOUT).ok
        refused = netcheck.check_allow(proxy, ("api.example.invalid", 443), TIMEOUT)
        assert not refused.ok and "403" in refused.detail
    with stand_in(200) as proxy:
        opened = netcheck.check_deny(proxy, ("github.com", 443), TIMEOUT)
        assert not opened.ok and "200" in opened.detail
    assert not netcheck.check_deny(closed_port(), ("github.com", 443), TIMEOUT).ok
    with listening() as address:
        assert netcheck.check_service(address, TIMEOUT).ok
        assert not netcheck.check_direct(address, TIMEOUT).ok
    address = closed_port()
    assert netcheck.check_direct(address, TIMEOUT).ok
    assert not netcheck.check_service(address, TIMEOUT).ok


def test_main_writes_the_report_and_exits_by_it(tmp_path, monkeypatch):
    mounts = tmp_path / "mounts"
    mounts.write_text(MOUNTS, encoding="utf-8")
    monkeypatch.setattr(netcheck, "MOUNTS_FILE", mounts)
    monkeypatch.setattr(socket, "getaddrinfo", resolver({}))
    out = tmp_path / "netcheck.json"
    host, closed = closed_port()
    with stand_in(403) as (proxy_host, proxy_port), listening() as (service_host, service_port):
        argv = ["--proxy", f"http://{proxy_host}:{proxy_port}",
                "--deny", "github.com:443,pypi.org:443", "--deny", "peps.python.org:443",
                "--direct", f"{host}:{closed}", "--services", f"{service_host}:{service_port}",
                "--mounts", ",".join(ALLOWED), "--report", str(out)]
        assert netcheck.main(argv) == 0
        written = json.loads(out.read_text(encoding="utf-8"))
        assert written["ok"] is True
        assert [(check["kind"], check["target"], check["ok"]) for check in written["checks"]] == [
            ("deny", "github.com:443", True), ("deny", "pypi.org:443", True),
            ("deny", "peps.python.org:443", True), ("direct", f"{host}:{closed}", True),
            ("service", f"{service_host}:{service_port}", True), ("dns", "github.com", True),
            ("dns", "pypi.org", True), ("dns", "peps.python.org", True),
            ("mounts", str(mounts), True)]
        assert netcheck.main([*argv[:-4], "--mounts", "/input", "--report", str(out)]) == 1
        assert json.loads(out.read_text(encoding="utf-8"))["ok"] is False
        # The proxy refuses pypi.org, but a resolver that answers for it is a route out.
        monkeypatch.setattr(socket, "getaddrinfo", resolver({"pypi.org": "151.101.0.223"}))
        assert netcheck.main(argv) == 1
        failed = [check for check in json.loads(out.read_text(encoding="utf-8"))["checks"]
                  if not check["ok"]]
        assert [(check["kind"], check["target"]) for check in failed] == [("dns", "pypi.org")]
