"""The network check (env/netcheck.py), which runs inside a system's container and checks the
environment that section 8 of the registration requires. These tests cover its pure parts, and
the checks that make connections against stand-in servers on 127.0.0.1; the check against the
real proxy and the real API is made by hand, as systems/README.md describes."""

from __future__ import annotations

import ipaddress
import json
import socket
import socketserver
import sys
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


def word(address: str) -> str:
    """An IPv4 address as /proc/net/route writes it: its four bytes read as one host-order word."""
    return format(int.from_bytes(ipaddress.IPv4Address(address).packed, sys.byteorder), "08X")


def route(iface: str, destination: str, gateway: str, flags: int, mask: str) -> str:
    return (f"{iface}\t{word(destination)}\t{word(gateway)}\t{flags:04X}\t0\t0\t0\t{word(mask)}"
            "\t0\t0\t0\n")


ROUTE_HEADER = ("Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\t\tMTU\tWindow"
                "\tIRTT\n")
# A container's route tables as Docker gives them: on an internal network, one subnet and no
# default route; on a network with a route out, a default route through the network's gateway.
ROUTES_INNER = ROUTE_HEADER + route("eth0", "172.22.0.0", "0.0.0.0", 0x0001, "255.255.0.0")
ROUTES_OUTER = (ROUTE_HEADER + route("eth0", "0.0.0.0", "172.23.0.1", 0x0003, "0.0.0.0")
                + route("eth0", "172.23.0.0", "0.0.0.0", 0x0001, "255.255.0.0"))
# /proc/net/ipv6_route with IPv6 on loopback only: ::1, and the two reject (unreachable) entries
# that the kernel lists, which are no route; then a default route through fe80::1.
NO_ROUTE_V6 = ("00000000000000000000000000000000 00 00000000000000000000000000000000 00 "
               "00000000000000000000000000000000 ffffffff 00000001 00000000 00200200       lo\n")
IPV6_ROUTES_LO = (NO_ROUTE_V6 + "00000000000000000000000000000001 80 0000000000000000000000000000"
                  "0000 00 00000000000000000000000000000000 00000000 00000004 00000000 80200001"
                  "       lo\n" + NO_ROUTE_V6)
IPV6_DEFAULT = ("00000000000000000000000000000000 00 00000000000000000000000000000000 00 "
                "fe800000000000000000000000000001 00000400 00000001 00000000 00000003     eth0\n")
# /proc/net/fib_trie and /proc/net/if_inet6 for a container whose address on its one network is
# 172.22.0.2.
FIB_TRIE = """\
Main:
  +-- 0.0.0.0/0 2 0 2
     +-- 127.0.0.0/8 2 0 2
        +-- 127.0.0.0/31 1 0 0
           |-- 127.0.0.0
              /8 host LOCAL
           |-- 127.0.0.1
              /32 host LOCAL
        |-- 127.255.255.255
           /32 link BROADCAST
     +-- 172.22.0.0/16 2 0 2
        +-- 172.22.0.0/30 2 0 2
           |-- 172.22.0.0
              /16 link UNICAST
           |-- 172.22.0.2
              /32 host LOCAL
        |-- 172.22.255.255
           /32 link BROADCAST
"""
IF_INET6 = "00000000000000000000000000000001 01 80 10 80       lo\n"
DOCKER_HOSTS = ("192.168.65.1", "192.168.65.254", "172.17.0.1")


def test_route_tables_with_and_without_a_default_route():
    inner, outer = netcheck.parse_routes(ROUTES_INNER), netcheck.parse_routes(ROUTES_OUTER)
    assert [(r.iface, str(r.network), r.gateway) for r in inner] == [
        ("eth0", "172.22.0.0/16", None)]
    assert [(r.iface, str(r.network), r.gateway) for r in outer] == [
        ("eth0", "0.0.0.0/0", "172.23.0.1"), ("eth0", "172.23.0.0/16", None)]
    lo_only = netcheck.parse_ipv6_routes(IPV6_ROUTES_LO)
    assert [(r.iface, str(r.network)) for r in lo_only] == [("lo", "::/0"), ("lo", "::1/128"),
                                                            ("lo", "::/0")]
    assert netcheck.check_routes(inner + lo_only).ok  # loopback's reject entries are no route
    out = netcheck.check_routes(outer + lo_only)
    assert not out.ok and "a default route out" in out.detail and "172.23.0.1" in out.detail
    with_default = netcheck.parse_ipv6_routes(IPV6_ROUTES_LO + IPV6_DEFAULT)
    out6 = netcheck.check_routes(inner + with_default)
    assert not out6.ok and "a default route out" in out6.detail and "fe80::1" in out6.detail
    with pytest.raises(ValueError):
        netcheck.parse_routes(ROUTE_HEADER + "eth0\tnot-hex\n")


def test_gateway_candidates_from_a_route_table():
    assert netcheck.gateway_candidates(netcheck.parse_routes(ROUTES_INNER), DOCKER_HOSTS, []) == [
        "172.22.0.1", *DOCKER_HOSTS]
    outer = netcheck.parse_routes(ROUTES_OUTER)
    assert netcheck.gateway_candidates(outer, DOCKER_HOSTS, ["10.9.9.9", "172.17.0.1"]) == [
        "172.23.0.1", *DOCKER_HOSTS, "10.9.9.9"]
    # Loopback and a host route give no candidate; a route through a gateway gives its next hop.
    other = netcheck.parse_routes(ROUTE_HEADER + route("lo", "127.0.0.0", "0.0.0.0", 0x0001,
                                                       "255.0.0.0")
                                  + route("eth1", "10.1.2.3", "0.0.0.0", 0x0005,
                                          "255.255.255.255")
                                  + route("eth1", "10.8.0.0", "10.7.0.254", 0x0003,
                                          "255.255.0.0"))
    assert netcheck.gateway_candidates(other, (), []) == ["10.7.0.254"]
    assert netcheck.local_addresses(FIB_TRIE, IF_INET6) == {"127.0.0.1", "172.22.0.2", "::1"}


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


def test_the_gateway_probe_against_127_0_0_1():
    closed = [closed_port()[1] for _ in range(3)]
    with closing(socket.create_server(("127.0.0.1", 0))) as listener:
        port = listener.getsockname()[1]
        ports = sorted([*closed, port])
        (found,), _ = netcheck.check_gateways(["127.0.0.1"], {}, ports, timeout=TIMEOUT,
                                              concurrency=2)
        assert not found.ok and found.detail == f"accepts on 127.0.0.1:{port}"
        own = {"127.0.0.1": "the container's own address"}
        (skipped,), _ = netcheck.check_gateways(["127.0.0.1"], own, ports, timeout=TIMEOUT,
                                                concurrency=2)
        assert skipped.ok and skipped.detail == "not probed: the container's own address"
    (held,), seconds = netcheck.check_gateways(["127.0.0.1"], {}, ports, timeout=TIMEOUT,
                                               concurrency=2)
    assert held.ok and held.detail == "no TCP port of the 4 probed accepts (ECONNREFUSED 4)"
    assert seconds >= 0


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


def kernel_files(tmp_path, monkeypatch, routes: str) -> None:
    """Points netcheck at route tables and address lists written under ``tmp_path``, with no
    Docker host address to probe, so that nothing outside this machine is reached."""
    for constant, text in (("ROUTE_FILE", routes), ("IPV6_ROUTE_FILE", IPV6_ROUTES_LO),
                           ("FIB_TRIE_FILE", FIB_TRIE), ("IF_INET6_FILE", IF_INET6)):
        path = tmp_path / constant.lower()
        path.write_text(text, encoding="utf-8")
        monkeypatch.setattr(netcheck, constant, path)
    monkeypatch.setattr(netcheck, "DOCKER_HOST_ADDRESSES", ())


def failures(out) -> list[tuple[str, str]]:
    checks = json.loads(out.read_text(encoding="utf-8"))["checks"]
    return [(check["kind"], check["target"]) for check in checks if not check["ok"]]


def test_a_default_route_fails_the_gateway_check_without_a_scan(tmp_path, monkeypatch):
    """With a route out the environment fails already, so no candidate is scanned through it, and
    none passes unprobed."""
    kernel_files(tmp_path, monkeypatch, ROUTES_OUTER)
    monkeypatch.setattr(netcheck, "DOCKER_HOST_ADDRESSES", ("192.0.2.1",))
    monkeypatch.setattr(socket, "getaddrinfo", resolver({}))

    async def no_scan(*args, **kwargs):
        raise AssertionError("the gateway scan ran")

    monkeypatch.setattr(netcheck, "_scan", no_scan)
    checks, seconds = netcheck.gateway_checks([], [], ("egress", 3128))
    assert seconds is None
    assert [(check.target, check.ok) for check in checks] == [
        ("default route", False), ("172.23.0.1", False), ("192.0.2.1", False)]
    assert all(check.detail.startswith("not probed: ") for check in checks[1:])


def test_main_writes_the_report_and_exits_by_it(tmp_path, monkeypatch):
    mounts = tmp_path / "mounts"
    mounts.write_text(MOUNTS, encoding="utf-8")
    monkeypatch.setattr(netcheck, "MOUNTS_FILE", mounts)
    monkeypatch.setattr(socket, "getaddrinfo", resolver({}))
    loopback = ROUTE_HEADER + route("lo", "127.0.0.0", "0.0.0.0", 0x0001, "255.0.0.0")
    kernel_files(tmp_path, monkeypatch, loopback)
    out = tmp_path / "netcheck.json"
    host, closed = closed_port()
    with stand_in(403) as (proxy_host, proxy_port), listening() as (service_host, service_port):
        argv = ["--proxy", f"http://{proxy_host}:{proxy_port}",
                "--deny", "github.com:443,pypi.org:443", "--deny", "peps.python.org:443",
                "--direct", f"{host}:{closed}", "--services", f"{service_host}:{service_port}",
                "--gateway", "127.0.0.1", "--mounts", ",".join(ALLOWED), "--report", str(out)]
        assert netcheck.main(argv) == 0
        written = json.loads(out.read_text(encoding="utf-8"))
        assert written["ok"] is True
        assert [(check["kind"], check["target"], check["ok"]) for check in written["checks"]] == [
            ("deny", "github.com:443", True), ("deny", "pypi.org:443", True),
            ("deny", "peps.python.org:443", True), ("direct", f"{host}:{closed}", True),
            ("service", f"{service_host}:{service_port}", True), ("dns", "github.com", True),
            ("dns", "pypi.org", True), ("dns", "peps.python.org", True),
            ("gateway", "default route", True), ("gateway", "127.0.0.1", True),
            ("mounts", str(mounts), True)]
        assert written["checks"][9]["detail"] == "not probed: the container's own address"
        assert netcheck.main([*argv[:-4], "--mounts", "/input", "--report", str(out)]) == 1
        assert json.loads(out.read_text(encoding="utf-8"))["ok"] is False
        # The proxy refuses pypi.org, but a resolver that answers for it is a route out.
        monkeypatch.setattr(socket, "getaddrinfo", resolver({"pypi.org": "151.101.0.223"}))
        assert netcheck.main(argv) == 1
        assert failures(out) == [("dns", "pypi.org")]
        # And so is a default route, whatever the proxy refuses.
        monkeypatch.setattr(socket, "getaddrinfo", resolver({}))
        kernel_files(tmp_path, monkeypatch,
                     ROUTE_HEADER + route("lo", "0.0.0.0", "127.0.0.1", 0x0003, "0.0.0.0"))
        assert netcheck.main(argv) == 1
        assert failures(out) == [("gateway", "default route")]
