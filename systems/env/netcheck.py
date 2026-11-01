"""The network check: run inside a system's container, it checks the environment that section 8
of the registration requires before the system ingests or answers anything.

    python netcheck.py --proxy http://egress:3128 --allow api.anthropic.com:443 \\
        --deny github.com:443,huggingface.co:443,pypi.org:443 \\
        --deny peps.python.org:443,eips.ethereum.org:443,ercs.ethereum.org:443 \\
        --deny host.docker.internal:443,gateway.docker.internal:443 \\
        --direct 140.82.112.3:443,1.1.1.1:443 --services embedder:8080,neo4j:7687 \\
        --mounts /input,/output,/run/secrets --report /output/netcheck.json

Each list is comma-separated, and a flag given twice adds to its list.

The checks, each written to the report with its result:

- allow: for each target, a CONNECT through the proxy, a TLS handshake with SNI and the
  certificate verified, then ``GET /v1/models`` with no key, which must answer 401: the API is
  reachable, and netcheck never sends a key or any other credential;
- deny: for each target, a CONNECT through the proxy, which the proxy must refuse with 403;
- direct: for each target, a TCP connection made directly, not through the proxy, which must
  fail within the timeout;
- service: each target, a service the system runs itself inside the environment, must accept a
  TCP connection;
- dns: no ``--allow`` or ``--deny`` host may resolve from inside the container, since the proxy,
  not the system, resolves the hosts it reaches, and a name that resolves here is a DNS route out
  (queries leave, and answers come back); each ``--services`` host must resolve, through Docker's
  embedded DNS. An IP address is looked up without asking DNS, so it is not checked;
- gateway: the container may have no default route, IPv4 or IPv6 (/proc/net/route and, when it
  exists, /proc/net/ipv6_route), and no TCP port from 1 to 65535 may accept a connection on a
  gateway candidate: the next hop of each IPv4 route through a gateway, the first host address of
  each IPv4 subnet the container is on (the address Docker gives a network's gateway by default),
  Docker Desktop's host addresses 192.168.65.1 and 192.168.65.254, the default bridge's gateway
  172.17.0.1 and each ``--gateway`` address. The container's own addresses (/proc/net/fib_trie,
  /proc/net/if_inet6) and those of the proxy and the ``--services`` are not probed. A connection
  that is refused, unreachable or not accepted within ``SCAN_TIMEOUT`` passes; any other error
  leaves its port unproven and fails the check. With a default route the check fails on it and
  no candidate is scanned, since a scan through a route out would load the host's connection
  tracking;
- mounts: /proc/mounts may show mounts only at the ``--mounts`` paths and beneath them, besides
  the container's own root, the /etc/hosts, /etc/hostname and /etc/resolv.conf that Docker
  manages, and the kernel's own filesystems at and beneath /proc, /sys and /dev. A mount anywhere
  else, of any type, or one beneath those three of a type that can hold files from the host,
  fails the check and is named.

It exits 0 only when every check passes, and 1 otherwise. It uses only the standard library.
"""

from __future__ import annotations

import argparse
import asyncio
import errno
import http.client
import ipaddress
import json
import re
import socket
import ssl
import struct
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

TIMEOUT = 5.0
MOUNTS_FILE = Path("/proc/mounts")
# The path the allow check asks for, with no key: the API answers 401 when it is reachable.
PROBE_PATH = "/v1/models"
PROBE_STATUS = 401
DENIED_STATUS = 403
# The mounts every container has: its root and three files that Docker manages, exactly, and at
# and beneath /proc, /sys and /dev the kernel's own filesystems, which hold no file from the host
# (Docker masks parts of /proc with tmpfs).
CONTAINER_FILES = ("/", "/etc/hosts", "/etc/hostname", "/etc/resolv.conf")
CONTAINER_TREES = ("/proc", "/sys", "/dev")
KERNEL_FILESYSTEMS = frozenset({"proc", "sysfs", "tmpfs", "devpts", "mqueue", "cgroup", "cgroup2",
                                "devtmpfs", "securityfs", "debugfs", "tracefs", "bpf", "pstore",
                                "fusectl", "configfs", "binfmt_misc", "hugetlbfs", "efivarfs"})
STATUS_LINE = re.compile(r"HTTP/1\.[0-9] (?P<status>[0-9]{3})(?: .*)?")
HEAD_LIMIT = 16 * 1024
TARGET = re.compile(r"(?:\[(?P<ipv6>[0-9A-Fa-f:.]+)\]|(?P<host>[^\s:\[\]/]+)):(?P<port>[0-9]{1,5})")
ROUTE_FILE = Path("/proc/net/route")
IPV6_ROUTE_FILE = Path("/proc/net/ipv6_route")
FIB_TRIE_FILE = Path("/proc/net/fib_trie")
IF_INET6_FILE = Path("/proc/net/if_inet6")
# Route flags (linux/route.h): up, through a gateway, and reject, an unreachable route that the
# kernel lists on loopback and that leads nowhere.
RTF_UP, RTF_GATEWAY, RTF_REJECT = 0x0001, 0x0002, 0x0200
# Gateway candidates besides those of the route table: Docker Desktop's host addresses and the
# default bridge's gateway.
DOCKER_HOST_ADDRESSES = ("192.168.65.1", "192.168.65.254", "172.17.0.1")
# The gateway scan: every TCP port, at most SCAN_CONCURRENCY connections at a time (and half the
# open-file limit at most), each given SCAN_TIMEOUT seconds to be accepted, which is far longer than
# a listener on a local network takes to answer.
SCAN_PORTS = range(1, 65536)
SCAN_CONCURRENCY = 512
SCAN_TIMEOUT = 0.25
# Connection errors that mean a port does not accept: refused, unreachable or blocked.
NOT_ACCEPTED = frozenset({errno.ECONNREFUSED, errno.ENETUNREACH, errno.EHOSTUNREACH,
                          errno.EHOSTDOWN, errno.ENETDOWN, errno.ETIMEDOUT, errno.ECONNRESET,
                          errno.ECONNABORTED, errno.EPERM, errno.EACCES})
# How many accepting ports a failed gateway check names.
SHOWN_PORTS = 20


@dataclass(frozen=True)
class Check:
    """One check and its result: ``kind`` is allow, deny, direct, service, dns, gateway or
    mounts."""

    kind: str
    target: str
    ok: bool
    detail: str


def parse_targets(text: str) -> list[tuple[str, int]]:
    """The ``host:port`` targets of a comma-separated list, an IPv6 address in brackets;
    ValueError for an entry that is not one."""
    targets = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        match = TARGET.fullmatch(item)
        if match is None or not 1 <= int(match["port"]) <= 65535:
            raise ValueError(f"{item!r} is not host:port with a port from 1 to 65535")
        targets.append((match["ipv6"] or match["host"], int(match["port"])))
    return targets


def parse_proxy(url: str) -> tuple[str, int]:
    """The host and port of the proxy's ``http://host:port`` URL; ValueError otherwise."""
    parts = urlsplit(url)
    if parts.scheme != "http" or not parts.hostname or parts.port is None:
        raise ValueError(f"{url!r} is not http://host:port")
    return parts.hostname, parts.port


def parse_paths(text: str) -> list[str]:
    """The absolute paths of a comma-separated list, other than /; ValueError otherwise."""
    paths = [item.strip().rstrip("/") for item in text.split(",") if item.strip()]
    if not all(path.startswith("/") for path in paths):
        raise ValueError(f"{text!r} names a path that is not absolute, or /")
    return paths


def name(target: tuple[str, int]) -> str:
    host, port = target
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


def report(checks: Sequence[Check]) -> dict:
    """The report: every check and its result, and ``ok``, true only when there is at least one
    check and every check passed."""
    return {"ok": bool(checks) and all(check.ok for check in checks),
            "checks": [asdict(check) for check in checks]}


def exit_code(written: dict) -> int:
    return 0 if written["ok"] else 1


def unescape(field: str) -> str:
    """A /proc/mounts field as the kernel means it: it writes space, tab, newline and backslash
    as three octal digits after a backslash."""
    return re.sub(r"\\([0-7]{3})", lambda digits: chr(int(digits[1], 8)), field)


def beneath(path: str, roots: Sequence[str]) -> bool:
    return any(path == root or path.startswith(root + "/") for root in roots)


def check_mounts(text: str, allowed: Sequence[str]) -> Check:
    """The mounts check over the text of /proc/mounts, with the ``allowed`` paths."""
    unexpected, expected = [], []
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 3:
            unexpected.append(f"a line that is not a mount: {line!r}")
            continue
        device, point, kind = unescape(fields[0]), unescape(fields[1]), fields[2]
        if point in CONTAINER_FILES or (beneath(point, CONTAINER_TREES)
                                        and kind in KERNEL_FILESYSTEMS):
            continue
        if beneath(point, allowed):
            expected.append(point)
        else:
            unexpected.append(f"{point} ({kind} from {device})")
    if unexpected:
        return Check("mounts", str(MOUNTS_FILE), False, "unexpected: " + "; ".join(unexpected))
    found = ", ".join(expected) if expected else "none"
    return Check("mounts", str(MOUNTS_FILE), True,
                 f"only the container's own and the allowed mounts ({found})")


def tunnel(proxy: tuple[str, int], target: tuple[str, int],
           timeout: float) -> tuple[int, socket.socket]:
    """Sends ``CONNECT target`` to the proxy and returns the status it answers and the socket."""
    sock = socket.create_connection(proxy, timeout=timeout)
    try:
        authority = name(target)
        sock.sendall(f"CONNECT {authority} HTTP/1.1\r\nHost: {authority}\r\n\r\n".encode("ascii"))
        head = b""
        while not head.endswith(b"\r\n\r\n"):
            byte = sock.recv(1)
            if not byte:
                raise ConnectionError("the proxy closed the connection without an answer")
            head += byte
            if len(head) > HEAD_LIMIT:
                raise ConnectionError("the proxy's answer has no end")
        status_line = head.split(b"\r\n", 1)[0].decode("latin-1")
        match = STATUS_LINE.fullmatch(status_line)
        if match is None:
            raise ConnectionError(f"the proxy answered {status_line!r}")
        return int(match["status"]), sock
    except BaseException:
        sock.close()
        raise


def check_allow(proxy: tuple[str, int], target: tuple[str, int], timeout: float) -> Check:
    host = target[0]
    try:
        status, sock = tunnel(proxy, target, timeout)
    except OSError as error:
        return Check("allow", name(target), False, f"no tunnel: {error}")
    with sock:
        if status != 200:
            return Check("allow", name(target), False, f"CONNECT answered {status}, not 200")
        try:
            with ssl.create_default_context().wrap_socket(sock, server_hostname=host) as tls:
                tls.sendall(f"GET {PROBE_PATH} HTTP/1.1\r\nHost: {host}\r\nAccept: "
                            "application/json\r\nUser-Agent: asofbench-netcheck\r\nConnection: "
                            "close\r\n\r\n".encode("ascii"))
                answer = http.client.HTTPResponse(tls, method="GET")
                answer.begin()
                version, code = tls.version(), answer.status
        except (OSError, http.client.HTTPException) as error:
            return Check("allow", name(target), False, f"CONNECT 200, then {error}")
    detail = (f"CONNECT 200; {version} with the certificate verified for {host}; GET {PROBE_PATH} "
              f"with no key answered {code}")
    if code != PROBE_STATUS:
        detail += f", not {PROBE_STATUS}"
    return Check("allow", name(target), code == PROBE_STATUS, detail)


def check_deny(proxy: tuple[str, int], target: tuple[str, int], timeout: float) -> Check:
    try:
        status, sock = tunnel(proxy, target, timeout)
    except OSError as error:
        return Check("deny", name(target), False, f"no answer from the proxy: {error}")
    sock.close()
    detail = f"CONNECT answered {status}" + ("" if status == DENIED_STATUS else
                                             f", not {DENIED_STATUS}")
    return Check("deny", name(target), status == DENIED_STATUS, detail)


def check_direct(target: tuple[str, int], timeout: float) -> Check:
    try:
        sock = socket.create_connection(target, timeout=timeout)
    except OSError as error:
        return Check("direct", name(target), True, f"no connection: {error}")
    sock.close()
    return Check("direct", name(target), False, "a direct connection was made")


def check_service(target: tuple[str, int], timeout: float) -> Check:
    try:
        socket.create_connection(target, timeout=timeout).close()
    except OSError as error:
        return Check("service", name(target), False, f"no connection: {error}")
    return Check("service", name(target), True, "accepts a TCP connection")


def is_address(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def check_dns(host: str, must_resolve: bool) -> Check:
    """Whether ``host`` resolves from inside the container, as ``must_resolve`` says it must or
    must not: a service the system runs must resolve, through Docker's embedded DNS, and a host
    beyond the proxy must not, since a name that resolves here is a DNS route out."""
    try:
        answers = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as error:
        return Check("dns", host, not must_resolve, f"does not resolve: {error}")
    except (OSError, UnicodeError) as error:
        return Check("dns", host, False, f"cannot be looked up: {error}")
    addresses = ", ".join(sorted({answer[4][0] for answer in answers}))
    detail = f"resolves to {addresses}" + ("" if must_resolve else ", a DNS route out")
    return Check("dns", host, must_resolve, detail)


def dns_checks(beyond: Sequence[tuple[str, int]],
               services: Sequence[tuple[str, int]]) -> list[Check]:
    """The DNS checks: each host ``beyond`` the proxy (the ``--allow`` and ``--deny`` targets)
    must not resolve, and each service must. Each name is checked once, in lower case, and an IP
    address not at all."""

    def names(targets: Sequence[tuple[str, int]]) -> list[str]:
        return list(dict.fromkeys(host.lower() for host, _ in targets if not is_address(host)))

    return ([check_dns(host, must_resolve=False) for host in names(beyond)]
            + [check_dns(host, must_resolve=True) for host in names(services)])


@dataclass(frozen=True)
class Route:
    """A route of the container's own table: its interface, the network it leads to, its next
    hop when it goes through a gateway, and its flags."""

    iface: str
    network: ipaddress.IPv4Network | ipaddress.IPv6Network
    gateway: str | None
    flags: int

    @property
    def is_default(self) -> bool:
        return (self.network.prefixlen == 0 and bool(self.flags & RTF_UP)
                and not self.flags & RTF_REJECT)


def _ipv4(word: str) -> ipaddress.IPv4Address:
    """An IPv4 address that /proc/net/route writes as one hexadecimal word in host byte order."""
    if not re.fullmatch("[0-9A-Fa-f]{8}", word):
        raise ValueError(f"{word!r} is not an IPv4 address of /proc/net/route")
    return ipaddress.IPv4Address(struct.pack("=I", int(word, 16)))


def _ipv6(digits: str) -> ipaddress.IPv6Address:
    """An IPv6 address that /proc/net writes as 32 hexadecimal digits."""
    if not re.fullmatch("[0-9A-Fa-f]{32}", digits):
        raise ValueError(f"{digits!r} is not an IPv6 address of /proc/net")
    return ipaddress.IPv6Address(bytes.fromhex(digits))


def parse_routes(text: str) -> list[Route]:
    """The routes of /proc/net/route; ValueError for a line that is not one."""
    routes = []
    for line in text.splitlines()[1:]:
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 8:
            raise ValueError(f"not a route: {line!r}")
        destination, gateway, mask = _ipv4(fields[1]), _ipv4(fields[2]), _ipv4(fields[7])
        flags = int(fields[3], 16)
        network = ipaddress.IPv4Network(f"{destination}/{mask}", strict=False)
        routes.append(Route(fields[0], network, str(gateway) if flags & RTF_GATEWAY else None,
                            flags))
    return routes


def parse_ipv6_routes(text: str) -> list[Route]:
    """The routes of /proc/net/ipv6_route; ValueError for a line that is not one."""
    routes = []
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 10:
            raise ValueError(f"not a route: {line!r}")
        flags = int(fields[8], 16)
        network = ipaddress.IPv6Network((_ipv6(fields[0]), int(fields[1], 16)), strict=False)
        gateway = str(_ipv6(fields[4])) if flags & RTF_GATEWAY else None
        routes.append(Route(fields[9], network, gateway, flags))
    return routes


def check_routes(routes: Sequence[Route]) -> Check:
    """The first gateway check: no route of ``routes`` may be a default route."""
    out = [route for route in routes if route.is_default]
    if out:
        found = "; ".join(f"{route.network} via {route.gateway or 'no gateway'} on {route.iface}"
                          for route in out)
        return Check("gateway", "default route", False, f"a default route out: {found}")
    return Check("gateway", "default route", True, "none, IPv4 or IPv6")


def gateway_candidates(routes: Sequence[Route], hosts: Sequence[str],
                       extra: Sequence[str]) -> list[str]:
    """The addresses the gateway scan probes, in order and each once: the next hop of each IPv4
    route through a gateway and the first host address of each IPv4 subnet the container is on,
    loopback aside, then ``hosts`` and ``extra``."""
    found = []
    for route in routes:
        if route.network.version != 4:
            continue
        if route.gateway is not None:
            found.append(route.gateway)
        elif (route.iface != "lo" and not route.network.is_loopback
              and 0 < route.network.prefixlen <= 30):
            found.append(str(route.network.network_address + 1))
    return list(dict.fromkeys([*found, *hosts, *extra]))


def local_addresses(fib_trie: str, if_inet6: str) -> set[str]:
    """The container's own addresses: the IPv4 ones that /proc/net/fib_trie lists as local
    hosts, and the IPv6 ones of /proc/net/if_inet6."""
    found, current = set(), None
    for line in fib_trie.splitlines():
        words = line.split()
        if len(words) == 2 and words[0] == "|--":
            current = words[1]
        elif words[:3] == ["/32", "host", "LOCAL"] and current is not None:
            found.add(current)
    found.update(str(_ipv6(line.split()[0])) for line in if_inet6.splitlines() if line.split())
    return found


@dataclass
class Probe:
    """What the scan found at one address: the ports that accepted, how many ports gave each
    other outcome, and the errors that left a port unproven."""

    accepted: list[int] = field(default_factory=list)
    outcomes: Counter = field(default_factory=Counter)
    errors: list[str] = field(default_factory=list)


async def _scan(addresses: Sequence[str], ports: Sequence[int], timeout: float,
                concurrency: int) -> dict[str, Probe]:
    loop = asyncio.get_running_loop()
    probes = {address: Probe() for address in addresses}
    pairs = ((address, port) for address in addresses for port in ports)

    async def worker() -> None:
        for address, port in pairs:
            probe = probes[address]
            try:
                sock = socket.socket(socket.AF_INET6 if ":" in address else socket.AF_INET,
                                     socket.SOCK_STREAM)
            except OSError as error:
                probe.errors.append(f"port {port}: {error}")
                continue
            try:
                sock.setblocking(False)
                await asyncio.wait_for(loop.sock_connect(sock, (address, port)), timeout)
            except TimeoutError:
                probe.outcomes["timed out"] += 1
            except OSError as error:
                if error.errno in NOT_ACCEPTED:
                    probe.outcomes[errno.errorcode[error.errno]] += 1
                else:
                    probe.errors.append(f"port {port}: {error}")
            else:
                probe.accepted.append(port)
            finally:
                sock.close()

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return probes


def _bounded(concurrency: int) -> int:
    """``concurrency``, but at most half the process's limit of open files."""
    try:
        import resource

        soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (ImportError, OSError, ValueError):
        return concurrency
    if soft == resource.RLIM_INFINITY or soft <= 0:
        return concurrency
    return max(1, min(concurrency, soft // 2))


def check_gateways(candidates: Sequence[str], excluded: dict[str, str],
                   ports: Sequence[int] = SCAN_PORTS, timeout: float = SCAN_TIMEOUT,
                   concurrency: int = SCAN_CONCURRENCY) -> tuple[list[Check], float]:
    """A check for each candidate, and the scan's wall time in seconds. A candidate that
    ``excluded`` names, with the reason, is not probed; on each other one every port of
    ``ports`` is tried, and its check passes only when none accepts and none is unproven."""
    probed = [address for address in candidates if address not in excluded]
    started = time.monotonic()
    probes = (asyncio.run(_scan(probed, ports, timeout, _bounded(concurrency))) if probed
              else {})
    seconds = time.monotonic() - started
    span = (f"from {ports[0]} to {ports[-1]}" if isinstance(ports, range) and ports.step == 1
            and len(ports) > 1 else f"of the {len(ports)} probed")
    checks = []
    for address in candidates:
        if address in excluded:
            checks.append(Check("gateway", address, True, f"not probed: {excluded[address]}"))
            continue
        probe = probes[address]
        if probe.accepted:
            accepted = sorted(probe.accepted)
            shown = ", ".join(name((address, port)) for port in accepted[:SHOWN_PORTS])
            more = len(accepted) - SHOWN_PORTS
            detail = f"accepts on {shown}" + (f" and {more} more ports" if more > 0 else "")
            checks.append(Check("gateway", address, False, detail))
        elif probe.errors:
            checks.append(Check("gateway", address, False,
                                f"{len(probe.errors)} ports unproven: {probe.errors[0]}"))
        else:
            outcomes = ", ".join(f"{outcome} {count}"
                                 for outcome, count in sorted(probe.outcomes.items()))
            checks.append(Check("gateway", address, True,
                                f"no TCP port {span} accepts ({outcomes})"))
    return checks, seconds


def _resolved(host: str) -> set[str]:
    try:
        return {answer[4][0] for answer in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)}
    except (OSError, UnicodeError):
        return set()


def gateway_checks(extra: Sequence[str], services: Sequence[tuple[str, int]],
                   proxy: tuple[str, int]) -> tuple[list[Check], float | None]:
    """The gateway checks of this container, from its own route tables and addresses, and the
    scan's wall time (None when no scan ran). With a default route the check fails on it and no
    candidate is scanned: the environment fails already, and a scan through a route out would
    only load the host's connection tracking, which its other containers share."""
    try:
        routes = parse_routes(ROUTE_FILE.read_text(encoding="utf-8"))
        try:
            ipv6_routes = IPV6_ROUTE_FILE.read_text(encoding="utf-8")
        except FileNotFoundError:
            ipv6_routes = ""
        routes6 = parse_ipv6_routes(ipv6_routes)
    except (OSError, ValueError) as error:
        return [Check("gateway", "default route", False,
                      f"the route tables cannot be read: {error}")], None
    try:
        try:
            if_inet6 = IF_INET6_FILE.read_text(encoding="utf-8")
        except FileNotFoundError:
            if_inet6 = ""
        own = local_addresses(FIB_TRIE_FILE.read_text(encoding="utf-8"), if_inet6)
    except (OSError, ValueError):
        own = set()  # unknown, so no address is spared: a listener of its own fails the check
    excluded = {address: "the container's own address" for address in sorted(own)}
    for (host, port), what in [*((service, "the service") for service in services),
                               (proxy, "the proxy")]:
        for address in sorted(_resolved(host)):
            excluded.setdefault(address, f"an address of {what} {name((host, port))}")
    candidates = gateway_candidates(routes, DOCKER_HOST_ADDRESSES, extra)
    routes_check = check_routes([*routes, *routes6])
    if not routes_check.ok:
        return [routes_check, *(
            Check("gateway", address, True, f"not probed: {excluded[address]}")
            if address in excluded else
            Check("gateway", address, False, "not probed: the default route out fails the check")
            for address in candidates)], None
    checks, seconds = check_gateways(candidates, excluded, SCAN_PORTS, SCAN_TIMEOUT,
                                     SCAN_CONCURRENCY)
    return [routes_check, *checks], seconds


def parse_addresses(text: str) -> list[str]:
    """The IP addresses of a comma-separated list; ValueError for an entry that is not one."""
    try:
        return [str(ipaddress.ip_address(item.strip())) for item in text.split(",")
                if item.strip()]
    except ValueError:
        raise ValueError(f"{text!r} is not a comma-separated list of IP addresses") from None


def _argument(parse):
    def parsed(text: str):
        try:
            return parse(text)
        except ValueError as error:
            raise argparse.ArgumentTypeError(str(error)) from None

    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Checks, inside a system's container, the "
                                                 "environment that section 8 requires.")
    parser.add_argument("--proxy", required=True, type=_argument(parse_proxy),
                        help="the egress proxy, as http://host:port")
    for flag, kind in (("--allow", "targets the proxy must open, each answering 401 with no key, "
                                   "and that must not resolve here"),
                       ("--deny", "targets the proxy must refuse with 403, and that must not "
                                  "resolve here"),
                       ("--direct", "targets a direct connection must not reach"),
                       ("--services", "the environment's own services, which must resolve and "
                                      "accept")):
        parser.add_argument(flag, default=[], type=_argument(parse_targets), action="extend",
                            help=f"comma-separated host:port {kind}")
    parser.add_argument("--gateway", default=[], type=_argument(parse_addresses),
                        action="extend", help="comma-separated further addresses whose every "
                                              "TCP port the gateway check probes")
    parser.add_argument("--mounts", default=[], type=_argument(parse_paths), action="extend",
                        help="comma-separated paths at which mounts are allowed")
    parser.add_argument("--report", required=True, type=Path, help="where to write the report")
    parser.add_argument("--timeout", default=TIMEOUT, type=float,
                        help=f"seconds for each connection (default {TIMEOUT:g})")
    args = parser.parse_args(argv)

    checks = [check_allow(args.proxy, target, args.timeout) for target in args.allow]
    checks += [check_deny(args.proxy, target, args.timeout) for target in args.deny]
    checks += [check_direct(target, args.timeout) for target in args.direct]
    checks += [check_service(target, args.timeout) for target in args.services]
    checks += dns_checks([*args.allow, *args.deny], args.services)
    gateway, seconds = gateway_checks(args.gateway, args.services, args.proxy)
    checks += gateway
    try:
        checks.append(check_mounts(MOUNTS_FILE.read_text(encoding="utf-8"), args.mounts))
    except OSError as error:
        checks.append(Check("mounts", str(MOUNTS_FILE), False, f"unreadable: {error}"))
    written = report(checks)
    written["ts"] = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    if seconds is not None:
        written["gateway_scan_seconds"] = round(seconds, 1)
    for check in checks:
        print(f"{'pass' if check.ok else 'FAIL'}  {check.kind:7} {check.target}: {check.detail}")
    try:
        args.report.write_text(json.dumps(written, indent=2) + "\n", encoding="utf-8")
    except OSError as error:
        print(f"netcheck: cannot write {args.report}: {error}", file=sys.stderr)
        return 1
    failed = sum(not check.ok for check in checks)
    scan = "" if seconds is None else f"; the gateway scan took {seconds:.1f} s"
    print(f"netcheck: {len(checks) - failed} of {len(checks)} checks passed{scan}; report in "
          f"{args.report}")
    return exit_code(written)


if __name__ == "__main__":
    sys.exit(main())
