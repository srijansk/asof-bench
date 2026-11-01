"""The network check: run inside a system's container, it checks the environment that section 8
of the registration requires before the system ingests or answers anything.

    python netcheck.py --proxy http://egress:3128 --allow api.anthropic.com:443 \\
        --deny github.com:443,huggingface.co:443,pypi.org:443 \\
        --deny peps.python.org:443,eips.ethereum.org:443,ercs.ethereum.org:443 \\
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
- mounts: /proc/mounts may show mounts only at the ``--mounts`` paths and beneath them, besides
  the container's own root, the /etc/hosts, /etc/hostname and /etc/resolv.conf that Docker
  manages, and the kernel's own filesystems at and beneath /proc, /sys and /dev. A mount anywhere
  else, of any type, or one beneath those three of a type that can hold files from the host,
  fails the check and is named.

It exits 0 only when every check passes, and 1 otherwise. It uses only the standard library.
"""

from __future__ import annotations

import argparse
import http.client
import ipaddress
import json
import re
import socket
import ssl
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class Check:
    """One check and its result: ``kind`` is allow, deny, direct, service, dns or mounts."""

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
    try:
        checks.append(check_mounts(MOUNTS_FILE.read_text(encoding="utf-8"), args.mounts))
    except OSError as error:
        checks.append(Check("mounts", str(MOUNTS_FILE), False, f"unreadable: {error}"))
    written = report(checks)
    written["ts"] = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    for check in checks:
        print(f"{'pass' if check.ok else 'FAIL'}  {check.kind:7} {check.target}: {check.detail}")
    try:
        args.report.write_text(json.dumps(written, indent=2) + "\n", encoding="utf-8")
    except OSError as error:
        print(f"netcheck: cannot write {args.report}: {error}", file=sys.stderr)
        return 1
    failed = sum(not check.ok for check in checks)
    print(f"netcheck: {len(checks) - failed} of {len(checks)} checks passed; report in "
          f"{args.report}")
    return exit_code(written)


if __name__ == "__main__":
    sys.exit(main())
