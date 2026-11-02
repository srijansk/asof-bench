"""The plumbing of a run that does not depend on the system, for every system's runner to share
(Graphiti's ingestion, graphiti/ingest.py, is the first to use it). It holds:

- the output directory's lock (``hold_lock``): ``run.lock``, created with O_EXCL, touched every
  LOCK_HEARTBEAT_S while the invocation runs and removed at its end, so that one invocation at a
  time works in a directory; and ``cancelled_on_sigterm``, which turns the SIGTERM that ``docker
  stop`` sends into the cancellation of the running task, so that the invocation unwinds, records
  itself and releases the lock;
- run.json's invocation records: ``new_invocation`` makes one, ``recorded`` writes it to run.json
  at once and again at the end, with its elapsed time and its exit status, and ``write_json``
  writes the file atomically;
- ``read_jsonl`` and ``drop_unfinished_line``, which read a log of JSON lines and cut the
  unfinished last line that an interrupted write leaves;
- ``parse_image_ids``, the IDs of the images a run uses, as the launcher gives them in
  ASOFBENCH_IMAGE_IDS;
- ``wait_for_services`` and ``netcheck_preflight``: the wait for the services a system runs and
  the network check (env/netcheck.py), whose report goes to netcheck.json in the output
  directory;
- ``infrastructure_error`` and ``Stop``: which exceptions stop a run because the infrastructure
  failed (the API, or a service the system runs, is unavailable) and not because of the work
  item, and the first reason to stop the run.

An invocation's record in run.json is, in this order, ``command``, ``started_at``, the command's
own fields, ``images`` (the image IDs it ran with), ``elapsed_s``, ``exit``, ``stop`` (what
stopped it, as ``Stop`` holds it) and ``crash`` (the class of the exception that crashed it),
the last four null until they are known.

Exit status, the same for every runner that shares this module: EXIT_DONE, 0, when the run did
all its work; EXIT_FAILED, 1, when the run reached the end of its work with a unit logged as
failed; EXIT_STOPPED, 2, when the run is refused before it starts (``Refused``), the network
check fails or an infrastructure error stops it (a resume goes on after a stop); EXIT_FAILURES,
3, when consecutive failures stop it; and EXIT_CRASHED, 4, when it crashes, which run.json
records with the exception's class. A system may name the status 0 for what its work is, as
Graphiti's ingestion does.

Nothing here belongs to one system: a system gives the table of its infrastructure errors, the
names of its images, its services, mounts and network-check arguments, the waits and the README
that its refusals point at as arguments. It uses only the standard library.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Collection, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

# The files of an output directory that every system's run names the same way: the network
# check's report, the record of the run's invocations, and the lock.
NETCHECK, RUN, LOCK = "netcheck.json", "run.json", "run.lock"
EXIT_DONE, EXIT_FAILED, EXIT_STOPPED, EXIT_FAILURES, EXIT_CRASHED = 0, 1, 2, 3, 4
# Seconds between the touches that show a held lock's invocation is alive.
LOCK_HEARTBEAT_S = 30.0
# An image ID, as Docker reports it.
IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
# The exceptions that stop a run as infrastructure errors, by the top-level package that defines
# them and their class name, matched along each class's MRO and through an exception's causes
# (``infrastructure_error``): the part that every system shares, the API after the SDK's own
# retries, unreachable, rate-limited, overloaded or failing, or refusing the key, and the
# standard library's connection errors. A system adds the packages of the services it runs, in a
# table of its own that starts from a copy of this one; the classes are frozensets, so that no
# system can change what another's table holds.
INFRASTRUCTURE = {
    "anthropic": frozenset({"APIConnectionError", "RateLimitError", "OverloadedError",
                            "InternalServerError", "AuthenticationError",
                            "PermissionDeniedError"}),
    "builtins": frozenset({"ConnectionError"}),
}


class Refused(Exception):
    """The invocation cannot start, or cannot go on with the state it found: it does nothing."""


class Stop:
    """The first reason to stop the run: its exit status, its message and what run.json records
    of it; an attempt under way finishes."""

    def __init__(self) -> None:
        self.code: int | None = None
        self.reason: str | None = None
        self.record: dict | None = None
        self._requested = asyncio.Event()

    def request(self, code: int, reason: str, record: dict) -> None:
        if self.code is None:
            self.code, self.reason, self.record = code, reason, record
            self._requested.set()

    async def wait(self, seconds: float) -> bool:
        """Waits ``seconds``; True when a stop is requested meanwhile."""
        try:
            await asyncio.wait_for(self._requested.wait(), timeout=seconds)
        except TimeoutError:
            return False
        return True


def infrastructure_error(error: BaseException,
                         table: Mapping[str, Collection[str]]) -> str | None:
    """``package Class`` of the first exception in ``error``, its causes and, for a group, its
    members, that ``table`` names (package to class names, as INFRASTRUCTURE gives them); None
    when there is none, so that the error is the unit of work's own."""
    seen, pending = set(), [error]
    while pending:
        current = pending.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        for cls in type(current).__mro__:
            package = cls.__module__.split(".", 1)[0]
            if cls.__name__ in table.get(package, ()):
                return f"{package} {cls.__name__}"
        pending += [current.__cause__, current.__context__,
                    *getattr(current, "exceptions", ())]
    return None


def parse_image_ids(text: str | None, names: Sequence[str], readme: str) -> dict[str, str]:
    """The image IDs of ``name=sha256:...`` entries, comma-separated, one for each of ``names``;
    Refused otherwise, with a message that points at ``readme``."""
    entries = [entry.strip() for entry in (text or "").split(",") if entry.strip()]
    images = dict(entry.partition("=")[::2] for entry in entries)
    if (len(images) != len(entries) or set(images) != set(names)
            or not all(IMAGE_ID.fullmatch(value) for value in images.values())):
        raise Refused("ASOFBENCH_IMAGE_IDS must give the ID (sha256:...) of each image the run "
                      f"uses, as name=ID entries for {', '.join(names)} ({readme})")
    return {name: images[name] for name in names}


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def held_by(path: Path) -> str:
    """Who holds the lock at ``path``, and how old its heartbeat is, as far as the file says."""
    try:
        holder = json.loads(path.read_text(encoding="utf-8"))
        age = time.time() - path.stat().st_mtime
    except (OSError, ValueError):
        return "its holder cannot be read"
    return (f"{holder.get('command')} in container {holder.get('host')}, pid {holder.get('pid')}, "
            f"since {holder.get('since')}; its heartbeat is {age:.0f} s old")


@contextlib.contextmanager
def hold_lock(out_dir: Path, command: str, *, readme: str = "README.md") -> Iterator[None]:
    """Holds the output directory's lock for the invocation ``command``: the invocation creates
    run.lock with O_CREAT|O_EXCL, which excludes across the containers that share a bind mount
    (flock and lockf do not, on Docker Desktop's), touches it every LOCK_HEARTBEAT_S while it
    runs, and removes it at its end if it is still its own. Refused at once when the file
    exists: another invocation holds it, or a killed one left it, whose heartbeat has stopped.
    The refusal says that ``readme``, the system's README, tells how to check that a killed
    invocation is gone."""
    path = out_dir / LOCK
    token = uuid.uuid4().hex
    # The lock is the file's existence, not flock(2) or lockf(3). Two containers that shared one
    # bind-mounted directory on Docker Desktop were each granted an exclusive flock, and each a
    # lockf, on the same file there, so neither excludes the other invocation; an O_EXCL create
    # of a file that exists fails with EEXIST in the second container, so it does. The cost is
    # that an invocation killed outright leaves the file, which its stopped heartbeat shows and
    # the operator removes (the system's README says how).
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise Refused(f"another invocation holds the lock on this output directory ({LOCK}: "
                      f"{held_by(path)}): one invocation at a time. A killed invocation leaves "
                      f"its lock, whose heartbeat stops (a live one touches it every "
                      f"{LOCK_HEARTBEAT_S:g} s): once you are sure that its invocation is gone "
                      f"({readme} says how to check), remove {LOCK} and run again") from None
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"token": token, "command": command,
                                 "host": socket.gethostname(), "pid": os.getpid(),
                                 "since": utc_now()}) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    released = threading.Event()

    def heartbeat() -> None:
        while not released.wait(LOCK_HEARTBEAT_S):
            with contextlib.suppress(OSError):
                os.utime(path)

    beating = threading.Thread(target=heartbeat, name="run.lock heartbeat", daemon=True)
    beating.start()
    try:
        yield
    finally:
        released.set()
        beating.join()
        with contextlib.suppress(OSError, ValueError):
            if json.loads(path.read_text(encoding="utf-8")).get("token") == token:
                path.unlink()


@contextlib.contextmanager
def cancelled_on_sigterm() -> Iterator[None]:
    """Cancels the running task on SIGTERM, which ``docker stop`` sends and which a container's
    first process otherwise ignores, so that the invocation unwinds: it records its exit status
    4 in run.json, closes what it opened and releases its lock. The handler runs on the event
    loop, so while the wait for the services or the network check blocks the loop it cannot run;
    Docker's SIGKILL, 10 s after the SIGTERM, then ends the invocation outright, which leaves its
    lock and a null exit in run.json (the system's README says how to recover)."""
    loop, task = asyncio.get_running_loop(), asyncio.current_task()
    try:
        loop.add_signal_handler(signal.SIGTERM, task.cancel)
    except (NotImplementedError, RuntimeError, ValueError):
        yield
        return
    try:
        yield
    finally:
        loop.remove_signal_handler(signal.SIGTERM)


def read_jsonl(path: Path) -> list[dict]:
    """The JSON lines of ``path``, split at newlines only, none when it does not exist; Refused
    for a line that is not JSON."""
    if not path.exists():
        return []
    lines = []
    with open(path, "rb") as handle:
        for number, raw in enumerate(handle, 1):
            try:
                lines.append(json.loads(raw.decode("utf-8")))
            except ValueError:
                raise Refused(f"{path.name}: line {number} is not JSON") from None
    return lines


def drop_unfinished_line(path: Path) -> int:
    """Removes a last line that has no line break, which an interrupted write leaves, from the
    file at ``path``; returns the number of bytes removed."""
    if not path.exists():
        return 0
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return 0
    keep = data.rfind(b"\n") + 1
    with open(path, "r+b") as handle:
        handle.truncate(keep)
    return len(data) - keep


def write_json(path: Path, value: dict) -> None:
    """Writes ``value`` to ``path`` atomically: a temporary file in the same directory, then a
    rename over the old one, so that an interrupted write never leaves half a file."""
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def wait_for_services(services: Sequence[tuple[str, int]], wait_s: float,
                      retry_s: float) -> None:
    """Waits until every service accepts a TCP connection, trying every ``retry_s`` seconds, or
    ``wait_s`` seconds have passed (the network check then fails on the one that does not)."""
    deadline = time.monotonic() + wait_s
    for target in services:
        while True:
            try:
                socket.create_connection(target, timeout=retry_s).close()
                break
            except OSError:
                if time.monotonic() > deadline:
                    return
                time.sleep(retry_s)


def netcheck_preflight(out_dir: Path, *, script: Path, args: Sequence[str],
                       services: Sequence[tuple[str, int]], mounts: Sequence[str],
                       wait_s: float, retry_s: float) -> int:
    """Waits for the services (``wait_for_services`` with ``wait_s`` and ``retry_s``) and runs
    the network check, ``script`` (env/netcheck.py), which writes its report to
    ``out_dir/netcheck.json``; an earlier invocation's report is kept as ``netcheck-<n>.json``.
    ``args`` are the check's flags that every system shares (the proxy and the allowed, denied
    and direct targets); the system's ``services`` and ``mounts`` follow them as ``--services``
    and ``--mounts``, the services sorted, so that the report lists them in one order whatever
    order the system waits for them in. Returns the check's exit status."""
    wait_for_services(services, wait_s, retry_s)
    report = out_dir / NETCHECK
    if report.exists():
        n = 1
        while (out_dir / f"netcheck-{n}.json").exists():
            n += 1
        report.rename(out_dir / f"netcheck-{n}.json")
    done = subprocess.run([sys.executable, str(script), *args,
                           "--services", ",".join(f"{host}:{port}"
                                                  for host, port in sorted(services)),
                           "--mounts", ",".join(mounts), "--report", str(report)], check=False)
    return done.returncode


def new_invocation(command: str, images: Mapping[str, str], **fields) -> dict:
    """An invocation's record for run.json (the module's description): its ``command`` and the
    time it starts, now, then the command's own ``fields`` in the order given, a copy of
    ``images`` and the four fields that are null until known."""
    return {"command": command, "started_at": utc_now(), **fields, "images": dict(images),
            "elapsed_s": None, "exit": None, "stop": None, "crash": None}


@contextlib.contextmanager
def recorded(record: dict, invocation: dict, run_path: Path, begun: float) -> Iterator[None]:
    """Records ``invocation`` in run.json at once, and again at the end with its elapsed time
    and, unless the body set it, its exit status: 2 for a refusal, 4 with the exception's class
    for a crash."""
    record["invocations"].append(invocation)
    write_json(run_path, record)
    try:
        yield
    except Refused:
        if invocation["exit"] is None:
            invocation["exit"] = EXIT_STOPPED
        raise
    except BaseException as error:
        invocation["exit"], invocation["crash"] = EXIT_CRASHED, type(error).__name__
        raise
    finally:
        if invocation["elapsed_s"] is None:
            invocation["elapsed_s"] = round(time.monotonic() - begun, 3)
        write_json(run_path, record)
