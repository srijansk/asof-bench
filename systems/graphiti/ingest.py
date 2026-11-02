"""Graphiti's ingestion of one arm's stream, inside the run environment (systems/README.md).

    ingest --stream /input/<file>.jsonl --arm H+|H-|CF --out /output [--resume]
           [--skip-episode <episode_id>]...

Every invocation first takes the output directory's lock, by creating ``run.lock`` there with
O_EXCL (``hold_lock``), and holds it to its end; a second invocation on the same directory stops at
once. A SIGTERM, which ``docker stop`` sends, cancels the invocation, which then records itself
and releases the lock (``cancelled_on_sigterm``). Then it waits for the
environment's services and runs the network check (env/netcheck.py, with the README's arguments,
``--services embedder:8080,neo4j:7687`` and ``--mounts`` its input, output and secret paths),
whose report it writes to ``/output/netcheck.json``, and ingests nothing unless every check
passes. Then it builds Graphiti as adapter.py configures it, checks that the store holds exactly
the episodes that the ingest log records as ingested (none, for a new run), and ingests the
stream's episodes: one group per corpus (Graphiti's ``group_id``), the episodes of a group
strictly in stream order, each awaited before the next, and the two groups concurrently. While an
episode is ingested, every model call is recorded in the usage log under its id, the invocation
(its place in run.json's ``invocations``, from 1) and the attempt (from 1, within the invocation),
so that a call that raised can be placed in the attempt it fell in (``attempt_outcome``).

An episode that raises is retried in place, after RETRY_DELAYS_S (two more attempts), with the
same previous episodes and saga predecessor; one that still raises is logged as failed and is
never attempted again, by this run or a resumed one, so that no episode is ever ingested after a
later one of its group. An infrastructure error (``infrastructure_error``: Neo4j unavailable or a
session lost; the embedder unreachable or failing; after the SDK's own retries, the API
unreachable, rate-limited, overloaded or failing, or refusing the key) stops the run with exit
status 2 and leaves the episode unlogged, so that a resumed run attempts it again; so does a failed
attempt that left the episode in the store. MAX_CONSECUTIVE_FAILURES logged failures in a row in
a group stop the run with exit status 3. A stop lets each group finish the attempt it is making
and starts nothing more. ``--resume`` continues each group from its first episode with no log
line, after checking that the stream and the arm are those the run began with. With
``--skip-episode``, which needs ``--resume``, the operator logs an episode that keeps stopping the
run as failed (SKIPPED) before the resumed run goes on; it must be the first episode of its group
with no log line.

A proposal's previous episodes are therefore its last EPISODE_WINDOW_LEN ingested episodes, all
recorded before the episode, oldest first, and its saga predecessor the last of them.

It writes, in ``--out``, and none of these files holds a local path:

- ``run.json``, written atomically: the input stream's file name and SHA-256 and the arm,
  recorded when the run begins, and each invocation's command, start, whether it resumed, the
  episodes the operator skipped, the image IDs it ran with (``ASOFBENCH_IMAGE_IDS``, given by the
  launcher), its elapsed time, its exit status, and the class of the error that stopped or
  crashed it;
- ``usage.jsonl``: one line per model call (asofbench_systems.usage);
- ``ingest-log.jsonl``: one line per episode, after its last attempt, with ``episode_id``,
  ``proposal``, ``group_id``, ``episode_uuid`` (null for a failed episode), ``ok``, ``error``
  (null when the first attempt succeeded, else the last failed attempt's exception class and the
  first 300 characters of its message, the key taken out, or SKIPPED), ``started_at`` (its first
  attempt), ``wall_s`` (its attempts' time, the waits between them not included), ``text_chars``
  (its ``text`` in code points, as the manifest counts them), ``attempts`` and ``invocation``
  (the invocation that logged it);
- ``summary.json``, written atomically: the totals over the whole log and every usage line, each
  counted once, as ``summarize`` describes.

Exit status: 0 when every episode of the stream is ingested; 1 when the run reached the stream's
end with an episode logged as failed; 2 when the run is refused before it starts (another
invocation holds the lock, a bad argument, an output directory that holds a run without
``--resume``, a log, stream, arm or store that does not match), when the network check fails, or
when an infrastructure error stops it; 3 when consecutive failures stop it; 4 when it crashes,
which run.json records with the exception's class.

``remove_episodes`` (``python -m graphiti.adapter --remove-episodes``) removes from the store an
episode that an attempt left there, under the same lock, after the same network check, and
records its invocation in run.json.

The state, log and summary functions use only the standard library; graphiti-core and anthropic
are imported only by adapter.open_engine.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from collections.abc import Awaitable, Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from asofbench_systems import llm, usage
from asofbench_systems.stream import read_episodes
from graphiti import adapter

ARMS = ("H+", "H-", "CF")
LOG, USAGE, SUMMARY, NETCHECK, RUN, LOCK = ("ingest-log.jsonl", "usage.jsonl", "summary.json",
                                            "netcheck.json", "run.json", "run.lock")
LOG_FIELDS = ("episode_id", "proposal", "group_id", "episode_uuid", "ok", "error", "started_at",
              "wall_s", "text_chars", "attempts", "invocation")
# The outcomes of the attempt that a call which raised fell in (``attempt_outcome``), by which
# summary.json splits its ``api_errors``.
ATTEMPT_OUTCOMES = ("ingested", "failed", "stopped", "unknown")
# The fields of a stream's episode (section 8).
EPISODE_FIELDS = ("episode_id", "corpus", "repo", "proposal", "ref", "recorded_at", "commit",
                  "message", "kind", "text")
# Seconds to wait before an episode's second and third attempts.
RETRY_DELAYS_S = (30.0, 120.0)
# Logged failures in a row, in one group, that stop the run.
MAX_CONSECUTIVE_FAILURES = 5
EXIT_INGESTED, EXIT_FAILED, EXIT_STOPPED, EXIT_FAILURES, EXIT_CRASHED = 0, 1, 2, 3, 4
SKIPPED = "skipped by the operator after repeated infrastructure stops"
# The usage context of the model calls that remove_episode makes.
REMOVAL_CONTEXT = "remove-episode:"
# How much of an exception's message the log keeps.
ERROR_CHARS = 300
# The prompt length up to which the price basis's prices hold (section 13).
PRICED_INPUT_TOKENS = 100_000
# The exceptions that stop a run as infrastructure errors, by the top-level package that defines
# them and their class name, matched along each class's MRO and through an exception's causes.
# The SDKs raise theirs only after their own retries.
INFRASTRUCTURE = {
    "neo4j": {"ServiceUnavailable", "SessionExpired", "SessionError", "DatabaseUnavailable"},
    "openai": {"APIConnectionError", "InternalServerError"},
    "anthropic": {"APIConnectionError", "RateLimitError", "OverloadedError", "InternalServerError",
                  "AuthenticationError", "PermissionDeniedError"},
    "graphiti_core": {"RateLimitError"},
    "builtins": {"ConnectionError"},
}
# The images a run records, as the launcher gives them in ASOFBENCH_IMAGE_IDS.
IMAGE_NAMES = ("graphiti", "egress", "embedder", "neo4j")
IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
# The network check, as systems/README.md runs it, with this system's services and mounts.
NETCHECK_SCRIPT = Path(__file__).resolve().parents[1] / "env" / "netcheck.py"
NETCHECK_ARGS = (
    "--proxy", "http://egress:3128", "--allow", "api.anthropic.com:443",
    "--deny", "github.com:443,huggingface.co:443,pypi.org:443,peps.python.org:443,"
              "eips.ethereum.org:443,ercs.ethereum.org:443",
    "--deny", "host.docker.internal:443,gateway.docker.internal:443",
    "--direct", "140.82.112.3:443,1.1.1.1:443",
    "--services", "embedder:8080,neo4j:7687",
    "--mounts", "/input,/output,/run/secrets",
)
SERVICES = (("neo4j", 7687), ("embedder", 8080))
# Seconds between the touches that show a held lock's invocation is alive.
LOCK_HEARTBEAT_S = 30.0
# How long the run waits for the services to accept connections (the embedder loads its model
# first), and how often it tries.
SERVICE_WAIT_S, SERVICE_RETRY_S = 600.0, 2.0
CHUNK = 1 << 20


class Refused(Exception):
    """The invocation cannot start, or cannot go on with the store it found: nothing is
    ingested."""


@dataclass
class ProposalState:
    """A proposal's saga predecessor and its window of previous episodes."""

    window: deque = field(default_factory=lambda: deque(maxlen=adapter.EPISODE_WINDOW_LEN))
    saga_previous: str | None = None

    def ingested(self, uuid: str) -> None:
        self.window.append(uuid)
        self.saga_previous = uuid


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


def infrastructure_error(error: BaseException) -> str | None:
    """``package Class`` of the first exception in ``error``, its causes and, for a group, its
    members, that INFRASTRUCTURE names; None when there is none, so that the error is the
    episode's."""
    seen, pending = set(), [error]
    while pending:
        current = pending.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        for cls in type(current).__mro__:
            package = cls.__module__.split(".", 1)[0]
            if cls.__name__ in INFRASTRUCTURE.get(package, ()):
                return f"{package} {cls.__name__}"
        pending += [current.__cause__, current.__context__,
                    *getattr(current, "exceptions", ())]
    return None


def parse_image_ids(text: str | None) -> dict[str, str]:
    """The image IDs of ``name=sha256:...`` entries, comma-separated, one for each of
    IMAGE_NAMES; Refused otherwise."""
    entries = [entry.strip() for entry in (text or "").split(",") if entry.strip()]
    images = dict(entry.partition("=")[::2] for entry in entries)
    if (len(images) != len(entries) or set(images) != set(IMAGE_NAMES)
            or not all(IMAGE_ID.fullmatch(value) for value in images.values())):
        raise Refused("ASOFBENCH_IMAGE_IDS must give the ID (sha256:...) of each image the run "
                      f"uses, as name=ID entries for {', '.join(IMAGE_NAMES)} (systems/graphiti/"
                      "README.md)")
    return {name: images[name] for name in IMAGE_NAMES}


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
def hold_lock(out_dir: Path, command: str = "ingest") -> Iterator[None]:
    """Holds the output directory's lock: the invocation creates run.lock with O_CREAT|O_EXCL,
    which excludes across the containers that share a bind mount (flock and lockf do not, on
    Docker Desktop's), touches it every LOCK_HEARTBEAT_S while it runs, and removes it at its end
    if it is still its own. Refused at once when the file exists: another invocation holds it,
    or a killed one left it, whose heartbeat has stopped."""
    path = out_dir / LOCK
    token = uuid.uuid4().hex
    # The lock is the file's existence, not flock(2) or lockf(3). Two containers that shared one
    # bind-mounted directory on Docker Desktop were each granted an exclusive flock, and each a
    # lockf, on the same file there, so neither excludes the other invocation; an O_EXCL create
    # of a file that exists fails with EEXIST in the second container, so it does. The cost is
    # that an invocation killed outright leaves the file, which its stopped heartbeat shows and
    # the operator removes (README.md).
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise Refused(f"another invocation holds the lock on this output directory ({LOCK}: "
                      f"{held_by(path)}): one invocation at a time. A killed invocation leaves "
                      f"its lock, whose heartbeat stops (a live one touches it every "
                      f"{LOCK_HEARTBEAT_S:g} s): once you are sure that its invocation is gone "
                      f"(README.md says how to check), remove {LOCK} and run again") from None
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
    4 in run.json, closes Graphiti and releases its lock."""
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


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


def read_stream(stream: Path) -> list[dict]:
    """The stream's episodes; Refused for a stream that cannot be read, an episode that lacks a
    field of section 8, or an episode id that occurs twice (the log is keyed by it)."""
    try:
        episodes = list(read_episodes(str(stream)))
    except (OSError, ValueError) as error:
        raise Refused(f"the stream cannot be read: {error}") from None
    ids = set()
    for number, e in enumerate(episodes, 1):
        if missing := [name for name in EPISODE_FIELDS if name not in e]:
            raise Refused(f"{stream.name}: episode {number} has no {', '.join(missing)}")
        if e["episode_id"] in ids:
            raise Refused(f"{stream.name}: episode {number}'s id, {e['episode_id']}, occurs twice")
        ids.add(e["episode_id"])
    return episodes


def check_log(episodes: list[dict], log: list[dict]) -> None:
    """Refused unless every line of ``log`` is one episode's outcome, with that episode's
    proposal and group and the invocation and attempts that ended it, each episode has at most
    one line, and the episodes of each group that the log holds are the group's first ones, in
    stream order."""
    by_id = {e["episode_id"]: e for e in episodes}
    problems, logged = [], set()
    for number, line in enumerate(log, 1):
        if not isinstance(line, dict) or set(line) != set(LOG_FIELDS):
            problems.append(f"line {number} is not an episode line")
            continue
        e = by_id.get(line["episode_id"])
        if e is None:
            problems.append(f"line {number}: {line['episode_id']} is not in the stream")
        elif (line["proposal"], line["group_id"]) != (e["proposal"], e["corpus"]):
            problems.append(f"line {number}: {line['episode_id']} has another proposal or group")
        elif line["episode_id"] in logged:
            problems.append(f"line {number}: {line['episode_id']} has a line already")
        elif line["ok"] is True and (not isinstance(line["episode_uuid"], str)
                                     or not line["episode_uuid"]):
            problems.append(f"line {number}: an ingested episode with no uuid")
        elif line["ok"] is not True and (line["ok"] is not False
                                         or line["episode_uuid"] is not None):
            problems.append(f"line {number}: a failed episode with a uuid, or no ok")
        elif not (type(line["invocation"]) is int and line["invocation"] >= 1
                  and type(line["attempts"]) is int
                  and line["attempts"] >= (1 if line["ok"] else 0)):
            problems.append(f"line {number}: no invocation or no attempts")
        logged.add(line["episode_id"])
    if not problems:
        for group, members in by_group(episodes).items():
            order = [line["episode_id"] for line in log if line["group_id"] == group]
            if order != [e["episode_id"] for e in members[:len(order)]]:
                problems.append(f"the log's {group} episodes are not the group's first ones, in "
                                "stream order")
    if problems:
        shown = "; ".join(problems[:5]) + (f"; and {len(problems) - 5} more"
                                           if len(problems) > 5 else "")
        raise Refused(f"the ingest log does not match the stream: {shown}")


def pending(episodes: list[dict], log: list[dict]) -> list[dict]:
    """The episodes that have no log line, in stream order."""
    logged = {line["episode_id"] for line in log}
    return [e for e in episodes if e["episode_id"] not in logged]


def plan_skips(episodes: list[dict], log: list[dict], skip: list[str],
               invocation: int) -> list[dict]:
    """The log lines of the episodes the operator skips in ``invocation``, in the order given:
    each must be the first episode of its group with no log line, the earlier skips counted;
    Refused otherwise."""
    by_id = {e["episode_id"]: e for e in episodes}
    logged = {line["episode_id"] for line in log}
    lines = []
    for episode_id in skip:
        e = by_id.get(episode_id)
        if e is None:
            raise Refused(f"--skip-episode {episode_id} is not an episode of the stream")
        first = next((member for member in by_group(episodes)[e["corpus"]]
                      if member["episode_id"] not in logged), None)
        if first is None or first["episode_id"] != episode_id:
            first_id = None if first is None else first["episode_id"]
            raise Refused(f"--skip-episode {episode_id} is not the first {e['corpus']} episode "
                          f"with no log line ({first_id}): the group's order must hold")
        lines.append({"episode_id": episode_id, "proposal": e["proposal"],
                      "group_id": e["corpus"], "episode_uuid": None, "ok": False,
                      "error": SKIPPED, "started_at": utc_now(), "wall_s": 0.0,
                      "text_chars": len(e["text"]), "attempts": 0, "invocation": invocation})
        logged.add(episode_id)
    return lines


def rebuild_states(log: list[dict]) -> dict[str, ProposalState]:
    """Each proposal's state after the episodes that ``log`` records as ingested, in its order."""
    states: dict[str, ProposalState] = {}
    for line in log:
        if line["ok"]:
            states.setdefault(line["proposal"], ProposalState()).ingested(line["episode_uuid"])
    return states


def consecutive_failures(log: list[dict]) -> dict[str, int]:
    """Each group's logged failures since its last ingested episode."""
    failures: dict[str, int] = {}
    for line in log:
        failures[line["group_id"]] = 0 if line["ok"] else failures.get(line["group_id"], 0) + 1
    return failures


def by_group(episodes: Iterable[dict]) -> dict[str, list[dict]]:
    """The episodes of each corpus, Graphiti's group, in stream order."""
    groups: dict[str, list[dict]] = {}
    for e in episodes:
        groups.setdefault(e["corpus"], []).append(e)
    return groups


def input_tokens(line: dict) -> int:
    """A call's input tokens, those written to and read from the prompt cache included."""
    return (line["input_tokens"] + line["cache_creation_input_tokens"]
            + line["cache_read_input_tokens"])


def totals(lines: list[dict]) -> dict:
    """The tokens, the responses (``calls``) and the calls that raised (``api_errors``) of
    ``lines``."""
    return {"input_tokens": sum(input_tokens(line) for line in lines),
            "output_tokens": sum(line["output_tokens"] for line in lines),
            "calls": sum(1 for line in lines if not line.get("error")),
            "api_errors": sum(1 for line in lines if line.get("error"))}


def attempt_outcome(line: dict, logged: dict[str, dict]) -> str:
    """The outcome of the attempt that ``line``, a usage line of an episode, fell in, from the
    episode's log line in ``logged`` (by episode id): ``ingested`` for the attempt that ingested
    the episode; ``failed`` for one that failed, retried in place or the last of an episode
    logged as failed; ``stopped`` for one whose invocation left the episode unlogged (a stop, a
    crash or a kill), whatever a later invocation did with it; and ``unknown`` for a line that
    names no attempt that the log can place. A resumed invocation numbers its attempts afresh, so
    an attempt is named by its invocation and its number together."""
    invocation, attempt = line.get("invocation"), line.get("attempt")
    if type(invocation) is not int or type(attempt) is not int or attempt < 1:
        return "unknown"
    entry = logged.get(line["context"])
    if entry is None or entry["invocation"] > invocation:
        return "stopped"
    if entry["invocation"] < invocation or attempt > entry["attempts"]:
        return "unknown"
    return "ingested" if entry["ok"] and attempt == entry["attempts"] else "failed"


def summarize(episodes: list[dict], log: list[dict], usage_lines: list[dict], record: dict, *,
              graphiti_core: str) -> dict:
    """summary.json: the run's stream (file name and SHA-256) and arm; the stream's ``episodes``
    and ``text_chars`` (code points of ``text``); the episodes ``ok`` (ingested) and ``failed``
    (logged as failed, skipped ones included); the ingestion attempts' ``input_tokens`` (cache
    tokens included), ``output_tokens``, ``calls`` (responses), ``api_errors`` (calls that
    raised), ``api_errors_by_outcome`` (those calls by the outcome of the attempt they fell in,
    as ``attempt_outcome`` gives it) and ``calls_over_100k_input`` (prompts beyond the price
    basis's 100,000 tokens), the lines of every attempt counted, failed ones included; the same
    totals but the split for the calls that removing episodes made (``removal``) and for any
    other line (``unattributed``); ``wall_s``, the sum of every episode's attempt time, which
    counts the two groups' time apart, and
    ``elapsed_s``, the wall-clock time of the run's invocations, summed (an invocation that was
    killed has none); ``invocations``; ``exit``, the exit status of the last invocation, the one
    that wrote it; ``graphiti_core``; ``model``, the models of the usage lines (one name, or a
    list when there are several, null when there is no line); and ``images``, each image's IDs
    over the invocations. Each line is counted once."""
    ids = {e["episode_id"] for e in episodes}
    logged = {line["episode_id"]: line for line in log if line["episode_id"] in ids}
    outcome = {episode_id: line["ok"] for episode_id, line in logged.items()}
    ingestion, removal, unattributed = [], [], []
    by_outcome = dict.fromkeys(ATTEMPT_OUTCOMES, 0)
    for line in usage_lines:
        context = line["context"]
        if context in ids:
            ingestion.append(line)
            if line.get("error"):
                by_outcome[attempt_outcome(line, logged)] += 1
        elif isinstance(context, str) and context.startswith(REMOVAL_CONTEXT):
            removal.append(line)
        else:
            unattributed.append(line)
    models = sorted({line["model"] for line in usage_lines if line["model"]})
    invocations = record["invocations"]
    images = {name: sorted({invocation["images"][name] for invocation in invocations})
              for name in IMAGE_NAMES}
    return {"arm": record["arm"], "stream": record["stream"],
            "stream_sha256": record["stream_sha256"], "episodes": len(episodes),
            "ok": sum(outcome.values()), "failed": len(outcome) - sum(outcome.values()),
            "text_chars": sum(len(e["text"]) for e in episodes),
            **totals(ingestion), "api_errors_by_outcome": by_outcome,
            "calls_over_100k_input": sum(input_tokens(line) > PRICED_INPUT_TOKENS
                                         for line in ingestion),
            "removal": totals(removal), "unattributed": totals(unattributed),
            "wall_s": round(sum(line["wall_s"] for line in log), 3),
            "elapsed_s": round(sum(invocation["elapsed_s"] or 0 for invocation in invocations),
                               3),
            "invocations": len(invocations), "exit": invocations[-1]["exit"],
            "graphiti_core": graphiti_core,
            "model": (models[0] if len(models) == 1 else models) if models else None,
            "images": images}


def error_text(error: BaseException, key_file: str) -> str:
    """The log's ``error``: the exception's class and the first ERROR_CHARS characters of its
    message, with the key taken out first."""
    message = llm.redact(str(error), key_file)[:ERROR_CHARS]
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def append_line(path: Path, line: dict) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, ensure_ascii=False) + "\n")


def write_json(path: Path, value: dict) -> None:
    """Writes ``value`` to ``path`` atomically: a temporary file in the same directory, then a
    rename over the old one, so that an interrupted write never leaves half a file."""
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


async def check_store(engine, log: list[dict]) -> None:
    """Refused unless the store holds exactly the episodes that ``log`` records as ingested."""
    stored = await engine.stored_episodes()
    logged = {line["episode_uuid"] for line in log if line["ok"]}
    held = {episode["uuid"] for episode in stored}
    if missing := logged - held:
        raise Refused(f"the store lacks {len(missing)} of the {len(logged)} episodes that the log "
                      "records as ingested: resume on the store the run began with")
    if extra := [episode for episode in stored if episode["uuid"] not in logged]:
        shown = " ".join(f"{episode['uuid']} ({episode['name']})" for episode in extra[:10])
        what = ("a new run needs an empty store (a new volume)" if not logged else
                "left by an attempt that wrote to the store before it stopped; remove them with "
                "`python -m graphiti.adapter --remove-episodes <uuid>... --out /output` "
                "(README.md), then resume")
        raise Refused(f"the store holds {len(extra)} episodes that the log does not record as "
                      f"ingested, {what}: {shown}")


async def ingest_group(engine, group: str, episodes: list[dict],
                       states: dict[str, ProposalState], failures: dict[str, int], stop: Stop,
                       log_path: Path, key_file: str, delays: tuple[float, ...],
                       progress: Callable[[str], None], *, invocation: int) -> None:
    """Ingests ``episodes``, one group's, one at a time in their order (the module's
    description), in the run's ``invocation``."""
    for number, e in enumerate(episodes, 1):
        if stop.code is not None:
            return
        state = states.setdefault(e["proposal"], ProposalState())
        started_at, attempts, spent, uuid, error = utc_now(), 0, 0.0, None, None
        while True:
            attempts += 1
            start = time.monotonic()
            try:
                with usage.attributed(e["episode_id"], invocation, attempts):
                    arguments = adapter.episode_arguments(
                        e, previous_episode_uuids=list(state.window),
                        saga_previous_episode_uuid=state.saga_previous, source=engine.source)
                    uuid = await engine.add_episode(**arguments)
            except Exception as raised:  # noqa: BLE001 - classified below, and never lost
                if infrastructure := infrastructure_error(raised):
                    stop.request(EXIT_STOPPED, f"{e['episode_id']} (attempt {attempts}): "
                                 f"{infrastructure}: {error_text(raised, key_file)}",
                                 {"episode_id": e["episode_id"], "error": infrastructure})
                    progress(f"{group} {number}/{len(episodes)} {e['episode_id']}: stopped by "
                             f"an infrastructure error ({infrastructure}); not logged")
                    return
                error = error_text(raised, key_file)
            finally:
                spent += time.monotonic() - start
            if uuid is not None:
                break
            try:
                left = [s["uuid"] for s in await engine.stored_episodes()
                        if s["name"] == e["episode_id"]]
            except Exception as raised:  # noqa: BLE001 - the store cannot be read: stop
                stop.request(EXIT_STOPPED, f"{e['episode_id']}: the store cannot be read: "
                             f"{error_text(raised, key_file)}",
                             {"episode_id": e["episode_id"], "error": type(raised).__name__})
                return
            if left:
                stop.request(EXIT_STOPPED, f"{e['episode_id']}: a failed attempt left it in the "
                             f"store ({' '.join(left)}); remove it, then resume",
                             {"episode_id": e["episode_id"], "error": "left in the store"})
                return
            if attempts > len(delays):
                break
            progress(f"{group} {number}/{len(episodes)} {e['episode_id']}: attempt {attempts} "
                     f"failed ({error.split(':', 1)[0]}); again in {delays[attempts - 1]:g} s")
            if await stop.wait(delays[attempts - 1]):
                return
        append_line(log_path, {"episode_id": e["episode_id"], "proposal": e["proposal"],
                               "group_id": e["corpus"], "episode_uuid": uuid,
                               "ok": uuid is not None, "error": error, "started_at": started_at,
                               "wall_s": round(spent, 3), "text_chars": len(e["text"]),
                               "attempts": attempts, "invocation": invocation})
        if uuid is not None:
            state.ingested(uuid)
            failures[group] = 0
        else:
            failures[group] = failures.get(group, 0) + 1
        outcome = "ok" if uuid is not None else f"FAILED ({error.split(':', 1)[0]})"
        progress(f"{group} {number}/{len(episodes)} {e['episode_id']}: {outcome} in "
                 f"{spent:.1f} s, {attempts} attempt{'s' if attempts > 1 else ''}")
        if failures[group] >= MAX_CONSECUTIVE_FAILURES:
            stop.request(EXIT_FAILURES, f"{failures[group]} consecutive {group} episodes failed",
                         {"group": group, "error": f"{failures[group]} consecutive failures"})
            return


def wait_for_services(services=SERVICES, wait_s: float = SERVICE_WAIT_S) -> None:
    """Waits until every service accepts a TCP connection, or ``wait_s`` seconds have passed (the
    network check then fails on the one that does not)."""
    deadline = time.monotonic() + wait_s
    for target in services:
        while True:
            try:
                socket.create_connection(target, timeout=SERVICE_RETRY_S).close()
                break
            except OSError:
                if time.monotonic() > deadline:
                    return
                time.sleep(SERVICE_RETRY_S)


def netcheck_preflight(out_dir: Path) -> int:
    """Waits for the services and runs the network check, which writes its report to
    ``out_dir/netcheck.json``; an earlier invocation's report is kept as ``netcheck-<n>.json``.
    Returns the check's exit status."""
    wait_for_services()
    report = out_dir / NETCHECK
    if report.exists():
        n = 1
        while (out_dir / f"netcheck-{n}.json").exists():
            n += 1
        report.rename(out_dir / f"netcheck-{n}.json")
    done = subprocess.run([sys.executable, str(NETCHECK_SCRIPT), *NETCHECK_ARGS,
                           "--report", str(report)], check=False)
    return done.returncode


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


async def run(stream: Path, arm: str, out_dir: Path, *, resume: bool, images: dict[str, str],
              skip: list[str] = (), open_engine: Callable[..., Awaitable] = adapter.open_engine,
              preflight: Callable[[Path], int] | None = netcheck_preflight,
              key_file: str = llm.KEY_FILE, delays: tuple[float, ...] = RETRY_DELAYS_S,
              progress: Callable[[str], None] = print) -> int:
    """One invocation of an ingestion run (the module's description); Refused when it cannot
    start, or finds a store that does not match its log."""
    if arm not in ARMS:
        raise Refused(f"--arm {arm!r} is not one of {', '.join(ARMS)}")
    if skip and not resume:
        raise Refused("--skip-episode needs --resume")
    if not out_dir.is_dir():
        raise Refused(f"{out_dir} is not a directory")
    with hold_lock(out_dir, "ingest"), cancelled_on_sigterm():
        return await _run(stream, arm, out_dir, resume=resume, images=images, skip=list(skip),
                          open_engine=open_engine, preflight=preflight, key_file=key_file,
                          delays=delays, progress=progress)


async def _run(stream: Path, arm: str, out_dir: Path, *, resume: bool, images: dict[str, str],
               skip: list[str], open_engine, preflight, key_file: str,
               delays: tuple[float, ...], progress: Callable[[str], None]) -> int:
    begun = time.monotonic()
    log_path, usage_path, run_path = out_dir / LOG, out_dir / USAGE, out_dir / RUN
    episodes = read_stream(stream)
    stream_sha256 = sha256_file(stream)
    if resume:
        if not run_path.exists():
            raise Refused(f"{out_dir} holds no run to resume (no {RUN}): start one without "
                          "--resume")
        record = json.loads(run_path.read_text(encoding="utf-8"))
        if (record["stream_sha256"], record["arm"]) != (stream_sha256, arm):
            raise Refused(f"the run began with the {record['arm']} stream {record['stream']} "
                          f"(SHA-256 {record['stream_sha256']}), not with the {arm} stream "
                          f"{stream.name} (SHA-256 {stream_sha256})")
        for path in (log_path, usage_path):
            if dropped := drop_unfinished_line(path):
                progress(f"ingest: dropped the unfinished last line of {path.name} ({dropped} "
                         "bytes), left by an interrupted invocation")
    else:
        if any((out_dir / name).exists() for name in (LOG, USAGE, SUMMARY, RUN)):
            raise Refused(f"{out_dir} already holds a run: resume it with --resume, or give a new "
                          "output directory and a new store")
        record = {"stream": stream.name, "stream_sha256": stream_sha256, "arm": arm,
                  "invocations": []}
    log = read_jsonl(log_path)
    check_log(episodes, log)
    # This invocation's place in run.json, which the lock keeps from changing meanwhile.
    number = len(record["invocations"]) + 1
    skip_lines = plan_skips(episodes, log, skip, number)
    invocation = {"command": "ingest", "started_at": utc_now(), "resume": resume, "skipped": [],
                  "images": dict(images), "elapsed_s": None, "exit": None, "stop": None,
                  "crash": None}
    with recorded(record, invocation, run_path, begun):
        if preflight is not None and preflight(out_dir) != 0:
            progress(f"ingest: the network check failed (see {NETCHECK}); nothing was ingested")
            invocation["exit"], invocation["stop"] = EXIT_STOPPED, {"error": "network check"}
            return EXIT_STOPPED
        stop = Stop()
        engine = await open_engine(usage.UsageLog(str(usage_path)), key_file)
        try:
            await check_store(engine, log)
            for line in skip_lines:
                append_line(log_path, line)
                invocation["skipped"].append(line["episode_id"])
                progress(f"ingest: {line['episode_id']} logged as failed: {SKIPPED}")
            if skip_lines:
                write_json(run_path, record)
                log = read_jsonl(log_path)
            todo = pending(episodes, log)
            states, failures = rebuild_states(log), consecutive_failures(log)
            groups = by_group(todo)
            progress(f"ingest: {len(todo)} of the stream's {len(episodes)} episodes to ingest, "
                     f"in {len(groups)} groups "
                     f"({', '.join(f'{g} {len(e)}' for g, e in groups.items())})")
            async with asyncio.TaskGroup() as tasks:
                for group, members in groups.items():
                    tasks.create_task(ingest_group(engine, group, members, states, failures,
                                                   stop, log_path, key_file, delays, progress,
                                                   invocation=number))
        finally:
            await engine.close()
        log = read_jsonl(log_path)
        outcomes = {line["episode_id"]: line["ok"] for line in log}
        code = (stop.code if stop.code is not None else
                EXIT_INGESTED if all(outcomes.get(e["episode_id"]) for e in episodes)
                else EXIT_FAILED)
        invocation["exit"], invocation["stop"] = code, stop.record
        invocation["elapsed_s"] = round(time.monotonic() - begun, 3)
        summary = summarize(episodes, log, read_jsonl(usage_path), record,
                            graphiti_core=engine.version)
        write_json(out_dir / SUMMARY, summary)
        if stop.reason is not None:
            progress(f"ingest: stopped (exit {code}): {stop.reason}")
        progress(f"ingest: {summary['ok']} of {summary['episodes']} episodes ingested, "
                 f"{summary['failed']} failed; {summary['calls']} model calls and "
                 f"{summary['api_errors']} that raised (by the outcome of their attempt: "
                 + ", ".join(f"{name} {count}" for name, count
                             in summary["api_errors_by_outcome"].items())
                 + f"), {summary['input_tokens']} input and "
                 f"{summary['output_tokens']} output tokens; summary in {SUMMARY}")
        return code


def ingested_uuids(out_dir: Path) -> set[str]:
    """The uuids of the episodes that the run's ingest log records as ingested."""
    return {line["episode_uuid"] for line in read_jsonl(out_dir / LOG) if line.get("ok")}


async def remove_episodes(uuids: list[str], out_dir: Path, *, images: dict[str, str],
                          open_engine: Callable[..., Awaitable] = adapter.open_engine,
                          preflight: Callable[[Path], int] | None = netcheck_preflight,
                          key_file: str = llm.KEY_FILE,
                          progress: Callable[[str], None] = print) -> int:
    """Removes, with Graphiti's own remove_episode, each episode of ``uuids`` that the store
    holds and the run's ingest log does not record as ingested: what an attempt wrote before it
    stopped. It works on a run's output directory, under the run's lock, after the network check
    (remove_episode rebuilds the summaries of the entities the episode mentioned with the
    construction model), records its model calls in the run's usage log under
    ``remove-episode:<episode id>``, with its invocation and no attempt, and records its
    invocation in run.json. Returns the exit status: 0 when every uuid is removed, 2 when one is
    not in the store; Refused for a uuid the log records as ingested."""
    if not out_dir.is_dir():
        raise Refused(f"{out_dir} is not a directory")
    with hold_lock(out_dir, "remove-episodes"), cancelled_on_sigterm():
        begun = time.monotonic()
        run_path = out_dir / RUN
        if not run_path.exists():
            raise Refused(f"{out_dir} holds no run ({RUN}): --remove-episodes works on a run's "
                          "output directory")
        record = json.loads(run_path.read_text(encoding="utf-8"))
        if refused := sorted(set(uuids) & ingested_uuids(out_dir)):
            raise Refused(f"the log records {', '.join(refused)} as ingested; only an episode it "
                          "does not record is removed")
        number = len(record["invocations"]) + 1
        invocation = {"command": "remove-episodes", "started_at": utc_now(),
                      "uuids": list(uuids), "removed": [], "images": dict(images),
                      "elapsed_s": None, "exit": None, "stop": None, "crash": None}
        with recorded(record, invocation, run_path, begun):
            if preflight is not None and preflight(out_dir) != 0:
                progress(f"remove-episodes: the network check failed (see {NETCHECK}); nothing "
                         "was removed")
                invocation["exit"], invocation["stop"] = EXIT_STOPPED, {"error": "network check"}
                return EXIT_STOPPED
            engine = await open_engine(usage.UsageLog(str(out_dir / USAGE)), key_file)
            status = EXIT_INGESTED
            try:
                stored = {episode["uuid"]: episode for episode in await engine.stored_episodes()}
                for uuid in uuids:
                    if uuid not in stored:
                        progress(f"remove-episodes: the store holds no episode {uuid}")
                        status = EXIT_STOPPED
                        continue
                    with usage.attributed(f"{REMOVAL_CONTEXT}{stored[uuid]['name']}", number):
                        await engine.remove_episode(uuid)
                    invocation["removed"].append(uuid)
                    progress(f"remove-episodes: removed {uuid} ({stored[uuid]['name']})")
            finally:
                await engine.close()
            invocation["exit"] = status
            return status


def crashed(error: BaseException, key_file: str, prefix: str) -> int:
    """Prints a crash, the key taken out, and returns its exit status."""
    text = "".join(traceback.format_exception(error))
    print(f"{prefix}: crashed ({type(error).__name__}); run.json records exit "
          f"{EXIT_CRASHED}\n{llm.redact(text, key_file)}", file=sys.stderr)
    return EXIT_CRASHED


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ingest", description="Ingests one arm's stream into "
                                     "Graphiti, inside the run environment.", allow_abbrev=False)
    parser.add_argument("--stream", required=True, type=Path, help="the run's input stream")
    parser.add_argument("--arm", required=True, choices=ARMS, help="the stream's arm")
    parser.add_argument("--out", required=True, type=Path, help="the run's output directory")
    parser.add_argument("--resume", action="store_true",
                        help="continue each group from its first episode with no log line")
    parser.add_argument("--skip-episode", action="append", default=[], metavar="EPISODE_ID",
                        help="with --resume: log this episode, the first of its group with no "
                             "log line, as failed (README.md); repeatable")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    # Neo4j's notices about property keys that no node has yet, for Graphiti's own queries.
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
    try:
        images = parse_image_ids(os.environ.get("ASOFBENCH_IMAGE_IDS"))
        return asyncio.run(run(args.stream, args.arm, args.out, resume=args.resume,
                               images=images, skip=args.skip_episode))
    except Refused as refusal:
        print(f"ingest: {refusal}", file=sys.stderr)
        return EXIT_STOPPED
    except SystemExit:
        raise
    except BaseException as error:  # noqa: BLE001 - a crash: recorded in run.json, exit 4
        return crashed(error, llm.KEY_FILE, "ingest")


if __name__ == "__main__":
    sys.exit(main())
