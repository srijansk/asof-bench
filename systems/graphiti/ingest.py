"""Graphiti's ingestion of one arm's stream, inside the run environment (systems/README.md).

    ingest --stream /input/<file>.jsonl --arm H+|H-|CF --out /output [--resume]

It first waits for the environment's services, then runs the network check (env/netcheck.py,
with the README's arguments, ``--services embedder:8080,neo4j:7687`` and ``--mounts`` its input,
output and secret paths), whose report it writes to ``/output/netcheck.json``, and ingests
nothing unless every check passes. Then it builds Graphiti as adapter.py configures it, checks
that the store holds exactly the episodes that the ingest log records as ingested (none, for a
new run), and ingests the stream's episodes: one group per corpus (Graphiti's ``group_id``), the
episodes of a group one at a time in stream order, each awaited before the next, as Graphiti
requires, and the two groups concurrently. While an episode is ingested, every model call is
recorded in the usage log under its id. An episode that still raises after Graphiti's own retries
is logged as failed and the run goes on. With ``--resume`` it attempts only the episodes that the
log does not record as ingested, in stream order, after rebuilding each proposal's state from the
log; without it, it refuses an output directory that already holds a run.

It writes, in ``--out``:

- ``usage.jsonl``: one line per model response (asofbench_systems.usage);
- ``ingest-log.jsonl``: one line per episode attempt, with ``episode_id``, ``proposal``,
  ``group_id``, ``episode_uuid`` (null for a failed attempt), ``ok``, ``error`` (null, or the
  exception's class and the first 300 characters of its message, the key taken out), ``started_at``,
  ``wall_s`` and ``text_chars`` (the episode's ``text`` in code points, as the manifest counts);
- ``summary.json``: the totals over the whole log and every usage line, each counted once, as
  ``summarize`` describes.

A proposal's state is its saga predecessor, the uuid of its last ingested episode, and its window,
the uuids of its last EPISODE_WINDOW_LEN ingested episodes, oldest first: the episodes the log
records as ingested, in the log's order, which within a proposal is the order of ingestion. An
episode retried after later ones of its proposal were ingested follows them.

Exit status: 0 when every episode of the stream is ingested, 1 when one is not or the network
check fails, 2 when the run is refused before it starts (a bad argument, an output directory that
holds a run without ``--resume``, a log or a store that does not match the stream).

The state, log and summary functions use only the standard library; graphiti-core and anthropic
are imported only by adapter.open_engine.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import socket
import subprocess
import sys
import time
from collections import deque
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from asofbench_systems import llm, usage
from asofbench_systems.stream import read_episodes
from graphiti import adapter

ARMS = ("H+", "H-", "CF")
LOG, USAGE, SUMMARY, NETCHECK = "ingest-log.jsonl", "usage.jsonl", "summary.json", "netcheck.json"
LOG_FIELDS = ("episode_id", "proposal", "group_id", "episode_uuid", "ok", "error", "started_at",
              "wall_s", "text_chars")
# The fields of a stream's episode (section 8).
EPISODE_FIELDS = ("episode_id", "corpus", "repo", "proposal", "ref", "recorded_at", "commit",
                  "message", "kind", "text")
# How much of an exception's message the log keeps.
ERROR_CHARS = 300
# The prompt length up to which the price basis's prices hold (section 13).
PRICED_INPUT_TOKENS = 100_000
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
# How long the run waits for the services to accept connections (the embedder loads its model
# first), and how often it tries.
SERVICE_WAIT_S, SERVICE_RETRY_S = 600.0, 2.0


class Refused(Exception):
    """The run cannot start: nothing has been ingested."""


@dataclass
class ProposalState:
    """A proposal's saga predecessor and its window of previous episodes."""

    window: deque = field(default_factory=lambda: deque(maxlen=adapter.EPISODE_WINDOW_LEN))
    saga_previous: str | None = None

    def ingested(self, uuid: str) -> None:
        self.window.append(uuid)
        self.saga_previous = uuid


def read_jsonl(path: Path) -> list[dict]:
    """The JSON lines of ``path``, none when it does not exist; Refused for a line that is not
    JSON."""
    if not path.exists():
        return []
    lines = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                lines.append(json.loads(line))
            except ValueError:
                raise Refused(f"{path}: line {number} is not JSON") from None
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
            raise Refused(f"{stream}: episode {number} has no {', '.join(missing)}")
        if e["episode_id"] in ids:
            raise Refused(f"{stream}: episode {number}'s id, {e['episode_id']}, occurs twice")
        ids.add(e["episode_id"])
    return episodes


def check_log(episodes: list[dict], log: list[dict]) -> None:
    """Refused unless every line of ``log`` is an attempt at an episode of ``episodes``, with that
    episode's proposal and group, and each episode is recorded as ingested at most once."""
    by_id = {e["episode_id"]: e for e in episodes}
    problems, ingested = [], set()
    for number, line in enumerate(log, 1):
        if not isinstance(line, dict) or set(line) != set(LOG_FIELDS):
            problems.append(f"line {number} is not an attempt line")
            continue
        e = by_id.get(line["episode_id"])
        if e is None:
            problems.append(f"line {number}: {line['episode_id']} is not in the stream")
        elif (line["proposal"], line["group_id"]) != (e["proposal"], e["corpus"]):
            problems.append(f"line {number}: {line['episode_id']} has another proposal or group")
        elif line["ok"] is True:
            if not isinstance(line["episode_uuid"], str) or not line["episode_uuid"]:
                problems.append(f"line {number}: an ingested episode with no uuid")
            elif line["episode_id"] in ingested:
                problems.append(f"line {number}: {line['episode_id']} is ingested twice")
            ingested.add(line["episode_id"])
        elif line["ok"] is not False or line["episode_uuid"] is not None:
            problems.append(f"line {number}: a failed attempt with a uuid, or no ok")
    if problems:
        shown = "; ".join(problems[:5]) + (f"; and {len(problems) - 5} more"
                                           if len(problems) > 5 else "")
        raise Refused(f"the ingest log does not match the stream: {shown}")


def pending(episodes: list[dict], log: list[dict]) -> list[dict]:
    """The episodes that the log does not record as ingested, in stream order."""
    done = {line["episode_id"] for line in log if line["ok"]}
    return [e for e in episodes if e["episode_id"] not in done]


def rebuild_states(log: list[dict]) -> dict[str, ProposalState]:
    """Each proposal's state after the episodes that ``log`` records as ingested, in its order."""
    states: dict[str, ProposalState] = {}
    for line in log:
        if line["ok"]:
            states.setdefault(line["proposal"], ProposalState()).ingested(line["episode_uuid"])
    return states


def by_group(episodes: Iterable[dict]) -> dict[str, list[dict]]:
    """The episodes of each corpus, Graphiti's group, in stream order."""
    groups: dict[str, list[dict]] = {}
    for e in episodes:
        groups.setdefault(e["corpus"], []).append(e)
    return groups


def input_tokens(line: dict) -> int:
    """A response's input tokens, those written to and read from the prompt cache included."""
    return (line["input_tokens"] + line["cache_creation_input_tokens"]
            + line["cache_read_input_tokens"])


def summarize(episodes: list[dict], log: list[dict], usage_lines: list[dict], *, arm: str,
              graphiti_core: str, model: str = adapter.MODEL) -> dict:
    """summary.json: the stream's ``episodes`` and ``text_chars`` (code points of ``text``), the
    episodes ``ok`` (ingested) and ``failed`` (attempted and never ingested), the usage lines'
    ``input_tokens`` (cache tokens included), ``output_tokens``, ``calls`` and
    ``calls_over_100k_input`` (prompts beyond the price basis's 100,000 tokens), and ``wall_s``,
    the sum of every attempt's wall time; each line counted once."""
    ids = {e["episode_id"] for e in episodes}
    done = {line["episode_id"] for line in log if line["ok"]} & ids
    tried = {line["episode_id"] for line in log} & ids
    return {"arm": arm, "episodes": len(episodes), "ok": len(done), "failed": len(tried - done),
            "text_chars": sum(len(e["text"]) for e in episodes),
            "input_tokens": sum(input_tokens(line) for line in usage_lines),
            "output_tokens": sum(line["output_tokens"] for line in usage_lines),
            "calls": len(usage_lines),
            "calls_over_100k_input": sum(input_tokens(line) > PRICED_INPUT_TOKENS
                                         for line in usage_lines),
            "wall_s": round(sum(line["wall_s"] for line in log), 3),
            "graphiti_core": graphiti_core, "model": model}


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


async def check_store(engine, log: list[dict]) -> None:
    """Refused unless the store holds exactly the episodes that ``log`` records as ingested."""
    stored = await engine.stored_episodes()
    logged = {line["episode_uuid"] for line in log if line["ok"]}
    held = {episode["uuid"] for episode in stored}
    if missing := logged - held:
        raise Refused(f"the store lacks {len(missing)} of the {len(logged)} episodes that the log "
                      "records as ingested: resume on the store the run began with")
    if extra := [episode for episode in stored if episode["uuid"] not in logged]:
        shown = ", ".join(f"{episode['uuid']} ({episode['name']})" for episode in extra[:10])
        what = ("a new run needs an empty store (a new volume)" if not logged else
                "orphans of an attempt that wrote to the store before it failed or was stopped; "
                "remove them (Graphiti's remove_episode) or start a new run")
        raise Refused(f"the store holds {len(extra)} episodes that the log does not record as "
                      f"ingested, {what}: {shown}")


async def ingest_group(engine, episodes: list[dict], states: dict[str, ProposalState],
                       log_path: Path, key_file: str, progress: Callable[[str], None]) -> None:
    """Ingests ``episodes``, one group's, one at a time in their order."""
    for number, e in enumerate(episodes, 1):
        state = states.setdefault(e["proposal"], ProposalState())
        started_at, start = utc_now(), time.monotonic()
        token = usage.usage_context.set(e["episode_id"])
        try:
            arguments = adapter.episode_arguments(
                e, previous_episode_uuids=list(state.window),
                saga_previous_episode_uuid=state.saga_previous, source=engine.source)
            uuid, error = await engine.add_episode(**arguments), None
        except Exception as raised:  # noqa: BLE001 - every failure is logged, and the run goes on
            uuid, error = None, error_text(raised, key_file)
        finally:
            usage.usage_context.reset(token)
        wall_s = round(time.monotonic() - start, 3)
        append_line(log_path, {"episode_id": e["episode_id"], "proposal": e["proposal"],
                               "group_id": e["corpus"], "episode_uuid": uuid,
                               "ok": uuid is not None, "error": error, "started_at": started_at,
                               "wall_s": wall_s, "text_chars": len(e["text"])})
        if uuid is not None:
            state.ingested(uuid)
        outcome = "ok" if uuid is not None else f"FAILED ({error.split(':', 1)[0]})"
        progress(f"{e['corpus']} {number}/{len(episodes)} {e['episode_id']}: {outcome} in "
                 f"{wall_s:.1f} s")


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
    ``out_dir/netcheck.json``; an earlier run's report is kept as ``netcheck-<n>.json``. Returns
    the check's exit status."""
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


async def run(stream: Path, arm: str, out_dir: Path, *, resume: bool,
              open_engine: Callable[..., Awaitable] = adapter.open_engine,
              preflight: Callable[[Path], int] | None = netcheck_preflight,
              key_file: str = llm.KEY_FILE, progress: Callable[[str], None] = print) -> int:
    """One ingestion run (the module's description); Refused when it cannot start."""
    if arm not in ARMS:
        raise Refused(f"--arm {arm!r} is not one of {', '.join(ARMS)}")
    if not out_dir.is_dir():
        raise Refused(f"{out_dir} is not a directory")
    log_path, usage_path = out_dir / LOG, out_dir / USAGE
    if not resume and any((out_dir / name).exists() for name in (LOG, USAGE, SUMMARY)):
        raise Refused(f"{out_dir} already holds a run: resume it with --resume, or give a new "
                      "output directory and a new store")
    episodes = read_stream(stream)
    if resume:
        for path in (log_path, usage_path):
            if dropped := drop_unfinished_line(path):
                progress(f"ingest: dropped the unfinished last line of {path.name} ({dropped} "
                         "bytes), left by an interrupted run")
    log = read_jsonl(log_path)
    check_log(episodes, log)
    todo = pending(episodes, log)
    if preflight is not None and preflight(out_dir) != 0:
        progress(f"ingest: the network check failed (see {out_dir / NETCHECK}); nothing was "
                 "ingested")
        return 1
    engine = await open_engine(usage.UsageLog(str(usage_path)), key_file)
    try:
        await check_store(engine, log)
        states = rebuild_states(log)
        groups = by_group(todo)
        progress(f"ingest: {len(todo)} of the stream's {len(episodes)} episodes to ingest, in "
                 f"{len(groups)} groups ({', '.join(f'{g} {len(e)}' for g, e in groups.items())})")
        # A task per group; should one fail outside an episode, the other is cancelled.
        async with asyncio.TaskGroup() as tasks:
            for group in groups.values():
                tasks.create_task(ingest_group(engine, group, states, log_path, key_file,
                                               progress))
    finally:
        await engine.close()
    summary = summarize(episodes, read_jsonl(log_path), read_jsonl(usage_path), arm=arm,
                        graphiti_core=engine.version)
    (out_dir / SUMMARY).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    progress(f"ingest: {summary['ok']} of {summary['episodes']} episodes ingested, "
             f"{summary['failed']} failed; {summary['calls']} model calls, "
             f"{summary['input_tokens']} input and {summary['output_tokens']} output tokens; "
             f"summary in {out_dir / SUMMARY}")
    return 0 if summary["ok"] == summary["episodes"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ingest", description="Ingests one arm's stream into "
                                     "Graphiti, inside the run environment.")
    parser.add_argument("--stream", required=True, type=Path, help="the run's input stream")
    parser.add_argument("--arm", required=True, choices=ARMS, help="the stream's arm")
    parser.add_argument("--out", required=True, type=Path, help="the run's output directory")
    parser.add_argument("--resume", action="store_true",
                        help="attempt only the episodes the log does not record as ingested")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    # Neo4j's notices about property keys that no node has yet, for Graphiti's own queries.
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
    try:
        return asyncio.run(run(args.stream, args.arm, args.out, resume=args.resume))
    except Refused as refusal:
        print(f"ingest: {refusal}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
