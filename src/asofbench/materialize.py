"""``asofbench materialize``: an arm's stream as the proposal files as of a date, with the log of
the commits up to it (spec section 11).

Grep at D reads its arm's stream materialised as of the end of D: the proposal files as they
stood, and the log of the commits that changed them, rebuilt from that arm's episodes and nothing
else from the repositories, so it sees no header its arm withholds, no index such as PEP 0 and
nothing recorded after D. The command replays every episode recorded at or before D 23:59:59 UTC,
in stream order: a create gives the proposal its file's full text, a modify or a rename applies
its diff to that text, which it must fit exactly (section 8), and a delete takes the file away.
It then writes each present proposal's text, byte for byte as UTF-8, and ``git-log.txt`` beside
them, and nothing else: no index, no canary, no other field of the stream.

**File names.** An episode records the repository its file came from but not the file's path
(its fields are those of section 8), so a file is named from its episode: ``<repo>/<series>-
<number>``, with the repository's series in lower case and the number in four digits for a PEP,
as the repositories name their files (``peps/pep-0008``, ``EIPs/eip-1559``, ``ERCs/erc-20``). An
ERC is ``EIPs/eip-<number>`` until the episode that moves it to the ERC repository, and
``ERCs/erc-<number>`` from then on.

**The log** (``git-log.txt``, ``git_log``). One entry per commit that gives the stream an episode
recorded by the end of D, newest first (the stream's order reversed), with no diff and no text::

    commit <the commit id>
    Date: <its recorded time, the episodes' recorded_at>
    File: <each proposal file it touched, as the tree names it, in stream order>

        <its message as the arm's stream keeps it, each line indented by four spaces>

A blank line follows the files, and another ends the message. Every line of the message, split at
newline characters only, is written after four spaces, so the message can be read back exactly.
When the episodes of one commit carry different messages, as the CF stream's do in a commit that
changes a CF status line (section 7), the entry gives each message after the files whose episodes
carry it, in stream order.

With ``--arm`` and ``--data``, the stream must be the one the build's ``MANIFEST.json`` registers
for that arm, by its SHA-256, or nothing is written.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterable, Iterator
from datetime import date
from hashlib import sha256
from pathlib import Path

from .corpora import REPOS
from .manifest import ManifestError, read_manifest, registered_stream
from .stream import KINDS, NO_NEWLINE
from .stream import _lines as lines_of  # the splitter the diffs were cut with (section 8)

LOG_FILE = "git-log.txt"  # beside the proposal files: the commits up to the date

_HUNK = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_MARKER = NO_NEWLINE.rstrip("\n")
_PID = re.compile(r"(PEP|EIP|ERC)-([0-9]+)")
_STAMP = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
_COMMIT = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")  # a git object id
_LOGGED = ("commit", "recorded_at", "repo", "proposal", "message")  # what the log reads


class MaterializeError(Exception):
    """A stream or an output directory the command cannot use; it reports it and exits 2."""


def file_name(repo: str, proposal: str) -> str:
    """The name of a proposal's file in ``repo``'s directory: ``pep-0008``, ``eip-20``."""
    match = _PID.fullmatch(proposal)
    if match is None or repo not in REPOS:
        raise MaterializeError(f"no file name for {proposal!r} in {repo!r}")
    spec, number = REPOS[repo], int(match.group(2))
    return f"{spec.series.lower()}-{number:04d}" if spec.family == "pep" else (
        f"{spec.series.lower()}-{number}")


def apply_diff(before: str, diff: str) -> str:
    """``before`` with a diff of ``stream.unified_diff`` applied; it must fit exactly.

    Lines are split at newline characters only, as when the diff was made. Every context and
    removed line must be the line of ``before`` that stands there, both sides of every hunk header
    must agree with its lines, and the marker line ``\\ No newline at end of file`` must follow,
    and only follow, a last line that has no newline. An empty diff leaves ``before`` as it is.
    """
    if not diff:
        return before
    old = lines_of(before)
    rows = diff.split("\n")
    if rows.pop() != "" or rows[:2] != ["--- a", "+++ b"]:
        raise MaterializeError("a diff that is not one of the stream's unified diffs")
    out: list[str] = []
    pos, i = 0, 2
    while i < len(rows):
        hunk = _HUNK.fullmatch(rows[i])
        if hunk is None:
            raise MaterializeError(f"a bad hunk header {rows[i]!r}")
        i += 1
        old_count = 1 if hunk[2] is None else int(hunk[2])
        new_count = 1 if hunk[4] is None else int(hunk[4])
        start = int(hunk[1]) - (1 if old_count else 0)
        if start < pos or start > len(old):
            raise MaterializeError("hunks that overlap or run past the text")
        out.extend(old[pos:start])
        pos = start
        if int(hunk[3]) - (1 if new_count else 0) != len(out):
            raise MaterializeError("a hunk whose new side does not start where it should")
        seen_old = seen_new = 0
        while seen_old < old_count or seen_new < new_count:
            if i >= len(rows):
                raise MaterializeError("a hunk that is cut short")
            row = rows[i]
            i += 1
            sign, content = row[:1], row[1:]
            if sign not in (" ", "-", "+"):
                raise MaterializeError(f"an unexpected diff line {row!r}")
            open_end = i < len(rows) and rows[i] == _MARKER
            i += open_end
            line = content if open_end else content + "\n"
            if sign != "+":
                if pos >= len(old) or old[pos] != line:
                    raise MaterializeError(f"a diff that does not fit its text at line {pos + 1}")
                pos += 1
                seen_old += 1
            if sign != "-":
                out.append(line)
                seen_new += 1
            if seen_old > old_count or seen_new > new_count:
                raise MaterializeError("a hunk that holds more lines than its header says")
    out.extend(old[pos:])
    if any(not line.endswith("\n") for line in out[:-1]):
        raise MaterializeError("a line without a newline that is not the last one")
    return "".join(out)


def _problem(row: object) -> str | None:
    if not isinstance(row, dict):
        return "not an episode"
    missing = [name for name in ("repo", "proposal", "recorded_at", "kind", "text", "commit",
                                 "message") if not isinstance(row.get(name), str)]
    if missing:
        return f"an episode with no {', '.join(missing)}"
    if _COMMIT.fullmatch(row["commit"]) is None:
        return f"an episode of the commit {row['commit']!r}"
    if row["kind"] not in KINDS:
        return f"an episode of the unknown kind {row['kind']!r}"
    if row["repo"] not in REPOS:
        return f"an episode of the unknown repository {row['repo']!r}"
    if _STAMP.fullmatch(row["recorded_at"]) is None:
        return f"an episode recorded at {row['recorded_at']!r}"
    if _PID.fullmatch(row["proposal"]) is None:
        return f"an episode of the proposal {row['proposal']!r}"
    return None


def _end_of(day: date) -> str:
    return f"{day.isoformat()}T23:59:59Z"


def replay(lines: Iterable[str], days: Iterable[date], where: str = "the stream",
           log: list[dict] | None = None) -> Iterator[tuple[date, dict[tuple[str, str], str]]]:
    """Each of ``days``, in date order, with the files as of its end: ``(repo, proposal) -> text``.

    ``lines`` are the stream's JSONL lines. Every line is checked, and must be in recorded-time
    order; episodes after the last day are not applied. Given ``log``, a list, each episode
    applied is appended to it as its ``commit``, ``recorded_at``, ``repo``, ``proposal`` and
    ``message``, in stream order (``git_log``).
    """
    days = sorted(days)
    files: dict[str, tuple[str, str]] = {}  # proposal -> (repo, text)
    k, previous = 0, ""
    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except ValueError:
            raise MaterializeError(f"{where} line {number}: not JSON") from None
        problem = _problem(row)
        if problem is not None:
            raise MaterializeError(f"{where} line {number}: {problem}")
        if row["recorded_at"] < previous:
            raise MaterializeError(f"{where} line {number}: not in recorded-time order")
        previous = row["recorded_at"]
        while k < len(days) and row["recorded_at"] > _end_of(days[k]):
            yield days[k], {(repo, pid): text for pid, (repo, text) in files.items()}
            k += 1
        if k == len(days):
            continue
        if log is not None:
            log.append({name: row[name] for name in _LOGGED})
        pid, repo, kind, text = row["proposal"], row["repo"], row["kind"], row["text"]
        if kind == "create":
            if pid in files:
                raise MaterializeError(f"{where} line {number}: {pid} is created again")
            files[pid] = (repo, text)
            continue
        if pid not in files:
            raise MaterializeError(f"{where} line {number}: a {kind} of {pid}, which has no file")
        if kind == "delete":
            del files[pid]
            continue
        before_repo, before = files[pid]
        if kind == "modify" and before_repo != repo:
            raise MaterializeError(f"{where} line {number}: a modify of {pid} that moves it")
        try:
            files[pid] = (repo, apply_diff(before, text))
        except MaterializeError as e:
            raise MaterializeError(f"{where} line {number}: {e}") from None
    for day in days[k:]:
        yield day, {(repo, pid): text for pid, (repo, text) in files.items()}


def write_tree(files: dict[tuple[str, str], str], out: Path) -> int:
    """Write each file under ``out/<repo>/<file name>``; return how many were written."""
    out = Path(out)
    paths: dict[Path, str] = {}
    for repo, pid in files:
        path = out / repo / file_name(repo, pid)
        if path in paths:
            raise MaterializeError(f"{paths[path]} and {pid} would both be written to {path}")
        paths[path] = pid
    out.mkdir(parents=True, exist_ok=True)
    for (repo, pid), text in sorted(files.items()):
        path = out / repo / file_name(repo, pid)
        path.parent.mkdir(exist_ok=True)
        try:
            data = text.encode("utf-8")
        except UnicodeEncodeError:
            raise MaterializeError(f"the text of {pid} is not UTF-8") from None
        path.write_bytes(data)
    return len(files)


def git_log(episodes: Iterable[dict]) -> str:
    """The text of ``git-log.txt`` for ``episodes``, the stream's episodes up to the date in
    stream order (``replay``'s log): one entry per commit, newest first, in the form the module
    docstring gives."""
    commits: dict[str, tuple[str, dict[str, list[str]]]] = {}
    for e in episodes:
        when, messages = commits.setdefault(e["commit"], (e["recorded_at"], {}))
        messages.setdefault(e["message"], []).append(
            f"{e['repo']}/{file_name(e['repo'], e['proposal'])}")
    lines: list[str] = []
    for commit, (when, messages) in reversed(commits.items()):
        lines += [f"commit {commit}", f"Date: {when}"]
        for message, paths in messages.items():
            lines += [*(f"File: {path}" for path in paths), "",
                      *(f"    {line}" for line in message.split("\n")), ""]
    return "".join(line + "\n" for line in lines)


def write_log(episodes: list[dict], out: Path) -> int:
    """Write ``git-log.txt`` into ``out``; return how many commits it lists."""
    try:
        data = git_log(episodes).encode("utf-8")
    except UnicodeEncodeError:
        raise MaterializeError("a commit message that is not UTF-8") from None
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / LOG_FILE).write_bytes(data)
    return len({e["commit"] for e in episodes})


def _check_out(out: Path) -> None:
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise MaterializeError(f"{out} already holds files: materialise into a new directory")


def _check_registered(stream: Path, arm: str, data: Path) -> None:
    """The stream must be the one ``data/MANIFEST.json`` registers for ``arm`` (section 8)."""
    manifest = data / "MANIFEST.json"
    entry = registered_stream(read_manifest(manifest), arm, manifest)
    if entry is None:
        raise MaterializeError(f"{manifest} registers no {arm} stream")
    digest = sha256()
    with stream.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != entry["sha256"]:
        raise MaterializeError(f"{stream} is not the {arm} stream registered in {manifest}: its "
                               f"SHA-256 is {digest.hexdigest()}, not {entry['sha256']}")


def main(args) -> int:
    """``asofbench materialize --stream FILE --as-of YYYY-MM-DD --out DIR [--arm A --data DIR]``."""
    stream, out, day = Path(args.stream), Path(args.out), args.as_of
    try:
        if not stream.is_file():
            raise MaterializeError(f"{stream} not found: write it with `asofbench stream`")
        _check_out(out)
        if getattr(args, "arm", None) is not None:
            _check_registered(stream, args.arm, Path(args.data))
        applied: list[dict] = []
        with stream.open(encoding="utf-8", newline="\n") as f:
            ((_, files),) = replay(f, [day], where=str(stream), log=applied)
        written = write_tree(files, out)
        logged = write_log(applied, out)
    except UnicodeDecodeError:
        print(f"asofbench materialize: {stream} is not UTF-8 text", file=sys.stderr)
        return 2
    except (MaterializeError, ManifestError) as e:
        print(f"asofbench materialize: {e}", file=sys.stderr)
        return 2
    print(f"asofbench materialize: {written} files as of the end of {day.isoformat()} (UTC), and "
          f"{LOG_FILE} with the {logged} commit{'' if logged == 1 else 's'} up to it, written to "
          f"{out}")
    return 0
