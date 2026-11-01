"""Read-only walk over the first-parent chain of a git repository (spec section 3).

A repository's history is its first-parent chain from the root to a tip, each merge commit
read as its diff against its first parent. A commit's recorded time is the larger of its
committer timestamp and its first parent's recorded time, so it never goes backwards along
the chain; "as of date D" means the last commit recorded at or before D 23:59:59 UTC.

Only the git commands in ``_READ_ONLY`` are ever run, always as ``git -C <repo>`` with no
system or user configuration. Blob text read from a repository is data and is never executed.
"""

from __future__ import annotations

import os
import re
import subprocess
from bisect import bisect_right
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path

_READ_ONLY = frozenset({"log", "rev-list", "cat-file", "show", "rev-parse"})

# Variables that would point git at a repository, index or configuration other than the
# clone being read (a git hook or a review script exports some of them).
_REDIRECTING = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_NAMESPACE",
    "GIT_CONFIG_PARAMETERS",
    "GIT_CONFIG_COUNT",
)

_EPOCH = date(1970, 1, 1)
_RECORD = re.compile(r"\x1e([0-9a-f]{40}|[0-9a-f]{64})\x1f")
_OBJECT_ID = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
# git's raw status letter -> FileChange.kind. A typechange (a symlink replaced by a regular file,
# as when PEP 495 was first added) keeps its path and gets new content, so it is a modification.
_KINDS = {"A": "A", "M": "M", "R": "R", "D": "D", "T": "M"}


@dataclass(frozen=True)
class Commit:
    sha: str
    committer_ts: int
    recorded_ts: int
    message: str


@dataclass(frozen=True)
class FileChange:
    commit: Commit
    kind: str  # "A" | "M" | "R" | "D"
    path: str
    old_path: str | None
    new_blob: str | None
    old_blob: str | None
    # git's six-digit modes: 100644 file, 100755 executable file, 120000 symlink, 160000
    # submodule. None on the side that does not exist (old for an add, new for a delete).
    old_mode: str | None = None
    new_mode: str | None = None


def _env() -> dict[str, str]:
    # GIT_TRACE* is dropped too: the cat-file process's stderr is only read when it fails, so
    # a flood of trace lines could fill the pipe and stall it.
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in _REDIRECTING and not k.startswith("GIT_TRACE")
    }
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_CONFIG_GLOBAL"] = "/dev/null"
    # In a partial clone, git would fetch a missing object from the network (and write it to the
    # clone) on demand; check_clone refuses such clones, and this makes a missed one fail instead.
    env["GIT_NO_LAZY_FETCH"] = "1"
    return env


def _command(repo: Path, *args: str) -> list[str]:
    if args[0] not in _READ_ONLY:
        raise ValueError(f"git {args[0]} is not a read-only command")
    return ["git", "-C", str(repo), *args]


def _git(repo: Path, *args: str) -> bytes:
    cmd = _command(repo, *args)
    done = subprocess.run(cmd, env=_env(), capture_output=True, check=False)
    if done.returncode != 0:
        detail = done.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed in {repo}: {detail}")
    return done.stdout


class CloneError(Exception):
    """A clone the generator refuses to read (spec section 2); the command exits 2 with it."""


_PROMISOR_CONFIG = re.compile(
    r"^\s*(?:promisor\s*=\s*true|partialclonefilter\s*=|partialclone\s*=)",
    re.IGNORECASE | re.MULTILINE,
)


def _is_partial(common: Path) -> bool:
    """Whether the repository whose common git directory is ``common`` is a partial clone
    (``git clone --filter``): it holds packs from a promisor remote, or its configuration names
    one. Read from the files, since ``git config`` is not a command the walk may run."""
    if any((common / "objects" / "pack").glob("*.promisor")):
        return True
    try:
        config = (common / "config").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return _PROMISOR_CONFIG.search(config) is not None


def check_clone(repo: Path) -> None:
    """Refuse a clone that is not there, is not the top level of its own work tree, or is shallow
    or partial.

    A directory inside some other repository is refused because git would read that repository's
    history instead; a shallow clone because its history is cut short, so the timelines it gives
    are wrong without any error; a partial clone because it lacks file contents, which git would
    fetch from the network while the walk reads them (section 2: a full, non-shallow clone).
    """
    path = Path(repo)
    if not path.is_dir():
        raise CloneError(f"{path} is not a directory: a full clone of the repository belongs there")
    try:
        top = _git(path, "rev-parse", "--show-toplevel").decode("utf-8", errors="replace").strip()
        same = os.path.samefile(top, path)
    except (RuntimeError, OSError):
        raise CloneError(f"{path} is not a git work tree: a full, non-shallow clone belongs there "
                         "(a bare repository is not one)") from None
    if not same:
        raise CloneError(f"{path} is not the top level of a clone: git would read the repository "
                         f"at {top} instead")
    shallow = _git(path, "rev-parse", "--is-shallow-repository").decode("ascii", "replace").strip()
    if shallow != "false":
        raise CloneError(f"{path} is a shallow clone: its history is cut short, so the timelines "
                         "would be wrong; fetch the whole history (git fetch --unshallow)")
    common = _git(path, "rev-parse", "--path-format=absolute", "--git-common-dir")
    if _is_partial(Path(common.decode("utf-8", errors="replace").strip())):
        raise CloneError(f"{path} is a partial clone: it lacks file contents, which git would "
                         "fetch from the network during the walk; clone it without --filter")


def end_of_day_ts(d: date) -> int:
    """The epoch second of ``d`` 23:59:59 UTC."""
    return (d - _EPOCH).days * 86400 + 86399


def first_parent_commits(repo: Path, tip: str) -> list[Commit]:
    """The first-parent chain from the root to ``tip``, with recorded times; messages are empty."""
    out = _git(
        repo,
        "log",
        "--first-parent",
        "--reverse",
        "--format=%H%x09%ct",
        "--end-of-options",
        tip,
        "--",
    )
    commits: list[Commit] = []
    recorded: int | None = None
    for line in out.decode("ascii").splitlines():
        sha, _, stamp = line.partition("\t")
        committer_ts = int(stamp)
        recorded = committer_ts if recorded is None else max(recorded, committer_ts)
        commits.append(Commit(sha, committer_ts, recorded, ""))
    return commits


def author_names(repo: Path, tip: str) -> dict[str, str]:
    """The author name of each commit on the first-parent chain to ``tip``, by sha.

    The name is as recorded in the commit (``%an``, so no mailmap), decoded as UTF-8 with
    undecodable bytes replaced.
    """
    out = _git(repo, "log", "--first-parent", "--format=%H%x1f%an", "--end-of-options", tip, "--")
    names: dict[str, str] = {}
    # Split at newlines only: git ends each record with one, and a name may hold a character
    # (a line or paragraph separator, a form feed) at which str.splitlines would cut it.
    for line in out.decode("utf-8", errors="replace").split("\n"):
        if line:
            sha, _, name = line.partition("\x1f")
            names[sha] = name
    return names


def _records(stream: str) -> Iterator[tuple[str, str, str]]:
    """Split the log stream into (sha, message, raw diff entries), newest commit first."""
    starts = list(_RECORD.finditer(stream))
    for i, start in enumerate(starts):
        stop = starts[i + 1].start() if i + 1 < len(starts) else len(stream)
        message, sep, raw = stream[start.end() : stop].partition("\x1f\x00")
        if not sep:
            raise ValueError(f"malformed git log record for {start.group(1)}")
        yield start.group(1), message, raw


def _raw_entries(raw: str) -> Iterator[tuple[str, str, str, str, str, list[str]]]:
    """Parse ``--raw -z`` output into (status, old mode, new mode, old blob, new blob, paths).

    Each entry is ``:<old mode> <new mode> <old blob> <new blob> <status>`` and then one NUL
    separated path, or two for a rename; the status says which, so a path that starts with a
    colon is never mistaken for the next entry.
    """
    tokens = raw.lstrip("\n").split("\0")
    if tokens[-1] == "":
        tokens.pop()
    i = 0
    while i < len(tokens):
        meta = tokens[i]
        if not meta.startswith(":"):
            raise ValueError(f"unexpected token in git raw diff output: {meta!r}")
        old_mode, new_mode, old_blob, new_blob, status = meta[1:].split(" ")
        count = 2 if status[0] in "RC" else 1
        paths = tokens[i + 1 : i + 1 + count]
        if len(paths) != count:
            raise ValueError(f"truncated git raw diff entry: {meta!r}")
        yield status[0], old_mode, new_mode, old_blob, new_blob, paths
        i += 1 + count


def _none_if_zero(value: str) -> str | None:
    """git prints an all-zero blob id and mode for the side of a change that does not exist."""
    return None if set(value) == {"0"} else value


def file_changes(
    repo: Path, tip: str, path_re: re.Pattern[str], commits: list[Commit]
) -> list[FileChange]:
    """Every first-parent change to a path matching ``path_re``, in chain order.

    A rename is reported when either side of it matches. Renames are paired at 50% similarity
    with no limit on how many files are compared (``-M50% -l0``), so the pairing never depends
    on git's defaults or on a clone's own ``diff.renames`` or ``diff.renameLimit``.

    ``commits`` is the walk from ``first_parent_commits(repo, tip)`` (or a longer one); it
    supplies each change's recorded time and fixes the order, and the full commit message is
    read here. Changes of one commit keep the order git lists them in (by path).
    """
    position = {c.sha: i for i, c in enumerate(commits)}
    out = _git(
        repo,
        "log",
        "--first-parent",
        "--diff-merges=first-parent",
        "-M50%",
        "-l0",
        "--raw",
        "--no-abbrev",
        "-z",
        "--root",
        "--format=%x1e%H%x1f%B%x1f",
        "--end-of-options",
        tip,
        "--",
    )
    found: list[tuple[int, FileChange]] = []
    for sha, message, raw in _records(out.decode("utf-8", errors="replace")):
        commit: Commit | None = None
        for status, old_mode, new_mode, old_blob, new_blob, paths in _raw_entries(raw):
            if not any(path_re.match(p) for p in paths):
                continue
            if status not in _KINDS:
                raise ValueError(f"unsupported change {status!r} of {paths[-1]!r} in {sha}")
            if commit is None:
                if sha not in position:
                    raise ValueError(f"commit {sha} is not in the commit list for {tip}")
                base = commits[position[sha]]
                commit = Commit(sha, base.committer_ts, base.recorded_ts, message.rstrip("\n"))
            found.append(
                (
                    position[sha],
                    FileChange(
                        commit=commit,
                        kind=_KINDS[status],
                        path=paths[-1],
                        old_path=paths[0] if status == "R" else None,
                        new_blob=_none_if_zero(new_blob),
                        old_blob=_none_if_zero(old_blob),
                        old_mode=_none_if_zero(old_mode),
                        new_mode=_none_if_zero(new_mode),
                    ),
                )
            )
    found.sort(key=lambda pair: pair[0])  # stable: git's path order is kept within a commit
    return [change for _, change in found]


class BlobReader:
    """Reads blobs through one persistent ``git cat-file --batch`` process.

    Use it as a context manager; the process lives until the block ends.
    """

    def __init__(self, repo: Path) -> None:
        self.repo = Path(repo)
        self._proc: subprocess.Popen[bytes] | None = None

    def __enter__(self) -> BlobReader:
        self._proc = subprocess.Popen(
            _command(self.repo, "cat-file", "--batch"),
            env=_env(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.stdin.close()  # end of input makes cat-file exit
        except BrokenPipeError:
            pass
        proc.stdout.close()
        proc.wait()
        proc.stderr.close()

    def _stopped(self, proc: subprocess.Popen[bytes]) -> RuntimeError:
        detail = proc.stderr.read().decode("utf-8", errors="replace").strip()
        return RuntimeError(f"git cat-file stopped in {self.repo}: {detail}")

    def read(self, blob: str) -> str:
        """The blob as text: UTF-8 with undecodable bytes replaced; a BOM and CRs are kept."""
        proc = self._proc
        if proc is None:
            raise RuntimeError("BlobReader must be used in a with block")
        if not _OBJECT_ID.fullmatch(blob):
            raise ValueError(f"not a full object id: {blob!r}")
        try:
            proc.stdin.write(blob.encode("ascii") + b"\n")
            proc.stdin.flush()
        except BrokenPipeError:
            raise self._stopped(proc) from None
        header = proc.stdout.readline().split()
        if not header:
            raise self._stopped(proc)
        if header[1:] == [b"missing"]:
            raise LookupError(f"object {blob} is not in {self.repo}")
        if len(header) != 3:
            raise RuntimeError(f"unexpected git cat-file reply for {blob}: {header!r}")
        size = int(header[2])
        data = proc.stdout.read(size + 1)  # the content and the newline that ends the reply
        if len(data) != size + 1:
            raise RuntimeError(f"git cat-file ended in the middle of {blob}")
        if header[1] != b"blob":
            raise ValueError(f"{blob} is a {header[1].decode()}, not a blob")
        return data[:size].decode("utf-8", errors="replace")


def as_of_commit(commits: list[Commit], d: date) -> Commit | None:
    """The last commit recorded at or before the end of ``d`` (UTC), or None if there is none."""
    index = bisect_right(commits, end_of_day_ts(d), key=lambda c: c.recorded_ts)
    return commits[index - 1] if index else None


def head_commit(repo: Path) -> str | None:
    """The commit HEAD names in ``repo``, or None when there is none (no commits, no repository)."""
    try:
        out = _git(Path(repo), "rev-parse", "--verify", "--quiet", "HEAD^{commit}")
    except RuntimeError:
        return None
    sha = out.decode("ascii", errors="replace").strip()
    return sha if _OBJECT_ID.fullmatch(sha) else None


def blob_at(repo: Path, commit: str, path: str) -> str | None:
    """The object ``path`` names at ``commit`` (``git rev-parse <commit>:<path>``), or None when
    there is no such commit or path. ``commit`` must be a full object id."""
    if not _OBJECT_ID.fullmatch(commit) or not path or "\n" in path:
        return None
    try:
        out = _git(Path(repo), "rev-parse", "--verify", "--quiet", f"{commit}:{path}")
    except RuntimeError:
        return None
    oid = out.decode("ascii", errors="replace").strip()
    return oid if _OBJECT_ID.fullmatch(oid) else None


def is_top_level(directory: Path) -> bool:
    """Whether ``directory`` is the top level of its own git work tree (not a directory inside
    one, not a bare repository, not outside any repository)."""
    try:
        top = _git(Path(directory), "rev-parse", "--show-toplevel").decode("utf-8", "replace")
        return os.path.samefile(top.strip(), directory)
    except (RuntimeError, OSError):
        return False


def files_at_head(repo: Path, directory: str) -> dict[str, bytes]:
    """``files_at`` at HEAD: the regular files directly in ``directory`` there."""
    return files_at(repo, directory)


def files_at(repo: Path, directory: str, revision: str = "HEAD") -> dict[str, bytes]:
    """The regular files directly in ``directory`` at ``revision`` (HEAD, or a full object id),
    by name, as their bytes: blobs of mode 100644 or 100755, so never a symbolic link or a
    submodule; empty when the revision has no such directory. ``cat-file`` lists the tree and
    reads each blob."""
    if revision != "HEAD" and not _OBJECT_ID.fullmatch(revision):
        raise ValueError(f"{revision!r} is neither HEAD nor a full object id")
    try:
        listing = _git(Path(repo), "cat-file", "-p", f"{revision}:{directory}")
    except RuntimeError:
        return {}
    files = {}
    for line in listing.decode("utf-8", errors="replace").split("\n"):
        meta, _, name = line.partition("\t")
        fields = meta.split()
        if len(fields) == 3 and fields[0] in ("100644", "100755") and fields[1] == "blob":
            files[name] = _git(Path(repo), "cat-file", "blob", fields[2])
    return files
