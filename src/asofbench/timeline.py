"""Proposal timelines, ERC stitching and the data-quality report (spec sections 2 to 6).

Each repository's first-parent chain is walked once, from the root to its pin, the last commit
recorded at or before the end of S (section 3). Every commit that adds, modifies, renames or
deletes a proposal's file gives that proposal a ``Version``, read from the file as it stands at
the commit (section 4). Runs of versions with one normalised status make an ``Interval``. An ERC's
timeline is the EIP repository's history of its number before the ERC file existed, then the ERC
repository's (section 6).

How the awkward shapes of the history are read (sections 2 to 4):

* A path is a proposal file if it matches its repository's pattern or is one of its alias paths
  (``corpora.ALIASES``). A rename whose two paths are different proposals is a deletion of one
  and an addition of the other; a rename into a proposal path is an addition, out of one a
  deletion.
* Within one commit, a proposal's removed files are taken away before its added and modified
  files are put in, so a deletion never outweighs an addition or a modification.
* After a commit, the proposal's file is the one changed in that commit, and of several the one
  in the newer layout (``corpora.NEWER_LAYOUT``). If none was changed (one of two files went), it
  keeps its file while that is still there, else takes the newest layout of those left. Each
  commit that adds, modifies, renames or deletes a file of a proposal and leaves two or more of its
  files live is recorded as a collision.
* A symlink is not a proposal text: its version counts as absent. A mode-only change still gives
  a version, whose header is the same.
* Of several versions recorded at the same second, the last in chain order is the one in effect
  at that time, so the others hold for no time and get no interval.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from itertools import groupby, pairwise
from pathlib import Path

from .corpora import (
    ALIASES,
    APRIL_FOOL,
    ERC_MOVE_START,
    EXCLUDED,
    LIKE_A_PROPOSAL,
    NEWER_LAYOUT,
    NOT_PROPOSAL_PLACES,
    REPOS,
    STAGNANT_BOT,
    S,
)
from .gitwalk import (
    BlobReader,
    CloneError,
    Commit,
    FileChange,
    as_of_commit,
    author_names,
    check_clone,
    end_of_day_ts,
    file_changes,
    first_parent_commits,
)
from .headers import header_span, number_list, parse_header, superseders
from .statuses import CANONICAL, MISSING, MOVED, UNKNOWN, normalize_status

ABSENT = "absent"  # the status of a version after a deletion, or of a symlink
STAGNANT = "Stagnant"
NOT_A_STATUS = frozenset({MISSING, MOVED})  # recorded, but not statuses (sections 5 and 6)
_SYMLINK = "120000"
_PEPS, _EIPS, _ERCS = "peps", "EIPs", "ERCs"


@dataclass(frozen=True)
class Version:
    repo: str
    recorded_ts: int
    commit: str
    path: str | None  # the proposal's file after the commit; for a deletion, the file deleted
    blob: str | None  # None after a deletion
    # File bookkeeping, not presence: "create" when the proposal had no file before this commit,
    # "modify" or "rename" when it had one (same path or another), "delete" when none is left. A
    # symlink's version is a "create" whose status is absent, so the revision stream derives its
    # episode kinds from presence (``stream._changes``), never from this field.
    kind: str
    header: dict[str, str]
    raw_status: str | None
    status: str  # canonical, missing, unknown or Moved; "absent" after a deletion or for a symlink


@dataclass(frozen=True)
class Interval:
    status: str
    start_ts: int
    end_ts: int | None  # None while the status still holds at the pin
    start_commit: str
    is_transition: bool


@dataclass
class Proposal:
    """One proposal's history. ``versions`` are in recorded-time order; ``series`` is at S."""

    family: str
    series: str
    number: int
    versions: list[Version] = field(default_factory=list)

    @property
    def pid(self) -> str:
        return f"{self.series}-{self.number}"

    def version_at(self, ts: int) -> Version | None:
        """The version in effect at epoch second ``ts``: the last one recorded at or before it."""
        index = bisect_right(self.versions, ts, key=lambda v: v.recorded_ts)
        return self.versions[index - 1] if index else None

    def _present_at(self, ts: int) -> Version | None:
        version = self.version_at(ts)
        return None if version is None or version.status == ABSENT else version

    def status_at(self, ts: int) -> str | None:
        """The normalised status at ``ts``; None before the proposal's creation or while absent."""
        version = self._present_at(ts)
        return None if version is None else version.status

    def title_at(self, ts: int) -> str | None:
        version = self._present_at(ts)
        return None if version is None else version.header.get("title")

    def requires_at(self, ts: int) -> list[int]:
        version = self._present_at(ts)
        return [] if version is None else number_list(version.header.get("requires"))

    def superseders_at(self, ts: int) -> list[int]:
        """Superseded-by and, under its pre-2011 PEP name, replaced-by."""
        version = self._present_at(ts)
        return [] if version is None else superseders(version.header)

    def replaces_at(self, ts: int) -> list[int]:
        version = self._present_at(ts)
        return [] if version is None else number_list(version.header.get("replaces"))

    def text_at(self, ts: int, readers: dict[str, BlobReader]) -> str:
        """The file's whole text at ``ts``; empty while there is no proposal text."""
        version = self._present_at(ts)
        if version is None or version.blob is None:
            return ""
        return readers[version.repo].read(version.blob)

    def body_at(self, ts: int, readers: dict[str, BlobReader]) -> str:
        """The text after the header block at ``ts``; empty while there is no proposal text."""
        text = self.text_at(ts, readers)
        return text[header_span(self.family, text)[1] :]

    def intervals(self) -> list[Interval]:
        """The status over time, consecutive equal statuses merged.

        A transition is a change from one status to a different one, and a proposal's first
        status is not one (section 7). So an interval is a transition when it holds a status and
        the last earlier interval that holds one exists, comes after the last absence, and holds
        a different status. Absent, missing and Moved are not statuses; a missing or Moved
        interval in between (a header without a status) is looked past, an absence is not, so
        the first status after a re-addition is not a transition either. Every interval holds
        for a positive length of time.
        """
        in_effect: list[Version] = []
        for version in self.versions:
            if in_effect and in_effect[-1].recorded_ts == version.recorded_ts:
                in_effect[-1] = version  # the earlier one held for no time
            else:
                in_effect.append(version)
        out: list[Interval] = []
        last_status: str | None = None  # of the last status interval since the last absence
        for version in in_effect:
            if out and out[-1].status == version.status:
                continue
            if out:
                last = out[-1]
                out[-1] = Interval(
                    last.status, last.start_ts, version.recorded_ts, last.start_commit,
                    last.is_transition,
                )
            transition = False
            if version.status == ABSENT:
                last_status = None
            elif version.status not in NOT_A_STATUS:
                transition = last_status is not None and last_status != version.status
                last_status = version.status
            out.append(
                Interval(version.status, version.recorded_ts, None, version.commit, transition)
            )
        return out


@dataclass
class Corpus:
    """Every proposal of the corpora, keyed by its id at S, and the walk it was read from.

    ``commits`` is each repository's first-parent chain up to its pin; a commit that changed a
    proposal file carries its full message. ``paths`` are the clone directories; ``authors``
    maps every walked commit to its author name.
    """

    proposals: dict[str, Proposal]
    commits: dict[str, list[Commit]] = field(default_factory=dict)
    pins: dict[str, str] = field(default_factory=dict)
    stitch_conflicts: list[dict] = field(default_factory=list)
    collisions: list[dict] = field(default_factory=list)
    paths: dict[str, Path] = field(default_factory=dict)
    authors: dict[str, str] = field(default_factory=dict)
    near_misses: list[dict] = field(default_factory=list)  # {"repo", "path"}, sorted
    excluded: list[str] = field(default_factory=list)  # ids of the proposals left out (section 2)
    stitch_dropped: dict[str, int] = field(default_factory=dict)  # EIP versions left out, by status

    def as_of_commits(self, d: date) -> dict[str, str | None]:
        """Each repository's as-of-``d`` commit (section 3), or None before its first commit."""
        found = {}
        for name, commits in self.commits.items():
            commit = as_of_commit(commits, d)
            found[name] = None if commit is None else commit.sha
        return found

    @contextmanager
    def readers(self) -> Iterator[dict[str, BlobReader]]:
        """One open ``BlobReader`` per repository, keyed by repository name."""
        with ExitStack() as stack:
            yield {name: stack.enter_context(BlobReader(path)) for name, path in self.paths.items()}


def _iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _number(repo: str, path: str | None) -> int | None:
    """The proposal number a path holds in ``repo``, or None if it is not a proposal file."""
    if path is None:
        return None
    match = REPOS[repo].pattern.match(path)
    if match is not None:
        return int(match.group(1))
    return ALIASES[repo].get(path)


def _is_near_miss(path: str) -> bool:
    """Named like a proposal file, but no repository's pattern or alias, and not a fixture."""
    if LIKE_A_PROPOSAL.search(path) is None:
        return False
    if any(spec.pattern.match(path) or path in ALIASES[name] for name, spec in REPOS.items()):
        return False
    return not any(place.search(path) for place in NOT_PROPOSAL_PLACES)


def _walk_pattern(repo: str) -> re.Pattern[str]:
    """The paths whose changes are read: proposal files, alias paths, and near misses."""
    parts = [
        REPOS[repo].path_re,
        *(f"^{re.escape(path)}$" for path in ALIASES[repo]),
        "^.*" + LIKE_A_PROPOSAL.pattern,  # file_changes matches from the start of the path
    ]
    return re.compile("|".join(f"(?:{part})" for part in parts))


def _layout(repo: str, path: str) -> tuple[int, int, str]:
    """Sorts the newer layout first (section 4); the path itself only breaks a tie."""
    directory, extensions = NEWER_LAYOUT[repo]
    top = path.split("/", 1)[0].casefold() if "/" in path else ""
    extension = path[path.rfind(".") :] if "." in path else ""
    rank = extensions.index(extension) if extension in extensions else len(extensions)
    return (0 if top == directory else 1, rank, path)


@dataclass
class _Files:
    live: dict[str, tuple[str, str | None]] = field(default_factory=dict)  # path -> (blob, mode)
    current: str | None = None  # the proposal's file


@dataclass
class _Walk:
    pin: Commit
    commits: list[Commit]
    versions: dict[int, list[Version]]
    live_at_pin: set[int]
    collisions: list[dict]
    near_misses: set[str]
    authors: dict[str, str]


def _effects(repo: str, group: list[FileChange], near_misses: set[str]):
    """Per proposal number: the paths a commit removes, and the (path, blob, mode) it puts in."""
    removed: dict[int, list[str]] = defaultdict(list)
    added: dict[int, list[tuple[str, str | None, str | None]]] = defaultdict(list)
    for change in group:
        old = change.old_path if change.kind == "R" else change.path if change.kind == "D" else None
        new = None if change.kind == "D" else change.path
        for path in (old, new):
            if path is not None and _is_near_miss(path):
                near_misses.add(path)
        old_number, new_number = _number(repo, old), _number(repo, new)
        if old_number is not None:
            removed[old_number].append(old)
        if new_number is not None:
            added[new_number].append((new, change.new_blob, change.new_mode))
    return removed, added


def _walk(repo: str, path: Path, s: date, expected: str | None = None) -> _Walk:
    spec = REPOS[repo]
    chain = first_parent_commits(path, "HEAD")
    pin = as_of_commit(chain, s)
    if pin is None:
        raise CloneError(f"{repo} has no commit recorded at or before the end of {s.isoformat()}")
    if expected is not None and pin.sha != expected:
        raise CloneError(f"the {repo} clone's pin (its as-of-{s.isoformat()} commit) is {pin.sha}, "
                         f"but the registered pins give {expected}")
    chain = chain[: chain.index(pin) + 1]
    changes = file_changes(path, pin.sha, _walk_pattern(repo), chain)
    with_message = {change.commit.sha: change.commit for change in changes}
    commits = [with_message.get(commit.sha, commit) for commit in chain]

    states: dict[int, _Files] = defaultdict(_Files)
    versions: dict[int, list[Version]] = defaultdict(list)
    collisions: list[dict] = []
    near_misses: set[str] = set()
    with BlobReader(path) as reader:
        for _, grouped in groupby(changes, key=lambda change: change.commit.sha):
            group = list(grouped)
            commit = group[0].commit
            removed, added = _effects(repo, group, near_misses)
            for number in sorted(removed.keys() | added.keys()):
                files = states[number]
                before = files.current
                for gone in removed.get(number, ()):
                    files.live.pop(gone, None)
                changed = []
                for new, blob, mode in added.get(number, ()):
                    files.live[new] = (blob, mode)
                    changed.append(new)
                if not files.live:
                    files.current = None
                    if before is not None:
                        versions[number].append(
                            Version(repo, commit.recorded_ts, commit.sha, before, None, "delete",
                                    {}, None, ABSENT)
                        )
                    continue
                if changed:
                    kept = min(changed, key=lambda p: _layout(repo, p))
                elif before in files.live:
                    kept = before
                else:
                    kept = min(files.live, key=lambda p: _layout(repo, p))
                if len(files.live) > 1:
                    collisions.append(
                        {
                            "repo": repo,
                            "commit": commit.sha,
                            "recorded_at": _iso(commit.recorded_ts),
                            "number": number,
                            "paths": sorted(files.live),
                            "kept": kept,
                        }
                    )
                files.current = kept
                kind = "create" if before is None else "modify" if kept == before else "rename"
                blob, mode = files.live[kept]
                if mode == _SYMLINK:
                    header: dict[str, str] = {}
                    raw = None
                    status = ABSENT
                else:
                    header = parse_header(spec.family, reader.read(blob))
                    raw = header.get("status")
                    status = normalize_status(spec.family, raw)
                versions[number].append(
                    Version(repo, commit.recorded_ts, commit.sha, kept, blob, kind, header, raw,
                            status)
                )
    return _Walk(
        pin=pin,
        commits=commits,
        versions=dict(versions),
        live_at_pin={number for number, files in states.items() if files.current is not None},
        collisions=collisions,
        near_misses=near_misses,
        authors=author_names(path, pin.sha),
    )


def _conflict(pid: str, kind: str, eip: Version, erc: Version) -> dict:
    return {
        "proposal": pid,
        "kind": kind,
        "eip_commit": eip.commit,
        "eip_recorded_at": _iso(eip.recorded_ts),
        "eip_status": eip.status,
        "erc_commit": erc.commit,
        "erc_recorded_at": _iso(erc.recorded_ts),
        "erc_status": erc.status,
    }


def _stitch(pid: str, eip_versions: list[Version], erc_versions: list[Version]):
    """An ERC's versions: its EIP history before the ERC file existed, then the ERC history.

    EIP versions recorded on or after ``ERC_MOVE_START`` whose status is missing or Moved are the
    move's stubs and markers and are left out; so is every EIP version recorded once the ERC file
    exists, where the ERC repository's values win. A conflict is reported if the last EIP version
    kept has a status other than the ERC file's first, or if an EIP version recorded after the
    ERC file's creation sets a status other than missing or Moved (section 6).
    """
    first = erc_versions[0]
    move_start = end_of_day_ts(ERC_MOVE_START) - 86399
    kept: list[Version] = []
    dropped: list[Version] = []
    late_conflicts: list[dict] = []
    for version in eip_versions:
        if version.recorded_ts >= first.recorded_ts:
            dropped.append(version)
            if version.status not in (MISSING, MOVED, ABSENT):
                late_conflicts.append(
                    _conflict(pid, "eip_status_after_erc_created", version, first)
                )
        elif version.recorded_ts >= move_start and version.status in (MISSING, MOVED):
            dropped.append(version)
        else:
            kept.append(version)
    conflicts = []
    if kept and kept[-1].status != first.status:
        conflicts.append(_conflict(pid, "first_erc_status_differs", kept[-1], first))
    return kept + erc_versions, conflicts + late_conflicts, dropped


def load_corpus(clones: Path, s: date = S, expect_pins: dict[str, str] | None = None) -> Corpus:
    """Walk ``clones/peps``, ``clones/EIPs`` and ``clones/ERCs`` up to their as-of-``s`` pins.

    Each clone must be a full, non-shallow clone read as itself (``gitwalk.check_clone``). With
    ``expect_pins`` (a registered manifest's ``pins``), a clone whose pin differs is refused
    before its history is read. Both raise ``CloneError``.
    """
    clones = Path(clones)
    for repo in REPOS:
        check_clone(clones / repo)
        if expect_pins is not None and not isinstance(expect_pins.get(repo), str):
            raise CloneError(f"the registered pins give no commit for {repo}")
    walks = {repo: _walk(repo, clones / repo, s, None if expect_pins is None else expect_pins[repo])
             for repo in REPOS}
    corpus = Corpus(
        proposals={},
        commits={repo: walk.commits for repo, walk in walks.items()},
        pins={repo: walk.pin.sha for repo, walk in walks.items()},
        paths={repo: clones / repo for repo in REPOS},
    )
    for walk in walks.values():
        corpus.authors.update(walk.authors)

    pep, eip, erc = REPOS[_PEPS], REPOS[_EIPS], REPOS[_ERCS]
    proposals = [
        Proposal(pep.family, pep.series, number, versions)
        for number, versions in sorted(walks[_PEPS].versions.items())
    ]
    dropped: Counter[str] = Counter()
    eips, ercs = walks[_EIPS], walks[_ERCS]
    for number in sorted(eips.versions.keys() | ercs.versions.keys()):
        # Both repositories draw on one number space. The series at S is ERC when the ERC file
        # exists at the ERC pin; a number with any ERC history is stitched, so no version is
        # lost if an ERC file were ever deleted (none is, up to the pins).
        series = erc.series if number in ercs.live_at_pin else eip.series
        versions = eips.versions.get(number, [])
        if number in ercs.versions:
            versions, conflicts, left_out = _stitch(
                f"{series}-{number}", versions, ercs.versions[number]
            )
            corpus.stitch_conflicts.extend(conflicts)
            dropped.update(version.status for version in left_out)
        proposals.append(Proposal(eip.family, series, number, versions))
    corpus.stitch_dropped = dict(sorted(dropped.items()))

    pid_of = {(p.family, p.number): p.pid for p in proposals}
    for repo, walk in walks.items():
        for collision in walk.collisions:
            number = collision.pop("number")
            collision["proposal"] = pid_of[(REPOS[repo].family, number)]
            corpus.collisions.append(collision)

    for p in proposals:
        statuses = {v.status for v in p.versions} & set(CANONICAL[p.family])
        if (p.family, p.number) in EXCLUDED or statuses == {APRIL_FOOL}:
            corpus.excluded.append(p.pid)
        else:
            corpus.proposals[p.pid] = p
    corpus.near_misses = sorted(
        ({"repo": repo, "path": path} for repo, walk in walks.items() for path in walk.near_misses),
        key=lambda item: (item["repo"], item["path"]),
    )
    return corpus


_MONTHS = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)
_PIPERMAIL = re.compile(r"pipermail/([^/\s]+)/(\d{4})-([A-Za-z]+)/")


def _resolution_lag(corpus: Corpus) -> dict:
    """Months from a PEP's announced decision to its recorded one (section 3's cross-check).

    For each PEP, the first version whose Resolution links a dated ``pipermail/<list>/<YYYY>-
    <Month>/`` archive gives the decided month; the recorded month is when the status in effect
    at that version began, since questions ask about statuses (a link added later than its
    status does not count as a late decision).
    """
    buckets = Counter({"same_month": 0, "1_to_3_months_later": 0, "more_than_3_later": 0,
                       "earlier": 0})
    n = 0
    for p in corpus.proposals.values():
        if p.family != REPOS[_PEPS].family:
            continue
        for version in p.versions:
            match = _PIPERMAIL.search(version.header.get("resolution", ""))
            if match is None or match.group(3).casefold() not in _MONTHS:
                continue
            archived = int(match.group(2)) * 12 + _MONTHS.index(match.group(3).casefold())
            began = next(
                i.start_ts
                for i in reversed(p.intervals())
                if i.start_ts <= version.recorded_ts
            )
            recorded = datetime.fromtimestamp(began, tz=UTC)
            lag = recorded.year * 12 + recorded.month - 1 - archived
            n += 1
            if lag < 0:
                buckets["earlier"] += 1
            elif lag == 0:
                buckets["same_month"] += 1
            elif lag <= 3:
                buckets["1_to_3_months_later"] += 1
            else:
                buckets["more_than_3_later"] += 1
            break  # the first version whose Resolution links a dated archive
    return {"n": n, **buckets}


def quality_report(corpus: Corpus) -> dict:
    """Counts and lists that describe the timelines and every irregularity found while reading.

    Proposals, versions, intervals and transitions are counted per series; an interval is a
    status interval unless it is absent. Inversions are first-parent commits whose committer
    time is earlier than their parent's (section 3), with the largest gap in days. Moves to
    Stagnant are counted by the bot account that makes most of them and by people, without
    names: the report publishes no personal names.
    """
    by_series: dict[str, list[Proposal]] = {spec.series: [] for spec in REPOS.values()}
    for p in corpus.proposals.values():
        by_series.setdefault(p.series, []).append(p)

    proposals, versions, intervals, transitions, missing = {}, {}, {}, {}, {}
    unknown: Counter[str] = Counter()
    stagnant: Counter[str] = Counter()
    for name, ps in by_series.items():
        present = sum(1 for p in ps if p.versions and p.versions[-1].status != ABSENT)
        proposals[name] = {"proposals": len(ps), "present_at_pin": present}
        versions[name] = sum(len(p.versions) for p in ps)
        found = [i for p in ps for i in p.intervals()]
        absent = sum(1 for i in found if i.status == ABSENT)
        intervals[name] = {"status": len(found) - absent, "absent": absent}
        transitions[name] = sum(1 for i in found if i.is_transition)
        for i in found:
            if i.status == STAGNANT and i.is_transition:  # by the bot account, or by a person
                bot = corpus.authors.get(i.start_commit) == STAGNANT_BOT
                stagnant["by_eth_bot_account" if bot else "by_people"] += 1
        where = []
        for p in ps:
            for v in p.versions:
                if v.status == UNKNOWN:
                    unknown[v.raw_status or ""] += 1
                elif v.status == MISSING:
                    where.append({"proposal": p.pid, "commit": v.commit, "path": v.path})
        missing[name] = {"versions": len(where), "where": where}

    inversions = {}
    for repo, commits in corpus.commits.items():
        gaps = [a.committer_ts - b.committer_ts for a, b in pairwise(commits)
                if b.committer_ts < a.committer_ts]
        largest = round(max(gaps) / 86400, 3) if gaps else 0.0
        inversions[repo] = {"count": len(gaps), "largest_gap_days": largest}

    ercs = by_series.get(REPOS[_ERCS].series, [])
    moved_in = sum(1 for p in ercs if p.versions and p.versions[0].repo == _EIPS)
    return {
        "proposals_by_series": proposals,
        "versions": versions,
        "intervals": intervals,
        "transitions": transitions,
        "unknown_raw_values": dict(sorted(unknown.items())),
        "missing_values": missing,
        "collisions": list(corpus.collisions),
        "stitch_conflicts": list(corpus.stitch_conflicts),
        "inversions": inversions,
        "stagnant_moves": {"total": sum(stagnant.values()),
                           "by_eth_bot_account": stagnant["by_eth_bot_account"],
                           "by_people": stagnant["by_people"]},
        "pep_resolution_lag_months": _resolution_lag(corpus),
        "near_miss_paths": list(corpus.near_misses),
        "excluded": list(corpus.excluded),
        "stitched": {
            "ercs": len(ercs),
            "with_eip_history": moved_in,
            "eip_versions_left_out": dict(corpus.stitch_dropped),
        },
    }
