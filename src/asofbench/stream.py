"""The dated revision stream (spec section 8) and its counterfactual arm (section 7, CF).

This module builds one arm's dated revision stream at a time: one episode per proposal file changed
by one first-parent commit, in recorded-time order. What each system receives of it is section 8's
and section 11's rule. ``asofbench stream --arm H+`` writes it as recorded, ``--arm H-``
withholds the header fields of ``headers.WITHHELD`` from both versions of the file before any text
or diff is computed, and ``--arm CF`` permutes the status header values of the counterfactual
proposals.

How the history is read (sections 6 to 8):

* **Kinds follow presence**, version by version, not file bookkeeping: absent -> present is a
  ``create`` (the full text), present -> present a ``modify`` or, when the file's path changed, a
  ``rename`` (a unified diff of the two texts), present -> absent a ``delete`` (no text). A
  symlink version is absent (the timeline says so), so a symlink replaced by a regular file is a
  ``create``. A change that leaves the content identical, a mode change, is a ``modify`` with an
  empty diff. Versions the ERC stitching dropped give no episodes; the first ERC-repository
  version of an ERC that began in the EIP repository is a ``rename`` against the last EIP text.
* **``proposal`` is the id at S** (``ERC-20`` even for EIP-repository episodes) and ``repo`` is the
  repository the version came from. ``ref`` is the string the questions use.
* **Order** is ``(recorded_ts, repo, position of the commit in that repository's first-parent
  chain, proposal id)``; the proposal ids are compared as strings. PEP 0 and the other excluded
  proposals are not in the corpus and so never appear.
* **H-**: both versions pass through ``strip_withheld`` before diffing and the stripped text is
  used for a create. An episode whose diff is empty keeps its commit message; those whose H+
  diff is not empty, commits that changed only withheld header lines, are what
  ``header_only_times`` finds for the F1 items' gold (section 12).
* **CF**: for the counterfactual proposals only, in both versions of every episode, each status
  header value is replaced by ``permute(family, normalize_status(family, raw))`` when that differs
  from the canonical status (a value the permutation leaves alone, such as ``Incomplete``, stays
  as written), then the diff is taken. A modification or rename whose before and after raw status
  values differ changes the status line; a create or a delete never does. In every commit that
  changes a CF status line, the episode of that proposal carries the message
  ``Update status of {ref}`` and every other episode of the commit carries ``Update {ref}`` with
  its own proposal's ref, so that no sibling's message can name the true status. Episodes of
  commits that change no CF status line keep the commit's message, and only a CF proposal's text
  differs from H+.

The messages of commits that change no CF status line are not edited, so some still mention a CF
proposal and name a status; ``cf_message_counts`` counts those commits, each once, per CF
proposal, and the CF command writes the counts beside the stream for the scorer's report (section
7). The H- arm keeps every message, and ``message_cues`` lists per proposal the times and status
values of those that mention it, which the H- command writes beside its stream so that the scorer
can say what the messages alone reveal (section 12).

The stats of a stream are its episode count, ``chars``, the number of characters of episode
``text`` (what a system reads of the files), ``tokens_est``, ``chars // 4``, and the file's
SHA-256 (section 8). ``asofbench build --streams`` records them in the manifest with the SHA-256
and size of each file written beside the stream (``manifest.SIDECARS``), and a stream that
``asofbench stream`` builds for a build whose manifest registers its arm must have the same
stats and the same files beside it.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import sys
from collections import defaultdict
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

from .corpora import CANARY
from .gitwalk import BlobReader, CloneError
from .headers import _scan, strip_withheld  # one scanner finds the header, so it is shared
from .manifest import (
    CF_MESSAGES_FILE,
    CUES_FILE,
    STATS_FILE,
    ManifestError,
    read_manifest,
    registered_pins,
    registered_stream,
)
from .manifest import STREAM_STATS as STATS
from .mentions import mentions
from .questions import CF_STRATUM, dumps, iso_ts
from .questions import ref as question_ref
from .statuses import CANONICAL, normalize_status, permute
from .timeline import ABSENT, Corpus, Proposal, Version, load_corpus

ARMS = ("H+", "H-", "CF")
KINDS = ("create", "modify", "rename", "delete")
NO_NEWLINE = "\\ No newline at end of file\n"
# The names ``asofbench build --streams`` writes each arm's stream under.
STREAM_FILES = {"H+": "h-plus.jsonl", "H-": "h-minus.jsonl", "CF": "cf.jsonl"}

_CF_ID = re.compile(rf"CF-{re.escape(CF_STRATUM)}-((?:PEP|EIP|ERC)-\d+)-\d{{4}}-\d{{2}}-\d{{2}}")
_KEY_PREFIX = re.compile(r"[ \t]*[A-Za-z][A-Za-z0-9-]*[ \t]*:[ \t]*")


class StreamError(Exception):
    """An input problem that ``main`` reports instead of a traceback."""


@dataclass(frozen=True)
class Episode:
    """One proposal file changed by one first-parent commit (section 8)."""

    episode_id: str  # f"{commit[:12]}:{proposal}"
    corpus: str  # "pep" or "eth", as in the questions
    repo: str  # the repository the version came from: peps, EIPs or ERCs
    proposal: str  # the id at S, such as PEP-572 or ERC-20
    ref: str  # as the questions write it: PEP 572 ("Assignment Expressions")
    recorded_at: str  # the commit's recorded time, UTC
    commit: str
    message: str
    kind: str  # create, modify, rename or delete
    text: str  # create: the full file; modify and rename: a unified diff; delete: empty


def _lines(text: str) -> list[str]:
    """``text`` as lines split at newline characters only, each keeping its newline.

    A form feed, a lone carriage return or a Unicode line separator stays inside its line, which
    ``str.splitlines`` would cut at. The last line has no newline if the text does not end with one.
    """
    parts = text.split("\n")
    lines = [part + "\n" for part in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines


def unified_diff(before: str, after: str) -> str:
    """The unified diff of two texts with three lines of context, labelled ``a`` and ``b``.

    Lines are split at newline characters only (``_lines``), and a last line without a newline,
    on either side and as a changed or a context line, is followed by git's marker line
    ``\\ No newline at end of file``. Every row of the diff is then one line of a text, the hunk
    counts are those of its rows, and the diff applied to ``before`` gives ``after`` exactly. It
    is the one function that computes every diff of every arm (section 8); git is not asked.
    """
    rows = []
    for row in difflib.unified_diff(_lines(before), _lines(after), "a", "b", n=3):
        rows.append(row)
        if not row.endswith("\n"):  # only a text's open last line can lack a newline here
            rows.append("\n" + NO_NEWLINE)
    return "".join(rows)


def _changes(p: Proposal) -> Iterator[tuple[Version, Version | None, Version | None, str]]:
    """``(version, before, after, kind)`` for each change in the proposal's presence.

    ``before`` and ``after`` are the versions whose text stood before and after the change, or
    None where the proposal had no text (not yet created, deleted, or a symlink).
    """
    before: Version | None = None
    for version in p.versions:
        after = None if version.status == ABSENT else version
        if before is None and after is None:
            continue  # absent before and after: a symlink, or a deletion of what was a symlink
        if before is None:
            kind = "create"
        elif after is None:
            kind = "delete"
        elif (before.repo, before.path) != (after.repo, after.path):
            kind = "rename"
        else:
            kind = "modify"
        yield version, before, after, kind
        before = after


def header_only_times(p: Proposal, reader: dict[str, BlobReader],
                      after: int | None = None) -> list[int]:
    """The recorded times of the proposal's H- episodes whose diff the withholding empties.

    Such an episode is a modify or a rename (``_changes``, the stream's own episodes) whose two
    texts differ, so that its H+ diff is not empty, and are equal once ``strip_withheld`` has
    removed the withheld fields from both, as the H- arm removes them before any diff is
    computed (``_generate``), so that its H- diff is empty (two texts have an empty diff exactly
    when they are equal, ``unified_diff``). Its commit changed only withheld header lines, and
    section 8 keeps the episode with its message. With ``after``, an epoch second, only the
    episodes recorded after it are read.
    """
    found = []
    for version, before, now, kind in _changes(p):
        if kind not in ("modify", "rename") or (after is not None and version.recorded_ts <= after):
            continue
        old, new = reader[before.repo].read(before.blob), reader[now.repo].read(now.blob)
        if old != new and strip_withheld(p.family, old) == strip_withheld(p.family, new):
            found.append(version.recorded_ts)
    return found


def _permute_status(family: str, text: str) -> str:
    """``text`` with each status header value replaced by its counterfactual value.

    The field is found by the scanner that reads the headers, so it is the same field that
    ``parse_header`` reads (and any repeated one). Only the value changes: the key as written,
    the colon, the indentation and the line ending stay, and any continuation lines of the field
    go. A value that the permutation leaves alone, or that names no status, is left untouched.
    """
    edits = []
    for field in _scan(family, text).fields:
        if field.key != "status":
            continue
        status = normalize_status(family, field.value)
        permuted = permute(family, status)
        if permuted == status:
            continue
        block = text[field.start : field.end]
        prefix = _KEY_PREFIX.match(block)
        if prefix is None:
            raise AssertionError(f"a status field with no key to rewrite: {block!r}")
        ending = block[len(block.rstrip("\r\n")) :]
        edits.append((field.start, field.end, block[: prefix.end()] + permuted + ending))
    for start, end, replacement in reversed(edits):
        text = text[:start] + replacement + text[end:]
    return text


def _plan(corpus: Corpus, counterfactual: dict[str, str]):
    """Every episode in stream order, as ``(key, proposal, version, before, after, kind,
    sets_status)``, with the set of commits that change a CF status line, the commit messages and
    the refs. No blob is read."""
    position = {
        repo: {commit.sha: index for index, commit in enumerate(chain)}
        for repo, chain in corpus.commits.items()
    }
    messages = {(repo, c.sha): c.message for repo, chain in corpus.commits.items() for c in chain}
    refs: dict[str, str] = {}
    pending = []
    for pid, p in corpus.proposals.items():
        for version, before, after, kind in _changes(p):
            try:
                where = position[version.repo][version.commit]
            except KeyError:
                raise ValueError(
                    f"{pid}: commit {version.commit} is not in the {version.repo} chain"
                ) from None
            key = (version.recorded_ts, version.repo, where, pid)
            # A modification or rename that changes a CF proposal's status line (its raw value).
            sets_status = (pid in counterfactual and kind in ("modify", "rename")
                           and before.raw_status != after.raw_status)
            pending.append((key, p, version, before, after, kind, sets_status))
        refs[pid] = question_ref(p)
    pending.sort(key=lambda item: item[0])
    # Every episode of a commit that changes a CF status line has its message replaced (section 7),
    # so that no sibling episode's message can name the true status.
    status_commits = {item[2].commit for item in pending if item[6]}
    return pending, status_commits, messages, refs


def _generate(
    corpus: Corpus, reader: dict[str, BlobReader], arm: str, counterfactual: dict[str, str]
) -> Iterator[Episode]:
    pending, status_commits, messages, refs = _plan(corpus, counterfactual)

    def view(p: Proposal, v: Version) -> str:
        text = reader[v.repo].read(v.blob)
        if arm == "H-":
            return strip_withheld(p.family, text)
        if p.pid in counterfactual:
            return _permute_status(p.family, text)
        return text

    for _, p, version, before, after, kind, sets_status in pending:
        pid = p.pid
        message = messages[(version.repo, version.commit)]
        if version.commit in status_commits:
            message = (f"Update status of {counterfactual[pid]}" if sets_status
                       else f"Update {refs[pid]}")
        if kind == "create":
            text = view(p, after)
        elif kind == "delete":
            text = ""
        else:
            text = unified_diff(view(p, before), view(p, after))
        yield Episode(
            episode_id=f"{version.commit[:12]}:{pid}",
            corpus=p.family,
            repo=version.repo,
            proposal=pid,
            ref=refs[pid],
            recorded_at=iso_ts(version.recorded_ts),
            commit=version.commit,
            message=message,
            kind=kind,
            text=text,
        )


def _status_pattern(family: str) -> re.Pattern[str]:
    """The canonical status values of a corpus as whole words, in any letter case; a value of
    several words (Last Call) may be split by any whitespace. The value matched is the n-th of
    ``CANONICAL[family]`` for the group ``s<n>`` that matched (``_matched_status``), since a
    spelling the pattern accepts, such as FINAL with a dotted capital I (U+0130), need not fold
    back to the value's own letters."""
    values = [f"(?P<s{n}>" + r"\s+".join(re.escape(word) for word in status.split()) + ")"
              for n, status in enumerate(CANONICAL[family])]
    return re.compile(r"(?<!\w)(?:" + "|".join(values) + r")(?!\w)", re.IGNORECASE)


def _matched_status(family: str, match: re.Match[str]) -> str:
    """The canonical value a match of ``_status_pattern(family)`` stands for."""
    return CANONICAL[family][int(match.lastgroup[1:])]


def _kept_messages(pending: list, messages: dict, skip: set[str]) -> list[tuple[int, str]]:
    """``(recorded_ts, message)`` of every commit that has an episode in the stream, each commit
    once and in stream order, leaving out the commits in ``skip``."""
    seen: set[str] = set()
    found = []
    for _, _, version, _, _, _, _ in pending:
        if version.commit in skip or version.commit in seen:
            continue
        seen.add(version.commit)
        found.append((version.recorded_ts, messages[(version.repo, version.commit)]))
    return found


def cf_message_counts(corpus: Corpus, cf: dict[str, str]) -> dict[str, int]:
    """For each CF proposal, how many commits outside the rewritten ones have a message that
    mentions it and contains one of its corpus's canonical status values (section 7), counting
    each commit once however many of its files changed.

    A message mentions a proposal by the mention rule of section 7 (``mentions.mentions``, with no
    proposal of its own to leave out, so a message that says PEP 1 mentions PEP 1 whichever
    episode carries it), read in the proposal's own corpus; a status value counts as a whole word
    in any letter case. The rewritten commits are those that change a CF status line, whose
    messages the CF stream replaces; every other commit with an episode in the stream keeps its
    message, and it is those that are counted.
    """
    absent = sorted(set(cf) - set(corpus.proposals))
    if absent:
        raise ValueError(f"CF proposals that are not in the corpus: {', '.join(absent)}")
    pending, status_commits, messages, _ = _plan(corpus, dict(cf))
    patterns = {family: _status_pattern(family) for family in CANONICAL}
    counts = dict.fromkeys(cf, 0)
    for _, message in _kept_messages(pending, messages, status_commits):
        named = {family: mentions(family, message, -1) for family in patterns}
        stated = {family: patterns[family].search(message) is not None for family in patterns}
        for pid in cf:
            p = corpus.proposals[pid]
            if stated[p.family] and p.number in named[p.family]:
                counts[pid] += 1
    return counts


def message_cues(corpus: Corpus) -> dict[str, list[list[str]]]:
    """For each proposal, the ``[recorded_at, status]`` pairs of the messages the H- stream keeps
    (every commit's, once) that mention it and contain a canonical status value (section 12).

    A message mentions a proposal by the mention rule of section 7, read in the proposal's own
    corpus, as ``cf_message_counts`` reads it; each canonical status value of that corpus that the
    message contains as a whole word, in any letter case, gives one pair. The pairs are sorted and
    de-duplicated; a proposal with none is left out. The scorer reads them with ``--message-cues``
    to say, for each H- F1 item, whether a message at or before its date names its gold status.
    """
    pending, _, messages, _ = _plan(corpus, {})
    by_number: dict[str, dict[int, str]] = {family: {} for family in CANONICAL}
    for pid, p in corpus.proposals.items():
        by_number[p.family][p.number] = pid
    patterns = {family: _status_pattern(family) for family in CANONICAL}
    found: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for recorded_ts, message in _kept_messages(pending, messages, set()):
        when = iso_ts(recorded_ts)
        for family, pattern in patterns.items():
            statuses = {_matched_status(family, m) for m in pattern.finditer(message)}
            if not statuses:
                continue
            for number in mentions(family, message, -1):
                pid = by_number[family].get(number)
                if pid is not None:
                    found[pid].update((when, status) for status in statuses)
    return {pid: [list(pair) for pair in sorted(pairs)] for pid, pairs in sorted(found.items())}


def episodes(
    corpus: Corpus,
    reader: dict[str, BlobReader],
    arm: str,
    cf: dict[str, str] | None = None,
) -> Iterator[Episode]:
    """The episodes of ``arm`` (H+, H- or CF) in recorded-time order.

    ``reader`` is ``Corpus.readers()``: one open ``BlobReader`` per repository, by name. For the
    CF arm, ``cf`` maps the id of each counterfactual proposal to its ref. The arguments are
    checked now; the episodes are read from git as they are consumed.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}: expected one of {', '.join(ARMS)}")
    counterfactual: dict[str, str] = {}
    if arm == "CF":
        if not cf:
            raise ValueError("arm CF needs cf, the counterfactual proposals and their refs")
        absent = sorted(set(cf) - set(corpus.proposals))
        if absent:
            raise ValueError(f"CF proposals that are not in the corpus: {', '.join(absent)}")
        counterfactual = dict(cf)
    return _generate(corpus, reader, arm, counterfactual)


def _json_bytes(content: dict) -> bytes:
    return (dumps(content) + "\n").encode("utf-8")


def _file_entry(entry: dict | None) -> str:
    return "none" if entry is None else f"sha256 {entry['sha256']}, {entry['bytes']} bytes"


def write_stream(
    corpus: Corpus,
    reader: dict[str, BlobReader],
    arm: str,
    cf: dict[str, str] | None,
    out: Path,
    expected: dict | None = None,
    beside: dict[str, bytes] | None = None,
) -> dict:
    """Write the stream to ``out`` as JSONL, with ``<out>.stats.json`` and the files of
    ``beside`` (suffix -> the bytes of ``<out>.<suffix>``) next to it, and return its entry for
    the manifest: the episode count, the characters of episode text, the tokens estimated from
    them, the file's SHA-256 and, under ``sidecars``, each file beside it with its SHA-256 and
    size.

    The stream appears whole or not at all, and an episode id may not repeat. With ``expected``
    (what a manifest registers for the arm, section 8), a stream whose stats differ, or a file
    beside it that is not the one registered, is not written, nor is anything else, and
    ``StreamError`` says what differs; a manifest that registers no files beside the stream
    checks the stats alone.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    seen: set[str] = set()
    count = chars = 0
    digest = sha256()
    try:
        with partial.open("wb") as f:
            for episode in episodes(corpus, reader, arm, cf):
                if episode.episode_id in seen:
                    raise ValueError(f"episode id {episode.episode_id} is not unique")
                seen.add(episode.episode_id)
                line = (dumps(asdict(episode)) + "\n").encode("utf-8")
                f.write(line)
                digest.update(line)
                count += 1
                chars += len(episode.text)
        stats = {"episodes": count, "chars": chars, "tokens_est": chars // 4,
                 "sha256": digest.hexdigest()}
        files = {STATS_FILE: _json_bytes(stats), **(beside or {})}
        sidecars = {suffix: {"bytes": len(data), "sha256": sha256(data).hexdigest()}
                    for suffix, data in sorted(files.items())}
        if expected is not None:
            differs = [f"{key} {stats[key]} here, {expected.get(key)} registered"
                       for key in STATS if stats[key] != expected.get(key)]
            registered = expected.get("sidecars")
            for suffix in sorted(set(sidecars) | set(registered or ())):
                if registered is not None and sidecars.get(suffix) != registered.get(suffix):
                    differs.append(f"{out.name}.{suffix} {_file_entry(sidecars.get(suffix))} "
                                   f"here, {_file_entry(registered.get(suffix))} registered")
            if differs:
                raise StreamError(f"the {arm} stream is not the one registered in the manifest: "
                                  + "; ".join(differs))
        os.replace(partial, out)
        for suffix, data in files.items():
            Path(f"{out}.{suffix}").write_bytes(data)
    finally:
        with suppress(FileNotFoundError):
            partial.unlink()
    return {**stats, "sidecars": sidecars}


def write_arm(
    corpus: Corpus,
    reader: dict[str, BlobReader],
    arm: str,
    cf: dict[str, str] | None,
    out: Path,
    expected: dict | None = None,
) -> dict:
    """Write one arm's stream to ``out`` with the files beside it, and return its entry for the
    manifest (``write_stream``).

    Beside it go ``<out>.stats.json``; for the CF arm ``<out>.cf_messages.json``, the canary and
    each CF proposal's ``cf_message_counts`` (section 7); and for the H- arm
    ``<out>.message_cues.json``, the canary and each proposal's ``message_cues`` (section 12).
    """
    beside = {}
    if arm == "CF":
        beside[CF_MESSAGES_FILE] = _json_bytes({"canary": CANARY, **cf_message_counts(corpus, cf)})
    if arm == "H-":
        beside[CUES_FILE] = _json_bytes({"canary": CANARY, **message_cues(corpus)})
    return write_stream(corpus, reader, arm, cf, out, expected, beside)


def write_streams(corpus: Corpus, cf: dict[str, str], root: Path) -> dict[str, dict]:
    """Write the three arms into ``root`` under ``STREAM_FILES``' names (``asofbench build
    --streams``); return each arm's entry for the manifest."""
    root = Path(root)
    with corpus.readers() as reader:
        return {arm: write_arm(corpus, reader, arm, cf if arm == "CF" else None, root / name)
                for arm, name in STREAM_FILES.items()}


def read_cf(data: Path) -> dict[str, str]:
    """The counterfactual proposals and their refs, from the CF items of ``data/gold.jsonl``."""
    path = Path(data) / "gold.jsonl"
    if not path.is_file():
        raise StreamError(f"{path} not found: run `asofbench build` first")
    found: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                item_id = record["id"]
            except (ValueError, KeyError, TypeError):
                raise StreamError(f"{path} line {number}: not a gold record") from None
            if not item_id.startswith("CF-"):
                continue
            match = _CF_ID.fullmatch(item_id)
            if match is None:
                raise StreamError(f"{path} line {number}: no proposal in the CF id {item_id!r}")
            try:
                ref = record["meta"]["ref"]
            except (KeyError, TypeError):
                raise StreamError(f"{path} line {number}: {item_id} has no meta.ref") from None
            if found.setdefault(match.group(1), ref) != ref:
                raise StreamError(f"{path} line {number}: two refs for {match.group(1)}")
    if not found:
        raise StreamError(f"{path} has no CF items")
    return found


def _registered(data: Path) -> dict | None:
    """The build's manifest in ``data``, or None if there is none (a build without one)."""
    path = Path(data) / "MANIFEST.json"
    return read_manifest(path) if path.is_file() else None


def main(args) -> int:
    """``asofbench stream``: write one arm's stream, with the files beside it that ``write_arm``
    describes, and print its stats as JSON.

    The stream and the questions must be read at the same pinned commits: with a manifest in
    ``--data``, clones whose pins differ from its own are refused, and when the manifest registers
    the arm's stream (``build --streams``) a stream, or a file beside it, that differs from it is
    not written.
    """
    out = Path(args.out)
    manifest_path = Path(args.data) / "MANIFEST.json"
    try:
        cf = read_cf(args.data) if args.arm == "CF" else None
        manifest = _registered(args.data)
        pins = None
        if manifest is not None and manifest.get("pins") is not None:
            pins = registered_pins(manifest, manifest_path)
        expected = registered_stream(manifest, args.arm, manifest_path)
        corpus = load_corpus(Path(args.clones), expect_pins=pins)
        if cf is not None:
            absent = sorted(set(cf) - set(corpus.proposals))
            if absent:
                raise StreamError(f"CF proposals that are not in the clones: {', '.join(absent)}")
        with corpus.readers() as reader:
            entry = write_arm(corpus, reader, args.arm, cf, out, expected)
    except (StreamError, CloneError, ManifestError) as e:
        print(f"asofbench stream: {e}", file=sys.stderr)
        return 2
    print(dumps({key: entry[key] for key in STATS}))
    return 0
