"""Timelines on the real clones (corpus-marked; set ASOFBENCH_CLONES to run them).

The corpus is built once for the module. Checks that read git do so independently of the
walker: ``git show <pin>:<path>`` resolves a path in the pinned tree, and the inversion count is
taken from ``git log --format=%ct``.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

from asofbench.corpora import EXCLUDED, REPOS, S
from asofbench.gitwalk import end_of_day_ts
from asofbench.headers import parse_header
from asofbench.statuses import normalize_status
from asofbench.timeline import load_corpus, quality_report

pytestmark = pytest.mark.corpus

EOD = end_of_day_ts(S)


def git(repo, *args: str) -> bytes:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL="/dev/null")
    cmd = ["git", "-C", str(repo), *args]
    return subprocess.run(cmd, env=env, capture_output=True, check=True).stdout


@pytest.fixture(scope="module")
def loaded(clones):
    start = time.monotonic()
    corpus = load_corpus(clones)
    return corpus, time.monotonic() - start


# Every header key the corpora use, by its name without a qualifier, judged by hand (section 8).
# Withheld keys carry a status or a relation, or a date or link that implies one; identity keys are
# the number, title, author, created, type and category; allowed keys carry neither.
KEY_TABLE = {
    "withheld": {
        "status", "resolution", "accepted", "superseded-by", "replaced-by", "replaces",
        "requires", "require", "dependencies", "last-call-deadline", "review-period-end",
        "review-period-ends", "review-end-date", "withdrawal-reason",
    },
    "identity": {
        "pep", "apep",  # "aPEP: 7" is one version's misspelt number line
        "eip", "title", "author", "authors", "owner",  # PEP 1's first header said Owner
        "created", "type", "category",
    },
    "allowed": {
        "post-history", "posted", "version", "content-type", "python-version", "last-modified",
        "discussions-to", "discussion-to", "bdfl-delegate", "pep-delegate", "sponsor", "editor",
        "topic", "release-date", "description", "updated", "edited", "layer",
    },
}


def test_every_header_key_is_classified(corpus):
    """A key that is new to the corpora fails here until it is judged: a status-bearing one must
    join ``headers.WITHHELD`` before the H- stream can be trusted (section 8)."""
    from asofbench.headers import WITHHELD

    seen: dict[str, set[str]] = {"pep": set(), "eth": set()}
    for p in corpus.proposals.values():
        for v in p.versions:
            seen[p.family].update(key.split(" (", 1)[0] for key in v.header)
    withheld, identity, allowed = (KEY_TABLE[k] for k in ("withheld", "identity", "allowed"))
    assert not (withheld & identity or withheld & allowed or identity & allowed)
    assert withheld == set(WITHHELD)
    used = seen["pep"] | seen["eth"]
    assert sorted(used - (withheld | identity | allowed)) == []  # every key is judged
    assert sorted((withheld | identity | allowed) - used) == []  # and the table holds no others
    assert (len(seen["pep"]), len(seen["eth"])) == (29, 25)


@pytest.fixture(scope="module")
def corpus(loaded):
    return loaded[0]


@pytest.fixture(scope="module")
def report(corpus):
    return quality_report(corpus)


def present(corpus):
    """Proposals with a proposal text at S, and the version it is."""
    found = {}
    for pid, p in corpus.proposals.items():
        v = p.version_at(EOD)
        if v is not None and v.status != "absent":
            found[pid] = v
    return found


def test_status_at_pin_matches_git_show(corpus):
    mismatches = []
    for pid, v in present(corpus).items():
        p = corpus.proposals[pid]
        assert v is p.versions[-1], pid  # nothing after the pin was walked
        text = git(corpus.paths[v.repo], "show", "--no-textconv", f"{corpus.pins[v.repo]}:{v.path}")
        header = parse_header(p.family, text.decode("utf-8", errors="replace"))
        expected = normalize_status(p.family, header.get("status"))
        if p.status_at(EOD) != expected:
            mismatches.append((pid, v.repo, v.path, p.status_at(EOD), expected))
    assert mismatches == []
    assert len(present(corpus)) > 1300


def files_at_pin(corpus, repo: str, directory: str) -> list[str]:
    """Regular files in one directory of the pinned tree (``cat-file -p`` lists a tree)."""
    spec = f"{corpus.pins[repo]}:{directory}" if directory else f"{corpus.pins[repo]}^{{tree}}"
    names = []
    for line in git(corpus.paths[repo], "cat-file", "-p", spec).decode("utf-8").splitlines():
        meta, _, name = line.partition("\t")
        mode, kind = meta.split()[:2]
        if kind == "blob" and mode != "120000":  # a symlink is not a proposal text
            names.append(f"{directory}/{name}" if directory else name)
    return names


def test_every_proposal_file_at_the_pin_has_a_present_proposal(corpus):
    expected = {}
    for repo, directories in (("peps", ("", "peps")), ("EIPs", ("", "EIPS")), ("ERCs", ("ERCS",))):
        spec = REPOS[repo]
        for path in (p for d in directories for p in files_at_pin(corpus, repo, d)):
            match = spec.pattern.match(path)
            if match is None or (spec.family, int(match.group(1))) in EXCLUDED:
                continue
            expected.setdefault((spec.family, int(match.group(1))), set()).add((repo, path))
    found = {(corpus.proposals[pid].family, corpus.proposals[pid].number): (v.repo, v.path)
             for pid, v in present(corpus).items()}
    assert set(found) == set(expected)
    assert all(found[key] in paths for key, paths in expected.items())
    ercs = {number for family, number in expected if (family, number) in found
            and found[(family, number)][0] == "ERCs"}
    assert {p.number for p in corpus.proposals.values() if p.series == "ERC"} == ercs


def test_pep_count(corpus):
    peps = [pid for pid in present(corpus) if pid.startswith("PEP-")]
    assert 740 <= len(peps) <= 750


def test_erc_move(corpus):
    subjects = {c.sha: c.message.split("\n", 1)[0] for c in corpus.commits["ERCs"]}
    moved = [
        p.pid
        for p in corpus.proposals.values()
        if p.series == "ERC"
        and subjects[next(v for v in p.versions if v.repo == "ERCs").commit] == "complete erc move"
    ]
    assert len(moved) == 365


def test_inversions_match_git(corpus, report):
    for repo in REPOS:
        out = git(corpus.paths[repo], "log", "--first-parent", "--format=%ct", corpus.pins[repo])
        stamps = [int(line) for line in out.decode("ascii").split()][::-1]  # root first
        independent = sum(1 for i in range(1, len(stamps)) if stamps[i] < stamps[i - 1])
        assert report["inversions"][repo]["count"] == independent, repo
    assert {repo: report["inversions"][repo]["count"] for repo in REPOS} == {
        "peps": 13,
        "EIPs": 1,
        "ERCs": 1,
    }


def test_measured_history_shapes(corpus):
    """The same-commit, rename, symlink and move cases of sections 4 and 6, measured on the
    clones."""
    by_commit = {(p.pid, v.commit[:10]): v for p in corpus.proposals.values() for v in p.versions}
    # A file converted by deleting one and adding another in one commit stays present.
    for pid, sha in (("PEP-505", "791464129f"), ("EIP-1", "7a85a6a9aa"), ("EIP-2", "c83a4aa858")):
        assert by_commit[(pid, sha)].kind == "rename", pid
        assert by_commit[(pid, sha)].status != "absent", pid
    def kind_and_status(pid, sha):
        return by_commit[(pid, sha)].kind, by_commit[(pid, sha)].status

    # The one rename that also changes a status.
    assert kind_and_status("PEP-566", "791464129f") == ("rename", "Final")
    # PEP 495 came back as a symlink, then 19 minutes later as a regular file. (Before that, a
    # file numbered 495 that was really the 3.6 release schedule lived for seven minutes.)
    assert kind_and_status("PEP-495", "135c01161b") == ("create", "absent")
    assert by_commit[("PEP-495", "7acfbeaf9f")].status == "Draft"
    # Two files of EIP 1175 (an ERC since the move) lived side by side for four months.
    assert [(c["proposal"], c["repo"], c["commit"][:10]) for c in corpus.collisions] == [
        ("ERC-1175", "EIPs", "7ceb961c3e")
    ]
    # No ERC timeline shows the move's stubs or markers.
    for p in corpus.proposals.values():
        if p.series == "ERC":
            assert all(v.status not in ("Moved", "missing") for v in p.versions if v.repo == "EIPs")


def test_intervals_partition_time_and_agree_with_status_at(corpus):
    for p in corpus.proposals.values():
        stamps = [v.recorded_ts for v in p.versions]
        assert stamps == sorted(stamps), p.pid
        intervals = p.intervals()
        assert intervals[0].start_ts == stamps[0], p.pid
        for i, nxt in zip(intervals, intervals[1:] + [None], strict=True):
            assert i.end_ts == (None if nxt is None else nxt.start_ts), p.pid
            assert i.end_ts is None or i.start_ts < i.end_ts, p.pid
            expected = None if i.status == "absent" else i.status
            assert p.status_at(i.start_ts) == expected, p.pid
            if i.end_ts is not None:
                assert p.status_at(i.end_ts - 1) == expected, p.pid
            assert not (i.is_transition and i.status in ("absent", "missing")), p.pid


def test_load_time(loaded):
    assert loaded[1] < 300


def legacy_records(corpus) -> dict[str, dict[int, list[int]]]:
    """For each recorded PEP value that section 5's legacy table maps (case-folded, as the table
    reads it), the PEPs whose versions record it, each with the years of those versions."""
    found: dict[str, dict[int, list[int]]] = {}
    for p in corpus.proposals.values():
        if p.family != "pep":
            continue
        for v in p.versions:
            key = (v.raw_status or "").strip().casefold()
            if key in LEGACY:
                years = found.setdefault(key, {}).setdefault(p.number, [])
                years.append(time.gmtime(v.recorded_ts).tm_year)
    return found


# Section 5's legacy PEP values, as the reasons of its table state their history: the PEPs whose
# versions record each value and the years of those versions, first to last.
LEGACY = {"incomplete", "draught", "approved", "finished", "complete", "postponed", "withdrawal"}
LEGACY_HISTORY = {
    "approved": {311: (2003, 2003), 411: (2012, 2012)},
    "complete": {160: (2000, 2000)},
    "finished": {160: (2001, 2006), 366: (2007, 2007)},
    "postponed": {431: (2015, 2015)},
    "withdrawal": {3150: (2012, 2012)},
}
# The table's reasons, in its own words, for the facts above and those of Incomplete below.
LEGACY_REASONS = (
    "| Incomplete | Draft | Recorded by 20 PEPs, PEP 1's own first header among them, for PEPs "
    "whose text was not finished, so no decision had been taken: first in 2000, last given to a "
    "new PEP in 2003 (PEP 320), and held by PEPs 205 and 226 until 2007 |",
    "| Draught | Draft | A misspelling the table maps; no PEP file records it in any version |",
    "| Approved | Accepted | A variant, recorded in one version each of PEP 311 (2003) and PEP 411 "
    "(2012) |",
    "| Finished, Complete | Final | Variants, recorded by PEP 160 (Complete in 2000, Finished from "
    "2001) and PEP 366 (Finished, in 2007) |",
    "| Postponed | Deferred | A variant, recorded in one version of PEP 431 (2015) |",
    "| Withdrawal | Withdrawn | A misspelling, recorded in one version of PEP 3150 (2012) |",
)


def test_section_5_states_the_history_of_the_legacy_pep_values(corpus):
    """Section 5's reasons for the PEP legacy table, as the pinned history records each value:
    Incomplete in 20 PEPs from 2000, PEP 1's first header among them, last given to a new PEP in
    2003 (PEP 320) and given up last by PEPs 205 and 226, in 2007; Draught in no version; the other
    values in the PEPs and years the table names. The table's rows state exactly these facts."""
    found = legacy_records(corpus)
    assert "draught" not in found
    assert {key: {n: (min(y), max(y)) for n, y in peps.items()} for key, peps in found.items()
            if key != "incomplete"} == LEGACY_HISTORY
    incomplete = found["incomplete"]
    assert len(incomplete) == 20 and min(min(y) for y in incomplete.values()) == 2000
    assert corpus.proposals["PEP-1"].versions[0].raw_status == "Incomplete"
    created = {n: corpus.proposals[f"PEP-{n}"].versions[0] for n in incomplete}
    newest = max((v.recorded_ts, n) for n, v in created.items()
                 if (v.raw_status or "").strip().casefold() == "incomplete")
    assert (newest[1], time.gmtime(newest[0]).tm_year) == (320, 2003)

    def given_up(n: int) -> int:  # the recorded time of the version that replaced Incomplete
        versions = corpus.proposals[f"PEP-{n}"].versions
        last = max(i for i, v in enumerate(versions)
                   if (v.raw_status or "").strip().casefold() == "incomplete")
        return versions[last + 1].recorded_ts

    latest = sorted(incomplete, key=given_up)[-2:]
    assert (sorted(latest), {time.gmtime(given_up(n)).tm_year for n in latest}) == ([205, 226],
                                                                                 {2007})
    assert all(given_up(n) < given_up(latest[0]) for n in incomplete if n not in latest)
    registration = (Path(__file__).resolve().parents[1] / "PREREGISTRATION.md").read_text(
        encoding="utf-8")
    table = registration.split("\n## 5. Status normalisation\n", 1)[1].split("\n## 6. ", 1)[0]
    assert [row for row in LEGACY_REASONS if row not in table.splitlines()] == []
