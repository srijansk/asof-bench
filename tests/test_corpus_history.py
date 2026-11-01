"""Sections 3 and 6's numbers about the repositories' histories, from the pinned clones with
read-only git (corpus-marked; set ASOFBENCH_CLONES to run them).

Nothing here reads the generator's walker: the first-parent chains and the pull merges are read
by tests/githistory.py, the ERC move with ``git log``, ``git ls-tree`` and ``git cat-file``, the
pins from data/v0.1/MANIFEST.json, and the items and timelines from data/v0.1.
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from githistory import DAY, FILES, git, git_env, utc_day
from githistory import chain as chain_to
from githistory import pull_merges as pull_merges_to

pytestmark = pytest.mark.corpus

DATA = Path(__file__).resolve().parents[1] / "data" / "v0.1"
PINS = json.loads((DATA / "MANIFEST.json").read_text(encoding="utf-8"))["pins"]
CORPUS = {"peps": "pep", "EIPs": "eth", "ERCs": "eth"}


def chain(clones: Path, repo: str) -> list[dict]:
    """``repo``'s first-parent chain up to data/v0.1's pin."""
    return chain_to(clones, repo, PINS[repo])


def pull_merges(clones: Path, repo: str) -> list[dict]:
    """``repo``'s pull merges up to data/v0.1's pin."""
    return pull_merges_to(clones, repo, PINS[repo])


@pytest.fixture(scope="module")
def merges(clones) -> dict[str, list[dict]]:
    return {repo: pull_merges(clones, repo) for repo in PINS}


def read_jsonl(name: str) -> list[dict]:
    with (DATA / name).open(encoding="utf-8", newline="\n") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_the_pep_chain_has_10959_commits_13_of_them_dated_before_their_parent(clones):
    """Section 3: 13 of the 10,959 first-parent commits up to the PEP repository's pin carry a
    committer date earlier than their parent's, one of them by 3,911 days, about eleven years."""
    assert int(git(clones, "peps", "rev-list", "--first-parent", "--count", PINS["peps"])) == 10959
    commits = chain(clones, "peps")
    gaps = [(a["committed"] - b["committed"]) / DAY
            for a, b in zip(commits, commits[1:], strict=False) if b["committed"] < a["committed"]]
    assert (len(commits), len(gaps), int(max(gaps))) == (10959, 13, 3911)
    assert round(max(gaps) / 365.25) == 11


def test_the_pull_merges_of_python_peps(clones, merges):
    """Section 3: 14 first-parent merges named "Merge branch 'master' of github.com:python/peps",
    committed from 2016-06-26 to 2018-08-29 (2016-06-27 to 2018-08-29 in UTC), hide changes to
    130 PEPs; the largest, 7914641, dates changes made from 2018-02-05 at 2018-07-20, 164 days
    after the first of them, so PEP 572 first appears as Accepted."""
    peps = merges["peps"]
    named = [c for c in chain(clones, "peps")
             if len(c["parents"]) == 2 and c["subject"].startswith("Merge branch 'master' of")]
    assert len(peps) == len(named) == 14  # no other merge of that name
    assert (peps[0]["local"], peps[-1]["local"]) == ("2016-06-26", "2018-08-29")
    assert (peps[0]["end"], peps[-1]["end"]) == (date(2016, 6, 27), date(2018, 8, 29))
    assert len(set().union(*(m["numbers"] for m in peps))) == 130
    largest = max(peps, key=lambda m: m["recorded"] - m["earliest"])
    assert largest["sha"].startswith("7914641")
    assert (largest["start"], largest["end"]) == (date(2018, 2, 5), date(2018, 7, 20))
    assert int((largest["recorded"] - largest["earliest"]) / DAY) == 164
    assert 572 in largest["numbers"]
    (pep572,) = [t for t in read_jsonl("timelines.jsonl") if t["pid"] == "PEP-572"]
    first = pep572["intervals"][0]
    # Its first interval starts at the merge's committer time, as git gives it.
    merged = datetime.fromtimestamp(largest["committed"], UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert (first["status"], first["start"]) == ("Accepted", merged) == (
        "Accepted", "2018-07-20T02:36:00Z")
    assert "Draft" not in {interval["status"] for interval in pep572["intervals"]}


def test_the_pull_merges_of_the_ethereum_repositories(clones, merges):
    """Section 3: the EIP repository has 4 such merges, each hiding at most two days, and the ERC
    repository, which shares that early history, has the same 4, which touch no proposal file."""
    eips, ercs = merges["EIPs"], merges["ERCs"]
    for repo in ("EIPs", "ERCs"):  # no other first-parent merge of that name
        named = [c["sha"] for c in chain(clones, repo) if len(c["parents"]) == 2
                 and c["subject"].startswith("Merge branch 'master' of")]
        assert named == [m["sha"] for m in merges[repo]], repo
    assert len(eips) == 4 and [m["sha"] for m in ercs] == [m["sha"] for m in eips]
    assert sum(m["subject"].endswith("https://github.com/ethereum/EIPs") for m in eips) == 2
    assert all(m["recorded"] - m["earliest"] <= 2 * DAY for m in eips)
    assert all(m["numbers"] for m in eips) and not any(m["numbers"] for m in ercs)


def test_no_gold_value_depends_on_the_pull_merges(merges):
    """Section 3: no item is dated inside a span that hides a change to its own proposal, and no
    F4 or F5 date falls inside any such span, of either corpus. One item,
    F1-change-PEP-498-2016-06-23, is dated inside a span, one of its own corpus (2016-06-19 to
    2016-06-27) that hides 13 other PEPs' changes."""
    spans = [(CORPUS[repo], m) for repo, found in merges.items() for m in found]
    inside, own = [], []
    for q in read_jsonl("questions.jsonl"):
        day = date.fromisoformat(q["as_of"])
        number = None if q["proposal"] is None else int(q["proposal"].split("-")[1])
        for corpus, m in spans:
            if m["start"] <= day <= m["end"]:
                inside.append((q["id"], corpus, m["sha"][:7], m["start"], m["end"],
                               len(m["numbers"])))
                if corpus == q["corpus"] and number in m["numbers"]:
                    own.append(q["id"])
    assert own == []
    assert not any(item_id.startswith(("F4-", "F5-")) for item_id, *_ in inside)
    assert inside == [
        ("F1-change-PEP-498-2016-06-23", "pep", "b409f87", date(2016, 6, 19), date(2016, 6, 27),
         13),
    ]


def blobs(clones: Path, repo: str, specs: list[str]) -> list[str]:
    """The texts of ``<commit>:<path>`` specs, read in one ``git cat-file --batch``."""
    raw = subprocess.run(["git", "-C", str(clones / repo), "cat-file", "--batch"],
                         input="".join(f"{s}\n" for s in specs).encode(), capture_output=True,
                         check=True, env=git_env()).stdout
    texts, pos = [], 0
    for _ in specs:
        header_end = raw.index(b"\n", pos)
        size = int(raw[pos:header_end].split()[2])
        texts.append(raw[header_end + 1: header_end + 1 + size].decode("utf-8", errors="replace"))
        pos = header_end + 1 + size + 1
    return texts


def test_the_erc_move_of_section_6(clones):
    """Section 6: the ERC repository created ERCS/erc-N.md for 365 proposals in one commit
    recorded on 2023-10-25; the EIP repository then touched each of those files in three
    commits: four minutes later a stub with no header, later that day a one-line change of the
    stub's link to the file's new path, and on 2023-10-26 the Moved marker. All 624 ERC files
    that ever existed up to the pin exist at it."""
    out = git(clones, "ERCs", "log", "--first-parent", "--format=%x1e%H%x00%ct", "--name-status",
              "--diff-filter=AR", PINS["ERCs"], "--", "ERCS/")
    brought = {}
    for block in out.split("\x1e")[1:]:
        head, _, rest = block.partition("\n")
        sha, committed = head.split("\x00")
        numbers = {int(m.group(1)) for line in rest.splitlines()
                   if (m := FILES["ERCs"].match(line.split("\t")[-1]))}
        if numbers:
            brought[sha] = (int(committed), numbers)
    ((moved_at, moved),) = [v for v in brought.values() if len(v[1]) > 100]  # one such commit
    assert (len(moved), utc_day(moved_at)) == (365, date(2023, 10, 25))
    at_pin = {int(m.group(1)) for line in git(clones, "ERCs", "ls-tree", "--name-only",
                                               PINS["ERCs"], "ERCS/").splitlines()
              if (m := FILES["ERCs"].match(line))}
    ever = {int(m.group(1)) for line in git(clones, "ERCs", "log", "--first-parent", "--format=",
                                            "--name-only", PINS["ERCs"], "--",
                                            "ERCS/").splitlines()
            if (m := FILES["ERCs"].match(line))}
    assert len(at_pin) == 624 and ever == at_pin
    # The EIP repository's first-parent commits that change exactly the files of those 365.
    out = git(clones, "EIPs", "log", "--first-parent", "--format=%x1e%H%x00%ct", "--numstat",
              PINS["EIPs"], "--", "EIPS/")
    touching = []
    for block in out.split("\x1e")[1:]:
        head, _, rest = block.partition("\n")
        sha, committed = head.split("\x00")
        counts = {}
        for line in rest.splitlines():
            parts = line.split("\t")
            if len(parts) == 3 and (m := re.fullmatch(r"EIPS/eip-(\d+)\.md", parts[2])):
                counts[int(m.group(1))] = (parts[0], parts[1])
        if set(counts) == moved:
            touching.append((int(committed), sha, counts))
    stub, link, marker = sorted(touching)
    assert (utc_day(stub[0]), round((stub[0] - moved_at) / 60)) == (date(2023, 10, 25), 4)
    assert all(added == "1" for added, _ in stub[2].values())
    assert utc_day(link[0]) == date(2023, 10, 25) and link[0] > stub[0]
    assert all(counts == ("1", "1") for counts in link[2].values())  # the 365 one-line changes
    assert utc_day(marker[0]) == date(2023, 10, 26)
    order = sorted(moved)
    stubs = blobs(clones, "EIPs", [f"{stub[1]}:EIPS/eip-{n}.md" for n in order])
    linked = blobs(clones, "EIPs", [f"{link[1]}:EIPS/eip-{n}.md" for n in order])
    marked = blobs(clones, "EIPs", [f"{marker[1]}:EIPS/eip-{n}.md" for n in order])
    for n, before, after, final in zip(order, stubs, linked, marked, strict=True):
        assert before.count("\n") <= 1 and not re.match(r"---|[A-Za-z-]+:", before), n
        assert f"/ERCS/erc-{n}.md" in after and f"/ERCS/erc-{n}.md" not in before, n
        assert re.match(r"---\n(?:.*\n)*?status: Moved\n(?:.*\n)*?---\n", final), n
