"""The context builders of long context and the oracle (systems/harness.py; PREREGISTRATION.md
section 11): what each puts in a request's context and from which stream, on the toy world with
its three streams registered in the manifest, and on synthetic streams for what the toy world
lacks (a deleted proposal, a header without a status, a proposal of 1,600,000 characters); and the
harness's reading of the streams (--streams-dir) and the order of the requests file. It needs git,
which builds the toy world's repositories, and the asofbench package, so the reader image skips
it; it makes no model call and reads no v0.1 file."""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import re
import shutil
import uuid
from datetime import date
from pathlib import Path

import pytest

pytest.importorskip("asofbench")
if shutil.which("git") is None:
    pytest.skip("the toy world's repositories need git", allow_module_level=True)

import harness  # noqa: E402 - systems/ is on the path (conftest.py)
from asofbench_systems.stream import render_episode  # noqa: E402
from reader import answer_requests  # noqa: E402

from asofbench import cli  # noqa: E402
from asofbench.stream import STREAM_FILES, unified_diff  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SYSTEMS = ROOT / "systems"
SEPARATOR = "\n---\n"
# The two context descriptions, the first line of a request's context, written out again here so
# that an edit of the code's strings fails these tests.
LONG_CONTEXT = ("The context holds every recorded change to the file of the proposal the question "
                "names, oldest first: its full text when the file was created, then a unified diff "
                "for each later change, each with the time it was recorded and its commit message.")
ORACLE = ("The context lists every proposal of the question's corpus present at the end of the "
          "question's date, with its status and its requires, replaces and superseded-by lists as "
          "its file's header recorded them then.")


def jsonl(path: Path) -> list[dict]:
    """The JSON lines of a file, split at newlines only, as the reader splits a requests file."""
    return [json.loads(line) for line in path.read_bytes().decode("utf-8").split("\n") if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> Path:
    """The toy world's repositories, built into questions and the three arms' streams by `asofbench
    build --streams`, so that the manifest registers the streams: data/ and streams/."""
    spec = importlib.util.spec_from_file_location("toy_world", ROOT / "examples" / "toy_world.py")
    toy_world = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(toy_world)
    root = tmp_path_factory.mktemp("toy-streams") / "world"
    clones = toy_world.toy_clones(root / "clones")
    assert cli.main(["build", "--clones", str(clones), "--out", str(root / "data"),
                     "--streams", str(root / "streams")]) == 0
    return root


@pytest.fixture(scope="module")
def questions(world) -> dict[str, dict]:
    return {q["id"]: q for q in jsonl(world / "data" / "questions.jsonl")}


@pytest.fixture(scope="module")
def runs(world, tmp_path_factory) -> dict[str, Path]:
    """One run directory for long context and one for the oracle, each for the toy world's items
    in every arm they are asked, three times; the tests read them and change nothing."""
    root = tmp_path_factory.mktemp("runs")
    for system in ("long-context", "oracle"):
        assert run_requests(world, root / system, system) == 0
    return {system: root / system for system in ("long-context", "oracle")}


def run_requests(world: Path, out: Path, system: str, *extra: str) -> int:
    streams = [] if system == "closed-book" else ["--streams-dir", str(world / "streams")]
    return harness.main(["requests", "--system", system, "--data", str(world / "data"),
                         "--out-dir", str(out), *streams, *extra])


def written(run_dir: Path) -> list[tuple[dict, dict]]:
    """Each request of a run, in the file's order, with the key line (the item) that stands for
    it."""
    keys = {line["key"]: line for line in jsonl(run_dir / "keys.jsonl")}
    requests = jsonl(run_dir / "input" / "requests.jsonl")
    return [(request, keys[request["key"]]) for request in requests]


def statements_of(context: str) -> list[str]:
    """The lines that follow a context's first line and the blank line after it."""
    description, _, body = context.partition("\n\n")
    assert description in (LONG_CONTEXT, ORACLE)
    return body.split("\n") if body else []


def contexts_of(pairs, item_id: str, arm: str) -> set[str]:
    """The contexts of one item in one arm, over its runs."""
    return {request["context"] for request, key in pairs
            if (key["id"], key["arm"]) == (item_id, arm)}


# --- the toy world: long context -----------------------------------------------------------------


def test_long_context_holds_every_episode_of_the_named_proposal_in_stream_order(world, questions,
                                                                                runs):
    pairs = written(runs["long-context"])
    streams = {arm: jsonl(world / "streams" / name) for arm, name in STREAM_FILES.items()}
    assert pairs
    for request, key in pairs:
        proposal = questions[key["id"]]["proposal"]
        episodes = [e for e in streams[key["arm"]] if e["proposal"] == proposal]
        assert request["context"] == (LONG_CONTEXT + "\n\n"
                                      + SEPARATOR.join(render_episode(e) for e in episodes))
        # No other proposal's episode: its commit is in none of the context's pieces.
        for other in streams[key["arm"]]:
            if other["proposal"] != proposal:
                assert other["commit"] not in request["context"]


def test_those_recorded_after_d_are_in_the_context_and_each_episode_follows_a_separator_line(
        world, runs):
    # D is 2017-01-25: the proposal's Final status (2017-09-11) and its move into the ERC
    # repository (2023-10-25) are both recorded after it, and both are in the context.
    pairs = written(runs["long-context"])
    (context,) = contexts_of(pairs, "F1-change-ERC-20-2017-01-25", "H+")
    head = re.compile(r'^ERC-20 \("Proposal 20"\): (\w+) recorded at (\S+) in the (\w+) '
                      r"repository, commit [0-9a-f]{40}\.$", re.M)
    assert [found.groups() for found in head.finditer(context)] == [
        ("create", "2015-11-19T12:00:00Z", "EIPs"), ("modify", "2017-09-11T12:00:00Z", "EIPs"),
        ("rename", "2023-10-25T15:11:00Z", "ERCs")]
    # Each episode after the first begins on the line after a line that holds --- alone. (Ethereum
    # front matter holds such lines too, so the episodes are told apart by their first lines.)
    assert context.count(SEPARATOR + 'ERC-20 ("Proposal 20"): ') == 2
    assert context.startswith(LONG_CONTEXT + '\n\nERC-20 ("Proposal 20"): create recorded at ')
    assert "\n+status: Final\n" in context
    assert context.endswith("Unified diff of the file, which moved here from another repository "
                            "or path:\n(empty: the change touched only lines this arm withholds, "
                            "or the file's mode)")
    # None of the other proposals' episodes: the toy world's other proposals are named by refs.
    for other in ("EIP-1", "EIP-2", "EIP-3", "PEP 1", "PEP 2"):
        assert f'{other} ("' not in context


def test_a_proposal_with_one_episode_has_that_episode_alone_after_the_description(world, runs):
    pairs = written(runs["long-context"])
    (episode,) = [e for e in jsonl(world / "streams" / "h-plus.jsonl") if e["proposal"] == "EIP-3"]
    assert contexts_of(pairs, "F1-current-EIP-3-2026-10-27", "H+") == {
        LONG_CONTEXT + "\n\n" + render_episode(episode)}
    assert render_episode(episode).endswith("Full text of the file:\n---\neip: 3\n"
                                            "title: Proposal 3\nstatus: Draft\nrequires: 2\n"
                                            "---\nPlain text.\n")


def test_each_item_reads_the_stream_of_its_own_arm(world, runs):
    pairs = written(runs["long-context"])
    (plus,) = contexts_of(pairs, "F1-current-EIP-3-2026-10-27", "H+")
    (minus,) = contexts_of(pairs, "F1-current-EIP-3-2026-10-27", "H-")
    assert "status: Draft" in plus and "status" not in minus.split("Full text of the file:")[1]
    # A counterfactual item reads the counterfactual stream: EIP-2's status, edited to Withdrawn.
    (edited,) = contexts_of(pairs, "CF-counterfactual-EIP-2-2017-01-19", "CF")
    assert "\n+status: Withdrawn\n" in edited and "Final" not in edited
    assert "Update status of EIP-2" in edited


def test_every_run_of_an_item_has_the_same_context(runs):
    for system in ("long-context", "oracle"):
        pairs = written(runs[system])
        by_item: dict[tuple[str, str], set[str]] = {}
        for request, key in pairs:
            by_item.setdefault((key["id"], key["arm"]), set()).add(request["context"])
        assert all(len(contexts) == 1 for contexts in by_item.values())


def test_long_context_is_never_asked_an_f3_f4_or_f5_item(world, questions, runs):
    pairs = written(runs["long-context"])
    families = {questions[key["id"]]["family"] for _, key in pairs}
    assert families == {"F1", "CF"}
    # The toy world has F4 and F5 items, so the rule left some out.
    assert {"F4", "F5"} <= {q["family"] for q in questions.values()}
    asked = {(key["id"], key["arm"]) for _, key in pairs}
    assert asked == {(q["id"], arm) for q in questions.values() if q["family"] in ("F1", "F2", "CF")
                     for arm in q["arms"]}


# --- the toy world: the oracle -------------------------------------------------------------------


def lines_for(pairs, item_id: str, arm: str) -> list[str]:
    (context,) = contexts_of(pairs, item_id, arm)
    return statements_of(context)


def test_the_oracle_lists_the_proposals_present_at_the_end_of_the_date_as_the_headers_record_them(
        world, runs):
    pairs = written(runs["oracle"])
    assert lines_for(pairs, "F1-change-PEP-2-2002-06-20", "H+") == [
        "PEP-1: status Draft; requires -; replaces -; superseded-by -",
        "PEP-2: status Draft; requires -; replaces -; superseded-by -"]
    # ERC-20 is listed under its id at S, in ascending number: after EIP-3. On 2017-01-25 EIP-2 is
    # Final (2017-01-01) and ERC-20 is still Draft (it is Final from 2017-09-11).
    assert lines_for(pairs, "F1-change-ERC-20-2017-01-25", "H+") == [
        "EIP-1: status Draft; requires -; replaces -; superseded-by -",
        "EIP-2: status Final; requires 1; replaces -; superseded-by -",
        "EIP-3: status Draft; requires 2; replaces -; superseded-by -",
        "ERC-20: status Draft; requires -; replaces -; superseded-by -"]
    # The F5 item's date, 2018-01-01, holds the change recorded at noon that day.
    assert lines_for(pairs, "F5-count-eth-Final-2018-01-01", "H+") == [
        "EIP-1: status Living; requires -; replaces -; superseded-by -",
        "EIP-2: status Final; requires 1; replaces -; superseded-by -",
        "EIP-3: status Draft; requires 2; replaces -; superseded-by -",
        "ERC-20: status Final; requires -; replaces -; superseded-by -"]
    # The toy world has no item on a date before a proposal exists, so ask the builder directly:
    # on 2001-06-01 PEP-2 is not created yet, and on 2015-12-01 only ERC-20 exists.
    plus = world / "streams" / "h-plus.jsonl"
    assert statements_of(harness.oracle_statements(plus, "pep", date(2001, 6, 1))) == [
        "PEP-1: status Draft; requires -; replaces -; superseded-by -"]
    assert statements_of(harness.oracle_statements(plus, "eth", date(2015, 12, 1))) == [
        "ERC-20: status Draft; requires -; replaces -; superseded-by -"]


def test_an_h_minus_item_gets_the_statements_read_from_the_h_plus_stream(runs):
    pairs = written(runs["oracle"])
    for item_id in ("F1-change-ERC-20-2017-01-25", "F1-change-PEP-2-2002-06-20",
                    "F1-current-EIP-1-2026-10-27"):
        plus, minus = contexts_of(pairs, item_id, "H+"), contexts_of(pairs, item_id, "H-")
        assert plus == minus and len(plus) == 1
    # The H- arm withholds every status, yet its item's context has them.
    assert all(" status -;" not in line
               for line in lines_for(pairs, "F1-change-ERC-20-2017-01-25", "H-"))


def test_a_counterfactual_item_gets_the_statements_as_the_counterfactual_stream_edits_them(runs):
    pairs = written(runs["oracle"])
    # PEP-1 is Final and PEP-2 Rejected as recorded (2005, 2006); the edit swaps Final with
    # Withdrawn and Accepted with Rejected, and edits nothing else.
    assert lines_for(pairs, "CF-counterfactual-PEP-1-2013-09-02", "CF") == [
        "PEP-1: status Withdrawn; requires -; replaces -; superseded-by -",
        "PEP-2: status Accepted; requires -; replaces -; superseded-by -"]
    assert lines_for(pairs, "F1-current-PEP-1-2026-10-27", "H+") == [
        "PEP-1: status Final; requires -; replaces -; superseded-by -",
        "PEP-2: status Rejected; requires -; replaces -; superseded-by -"]
    # On the Ethereum side only the counterfactual proposals' statuses change: EIP-1 and EIP-3 are
    # not among them.
    assert lines_for(pairs, "CF-counterfactual-ERC-20-2025-04-03", "CF") == [
        "EIP-1: status Living; requires -; replaces -; superseded-by -",
        "EIP-2: status Withdrawn; requires 1; replaces -; superseded-by -",
        "EIP-3: status Draft; requires 2; replaces -; superseded-by -",
        "ERC-20: status Withdrawn; requires -; replaces -; superseded-by -"]


def test_the_oracle_is_asked_every_item_in_every_arm_it_lists(runs, questions):
    pairs = written(runs["oracle"])
    assert {(key["id"], key["arm"]) for _, key in pairs} == {
        (q["id"], arm) for q in questions.values() for arm in q["arms"]}
    assert {questions[key["id"]]["family"] for _, key in pairs} == {"F1", "F4", "F5", "CF"}


def parsed(context: str) -> dict[int, tuple[str, str, list[int]]]:
    """A context's statements read back: number to (id, status, requires)."""
    found = {}
    for line in statements_of(context):
        proposal, _, rest = line.partition(": status ")
        status, _, rest = rest.partition("; requires ")
        requires, _, _ = rest.partition("; replaces ")
        found[int(proposal.rpartition("-")[2])] = (
            proposal, status, [] if requires == "-" else [int(n) for n in requires.split(", ")])
    return found


def test_the_oracle_context_holds_what_the_gold_of_every_toy_item_is_computed_from(
        world, questions, runs):
    # The ceiling's meaning: from its context alone, the toy world's items are all answerable.
    # The gold is the generator's own (its timelines, not this harness's replay).
    gold = {g["id"]: g["gold"] for g in jsonl(world / "data" / "gold.jsonl")}
    checked = set()
    for request, key in written(runs["oracle"]):
        item = questions[key["id"]]
        records = parsed(request["context"])
        if item["family"] in ("F1", "CF"):
            # F1 in H- as in H+ (the oracle reads H+), and CF as the edited stream has it.
            number = int(item["proposal"].rpartition("-")[2])
            assert records[number][:2] == (item["proposal"], gold[key["id"]])
        elif item["family"] == "F5":
            status = key["id"].split("-")[3]
            assert sum(record[1] == status for record in records.values()) == gold[key["id"]]
        elif item["family"] == "F4":
            closure, pending = set(), [int(item["proposal"].rpartition("-")[2])]
            while pending:
                for required in records[pending.pop()][2]:
                    if required in records and required not in closure:
                        closure.add(required)
                        pending.append(required)
            assert sorted(n for n in closure if records[n][1] == "Final") == gold[key["id"]]
        checked.add(item["family"])
    assert checked == {"F1", "F4", "F5", "CF"}


# --- synthetic streams ---------------------------------------------------------------------------


class Stream:
    """A synthetic stream of made-up proposals: episodes in recorded-time order, each diff made by
    the generator's own ``unified_diff``, so that ``asofbench.materialize.replay`` applies it."""

    REPOS = {"PEP": "peps", "EIP": "EIPs", "ERC": "ERCs"}

    def __init__(self) -> None:
        self.episodes: list[dict] = []
        self.texts: dict[str, str] = {}
        self.repos: dict[str, str] = {}

    def record(self, when: str, proposal: str, text: str | None, *, repo: str | None = None,
               message: str = "Update") -> dict:
        """The proposal's file as ``text`` from ``when`` on (a deletion for None): a create when it
        has no file, a rename when its repository changes, otherwise a modify."""
        repo = repo or self.repos.get(proposal) or self.REPOS[proposal.split("-")[0]]
        before = self.texts.get(proposal)
        if text is None:
            kind, body = "delete", ""
            del self.texts[proposal]
        elif before is None:
            kind, body = "create", text
        else:
            kind = "modify" if repo == self.repos[proposal] else "rename"
            body = unified_diff(before, text)
        if text is not None:
            self.texts[proposal], self.repos[proposal] = text, repo
        commit = hashlib.sha1(f"commit {len(self.episodes)}".encode()).hexdigest()
        number = proposal.split("-")[1]
        ref = f'{proposal.replace("-", " ")} ("Made-up {number}")'
        episode = {"episode_id": f"{commit[:12]}:{proposal}",
                   "corpus": "pep" if repo == "peps" else "eth", "repo": repo,
                   "proposal": proposal, "ref": ref, "recorded_at": when, "commit": commit,
                   "message": message, "kind": kind, "text": body}
        self.episodes.append(episode)
        return episode

    def write(self, path: Path) -> Path:
        path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in self.episodes),
                        encoding="utf-8")
        return path


def pep_text(number: int, *header: str, body: str = "Made-up body.\n") -> str:
    return "\n".join([f"PEP: {number}", f"Title: Made-up {number}", *header]) + f"\n\n{body}"


def eip_text(number: int, *header: str, body: str = "Made-up body.\n") -> str:
    return "\n".join(["---", f"eip: {number}", f"title: Made-up {number}", *header, "---"]) + (
        f"\n{body}")


def proposals_of(context: str) -> list[str]:
    return [line.split(":", 1)[0] for line in statements_of(context)]


def test_a_proposal_deleted_or_not_yet_created_at_the_date_is_absent(tmp_path):
    stream = Stream()
    stream.record("2001-01-01T12:00:00Z", "PEP-9", pep_text(9, "Status: Draft"))
    stream.record("2001-02-01T12:00:00Z", "PEP-100", pep_text(100, "Status: Draft"))
    stream.record("2001-03-01T12:00:00Z", "PEP-10", pep_text(10, "Status: Draft"))
    stream.record("2001-04-01T12:00:00Z", "PEP-11", pep_text(11, "Status: Draft"))
    stream.record("2001-05-01T12:00:00Z", "PEP-11", None)
    stream.record("2001-06-01T12:00:00Z", "PEP-12", pep_text(12, "Status: Draft"))
    stream.record("2001-12-31T23:59:59Z", "PEP-13", pep_text(13, "Status: Draft"))
    stream.record("2002-01-01T00:00:00Z", "PEP-14", pep_text(14, "Status: Draft"))
    stream.record("2002-02-01T12:00:00Z", "PEP-12", None)
    path = stream.write(tmp_path / "h-plus.jsonl")
    # In ascending number (PEP-100 last, though "PEP-100" sorts before "PEP-11" as text); PEP-11 was
    # deleted before the date; PEP-13 is recorded in the date's last second; PEP-14 is created a
    # second after it.
    assert proposals_of(harness.oracle_statements(path, "pep", date(2001, 12, 31))) == [
        "PEP-9", "PEP-10", "PEP-12", "PEP-13", "PEP-100"]
    # A day later PEP-14 is there, and a day after the deletion PEP-12 is gone.
    assert proposals_of(harness.oracle_statements(path, "pep", date(2002, 1, 1))) == [
        "PEP-9", "PEP-10", "PEP-12", "PEP-13", "PEP-14", "PEP-100"]
    assert proposals_of(harness.oracle_statements(path, "pep", date(2002, 2, 1))) == [
        "PEP-9", "PEP-10", "PEP-13", "PEP-14", "PEP-100"]
    # Before the first episode no proposal is listed: the description and the blank line remain.
    assert harness.oracle_statements(path, "pep", date(2000, 12, 31)) == ORACLE + "\n\n"
    # A proposal deleted and created again is listed from its second creation on.
    again = Stream()
    again.record("2001-01-01T12:00:00Z", "PEP-5", pep_text(5, "Status: Draft"))
    again.record("2001-02-01T12:00:00Z", "PEP-5", None)
    again.record("2001-03-01T12:00:00Z", "PEP-5", pep_text(5, "Status: Final"))
    path = again.write(tmp_path / "again.jsonl")
    assert [proposals_of(harness.oracle_statements(path, "pep", date(2001, month, 15)))
            for month in (1, 2, 3)] == [["PEP-5"], [], ["PEP-5"]]


def test_each_statement_is_the_header_as_the_file_recorded_it(tmp_path):
    stream = Stream()
    stream.record("2001-01-01T12:00:00Z", "PEP-1", pep_text(
        1, "Status: Final", "Requires: 3, 7, 3", "Replaces: 4 5", "Superseded-By: 9, 10"))
    # The pre-2011 name of Superseded-By, and a legacy status: both as recorded, not normalised.
    stream.record("2001-01-01T12:01:00Z", "PEP-2", pep_text(
        2, "Status: Incomplete", "Replaced-By: 8"))
    # No Status line, and a qualified Requires that is not the requires field.
    stream.record("2001-01-01T12:02:00Z", "PEP-3", pep_text(3, "Requires (*optional): 1"))
    stream.record("2001-01-01T12:03:00Z", "PEP-4", pep_text(4, "Status:"))
    # Ethereum: the lists are "-" unless the header has them.
    stream.record("2001-01-01T12:04:00Z", "EIP-1559", eip_text(
        1559, "status: Last Call", "requires: 1, 2718"))
    stream.record("2001-01-01T12:05:00Z", "EIP-5", eip_text(
        5, "status: Draft", "replaces: 4", "superseded-by: 6"))
    stream.record("2001-01-01T12:06:00Z", "EIP-6", "Not a header at all.\n")
    path = stream.write(tmp_path / "h-plus.jsonl")
    assert statements_of(harness.oracle_statements(path, "pep", date(2001, 1, 1))) == [
        "PEP-1: status Final; requires 3, 7; replaces 4, 5; superseded-by 9, 10",
        "PEP-2: status Incomplete; requires -; replaces -; superseded-by 8",
        "PEP-3: status -; requires -; replaces -; superseded-by -",
        "PEP-4: status -; requires -; replaces -; superseded-by -"]
    assert statements_of(harness.oracle_statements(path, "eth", date(2001, 1, 1))) == [
        "EIP-5: status Draft; requires -; replaces 4; superseded-by 6",
        "EIP-6: status -; requires -; replaces -; superseded-by -",
        "EIP-1559: status Last Call; requires 1, 2718; replaces -; superseded-by -"]


def test_the_oracle_reads_the_header_in_effect_at_the_date_and_lists_only_its_own_corpus(
        tmp_path):
    stream = Stream()
    stream.record("2001-01-01T12:00:00Z", "PEP-1", pep_text(1, "Status: Draft"))
    stream.record("2001-03-01T12:00:00Z", "ERC-20", eip_text(20, "status: Draft"), repo="EIPs")
    stream.record("2001-06-01T12:00:00Z", "PEP-1", pep_text(1, "Status: Final", "Requires: 2"))
    stream.record("2001-09-01T12:00:00Z", "ERC-20", eip_text(20, "status: Final"), repo="ERCs")
    path = stream.write(tmp_path / "h-plus.jsonl")
    assert statements_of(harness.oracle_statements(path, "pep", date(2001, 5, 31))) == [
        "PEP-1: status Draft; requires -; replaces -; superseded-by -"]
    assert statements_of(harness.oracle_statements(path, "pep", date(2001, 6, 1))) == [
        "PEP-1: status Final; requires 2; replaces -; superseded-by -"]
    # The Ethereum proposal that moved to the ERC repository is one entry, under its id at S.
    assert statements_of(harness.oracle_statements(path, "eth", date(2001, 8, 31))) == [
        "ERC-20: status Draft; requires -; replaces -; superseded-by -"]
    assert statements_of(harness.oracle_statements(path, "eth", date(2001, 9, 1))) == [
        "ERC-20: status Final; requires -; replaces -; superseded-by -"]


def test_a_stream_that_cannot_be_replayed_is_refused_not_listed(tmp_path):
    stream = Stream()
    stream.record("2001-01-01T12:00:00Z", "PEP-1", pep_text(1, "Status: Draft"))
    stream.record("2001-02-01T12:00:00Z", "PEP-1", pep_text(1, "Status: Final"))
    path = stream.write(tmp_path / "h-plus.jsonl")
    broken = path.read_text(encoding="utf-8").replace("-Status: Draft", "-Status: Other")
    (tmp_path / "broken.jsonl").write_text(broken, encoding="utf-8")
    with pytest.raises(harness.Refused, match=r"broken\.jsonl line 2"):
        harness.oracle_statements(tmp_path / "broken.jsonl", "pep", date(2001, 3, 1))
    (tmp_path / "text.jsonl").write_text("not json\n", encoding="utf-8")
    with pytest.raises(harness.Refused, match=r"text\.jsonl line 1: not JSON"):
        harness.oracle_statements(tmp_path / "text.jsonl", "pep", date(2001, 3, 1))
    (tmp_path / "bytes.jsonl").write_bytes(b'{"a": "\xff"}\n')
    with pytest.raises(harness.Refused, match="not UTF-8"):
        harness.oracle_statements(tmp_path / "bytes.jsonl", "pep", date(2001, 3, 1))
    with pytest.raises(harness.Refused, match="cannot be read"):
        harness.oracle_statements(tmp_path / "missing.jsonl", "pep", date(2001, 3, 1))
    # An episode that cannot be rendered (a kind that no template knows) is refused as well.
    odd = [{**stream.episodes[0], "kind": "copy"}]
    (tmp_path / "odd.jsonl").write_text(json.dumps(odd[0]) + "\n", encoding="utf-8")
    with pytest.raises(harness.Refused, match="odd.jsonl"):
        harness.long_context(tmp_path / "odd.jsonl", "PEP-1")


def test_one_replay_gives_every_corpus_and_date_asked_and_the_same_text_as_one_call_each(tmp_path):
    stream = Stream()
    stream.record("2001-01-01T12:00:00Z", "PEP-1", pep_text(1, "Status: Draft"))
    stream.record("2001-02-01T12:00:00Z", "EIP-2", eip_text(2, "status: Draft"))
    stream.record("2001-03-01T12:00:00Z", "PEP-1", pep_text(1, "Status: Final"))
    path = stream.write(tmp_path / "h-plus.jsonl")
    wanted = {(corpus, date(2001, month, 15)) for corpus in ("pep", "eth") for month in (1, 2, 3)}
    together = harness.oracle_contexts(path, wanted)
    assert set(together) == wanted
    for corpus, day in wanted:
        assert together[corpus, day] == harness.oracle_statements(path, corpus, day)
    assert harness.oracle_contexts(path, set()) == {}


def test_long_context_is_the_proposal_s_episodes_in_stream_order_and_no_other(tmp_path):
    stream = Stream()
    first = stream.record("2001-01-01T12:00:00Z", "PEP-1", pep_text(1, "Status: Draft"),
                          message="Add PEP 1\n\nWith a second paragraph.")
    stream.record("2001-02-01T12:00:00Z", "PEP-2", pep_text(2, "Status: Draft"))
    second = stream.record("2001-03-01T12:00:00Z", "PEP-1", pep_text(1, "Status: Final"))
    stream.record("2001-04-01T12:00:00Z", "PEP-2", pep_text(2, "Status: Final"))
    third = stream.record("2001-05-01T12:00:00Z", "PEP-1", None)
    fourth = stream.record("2001-06-01T12:00:00Z", "PEP-1", pep_text(1, "Status: Draft"))
    path = stream.write(tmp_path / "h-plus.jsonl")
    context = harness.long_context(path, "PEP-1")
    assert context == (LONG_CONTEXT + "\n\n" + SEPARATOR.join(
        render_episode(e) for e in (first, second, third, fourth)))
    assert third["kind"] == "delete" and "The file was deleted." in context
    assert "With a second paragraph." in context
    assert all(e["commit"] not in context for e in stream.episodes if e["proposal"] == "PEP-2")
    assert harness.long_contexts(path, ["PEP-1", "PEP-2"]) == {
        "PEP-1": context, "PEP-2": harness.long_context(path, "PEP-2")}
    with pytest.raises(harness.Refused, match="holds no episode of PEP-3"):
        harness.long_context(path, "PEP-3")


def test_a_proposal_of_1_600_000_characters_is_sent_whole(tmp_path):
    head = pep_text(5, "Status: Draft", body="")
    filler = "".join(f"line {n:07d} of a made-up proposal\n" for n in range(50_000))
    text = head + filler[:1_600_000 - len(head)]
    assert len(text) == 1_600_000
    stream = Stream()
    created = stream.record("2001-01-01T12:00:00Z", "PEP-5", text)
    changed = stream.record("2001-06-01T12:00:00Z", "PEP-5", text.replace("Draft", "Final", 1))
    stream.record("2001-07-01T12:00:00Z", "PEP-6", pep_text(6, "Status: Draft"))
    questions = [{"id": "F1-change-PEP-5-2001-03-01", "family": "F1", "stratum": "change",
                  "corpus": "pep", "proposal": "PEP-5", "as_of": "2001-03-01",
                  "arms": ["H+", "H-"], "question": "A made-up question?",
                  "answer_type": "status", "choices": None}]
    data, streams = build_data(tmp_path / "world", questions, {"H+": stream, "H-": stream})
    run_dir = tmp_path / "run"
    assert harness.main(["requests", "--system", "long-context", "--data", str(data), "--out-dir",
                         str(run_dir), "--streams-dir", str(streams)]) == 0
    pairs = written(run_dir)
    assert len(pairs) == 6  # two arms, three runs
    expected = LONG_CONTEXT + "\n\n" + render_episode(created) + SEPARATOR + render_episode(changed)
    for request, _ in pairs:
        assert request["context"] == expected
        assert text in request["context"] and len(request["context"]) > 1_600_000
        assert answer_requests.request_problem(request) is None
    # The whole line is in the file: the request is not cut at any size.
    sizes = [len(line) for line in (run_dir / "input" / "requests.jsonl").read_bytes().split(b"\n")
             if line]
    assert min(sizes) > 1_600_000


def question(item_id: str, family: str, proposal: str | None, as_of: str, *,
             arms: tuple[str, ...] = ("H+", "H-"), corpus: str = "pep",
             answer_type: str = "status") -> dict:
    return {"id": item_id, "family": family, "stratum": "made-up", "corpus": corpus,
            "proposal": proposal, "as_of": as_of, "arms": list(arms),
            "question": f"A made-up question about {proposal}?", "answer_type": answer_type,
            "choices": None}


def build_data(root: Path, questions: list[dict], streams: dict[str, Stream]) -> tuple[Path, Path]:
    """A build's questions.jsonl, and a MANIFEST.json that registers it and the streams, in
    root/data; the streams, in root/streams under the generator's names."""
    data, directory = root / "data", root / "streams"
    data.mkdir(parents=True)
    directory.mkdir()
    (data / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in questions),
                                          encoding="utf-8")
    registered = {}
    for arm, stream in streams.items():
        path = stream.write(directory / STREAM_FILES[arm])
        chars = sum(len(e["text"]) for e in stream.episodes)
        registered[arm] = {"episodes": len(stream.episodes), "chars": chars,
                           "tokens_est": chars // 4, "sha256": sha256(path)}
    (data / "MANIFEST.json").write_text(json.dumps({
        "files": {"questions.jsonl": {"sha256": sha256(data / "questions.jsonl")}},
        "streams": registered}), encoding="utf-8")
    return data, directory


def test_a_stream_that_the_replay_refuses_stops_the_oracle_before_anything_is_written(tmp_path):
    stream = Stream()
    stream.record("2001-01-01T12:00:00Z", "PEP-1", pep_text(1, "Status: Draft"))
    stream.record("2001-02-01T12:00:00Z", "PEP-1", pep_text(1, "Status: Final"))
    stream.episodes[1]["text"] = stream.episodes[1]["text"].replace("-Status: Draft",
                                                                    "-Status: Other")
    # The replay applies the episodes recorded by the item's date only, so the date follows the
    # broken one.
    questions = [question("F1-change-PEP-1-2001-03-01", "F1", "PEP-1", "2001-03-01")]
    data, streams = build_data(tmp_path / "world", questions, {"H+": stream, "H-": stream})
    assert harness.main(["requests", "--system", "oracle", "--data", str(data), "--out-dir",
                         str(tmp_path / "run"), "--streams-dir", str(streams)]) == 2
    assert not (tmp_path / "run").exists()


def test_an_item_whose_proposal_the_stream_lacks_stops_long_context_before_anything_is_written(
        tmp_path):
    stream = Stream()
    stream.record("2001-01-01T12:00:00Z", "PEP-1", pep_text(1, "Status: Draft"))
    questions = [question("F1-change-PEP-7-2001-01-15", "F1", "PEP-7", "2001-01-15")]
    data, streams = build_data(tmp_path / "world", questions, {"H+": stream, "H-": stream})
    assert harness.main(["requests", "--system", "long-context", "--data", str(data),
                         "--out-dir", str(tmp_path / "run"), "--streams-dir", str(streams)]) == 2
    assert not (tmp_path / "run").exists()


def test_f2_and_f3_items_are_asked_at_s_and_long_context_is_asked_f2_only(tmp_path):
    # The toy world has no F2 or F3 item. Their date is S (the build gives them as_of = S).
    stream = Stream()
    stream.record("2001-01-01T12:00:00Z", "PEP-9", pep_text(9, "Status: Draft"))
    stream.record("2001-02-01T12:00:00Z", "PEP-10", pep_text(10, "Status: Final", "Replaces: 9"))
    stream.record("2001-03-01T12:00:00Z", "PEP-9", pep_text(9, "Status: Superseded",
                                                             "Superseded-By: 10"))
    stream.record("2001-04-01T12:00:00Z", "EIP-1", eip_text(1, "status: Final"))
    stream.record("2001-05-01T12:00:00Z", "EIP-2", eip_text(2, "status: Final", "requires: 1"))
    questions = [
        question("F2-positive-PEP-9-2026-10-27", "F2", "PEP-9", "2026-10-27", answer_type="set"),
        question("F3-size-2-4-EIP-2-2026-10-27", "F3", "EIP-2", "2026-10-27", corpus="eth",
                 answer_type="set")]
    data, streams = build_data(tmp_path / "world", questions, {"H+": stream, "H-": stream})
    ids = {}
    for system in ("oracle", "long-context"):
        run_dir = tmp_path / system
        assert harness.main(["requests", "--system", system, "--data", str(data), "--out-dir",
                             str(run_dir), "--streams-dir", str(streams)]) == 0
        ids[system] = {(key["id"], key["arm"]) for _, key in written(run_dir)}
    assert ids["oracle"] == {(q["id"], arm) for q in questions for arm in ("H+", "H-")}
    assert ids["long-context"] == {("F2-positive-PEP-9-2026-10-27", arm) for arm in ("H+", "H-")}
    # The oracle's context for the F2 item lists the PEPs as they stand at S: both lists, as the
    # headers record them.
    (context,) = {request["context"] for request, key in written(tmp_path / "oracle")
                  if key["id"].startswith("F2")}
    assert statements_of(context) == [
        "PEP-9: status Superseded; requires -; replaces -; superseded-by 10",
        "PEP-10: status Final; requires -; replaces 9; superseded-by -"]
    (context,) = {request["context"] for request, key in written(tmp_path / "oracle")
                  if key["id"].startswith("F3")}
    assert statements_of(context) == [
        "EIP-1: status Final; requires -; replaces -; superseded-by -",
        "EIP-2: status Final; requires 1; replaces -; superseded-by -"]


def test_a_question_without_a_corpus_a_proposal_that_is_text_or_a_date_is_refused(tmp_path):
    stream = Stream()
    stream.record("2001-01-01T12:00:00Z", "PEP-9", pep_text(9, "Status: Draft"))
    good = question("F1-change-PEP-9-2001-06-01", "F1", "PEP-9", "2001-06-01")
    without_corpus = {name: value for name, value in good.items() if name != "corpus"}
    for number, bad in enumerate((
            {**good, "corpus": "both"}, {**good, "as_of": "2001-13-01"},
            {**good, "as_of": "20010601"}, {**good, "as_of": None}, {**good, "proposal": 9},
            without_corpus)):
        data, streams = build_data(tmp_path / f"world-{number}", [bad],
                                   {"H+": stream, "H-": stream})
        for system in ("closed-book", "long-context", "oracle"):
            extra = [] if system == "closed-book" else ["--streams-dir", str(streams)]
            assert harness.main(["requests", "--system", system, "--data", str(data), "--out-dir",
                                 str(tmp_path / f"run-{number}-{system}"), *extra]) == 2, bad
            assert not (tmp_path / f"run-{number}-{system}").exists()
    # The proposal of an F5 item is none, and that is no refusal.
    five = question("F5-count-pep-Draft-2001-06-01", "F5", None, "2001-06-01", arms=("H+",),
                    answer_type="count")
    data, streams = build_data(tmp_path / "world-five", [five], {"H+": stream})
    assert harness.main(["requests", "--system", "oracle", "--data", str(data), "--out-dir",
                         str(tmp_path / "run-five"), "--streams-dir", str(streams)]) == 0


# --- the harness: the streams it reads, plan.json, the order of the requests ---------------------


def test_each_distinct_stream_corpus_and_date_is_computed_once(world, questions, tmp_path,
                                                                monkeypatch):
    replayed, computed = [], []
    replay, statements = harness.materialize.replay, harness.statements

    def counting_replay(lines, days, where="the stream", log=None):
        days = sorted(days)
        replayed.append((where, days))
        return replay(lines, days, where, log)

    def counting_statements(files, corpus):
        computed.append(corpus)
        return statements(files, corpus)

    monkeypatch.setattr(harness.materialize, "replay", counting_replay)
    monkeypatch.setattr(harness, "statements", counting_statements)
    assert run_requests(world, tmp_path / "oracle", "oracle") == 0
    pairs = written(tmp_path / "oracle")
    # The oracle reads the H+ stream for H+ and H- items and the CF stream for CF items.
    triples = {("cf.jsonl" if key["arm"] == "CF" else "h-plus.jsonl",
                questions[key["id"]]["corpus"], questions[key["id"]]["as_of"])
               for _, key in pairs}
    assert len(triples) < len(pairs) / 2
    assert len(computed) == len(triples)
    assert sorted(where for where, _ in replayed) == ["cf.jsonl", "h-plus.jsonl"]
    for where, days in replayed:
        assert days == sorted({date.fromisoformat(as_of) for name, _, as_of in triples
                               if name == where})
    # Long context reads each arm's stream once, not once for each proposal or request.
    reads = []
    read_episodes = harness.read_episodes
    monkeypatch.setattr(harness, "read_episodes", lambda path: reads.append(Path(path).name)
                        or read_episodes(path))
    assert run_requests(world, tmp_path / "long-context", "long-context") == 0
    assert sorted(reads) == ["cf.jsonl", "h-minus.jsonl", "h-plus.jsonl"]


def test_plan_json_names_the_streams_a_run_read_and_closed_book_reads_none(world, tmp_path):
    for system, arms in (("long-context", ["CF", "H+", "H-"]), ("oracle", ["CF", "H+"])):
        assert run_requests(world, tmp_path / system, system) == 0
        plan = json.loads((tmp_path / system / "plan.json").read_text(encoding="utf-8"))
        assert plan["system"] == system
        assert plan["streams"] == {arm: {"file": STREAM_FILES[arm],
                                         "sha256": sha256(world / "streams" / STREAM_FILES[arm])}
                                   for arm in arms}
        assert str(tmp_path) not in json.dumps(plan) and str(ROOT) not in json.dumps(plan)
        assert plan["requests_sha256"] == sha256(tmp_path / system / "input" / "requests.jsonl")
        assert plan["keys_sha256"] == sha256(tmp_path / system / "keys.jsonl")
        assert sorted(p.name for p in (tmp_path / system).iterdir()) == [
            "input", "keys.jsonl", "output", "plan.json"]
        assert sorted(p.name for p in (tmp_path / system / "input").iterdir()) == [
            "requests.jsonl"]
    # Only the streams the asked items need are read: one counterfactual item reads cf.jsonl.
    one = tmp_path / "one.json"
    one.write_text(json.dumps(["CF-counterfactual-PEP-1-2013-09-02"]), encoding="utf-8")
    assert run_requests(world, tmp_path / "cf-only", "oracle", "--items", str(one),
                        "--runs", "2") == 0
    plan = json.loads((tmp_path / "cf-only" / "plan.json").read_text(encoding="utf-8"))
    assert list(plan["streams"]) == ["CF"] and plan["requests"] == 1
    assert run_requests(world, tmp_path / "closed", "closed-book") == 0
    plan = json.loads((tmp_path / "closed" / "plan.json").read_text(encoding="utf-8"))
    assert plan["streams"] is None


@pytest.mark.parametrize("system", ["closed-book", "long-context", "oracle"])
def test_every_entrant_s_requests_are_written_in_ascending_key_order(system, world, tmp_path):
    assert run_requests(world, tmp_path / "run", system) == 0
    requests = jsonl(tmp_path / "run" / "input" / "requests.jsonl")
    keys = jsonl(tmp_path / "run" / "keys.jsonl")
    assert [r["key"] for r in requests] == sorted(r["key"] for r in requests)
    assert sorted(r["key"] for r in requests) == sorted(k["key"] for k in keys)
    # keys.jsonl keeps the items' order, so the requests are not in it: a random permutation.
    assert [r["key"] for r in requests] != [k["key"] for k in keys]
    planned = harness.plan_requests(list(jsonl(world / "data" / "questions.jsonl")), system, None,
                                    (1, 2, 3))
    assert [(k["id"], k["arm"], k["run"]) for k in keys] == [
        (q["id"], arm, run) for q, arm, run in planned]
    # Each request holds the fields of its entrant and a UUID4 key, in canonical form.
    for request in requests:
        assert set(request) == ({"key", "question"} if system == "closed-book"
                                else {"key", "question", "context"})
        assert str(uuid.UUID(request["key"], version=4)) == request["key"]


def test_the_requests_follow_the_keys_and_not_the_items(world, tmp_path, monkeypatch):
    counter = itertools.count()

    def descending() -> uuid.UUID:
        return uuid.UUID(int=(0xFFFF_FFFF - next(counter)) << 96, version=4)

    monkeypatch.setattr(harness.uuid, "uuid4", descending)
    assert run_requests(world, tmp_path / "run", "oracle") == 0
    requests = jsonl(tmp_path / "run" / "input" / "requests.jsonl")
    keys = jsonl(tmp_path / "run" / "keys.jsonl")
    # The first item's keys are the largest, so the file ends with them and holds the last item's
    # first.
    assert [r["key"] for r in requests] == [k["key"] for k in reversed(keys)]


def test_no_request_carries_the_item_and_the_reader_accepts_every_request_file(world, questions,
                                                                                 runs):
    for system, run_dir in runs.items():
        path = run_dir / "input" / "requests.jsonl"
        text = path.read_text(encoding="utf-8")
        for item_id in questions:
            assert item_id not in text
        for request in jsonl(path):
            assert set(request) == {"key", "question", "context"}
            assert answer_requests.request_problem(request) is None
        assert answer_requests.scan_requests(path) == [r["key"] for r in jsonl(path)]
        assert sorted(p.name for p in (run_dir / "input").iterdir()) == ["requests.jsonl"], system


def test_a_run_that_cannot_read_its_streams_is_refused_and_writes_nothing(world, tmp_path, capsys):
    data = str(world / "data")

    def attempt(name: str, system: str, *extra: str) -> int:
        return harness.main(["requests", "--system", system, "--data", data, "--out-dir",
                             str(tmp_path / name), *extra])

    # No --streams-dir for a system that reads streams, and one for a system that reads none.
    assert attempt("none", "long-context") == 2
    assert "--streams-dir" in capsys.readouterr().err
    assert attempt("oracle-none", "oracle") == 2
    assert attempt("closed", "closed-book", "--streams-dir", str(world / "streams")) == 2
    assert "--streams-dir" in capsys.readouterr().err
    # A directory without the stream, one with a stream that is not the registered one, and a
    # build whose manifest registers no stream.
    (tmp_path / "empty").mkdir()
    assert attempt("empty-dir", "oracle", "--streams-dir", str(tmp_path / "empty")) == 2
    assert re.search(r"(cf|h-plus)\.jsonl", capsys.readouterr().err)
    edited = tmp_path / "edited"
    shutil.copytree(world / "streams", edited)
    with open(edited / "h-plus.jsonl", "a", encoding="utf-8") as handle:
        handle.write("\n")
    assert attempt("edited-run", "oracle", "--streams-dir", str(edited)) == 2
    assert "registers" in capsys.readouterr().err
    bare = tmp_path / "bare"
    shutil.copytree(world / "data", bare)
    manifest = json.loads((bare / "MANIFEST.json").read_text(encoding="utf-8"))
    manifest["streams"] = None
    (bare / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert harness.main(["requests", "--system", "oracle", "--data", str(bare), "--out-dir",
                         str(tmp_path / "bare-run"), "--streams-dir", str(world / "streams")]) == 2
    assert "registers no" in capsys.readouterr().err
    for name in ("none", "oracle-none", "closed", "empty-dir", "edited-run", "bare-run"):
        assert not (tmp_path / name).exists(), name


def test_the_context_descriptions_are_exactly_these_and_the_configs_record_them():
    assert harness.LONG_CONTEXT_DESCRIPTION == LONG_CONTEXT
    assert harness.ORACLE_DESCRIPTION == ORACLE
    for system, description in (("long-context", LONG_CONTEXT), ("oracle", ORACLE)):
        config = json.loads((SYSTEMS / system / "config.json").read_text(encoding="utf-8"))
        assert config["system"] == system
        assert config["prompts"]["context_description"] == description
        assert config["requests"]["fields"] == ["key", "question", "context"]
