"""The dated revision stream on synthetic repositories (spec section 8 and the CF paragraph of 7).

Each test builds three throwaway repositories laid out like the corpus clones (``peps``, ``EIPs``
and ``ERCs``), loads them with ``load_corpus`` and reads the stream through ``episodes`` or through
the ``asofbench stream`` command. Expected texts are written out by hand or taken from ``difflib``
directly, never from the code under test.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from gitrepo import TempRepo
from streamcheck import apply_diff

from asofbench import cli
from asofbench.corpora import CANARY
from asofbench.headers import WITHHELD
from asofbench.questions import dumps
from asofbench.stream import Episode, episodes, unified_diff
from asofbench.timeline import load_corpus

BODY = "".join(f"Body line {i} of the proposal text.\n" for i in range(24))
# Shares no line with BODY, so git never pairs the two texts as a rename.
OTHER_BODY = "".join(f"Rewritten paragraph {i}, nothing like the old one.\n" for i in range(30))


def pep(number: int, status: str | None, *, title="Example", headers=(), body=BODY) -> str:
    lines = [f"PEP: {number}", f"Title: {title}"]
    if status is not None:
        lines.append(f"Status: {status}")
    lines.extend(headers)
    return "\n".join(lines) + "\n\n" + body


def eip(number: int, status: str | None, *, title="Example", headers=(), body=BODY) -> str:
    lines = ["---", f"eip: {number}", f"title: {title}"]
    if status is not None:
        lines.append(f"status: {status}")
    lines.extend(headers)
    lines.append("---")
    return "\n".join(lines) + "\n" + body


def noon(day: str) -> str:
    return f"{day}T12:00:00Z"


class World:
    """Three repositories named and laid out like the clones, each with one early commit."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.repos: dict[str, TempRepo] = {}
        for name in ("peps", "EIPs", "ERCs"):
            repo = TempRepo(root / name)
            repo.write("README.md", "Proposals.\n")
            repo.commit("start", "2000-01-01T00:00:00Z")
            self.repos[name] = repo

    def __getitem__(self, name: str) -> TempRepo:
        return self.repos[name]

    def load(self):
        return load_corpus(self.root)


@pytest.fixture
def world(tmp_path):
    return World(tmp_path / "clones")


def stream_of(world: World, arm: str, cf: dict[str, str] | None = None) -> list[Episode]:
    corpus = world.load()
    with corpus.readers() as readers:
        return list(episodes(corpus, readers, arm, cf))


def only(found: list[Episode], proposal: str) -> list[Episode]:
    return [e for e in found if e.proposal == proposal]


# --- the diff function ---------------------------------------------------------------------------


def test_unified_diff_is_difflib_with_three_lines_of_context():
    assert unified_diff("a\nb\n", "a\nb\n") == ""
    assert unified_diff("a\n", "b\n") == "--- a\n+++ b\n@@ -1 +1 @@\n-a\n+b\n"
    before = "".join(f"{i}\n" for i in range(20))
    after = before.replace("10\n", "ten\n")
    assert unified_diff(before, after) == (
        "--- a\n+++ b\n@@ -8,7 +8,7 @@\n 7\n 8\n 9\n-10\n+ten\n 11\n 12\n 13\n"
    )
    wrapped = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True), "a", "b", n=3
        )
    )
    assert unified_diff(before, after) == wrapped


def test_unified_diff_marks_a_last_line_without_a_newline():
    marker = "\\ No newline at end of file\n"
    assert unified_diff("a", "a") == ""
    assert unified_diff("a\nb", "a\nb\n") == (
        f"--- a\n+++ b\n@@ -1,2 +1,2 @@\n a\n-b\n{marker}+b\n"
    )
    assert unified_diff("a\nb\n", "a\nb") == (
        f"--- a\n+++ b\n@@ -1,2 +1,2 @@\n a\n-b\n+b\n{marker}"
    )
    assert unified_diff("a\nb", "a\nc") == (
        f"--- a\n+++ b\n@@ -1,2 +1,2 @@\n a\n-b\n{marker}+c\n{marker}"
    )
    # An open last line that is only context is marked too, as git marks it.
    assert unified_diff("x\ny\nz", "X\ny\nz") == (
        f"--- a\n+++ b\n@@ -1,3 +1,3 @@\n-x\n+X\n y\n z\n{marker}"
    )
    assert unified_diff("", "no newline") == f"--- a\n+++ b\n@@ -0,0 +1 @@\n+no newline\n{marker}"
    assert unified_diff("no newline", "") == f"--- a\n+++ b\n@@ -1 +0,0 @@\n-no newline\n{marker}"


def test_unified_diff_cuts_lines_at_newlines_only():
    # A form feed, a lone carriage return, a next-line character and a line separator stay inside
    # their line, so every physical line of the diff is one line of the text.
    before = "page one\n\x0c\npage two\n"
    after = "page one\n\x0c\nPage two\n"
    assert unified_diff(before, after) == (
        "--- a\n+++ b\n@@ -1,3 +1,3 @@\n page one\n \x0c\n-page two\n+Page two\n"
    )
    odd = "a\rb\x85c" + chr(0x2028) + "d\n"  # one line, with three characters splitlines cuts at
    assert unified_diff(odd + "z\n", odd + "Z\n") == (
        "--- a\n+++ b\n@@ -1,2 +1,2 @@\n " + odd + "-z\n+Z\n"
    )
    assert unified_diff("a\r\nb\r\n", "a\r\nB\r\n") == (
        "--- a\n+++ b\n@@ -1,2 +1,2 @@\n a\r\n-b\r\n+B\r\n"
    )
    # A form feed on a line of its own is a line: a change next to it keeps it as context.
    diff = unified_diff("x\n\x0c\ny\n", "x\n\x0c\nY\n")
    assert [row for row in diff.split("\n")][3:-1] == [" x", " \x0c", "-y", "+Y"]


def test_unified_diff_is_the_registered_difflib_call_for_ordinary_texts():
    # For a text that ends with a newline and has no character that splitlines cuts at but a
    # newline, the split changes nothing: the diff is the plain difflib one.
    before = "".join(f"line {i}\r\n" if i % 3 else f"line {i}\n" for i in range(30))
    after = before.replace("line 14", "changed").replace("line 3\n", "")
    plain = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True), "a", "b", n=3
        )
    )
    assert unified_diff(before, after) == plain != ""


AWKWARD_HISTORY = [
    "page one\n\x0c\npage two\nlast line\n",  # a form feed on a line of its own
    "page one\n\x0c\npage 2\nlast line",  # no final newline
    "page one\n\x0c\npage 2\nlast line\n",  # the newline comes back
    "page one\r\n\x0c\r\npage 2\r\nlast line\r\n",  # CRLF
    "page one\r\n\x0c\r\npage 2\r\nthe last line",  # CRLF, open, the last line changed
]


def test_every_diff_applies_exactly_to_awkward_texts(world):
    peps = world["peps"]
    for n, body in enumerate(AWKWARD_HISTORY):
        peps.write("pep-0001.txt", pep(1, "Draft", body=body))
        peps.commit(f"edit {n}", noon(f"2020-01-{n + 1:02d}"))
    found = stream_of(world, "H+")
    texts = [pep(1, "Draft", body=body) for body in AWKWARD_HISTORY]
    assert found[0].text == texts[0]
    for earlier, later, episode in zip(texts[:-1], texts[1:], found[1:], strict=True):
        rows = episode.text.split("\n")
        assert rows.pop() == ""
        assert all(row[:1] in " +-@\\" for row in rows)  # no row holds pieces of two lines
        assert apply_diff(earlier, episode.text) == later


# --- episode kinds and texts ---------------------------------------------------------------------


def test_create_is_full_text(world):
    peps, eips = world["peps"], world["EIPs"]
    ptext, etext = pep(1, "Draft", title="Purpose"), eip(5, "Draft", title="Five")
    peps.write("pep-0001.txt", ptext)
    psha = peps.commit("Add PEP 1", noon("2020-01-01"))
    eips.write("EIPS/eip-5.md", etext)
    esha = eips.commit("Add EIP 5", noon("2020-01-02"))
    assert stream_of(world, "H+") == [
        Episode(
            episode_id=f"{psha[:12]}:PEP-1",
            corpus="pep",
            repo="peps",
            proposal="PEP-1",
            ref='PEP 1 ("Purpose")',
            recorded_at="2020-01-01T12:00:00Z",
            commit=psha,
            message="Add PEP 1",
            kind="create",
            text=ptext,
        ),
        Episode(
            episode_id=f"{esha[:12]}:EIP-5",
            corpus="eth",
            repo="EIPs",
            proposal="EIP-5",
            ref='EIP-5 ("Five")',
            recorded_at="2020-01-02T12:00:00Z",
            commit=esha,
            message="Add EIP 5",
            kind="create",
            text=etext,
        ),
    ]


def test_modify_is_diff_with_three_lines_context(world):
    peps = world["peps"]
    before = pep(1, "Draft")
    after = before.replace("Body line 12 of the", "Edited line 12 of the")
    peps.write("pep-0001.txt", before)
    peps.commit("Add PEP 1", noon("2020-01-01"))
    peps.write("pep-0001.txt", after)
    edited = peps.commit("Edit PEP 1", noon("2020-01-02"))
    create, modify = stream_of(world, "H+")
    assert (create.kind, modify.kind, modify.commit, modify.message) == (
        "create",
        "modify",
        edited,
        "Edit PEP 1",
    )
    expected = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True), "a", "b", n=3
        )
    )
    assert modify.text == expected
    lines = modify.text.splitlines()
    assert lines[:2] == ["--- a", "+++ b"]
    assert re.fullmatch(r"@@ -\d+,7 \+\d+,7 @@", lines[2])
    assert [line[0] for line in lines[3:]] == [" "] * 3 + ["-", "+"] + [" "] * 3


def test_delete_is_empty(world):
    peps = world["peps"]
    peps.write("pep-0001.txt", pep(1, "Draft", title="Doomed"))
    peps.commit("Add PEP 1", noon("2020-01-01"))
    peps.rm("pep-0001.txt")
    deleted = peps.commit("Remove PEP 1", noon("2020-02-01"))
    create, delete = stream_of(world, "H+")
    assert (delete.kind, delete.text, delete.message, delete.commit) == (
        "delete",
        "",
        "Remove PEP 1",
        deleted,
    )
    # A deleted proposal keeps the id and the title it had.
    assert (delete.proposal, delete.ref) == ("PEP-1", 'PEP 1 ("Doomed")')
    assert delete.recorded_at == "2020-02-01T12:00:00Z"


def test_a_deleted_then_readded_proposal_is_created_again(world):
    peps = world["peps"]
    peps.write("pep-0004.txt", pep(4, "Draft"))
    peps.commit("add", noon("2020-01-01"))
    peps.rm("pep-0004.txt")
    peps.commit("delete", noon("2020-02-01"))
    again = pep(4, "Draft", title="Second life")
    peps.write("pep-0004.txt", again)
    peps.commit("add again", noon("2020-05-01"))
    found = stream_of(world, "H+")
    assert [e.kind for e in found] == ["create", "delete", "create"]
    assert found[2].text == again  # the full file, not a diff against the deleted text
    assert {e.ref for e in found} == {'PEP 4 ("Second life")'}


def test_a_rename_is_a_diff_of_the_two_texts(world):
    peps = world["peps"]
    before, after = pep(1, "Draft"), pep(1, "Final")
    peps.write("pep-0001.txt", before)
    peps.commit("add", noon("2020-01-01"))
    peps.rm("pep-0001.txt")
    peps.write("peps/pep-0001.rst", after)
    moved = peps.commit("move into peps/ and finalise", noon("2020-01-02"))
    create, rename = stream_of(world, "H+")
    assert (create.kind, rename.kind, rename.commit) == ("create", "rename", moved)
    assert rename.text == unified_diff(before, after) != ""


def test_a_mode_only_change_is_a_modify_with_an_empty_diff(world):
    peps = world["peps"]
    peps.write("pep-0365.txt", pep(365, "Final"))
    peps.commit("add", noon("2009-01-01"))
    os.chmod(peps.path / "pep-0365.txt", 0o755)
    chmod = peps.commit("make it executable", noon("2010-01-01"))
    create, modify = stream_of(world, "H+")
    assert (modify.kind, modify.text, modify.message, modify.commit) == (
        "modify",
        "",
        "make it executable",
        chmod,
    )


def test_a_symlink_replaced_by_a_file_is_a_create(world):
    # PEP 495 was first added as a symlink and 19 minutes later became a regular file. The symlink
    # is no proposal text, so the file that replaced it is the proposal's first appearance.
    peps = world["peps"]
    (peps.path / "pep-0495.rst").symlink_to("../elsewhere/pep-0495.rst")
    peps.commit("add as a symlink", "2014-07-27T12:00:00Z")
    peps.rm("pep-0495.rst")
    text = pep(495, "Draft")
    peps.write("pep-0495.rst", text)
    made = peps.commit("make it a file", "2014-07-27T12:19:00Z")
    found = stream_of(world, "H+")
    assert [(e.kind, e.commit, e.text) for e in found] == [("create", made, text)]


def test_proposals_that_are_not_in_the_corpus_have_no_episodes(world):
    peps = world["peps"]
    peps.write("pep-0000.txt", pep(0, "Active", title="Index"))
    peps.write("pep-0401.txt", pep(401, "April Fool!"))
    peps.write("pep-0008.txt", pep(8, "Active"))
    peps.commit("add", noon("2020-01-01"))
    assert {e.proposal for e in stream_of(world, "H+")} == {"PEP-8"}


def test_the_message_is_the_whole_commit_message(world):
    peps = world["peps"]
    peps.write("pep-0001.txt", pep(1, "Draft"))
    peps.commit("Add PEP 1\n\nAccepted by the steering council.\nSecond line.", noon("2020-01-01"))
    (episode,) = stream_of(world, "H+")
    assert episode.message == "Add PEP 1\n\nAccepted by the steering council.\nSecond line."


def test_crlf_and_bom_are_kept_in_a_create(world):
    peps = world["peps"]
    text = "\ufeff" + pep(1, "Draft").replace("\n", "\r\n")
    peps.write("pep-0001.txt", text)
    peps.commit("add", noon("2020-01-01"))
    (episode,) = stream_of(world, "H+")
    assert episode.text == text


# --- ERCs ----------------------------------------------------------------------------------------


def test_erc_episodes_come_from_the_stitched_versions(world):
    eips, ercs = world["EIPs"], world["ERCs"]
    draft, final = eip(20, "Draft"), eip(20, "Final")
    eips.write("EIPS/eip-20.md", draft)
    drafted = eips.commit("draft", noon("2017-01-01"))
    eips.write("EIPS/eip-20.md", final)
    finalised = eips.commit("final", noon("2018-01-01"))
    # The move's stubs and marker are dropped by the stitching and so yield no episode.
    eips.write("EIPS/eip-20.md", "This file was moved.\n")
    eips.commit("stub", "2023-10-25T15:45:00Z")
    moved_text = eip(20, "Final", body=BODY + "Moved to the ERC repository.\n")
    ercs.write("ERCS/erc-20.md", moved_text)
    created = ercs.commit("complete erc move", "2023-10-25T15:11:00Z")
    eips.write("EIPS/eip-20.md", eip(20, "Moved", body="This file was moved.\n"))
    eips.commit("moved", noon("2023-10-26"))
    edited_text = eip(20, "Final", body=BODY + "Moved to the ERC repository.\nOne more.\n")
    ercs.write("ERCS/erc-20.md", edited_text)
    edited = ercs.commit("edit", noon("2024-01-01"))
    found = stream_of(world, "H+")
    assert [(e.proposal, e.repo, e.kind, e.commit) for e in found] == [
        ("ERC-20", "EIPs", "create", drafted),
        ("ERC-20", "EIPs", "modify", finalised),
        ("ERC-20", "ERCs", "rename", created),
        ("ERC-20", "ERCs", "modify", edited),
    ]
    assert {e.ref for e in found} == {'ERC-20 ("Example")'}
    assert {e.corpus for e in found} == {"eth"}
    assert found[0].text == draft
    assert found[1].text == unified_diff(draft, final)
    # The first ERC-repository version moves the file: a diff against the last EIP text kept.
    assert found[2].text == unified_diff(final, moved_text) != ""
    assert found[3].text == unified_diff(moved_text, edited_text)
    assert [e.recorded_at for e in found] == [
        "2017-01-01T12:00:00Z",
        "2018-01-01T12:00:00Z",
        "2023-10-25T15:11:00Z",
        "2024-01-01T12:00:00Z",
    ]


# --- H-: the withheld header lines ----------------------------------------------------------------

PEP_HEADERS = [
    "Author: Alice",
    "Requires: 2",
    "Replaces: 3",
    "Superseded-By: 4",
    "Resolution: https://example.org/msg",
    "    and a continuation line",
    "Created: 2020-01-01",
]
WITHHELD_LINE = re.compile(  # a line, or a diff line, that starts with a withheld field's name
    r"^[ +-]?(" + "|".join(map(re.escape, sorted(WITHHELD, key=len, reverse=True))) + r")\b",
    re.IGNORECASE,
)


def test_hminus_strips_header_in_text_and_diff(world):
    peps = world["peps"]
    first = pep(1, "Draft", headers=PEP_HEADERS, body="Status: in body\n" + BODY)
    second = (
        pep(1, "Accepted", headers=PEP_HEADERS, body="Status: in body\n" + BODY)
        .replace("Requires: 2", "Requires: 5")
        .replace("Body line 0 of", "Edited line 0 of")
    )
    peps.write("pep-0001.txt", first)
    peps.commit("Add PEP 1", noon("2020-01-01"))
    peps.write("pep-0001.txt", second)
    peps.commit("Accept PEP 1 and edit", noon("2020-02-01"))

    kept = "PEP: 1\nTitle: Example\nAuthor: Alice\nCreated: 2020-01-01\n\n"
    shown_first = kept + "Status: in body\n" + BODY
    shown_second = shown_first.replace("Body line 0 of", "Edited line 0 of")
    create, modify = stream_of(world, "H-")
    assert create.text == shown_first
    assert modify.text == unified_diff(shown_first, shown_second)
    # The body line that looks like a header survives, as does the header's context.
    assert [x for x in create.text.splitlines() if WITHHELD_LINE.match(x)] == ["Status: in body"]
    assert [x for x in modify.text.splitlines() if WITHHELD_LINE.match(x)] == [" Status: in body"]
    assert " Created: 2020-01-01" in modify.text.splitlines()
    assert "continuation" not in create.text + modify.text
    # The same history with headers visible shows the withheld lines.
    plus_create, plus_modify = stream_of(world, "H+")
    assert plus_create.text == first
    assert "-Status: Draft" in plus_modify.text and "+Status: Accepted" in plus_modify.text


def test_hminus_strips_front_matter_and_qualified_keys(world):
    eips = world["EIPs"]
    headers = [
        "author: Alice",
        "requires: 1",
        "requires (*optional): 2",
        "last-call-deadline: 2020-02-01",
        "withdrawal-reason: none",
        "resolution: x",
        "superseded-by: 9",
        "replaces: 3",
        "created: 2020-01-01",
    ]
    eips.write("EIPS/eip-7.md", eip(7, "Review", headers=headers, body="status: in body\n" + BODY))
    eips.commit("Add EIP 7", noon("2020-01-01"))
    # The second text changes every withheld value and drops the qualified requires line, and
    # its body no longer starts with the line that looks like a header.
    eips.write("EIPS/eip-7.md", eip(7, "Final", headers=headers[:2] + headers[4:], body=BODY))
    eips.commit("Finalise EIP 7", noon("2020-02-01"))
    head = "---\neip: 7\ntitle: Example\nauthor: Alice\ncreated: 2020-01-01\n---\n"
    shown_first, shown_second = head + "status: in body\n" + BODY, head + BODY
    create, modify = stream_of(world, "H-")
    assert create.text == shown_first
    assert modify.text == unified_diff(shown_first, shown_second)
    assert "-status: in body" in modify.text.splitlines()
    assert not [x for x in modify.text.splitlines() if re.match(r"[ +-]\s*requires", x)]


def test_hminus_strips_the_earlier_and_other_names_of_withheld_fields(world):
    """Section 8: the review period's earlier names, require, dependencies and accepted go too."""
    eips, peps = world["EIPs"], world["peps"]
    headers = ["author: Alice", "review-period-end: 2019-12-02", "review-period-ends: 2019-12-12",
               "review-end-date: 2018-11-06", "require: 1066", "dependencies: 165", "created: x"]
    eips.write("EIPS/eip-7.md", eip(7, "Last Call", headers=headers))
    eips.commit("Add EIP 7", noon("2020-01-01"))
    extended = [headers[0], "review-period-end: 2019-12-09", *headers[2:]]
    eips.write("EIPS/eip-7.md", eip(7, "Last Call", headers=extended))
    eips.commit("Extend the review period of EIP 7", noon("2020-02-01"))
    peps.write("pep-3147.txt", pep(3147, "Final", headers=["Accepted: http://example.invalid/a"]))
    peps.commit("Add PEP 3147", noon("2020-03-01"))
    shown = "---\neip: 7\ntitle: Example\nauthor: Alice\ncreated: x\n---\n" + BODY
    create, modify, pep_create = stream_of(world, "H-")
    assert create.text == shown
    assert modify.text == ""  # only a withheld value changed: the diff is empty, the message kept
    assert modify.message == "Extend the review period of EIP 7"
    assert pep_create.text == "PEP: 3147\nTitle: Example\n\n" + BODY
    plus = stream_of(world, "H+")
    assert "review-period-end: 2019-12-02" in plus[0].text and "Accepted: " in plus[2].text


def test_status_only_change_keeps_message(world):
    peps = world["peps"]
    peps.write("pep-0001.txt", pep(1, "Draft"))
    peps.commit("Add PEP 1", noon("2020-01-01"))
    peps.write("pep-0001.txt", pep(1, "Accepted"))
    accepted = peps.commit("Accept PEP 1", noon("2020-02-01"))
    create, modify = stream_of(world, "H-")
    assert (modify.kind, modify.text, modify.message, modify.commit) == (
        "modify",
        "",
        "Accept PEP 1",
        accepted,
    )
    # With headers visible, the same episode is the status diff.
    visible = stream_of(world, "H+")[1]
    assert visible.text == unified_diff(pep(1, "Draft"), pep(1, "Accepted")) != ""


def test_the_episodes_whose_diff_the_withholding_empties_are_the_streams_own(world):
    """Section 8 keeps an H- episode whose diff becomes empty, with its message: a modify or a
    rename that changed only withheld header lines. ``header_only_times`` gives the recorded
    times of exactly those episodes of a proposal: an H- episode with an empty text whose H+
    text is not empty. A content change, a status change made with one, an identity field, a
    mode-only change and a deletion are not among them."""
    from asofbench.stream import header_only_times

    peps, eips = world["peps"], world["EIPs"]
    steps = [
        ("2020-01-01", "Add PEP 1", pep(1, "Draft", headers=["Requires: 2"])),
        ("2020-02-01", "Accept PEP 1", pep(1, "Accepted", headers=["Requires: 2"])),  # status
        ("2020-03-01", "Edit PEP 1", pep(1, "Accepted", headers=["Requires: 2"],
                                         body=BODY + "More.\n")),  # content
        ("2020-04-01", "Finish PEP 1", pep(1, "Final", headers=["Requires: 2"])),  # both
        ("2020-05-01", "Require PEP 3", pep(1, "Final", headers=["Requires: 3"])),  # relation
        ("2020-07-01", "Retitle PEP 1", pep(1, "Final", title="Renamed",
                                            headers=["Requires: 3"])),  # an identity field
    ]
    for day, message, text in steps:
        peps.write("pep-0001.txt", text)
        peps.commit(message, noon(day))
        if day == "2020-05-01":
            os.chmod(peps.path / "pep-0001.txt", 0o755)
            peps.commit("Make PEP 1 executable", noon("2020-06-01"))  # a mode-only change
    peps.mv("pep-0001.txt", "peps/pep-0001.rst")
    peps.write("peps/pep-0001.rst", pep(1, "Superseded", title="Renamed", headers=["Requires: 3"]))
    peps.commit("Move PEP 1 and supersede it", noon("2020-08-01"))  # a rename, status only
    peps.rm("peps/pep-0001.rst")
    peps.commit("Remove PEP 1", noon("2020-09-01"))
    eips.write("EIPS/eip-7.md", eip(7, "Last Call", headers=["last-call-deadline: 2020-03-01"]))
    eips.commit("Add EIP 7", noon("2020-01-15"))
    eips.write("EIPS/eip-7.md", eip(7, "Last Call", headers=["last-call-deadline: 2020-04-01"]))
    eips.commit("Extend the last call of EIP 7", noon("2020-02-15"))  # a withheld date only
    minus, plus = stream_of(world, "H-"), stream_of(world, "H+")
    by_id = {e.episode_id: e for e in plus}
    shown = sorted((e.proposal, e.recorded_at) for e in minus
                   if e.kind in ("modify", "rename") and e.text == ""
                   and by_id[e.episode_id].text != "")
    assert shown == [("EIP-7", "2020-02-15T12:00:00Z"), ("PEP-1", "2020-02-01T12:00:00Z"),
                     ("PEP-1", "2020-05-01T12:00:00Z"), ("PEP-1", "2020-08-01T12:00:00Z")]
    corpus = world.load()
    with corpus.readers() as readers:
        found = {pid: header_only_times(corpus.proposals[pid], readers)
                 for pid in ("PEP-1", "EIP-7")}
        later = header_only_times(corpus.proposals["PEP-1"], readers,
                                  after=int(datetime.fromisoformat("2020-05-01T12:00:00+00:00")
                                            .timestamp()))
    assert sorted((pid, datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))
                  for pid, times in found.items() for t in times) == shown
    assert [datetime.fromtimestamp(t, UTC).date().isoformat() for t in later] == [
        "2020-08-01"]  # only those recorded after the given second


def test_hminus_leaves_everything_but_text_unchanged(world):
    peps = world["peps"]
    peps.write("pep-0001.txt", pep(1, "Draft", headers=["Requires: 2"]))
    peps.commit("Add PEP 1", noon("2020-01-01"))
    peps.write("pep-0001.txt", pep(1, "Final", headers=["Requires: 2"], body=OTHER_BODY))
    peps.commit("Rewrite", noon("2020-02-01"))
    peps.rm("pep-0001.txt")
    peps.commit("Remove", noon("2020-03-01"))
    plus, minus = stream_of(world, "H+"), stream_of(world, "H-")
    assert [replace(e, text="") for e in plus] == [replace(e, text="") for e in minus]
    assert minus[2].text == ""  # a delete has no text in either arm


# --- CF: the counterfactual stream ----------------------------------------------------------------

UPDATE = 'Update status of PEP 1 ("Example")'


def test_cf_permutes_whole_history(world):
    peps = world["peps"]
    edited = BODY.replace("Body line 0 of", "Edited line 0 of")
    history = [
        ("Add PEP 1", "2020-01-01", pep(1, "Accepted")),
        ("Mark PEP 1 as final", "2020-02-01", pep(1, "Final")),
        ("Fix typo", "2020-03-01", pep(1, "Final", body=edited)),
        ("Approve PEP 1", "2020-04-01", pep(1, "Approved", body=edited)),
        ("Back to incomplete", "2020-05-01", pep(1, "Incomplete", body=edited)),
        ("Tidy the body", "2020-06-01", pep(1, "Incomplete", body=BODY)),
    ]
    for message, day, text in history:
        peps.write("pep-0001.txt", text)
        peps.commit(message, noon(day))
    # An ordinary proposal, whose status commits are the very kind that CF rewrites for PEP 1.
    for message, day, status in [("Accept PEP 2", "2020-01-15", "Accepted"),
                                 ("Finalise PEP 2", "2020-02-15", "Final")]:
        peps.write("pep-0002.txt", pep(2, status))
        peps.commit(message, noon(day))
    cf = {"PEP-1": 'PEP 1 ("Example")'}
    plus, counterfactual = stream_of(world, "H+"), stream_of(world, "CF", cf)

    # Statuses as the stream shows them: Accepted/Rejected and Final/Withdrawn swap, the legacy
    # name Approved is read as Accepted before swapping, and Incomplete (read as Draft, which
    # the permutation leaves alone) is left exactly as written.
    shown = [
        pep(1, "Rejected"),
        pep(1, "Withdrawn"),
        pep(1, "Withdrawn", body=edited),
        pep(1, "Rejected", body=edited),
        pep(1, "Incomplete", body=edited),
        pep(1, "Incomplete", body=BODY),
    ]
    one = only(counterfactual, "PEP-1")
    assert one[0].text == shown[0]
    steps = [unified_diff(a, b) for a, b in zip(shown, shown[1:], strict=False)]
    assert [e.text for e in one[1:]] == steps
    # A diff keeps the permuted status as context, and the true status appears nowhere.
    assert " Status: Withdrawn" in one[2].text.splitlines()
    assert all("Final" not in e.text for e in one)
    # A commit that changes the status line is renamed; the create and the rest keep theirs.
    assert [e.message for e in one] == [
        "Add PEP 1",
        UPDATE,
        "Fix typo",
        UPDATE,
        UPDATE,
        "Tidy the body",
    ]
    # Nothing but text and message differs from H+ for the proposal, and nothing at all for
    # the other proposals.
    assert [replace(e, text="", message="") for e in one] == [
        replace(e, text="", message="") for e in only(plus, "PEP-1")
    ]
    assert only(counterfactual, "PEP-2") == only(plus, "PEP-2") != []
    assert len(counterfactual) == len(plus)


def test_cf_rewrites_every_episode_of_a_commit_that_changes_a_cf_status_line(world):
    peps, eips = world["peps"], world["EIPs"]

    def one(number, status, *, body=BODY):
        return pep(number, status, title=f"Title {number}", body=body)

    peps.write("pep-0001.txt", one(1, "Accepted"))  # CF
    peps.write("pep-0002.txt", one(2, "Final"))  # not CF
    peps.write("pep-0003.txt", one(3, "Draft"))  # CF
    peps.write("pep-0004.txt", one(4, "Draft"))  # not CF
    peps.commit("Add four PEPs, two of them accepted and final", noon("2020-01-01"))
    # One commit changes the status line of PEP 1 only (a CF proposal), and in the same commit:
    # a body edit to a proposal that is not CF, a body edit to a CF proposal whose status line
    # does not change, a deletion and a creation.
    peps.write("pep-0001.txt", one(1, "Final"))
    peps.write("pep-0002.txt", one(2, "Final", body=OTHER_BODY))
    peps.write("pep-0003.txt", one(3, "Draft", body=OTHER_BODY))
    peps.rm("pep-0004.txt")
    peps.write("pep-0005.txt", one(5, "Draft"))
    peps.commit("Mark PEP 1 as Final; tidy PEP 2 and PEP 3; drop PEP 4; add PEP 5",
                noon("2020-02-01"))
    # A commit that touches CF proposals but changes none of their status lines is left alone.
    peps.write("pep-0002.txt", one(2, "Final", body=BODY))
    peps.write("pep-0003.txt", one(3, "Draft", body=BODY))
    peps.commit("Tidy PEP 2 and PEP 3 again; PEP 1 is final", noon("2020-03-01"))
    # Two CF proposals change their status lines in one commit, with a sibling that is not CF.
    peps.write("pep-0001.txt", one(1, "Withdrawn"))
    peps.write("pep-0003.txt", one(3, "Accepted"))
    peps.write("pep-0005.txt", one(5, "Draft", body=OTHER_BODY))
    peps.commit("Withdraw PEP 1 and accept PEP 3", noon("2020-04-01"))
    # A different commit, in another repository, that mentions a status.
    eips.write("EIPS/eip-9.md", eip(9, "Final", title="Title 9"))
    eips.commit("Mark EIP 9 as Final", noon("2020-05-01"))

    cf = {"PEP-1": 'PEP 1 ("Title 1")', "PEP-3": 'PEP 3 ("Title 3")'}
    plus, edited = stream_of(world, "H+"), stream_of(world, "CF", cf)
    assert [e.episode_id for e in plus] == [e.episode_id for e in edited]

    by_commit = {}
    for e in edited:
        by_commit.setdefault(e.commit, []).append(e)
    order = list(dict.fromkeys(e.commit for e in edited))
    first, second, third, fourth, fifth = [by_commit[c] for c in order]
    # Commits that change no CF status line: every message as in H+.
    for group in (first, third, fifth):
        for e in group:
            assert e.message == next(x.message for x in plus if x.episode_id == e.episode_id)
    assert [e.message for e in first] == ["Add four PEPs, two of them accepted and final"] * 4
    assert [e.message for e in third] == ["Tidy PEP 2 and PEP 3 again; PEP 1 is final"] * 2
    assert fifth[0].message == "Mark EIP 9 as Final"
    # The commit that changes PEP 1's status line: its own episode says so, all others say Update.
    got = {e.proposal: (e.kind, e.message) for e in second}
    assert got == {
        "PEP-1": ("modify", 'Update status of PEP 1 ("Title 1")'),
        "PEP-2": ("modify", 'Update PEP 2 ("Title 2")'),  # a proposal that is not CF
        "PEP-3": ("modify", 'Update PEP 3 ("Title 3")'),  # CF, but its status line did not change
        "PEP-4": ("delete", 'Update PEP 4 ("Title 4")'),  # a deletion keeps the ref it had
        "PEP-5": ("create", 'Update PEP 5 ("Title 5")'),  # and a creation
    }
    # Two CF proposals change in one commit: each says so with its own ref.
    got = {e.proposal: e.message for e in fourth}
    assert got == {
        "PEP-1": 'Update status of PEP 1 ("Title 1")',
        "PEP-3": 'Update status of PEP 3 ("Title 3")',
        "PEP-5": 'Update PEP 5 ("Title 5")',
    }
    # Every sibling message is "Update {ref}" with the episode's own ref field.
    for e in second + fourth:
        assert e.message in (f"Update {e.ref}", f"Update status of {cf.get(e.proposal)}")
    # Nothing but the message (and, for a CF proposal, the permuted status in the text) differs
    # from H+, in every episode.
    for a, b in zip(plus, edited, strict=True):
        assert replace(a, text="", message="") == replace(b, text="", message="")
        if a.proposal not in cf:
            assert a.text == b.text
    # No message of the CF stream names the true status of a proposal in a commit that changed it.
    assert not [e for e in second + fourth if "Final" in e.message or "Accepted" in e.message
                or "Withdrawn" in e.message]


def test_cf_permutes_ethereum_statuses(world):
    eips = world["EIPs"]
    ref = 'EIP-5 ("Example")'
    history = [
        ("Add EIP 5", "2020-01-01", "Review"),
        ("Move to last call", "2020-02-01", "Last Call"),
        ("Finalise", "2020-03-01", "Final"),
        ("Stagnant", "2020-04-01", "Stagnant"),
    ]
    for message, day, status in history:
        eips.write("EIPS/eip-5.md", eip(5, status, headers=["requires: 1"]))
        eips.commit(message, noon(day))
    found = stream_of(world, "CF", {"EIP-5": ref})
    shown = [
        eip(5, "Last Call", headers=["requires: 1"]),
        eip(5, "Review", headers=["requires: 1"]),
        eip(5, "Withdrawn", headers=["requires: 1"]),
        eip(5, "Stagnant", headers=["requires: 1"]),  # not swapped
    ]
    assert found[0].text == shown[0]
    steps = [unified_diff(a, b) for a, b in zip(shown, shown[1:], strict=False)]
    assert [e.text for e in found[1:]] == steps
    assert [e.message for e in found] == ["Add EIP 5"] + [f"Update status of {ref}"] * 3
    # Headers stay visible in this arm: the requires line is still there.
    assert "requires: 1" in found[0].text


def test_cf_keeps_indentation_and_line_endings(world):
    eips = world["EIPs"]
    run = (
        "<pre>\r\n  eip: 6\r\n  title: Example\r\n  status: Final\r\n"
        "  type: Standards Track\r\n</pre>\r\n"
    )
    eips.write("eip-6.mediawiki", run + "\r\nBody.\r\n")
    eips.commit("Add EIP 6", noon("2016-01-01"))
    (episode,) = stream_of(world, "CF", {"EIP-6": 'EIP-6 ("Example")'})
    assert episode.text == run.replace("status: Final", "status: Withdrawn") + "\r\nBody.\r\n"


def test_cf_does_not_touch_the_body_or_other_proposals(world):
    peps = world["peps"]
    body = "Status: Final in the body, and a line saying Final and Accepted.\n" + BODY
    peps.write("pep-0001.txt", pep(1, "Final", body=body))
    peps.write("pep-0002.txt", pep(2, "Final", body=body))
    peps.commit("add both", noon("2020-01-01"))
    found = stream_of(world, "CF", {"PEP-1": 'PEP 1 ("Example")'})
    assert only(found, "PEP-1")[0].text == pep(1, "Withdrawn", body=body)
    assert only(found, "PEP-2")[0].text == pep(2, "Final", body=body)


def test_cf_needs_its_proposals(world):
    peps = world["peps"]
    peps.write("pep-0001.txt", pep(1, "Final"))
    peps.commit("add", noon("2020-01-01"))
    with pytest.raises(ValueError, match="CF"):
        stream_of(world, "CF")
    with pytest.raises(ValueError, match="PEP-99"):
        stream_of(world, "CF", {"PEP-99": "PEP 99"})


def test_the_arm_is_checked(world):
    with pytest.raises(ValueError, match="arm"):
        stream_of(world, "H")


# --- order and ids -------------------------------------------------------------------------------


def test_order_and_ids(world):
    peps, eips = world["peps"], world["EIPs"]
    peps.write("pep-0001.txt", pep(1, "Draft"))
    first = peps.commit("start PEP 1", noon("2019-12-01"))
    # Two proposals in one commit, a later commit in the same second, and an EIP in that second.
    peps.write("pep-0010.txt", pep(10, "Draft"))
    peps.write("pep-0002.txt", pep(2, "Draft"))
    both = peps.commit("add PEP 2 and PEP 10", noon("2020-01-01"))
    peps.write("pep-0001.txt", pep(1, "Final"))
    again = peps.commit("same second, next in the chain", noon("2020-01-01"))
    # A committer date before the parent's is recorded at the parent's time (section 3).
    peps.write("pep-0002.txt", pep(2, "Final"))
    skewed = peps.commit("clock skew", noon("2019-06-01"))
    eips.write("EIPS/eip-3.md", eip(3, "Draft"))
    eip_commit = eips.commit("add EIP 3", noon("2020-01-01"))

    found = stream_of(world, "H+")
    # Sorted by recorded time, then repository, then chain position, then proposal id; the
    # proposal ids are compared as strings, so PEP-10 comes before PEP-2.
    assert [(e.proposal, e.kind, e.commit) for e in found] == [
        ("PEP-1", "create", first),
        ("EIP-3", "create", eip_commit),
        ("PEP-10", "create", both),
        ("PEP-2", "create", both),
        ("PEP-1", "modify", again),
        ("PEP-2", "modify", skewed),
    ]
    assert found[-1].recorded_at == "2020-01-01T12:00:00Z"
    stamps = [e.recorded_at for e in found]
    assert stamps == sorted(stamps)
    assert [e.episode_id for e in found] == [f"{e.commit[:12]}:{e.proposal}" for e in found]
    assert len({e.episode_id for e in found}) == len(found)
    for e in found:
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", e.recorded_at)
        datetime.fromisoformat(e.recorded_at.replace("Z", "+00:00"))


# --- the command ---------------------------------------------------------------------------------


def run_stream(world: World, arm: str, out: Path, data: Path | None = None) -> int:
    data = data or world.root.parent / "data"
    argv = ["stream", "--clones", str(world.root), "--data", str(data), "--arm", arm,
            "--out", str(out)]
    return cli.main(argv)


def small_history(world: World) -> None:
    peps = world["peps"]
    peps.write("pep-0001.txt", pep(1, "Draft", title="Caf\u00e9 \u2014 \u201cquoted\u201d"))
    peps.commit("Add PEP 1", noon("2020-01-01"))
    peps.write("pep-0001.txt", pep(1, "Final", title="Caf\u00e9 \u2014 \u201cquoted\u201d"))
    peps.commit("Finalise PEP 1", noon("2020-02-01"))
    world["EIPs"].write("EIPS/eip-5.md", eip(5, "Draft"))
    world["EIPs"].commit("Add EIP 5", noon("2020-03-01"))


def test_stats(world, tmp_path, capsys):
    small_history(world)
    out = tmp_path / "streams" / "h-plus.jsonl"
    assert run_stream(world, "H+", out) == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").split("\n") if line]
    chars = sum(len(row["text"]) for row in rows)
    # Section 8: the episode count, the characters of episode text, the tokens estimated as the
    # characters divided by four, and the file's SHA-256.
    stats = {"episodes": 3, "chars": chars, "tokens_est": chars // 4,
             "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    assert len(rows) == 3 and chars > 0
    assert json.loads(Path(f"{out}.stats.json").read_text(encoding="utf-8")) == stats
    printed = capsys.readouterr().out
    assert json.loads(printed.strip().splitlines()[-1]) == stats


def test_the_stream_file_is_jsonl_with_the_registered_fields(world, tmp_path):
    small_history(world)
    out = tmp_path / "s.jsonl"
    assert run_stream(world, "H+", out) == 0
    raw = out.read_bytes()
    assert raw.endswith(b"\n") and b"\r" not in raw
    lines = raw.decode("utf-8").split("\n")[:-1]
    assert len(lines) == 3
    for line, episode in zip(lines, stream_of(world, "H+"), strict=True):
        record = json.loads(line)
        assert set(record) == {"episode_id", "corpus", "repo", "proposal", "ref", "recorded_at",
                               "commit", "message", "kind", "text"}
        assert line == dumps(asdict(episode))  # sorted keys, compact, non-ASCII kept
    assert "Caf\u00e9 \u2014" in lines[0]


def test_the_stream_is_deterministic(world, tmp_path):
    small_history(world)
    for arm in ("H+", "H-"):
        a, b = tmp_path / f"{arm}-a.jsonl", tmp_path / f"{arm}-b.jsonl"
        assert run_stream(world, arm, a) == 0 and run_stream(world, arm, b) == 0
        assert a.read_bytes() == b.read_bytes()
        assert Path(f"{a}.stats.json").read_bytes() == Path(f"{b}.stats.json").read_bytes()


def write_gold(data: Path, *records: dict) -> None:
    data.mkdir(parents=True, exist_ok=True)
    (data / "gold.jsonl").write_text(
        "".join(dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def cf_record(pid: str = "PEP-1", ref: str = 'PEP 1 ("Example")') -> dict:
    return {"id": f"CF-counterfactual-{pid}-2020-02-15", "gold": "Rejected",
            "meta": {"ref": ref, "true_status": "Accepted"}}


def test_the_cf_command_reads_its_proposals_from_gold(world, tmp_path):
    peps = world["peps"]
    peps.write("pep-0001.txt", pep(1, "Accepted"))
    peps.commit("Add PEP 1", noon("2020-01-01"))
    peps.write("pep-0001.txt", pep(1, "Final"))
    peps.commit("Finalise PEP 1", noon("2020-02-01"))
    peps.write("pep-0002.txt", pep(2, "Accepted"))
    peps.commit("Add PEP 2", noon("2020-03-01"))
    data = tmp_path / "data"
    write_gold(
        data,
        {"id": "F1-change-PEP-2-2020-03-05", "gold": "Accepted", "meta": {"ref": "PEP 2"}},
        cf_record(),
    )
    out = tmp_path / "cf.jsonl"
    assert run_stream(world, "CF", out, data) == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").split("\n") if line]
    by_id = {(r["proposal"], r["kind"]): r for r in rows}
    assert by_id[("PEP-1", "create")]["text"] == pep(1, "Rejected")  # Accepted, permuted
    assert by_id[("PEP-1", "modify")]["message"] == UPDATE
    assert by_id[("PEP-2", "create")]["text"] == pep(2, "Accepted")  # not a CF proposal
    assert by_id[("PEP-2", "create")]["message"] == "Add PEP 2"


def test_the_cf_command_counts_the_messages_it_leaves_alone(world, tmp_path, capsys):
    peps, eips = world["peps"], world["EIPs"]
    day = iter(range(1, 400))

    def when() -> str:
        n = next(day)
        return noon(f"2020-{(n - 1) // 28 + 1:02d}-{(n - 1) % 28 + 1:02d}")

    def edit(repo, files, message):
        """Change the body of each file (its status line stays) and commit with ``message``."""
        body = BODY.replace("Body line 0", f"Edit {message[:9]}")
        for path, (build, number, status) in files.items():
            repo.write(path, build(number, status, body=body))
        repo.commit(message, when())

    # --- PEP 1 is a CF proposal; PEP 2 and PEP 3 are not.
    pep1, pep2, pep3 = (pep, 1, "Final"), (pep, 2, "Draft"), (pep, 3, "Draft")
    for number, status in ((1, "Accepted"), (2, "Draft"), (3, "Draft")):
        peps.write(f"pep-000{number}.txt", pep(number, status))
    peps.commit("Add three PEPs", when())
    peps.write("pep-0001.txt", pep(1, "Final"))
    peps.commit("Mark PEP 1 as Final", when())  # a CF status line changes: this commit is rewritten
    edit(peps, {"pep-0002.txt": pep2, "pep-0003.txt": pep3},
         "PEP 1 was Accepted by the BDFL; pep-0001 stays.")  # two episodes, one commit: once
    edit(peps, {"pep-0002.txt": pep2}, "Note that PEP 10 is final")  # another proposal
    edit(peps, {"pep-0002.txt": pep2}, "Finally, PEP 1 is clear")  # Finally is not a status
    edit(peps, {"pep-0002.txt": pep2}, "PEP-1 draft text")  # a status in lower case
    edit(peps, {"pep-0002.txt": pep2}, "Review PEP 1")  # Review is no PEP status
    edit(peps, {"pep-0002.txt": pep2}, "See PEPs 1 and 2: final")  # PEPs 1 is no mention
    edit(peps, {"pep-0002.txt": pep2}, "The :pep:`1` role is Accepted")  # a role is a mention
    edit(peps, {"pep-0001.txt": pep1}, "PEP 1: Accepted wording")  # the CF proposal's own episode
    # --- EIP 5 is a CF proposal; EIP 6 is not.
    eip5, eip6 = (eip, 5, "Last Call"), (eip, 6, "Draft")
    eips.write("EIPS/eip-5.md", eip(5, "Review"))
    eips.write("EIPS/eip-6.md", eip(6, "Draft"))
    eips.commit("Add two EIPs", when())
    eips.write("EIPS/eip-5.md", eip(5, "Last Call"))
    eips.commit("Move EIP-5 to Last Call", when())  # rewritten
    edit(eips, {"EIPS/eip-6.md": eip6}, "EIP-5 is in last call now")  # a phrase, in lower case
    edit(eips, {"EIPS/eip-6.md": eip6}, "ERC-5 Review notes")  # ERC-5 is the same number
    edit(eips, {"EIPS/eip-6.md": eip6}, "PEP 5 final")  # a PEP, not EIP 5
    edit(eips, {"EIPS/eip-6.md": eip6}, "EIP-50 final")  # another number
    edit(eips, {"EIPS/eip-6.md": eip6}, "eip-5.md: Stagnant")
    edit(eips, {"EIPS/eip-6.md": eip6}, "EIP 5: accepted")  # an earlier-process value, kept
    edit(eips, {"EIPS/eip-6.md": eip6}, "EIP-5 finalised")  # not a whole word
    edit(eips, {"EIPS/eip-6.md": eip6}, "EIP-5 reached Last\n  Call")  # a wrapped phrase counts
    edit(eips, {"EIPS/eip-6.md": eip6}, "EIP-5 last-call")  # a hyphen is not a space
    edit(eips, {"EIPS/eip-5.md": eip5}, "EIP-5 has no status word")  # a mention only
    data = tmp_path / "data"
    write_gold(data, cf_record("PEP-1"), cf_record("EIP-5", 'EIP-5 ("Example")'))
    out = tmp_path / "cf.jsonl"
    assert run_stream(world, "CF", out, data) == 0

    counts = json.loads(Path(f"{out}.cf_messages.json").read_text(encoding="utf-8"))
    # Section 7 counts commits, each once, however many of its files changed.
    assert counts == {"canary": CANARY, "PEP-1": 4, "EIP-5": 5}
    # The stats file holds the stream's stats only; the other arms write no such file.
    assert set(json.loads(Path(f"{out}.stats.json").read_text(encoding="utf-8"))) == {
        "episodes", "chars", "tokens_est", "sha256"}
    for arm in ("H+", "H-"):
        plain = tmp_path / f"{arm}.jsonl"
        assert run_stream(world, arm, plain) == 0
        assert not Path(f"{plain}.cf_messages.json").exists()


def test_the_hminus_command_writes_the_message_cues(world, tmp_path):
    """Section 12: per proposal, the [recorded_at, status] pairs of the messages kept in the H-
    stream that mention it (in its own corpus) and contain a canonical status value."""
    peps, eips = world["peps"], world["EIPs"]
    peps.write("pep-0001.txt", pep(1, "Draft"))
    peps.write("pep-0002.txt", pep(2, "Draft"))
    peps.commit("Add PEP 1 and PEP 2", noon("2020-01-01"))  # no status value
    peps.write("pep-0001.txt", pep(1, "Final"))
    peps.commit("Mark PEP 1 as Final", noon("2020-02-01"))
    peps.write("pep-0001.txt", pep(1, "Final", title="Edited"))
    peps.write("pep-0002.txt", pep(2, "Rejected"))
    peps.commit("PEP 1 and pep-0002: draft, then Rejected", noon("2020-03-01"))  # two files
    peps.write("pep-0002.txt", pep(2, "Rejected", title="Edited"))
    peps.commit("Finally, PEP 2 is clear", noon("2020-04-01"))  # Finally is no status value
    peps.write("pep-0002.txt", pep(2, "Rejected", title="Edited again"))
    peps.commit("PEP 99 is Final", noon("2020-04-02"))  # no proposal 99 in the corpus
    eips.write("EIPS/eip-5.md", eip(5, "Review"))
    eips.commit("Move EIP-5 to Last\n  Call", noon("2020-05-01"))
    eips.write("EIPS/eip-5.md", eip(5, "Review", title="Edited"))
    # Read in each corpus: ERC-5 is proposal 5 with Review and Final (both Ethereum values), and
    # PEP 1 is named with Final; the message is one cue source for both.
    eips.commit("ERC-5 review notes, as PEP 1 says (Final)", noon("2020-06-01"))
    out = tmp_path / "h-minus.jsonl"
    assert run_stream(world, "H-", out) == 0
    cues = json.loads(Path(f"{out}.message_cues.json").read_text(encoding="utf-8"))
    assert cues == {
        "canary": CANARY,
        "PEP-1": [["2020-02-01T12:00:00Z", "Final"], ["2020-03-01T12:00:00Z", "Draft"],
                  ["2020-03-01T12:00:00Z", "Rejected"], ["2020-06-01T12:00:00Z", "Final"]],
        "PEP-2": [["2020-03-01T12:00:00Z", "Draft"], ["2020-03-01T12:00:00Z", "Rejected"]],
        "EIP-5": [["2020-05-01T12:00:00Z", "Last Call"], ["2020-06-01T12:00:00Z", "Final"],
                  ["2020-06-01T12:00:00Z", "Review"]],
    }
    plain = tmp_path / "h-plus.jsonl"  # only the H- arm writes them
    assert run_stream(world, "H+", plain) == 0
    assert not Path(f"{plain}.message_cues.json").exists()


@pytest.mark.parametrize(
    "records",
    [
        [],  # no CF items at all
        [{"id": "F1-change-PEP-1-2020-03-05", "gold": "Draft", "meta": {"ref": "PEP 1"}}],
        [cf_record("PEP-99", "PEP 99")],  # a proposal that is not in the corpus
        [{"id": "CF-counterfactual-nonsense", "gold": "Draft", "meta": {"ref": "x"}}],
        [{"id": "CF-counterfactual-PEP-1-2020-02-15", "gold": "Draft", "meta": {}}],  # no ref
        [cf_record(), cf_record(ref='PEP 1 ("Another title")')],  # two refs for one proposal
    ],
)
def test_the_cf_command_refuses_a_gold_file_without_usable_cf_items(
    world, tmp_path, records, capsys
):
    world["peps"].write("pep-0001.txt", pep(1, "Final"))
    world["peps"].commit("add", noon("2020-01-01"))
    data = tmp_path / "data"
    write_gold(data, *records)
    out = tmp_path / "cf.jsonl"
    assert run_stream(world, "CF", out, data) == 2
    assert not out.exists()
    assert "asofbench stream" in capsys.readouterr().err


def test_the_cf_command_refuses_a_gold_file_that_is_not_json(world, tmp_path, capsys):
    world["peps"].write("pep-0001.txt", pep(1, "Final"))
    world["peps"].commit("add", noon("2020-01-01"))
    data = tmp_path / "data"
    data.mkdir()
    (data / "gold.jsonl").write_text(dumps(cf_record()) + "\nthis is not json\n", encoding="utf-8")
    out = tmp_path / "cf.jsonl"
    assert run_stream(world, "CF", out, data) == 2
    assert not out.exists() and "line 2" in capsys.readouterr().err


def test_the_cf_command_needs_gold_jsonl(world, tmp_path, capsys):
    world["peps"].write("pep-0001.txt", pep(1, "Final"))
    world["peps"].commit("add", noon("2020-01-01"))
    out = tmp_path / "cf.jsonl"
    assert run_stream(world, "CF", out, tmp_path / "empty-data") == 2
    assert not out.exists() and "gold.jsonl" in capsys.readouterr().err


def test_the_command_checks_the_manifest_pins(world, tmp_path, capsys):
    small_history(world)
    data = tmp_path / "data"
    data.mkdir()
    pins = world.load().pins
    (data / "MANIFEST.json").write_text(json.dumps({"pins": pins}), encoding="utf-8")
    assert run_stream(world, "H+", tmp_path / "ok.jsonl", data) == 0
    (data / "MANIFEST.json").write_text(
        json.dumps({"pins": {**pins, "peps": "0" * 40}}), encoding="utf-8"
    )
    assert run_stream(world, "H+", tmp_path / "bad.jsonl", data) == 2
    assert not (tmp_path / "bad.jsonl").exists()
    assert "pins" in capsys.readouterr().err


def test_the_command_refuses_clones_it_cannot_read(world, tmp_path, capsys):
    argv = ["stream", "--clones", str(tmp_path / "nowhere"), "--data", str(tmp_path / "data"),
            "--arm", "H+", "--out", str(tmp_path / "s.jsonl")]
    assert cli.main(argv) == 2
    err = capsys.readouterr().err
    assert err.startswith("asofbench stream: ") and "not a directory" in err
    data = tmp_path / "data"
    data.mkdir()
    (data / "MANIFEST.json").write_text("{truncated", encoding="utf-8")
    small_history(world)
    assert run_stream(world, "H+", tmp_path / "s.jsonl", data) == 2
    assert "MANIFEST.json" in capsys.readouterr().err
    assert not (tmp_path / "s.jsonl").exists()


def test_a_status_word_written_with_a_dotted_or_dotless_i_is_that_status(world, tmp_path):
    """Status values match in any letter case, so FINAL with a dotted I (U+0130) and final with a
    dotless i (U+0131) read as Final in the H- message cues, as they already did in the CF count,
    and do not stop the stream."""
    peps = world["peps"]
    peps.write("pep-0001.txt", pep(1, "Draft"))
    peps.write("pep-0002.txt", pep(2, "Draft"))
    peps.commit("Add PEP 1 and PEP 2", noon("2020-01-01"))
    peps.write("pep-0002.txt", pep(2, "Draft", title="Edited"))
    peps.commit(f"PEP 1 is F{chr(0x130)}NAL", noon("2020-02-01"))
    peps.write("pep-0002.txt", pep(2, "Draft", title="Edited again"))
    peps.commit(f"pep-0001: f{chr(0x131)}nal wording", noon("2020-03-01"))
    out = tmp_path / "h-minus.jsonl"
    assert run_stream(world, "H-", out) == 0
    cues = json.loads(Path(f"{out}.message_cues.json").read_text(encoding="utf-8"))
    assert cues == {"canary": CANARY, "PEP-1": [["2020-02-01T12:00:00Z", "Final"],
                                                ["2020-03-01T12:00:00Z", "Final"]]}
    data = tmp_path / "data"
    write_gold(data, cf_record("PEP-1"))
    cf = tmp_path / "cf.jsonl"
    assert run_stream(world, "CF", cf, data) == 0
    counts = json.loads(Path(f"{cf}.cf_messages.json").read_text(encoding="utf-8"))
    assert counts == {"canary": CANARY, "PEP-1": 2}
