"""The independent stream checks of ``streamcheck.py``, tested on small texts.

These checks decide whether the real H- stream leaks and whether the CF stream is the permuted
history, so they get their own tests before the corpus tests rely on them.
"""

from __future__ import annotations

import random

import pytest
from streamcheck import (
    apply_diff,
    diff_header_hits,
    header_zone,
    leaked_lines,
    lines_of,
    permute_expected,
    status_value,
    strip_expected,
)

from asofbench.stream import unified_diff


def split_lines(text: str) -> list[str]:
    return text.split("\n")


# --- apply_diff ----------------------------------------------------------------------------------


def numbered(n: int) -> str:
    return "".join(f"line {i}\n" for i in range(n))


LS = chr(0x2028)  # LINE SEPARATOR: str.splitlines cuts at it, a newline-only splitter does not
AWKWARD = [
    "page one\n\x0c\npage two\npage three\n",  # a form feed on a line of its own
    "page one\x0c\npage two\n",  # a form feed ending a line
    "a\x85b\nc\nd\n",  # NEL inside a line
    f"a{LS}b\nc\nd\n",
    "a\rb\nc\nd\n",  # a lone carriage return inside a line
    "a\x1cb\x1dc\x1ed\nsecond\n",
]


def variants(text: str) -> list[str]:
    """The text as it stands, without its final newline, with CRLF line ends, and both."""
    crlf = text.replace("\n", "\r\n")
    return [text, text.rstrip("\n"), crlf, crlf.rstrip("\r\n")]


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("", ""),
        ("", "a\nb\n"),
        ("a\nb\n", ""),
        ("a\nb\n", "a\nb\n"),
        ("a\nb\n", "b\na\n"),
        (numbered(30), numbered(30).replace("line 15\n", "changed\n")),
        (numbered(30), numbered(30).replace("line 0\n", "")),
        (numbered(30), numbered(30) + "tail\n"),
        (numbered(30), "head\n" + numbered(30)),
        (numbered(30), numbered(30).replace("line 29\n", "")),
        (numbered(40), numbered(40).replace("line 3\n", "x\n").replace("line 36\n", "y\n")),
        ("a\r\nb\r\nc\r\n", "a\r\nB\r\nc\r\n"),
        ("only line\n", "different line\n"),
        # No final newline on either side, on one side, or on both and the same.
        ("a", "a"),
        ("a", "b"),
        ("a\nb", "a\nb\n"),
        ("a\nb\n", "a\nb"),
        ("a\nb", "a\nc"),
        ("x\ny\nz", "X\ny\nz"),
        ("x\ny\nz", "x\ny\nz\nw"),
        ("x\ny\nz", "x\ny"),
        ("", "no newline"),
        ("no newline", ""),
        ("a\r\nb", "a\r\nb\r\n"),
        ("a\r\nb\r\n", "A\r\nb"),
        (numbered(30).rstrip("\n"), numbered(30).replace("line 15\n", "x\n").rstrip("\n")),
    ],
)
def test_apply_diff_inverts_unified_diff(before, after):
    assert apply_diff(before, unified_diff(before, after)) == after


@pytest.mark.parametrize("text", AWKWARD)
def test_apply_diff_inverts_unified_diff_on_awkward_texts(text):
    texts = variants(text)
    for before in texts:
        for after in texts:
            assert apply_diff(before, unified_diff(before, after)) == after
        edited = before.replace("page two", "page 2").replace("c", "C", 1)
        assert apply_diff(before, unified_diff(before, edited)) == edited
        assert apply_diff("", unified_diff("", before)) == before
        assert apply_diff(before, unified_diff(before, "")) == ""


def test_apply_diff_inverts_unified_diff_on_random_edits():
    rng = random.Random(20261027)
    vocabulary = [f"word {i}\n" for i in range(12)] + [
        "\n", "  indented\n", "Status: x\r\n", "\x0c\n", "a\x0cb\n", f"s{LS}t\n", "cr\rhere\n",
    ]
    for _ in range(600):
        before = rng.choices(vocabulary, k=rng.randrange(0, 40))
        after = list(before)
        for _ in range(rng.randrange(0, 6)):
            op = rng.choice(("insert", "delete", "replace"))
            at = rng.randrange(0, len(after) + 1)
            if op == "insert" or not after:
                after.insert(at, rng.choice(vocabulary))
            elif op == "delete":
                del after[min(at, len(after) - 1)]
            else:
                after[min(at, len(after) - 1)] = rng.choice(vocabulary)
        a, b = "".join(before), "".join(after)
        if rng.random() < 0.3:
            a = a.rstrip("\n")  # a side with no final newline
        if rng.random() < 0.3:
            b = b.rstrip("\n")
        assert apply_diff(a, unified_diff(a, b)) == b, (a, b)


def test_lines_of_splits_at_newlines_only():
    assert lines_of("") == []
    assert lines_of("a") == ["a"]
    assert lines_of("a\n") == ["a\n"]
    assert lines_of("a\n\nb") == ["a\n", "\n", "b"]
    assert lines_of("a\r\nb\x0cc\rd\n") == ["a\r\n", "b\x0cc\rd\n"]


def test_apply_diff_is_exact_and_refuses_what_it_cannot_apply():
    with pytest.raises(ValueError, match="does not fit"):
        apply_diff("a\nb\n", "--- a\n+++ b\n@@ -1,2 +1,2 @@\n a\n-c\n+d\n")
    with pytest.raises(ValueError, match="newline"):
        apply_diff("a\n", "--- a\n+++ b\n@@ -1 +1 @@\n-a\n+b")
    # A hunk header whose counts are not those of its lines.
    with pytest.raises(ValueError, match="header says"):
        apply_diff("a\nb\n", "--- a\n+++ b\n@@ -1,3 +1,2 @@\n a\n-b\n+c\n")
    with pytest.raises(ValueError, match="header says"):
        apply_diff("a\nb\n", "--- a\n+++ b\n@@ -1,2 +1,3 @@\n a\n-b\n+c\n")
    # The final newline is part of the text: a removed line without its marker is a line with one.
    with pytest.raises(ValueError, match="does not fit"):
        apply_diff("a\nb", "--- a\n+++ b\n@@ -1,2 +1,2 @@\n a\n-b\n+c\n")
    # The marker says the line has no newline, but the text has one there.
    with pytest.raises(ValueError, match="does not fit"):
        apply_diff(
            "a\nb\n", "--- a\n+++ b\n@@ -1,2 +1,2 @@\n a\n-b\n\\ No newline at end of file\n+c\n"
        )
    # A line left without its newline has to be the last of the text.
    with pytest.raises(ValueError, match="not the last"):
        apply_diff(
            "a\n",
            "--- a\n+++ b\n@@ -1 +1,2 @@\n-a\n+x\n\\ No newline at end of file\n+y\n",
        )
    with pytest.raises(ValueError, match="unexpected"):
        apply_diff("a\n", "--- a\n+++ b\n\\ No newline at end of file\n")
    with pytest.raises(ValueError, match="header"):
        apply_diff("a\n", "not a diff\n")
    # A diff that applies to the text it was made from does not apply to another.
    diff = unified_diff("a\nb\nc\n", "a\nB\nc\n")
    assert apply_diff("a\nb\nc\n", diff) == "a\nB\nc\n"
    with pytest.raises(ValueError):
        apply_diff("a\nb\n", diff)  # the context line c is not there


# --- the header zone and what is withheld --------------------------------------------------------

PEP = (
    "PEP: 7\nTitle: Example\nAuthor: A. Writer\n"
    "Status: Final\nRequires: 2, 3\n    and 4\nSuperseded-By: 9\nCreated: 2020-01-01\n"
    "\nStatus: in the body\n\nBody.\n"
)
FRONT = (
    "---\neip: 7\ntitle: Example\nstatus: Review\nrequires: 1\nrequires (*optional): 2\n"
    "last-call-deadline: 2020-02-01\nauthor: A. Writer\n---\nstatus: in the body\n"
)
RUN = (
    "Some preface\n<pre>\n  eip: 7\n  title: Example\n  status: Final\n  replaces: 3\n"
    "</pre>\n\nBody.\n"
)


def test_header_zone():
    assert header_zone(split_lines(PEP)) == (0, 8)
    assert header_zone(split_lines(FRONT)) == (1, 8)
    assert header_zone(split_lines(RUN)) == (2, 6)
    assert header_zone(split_lines("\nPEP: 1\nTitle: x\n\nbody\n")) == (1, 3)  # leading blank line
    assert header_zone(split_lines("\ufeffPEP: 1\r\nStatus: Draft\r\n\r\nbody")) == (0, 2)
    assert header_zone(split_lines("no header here\nat all\n")) == (0, 0)
    assert header_zone(split_lines("")) == (0, 0)
    # An unterminated block runs to the end of the text.
    assert header_zone(split_lines("---\nkey: v\n")) == (1, 3)


def test_leaked_lines_reads_the_zone_only():
    assert leaked_lines(PEP) == ["Status: Final", "Requires: 2, 3", "Superseded-By: 9"]
    assert leaked_lines(FRONT) == [
        "status: Review",
        "requires: 1",
        "requires (*optional): 2",
        "last-call-deadline: 2020-02-01",
    ]
    assert leaked_lines(RUN) == ["  status: Final", "  replaces: 3"]
    assert leaked_lines("PEP: 1\nTitle: x\n\nStatus: body line\n") == []
    # Spellings headers.py does not know are still suspicious here.
    assert leaked_lines("PEP: 1\nSuperseded By: 3\nWithdrawal_Reason: x\nStatus : Draft\n") == [
        "Superseded By: 3",
        "Withdrawal_Reason: x",
        "Status : Draft",
    ]


def test_leaked_lines_knows_every_name_section_8_withholds():
    # The earlier names of last-call-deadline, require, dependencies and accepted (section 8).
    names = ["review-period-end", "Review-Period-Ends", "review_end_date", "require",
             "Dependencies", "accepted", "requires (*optional)"]
    text = "---\neip: 1\n" + "".join(f"{name}: x\n" for name in names) + "title: T\n---\nBody\n"
    assert leaked_lines(text) == [f"{name}: x" for name in names]
    pep = "PEP: 1\nAccepted: http://example.invalid/\nTitle: T\n\nAccepted: body\n"
    assert leaked_lines(pep) == ["Accepted: http://example.invalid/"]
    assert strip_expected("pep", pep) == "PEP: 1\nTitle: T\n\nAccepted: body\n"


def test_strip_expected():
    assert strip_expected("pep", PEP) == (
        "PEP: 7\nTitle: Example\nAuthor: A. Writer\nCreated: 2020-01-01\n"
        "\nStatus: in the body\n\nBody.\n"
    )
    assert strip_expected("eth", FRONT) == (
        "---\neip: 7\ntitle: Example\nauthor: A. Writer\n---\nstatus: in the body\n"
    )
    assert strip_expected("eth", RUN) == (
        "Some preface\n<pre>\n  eip: 7\n  title: Example\n</pre>\n\nBody.\n"
    )
    # A withheld field that closes the block does not take the closing line with it.
    closing = "---\neip: 1\nstatus: Draft\n---\nbody\n"
    assert strip_expected("eth", closing) == "---\neip: 1\n---\nbody\n"
    crlf = "PEP: 1\r\nStatus: Draft\r\n\r\nbody\r\n"
    assert strip_expected("pep", crlf) == "PEP: 1\r\n\r\nbody\r\n"
    assert strip_expected("pep", "no header here\n") == "no header here\n"
    assert leaked_lines(strip_expected("pep", PEP)) == []
    assert leaked_lines(strip_expected("eth", FRONT)) == []


def test_continuation_lines_follow_each_family_rule():
    # PEP: a line that starts with whitespace continues the field, even if it looks like a key.
    pep = (
        "PEP: 644\nResolution: https://example.org/msg/,\n"
        "    https://bugs.example.org/issue1\nCreated: x\n\nb\n"
    )
    assert strip_expected("pep", pep) == "PEP: 644\nCreated: x\n\nb\n"
    # Ethereum front matter: any line that is no key line continues the field, indented or not.
    front = "---\neip: 1\nrequires: 1,\n  2,\n3\nauthor: A\n---\nb\n"
    assert strip_expected("eth", front) == "---\neip: 1\nauthor: A\n---\nb\n"
    # In the run form blocks are indented, so an indented key line starts a field.
    run = "<pre>\n  eip: 1\n  requires: 1,\n  note: kept\n</pre>\n"
    assert strip_expected("eth", run) == "<pre>\n  eip: 1\n  note: kept\n</pre>\n"


def test_a_key_line_with_an_empty_value_starts_a_new_field():
    # PEP 3112 had "Type:" with nothing after it, right after its Status line.
    text = "PEP: 1\nStatus: Accepted\nType:\nCreated: 2007-02-23\n\nbody\n"
    assert strip_expected("pep", text) == "PEP: 1\nType:\nCreated: 2007-02-23\n\nbody\n"


# --- the counterfactual edit ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("family", "value", "expected"),
    [
        ("pep", "Accepted", "Rejected"),
        ("pep", "Rejected", "Accepted"),
        ("pep", "Final", "Withdrawn"),
        ("pep", "Withdrawn", "Final"),
        ("pep", "Active", "Deferred"),
        ("pep", "Deferred", "Active"),
        ("pep", "Provisional", "Superseded"),
        ("pep", "Superseded", "Provisional"),
        ("pep", "Draft", "Draft"),
        ("pep", "Approved", "Rejected"),  # the legacy name is read first
        ("pep", "Postponed", "Active"),
        ("pep", "Replaced", "Provisional"),
        ("pep", "Finished", "Withdrawn"),
        ("pep", "Incomplete", "Incomplete"),  # read as Draft, which is not swapped: as written
        ("pep", "final # comment", "Withdrawn"),
        ("eth", "Final", "Withdrawn"),
        ("eth", "Review", "Last Call"),
        ("eth", "Last Call", "Review"),
        ("eth", "Stagnant", "Stagnant"),
        ("eth", "Accepted", "Accepted"),  # swapped for PEPs only
    ],
)
def test_permute_expected(family, value, expected):
    tail = "\nStatus: Final in the body\n"
    assert permute_expected(family, f"X: 1\nStatus: {value}\n{tail}") == (
        f"X: 1\nStatus: {expected}\n{tail}"
    )


def test_permute_expected_keeps_indentation_and_line_endings():
    text = "<pre>\r\n  eip: 6\r\n  status: Final\r\n</pre>\r\n"
    assert permute_expected("eth", text) == "<pre>\r\n  eip: 6\r\n  status: Withdrawn\r\n</pre>\r\n"


def test_status_value():
    assert status_value(PEP) == "Final"
    assert status_value(FRONT) == "Review"
    assert status_value(RUN) == "Final"
    assert status_value("PEP: 1\nTitle: x\n\nStatus: body line\n") is None
    assert status_value("PEP: 1\r\nstatus:   Final # done  \r\n\r\nbody") == "Final # done"


# --- header lines in a diff ----------------------------------------------------------------------


def test_diff_header_hits_finds_withheld_lines_in_the_header_zone_only():
    before = "PEP: 1\nStatus: Draft\nRequires: 2\nAuthor: A\n\nStatus: in the body\n" + numbered(10)
    after = before.replace("Draft", "Final").replace("Requires: 2", "Requires: 3").replace(
        "Status: in the body", "Status: still in the body"
    )
    hits = diff_header_hits(unified_diff(before, after), before, after)
    assert hits == ["-Status: Draft", "-Requires: 2", "+Status: Final", "+Requires: 3"]
    # The same change made to the body line alone is no hit: it is outside the header zone.
    body_only = before.replace("Status: in the body", "Status: still in the body")
    assert diff_header_hits(unified_diff(before, body_only), before, body_only) == []
    # Nothing is withheld in a diff of texts that were stripped.
    stripped_before, stripped_after = strip_expected("pep", before), strip_expected("pep", after)
    assert diff_header_hits(
        unified_diff(stripped_before, stripped_after), stripped_before, stripped_after
    ) == []


def test_diff_header_hits_skips_the_no_newline_markers():
    before = "PEP: 1\nStatus: Draft\n\nbody"  # no final newline
    after = "PEP: 1\n\nbody 2"
    diff = unified_diff(before, after)
    assert "\\ No newline at end of file" in diff
    assert diff_header_hits(diff, before, after) == ["-Status: Draft"]


def test_diff_header_hits_places_lines_by_their_own_side():
    # A line added far below a removed header line: only the header one is in a zone.
    before = "PEP: 1\nStatus: Draft\n\n" + numbered(30)
    after = "PEP: 1\n\n" + numbered(30).replace("line 25\n", "Status: moved into the body\n")
    assert diff_header_hits(unified_diff(before, after), before, after) == ["-Status: Draft"]
    # An insertion into an empty file or at the very start is placed by the hunk numbering.
    created = "PEP: 1\nStatus: Draft\n"
    assert diff_header_hits(unified_diff("", created), "", created) == ["+Status: Draft"]
