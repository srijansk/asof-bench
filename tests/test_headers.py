"""Reading headers (spec section 4) and removing the withheld fields (section 8)."""

from __future__ import annotations

import pytest

from asofbench.headers import (
    WITHHELD,
    header_span,
    number_list,
    parse_header,
    strip_withheld,
    superseders,
)


def test_parse_pep_header_continuation():
    t = "PEP: 8\nTitle: Style Guide\nAuthor: A <a@x>,\n    B <b@y>\nStatus: Active\n\nBody: no\n"
    h = parse_header("pep", t)
    assert h == {
        "pep": "8",
        "title": "Style Guide",
        "author": "A <a@x>, B <b@y>",
        "status": "Active",
    }


def test_parse_pep_header_crlf_bom():
    assert parse_header("pep", "\ufeffPEP: 1\r\nStatus: Final\r\n\r\nBody\r\n")["status"] == "Final"


def test_parse_pep_header_ends_at_the_first_blank_line():
    t = "PEP: 8\nStatus: Active\n\nTitle: only in the body\nStatus: Rejected\n"
    assert parse_header("pep", t) == {"pep": "8", "status": "Active"}


def test_parse_pep_header_without_a_blank_line_is_the_whole_text():
    assert parse_header("pep", "PEP: 8\nStatus: Active") == {"pep": "8", "status": "Active"}


def test_parse_pep_header_keys_are_case_folded_and_values_unquoted():
    t = 'PEP: 8\nSUPERSEDED-by: 9\nTitle: "Quoted: title"\nPost-History:\n\nBody\n'
    assert parse_header("pep", t) == {
        "pep": "8",
        "superseded-by": "9",
        "title": "Quoted: title",
        "post-history": "",
    }


def test_a_repeated_key_keeps_its_first_value():
    pep = "PEP: 230\nStatus: Draft\nStatus: Incomplete\n\nBody\n"
    assert parse_header("pep", pep)["status"] == "Draft"
    eth = "---\neip: 2544\nstatus: Draft\nstatus: Review\n---\nBody\n"
    assert parse_header("eth", eth)["status"] == "Draft"


def test_parse_pep_header_continuation_of_an_empty_field():
    t = "PEP: 8\nPost-History:\n    01-Jan-2020,\n    02-Feb-2020\n\nBody\n"
    assert parse_header("pep", t)["post-history"] == "01-Jan-2020, 02-Feb-2020"


# pep-0263.txt at 1e0563b4e6 and pep-3118.txt at 5f33b4b632 begin with a blank line (section 4).
BLANK_FIRST = "PEP: 263\nTitle: T\nStatus: Final\nType: Standards Track\n\nStatus: in body\n"
BOM = chr(0xFEFF)


def test_a_pep_header_starts_at_the_first_non_blank_line():
    for lead in ("\n", "\n\n", "\r\n", BOM + "\n", BOM + "\r\n\r\n"):
        t = lead + BLANK_FIRST
        assert parse_header("pep", t) == {
            "pep": "263",
            "title": "T",
            "status": "Final",
            "type": "Standards Track",
        }, repr(lead)
        start, end = header_span("pep", t)
        assert t[start:end] == "PEP: 263\nTitle: T\nStatus: Final\nType: Standards Track\n\n"
        assert t[end:] == "Status: in body\n", repr(lead)


def test_a_pep_header_after_leading_blank_lines_strips_cleanly():
    for lead in ("\n", "\n\n", BOM + "\r\n\r\n"):
        t = lead + BLANK_FIRST
        stripped = strip_withheld("pep", t)
        assert stripped == lead + "PEP: 263\nTitle: T\nType: Standards Track\n\nStatus: in body\n"
        assert "status" not in parse_header("pep", stripped)


def test_a_pep_text_with_no_non_blank_line_has_no_header():
    for t in ("", "\n", "\n\n", "\r\n \n", BOM, BOM + "\n\n"):
        start, end = header_span("pep", t)
        assert start == end == (1 if t.startswith(BOM) else 0), repr(t)
        assert t[end:] == t[start:]  # the body is the whole text after any byte-order mark
        assert parse_header("pep", t) == {}
        assert strip_withheld("pep", t) == t


def test_parse_eip_yaml():
    t = (
        "---\neip: 20\ntitle: 'Token Standard'\nstatus: Final\nrequires: 165, 721\n---\n"
        "## Abstract\nstatus: no\n"
    )
    h = parse_header("eth", t)
    assert h["status"] == "Final" and h["title"] == "Token Standard" and h["requires"] == "165, 721"


def test_parse_eip_yaml_with_bom_blank_lines_and_double_quotes():
    t = (
        "\ufeff\n\n---\r\neip: 20\r\n"
        'title: "ERC-20: Token Standard"\r\nstatus: Final # done\r\n---\r\nBody\r\n'
    )
    h = parse_header("eth", t)
    assert h == {"eip": "20", "title": "ERC-20: Token Standard", "status": "Final # done"}


def test_parse_eip_yaml_keeps_empty_values():
    t = "---\neip: 20\nrequires:\nstatus: Draft\n---\nBody\n"
    assert parse_header("eth", t) == {"eip": "20", "requires": "", "status": "Draft"}


def test_parse_eip_indented_block():
    t = (
        "### Title\n\n      EIP: 101\n      Title: Serenity\n      Status: Active\n\n"
        "### Specification\nStatus: no\n"
    )
    assert parse_header("eth", t)["status"] == "Active"


def test_parse_eip_mediawiki_pre():
    t = "<pre>\n  EIP: 1\n  Title: EIP Purpose\n  Status: Draft\n</pre>\n==What==\n"
    assert parse_header("eth", t)["status"] == "Draft"


def test_parse_eip_block_must_start_in_the_first_40_lines():
    late = "\n" * 40 + "Status: Draft\n"
    assert parse_header("eth", late) == {}
    just_in_time = "\n" * 39 + "Status: Draft\n"
    assert parse_header("eth", just_in_time) == {"status": "Draft"}


def test_parse_eip_without_a_header():
    assert parse_header("eth", "Just prose.\nNo header here.\n") == {}
    assert parse_header("eth", "") == {}
    assert parse_header("pep", "") == {}


def test_unknown_family_is_an_error():
    with pytest.raises(ValueError, match="family"):
        parse_header("rfc", "x: y\n")


def test_header_span_pep():
    t = "PEP: 8\nStatus: Active\n\nBody\n"
    start, end = header_span("pep", t)
    assert t[start:end] == "PEP: 8\nStatus: Active\n\n"
    assert t[end:] == "Body\n"


def test_header_span_pep_crlf_bom():
    t = "\ufeffPEP: 1\r\nStatus: Final\r\n\r\nBody\r\n"
    start, end = header_span("pep", t)
    assert start == 1
    assert t[end:] == "Body\r\n"


def test_header_span_yaml():
    t = "---\neip: 20\nstatus: Final\n---\n## Abstract\nstatus: no\n"
    start, end = header_span("eth", t)
    assert t[start:end] == "---\neip: 20\nstatus: Final\n---\n"
    assert t[end:] == "## Abstract\nstatus: no\n"


def test_header_span_indented_block_ends_after_its_blank_line():
    t = "### Title\n\n      EIP: 101\n      Status: Active\n\n### Specification\n"
    start, end = header_span("eth", t)
    assert t[start:end] == "      EIP: 101\n      Status: Active\n\n"
    assert t[end:] == "### Specification\n"


def test_header_span_pre_block_ends_after_its_closing_pre_line():
    t = "<pre>\n  EIP: 1\n  Status: Draft\n</pre>\n==What==\n"
    start, end = header_span("eth", t)
    assert t[start:end] == "  EIP: 1\n  Status: Draft\n</pre>\n"
    assert t[end:] == "==What==\n"


def test_header_span_ends_at_the_end_of_the_text_when_nothing_closes_the_block():
    for family, t in [
        ("pep", "PEP: 8\nStatus: Active"),
        ("pep", "PEP: 8\nStatus: Active\n"),
        ("eth", "EIP: 5\nStatus: Draft"),
        ("eth", "EIP: 5\nStatus: Draft\n"),
        ("eth", "---\neip: 20\nstatus: Final\n"),
    ]:
        start, end = header_span(family, t)
        assert (start, end) == (0, len(t)), (family, t)


def test_header_span_closing_line_is_inside_the_span_in_every_form():
    cases = [
        ("pep", "PEP: 8\nStatus: Active\n\nBody\n", "\n"),
        ("eth", "---\neip: 20\nstatus: Final\n---\nBody\n", "---\n"),
        ("eth", "  EIP: 5\n  Status: Draft\n\nBody\n", "\n"),
        ("eth", "<pre>\n  EIP: 5\n  Status: Draft\n</pre>\nBody\n", "</pre>\n"),
    ]
    for family, text, closing in cases:
        start, end = header_span(family, text)
        assert text[start:end].endswith(closing), (family, text)
        assert text[end:] == "Body\n", (family, text)


def test_header_span_without_a_header():
    assert header_span("eth", "Just prose.\n") == (0, 0)
    assert header_span("pep", "") == (0, 0)


WRAPPED_TITLE = (
    "<pre>\n"
    "  EIP: 196\n"
    "  Title: Precompiled contracts for addition and scalar multiplication on the elliptic\n"
    "    curve alt_bn128\n"
    "  Author: A\n"
    "  Status: Draft\n"
    "  Requires: 5\n"
    "  Type: Standard Track\n"
    "</pre>\n"
    "Body\n"
)


def test_wrapped_title_in_a_pre_block_does_not_end_the_header():
    assert parse_header("eth", WRAPPED_TITLE) == {
        "eip": "196",
        "title": (
            "Precompiled contracts for addition and scalar multiplication on the elliptic"
            " curve alt_bn128"
        ),
        "author": "A",
        "status": "Draft",
        "requires": "5",
        "type": "Standard Track",
    }
    start, end = header_span("eth", WRAPPED_TITLE)
    assert WRAPPED_TITLE[start:end].startswith("  EIP: 196\n")
    assert WRAPPED_TITLE[start:end].endswith("  Type: Standard Track\n</pre>\n")
    assert WRAPPED_TITLE[end:] == "Body\n"
    # Status and Requires come after the wrapped line and are still withheld; the wrapped
    # line stays with its Title.
    assert strip_withheld("eth", WRAPPED_TITLE) == (
        "<pre>\n"
        "  EIP: 196\n"
        "  Title: Precompiled contracts for addition and scalar multiplication on the elliptic\n"
        "    curve alt_bn128\n"
        "  Author: A\n"
        "  Type: Standard Track\n"
        "</pre>\n"
        "Body\n"
    )


COLONLESS = (
    "    EIP: 211\n"
    "    Title: New opcodes\n"
    "    Type: Standard Track\n"
    "    Category Core\n"
    "    Status: Draft\n"
    "    Requires: 5\n"
    "    Created: 2017-02-13\n"
    "\n"
    "## Abstract\n"
)


def test_a_line_without_a_colon_continues_the_field_above():
    assert parse_header("eth", COLONLESS) == {
        "eip": "211",
        "title": "New opcodes",
        "type": "Standard Track Category Core",
        "status": "Draft",
        "requires": "5",
        "created": "2017-02-13",
    }
    start, end = header_span("eth", COLONLESS)
    assert COLONLESS[start:end].endswith("    Created: 2017-02-13\n\n")
    assert COLONLESS[end:] == "## Abstract\n"
    assert strip_withheld("eth", COLONLESS) == (
        "    EIP: 211\n"
        "    Title: New opcodes\n"
        "    Type: Standard Track\n"
        "    Category Core\n"
        "    Created: 2017-02-13\n"
        "\n"
        "## Abstract\n"
    )


LATE_BLOCK = (
    "\n" * 37
    + "EIP: 9\nTitle: T\nAuthor: A\nStatus: Draft\nType: Meta\nCreated: 2016-01-01\n"
    + "\nBody\n"
)


def test_a_block_that_starts_late_runs_past_line_40():
    assert LATE_BLOCK.split("\n")[37] == "EIP: 9"  # the block starts on line 38
    assert parse_header("eth", LATE_BLOCK) == {
        "eip": "9",
        "title": "T",
        "author": "A",
        "status": "Draft",
        "type": "Meta",
        "created": "2016-01-01",
    }
    start, end = header_span("eth", LATE_BLOCK)
    assert LATE_BLOCK[start:end].startswith("EIP: 9\n")
    assert LATE_BLOCK[end:] == "Body\n"
    assert strip_withheld("eth", LATE_BLOCK) == LATE_BLOCK.replace("Status: Draft\n", "")


WRAPPED_FRONT_MATTER = (
    "---\n"
    "eip: 20\n"
    "title: A very long title that was\n"
    "wrapped by hand\n"
    "status: Final\n"
    "requires: 5\n"
    "---\n"
    "Body\n"
)


def test_an_unindented_wrapped_line_in_front_matter_continues_the_field_above():
    assert parse_header("eth", WRAPPED_FRONT_MATTER) == {
        "eip": "20",
        "title": "A very long title that was wrapped by hand",
        "status": "Final",
        "requires": "5",
    }
    start, end = header_span("eth", WRAPPED_FRONT_MATTER)
    assert WRAPPED_FRONT_MATTER[end:] == "Body\n"
    assert strip_withheld("eth", WRAPPED_FRONT_MATTER) == (
        "---\neip: 20\ntitle: A very long title that was\nwrapped by hand\n---\nBody\n"
    )


PROSE_MID_BLOCK = (
    "EIP: 5\n"
    "Title: T\n"
    "Status: Draft\n"
    "this wraps Status\n"
    "Created: 2016-01-01\n"
    "\n"
    "Status: in body\n"
)


def test_a_prose_line_in_the_middle_of_a_block_continues_the_field_above():
    h = parse_header("eth", PROSE_MID_BLOCK)
    assert h == {
        "eip": "5",
        "title": "T",
        "status": "Draft this wraps Status",
        "created": "2016-01-01",
    }
    start, end = header_span("eth", PROSE_MID_BLOCK)
    assert PROSE_MID_BLOCK[end:] == "Status: in body\n"
    assert strip_withheld("eth", PROSE_MID_BLOCK) == (
        "EIP: 5\nTitle: T\nCreated: 2016-01-01\n\nStatus: in body\n"
    )


def test_a_wrapped_withheld_field_is_removed_with_its_continuation_lines():
    block = "  EIP: 5\n  Requires: 1,\n    2,\n    3\n  Status: Draft\n  Title: T\n\nBody\n"
    assert parse_header("eth", block)["requires"] == "1, 2, 3"
    assert strip_withheld("eth", block) == "  EIP: 5\n  Title: T\n\nBody\n"
    front_matter = "---\neip: 20\nrequires: 165,\n721\nstatus: Final\ntitle: T\n---\nBody\n"
    assert parse_header("eth", front_matter)["requires"] == "165, 721"
    assert strip_withheld("eth", front_matter) == "---\neip: 20\ntitle: T\n---\nBody\n"
    pep = "PEP: 8\nRequires: 1,\n    2\nTitle: T\n\nBody\n"
    assert strip_withheld("pep", pep) == "PEP: 8\nTitle: T\n\nBody\n"


def test_a_withheld_field_after_a_wrapped_line_is_still_removed():
    block = "EIP: 5\nTitle: a title\n  that wraps\nStatus: Draft\nType: Meta\n\nBody\n"
    assert parse_header("eth", block)["title"] == "a title that wraps"
    assert strip_withheld("eth", block) == (
        "EIP: 5\nTitle: a title\n  that wraps\nType: Meta\n\nBody\n"
    )


def test_an_empty_valued_key_inside_a_block_is_a_key_line():
    t = "EIP: 5\nRequires:\nStatus: Draft\n\nBody\n"
    assert parse_header("eth", t) == {"eip": "5", "requires": "", "status": "Draft"}
    assert strip_withheld("eth", t) == "EIP: 5\n\nBody\n"


def test_the_block_starts_at_the_first_key_line_that_has_a_value():
    t = "Note:\n\nEIP: 5\nStatus: Draft\n\nBody\n"
    assert parse_header("eth", t) == {"eip": "5", "status": "Draft"}
    start, end = header_span("eth", t)
    assert t[start:end] == "EIP: 5\nStatus: Draft\n\n"


def test_a_closing_pre_line_ends_the_block_even_without_a_blank_line():
    t = "<pre>\n  EIP: 1\n  Status: Draft\n</pre>\nStatus: Final\n"
    assert parse_header("eth", t) == {"eip": "1", "status": "Draft"}
    assert t[header_span("eth", t)[1] :] == "Status: Final\n"
    assert strip_withheld("eth", t) == "<pre>\n  EIP: 1\n</pre>\nStatus: Final\n"


def test_a_code_fence_directly_under_the_header_closes_the_block_before_it():
    t = (
        "```\n"
        "EIP: 5\n"
        "Title: T\n"
        "Created: 2016-01-01\n"
        "Requires: 3\n"
        "```\n"
        "\n"
        "## Abstract\n"
    )
    assert parse_header("eth", t) == {
        "eip": "5",
        "title": "T",
        "created": "2016-01-01",
        "requires": "3",
    }
    start, end = header_span("eth", t)
    assert t[start:end] == "EIP: 5\nTitle: T\nCreated: 2016-01-01\nRequires: 3\n"
    assert t[end:] == "```\n\n## Abstract\n"  # the closing fence is the first line of the body
    # Requires is the last field and is withheld; the fence that follows it stays.
    assert strip_withheld("eth", t) == (
        "```\nEIP: 5\nTitle: T\nCreated: 2016-01-01\n```\n\n## Abstract\n"
    )


def test_a_markdown_heading_directly_under_the_header_closes_the_block_before_it():
    t = "    EIP: 5\n    Title: T\n    Status: Draft\n## Abstract\nText.\n"
    assert parse_header("eth", t) == {"eip": "5", "title": "T", "status": "Draft"}
    start, end = header_span("eth", t)
    assert t[start:end] == "    EIP: 5\n    Title: T\n    Status: Draft\n"
    assert t[end:] == "## Abstract\nText.\n"
    # Status is the last field and is withheld; the heading that follows it stays.
    assert strip_withheld("eth", t) == "    EIP: 5\n    Title: T\n## Abstract\nText.\n"


def test_an_indented_fence_or_a_heading_without_a_space_closes_the_block_too():
    for closer in ("  ```", "\t```python", "  ## Abstract", "#Abstract", "# Title"):
        t = f"EIP: 5\nStatus: Draft\nCreated: 2016-01-01\n{closer}\nBody\n"
        assert parse_header("eth", t)["created"] == "2016-01-01", closer
        start, end = header_span("eth", t)
        assert t[end:] == f"{closer}\nBody\n", closer
        assert strip_withheld("eth", t) == f"EIP: 5\nCreated: 2016-01-01\n{closer}\nBody\n", closer


def test_header_span_ends_before_a_code_fence_or_heading():
    # Unlike a blank line or </pre>, which are inside the span, these start the body.
    for closer in ("```", "## Abstract"):
        text = f"EIP: 5\nStatus: Draft\n{closer}\nBody\n"
        start, end = header_span("eth", text)
        assert text[start:end] == "EIP: 5\nStatus: Draft\n"
        assert text[end:] == f"{closer}\nBody\n"


def test_a_fence_or_hash_inside_a_line_does_not_end_the_block():
    t = "EIP: 5\nTitle: Use `x` # not a heading\nStatus: Draft\nType: Meta ```\n\nBody\n"
    assert parse_header("eth", t) == {
        "eip": "5",
        "title": "Use `x` # not a heading",
        "status": "Draft",
        "type": "Meta ```",
    }
    assert header_span("eth", t)[1] == len(t) - len("Body\n")


def test_a_prose_line_directly_under_the_last_field_continues_it_until_the_blank_line():
    # Only a blank line, </pre>, a code fence or a heading closes the block.
    t = "EIP: 5\nStatus: Draft\nCreated: 2016-01-01\nSome prose.\n\nBody\n"
    assert parse_header("eth", t)["created"] == "2016-01-01 Some prose."
    assert t[header_span("eth", t)[1] :] == "Body\n"


def test_a_fence_or_heading_inside_front_matter_does_not_end_it():
    t = "---\neip: 20\nstatus: Final\n# note\n```\n---\nBody\n"
    assert parse_header("eth", t) == {"eip": "20", "status": "Final # note ```"}
    assert t[header_span("eth", t)[1] :] == "Body\n"
    assert strip_withheld("eth", t) == "---\neip: 20\n---\nBody\n"


def test_a_fence_or_heading_line_inside_a_pep_header_is_skipped_not_a_closer():
    t = "PEP: 8\n## junk\nStatus: Active\n```\nTitle: T\n\nBody\n"
    assert parse_header("pep", t) == {"pep": "8", "status": "Active", "title": "T"}
    assert t[header_span("pep", t)[1] :] == "Body\n"


QUALIFIED_FRONT_MATTER = (
    "---\n"
    "eip: 2711\n"
    "created: 2020-01-01\n"
    "requires (*optional): 2718\n"
    "title: T\n"
    "---\n"
    "Body\n"
)

QUALIFIED_BLOCK = (
    "    EIP: 2711\n"
    "    Created: 2020-01-01\n"
    "    Requires (*optional): 2718\n"
    "    Title: T\n"
    "\n"
    "Body\n"
)


def test_a_qualified_key_is_recorded_under_its_full_name_and_never_as_the_plain_field():
    for text in (QUALIFIED_FRONT_MATTER, QUALIFIED_BLOCK):
        h = parse_header("eth", text)
        assert h["requires (*optional)"] == "2718"
        assert "requires" not in h
        assert h["created"] == "2020-01-01"  # the qualified line did not continue created
        assert number_list(h.get("requires")) == []  # so it never feeds a requires list
        assert superseders(h) == []


def test_a_qualified_key_after_created_neither_continues_it_nor_survives_stripping():
    assert strip_withheld("eth", QUALIFIED_FRONT_MATTER) == (
        "---\neip: 2711\ncreated: 2020-01-01\ntitle: T\n---\nBody\n"
    )
    assert strip_withheld("eth", QUALIFIED_BLOCK) == (
        "    EIP: 2711\n    Created: 2020-01-01\n    Title: T\n\nBody\n"
    )
    # As the last field of a block, directly under created.
    last = "EIP: 5\nCreated: 2020-01-01\nrequires (*optional): 2718\n\nBody\n"
    assert parse_header("eth", last) == {
        "eip": "5",
        "created": "2020-01-01",
        "requires (*optional)": "2718",
    }
    assert strip_withheld("eth", last) == "EIP: 5\nCreated: 2020-01-01\n\nBody\n"


def test_a_qualified_key_is_case_folded_and_its_spaces_are_single():
    for line, key, value in [
        ("Requires (*Optional): 5", "requires (*optional)", "5"),
        ("requires(*optional): 5", "requires (*optional)", "5"),
        ("requires   (*optional   only)  : 5", "requires (*optional only)", "5"),
        ("REQUIRES ( *Optional ): 5", "requires (*optional)", "5"),
        (
            "Category (*only required for Standard Track): Core",
            "category (*only required for standard track)",
            "Core",
        ),
    ]:
        assert parse_header("eth", f"---\neip: 1\n{line}\n---\n") == {"eip": "1", key: value}, line


def test_a_plain_key_and_its_qualified_form_are_two_fields():
    for text in (
        "---\neip: 1\nrequires: 5\nrequires (*optional): 6\n---\n",
        "---\neip: 1\nrequires (*optional): 6\nrequires: 5\n---\n",
        "EIP: 1\nrequires: 5\nrequires (*optional): 6\n\n",
    ):
        h = parse_header("eth", text)
        assert h["requires"] == "5" and h["requires (*optional)"] == "6", text
        assert number_list(h["requires"]) == [5]


def test_strip_withheld_removes_a_qualified_form_of_every_withheld_name():
    for name in WITHHELD:
        front_matter = f"---\neip: 1\n{name} (*optional): x\ntitle: T\n---\nBody\n"
        assert strip_withheld("eth", front_matter) == "---\neip: 1\ntitle: T\n---\nBody\n", name
        block = f"EIP: 1\n  {name.title()} (see below): x\nTitle: T\n\nBody\n"
        assert strip_withheld("eth", block) == "EIP: 1\nTitle: T\n\nBody\n", name
        pep = f"PEP: 1\n{name} (opt): x\nTitle: T\n\nBody\n"
        assert strip_withheld("pep", pep) == "PEP: 1\nTitle: T\n\nBody\n", name


def test_a_qualified_key_that_is_not_withheld_stays_and_is_not_its_plain_field():
    t = (
        "---\neip: 1\ncategory (*only required for Standard Track): Core\nstatus: Draft\n---\n"
        "Body\n"
    )
    h = parse_header("eth", t)
    assert h["category (*only required for standard track)"] == "Core"
    assert "category" not in h
    assert strip_withheld("eth", t) == (
        "---\neip: 1\ncategory (*only required for Standard Track): Core\n---\nBody\n"
    )


def test_a_wrapped_qualified_withheld_field_is_removed_with_its_continuation_lines():
    front_matter = "---\neip: 1\nrequires (*optional): 165,\n721\ntitle: T\n---\nBody\n"
    assert parse_header("eth", front_matter)["requires (*optional)"] == "165, 721"
    assert strip_withheld("eth", front_matter) == "---\neip: 1\ntitle: T\n---\nBody\n"
    block = "  EIP: 1\n  Requires (*optional): 165,\n    721\n  Title: T\n\nBody\n"
    assert parse_header("eth", block)["requires (*optional)"] == "165, 721"
    assert strip_withheld("eth", block) == "  EIP: 1\n  Title: T\n\nBody\n"


def test_a_qualified_key_line_can_start_a_block():
    t = "Note:\n\nrequires (*optional): 5\nStatus: Draft\n\nBody\n"
    assert parse_header("eth", t) == {"requires (*optional)": "5", "status": "Draft"}
    start, end = header_span("eth", t)
    assert t[start:end] == "requires (*optional): 5\nStatus: Draft\n\n"


def test_a_qualified_key_in_a_pep_header():
    t = "PEP: 8\nRequires (optional): 5\nStatus: Active\n\nBody\n"
    assert parse_header("pep", t) == {"pep": "8", "requires (optional)": "5", "status": "Active"}
    assert strip_withheld("pep", t) == "PEP: 8\n\nBody\n"


def test_a_parenthesis_that_is_not_a_qualifier_does_not_make_a_key():
    # The parenthetical has to sit between the name and the colon.
    t = (
        "EIP: 5\nTitle: Foo (bar): baz\nStatus: Draft\n(see above): it wraps the status\n"
        "\nBody\n"
    )
    assert parse_header("eth", t) == {
        "eip": "5",
        "title": "Foo (bar): baz",
        "status": "Draft (see above): it wraps the status",
    }
    wrapped = "EIP: 5\nTitle: A title\nTitle (draft) of the thing: x\n\nBody\n"
    assert parse_header("eth", wrapped) == {
        "eip": "5",
        "title": "A title Title (draft) of the thing: x",
    }
    # A key with a space but no parenthesis is still not a key.
    spaced = "---\neip: 1\nauthor: A\ndiscussion to: https://example.org/1\n---\n"
    assert parse_header("eth", spaced) == {"eip": "1", "author": "A discussion to: https://example.org/1"}


def test_blank_lines_in_front_matter_belong_to_no_field():
    t = "---\neip: 20\nstatus: Final\n\ntitle: T\n---\nBody\n"
    assert parse_header("eth", t) == {"eip": "20", "status": "Final", "title": "T"}
    assert strip_withheld("eth", t) == "---\neip: 20\n\ntitle: T\n---\nBody\n"


def test_front_matter_that_is_never_closed_runs_to_the_end_of_the_text():
    t = "---\neip: 20\nstatus: Final\n"
    assert parse_header("eth", t) == {"eip": "20", "status": "Final"}
    assert header_span("eth", t) == (0, len(t))
    assert strip_withheld("eth", t) == "---\neip: 20\n"


def test_number_list():
    assert number_list("165, 721") == [165, 721]
    assert number_list("EIP-20,ERC-721 1155") == [20, 721, 1155]
    assert number_list("[20, 165]") == [20, 165]
    assert number_list("") == []
    assert number_list(None) == []
    assert number_list("n/a") == []


def test_number_list_order_duplicates_and_padding():
    assert number_list("0008, 257, 8,\n  257 , 3000") == [8, 257, 3000]
    assert number_list("PEP 8; PEP 9") == [8, 9]
    # Only the digits of each item are kept, as registered, so an item with two numbers fuses them.
    assert number_list("1-2, 3.4") == [12, 34]


def test_superseders_reads_replaced_by():
    header = {"replaced-by": "3000", "superseded-by": "3100"}
    assert set(superseders(header)) == {3000, 3100}
    assert superseders({"superseded-by": "5, 6", "replaced-by": "6"}) == [5, 6]
    assert superseders({"status": "Final"}) == []


def test_withheld_fields():
    # Section 8, in its order: last-call-deadline's earlier names follow it.
    assert WITHHELD == (
        "status",
        "resolution",
        "accepted",
        "superseded-by",
        "replaced-by",
        "replaces",
        "requires",
        "require",
        "dependencies",
        "last-call-deadline",
        "review-period-end",
        "review-period-ends",
        "review-end-date",
        "withdrawal-reason",
    )


def test_strip_withheld_removes_the_earlier_and_other_names_of_withheld_fields():
    # Shapes the corpora use: the review period's three earlier names of last-call-deadline,
    # require and dependencies for requires, and a PEP's Accepted line (a dated decision link).
    eth = (
        "---\neip: 152\ntitle: T\nstatus: Last Call\nreview-period-end: 2019-12-02\n"
        "review-period-ends: 2019-12-12\nreview-end-date: 2018-11-06\nrequire: 1066\n"
        "dependencies: 165,\n  721\ncreated: 2019-01-01\n---\nreview-period-end: in body\n"
    )
    assert strip_withheld("eth", eth) == (
        "---\neip: 152\ntitle: T\ncreated: 2019-01-01\n---\nreview-period-end: in body\n"
    )
    block = "  EIP: 9\n  Review-Period-End: 2019-12-02\n  Title: T\n\nBody\n"
    assert strip_withheld("eth", block) == "  EIP: 9\n  Title: T\n\nBody\n"
    pep = (
        "PEP: 3147\nTitle: T\nStatus: Final\nAccepted: http://mail.python.org/pipermail/\n"
        "    python-dev/2010-April/099414.html\nCreated: 16-Dec-2009\n\nAccepted: in body\n"
    )
    assert strip_withheld("pep", pep) == (
        "PEP: 3147\nTitle: T\nCreated: 16-Dec-2009\n\nAccepted: in body\n"
    )


def test_strip_withheld_pep():
    t = (
        "PEP: 8\n"
        "Status: Final\n"
        "Resolution: https://mail.python.org/archives/list/\n"
        "    python-dev@python.org/message/ABC/\n"
        "Superseded-By: 9\n"
        "Title: Style Guide\n"
        "\n"
        "Status: in body\n"
    )
    assert strip_withheld("pep", t) == "PEP: 8\nTitle: Style Guide\n\nStatus: in body\n"


def test_strip_withheld_eip_yaml():
    t = (
        "---\n"
        "eip: 20\n"
        "title: Token Standard\n"
        "author: A (@a)\n"
        "status: Final\n"
        "last-call-deadline: 2020-01-01\n"
        "type: Standards Track\n"
        "category: ERC\n"
        "created: 2015-11-19\n"
        "requires: 165\n"
        "withdrawal-reason: none\n"
        "---\n"
        "## Abstract\n"
        "status: body\n"
        "requires: body\n"
    )
    assert strip_withheld("eth", t) == (
        "---\n"
        "eip: 20\n"
        "title: Token Standard\n"
        "author: A (@a)\n"
        "type: Standards Track\n"
        "category: ERC\n"
        "created: 2015-11-19\n"
        "---\n"
        "## Abstract\n"
        "status: body\n"
        "requires: body\n"
    )


def test_strip_withheld_eip_yaml_removes_continuation_lines():
    t = "---\neip: 20\nrequires: 165,\n  721\ntitle: T\n---\nBody\n"
    assert strip_withheld("eth", t) == "---\neip: 20\ntitle: T\n---\nBody\n"


def test_strip_withheld_old_eip_blocks():
    block = (
        "### Title\n\n      EIP: 101\n      Status: Active\n      Requires: 86\n"
        "      Title: T\n\n### Body\nStatus: no\n"
    )
    assert strip_withheld("eth", block) == (
        "### Title\n\n      EIP: 101\n      Title: T\n\n### Body\nStatus: no\n"
    )
    pre = "<pre>\n  EIP: 1\n  Status: Draft\n  Created: 2015-10-27\n</pre>\nStatus: no\n"
    assert strip_withheld("eth", pre) == (
        "<pre>\n  EIP: 1\n  Created: 2015-10-27\n</pre>\nStatus: no\n"
    )


def test_strip_withheld_leaves_everything_else_byte_identical():
    untouched = "\ufeffPEP: 8\r\nTitle: T\r\n\r\nStatus: in body\r\n"
    assert strip_withheld("pep", untouched) == untouched
    crlf = "\ufeffPEP: 8\r\nStatus: Final\r\nTitle: T\r\n\r\nBody\r\n"
    assert strip_withheld("pep", crlf) == "\ufeffPEP: 8\r\nTitle: T\r\n\r\nBody\r\n"
    assert strip_withheld("pep", "") == ""
    assert strip_withheld("eth", "no header\n") == "no header\n"


def test_strip_withheld_is_case_insensitive_and_handles_a_last_line_without_newline():
    assert strip_withheld("pep", "PEP: 8\nSTATUS: Final") == "PEP: 8\n"
    assert strip_withheld("pep", "Status: Final") == ""


def test_strip_withheld_removes_nothing_that_parse_header_cannot_see():
    t = "PEP: 8\nTitle: T\nStatus: Final\nReplaces: 3\nCreated: 1-Jan-2000\n\nBody\n"
    kept = parse_header("pep", strip_withheld("pep", t))
    assert kept == {"pep": "8", "title": "T", "created": "1-Jan-2000"}
    assert not set(kept) & set(WITHHELD)
