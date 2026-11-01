"""Status normalisation, outcome classes and the CF permutation (spec sections 5, 7 and 12)."""

from __future__ import annotations

import pytest

from asofbench.statuses import (
    CANONICAL,
    CF_PERMUTATION,
    CHOICES,
    MISSING,
    MOVED,
    UNKNOWN,
    normalize_answer_status,
    normalize_status,
    outcome_class,
    permute,
)


def test_sentinels():
    assert (MISSING, UNKNOWN, MOVED) == ("missing", "unknown", "Moved")


def test_normalize_status_pep():
    for raw in ("Incomplete", "Draught"):
        assert normalize_status("pep", raw) == "Draft"
    assert normalize_status("pep", "Approved") == "Accepted"
    for raw in ("Finished", "Complete"):
        assert normalize_status("pep", raw) == "Final"
    assert normalize_status("pep", "Postponed") == "Deferred"
    assert normalize_status("pep", "Withdrawal") == "Withdrawn"
    assert normalize_status("pep", "Replaced") == "Superseded"
    assert normalize_status("pep", " final ") == "Final"
    assert normalize_status("pep", "april fool!") == "April Fool!"
    for raw in ("", None, "<REQUIRED: Draft | Active>"):
        assert normalize_status("pep", raw) == MISSING
    assert normalize_status("pep", "Accepted (pending)") == UNKNOWN


def test_normalize_status_pep_does_not_know_ethereum_values():
    for raw in ("Review", "Last Call", "Living", "Stagnant", "Abandoned", "Moved", "Idea"):
        assert normalize_status("pep", raw) == UNKNOWN


def test_normalize_status_eth():
    assert normalize_status("eth", "last call") == "Last Call"
    assert normalize_status("eth", "Moved") == MOVED
    assert normalize_status("eth", "Stagnant") == "Stagnant"
    assert normalize_status("eth", "Draft # note") == "Draft"
    for kept in ("Abandoned", "Replaced", "Accepted", "Active", "Superseded", "Deferred"):
        assert normalize_status("eth", kept) == kept
    assert normalize_status("eth", "Idea") == UNKNOWN


def test_normalize_status_eth_has_no_pep_legacy_table():
    legacy = (
        "Incomplete", "Draught", "Approved", "Finished", "Complete", "Postponed", "Withdrawal",
    )
    for raw in legacy:
        assert normalize_status("eth", raw) == UNKNOWN


def test_recorded_values_are_not_rewritten_beyond_case_and_surrounding_whitespace():
    assert normalize_status("eth", "Last-Call") == UNKNOWN
    assert normalize_status("eth", "LastCall") == UNKNOWN
    assert normalize_status("pep", "Final.") == UNKNOWN
    assert normalize_status("pep", '"Final"') == UNKNOWN  # quotes are removed by the header parser


def test_inline_comments_are_stripped_in_both_families():
    assert normalize_status("pep", "Final # signed off") == "Final"
    assert normalize_status("eth", "Review\t# bot") == "Review"
    assert normalize_status("eth", "# only a comment") == MISSING
    assert normalize_status("eth", "Draft#attached") == UNKNOWN  # no space, so not a comment


def test_template_placeholders_are_missing_even_with_a_hash_inside():
    assert normalize_status("eth", "<Draft | Review | Last Call | Final>") == MISSING
    assert normalize_status("pep", "<REQUIRED: see PEP #1>") == MISSING


def test_every_canonical_status_normalizes_to_itself():
    for family, statuses in CANONICAL.items():
        for status in statuses:
            assert normalize_status(family, status) == status
            assert normalize_status(family, status.upper()) == status
            assert normalize_status(family, f"  {status.lower()}  ") == status


def test_canonical_statuses():
    assert CANONICAL["pep"] == (
        "Draft",
        "Active",
        "Accepted",
        "Provisional",
        "Deferred",
        "Rejected",
        "Withdrawn",
        "Final",
        "Superseded",
        "April Fool!",
    )
    assert set(CANONICAL["eth"]) == {
        "Draft",
        "Review",
        "Last Call",
        "Final",
        "Stagnant",
        "Withdrawn",
        "Living",
        "Accepted",
        "Active",
        "Abandoned",
        "Superseded",
        "Deferred",
        "Replaced",
    }
    assert MOVED not in CANONICAL["eth"]


def test_choices_order():
    assert CHOICES["pep"] == (
        "Draft",
        "Active",
        "Accepted",
        "Provisional",
        "Deferred",
        "Rejected",
        "Withdrawn",
        "Final",
        "Superseded",
    )
    assert CHOICES["eth"] == (
        "Draft",
        "Review",
        "Last Call",
        "Final",
        "Withdrawn",
        "Living",
        "Accepted",
        "Active",
        "Abandoned",
        "Superseded",
        "Deferred",
        "Replaced",
    )


OUTCOMES = {
    "pep": {
        "adopted": ("Accepted", "Final", "Active", "Provisional"),
        "declined": ("Rejected", "Withdrawn"),
        "superseded": ("Superseded",),
        "open": ("Draft", "Deferred"),
    },
    "eth": {
        "adopted": ("Final", "Living", "Accepted", "Active"),
        "declined": ("Withdrawn", "Abandoned"),
        "superseded": ("Superseded", "Replaced"),
        "open": ("Draft", "Review", "Last Call", "Deferred"),
    },
}


def test_outcome_classes():
    for family, classes in OUTCOMES.items():
        for name, statuses in classes.items():
            for status in statuses:
                assert outcome_class(family, status) == name, (family, status)
        in_table = {s for statuses in classes.values() for s in statuses}
        assert in_table == set(CHOICES[family])  # every status that can be asked has a class
    assert outcome_class("eth", "Stagnant") is None
    assert outcome_class("pep", "April Fool!") is None
    for sentinel in (MISSING, UNKNOWN, MOVED, "absent"):
        assert outcome_class("eth", sentinel) is None
        assert outcome_class("pep", sentinel) is None
    assert outcome_class("pep", "Review") is None  # Ethereum-only values are not PEP statuses


def test_cf_permutation():
    assert CF_PERMUTATION == {
        "pep": {
            "Accepted": "Rejected",
            "Rejected": "Accepted",
            "Final": "Withdrawn",
            "Withdrawn": "Final",
            "Active": "Deferred",
            "Deferred": "Active",
            "Provisional": "Superseded",
            "Superseded": "Provisional",
        },
        "eth": {
            "Final": "Withdrawn",
            "Withdrawn": "Final",
            "Review": "Last Call",
            "Last Call": "Review",
        },
    }


def test_cf_permutation_is_a_plain_dict_of_the_swapped_pairs_only():
    for family, mapping in CF_PERMUTATION.items():
        assert type(mapping) is dict
        assert set(mapping) <= set(CHOICES[family])
        assert all(mapping[mapping[s]] == s != mapping[s] for s in mapping)  # swaps undo themselves
    eth = CF_PERMUTATION["eth"]
    assert "Draft" not in eth and "Living" not in eth and "Stagnant" not in eth
    assert eth.get("Stagnant") is None
    assert eth.get("Stagnant", "other") == "other"
    with pytest.raises(KeyError):
        eth["Stagnant"]


def test_permute():
    for family, mapping in CF_PERMUTATION.items():
        for status, partner in mapping.items():
            assert permute(family, status) == partner
    assert permute("pep", "Accepted") == "Rejected"
    assert permute("eth", "Last Call") == "Review"


def test_permute_leaves_every_other_value_unchanged():
    for family in CF_PERMUTATION:
        for status in CANONICAL[family]:
            if status not in CF_PERMUTATION[family]:
                assert permute(family, status) == status
        for other in (MOVED, UNKNOWN, MISSING, "absent", "Idea", ""):
            assert permute(family, other) == other
    assert permute("pep", "Draft") == "Draft"
    assert permute("pep", "April Fool!") == "April Fool!"
    assert permute("eth", "Living") == "Living"
    assert permute("eth", "Stagnant") == "Stagnant"
    assert permute("eth", "Draft") == "Draft"


def test_permute_undoes_itself():
    for family, statuses in CANONICAL.items():
        for status in statuses:
            assert permute(family, permute(family, status)) == status


def test_normalize_answer_status():
    assert normalize_answer_status("pep", ' "Final." ') == "Final"
    assert normalize_answer_status("eth", "last-call") == "Last Call"
    assert normalize_answer_status("eth", "lastcall") == "Last Call"
    assert normalize_answer_status("pep", "Replaced") == "Superseded"
    assert normalize_answer_status("pep", "finalized") is None


LDQUOTE, RDQUOTE, LSQUOTE, RSQUOTE = (chr(c) for c in (0x201C, 0x201D, 0x2018, 0x2019))


def test_normalize_answer_status_clears_quotes_full_stop_case_and_spacing():
    assert normalize_answer_status("pep", "'accepted'") == "Accepted"
    assert normalize_answer_status("pep", '"Final".') == "Final"
    assert normalize_answer_status("eth", "  LAST \t  CALL. ") == "Last Call"
    assert normalize_answer_status("eth", '"last-call."') == "Last Call"
    assert normalize_answer_status("pep", "Final..") is None  # one full stop only
    assert normalize_answer_status("pep", "april fool!") == "April Fool!"
    assert normalize_answer_status("pep", "Approved") == "Accepted"
    assert normalize_answer_status("pep", "Incomplete") == "Draft"


def test_normalize_answer_status_clears_typographic_quotes():
    assert normalize_answer_status("pep", LDQUOTE + "Final" + RDQUOTE) == "Final"
    assert normalize_answer_status("pep", LSQUOTE + "Final" + RSQUOTE) == "Final"
    assert normalize_answer_status("pep", " " + LDQUOTE + "Final." + RDQUOTE + " ") == "Final"
    assert normalize_answer_status("pep", LDQUOTE + "Final" + RDQUOTE + ".") == "Final"
    assert normalize_answer_status("eth", LDQUOTE + "last-call" + RDQUOTE) == "Last Call"
    assert normalize_answer_status("eth", LSQUOTE + "Last Call" + RSQUOTE + ".") == "Last Call"
    # Any of the six quote characters is cleared from either end, whatever it is paired with.
    assert normalize_answer_status("pep", LDQUOTE + 'Final"') == "Final"
    assert normalize_answer_status("pep", "'Final" + RDQUOTE) == "Final"
    assert normalize_answer_status("pep", '"Final') == "Final"
    # Quotes inside the answer are not surrounding quotes.
    assert normalize_answer_status("pep", "Fi" + RSQUOTE + "nal") is None
    assert normalize_answer_status("pep", '"Final" or "Accepted"') is None


def test_normalize_answer_status_is_family_specific():
    assert normalize_answer_status("pep", "Last Call") is None
    assert normalize_answer_status("pep", "Review") is None
    assert normalize_answer_status("eth", "Approved") is None  # the legacy table is PEP-only
    assert normalize_answer_status("eth", "Moved") is None  # not a status
    assert normalize_answer_status("eth", "Stagnant") == "Stagnant"
    assert normalize_answer_status("eth", "") is None
    assert normalize_answer_status("eth", "   ") is None


def test_unknown_family_is_an_error():
    with pytest.raises(ValueError, match="family"):
        normalize_status("rfc", "Final")
    with pytest.raises(ValueError, match="family"):
        normalize_answer_status("rfc", "Final")
    with pytest.raises(ValueError, match="family"):
        outcome_class("rfc", "Final")
