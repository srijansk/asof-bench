"""Which proposals a body mentions (spec section 7, common rules)."""

from __future__ import annotations

import pytest

from asofbench.mentions import mention_spans, mentions


def test_mentions_pep():
    body = "See PEP 8, PEP-0257, :pep:`484`, :pep:`the typing PEP <483>`, pep-0020.txt and PEP 572"
    assert mentions("pep", body, 572) == {8, 257, 484, 483, 20}


def test_mentions_eth():
    body = "Uses EIP-155 and ERC-20, EIP 712, see ./eip-1559.md and erc-721.md"
    assert mentions("eth", body, 1) == {155, 20, 712, 1559, 721}


def test_own_number_is_ignored_whatever_its_spelling():
    assert mentions("pep", "PEP 8 and PEP-0008 and pep-0008.rst", 8) == set()
    assert mentions("eth", "EIP-20, ERC 20, eip-0020.md", 20) == set()
    assert mentions("eth", "EIP-20 and EIP-21", 20) == {21}


def test_zero_padding_is_optional():
    assert mentions("pep", "PEP 0008 PEP-8 PEP 000009", 1) == {8, 9}
    assert mentions("eth", "EIP-0155 EIP 0000020 ERC-00721", 1) == {155, 20, 721}
    assert mentions("pep", "PEP 0 is the index", 1) == {0}


def test_pep_role_forms():
    assert mentions("pep", ":pep:`8`", 1) == {8}
    assert mentions("pep", ":pep:`0008`", 1) == {8}
    assert mentions("pep", ":pep:`style guide <8>`", 1) == {8}
    assert mentions("pep", ":pep:`8#naming-conventions`", 1) == {8}
    assert mentions("pep", ":pep:`the guide <8>` and :pep:`9`", 1) == {8, 9}
    assert mentions("pep", ":pep:`not a number`", 1) == set()
    assert mentions("pep", ":pep:`own <1>`", 1) == set()


def test_only_the_registered_spellings_count():
    # PEP M, PEP-M, pep-M and the :pep: role; nothing else.
    assert mentions("pep", "pep 8, Pep 9, PEPs 10, PEP8, PEP: 11, PEP #12, PEP  13", 1) == set()
    assert mentions("pep", "XPEP 8 and APEP-9 and pep_10", 1) == set()
    assert mentions("pep", "(PEP 8) [PEP-9] 'PEP 10' PEP 11. pep-12.rst", 1) == {8, 9, 10, 11, 12}
    # EIP-M, ERC-M, EIP M, ERC M, eip-M.md and erc-M.md.
    assert mentions("eth", "eip 8, Eip-9, EIPs 10, EIP8, ERC721, EIP: 11, EIP  12", 1) == set()
    assert mentions("eth", "see eip-1559, but ./eip-1559.mediawiki is not listed", 1) == set()
    assert mentions("eth", "[EIP-1](./eip-1.md) and [ERC](../ERCS/erc-165.md)", 2) == {1, 165}


def test_a_mention_is_a_whole_word():
    # Section 7: the spelling must stand alone, with no letter, digit or underscore next to it.
    assert mentions("pep", "PEP 8a, PEP 8_1, 3PEP 8, my_PEP 8, pep-9x.txt", 1) == set()
    standing_alone = "PEP 8, (PEP-9), PEP 10. And pep-11.rst, PEP 12's"
    assert mentions("pep", standing_alone, 1) == {8, 9, 10, 11, 12}
    assert mentions("eth", "ERC-721A, EIP-155x, EIP-20_, xEIP-1, my_ERC-5", 2) == set()
    standing_alone = "EIP-155's, (ERC-20), EIP 712. And erc-165.md."
    assert mentions("eth", standing_alone, 2) == {155, 20, 712, 165}
    assert mentions("eth", "eip-1559.mdx, neip-5.md, eip_6.md, eip-7.md.", 2) == {7}


def test_each_family_reads_only_its_own_spellings():
    body = "PEP 8, EIP-20, ERC 721, pep-0009.txt, eip-1559.md, :pep:`10`"
    assert mentions("pep", body, 1) == {8, 9, 10}
    assert mentions("eth", body, 1) == {20, 721, 1559}


def test_no_mentions():
    assert mentions("pep", "", 1) == set()
    assert mentions("eth", "Nothing to see here.", 1) == set()


def test_unknown_family_is_an_error():
    with pytest.raises(ValueError, match="family"):
        mentions("rfc", "RFC 2119", 1)


def test_mention_spans_locate_each_mention():
    body = "See PEP 8 and :pep:`style <8>`, then PEP-0257 and PEP 572 (this one)."
    spans = mention_spans("pep", body, 8)
    assert [body[a:b] for a, b in spans] == ["PEP 8", ":pep:`style <8>`"]
    assert [body[a:b] for a, b in mention_spans("pep", body, 257)] == ["PEP-0257"]
    assert mention_spans("pep", body, 9) == []
    eth = "ERC-20 here, eip-20.md there, EIP 21 elsewhere"
    assert [eth[a:b] for a, b in mention_spans("eth", eth, 20)] == ["ERC-20", "eip-20.md"]


def test_mention_spans_agree_with_mentions():
    bodies = {
        "pep": "See PEP 8, PEP-0257, :pep:`484`, :pep:`the typing PEP <483>`, pep-0020.txt, PEP 8",
        "eth": "Uses EIP-155 and ERC-20, EIP 712, see ./eip-1559.md and erc-721.md, EIP-155",
    }
    for family, body in bodies.items():
        found = mentions(family, body, 1)
        for number in range(0, 2000):
            assert bool(mention_spans(family, body, number)) == (number in found), number
