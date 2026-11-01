"""Which proposals a body mentions (spec section 7, common rules).

A body, the text after the header block, mentions proposal M if it contains, for PEPs, ``PEP M``,
``PEP-M``, ``pep-M`` or a ``:pep:`` role that targets M, and, for Ethereum, ``EIP-M``, ``ERC-M``,
``EIP M``, ``ERC M``, ``eip-M.md`` or ``erc-M.md``, as a whole word and with optional zero
padding. Nothing else counts: not other letter cases, not plurals, not a reference split across
lines. A body's mentions of its own proposal are ignored.
"""

from __future__ import annotations

import re

_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "pep": (
        re.compile(r"\b(?:PEP[ -]|pep-)0*(\d+)\b"),
        # :pep:`8`, :pep:`0008`, :pep:`8#section`, :pep:`a title <8>`
        re.compile(r":pep:`(?:[^`<]*<)?\s*0*(\d+)\s*(?:#[^`>]*)?>?`"),
    ),
    "eth": (
        re.compile(r"\b(?:EIP|ERC)[ -]0*(\d+)\b"),
        re.compile(r"\b(?:eip|erc)-0*(\d+)\.md\b"),
    ),
}


def _patterns(family: str) -> tuple[re.Pattern[str], ...]:
    try:
        return _PATTERNS[family]
    except KeyError:
        raise ValueError(f"unknown family {family!r}: expected 'pep' or 'eth'") from None


def mentions(family: str, body: str, self_number: int) -> set[int]:
    """The numbers of the proposals that ``body`` mentions, other than ``self_number``."""
    found = {int(m.group(1)) for pattern in _patterns(family) for m in pattern.finditer(body)}
    found.discard(self_number)
    return found


def mention_spans(family: str, body: str, number: int) -> list[tuple[int, int]]:
    """The (start, end) offsets of every mention of ``number`` in ``body``, in text order.

    The same spellings as ``mentions``, so a number has a span exactly when ``mentions`` lists it
    (for any number other than the body's own).
    """
    found = {
        m.span()
        for pattern in _patterns(family)
        for m in pattern.finditer(body)
        if int(m.group(1)) == number
    }
    return sorted(found)
