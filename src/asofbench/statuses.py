"""Status normalisation, outcome classes and the counterfactual permutation.

Spec section 5 (recorded statuses, outcome classes), section 7 (the CF permutation) and
section 12 (answers). ``family`` is ``"pep"`` for the Python Enhancement Proposals and
``"eth"`` for the Ethereum EIPs and ERCs.
"""

from __future__ import annotations

import re

MISSING = "missing"  # empty, or a template placeholder: not a status
UNKNOWN = "unknown"  # any other value that the tables do not list
MOVED = "Moved"  # an EIP whose text moved to the ERC repository (eth only); not a status

CANONICAL: dict[str, tuple[str, ...]] = {
    "pep": (
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
    ),
    "eth": (
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
    ),
}

# The values a question offers: every canonical status except the two that are never an answer.
_NEVER_AN_ANSWER = frozenset({"April Fool!", "Stagnant"})
CHOICES: dict[str, tuple[str, ...]] = {
    family: tuple(s for s in statuses if s not in _NEVER_AN_ANSWER)
    for family, statuses in CANONICAL.items()
}

# Earlier PEP names and misspellings (section 5). Ethereum keeps its earlier values as recorded.
_PEP_LEGACY = {
    "incomplete": "Draft",
    "draught": "Draft",
    "approved": "Accepted",
    "finished": "Final",
    "complete": "Final",
    "postponed": "Deferred",
    "withdrawal": "Withdrawn",
    "replaced": "Superseded",
}

# Case-folded value -> canonical status, per family.
_LOOKUP: dict[str, dict[str, str]] = {
    family: {s.casefold(): s for s in statuses} for family, statuses in CANONICAL.items()
}
_LOOKUP["pep"].update(_PEP_LEGACY)

_OUTCOMES: dict[str, dict[str, tuple[str, ...]]] = {
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
_OUTCOME_OF = {
    family: {status: name for name, statuses in classes.items() for status in statuses}
    for family, classes in _OUTCOMES.items()
}

_SWAPS: dict[str, tuple[tuple[str, str], ...]] = {
    "pep": (
        ("Accepted", "Rejected"),
        ("Final", "Withdrawn"),
        ("Active", "Deferred"),
        ("Provisional", "Superseded"),
    ),
    "eth": (("Final", "Withdrawn"), ("Review", "Last Call")),
}


def _both_ways(pairs: tuple[tuple[str, str], ...]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for a, b in pairs:
        mapping[a] = b
        mapping[b] = a
    return mapping


# The counterfactual edit of section 7: only the swapped statuses, in both directions. Every
# other value is unchanged; ``permute`` applies the edit to any status.
CF_PERMUTATION: dict[str, dict[str, str]] = {
    family: _both_ways(pairs) for family, pairs in _SWAPS.items()
}

_PLACEHOLDER = re.compile(r"<.*>")
_INLINE_COMMENT = re.compile(r"(?:^|\s)#.*")


def _check(family: str) -> None:
    if family not in CANONICAL:
        raise ValueError(f"unknown family {family!r}: expected 'pep' or 'eth'")


def permute(family: str, status: str) -> str:
    """The status after the counterfactual edit: its swap partner, or ``status`` itself.

    Stagnant, April Fool!, ``unknown``, ``missing``, ``absent`` and every status that section 7
    does not swap come back unchanged.
    """
    _check(family)
    return CF_PERMUTATION[family].get(status, status)


def normalize_status(family: str, raw: str | None) -> str:
    """A canonical status, or MISSING, UNKNOWN or (eth only) MOVED, for a recorded value."""
    _check(family)
    if raw is None:
        return MISSING
    value = raw.strip()
    if _PLACEHOLDER.fullmatch(value):
        return MISSING
    value = _INLINE_COMMENT.sub("", value, count=1).strip()
    if not value:
        return MISSING
    key = value.casefold()
    if family == "eth" and key == MOVED.casefold():
        return MOVED
    return _LOOKUP[family].get(key, UNKNOWN)


def outcome_class(family: str, status: str) -> str | None:
    """adopted, declined, superseded or open; None for a value that has no class."""
    _check(family)
    return _OUTCOME_OF[family].get(status)


# Straight quotes, and the typographic pairs U+201C/U+201D and U+2018/U+2019 (section 12).
_QUOTES = "\"'" + chr(0x201C) + chr(0x201D) + chr(0x2018) + chr(0x2019)


def _clear_quotes(text: str) -> str:
    return text.strip(_QUOTES).strip()


def normalize_answer_status(family: str, answer: str) -> str | None:
    """The canonical status that a system's answer names, or None if it names none (section 12).

    The answer is stripped, cleared of surrounding straight or typographic quotes (any of
    the six characters, from either end) and of one trailing full stop (inside or outside the
    quotes), case-folded and has its whitespace collapsed; ``last-call`` and ``lastcall`` read
    as ``last call``; then the family's table is applied. ``answer`` must be a string; the
    scorer rejects other types before it gets here.
    """
    _check(family)
    text = _clear_quotes(answer.strip())
    if text.endswith("."):
        text = _clear_quotes(text[:-1])
    text = " ".join(text.casefold().split())
    if text in ("last-call", "lastcall"):
        text = "last call"
    return _LOOKUP[family].get(text)
