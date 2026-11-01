"""Deterministic selection: hash order, the day choice and eligible days (spec section 7).

No pseudo-random generator is used. Candidates are ordered by the SHA-256 of the UTF-8 string
``asof-bench/v0.1|<family>|<stratum>|<candidate id>`` and taken in that order. A candidate's date
is the day at index k of its eligible days in date order, where k is the first eight bytes of the
SHA-256 of ``asof-bench/v0.1|date|<family>|<stratum>|<candidate id>``, read as a big-endian
integer, modulo the number of eligible days.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import date, timedelta
from hashlib import sha256
from typing import TypeVar

from .corpora import SEED, S

T = TypeVar("T")

MARGIN = 7  # days on each side of D over which an answer must hold (section 7)
_DAY = timedelta(days=1)


def _digest(text: str) -> bytes:
    return sha256(text.encode("utf-8")).digest()


def order(candidates: Iterable[T], family: str, stratum: str, ident: Callable[[T], str]) -> list[T]:
    """``candidates`` in ascending SHA-256 of ``f"{SEED}|{family}|{stratum}|{ident(c)}"``."""
    return sorted(candidates, key=lambda c: _digest(f"{SEED}|{family}|{stratum}|{ident(c)}"))


def pick_day(days: list[date], family: str, stratum: str, ident: str) -> date:
    """The day at index k of ``days``, with k from the hash of the candidate's date string."""
    if not days:
        raise ValueError(f"{ident} has no eligible day in {family} {stratum}")
    k = int.from_bytes(_digest(f"{SEED}|date|{family}|{stratum}|{ident}")[:8], "big")
    return days[k % len(days)]


def eligible_range(
    start: date, end: date | None, s: date = S, margin: int = MARGIN
) -> tuple[date, date] | None:
    """The first and last eligible day of an answer that held from ``start`` until ``end``.

    The answer held at the end of every day from ``start`` to the day before ``end`` (to ``s``
    when ``end`` is None, because it still held then), so D is eligible when D - margin >= start
    and D + margin <= end - 1 day (or <= s). None when no day is.
    """
    last = s if end is None else end - _DAY
    first, final = start + margin * _DAY, last - margin * _DAY
    return (first, final) if first <= final else None


def eligible_days(start: date, end: date | None, s: date = S, margin: int = MARGIN) -> list[date]:
    """Every day of ``eligible_range(start, end, s, margin)``, in date order."""
    found = eligible_range(start, end, s, margin)
    if found is None:
        return []
    first, last = found
    return [first + k * _DAY for k in range((last - first).days + 1)]
