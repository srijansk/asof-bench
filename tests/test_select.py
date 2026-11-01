"""Deterministic selection (spec section 7, common rules)."""

from __future__ import annotations

from datetime import date, timedelta
from hashlib import sha256

import pytest

from asofbench.select import eligible_days, eligible_range, order, pick_day


def test_order_is_hash_order():
    ids = ["PEP-1", "PEP-2", "PEP-3"]
    expected = sorted(ids, key=lambda x: sha256(f"asof-bench/v0.1|F1|change|{x}".encode()).digest())
    assert order(ids, "F1", "change", str) == expected


def test_order_hashes_the_ident_of_any_candidate():
    pairs = [("PEP-7", 1), ("EIP-20", 2), ("ERC-721", 3), ("PEP-8", 4)]
    by_hand = sorted(
        pairs, key=lambda c: sha256(f"asof-bench/v0.1|CF|counterfactual|{c[0]}".encode()).digest()
    )
    assert order(pairs, "CF", "counterfactual", lambda c: c[0]) == by_hand
    # The family and the stratum enter the hash string.
    labelling = sorted(
        pairs, key=lambda c: sha256(f"asof-bench/v0.1|labelling|eth|{c[0]}".encode()).digest()
    )
    assert order(pairs, "labelling", "eth", lambda c: c[0]) == labelling
    assert order(iter(pairs), "labelling", "eth", lambda c: c[0]) == labelling  # any iterable


def test_pick_day():
    days = [date(2020, 1, 1) + timedelta(i) for i in range(10)]
    k = int.from_bytes(sha256(b"asof-bench/v0.1|date|F1|change|PEP-1").digest()[:8], "big") % 10
    assert pick_day(days, "F1", "change", "PEP-1") == days[k]


def test_pick_day_needs_a_day():
    with pytest.raises(ValueError):
        pick_day([], "F1", "change", "PEP-1")


def test_eligible_days():
    assert eligible_days(date(2020, 1, 1), date(2020, 1, 16)) == [date(2020, 1, 8)]
    assert eligible_days(date(2020, 1, 1), date(2020, 1, 15)) == []
    still_holding = eligible_days(date(2026, 10, 1), None)
    assert still_holding == [date(2026, 10, 8) + timedelta(i) for i in range(13)]
    assert still_holding[-1] == date(2026, 10, 20)


def test_eligible_days_with_another_snapshot_and_margin():
    assert eligible_days(date(2020, 1, 1), None, s=date(2020, 1, 10), margin=3) == [
        date(2020, 1, 4) + timedelta(i) for i in range(4)
    ]
    assert eligible_days(date(2020, 1, 1), date(2020, 1, 2), margin=0) == [date(2020, 1, 1)]


def test_eligible_range_agrees_with_eligible_days():
    cases = [
        (date(2020, 1, 1), date(2020, 1, 16)),
        (date(2020, 1, 1), date(2020, 1, 15)),
        (date(2026, 10, 1), None),
        (date(2019, 3, 5), date(2021, 7, 30)),
    ]
    for start, end in cases:
        days = eligible_days(start, end)
        found = eligible_range(start, end)
        if not days:
            assert found is None
        else:
            assert found == (days[0], days[-1])
            assert len(days) == (days[-1] - days[0]).days + 1
