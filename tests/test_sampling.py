"""Hash order and day choice on pools larger than their caps (spec section 7, common rules).

The corpus here is hand-built (no git) and large enough that every sampled stratum has more
eligible candidates than it takes, so the cut at the registered size is exercised: F1-change,
F1-current, F1-recent, F2-negative, each F3 size stratum, F4, CF and the mention pairs.

Every expected value is computed in this file straight from the registration, never from the
generator's helpers: each pool from the histories built below, each eligible day from the margin
rule (the answer held at the end of every day from D-7 to D+7, so a run of one answer from day a
to day b offers a+7 to b-7), the order from SHA-256 of ``asof-bench/v0.1|<family>|<stratum>|<id>``,
and the date from the first eight bytes of SHA-256 of ``asof-bench/v0.1|date|<family>|<stratum>|
<id>``, read big-endian, modulo the number of eligible days.

The corpus:

* PEP group A, PEP 1000+j (j < 180): Draft from ``a_start(j)``, Final from ``a_final(j)``. PEPs
  1000, 1002, ..., 1008 each list the next number in Replaces (five F2 positives). PEPs 1100 to
  1179 mention PEP n+20, n+40 and n+60 in their bodies (F2 negatives, CF prominence, mention
  pairs).
* PEP group R, PEP 2000+j (j < 70): Draft from 2025-01-01, Accepted from ``r_accepted(j)`` in
  2026 (F1-recent).
* Ethereum leaves, EIP 5001 to 5020: a requires chain 5001 -> 5002 -> ... -> 5020, all Draft from
  2018-01-01; EIP-5020 becomes Final on 2020-06-01, which changes every closure's gold.
* Ethereum subjects, EIP 6000+i (i < 170): Draft from ``s_start(i)``, Review 400 days later; each
  requires one leaf (``s_leaf(i)``), so closures of every size stratum; each body mentions the
  next subject (mention pairs, CF prominence).
"""

from __future__ import annotations

import hashlib
from collections import Counter, deque
from datetime import date, timedelta

import pytest
from handbuilt import Hand

from asofbench.questions import build_items

SEED = "asof-bench/v0.1"
S = date(2026, 10, 27)
DAY = timedelta(days=1)
MARGIN = 7

A0 = date(2005, 1, 1)
GROUP_A = range(180)
GROUP_R = range(70)
R_DRAFT = date(2025, 1, 1)
LEAVES = range(1, 21)
LEAF_START = date(2018, 1, 1)
LEAF_FINAL = date(2020, 6, 1)  # EIP-5020 only
SUBJECTS = range(170)
REPLACING = (0, 2, 4, 6, 8)  # group A indices whose PEP lists the next number in Replaces


def a_start(j: int) -> date:
    return A0 + 3 * j * DAY


def a_final(j: int) -> date:
    return a_start(j) + (30 + j) * DAY


def r_accepted(j: int) -> date:
    return date(2026, 1, 10) + j * DAY


def s_start(i: int) -> date:
    return date(2018, 6, 1) + i * DAY


def s_review(i: int) -> date:
    return s_start(i) + 400 * DAY


def s_leaf(i: int) -> int:
    """The leaf subject i requires: 5017-5019 (closures of 2-4), 5007-5016 (5-14), 5001-5006
    (15 or more)."""
    if i < 45:
        return 5017 + i % 3
    if i < 125:
        return 5007 + i % 10
    return 5001 + i % 6


def pep_mentions(n: int) -> set[int]:
    return {n + 20, n + 40, n + 60} if 1100 <= n <= 1179 else set()


def build() -> Hand:
    h = Hand()
    for j in GROUP_A:
        n = 1000 + j
        headers = [("Replaces", str(n + 1))] if j in REPLACING else []
        mentioned = sorted(pep_mentions(n))
        body = ("See " + ", ".join(f"PEP {m}" for m in mentioned) + ".\n") if mentioned else (
            "Plain text.\n")
        h.add("pep", "PEP", n, f"Group A {j}",
              [(a_start(j).isoformat(), "Draft"), (a_final(j).isoformat(), "Final")],
              headers=headers, body=body)
    for j in GROUP_R:
        h.add("pep", "PEP", 2000 + j, f"Group R {j}",
              [(R_DRAFT.isoformat(), "Draft"), (r_accepted(j).isoformat(), "Accepted")])
    for k in LEAVES:
        steps = [(LEAF_START.isoformat(), "Draft")]
        if k == 20:
            steps.append((LEAF_FINAL.isoformat(), "Final"))
        headers = [("requires", str(5000 + k + 1))] if k < 20 else []
        h.add("eth", "EIP", 5000 + k, f"Leaf {k}", steps, headers=headers)
    for i in SUBJECTS:
        h.add("eth", "EIP", 6000 + i, f"Subject {i}",
              [(s_start(i).isoformat(), "Draft"), (s_review(i).isoformat(), "Review")],
              headers=[("requires", str(s_leaf(i)))],
              body=f"Related to EIP-{6000 + (i + 1) % 170}.\n")
    return h


# --- the registration's selection, computed here -------------------------------------------------


def digest(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


def hash_order(ids, family: str, stratum: str) -> list[str]:
    return sorted(ids, key=lambda c: digest(f"{SEED}|{family}|{stratum}|{c}"))


def chosen(days: list[date], family: str, stratum: str, ident: str) -> date:
    k = int.from_bytes(digest(f"{SEED}|date|{family}|{stratum}|{ident}")[:8], "big")
    return days[k % len(days)]


def window(first: date, last: date) -> list[date]:
    """The days D of a run of one answer from ``first`` to ``last`` with D-7..D+7 inside it."""
    lo, hi = first + MARGIN * DAY, last - MARGIN * DAY
    return [lo + k * DAY for k in range((hi - lo).days + 1)]


REQUIRES = {
    **{5000 + k: ([5000 + k + 1] if k < 20 else []) for k in LEAVES},
    **{6000 + i: [s_leaf(i)] for i in SUBJECTS},
}
ETH_MENTIONS = {6000 + i: {6000 + (i + 1) % 170} for i in SUBJECTS}
PEPS = [1000 + j for j in GROUP_A] + [2000 + j for j in GROUP_R]


def reach(n: int) -> dict[int, int]:
    dist, queue = {n: 0}, deque([n])
    while queue:
        x = queue.popleft()
        for y in REQUIRES.get(x, ()):
            if y not in dist:
                dist[y] = dist[x] + 1
                queue.append(y)
    del dist[n]
    return dist


# --- fixtures ------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def built():
    h = build()
    return build_items(h.corpus(), h.readers)


def stratum(built, family: str, name: str, corpus: str) -> list:
    found = [i for i in built if (i.family, i.stratum, i.corpus) == (family, name, corpus)]
    assert [i.rank for i in found] == list(range(len(found)))
    return found


def check(built, family, name, corpus, days_of: dict[str, list[date]], cap: int):
    """The stratum's items are the first ``cap`` of the hash order of ``days_of``'s candidates,
    in that order, each on the day the hash picks from its eligible days."""
    assert len(days_of) > cap, "the pool must exceed the cap"
    assert built.pools[(family, name, corpus)] == len(days_of)
    expected = hash_order(days_of, family, name)[:cap]
    found = stratum(built, family, name, corpus)
    assert [i.proposal for i in found] == expected
    for i in found:
        assert i.as_of == chosen(days_of[i.proposal], family, name, i.proposal).isoformat(), i.id
    return found


# --- F1 ------------------------------------------------------------------------------------------


def test_f1_change_takes_the_first_150_in_hash_order(built):
    pep = {f"PEP-{1000 + j}": window(a_start(j), a_final(j) - DAY) for j in GROUP_A}
    pep |= {f"PEP-{2000 + j}": window(R_DRAFT, r_accepted(j) - DAY) for j in GROUP_R}
    found = check(built, "F1", "change", "pep", pep, 150)
    assert {i.gold for i in found} == {"Draft"}
    eth = {f"EIP-{6000 + i}": window(s_start(i), s_review(i) - DAY) for i in SUBJECTS}
    eth["EIP-5020"] = window(LEAF_START, LEAF_FINAL - DAY)
    check(built, "F1", "change", "eth", eth, 150)


def test_f1_current_takes_the_first_50_in_hash_order(built):
    for corpus, pids in (
        ("pep", [f"PEP-{n}" for n in PEPS]),
        ("eth", [f"EIP-{5000 + k}" for k in LEAVES] + [f"EIP-{6000 + i}" for i in SUBJECTS]),
    ):
        assert built.pools[("F1", "current", corpus)] == len(pids) > 50
        found = stratum(built, "F1", "current", corpus)
        assert [i.proposal for i in found] == hash_order(pids, "F1", "current")[:50]
        assert {i.as_of for i in found} == {S.isoformat()}


def test_f1_recent_takes_the_first_60_in_hash_order(built):
    pep = {f"PEP-{2000 + j}": window(r_accepted(j), S) for j in GROUP_R}
    found = check(built, "F1", "recent", "pep", pep, 60)
    assert {i.gold for i in found} == {"Accepted"}


# --- F2, F3 --------------------------------------------------------------------------------------


def test_f2_negatives_are_the_first_in_hash_order(built):
    positives = stratum(built, "F2", "positive", "pep")
    assert [(i.proposal, i.gold) for i in positives] == [
        (f"PEP-{1000 + j + 1}", [1000 + j]) for j in REPLACING]
    present = set(PEPS)
    pool = [f"PEP-{n}" for n in PEPS if pep_mentions(n) & present]
    assert len(pool) == 60 > len(positives)
    assert built.pools[("F2", "negative", "pep")] == len(pool)
    negatives = stratum(built, "F2", "negative", "pep")
    assert [i.proposal for i in negatives] == hash_order(pool, "F2", "negative")[: len(positives)]
    assert all(i.gold == [] for i in negatives)


def test_f3_strata_take_the_first_40_in_hash_order(built):
    indegree = Counter(m for ms in REQUIRES.values() for m in ms)
    hubs = {n for n, _ in sorted(indegree.items(), key=lambda kv: (-kv[1], kv[0]))[:5]}
    pools: dict[str, dict[str, list[int]]] = {"size-2-4": {}, "size-5-14": {}, "size-15-plus": {}}
    for n in REQUIRES:
        dist = reach(n)
        if len(dist) < 2 or max(dist.values()) < 2 or set(dist) <= hubs:
            continue
        size = len(dist)
        name = "size-2-4" if size <= 4 else "size-5-14" if size <= 14 else "size-15-plus"
        pools[name][f"EIP-{n}"] = sorted(dist)
    for name, pool in pools.items():
        assert len(pool) > 40 and built.pools[("F3", name, "eth")] == len(pool), name
        found = stratum(built, "F3", name, "eth")
        assert [i.proposal for i in found] == hash_order(pool, "F3", name)[:40], name
        assert all(i.gold == pool[i.proposal] and i.as_of == S.isoformat() for i in found)


# --- F4 ------------------------------------------------------------------------------------------


def test_f4_takes_the_first_120_in_hash_order_on_the_hashed_day(built):
    """Every closure holds EIP-5020, which became Final on 2020-06-01: the gold changes there,
    so each candidate's windows are its closure's start to 2020-05-31, and 2020-06-01 to S."""
    after = window(LEAF_FINAL, S)
    days = {}
    for n in REQUIRES:
        size = len(reach(n))
        if not 2 <= size <= 14:
            continue
        start = s_start(n - 6000) if n >= 6000 else LEAF_START
        days[f"EIP-{n}"] = window(start, LEAF_FINAL - DAY) + after
    found = check(built, "F4", "closure-final", "eth", days, 120)
    for i in found:
        members = sorted(reach(int(i.proposal.split("-")[1])))
        assert i.meta["closure"] == members
        assert i.gold == ([5020] if i.as_of >= LEAF_FINAL.isoformat() else []), i.id


# --- CF ------------------------------------------------------------------------------------------


def prominent(scores: Counter, present: list[int]) -> list[int]:
    return sorted(present, key=lambda n: (-scores[n], n))[:50]


def test_cf_takes_the_first_15_in_hash_order_on_the_hashed_day(built):
    present = set(PEPS)
    pep_scores = Counter(m for n in PEPS for m in pep_mentions(n) & present)
    pool = prominent(pep_scores, PEPS)
    eligible = {f"PEP-{n}": window(a_final(n - 1000), S) for n in pool if n < 2000}
    eligible |= {f"PEP-{n}": window(r_accepted(n - 2000), S) for n in pool if n >= 2000}
    found = check(built, "CF", "counterfactual", "pep", eligible, 15)
    assert {(i.meta["true_status"], i.gold) for i in found} <= {("Final", "Withdrawn"),
                                                                ("Accepted", "Rejected")}
    eth = [5000 + k for k in LEAVES] + [6000 + i for i in SUBJECTS]
    eth_scores = Counter()
    for n in eth:
        eth_scores.update((set(REQUIRES[n]) | ETH_MENTIONS.get(n, set())) - {n})
    pool = prominent(eth_scores, eth)
    eligible = {}
    for n in pool:
        if n == 5020:
            eligible["EIP-5020"] = window(LEAF_FINAL, S)
        elif n >= 6000:
            eligible[f"EIP-{n}"] = window(s_review(n - 6000), S)
    assert len(eligible) == 31  # EIP-5020 and the 30 lowest-numbered subjects
    found = check(built, "CF", "counterfactual", "eth", eligible, 15)
    assert {(i.meta["true_status"], i.gold) for i in found} <= {("Final", "Withdrawn"),
                                                                ("Review", "Last Call")}


# --- mention pairs -------------------------------------------------------------------------------


def test_mention_pairs_take_the_first_100_in_hash_order(built):
    present = set(PEPS)
    pep = {f"PEP-{min(n, m)}~PEP-{max(n, m)}" for n in PEPS for m in pep_mentions(n) & present}
    eth = {f"EIP-{min(n, m)}~EIP-{max(n, m)}" for n, ms in ETH_MENTIONS.items() for m in ms}
    for corpus, pool in (("pep", pep), ("eth", eth)):
        assert len(pool) > 100 and built.pools[("labelling", corpus, corpus)] == len(pool)
        found = [r for r in built.labelling if r["corpus"] == corpus]
        assert [r["pair_id"] for r in found] == hash_order(pool, "labelling", corpus)[:100]
        assert [r["rank"] for r in found] == list(range(100))
