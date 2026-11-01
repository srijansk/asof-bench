"""Question families, the labelling sample and the outputs, on a hand-built corpus (spec section 7).

The corpus is built directly from ``Proposal`` and ``Version`` objects, with no git: eight PEPs
and eleven Ethereum proposals, each shaped to exercise a rule. Every proposal's role is written next
to it; the expected pools below follow from those histories by hand.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from handbuilt import DELETE, Hand, render, ts

from asofbench import questions
from asofbench.corpora import CANARY, S
from asofbench.gitwalk import end_of_day_ts
from asofbench.headers import header_span
from asofbench.mentions import mentions
from asofbench.questions import TEMPLATES, build_items, write_outputs
from asofbench.statuses import CHOICES, outcome_class, permute
from asofbench.timeline import Proposal, Version

EOD_S = end_of_day_ts(S)
DAY = timedelta(days=1)
PREREGISTRATION = Path(__file__).resolve().parents[1] / "PREREGISTRATION.md"


def eod(d: date) -> int:
    return end_of_day_ts(d)


def hand_built() -> Hand:
    h = Hand()
    # --- PEPs -----------------------------------------------------------------------------
    # F1-change (Draft, Accepted), F1-current (Final), CF; its body mentions PEP 10.
    h.add("pep", "PEP", 1, "Purpose and Guidelines",
          [("2001-01-01", "Draft"), ("2003-06-01", "Accepted"), ("2005-01-01", "Final")],
          body="This PEP explains the process.\nSee   PEP 10 for an example of an older "
               "approach.\n")
    # Deleted for four years, then re-added as Rejected: no date in the gap is ever sampled.
    h.add("pep", "PEP", 2, "Deleted And Readded",
          [("2010-01-01", "Draft"), ("2010-02-01", DELETE), ("2014-01-01", "Rejected")],
          body="The proposal was rejected.\n")
    # F2 positive by a one-sided record: nothing here, PEP 11 says Replaces: 10.
    h.add("pep", "PEP", 10, "Old Way", [("2005-03-01", "Draft"), ("2008-03-01", "Superseded")])
    # Replaces itself and a number that is no PEP: both are dropped from every gold set.
    h.add("pep", "PEP", 11, "New Way", [("2006-01-01", "Draft"), ("2008-06-01", "Final")],
          headers=[("Replaces", "10, 11, 9999")], body="This replaces the approach of PEP 10.\n")
    # A 2026 transition (F1-recent), then Final the day before S (the PEP F5 Final item at S is
    # unstable and dropped); Superseded-By lists only itself and a non-PEP, so its gold is empty.
    h.add("pep", "PEP", 12, "Recent Change",
          [("2015-01-01", "Draft"), ("2026-02-01", "Accepted"), ("2026-10-26", "Final")],
          headers=[("Superseded-By", "12, 9999")], body="Builds on PEP 1.\n")
    # Final two days after 2020-01-01: the F5 Draft and Final items move to 2020-01-06.
    h.add("pep", "PEP", 13, "Final In 2020",
          [("2018-01-01", "Draft"), ("2020-01-03T10:00:00Z", "Final")],
          headers=[("Superseded-By", "14")], body="No references here.\n")
    # Created in 2026: a first status is no transition, so it gives F1-recent nothing.
    h.add("pep", "PEP", 14, "Created In 2026", [("2026-02-01", "Draft")])
    # Final from S-7 exactly: its status at S has held at the end of S-7 to S (F1-current).
    h.add("pep", "PEP", 15, "Final At S Minus 7",
          [("2019-05-01", "Draft"), ("2026-10-20", "Final")])

    # --- Ethereum ---------------------------------------------------------------------------
    # An ERC at S whose history began in the EIP repository.
    h.add("eth", "ERC", 20, "Token Standard",
          [("2015-11-19", "Draft"), ("2017-09-11", "Final"),
           ("2023-10-25T15:11:00Z", "Final", "ERCs")],
          body="A standard interface for tokens.\n")
    h.add("eth", "EIP", 100, "Hub Hundred", [("2016-01-01", "Draft"), ("2018-06-01", "Final")],
          headers=[("requires", "101")])
    h.add("eth", "EIP", 101, "Hub Hundred One",
          [("2016-01-01", "Draft"), ("2019-01-01", "Review"), ("2019-06-01", "Last Call"),
           ("2019-09-01", "Final")])
    # A first version without a status: a month of "missing", never sampled.
    h.add("eth", "EIP", 102, "Withdrawn One",
          [("2016-12-01", None), ("2017-01-01", "Draft"), ("2018-02-01", "Withdrawn")],
          body="This proposal was withdrawn by its author.\n")
    # Closure {100, 101}, two hops, but both are hubs: dropped from F3 as hub recall.
    h.add("eth", "EIP", 103, "Hub Recall", [("2017-06-01", "Draft")], headers=[("requires", "100")])
    # Closure {20, 100, 101, 102}, all one hop away: not F3. Stagnant for a year: F1 never asks
    # Stagnant and F4 never samples those days. A 2026 transition to Last Call (F1-recent).
    h.add("eth", "EIP", 104, "Stagnant Gap",
          [("2019-03-01", "Draft"), ("2021-01-01", "Stagnant"), ("2022-02-01", "Draft"),
           ("2026-02-01", "Last Call")],
          headers=[("requires", "20, 100, 101, 102")], body="Uses EIP-20 tokens.\n")
    # A requires cycle 105 -> 106 -> 107 -> 106: closure {106, 107} for 105 (F3 and F4). Review
    # two days before S: the Ethereum F5 Draft and Review items at S are unstable and dropped.
    h.add("eth", "EIP", 105, "Cycle Start", [("2019-01-01", "Draft"), ("2026-10-25", "Review")],
          headers=[("requires", "106")], body="Compatible with ERC-20 balances.\n")
    h.add("eth", "EIP", 106, "Cycle Middle", [("2019-01-01", "Draft"), ("2021-06-01", "Final")],
          headers=[("requires", "107")])
    # Final two days after 2020-01-01: moves the Ethereum F5 Draft and Final items.
    h.add("eth", "EIP", 107, "Cycle End",
          [("2019-01-01", "Draft"), ("2020-01-03T12:00:00Z", "Final")],
          headers=[("requires", "106")])
    # Deleted before S: no item is about it (section 7), though its Draft interval is long.
    h.add("eth", "EIP", 108, "Deleted Proposal", [("2018-03-01", "Draft"), ("2019-06-01", DELETE)])
    # Final from S-6: one day short of F1-current's seven.
    h.add("eth", "EIP", 109, "Final At S Minus 6",
          [("2019-05-01", "Draft"), ("2026-10-21", "Final")])
    return h


@pytest.fixture(scope="module")
def hand():
    return hand_built()


@pytest.fixture(scope="module")
def corpus(hand):
    return hand.corpus()


@pytest.fixture(scope="module")
def built(hand, corpus):
    return build_items(corpus, hand.readers)


def of(items, family, stratum=None, corpus=None):
    return [
        i for i in items
        if i.family == family
        and (stratum is None or i.stratum == stratum)
        and (corpus is None or i.corpus == corpus)
    ]


def day_of(item) -> date:
    return date.fromisoformat(item.as_of)


def held(p: Proposal, d: date, status: str, margin: int = 7) -> bool:
    return all(p.status_at(eod(d + k * DAY)) == status for k in range(-margin, margin + 1))


# --- F1 ------------------------------------------------------------------------------------------


def test_f1_change_gold_differs_from_s(built, corpus):
    change = of(built, "F1", "change")
    assert {i.proposal for i in change} == {
        "PEP-1", "PEP-2", "PEP-10", "PEP-11", "PEP-12", "PEP-13", "PEP-15",
        "ERC-20", "EIP-100", "EIP-101", "EIP-102", "EIP-104", "EIP-105", "EIP-106", "EIP-107",
        "EIP-109",
    }  # not EIP-108, deleted before S
    for i in change:
        p = corpus.proposals[i.proposal]
        d = day_of(i)
        assert i.gold != p.status_at(EOD_S), i.id
        assert i.gold != "Stagnant" and i.gold in CHOICES[i.corpus], i.id
        assert i.gold == p.status_at(eod(d)) and held(p, d, i.gold), i.id
        assert d <= S - 7 * DAY, i.id
        assert i.gold_outcome == outcome_class(i.corpus, i.gold)
        assert i.choices == CHOICES[i.corpus] and i.answer_type == "status"
        assert i.arms == ("H+", "H-")


def test_f1_current_is_s(built, corpus):
    current = of(built, "F1", "current")
    assert {i.proposal for i in current} == {
        "PEP-1", "PEP-2", "PEP-10", "PEP-11", "PEP-13", "PEP-14", "PEP-15",
        "ERC-20", "EIP-100", "EIP-101", "EIP-102", "EIP-103", "EIP-104", "EIP-106", "EIP-107",
    }  # not PEP-12 (Final since S-1), EIP-105 (Review since S-2), EIP-109 (Final since S-6)
    for i in current:
        p = corpus.proposals[i.proposal]
        assert i.as_of == "2026-10-27"
        assert i.gold == p.status_at(EOD_S)
        assert all(p.status_at(eod(S - k * DAY)) == i.gold for k in range(8)), i.id


def test_f1_recent_starts_with_transition(built, corpus):
    recent = of(built, "F1", "recent")
    assert {i.proposal for i in recent} == {"PEP-12", "EIP-104"}  # PEP-14 was created in 2026
    start_2026 = ts("2026-01-01T00:00:00Z")
    for i in recent:
        p = corpus.proposals[i.proposal]
        d = day_of(i)
        assert held(p, d, i.gold) and d <= S - 7 * DAY
        for k in range(-7, 8):
            moment = eod(d + k * DAY)
            interval = next(
                iv for iv in p.intervals()
                if iv.start_ts <= moment and (iv.end_ts is None or moment < iv.end_ts)
            )
            assert interval.is_transition and interval.start_ts >= start_2026, i.id
    assert questions.f1_ranges(corpus.proposals["PEP-14"], "recent") == []
    assert questions.f1_ranges(corpus.proposals["PEP-12"], "recent") == [
        (date(2026, 2, 8), date(2026, 10, 18))
    ]


def test_f1_items_record_whether_the_gold_is_the_status_at_s(built, corpus):
    """The gold of every F1 item records whether its gold equals the status at S, so that the
    scorer can split F1-recent by it (section 10) without the timelines."""
    for i in of(built, "F1"):
        at_s = corpus.proposals[i.proposal].status_at(EOD_S)
        assert i.meta["status_at_s"] == at_s, i.id
        assert i.meta["gold_equals_status_at_s"] is (i.gold == at_s), i.id
    assert not any(i.meta["gold_equals_status_at_s"] for i in of(built, "F1", "change"))
    assert all(i.meta["gold_equals_status_at_s"] for i in of(built, "F1", "current"))
    # F1-recent has both: PEP-12 was Accepted at its date and is Final at S; EIP-104 has been in
    # Last Call since its 2026 transition.
    assert {i.proposal: i.meta["gold_equals_status_at_s"] for i in of(built, "F1", "recent")} == {
        "PEP-12": False, "EIP-104": True}
    assert not any("gold_equals_status_at_s" in i.meta for i in built if i.family != "F1")


def header_only_world(root: Path):
    """Three repositories laid out like the clones: PEP 1 has a status-only commit on
    2020-06-01; PEP 2 changes its status and its text together on 2020-06-01; PEP 3 has a
    status-only commit on 2020-03-01 and then edits its text."""
    from gitrepo import TempRepo

    repos = {name: TempRepo(root / name) for name in ("peps", "EIPs", "ERCs")}
    for repo in repos.values():
        repo.write("README.md", "Proposals.\n")
        repo.commit("start", "2000-01-01T00:00:00Z")
    peps = repos["peps"]

    def text(number, status, body="Plain text.\n"):
        return f"PEP: {number}\nTitle: Proposal {number}\nStatus: {status}\n\n{body}"

    for day, files in [
        ("2020-01-01", {1: text(1, "Draft"), 2: text(2, "Draft"), 3: text(3, "Draft")}),
        ("2020-03-01", {1: text(1, "Draft", "Edited.\n"), 3: text(3, "Accepted")}),
        ("2020-06-01", {1: text(1, "Accepted", "Edited.\n"), 2: text(2, "Accepted", "New.\n"),
                        3: text(3, "Accepted", "Edited.\n")}),
    ]:
        for number, content in files.items():
            peps.write(f"pep-{number:04d}.txt", content)
        peps.commit(f"Update on {day}", f"{day}T12:00:00Z")
    return root


def test_an_f1_item_records_a_header_only_h_minus_episode_after_its_date(tmp_path):
    """Section 8 keeps an H- episode whose diff the withholding empties, with its message: a
    stream system sees that something it cannot read changed. Each F1 item records whether its
    proposal has such an episode recorded after the end of its date
    (``meta.h_minus_header_only_after_d``); a content change, or a header-only change on or
    before the date, does not count."""
    from asofbench.timeline import load_corpus

    corpus = load_corpus(header_only_world(tmp_path / "clones"))
    with corpus.readers() as readers:
        def flag(pid: str, day: str) -> bool:
            p = corpus.proposals[pid]
            item = questions._status_item(corpus, p, "F1", "change", date.fromisoformat(day), 0,
                                          readers)
            assert item.meta["h_minus_header_only_after_d"] is questions.header_only_after(
                p, date.fromisoformat(day), readers)
            return item.meta["h_minus_header_only_after_d"]

        assert flag("PEP-1", "2020-05-31") is True  # the status-only commit comes after
        assert flag("PEP-1", "2020-06-01") is False  # on the date itself: not after its end
        assert flag("PEP-2", "2020-05-31") is False  # its status changed with its text
        assert flag("PEP-3", "2020-02-29") is True
        assert flag("PEP-3", "2020-03-01") is False  # before the date: grep at D sees it too
        cf = questions._status_item(corpus, corpus.proposals["PEP-1"], "CF", "counterfactual",
                                    date(2020, 5, 31), 0, readers)
        assert "h_minus_header_only_after_d" not in cf.meta  # asked over CF, not H-
        built = build_items(corpus, readers)
        f1 = of(built, "F1")
        assert f1 and all(isinstance(i.meta["h_minus_header_only_after_d"], bool) for i in f1)
        for i in f1:
            assert i.meta["h_minus_header_only_after_d"] is questions.header_only_after(
                corpus.proposals[i.proposal], day_of(i), readers), i.id


def test_the_header_only_flag_reads_the_end_of_the_date_to_the_second(tmp_path):
    """The flag reads episodes recorded after D 23:59:59 UTC: a status-only commit at 23:59:59
    on D is part of D, and one at 00:00:00 on the next day is after it."""
    from gitrepo import TempRepo

    from asofbench.timeline import load_corpus

    repos = {name: TempRepo(tmp_path / "clones" / name) for name in ("peps", "EIPs", "ERCs")}
    for repo in repos.values():
        repo.write("README.md", "Proposals.\n")
        repo.commit("start", "2000-01-01T00:00:00Z")
    peps = repos["peps"]

    def text(number, status):
        return f"PEP: {number}\nTitle: Proposal {number}\nStatus: {status}\n\nPlain text.\n"

    for number in (4, 5):
        peps.write(f"pep-{number:04d}.txt", text(number, "Draft"))
    peps.commit("Add PEPs 4 and 5", "2020-01-01T12:00:00Z")
    peps.write("pep-0004.txt", text(4, "Accepted"))
    peps.commit("Accept PEP 4", "2020-05-31T23:59:59Z")
    peps.write("pep-0005.txt", text(5, "Accepted"))
    peps.commit("Accept PEP 5", "2020-06-01T00:00:00Z")
    corpus = load_corpus(tmp_path / "clones")
    day = date(2020, 5, 31)
    with corpus.readers() as readers:
        assert questions.header_only_after(corpus.proposals["PEP-4"], day, readers) is False
        assert questions.header_only_after(corpus.proposals["PEP-5"], day, readers) is True
        assert questions.header_only_after(corpus.proposals["PEP-4"], day - DAY, readers) is True


def test_the_report_counts_the_f1_items_with_a_header_only_episode_after_their_date(built,
                                                                                   corpus):
    text = questions.report(built, corpus)
    section = text.split("\n## F1 items with a header-only H- episode after their date\n",
                         1)[1].split("\n## ", 1)[0]
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")]
            for line in section.splitlines() if line.startswith("| ")]
    assert rows[0] == ["Stratum", "Corpus", "Items", "Flagged"]
    expected = []
    for stratum in ("change", "current", "recent"):
        for family in ("pep", "eth"):
            found = of(built, "F1", stratum, family)
            if found:
                flagged = sum(i.meta["h_minus_header_only_after_d"] for i in found)
                expected.append([stratum, family, str(len(found)), str(flagged)])
    assert rows[1:] == expected
    # Every status step of the hand-built history changes the status line alone, so every
    # F1-change item has one after its date, except PEP 2's, whose only later episodes are its
    # deletion and its re-creation; an F1-current item, dated S, never has one.
    assert {i.proposal for i in of(built, "F1", "change")
            if not i.meta["h_minus_header_only_after_d"]} == {"PEP-2"}
    assert not any(i.meta["h_minus_header_only_after_d"] for i in of(built, "F1", "current"))
    assert "meta.h_minus_header_only_after_d" in section


def test_f1_recent_reads_transitions_at_the_end_of_each_day():
    """Section 7: statuses and transitions are read at the end of each day, so a change reverted
    within one day is no change; F1-recent's windows lie in the status runs that the ends of days
    show beginning with a transition on or after 2026-01-01."""
    h = Hand()
    # Accepted and back to Draft within one day: no end of day sees a 2026 transition.
    zigzag = h.add("pep", "PEP", 21, "Same Day Zigzag",
                   [("2015-01-01", "Draft"), ("2026-03-01T10:00:00Z", "Accepted"),
                    ("2026-03-01T12:00:00Z", "Draft")])
    # A zigzag inside a recent status leaves that status, and its window, whole.
    inside = h.add("pep", "PEP", 22, "Zigzag Inside",
                   [("2015-01-01", "Draft"), ("2026-02-01", "Accepted"),
                    ("2026-05-01T10:00:00Z", "Final"), ("2026-05-01T12:00:00Z", "Accepted")])
    # Deleted and added again within one day: no end of day sees the absence, so Accepted
    # follows Draft as a transition.
    readded = h.add("pep", "PEP", 23, "Readded Same Day",
                    [("2015-01-01", "Draft"), ("2026-03-01T10:00:00Z", DELETE),
                     ("2026-03-01T12:00:00Z", "Accepted")])
    assert questions.f1_ranges(zigzag, "recent") == []
    assert questions.f1_ranges(inside, "recent") == [(date(2026, 2, 8), date(2026, 10, 20))]
    assert questions.f1_ranges(readded, "recent") == [(date(2026, 3, 8), date(2026, 10, 20))]


def test_no_item_in_absent_gap(built, corpus):
    for i in built:
        if i.family not in ("F1", "CF"):
            continue
        p = corpus.proposals[i.proposal]
        d = day_of(i)
        assert all(p.status_at(eod(d + k * DAY)) is not None for k in range(-7, 1)), i.id
    pep2 = corpus.proposals["PEP-2"]
    assert questions.f1_ranges(pep2, "change") == [(date(2010, 1, 8), date(2010, 1, 24))]
    (item,) = [i for i in of(built, "F1", "change") if i.proposal == "PEP-2"]
    assert date(2010, 1, 8) <= day_of(item) <= date(2010, 1, 24) and item.gold == "Draft"
    # EIP-108 is absent at S, so no item is about it at all.
    assert not [i.id for i in built if i.proposal == "EIP-108"]


def test_missing_and_stagnant_are_never_sampled(corpus):
    eip102 = corpus.proposals["EIP-102"]
    assert eip102.status_at(eod(date(2016, 12, 15))) == "missing"
    assert questions.f1_ranges(eip102, "change") == [(date(2017, 1, 8), date(2018, 1, 24))]
    eip104 = corpus.proposals["EIP-104"]
    for lo, hi in questions.f1_ranges(eip104, "change"):
        assert all(eip104.status_at(eod(lo + k * DAY)) == "Draft" for k in range((hi - lo).days))


def test_f1_change_compares_each_window_with_the_last_offered_status_at_s():
    """Section 7: F1-change compares each window's status with the proposal's last offered status
    at S, the last status a question offers among its day runs up to S: its status at S when a
    question offers that, otherwise the last offered status it held before S. Stagnant, or any
    other value no question offers, records no new decision, so going to it from an offered
    status is no change from that status."""
    h = Hand()
    # Draft for a year, then Stagnant at S: Draft is its last offered status, so no change.
    stagnant = h.add("eth", "EIP", 31, "Draft Then Stagnant",
                     [("2020-01-01", "Draft"), ("2021-01-01", "Stagnant")])
    # Draft, then Review, then Stagnant at S: only the Draft days differ from Review.
    reviewed = h.add("eth", "EIP", 32, "Review Then Stagnant",
                     [("2020-01-01", "Draft"), ("2020-06-01", "Review"),
                      ("2021-01-01", "Stagnant")])
    # Draft, then Final at S: the Draft days, as before.
    final = h.add("eth", "EIP", 33, "Draft Then Final",
                  [("2020-01-01", "Draft"), ("2021-01-01", "Final")])
    # Draft, then Stagnant, then Final at S: the Draft days.
    revived = h.add("eth", "EIP", 34, "Stagnant Then Final",
                    [("2020-01-01", "Draft"), ("2021-01-01", "Stagnant"),
                     ("2022-01-01", "Final")])
    # Draft, then a header without a status at S: a missing status is offered by no question.
    blank = h.add("eth", "EIP", 35, "Draft Then No Status",
                  [("2020-01-01", "Draft"), ("2021-01-01", None)])
    # The status is read from the day runs: a Review recorded and moved to Stagnant within one
    # day is seen by no end of day, so Draft is the last offered status.
    same_day = h.add("eth", "EIP", 36, "Review For An Hour",
                     [("2020-01-01", "Draft"), ("2020-12-31T10:00:00Z", "Review"),
                      ("2020-12-31T11:00:00Z", "Stagnant")])
    # An absence is a day run that holds no status, looked past like Stagnant: deleted, then
    # added again as Stagnant, the proposal's last offered status is still Draft.
    readded = h.add("eth", "EIP", 37, "Deleted Then Stagnant",
                    [("2020-01-01", "Draft"), ("2021-01-01", DELETE),
                     ("2022-01-01", "Stagnant")])
    draft = (date(2020, 1, 8), date(2020, 12, 24))
    assert questions.f1_ranges(stagnant, "change") == []
    assert questions.f1_ranges(reviewed, "change") == [(date(2020, 1, 8), date(2020, 5, 24))]
    assert questions.f1_ranges(final, "change") == [draft]
    assert questions.f1_ranges(revived, "change") == [draft]
    assert questions.f1_ranges(blank, "change") == []
    assert questions.f1_ranges(same_day, "change") == []
    assert questions.f1_ranges(readded, "change") == []
    # The stratum follows the rule: no item about EIP-31, EIP-35, EIP-36 or EIP-37, and no item
    # whose gold is its proposal's last offered status at S.
    last_offered = {"EIP-32": "Review", "EIP-33": "Final", "EIP-34": "Final"}
    corpus = h.corpus()
    built = build_items(corpus, h.readers)
    change = of(built, "F1", "change", "eth")
    assert built.pools[("F1", "change", "eth")] == 3
    assert {i.proposal for i in change} == set(last_offered)
    for i in change:
        assert i.gold == "Draft" != last_offered[i.proposal], i.id
        assert date(2020, 1, 8) <= day_of(i) <= date(2020, 12, 24), i.id


def test_a_proposal_deleted_before_s_is_never_asked_about():
    """Section 7: every item about one proposal concerns a proposal present at S. Members of a
    requirement closure (F4) and the proposals counted in F5 need not be present at S."""
    h = Hand()
    h.add("eth", "EIP", 400, "Deleted With Requires",
          [("2019-01-01", "Draft"), ("2020-03-01", "Final"), ("2022-06-01", DELETE)],
          headers=[("requires", "401, 402")])
    h.add("eth", "EIP", 401, "Kept One", [("2019-01-01", "Final")])
    h.add("eth", "EIP", 402, "Kept Two", [("2019-01-01", "Final")])
    h.add("eth", "EIP", 403, "Deleted In 2026",
          [("2025-06-01", "Draft"), ("2026-02-01", "Review"), ("2026-09-01", DELETE)])
    h.add("eth", "EIP", 404, "Requires The Deleted", [("2019-06-01", "Draft")],
          headers=[("requires", "400, 401")])
    h.add("pep", "PEP", 400, "Deleted PEP",
          [("2010-01-01", "Draft"), ("2012-01-01", "Accepted"), ("2015-01-01", DELETE)])
    h.add("pep", "PEP", 401, "Mentions It", [("2010-01-01", "Final")], body="See PEP 400.\n")
    corpus = h.corpus()
    gone = ("EIP-400", "EIP-403", "PEP-400")
    eip400, eip403, pep400 = (corpus.proposals[pid] for pid in gone)
    # Each of them has days that every other rule would allow.
    assert questions.f1_ranges(eip400, "change") and questions.f1_ranges(pep400, "change")
    assert questions.f1_ranges(eip403, "recent") and questions.f1_ranges(eip403, "change")
    assert questions.cf_ranges(eip400) and questions.cf_ranges(pep400)
    assert "EIP-400" in questions.f4_ranges(corpus)
    built = build_items(corpus, h.readers)
    assert [i.id for i in built if i.proposal in gone] == []
    assert all(corpus.proposals[i.proposal].status_at(EOD_S) is not None
               for i in built if i.proposal is not None)
    assert {i.proposal for i in of(built, "CF")} == {"EIP-401", "EIP-402", "PEP-401"}
    # A closure member need not be present at S: EIP-404's closure held EIP-400 until 2022.
    (f4,) = of(built, "F4")
    assert f4.proposal == "EIP-404" and 400 in f4.meta["closure"]
    assert day_of(f4) < date(2022, 6, 1)
    # Nor need the proposals counted in F5: EIP-400 is one of the three Final at 2022-01-01.
    f5 = {i.id: i for i in of(built, "F5", corpus="eth")}
    assert f5["F5-count-eth-Final-2022-01-01"].gold == 3


# --- runs by end of day ----------------------------------------------------------------------


def test_day_runs_read_the_status_at_the_end_of_each_day():
    p = Proposal("pep", "PEP", 50)
    for when, status in [
        ("2020-01-01T10:00:00Z", "Draft"),
        ("2020-02-01T10:00:00Z", "Accepted"),  # back to Draft before the end of the day
        ("2020-02-01T11:00:00Z", "Draft"),
        ("2020-03-01T10:00:00Z", "Final"),
    ]:
        p.versions.append(Version("peps", ts(when), "c" * 40, "peps/pep-0050.rst", "b" * 40,
                                  "modify", {"status": status}, status, status))
    assert [(r.status, r.first, r.last) for r in questions.day_runs(p)] == [
        ("Draft", date(2020, 1, 1), date(2020, 2, 29)),
        ("Final", date(2020, 3, 1), S),
    ]
    # The Draft window spans the reverted Accepted: the end of every day saw Draft.
    assert questions.f1_ranges(p, "change") == [(date(2020, 1, 8), date(2020, 2, 22))]


def test_a_same_day_reversion_is_reported():
    h = Hand()
    h.add("pep", "PEP", 50, "Reverted Once",
          [("2020-01-01", "Draft"), ("2020-06-01T10:00:00Z", "Accepted"),
           ("2020-06-01T11:00:00Z", "Draft"), ("2021-01-01", "Final")])
    corpus = h.corpus()
    built = build_items(corpus, h.readers)
    found = built.notes["reversions"]
    assert found["reversions"] == [
        {"proposal": "PEP-50", "day": "2020-06-01", "status": "Draft", "reverted": "Accepted"}]
    (change,) = of(built, "F1", "change")
    assert found["items_of_these_proposals"] == sorted(
        i.id for i in built if i.family in ("F1", "CF"))
    # One Draft window across the reversion, where an interval-by-interval reading has two.
    assert questions.f1_ranges(corpus.proposals["PEP-50"], "change") == [
        (date(2020, 1, 8), date(2020, 12, 24))]
    assert found["items_that_would_differ"] == [change.id]


# --- refs ----------------------------------------------------------------------------------------


def test_refs(corpus):
    assert questions.ref(corpus.proposals["PEP-1"]) == 'PEP 1 ("Purpose and Guidelines")'
    assert questions.ref(corpus.proposals["EIP-100"]) == 'EIP-100 ("Hub Hundred")'
    assert questions.ref(corpus.proposals["ERC-20"]) == 'ERC-20 ("Token Standard")'
    # Absent at S: the latest known title.
    assert questions.ref(corpus.proposals["EIP-108"]) == 'EIP-108 ("Deleted Proposal")'
    untitled = Proposal("eth", "EIP", 9)
    untitled.versions.append(Version("EIPs", ts("2020-01-01"), "c" * 40, "EIPS/eip-9.md",
                                     "b" * 40, "create", {"status": "Draft"}, "Draft", "Draft"))
    assert questions.ref(untitled) == "EIP-9"
    retitled = Proposal("pep", "PEP", 9)
    for when, header in [("2020-01-01", {"title": "Old Title"}), ("2021-01-01", {})]:
        retitled.versions.append(Version("peps", ts(when), "c" * 40, "peps/pep-0009.rst",
                                         "b" * 40, "modify", header, None, "missing"))
    assert questions.ref(retitled) == 'PEP 9 ("Old Title")'


# --- F2 ------------------------------------------------------------------------------------------


def test_f2_one_sided_counts(built, corpus, hand):
    positives = of(built, "F2", "positive")
    negatives = of(built, "F2", "negative")
    assert [(i.proposal, i.gold) for i in positives] == [("PEP-10", [11]), ("PEP-13", [14])]
    assert len(negatives) == len(positives)
    assert built.pools[("F2", "negative", "pep")] == 3  # PEP 1, 11 and 12 mention another PEP
    present = {p.number for p in corpus.proposals.values()
               if p.family == "pep" and p.status_at(EOD_S) is not None}
    for i in negatives:
        assert i.gold == []
        assert i.proposal in {"PEP-1", "PEP-11", "PEP-12"}
        p = corpus.proposals[i.proposal]
        assert mentions("pep", p.body_at(EOD_S, hand.readers), p.number) & present
    for i in positives + negatives:
        assert i.as_of == "2026-10-27" and i.answer_type == "set" and i.choices is None
        assert i.arms == ("H+", "H-")


def test_f2_items_record_whether_the_asked_peps_own_episodes_can_show_the_gold(built):
    """An F2 item is recoverable in H+ when every gold member is listed in the asked PEP's
    Superseded-By (or Replaced-By) in some version of its file or mentioned in the body of some
    version; in H-, where those headers are withheld, when every member is mentioned. A negative
    is recoverable in both arms."""
    flags = {i.proposal: i.meta["recoverable"] for i in of(built, "F2")}
    # PEP 10's superseder, 11, is recorded only on 11's side (Replaces), and 10 never mentions it.
    assert flags["PEP-10"] == {"H+": False, "H-": False}
    # PEP 13 lists 14 in Superseded-By but never mentions it: recoverable in H+ only.
    assert flags["PEP-13"] == {"H+": True, "H-": False}
    for i in of(built, "F2", "negative"):
        assert i.meta["recoverable"] == {"H+": True, "H-": True}, i.id
    assert not any("recoverable" in i.meta for i in built if i.family != "F2")


def test_a_superseder_mentioned_in_an_earlier_version_is_recoverable_in_both_arms():
    h = Hand()
    old = h.add("pep", "PEP", 34, "Mentioned Earlier",
                [("2001-01-01", "Draft"), ("2006-01-01", "Superseded")])
    # The first version's body names PEP 35; the second's, the one at S, does not.
    text = render("pep", 34, "Mentioned Earlier", "Draft", (), "See PEP 35 instead.\n")
    blob = hashlib.sha1(text.encode()).hexdigest()
    h.readers["peps"].blobs[blob] = text
    old.versions[0] = replace(old.versions[0], blob=blob)
    assert questions.f2_recoverable(old, [35], h.readers) == {"H+": True, "H-": True}
    # A member that no version lists or mentions is not recoverable, nor is a set holding one.
    assert questions.f2_recoverable(old, [36], h.readers) == {"H+": False, "H-": False}
    assert questions.f2_recoverable(old, [35, 36], h.readers) == {"H+": False, "H-": False}
    assert questions.f2_recoverable(old, [], h.readers) == {"H+": True, "H-": True}


# --- F3 ------------------------------------------------------------------------------------------


def test_f3_closure(built):
    cycle = {1: [2], 2: [3], 3: [2]}
    assert questions.closure(cycle, 1) == {2: 1, 3: 2}
    assert questions.closure(cycle, 2) == {3: 1}  # the proposal itself is excluded
    assert questions.closure({1: [1]}, 1) == {}
    assert questions.closure({1: [2, 3], 2: [3]}, 1) == {2: 1, 3: 1}  # shortest distance
    graph = {1: [2], 3: [2, 4], 5: [2, 4, 6], 7: [6, 8]}
    # In-degree: 2 three times, 4 and 6 twice, 8 once; ties go to the lower number.
    assert questions.hubs(graph, 5) == [2, 4, 6, 8]
    ones = {1: [9], 2: [8], 3: [7], 4: [6], 5: [10], 11: [12]}
    assert questions.hubs(ones, 5) == [6, 7, 8, 9, 10]
    assert [questions.size_stratum(n) for n in (0, 1, 2, 4, 5, 14, 15, 99)] == [
        None, None, "size-2-4", "size-2-4", "size-5-14", "size-5-14", "size-15-plus",
        "size-15-plus",
    ]
    f3 = of(built, "F3")
    assert [(i.proposal, i.stratum, i.gold) for i in f3] == [("EIP-105", "size-2-4", [106, 107])]
    assert built.notes["F3"]["hubs"] == [100, 101, 106, 20, 102]
    assert built.notes["F3"]["dropped_as_hub_recall"] == ["EIP-103"]
    assert built.notes["F3"]["dropped_requires"] == []
    assert f3[0].arms == ("H+", "H-") and f3[0].as_of == "2026-10-27"


# --- F4 ------------------------------------------------------------------------------------------


def closure_at(corpus, number: int, moment: int) -> set[int]:
    present = {p.number: p for p in corpus.proposals.values()
               if p.family == "eth" and p.status_at(moment) is not None}
    seen, todo = set(), [number]
    while todo:
        n = todo.pop()
        for m in present[n].requires_at(moment) if n in present else ():
            if m != number and m in present and m not in seen:
                seen.add(m)
                todo.append(m)
    return seen


def test_f4_final_members(built, corpus):
    f4 = {i.proposal: i for i in of(built, "F4")}
    assert set(f4) == {"EIP-103", "EIP-104", "EIP-105"}
    by_number = {p.number: p for p in corpus.proposals.values() if p.family == "eth"}
    for i in f4.values():
        d = day_of(i)
        number = corpus.proposals[i.proposal].number
        members = closure_at(corpus, number, eod(d))
        assert 2 <= len(members) <= 14 and d <= S - 7 * DAY
        finals = sorted(m for m in members if by_number[m].status_at(eod(d)) == "Final")
        assert i.gold == finals, i.id
        assert i.meta["closure"] == sorted(members)
        for k in range(-7, 8):
            moment = eod(d + k * DAY)
            assert closure_at(corpus, number, moment) == members, (i.id, k)
            assert sorted(m for m in members if by_number[m].status_at(moment) == "Final") == finals
        assert corpus.proposals[i.proposal].status_at(eod(d)) in CHOICES["eth"]
        assert i.arms == ("H+",) and i.answer_type == "set"
    ranges = questions.f4_ranges(corpus)
    became_final = date(2020, 1, 3)  # EIP-107, a member of 105's closure
    for k in range(-7, 7):
        assert not questions.in_ranges(ranges["EIP-105"], became_final + k * DAY), k
    assert questions.in_ranges(ranges["EIP-105"], became_final - 8 * DAY)
    assert questions.in_ranges(ranges["EIP-105"], became_final + 7 * DAY)
    # EIP-104 was Stagnant from 2021-01-01 to 2022-01-31: none of those days is sampled.
    stagnant = [date(2021, 1, 1) + k * DAY for k in range(396)]
    assert not any(questions.in_ranges(ranges["EIP-104"], d) for d in stagnant)
    assert questions.in_ranges(ranges["EIP-104"], date(2020, 12, 31))
    assert built.notes["F4"]["gold_differs_from_s"] == sum(
        1 for i in f4.values() if i.meta["gold_at_s"] != i.gold
    )


def test_f4_closure_size_bounds():
    """A closure of 14 members is asked about; one of 15 is not (nor of 1)."""
    h = Hand()
    for n in range(200, 216):
        h.add("eth", "EIP", n, f"Leaf {n}", [("2019-01-01", "Final")])
    h.add("eth", "EIP", 300, "Fourteen", [("2019-01-01", "Draft")],
          headers=[("requires", ", ".join(str(n) for n in range(200, 214)))])
    h.add("eth", "EIP", 301, "Fifteen", [("2019-01-01", "Draft")],
          headers=[("requires", ", ".join(str(n) for n in range(200, 215)))])
    h.add("eth", "EIP", 302, "One", [("2019-01-01", "Draft")], headers=[("requires", "200")])
    ranges = questions.f4_ranges(h.corpus())
    assert set(ranges) == {"EIP-300"}
    assert ranges["EIP-300"] == [(date(2019, 1, 8), S - 7 * DAY)]


# --- F5 ------------------------------------------------------------------------------------------


def count_at(corpus, family: str, status: str, d: date) -> int:
    return sum(1 for p in corpus.proposals.values()
               if p.family == family and p.status_at(eod(d)) == status)


def test_f5_shift(built, corpus):
    f5 = {i.id: i for i in of(built, "F5")}
    assert len(of(built, "F5", corpus="pep")) == 27 and len(of(built, "F5", corpus="eth")) == 22
    moved = {"F5-count-pep-Final-2020-01-06", "F5-count-pep-Draft-2020-01-06",
             "F5-count-eth-Final-2020-01-06", "F5-count-eth-Draft-2020-01-06"}
    assert moved <= set(f5)
    assert "F5-count-pep-Rejected-2020-01-01" in f5 and "F5-count-eth-Review-2020-01-01" in f5
    assert not any(k.startswith(("F5-count-pep-Final-2020-01-0", "F5-count-eth-Final-2020-01-0"))
                   and k not in moved for k in f5)
    # Unstable at S, so dropped rather than moved.
    assert "F5-count-pep-Final-2026-10-27" not in f5
    assert "F5-count-eth-Draft-2026-10-27" not in f5
    assert "F5-count-eth-Review-2026-10-27" not in f5
    assert {"F5-count-pep-Draft-2026-10-27", "F5-count-eth-Final-2026-10-27"} <= set(f5)
    assert built.notes["F5"]["dropped"] == [
        "F5-count-eth-Draft-2026-10-27", "F5-count-eth-Review-2026-10-27",
        "F5-count-pep-Final-2026-10-27",
    ]
    assert f5["F5-count-pep-Final-2020-01-06"].gold == 3
    assert f5["F5-count-eth-Draft-2018-01-01"].gold == 4
    assert f5["F5-count-eth-Final-2026-10-27"].gold == 6
    for i in f5.values():
        d = day_of(i)
        status = i.meta["status"]
        assert i.gold == count_at(corpus, i.corpus, status, d)
        assert i.proposal is None and i.answer_type == "count" and i.arms == ("H+",)
        window = range(-3, 1) if d == S else range(-3, 4)
        for p in corpus.proposals.values():
            if p.family == i.corpus:
                members = {p.status_at(eod(d + k * DAY)) == status for k in window}
                assert len(members) == 1, (i.id, p.pid)


# --- CF ------------------------------------------------------------------------------------------


def test_cf_gold_permuted(built, corpus, hand):
    cf = of(built, "CF")
    assert {i.proposal for i in of(cf, "CF", corpus="pep")} == {
        "PEP-1", "PEP-2", "PEP-10", "PEP-11", "PEP-12", "PEP-13"}
    assert {i.proposal for i in of(cf, "CF", corpus="eth")} == {
        "ERC-20", "EIP-100", "EIP-101", "EIP-102", "EIP-104", "EIP-106", "EIP-107"}
    for i in cf:
        p = corpus.proposals[i.proposal]
        d = day_of(i)
        true = p.status_at(eod(d))
        assert permute(i.corpus, true) != true, i.id
        assert i.gold == permute(i.corpus, true) and i.meta["true_status"] == true
        assert held(p, d, true) and d <= S - 7 * DAY
        assert i.gold_outcome == outcome_class(i.corpus, i.gold)
        body = p.body_at(eod(d), hand.readers).casefold()
        words = ("accept", "reject", "withdraw", "final", "supersede")
        assert i.meta["decision_word"] == any(w in body for w in words)
        assert i.arms == ("CF",) and i.stratum == "counterfactual"
    flagged = {i.proposal for i in cf if i.meta["decision_word"]}
    assert flagged == {"PEP-2", "EIP-102"}
    pool = built.notes["CF"]["pool"]
    assert [entry["proposal"] for entry in pool["pep"]][:2] == ["PEP-10", "PEP-1"]
    assert [entry["proposal"] for entry in pool["eth"]][:4] == [
        "ERC-20", "EIP-100", "EIP-101", "EIP-106"]


def test_cf_items_record_header_fields_that_point_at_the_true_status(built, corpus):
    """A CF item is flagged when its file at D has a header field other than the status, which
    the permutation leaves alone, that records the true status: a last-call-deadline under any of
    its names, a Superseded-By, Replaced-By, withdrawal-reason or Accepted with a value, or a
    Resolution whose case-folded value contains a decision word. A title names its proposal's
    subject, so "Final In 2020" and "Withdrawn One" are no cues."""
    cues = {i.proposal: i.meta["header_cue_fields"] for i in of(built, "CF")}
    assert {pid: fields for pid, fields in cues.items() if fields} == {
        "PEP-12": ["superseded-by"],  # Superseded-By: 12, 9999 has a value
        "PEP-13": ["superseded-by"],  # Superseded-By: 14; its title, "Final In 2020", is no cue
    }
    assert cues["EIP-102"] == []  # its title, "Withdrawn One", is no cue
    for i in of(built, "CF"):
        assert i.meta["header_cue"] is bool(i.meta["header_cue_fields"]), i.id
    assert not any("header_cue" in i.meta for i in built if i.family != "CF")
    text = questions.report(built, corpus)
    assert f"that records the true status ({questions.HEADER_CUE_RULE}): 2 of " in text
    assert "- CF-counterfactual-PEP-13-" in text and "(superseded-by)" in text
    assert "(title" not in text


@pytest.mark.parametrize(("family", "field", "flagged"), [
    ("eth", ("last-call-deadline", "2024-06-26"), ["last-call-deadline"]),
    ("eth", ("review-period-end", "2019-01-01"), ["review-period-end"]),  # an earlier name
    ("eth", ("last-call-deadline", ""), ["last-call-deadline"]),  # the field, whatever its value
    ("pep", ("Superseded-By", "566"), ["superseded-by"]),
    ("pep", ("Replaced-By", "3000"), ["replaced-by"]),
    ("pep", ("Superseded-By", ""), []),  # without a value
    # A withdrawal-reason or an Accepted with a value records the decision, whatever it says.
    ("eth", ("withdrawal-reason", "No longer maintained"), ["withdrawal-reason"]),
    ("eth", ("withdrawal-reason", ""), []),
    ("pep", ("Accepted", "https://mail.example/pipermail/dev/2013-February/2.html"), ["accepted"]),
    ("pep", ("Accepted", ""), []),
    # A Resolution counts only when its value says which way the decision went.
    ("pep", ("Resolution", "https://discuss.example/t/pep-7-acceptance/9"), ["resolution"]),
    ("pep", ("Resolution", "https://mail.example/pipermail/dev/2013-February/1.html"), []),
    # No other field is read: subjects, people, venues and relations record no status.
    ("eth", ("description", "Withdrawals as operations"), []),
    ("eth", ("author", "Final Rejectson (@final-reject)"), []),
    ("pep", ("Discussions-To", "https://discuss.example/t/accept-or-reject/1"), []),
    ("pep", ("Replaces", "3000"), []),
    ("eth", ("requires", "1559"), []),
    ("pep", ("Status", "Final"), []),  # a status field is the one the permutation edits
])
def test_header_cues(family, field, flagged):
    status = "Accepted" if family == "pep" else "Review"
    text = render(family, 7, "Plain Title", status, [field],
                  "The body says final and accept, but a body is no header.\n")
    assert questions.header_cues(family, text) == flagged


@pytest.mark.parametrize("family", ["pep", "eth"])
def test_a_title_is_no_header_cue(family):
    """A title names its proposal's subject, so its words record no status."""
    status = "Accepted" if family == "pep" else "Review"
    text = render(family, 7, "Reject Withdrawn Finals, Accepted", status, [], "Body.\n")
    assert questions.header_cues(family, text) == []


def test_the_fields_a_header_cue_reads_are_named_once():
    """The header reader names every field a CF header cue reads once (section 7), and H-
    withholds each of them (section 8): the last-call names whatever their value, the
    Superseded-By, withdrawal-reason and Accepted names with a value, and the Resolution when its
    value holds a decision word. Superseded-By is also read through its names."""
    from asofbench.headers import (
        ACCEPTED_NAMES,
        CUE_NAMES,
        CUE_WITH_A_VALUE,
        LAST_CALL_NAMES,
        RESOLUTION_NAMES,
        SUPERSEDED_NAMES,
        WITHDRAWAL_REASON_NAMES,
        WITHHELD,
        superseders,
    )

    assert CUE_WITH_A_VALUE == (*SUPERSEDED_NAMES, *WITHDRAWAL_REASON_NAMES, *ACCEPTED_NAMES)
    assert CUE_NAMES == (*LAST_CALL_NAMES, *CUE_WITH_A_VALUE, *RESOLUTION_NAMES)
    assert set(CUE_NAMES) <= set(WITHHELD)
    header = {name: str(k) for k, name in enumerate(SUPERSEDED_NAMES, 1)}
    assert superseders(header) == list(range(1, len(SUPERSEDED_NAMES) + 1))
    for name in LAST_CALL_NAMES:
        text = render("eth", 7, "Plain Title", "Review", [(name, "2024-01-01")], "Body.\n")
        assert questions.header_cues("eth", text) == [name]
    for name in CUE_WITH_A_VALUE:
        text = render("pep", 7, "Plain Title", "Accepted", [(name, "8")], "Body.\n")
        assert questions.header_cues("pep", text) == [name]
        text = render("pep", 7, "Plain Title", "Accepted", [(name, "")], "Body.\n")
        assert questions.header_cues("pep", text) == []
    for name in RESOLUTION_NAMES:
        for word in questions.DECISION_WORDS:
            text = render("pep", 7, "Plain Title", "Accepted",
                          [(name, f"https://example.invalid/{word.upper()}")], "Body.\n")
            assert questions.header_cues("pep", text) == [name]
        text = render("pep", 7, "Plain Title", "Accepted", [(name, "https://example.invalid/1")],
                      "Body.\n")
        assert questions.header_cues("pep", text) == []


# --- labelling -----------------------------------------------------------------------------------


def test_labelling_pairs(built, corpus, hand):
    pairs = {record["pair_id"]: record for record in built.labelling}
    assert set(pairs) == {"PEP-1~PEP-10", "PEP-1~PEP-12", "ERC-20~EIP-105"}
    one_ten = pairs["PEP-1~PEP-10"]
    assert (one_ten["a"], one_ten["b"], one_ten["corpus"]) == ("PEP-1", "PEP-10", "pep")
    assert (one_ten["a_mentions_b"], one_ten["b_mentions_a"]) == (True, False)
    assert one_ten["a_title"] == "Purpose and Guidelines" and one_ten["b_title"] == "Old Way"
    # Section 9: the published record says where each mention is, never the text around it.
    (span,) = one_ten["a_mention_spans"]
    assert one_ten["b_mention_spans"] == []
    pep1 = corpus.proposals["PEP-1"].version_at(EOD_S)
    assert (span["repo"], span["path"], span["commit"]) == ("peps", pep1.path, corpus.pins["peps"])
    text = hand.readers["peps"].read(pep1.blob)
    assert text[span["start"]:span["end"]] == "PEP 10"  # character offsets into the blob
    # The annotator sheet's passage around it is the 200 characters centred on the mention.
    body = header_span("pep", text)[1]
    assert questions.passage(text[body:], span["start"] - body, span["end"] - body) == (
        "This PEP explains the process. See PEP 10 for an example of an older approach.")
    erc = pairs["ERC-20~EIP-105"]
    assert (erc["a_mentions_b"], erc["b_mentions_a"]) == (False, True)
    (erc_span,) = erc["b_mention_spans"]
    eip105 = corpus.proposals["EIP-105"].version_at(EOD_S)
    assert hand.readers["EIPs"].read(eip105.blob)[erc_span["start"]:erc_span["end"]] == "ERC-20"
    for record in built.labelling:
        assert record["canary"] == CANARY
        assert "a_snippets" not in record and "b_snippets" not in record
        assert set(record) == {"pair_id", "corpus", "rank", "a", "a_title", "a_mentions_b",
                               "a_mention_spans", "a_source", "b", "b_title", "b_mentions_a",
                               "b_mention_spans", "b_source", "canary"}
        published = json.dumps(record)
        assert "explains the process" not in published and "balances" not in published


def test_snippets_are_centred_and_at_most_two():
    body = ("x" * 300) + " PEP 7 " + ("y" * 300) + " PEP 7 again " + ("z" * 50) + " PEP 7 end"
    found = questions.snippets("pep", body, 7)
    assert len(found) == 2
    assert all(len(s) <= 200 and "PEP 7" in s for s in found)
    first = found[0]
    assert first.index("PEP 7") in range(90, 101)  # centred on the mention
    assert questions.snippets("pep", "No mention here.", 7) == []


# --- templates -----------------------------------------------------------------------------------


def test_templates_match_the_registration():
    registered = re.findall(r'Template: "([^"]*)"', PREREGISTRATION.read_text(encoding="utf-8"))
    assert dict(zip(("F1", "F2", "F3", "F4", "F5"), registered, strict=True)) == TEMPLATES


PEP_CHOICES = (
    "Draft, Active, Accepted, Provisional, Deferred, Rejected, Withdrawn, Final, Superseded"
)
ETH_CHOICES = ("Draft, Review, Last Call, Final, Withdrawn, Living, Accepted, Active, Abandoned, "
               "Superseded, Deferred, Replaced")


def test_templates_exact(built):
    def one(family, proposal, stratum=None):
        (item,) = [i for i in of(built, family, stratum) if i.proposal == proposal]
        return item

    pep1 = one("F1", "PEP-1", "change")
    assert pep1.question == (
        f'What was the status of PEP 1 ("Purpose and Guidelines") as of the end of {pep1.as_of} '
        f"(UTC)? Answer with exactly one value from: {PEP_CHOICES}."
    )
    erc = one("F1", "ERC-20", "change")
    assert erc.question == (
        f'What was the status of ERC-20 ("Token Standard") as of the end of {erc.as_of} (UTC)? '
        f"Answer with exactly one value from: {ETH_CHOICES}. "
        "Before 2023-10-26, ERCs were filed in the EIP repository as EIP-20."
    )
    assert one("F2", "PEP-10").question == (
        "According to the PEP repository as of the end of 2026-10-27 (UTC), which PEP or PEPs "
        'superseded PEP 10 ("Old Way")? Answer with a list of PEP numbers, or an empty list if '
        "none did."
    )
    assert one("F3", "EIP-105").question == (
        'As of the end of 2026-10-27 (UTC), which proposals does EIP-105 ("Cycle Start") require, '
        "directly or through the proposals it requires? Answer with a list of every proposal "
        "number in that set, or an empty list."
    )
    f4 = one("F4", "EIP-105")
    assert f4.question == (
        f'As of the end of {f4.as_of} (UTC), which of the proposals that EIP-105 ("Cycle Start") '
        "required, directly or through the proposals it required, had status Final? Answer with a "
        "list of proposal numbers, or an empty list."
    )
    f5 = {i.id: i for i in of(built, "F5")}
    assert f5["F5-count-pep-Final-2020-01-06"].question == (
        "How many PEPs (PEP 0 is excluded) had status Final as of the end of 2020-01-06 (UTC)? "
        "Answer with one integer."
    )
    assert f5["F5-count-eth-Withdrawn-2026-10-27"].question == (
        "How many proposals in the Ethereum EIP and ERC series had status Withdrawn as of the end "
        "of 2026-10-27 (UTC)? Answer with one integer."
    )
    cf = one("CF", "ERC-20")
    assert cf.question == (
        f'What was the status of ERC-20 ("Token Standard") as of the end of {cf.as_of} (UTC)? '
        f"Answer with exactly one value from: {ETH_CHOICES}. "
        "Before 2023-10-26, ERCs were filed in the EIP repository as EIP-20."
    )
    for i in built:
        if i.proposal is not None and corpus_series(i.proposal) == "ERC":
            assert i.question.endswith(
                f" Before 2023-10-26, ERCs were filed in the EIP repository as EIP-"
                f"{i.proposal.split('-')[1]}."), i.id
        else:
            assert "ERCs were filed" not in i.question, i.id


def corpus_series(pid: str) -> str:
    return pid.split("-")[0]


# --- items, ids, sources -------------------------------------------------------------------------


def test_ids_ranks_sources_and_commits(built, corpus):
    for i in built:
        if i.family == "F5":
            assert i.id == f"F5-count-{i.corpus}-{i.meta['status']}-{i.as_of}"
        else:
            assert i.id == f"{i.family}-{i.stratum}-{i.proposal}-{i.as_of}"
            assert i.meta["ref"] == questions.ref(corpus.proposals[i.proposal])
        d = day_of(i)
        assert i.as_of_commits == corpus.as_of_commits(d)
        if i.family in ("F1", "CF"):
            p = corpus.proposals[i.proposal]
            v = p.version_at(eod(d))
            assert i.source == {"repo": v.repo, "path": v.path,
                                "commit": i.as_of_commits[v.repo]}
        else:
            assert i.source is None
    erc = next(i for i in of(built, "F1", "change") if i.proposal == "ERC-20")
    assert erc.source["repo"] == "EIPs" and erc.source["path"] == "EIPS/eip-20.md"
    ranks = defaultdict(list)
    for i in built:
        ranks[(i.family, i.stratum, i.corpus)].append(i.rank)
    for key, found in ranks.items():
        if key[0] != "F5":
            assert found == list(range(len(found))), key


def test_hash_order_and_day_choice(built, corpus):
    """Each sampled stratum is a prefix of the hash order; each date is pick_day's choice."""
    from asofbench.select import order, pick_day

    change = of(built, "F1", "change", "eth")
    pool = [p.pid for p in corpus.proposals.values()
            if p.family == "eth" and p.status_at(EOD_S) is not None
            and questions.f1_ranges(p, "change")]
    assert [i.proposal for i in change] == order(pool, "F1", "change", str)
    for i in change:
        days = questions.expand(questions.f1_ranges(corpus.proposals[i.proposal], "change"))
        assert i.as_of == pick_day(days, "F1", "change", i.proposal).isoformat()


# --- outputs -------------------------------------------------------------------------------------


QUESTION_KEYS = {"id", "family", "stratum", "corpus", "proposal", "as_of", "arms", "question",
                 "answer_type", "choices", "canary"}
GOLD_KEYS = {"id", "gold", "gold_outcome", "source", "as_of_commits", "meta", "canary"}
OUTPUTS = ("questions.jsonl", "gold.jsonl", "pilot.json", "labelling/mention_pairs.jsonl",
           "timelines.jsonl", "edges.jsonl", "REPORT.md")


def read_jsonl(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n") and "\r" not in text
    return [json.loads(line) for line in text.splitlines()]


def test_ids_unique_and_output_deterministic(built, corpus, hand, tmp_path):
    ids = [i.id for i in built]
    assert len(ids) == len(set(ids))
    write_outputs(built, corpus, tmp_path / "a")
    write_outputs(build_items(corpus, hand.readers), corpus, tmp_path / "b")
    for name in OUTPUTS:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes(), name
    out = tmp_path / "a"
    for name in ("questions.jsonl", "gold.jsonl", "labelling/mention_pairs.jsonl",
                 "timelines.jsonl", "edges.jsonl"):
        for record in read_jsonl(out / name):
            assert record["canary"] == CANARY, name
    pilot = json.loads((out / "pilot.json").read_text(encoding="utf-8"))
    assert pilot["canary"] == CANARY
    assert CANARY in (out / "REPORT.md").read_text(encoding="utf-8")
    # JSON lines are written compactly with sorted keys.
    line = (out / "questions.jsonl").read_text(encoding="utf-8").splitlines()[0]
    assert line == json.dumps(json.loads(line), sort_keys=True, ensure_ascii=False,
                              separators=(",", ":"))


def test_output_formats(built, corpus, tmp_path):
    write_outputs(built, corpus, tmp_path)
    questions_rows = read_jsonl(tmp_path / "questions.jsonl")
    gold_rows = read_jsonl(tmp_path / "gold.jsonl")
    assert [row["id"] for row in questions_rows] == [i.id for i in built]
    assert [row["id"] for row in gold_rows] == [i.id for i in built]
    assert all(set(row) == QUESTION_KEYS for row in questions_rows)
    assert all(set(row) == GOLD_KEYS for row in gold_rows)
    by_id = {i.id: i for i in built}
    for row in questions_rows:
        item = by_id[row["id"]]
        assert row["arms"] == list(item.arms) and row["question"] == item.question
        assert row["choices"] == (None if item.choices is None else list(item.choices))
    for row in gold_rows:
        assert row["gold"] == by_id[row["id"]].gold
        assert row["meta"]["rank"] == by_id[row["id"]].rank
    pilot = json.loads((tmp_path / "pilot.json").read_text(encoding="utf-8"))
    groups = {(g["family"], g["stratum"], g["corpus"]): g for g in pilot["groups"]}
    change = of(built, "F1", "change", "pep")
    assert groups[("F1", "change", "pep")]["ids"] == [i.id for i in change if i.rank < 50]
    assert groups[("F1", "current", "eth")]["ids"] == [
        i.id for i in of(built, "F1", "current", "eth") if i.rank < 25]
    assert groups[("CF", "counterfactual", "pep")]["ids"] == [i.id for i in of(built, "CF", None,
                                                                                "pep")]
    assert pilot["ids"] == [i for g in pilot["groups"] for i in g["ids"]]
    assert pilot["m0_graphiti_proposals"] == [i.proposal for i in change if i.rank < 50]
    timelines = {row["pid"]: row for row in read_jsonl(tmp_path / "timelines.jsonl")}
    assert set(timelines) == set(corpus.proposals)
    pep2 = timelines["PEP-2"]
    assert [iv["status"] for iv in pep2["intervals"]] == ["Draft", "absent", "Rejected"]
    assert pep2["intervals"][0]["start"] == "2010-01-01T12:00:00Z"
    assert pep2["intervals"][-1]["end"] is None and pep2["title"] == "Deleted And Readded"
    assert timelines["EIP-108"]["present_at_s"] is False
    report = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "| F1 | change | pep |" in report and "all eligible" in report


def test_write_outputs_needs_the_build(corpus, tmp_path):
    with pytest.raises(TypeError):
        write_outputs([], corpus, tmp_path)


def test_edges_are_the_header_relations_at_s(tmp_path):
    """Section 9's relation gold: PEP m supersedes PEP n when, at S, n lists m in Superseded-By
    or m lists n in Replaces; N requires M when N's requires header at S lists M, in either
    corpus. Self-references and numbers that are no proposal at S are dropped."""
    h = Hand()
    h.add("pep", "PEP", 8, "Eight", [("2001-01-01", "Draft")], headers=[("Superseded-By", "9")])
    h.add("pep", "PEP", 9, "Nine", [("2001-01-01", "Draft")], headers=[("Superseded-By", "10")])
    h.add("pep", "PEP", 10, "Ten", [("2002-01-01", "Final")],
          headers=[("Requires", "9, 10, 4000")])
    # A one-sided Replaces record is an edge; itself and a PEP deleted before S are not.
    h.add("pep", "PEP", 11, "Eleven", [("2003-01-01", "Final")],
          headers=[("Replaces", "9, 11, 12")])
    h.add("pep", "PEP", 12, "Twelve", [("2003-01-01", "Draft"), ("2004-01-01", DELETE)])
    h.add("eth", "EIP", 5, "Five", [("2016-01-01", "Draft")], headers=[("requires", "5, 6, 999")])
    h.add("eth", "ERC", 6, "Six", [("2016-01-01", "Final")])
    corpus = h.corpus()
    records = questions.edge_records(corpus)
    assert all(r.pop("canary") == CANARY for r in records)
    # Sorted by relation, then source id, then target id, each compared as a string.
    assert records == [
        {"corpus": "eth", "relation": "requires", "source": "EIP-5", "target": "ERC-6"},
        {"corpus": "pep", "relation": "requires", "source": "PEP-10", "target": "PEP-9"},
        {"corpus": "pep", "relation": "supersedes", "source": "PEP-10", "target": "PEP-9"},
        {"corpus": "pep", "relation": "supersedes", "source": "PEP-11", "target": "PEP-9"},
        {"corpus": "pep", "relation": "supersedes", "source": "PEP-9", "target": "PEP-8"},
    ]
    # write_outputs writes them as data/v0.1/edges.jsonl, one record per line.
    write_outputs(build_items(corpus, h.readers), corpus, tmp_path)
    lines = read_jsonl(tmp_path / "edges.jsonl")
    assert [{k: v for k, v in r.items() if k != "canary"} for r in lines] == records
    assert all(r["canary"] == CANARY for r in lines)


def test_the_report_publishes_the_trivial_answer_floors(built, corpus):
    """For each family, stratum and corpus: the items, the most common gold value and how many
    items have it, and for F1 how many items have the status at S as their gold. When values tie
    for the most items, the value shown is the one whose JSON text sorts first, marked with the
    number of values tied."""
    text = questions.report(built, corpus)
    section = text.split("\n## Trivial-answer floors\n", 1)[1].split("\n## ", 1)[0]
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")]
            for line in section.splitlines() if line.startswith("| ")]
    assert rows[0] == ["Family", "Stratum", "Corpus", "Items", "Most common gold", "With it",
                       "Share", "Gold is the status at S", "Share"]
    by_key = {tuple(row[:3]): row for row in rows[1:]}
    groups = defaultdict(list)
    for i in built:
        groups[(i.family, i.stratum, i.corpus)].append(i)
    assert set(by_key) == set(groups)
    for key, found in groups.items():
        counts = defaultdict(int)
        for i in found:
            counts[json.dumps(i.gold)] += 1
        top = min(counts, key=lambda value: (-counts[value], value))
        value = json.loads(top)
        shown = value if isinstance(value, str) else json.dumps(value)
        n = counts[top]
        tied = sum(1 for count in counts.values() if count == n)
        if tied > 1:
            shown += f" (one of {tied} tied)"
        assert by_key[key][3:7] == [str(len(found)), shown, str(n), f"{n / len(found):.3f}"], key
        if key[0] == "F1":
            same = sum(1 for i in found if i.gold == i.meta["status_at_s"])
            assert by_key[key][7:] == [str(same), f"{same / len(found):.3f}"], key
        else:
            assert by_key[key][7:] == ["-", "-"], key
    # A unique mode is shown alone; a tie (PEP 10's [11] and PEP 13's [14]) is marked.
    assert by_key[("F2", "negative", "pep")][4:7] == ["[]", "2", "1.000"]
    assert by_key[("F2", "positive", "pep")][4:7] == ["[11] (one of 2 tied)", "1", "0.500"]
    assert "A tie shows the gold value whose JSON text sorts first" in section


def test_the_report_counts_the_f2_items_the_asked_peps_episodes_can_show(built, corpus):
    text = questions.report(built, corpus)
    section = text.split("\n## F2, supersession\n", 1)[1].split("\n## ", 1)[0]
    assert "H+ 1 of 2, H- 0 of 2" in section
    assert "Not recoverable in H+: F2-positive-PEP-10-2026-10-27" in section
    assert ("Not recoverable in H-: F2-positive-PEP-10-2026-10-27, F2-positive-PEP-13-2026-10-27"
            in section)


def test_build_records_pools(built):
    assert built.pools[("F1", "change", "pep")] == 7
    assert built.pools[("F1", "change", "eth")] == 9
    assert built.pools[("F3", "size-2-4", "eth")] == 1
    assert built.pools[("F3", "size-15-plus", "eth")] == 0
    assert built.pools[("labelling", "pep", "pep")] == 2
    assert built.pools[("CF", "counterfactual", "eth")] == 7


def test_utc_date():
    assert questions.utc_date(ts("2020-01-03T23:59:59Z")) == date(2020, 1, 3)
    assert questions.utc_date(ts("2020-01-04T00:00:00Z")) == date(2020, 1, 4)
    assert questions.utc_date(0) == date(1970, 1, 1)
    assert questions.utc_date(int(datetime(2026, 10, 27, tzinfo=UTC).timestamp())) == S
