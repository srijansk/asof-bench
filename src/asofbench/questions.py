"""The question families, the mention-pair sample and the build outputs (spec sections 7 to 10).

Every family follows section 7's common rules: candidates in hash order (``select.order``), one
item per proposal per stratum, and a date that ``select.pick_day`` chooses from the candidate's
eligible days, in date order.

**Day semantics.** The status "at the end of day d" is ``status_at(end_of_day_ts(d))``. An answer
is eligible on day D when it held at the end of every day from D-7 to D+7, so no date after S-7
is eligible except S itself in F1-current and F5. Statuses are therefore read as *day runs*:
maximal runs of days whose end-of-day status is the same. A status that is recorded and reverted
within one UTC day is seen by no end of day, so it splits no run (section 7 reads statuses and
transitions at the end of each day). No day is sampled on which the proposal is absent or its
status is not one a question offers (Stagnant, unknown, missing, Moved or April Fool!).

**Presence at S.** Every item about one proposal concerns a proposal present at S, in every
family (section 7, common rules): candidates come from ``_AtS.present``. Members of a
requirement closure (F4) and the proposals counted in F5 need not be present at S, so those read
every proposal of the corpus.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from .corpora import CANARY, SEED, S, erc_note
from .gitwalk import end_of_day_ts
from .headers import (  # one scanner reads headers, and one table names their fields
    CUE_WITH_A_VALUE,
    LAST_CALL_NAMES,
    RESOLUTION_NAMES,
    _scan,
    header_span,
    number_list,
    superseders,
)
from .mentions import mention_spans, mentions
from .select import MARGIN, eligible_range, order, pick_day
from .statuses import CHOICES, outcome_class, permute
from .timeline import ABSENT, NOT_A_STATUS, Corpus, Proposal, quality_report

# --- the registered numbers, in one place (section 7, section 9, section 13) ---------------------

CORPORA = ("pep", "eth")
FAMILIES = ("F1", "F2", "F3", "F4", "F5", "CF")
# The family and stratum strings, which enter the selection hash (section 7).
F1_STRATA = ("change", "current", "recent")
F2_POSITIVE, F2_NEGATIVE = "positive", "negative"  # positives are all taken, not sampled
F3_STRATA = ("size-2-4", "size-5-14", "size-15-plus")
F4_STRATUM = "closure-final"
F5_STRATUM = "count"  # F5 is not sampled; its items follow the registered dates and statuses
CF_STRATUM = "counterfactual"
LABELLING = "labelling"  # with the corpus as its stratum: "pep" or "eth"

# Items per stratum and corpus. "fixed" sizes must be met; the others take every eligible
# candidate when fewer are eligible.
SIZES: dict[tuple[str, str], int] = {
    ("F1", "change"): 150,
    ("F1", "current"): 50,
    ("F1", "recent"): 60,
    **{("F3", stratum): 40 for stratum in F3_STRATA},
    ("F4", F4_STRATUM): 120,
    ("CF", CF_STRATUM): 15,
    **{(LABELLING, corpus): 100 for corpus in CORPORA},
}
FIXED = {("F1", "change"), ("F1", "current"), ("F4", F4_STRATUM), ("CF", CF_STRATUM),
         *((LABELLING, corpus) for corpus in CORPORA)}

RECENT_FROM = date(2026, 1, 1)  # F1-recent: intervals that began with a transition from then on
F3_HUBS = 5  # the most-required proposals at S whose closure alone is hub recall
F4_CLOSURE = (2, 14)  # closure sizes F4 asks about
F5_MARGIN = 3  # days on each side of D over which F5 membership must not change
F5_DATES: dict[str, tuple[date, ...]] = {
    "pep": (date(2004, 1, 1), date(2008, 1, 1), date(2012, 1, 1), date(2016, 1, 1),
            date(2020, 1, 1), date(2024, 1, 1), S),
    "eth": (date(2018, 1, 1), date(2020, 1, 1), date(2022, 1, 1), date(2024, 1, 1),
            date(2026, 1, 1), S),
}
F5_STATUSES: dict[str, tuple[str, ...]] = {
    "pep": ("Draft", "Rejected", "Withdrawn", "Final"),
    "eth": ("Draft", "Review", "Final", "Withdrawn"),
}
F5_SERIES = {
    "pep": "PEPs (PEP 0 is excluded)",
    "eth": "proposals in the Ethereum EIP and ERC series",
}
CF_POOL = 50  # the most prominent proposals of each corpus at S
DECISION_WORDS = ("accept", "reject", "withdraw", "final", "supersede")
# Section 7's CF header-cue rule (``header_cues``) in the one wording REPORT.md and results.md use.
HEADER_CUE_RULE = ("a last-call-deadline under any of its names, whatever its value; a "
                   "Superseded-By, Replaced-By, withdrawal-reason or Accepted with a value; or a "
                   "Resolution whose case-folded value contains a decision word")
SNIPPET_CHARS = 200
SNIPPETS_PER_SIDE = 2
PILOT = {("F1", "change"): 50, ("F1", "current"): 25}  # section 13, per corpus; plus every CF item

# The question templates of section 7, exactly as registered. CF items ask F1's question.
TEMPLATES = {
    "F1": (
        "What was the status of {ref} as of the end of {D} (UTC)? "
        "Answer with exactly one value from: {choices}."
    ),
    "F2": (
        "According to the PEP repository as of the end of {S} (UTC), which PEP or PEPs "
        "superseded {ref}? Answer with a list of PEP numbers, or an empty list if none did."
    ),
    "F3": (
        "As of the end of {S} (UTC), which proposals does {ref} require, directly or through the "
        "proposals it requires? Answer with a list of every proposal number in that set, or an "
        "empty list."
    ),
    "F4": (
        "As of the end of {D} (UTC), which of the proposals that {ref} required, directly or "
        "through the proposals it required, had status Final? Answer with a list of proposal "
        "numbers, or an empty list."
    ),
    "F5": (
        "How many {series} had status {status} as of the end of {D} (UTC)? "
        "Answer with one integer."
    ),
}
ARMS = {"F1": ("H+", "H-"), "F2": ("H+", "H-"), "F3": ("H+", "H-"), "F4": ("H+",),
        "F5": ("H+",), "CF": ("CF",)}  # section 8; CF items are asked on the counterfactual stream
ANSWER_TYPES = {"F1": "status", "CF": "status", "F2": "set", "F3": "set", "F4": "set",
                "F5": "count"}

DAY = timedelta(days=1)
_EPOCH = date(1970, 1, 1)
Range = tuple[date, date]  # first and last day, both included


@dataclass(frozen=True)
class Item:
    id: str
    family: str
    stratum: str
    corpus: str  # "pep" or "eth"
    proposal: str | None  # the id at S; None for F5
    as_of: str
    arms: tuple[str, ...]
    question: str
    answer_type: str  # "status", "set" or "count"
    choices: tuple[str, ...] | None
    gold: object  # a status, a sorted list of proposal numbers, or a count
    gold_outcome: str | None
    source: dict | None  # F1 and CF: the file that sets the gold, at the as-of commit
    as_of_commits: dict[str, str | None]
    rank: int  # position in the family-stratum-corpus order
    meta: dict


class Build(list):
    """Every item, plus what the other outputs need: the eligible pool of each family, stratum
    and corpus (keyed ``(family, stratum, corpus)``), the mention-pair sample, and notes."""

    def __init__(self, items: Iterable[Item], pools: dict, labelling: list[dict], notes: dict):
        super().__init__(items)
        self.pools = pools
        self.labelling = labelling
        self.notes = notes


# --- days, runs and ranges -----------------------------------------------------------------------


def utc_date(ts: int) -> date:
    """The UTC date of epoch second ``ts``."""
    return _EPOCH + timedelta(days=ts // 86400)


def iso_ts(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class Run:
    status: str | None  # None while the proposal is absent
    first: date
    last: date


def day_runs(p: Proposal, s: date = S) -> list[Run]:
    """The proposal's status at the end of each day, as maximal runs of equal days, up to ``s``.

    An interval covers the days whose end falls inside it: from the day it starts to the day
    before it ends. One that starts and ends within a day covers none, so the runs on either
    side of it join when their statuses agree.
    """
    runs: list[Run] = []
    for i in p.intervals():
        first = utc_date(i.start_ts)
        last = s if i.end_ts is None else min(s, utc_date(i.end_ts) - DAY)
        if first > last:
            continue
        status = None if i.status == ABSENT else i.status
        if runs and runs[-1].status == status:
            runs[-1] = Run(status, runs[-1].first, last)
        else:
            runs.append(Run(status, first, last))
    return runs


def _window(first: date, last: date, s: date = S, margin: int = MARGIN) -> Range | None:
    """The days D of a run from ``first`` to ``last`` whose margins lie inside the run."""
    return eligible_range(first, None if last >= s else last + DAY, s, margin)


def _intersect(a: list[Range], b: list[Range]) -> list[Range]:
    out: list[Range] = []
    i = j = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if lo <= hi:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def expand(ranges: list[Range]) -> list[date]:
    """Every day of ``ranges``, in date order."""
    return [lo + k * DAY for lo, hi in ranges for k in range((hi - lo).days + 1)]


def in_ranges(ranges: list[Range], d: date) -> bool:
    return any(lo <= d <= hi for lo, hi in ranges)


def _count(ranges: list[Range]) -> int:
    return sum((hi - lo).days + 1 for lo, hi in ranges)


# --- refs and questions --------------------------------------------------------------------------


def title(p: Proposal, s: date = S) -> str | None:
    """The title at S; if there is none, the latest title the proposal had before S."""
    moment = end_of_day_ts(s)
    found = p.title_at(moment)
    if found:
        return found
    for v in reversed(p.versions):
        if v.recorded_ts <= moment and v.header.get("title"):
            return v.header["title"]
    return None


def ref(p: Proposal, s: date = S) -> str:
    """``PEP 572 ("Assignment Expressions")``, ``EIP-1559 ("...")`` or ``ERC-20 ("...")``."""
    head = f"PEP {p.number}" if p.family == "pep" else p.pid
    found = title(p, s)
    return f'{head} ("{found}")' if found else head


def _ask(p: Proposal, template: str, **values: str) -> str:
    """The template filled in for ``p``; a question about an ERC ends with the ERC note."""
    text = template.format(ref=ref(p), **values)
    return f"{text} {erc_note(p.number)}" if p.series == "ERC" else text


def _proposals(corpus: Corpus, family: str) -> list[Proposal]:
    return sorted((p for p in corpus.proposals.values() if p.family == family),
                  key=lambda p: p.number)


# --- F1 and CF windows ---------------------------------------------------------------------------


def day_transitions(runs: list[Run]) -> list[bool]:
    """For each day run, whether it begins with a transition as the ends of days show them.

    Section 7 reads statuses and transitions at the end of each day, so a change reverted within
    one day is no change. A run begins with a transition when it holds a status and the last
    earlier run that holds one, since the last absence, holds a different one; missing and Moved
    runs (a header without a status) are looked past, an absence is not. This is
    ``Proposal.intervals``' rule applied to the day runs.
    """
    found = []
    last: str | None = None  # of the last run holding a status since the last absence
    for r in runs:
        transition = False
        if r.status is None:
            last = None
        elif r.status not in NOT_A_STATUS:
            transition = last is not None and last != r.status
            last = r.status
        found.append(transition)
    return found


def _recent_spans(runs: list[Run], s: date) -> list[Range]:
    """The days of the runs that began with a transition on or after 2026-01-01 (by the ends of
    days), with each stretch of such days shrunk by the margin on both sides."""
    spans: list[Range] = []
    for r, began in zip(runs, day_transitions(runs), strict=True):
        if not began or r.first < RECENT_FROM:
            continue
        if spans and spans[-1][1] + DAY == r.first:
            spans[-1] = (spans[-1][0], r.last)
        else:
            spans.append((r.first, r.last))
    return [w for first, last in spans if (w := _window(first, last, s))]


def f1_ranges(p: Proposal, stratum: str, s: date = S, runs: list[Run] | None = None) -> list[Range]:
    """The eligible days of ``p`` in F1-change or F1-recent, as ranges in date order.

    F1-change compares each window's status with the proposal's last offered status at S: the
    last status a question offers among its day runs up to S, which is its status at S when a
    question offers that, and otherwise the last offered status it held before S (section 7).
    Stagnant, or any other value no question offers, records no new decision, so going to it from
    an offered status is no change from that status. A proposal with no offered status has no
    window.
    """
    runs = day_runs(p, s) if runs is None else runs
    choices = CHOICES[p.family]
    windows = [
        (r.status, w) for r in runs
        if r.status in choices and (w := _window(r.first, r.last, s))
    ]
    if stratum == "change":
        last = next((r.status for r in reversed(runs) if r.status in choices), None)
        return [w for status, w in windows if status != last]
    if stratum == "recent":
        return _intersect([w for _, w in windows], _recent_spans(runs, s))
    raise ValueError(f"F1 stratum {stratum!r} has no date window")


def f1_current(p: Proposal, s: date = S, runs: list[Run] | None = None) -> bool:
    """Whether the status at S is one a question offers and held at the end of S-7 to S."""
    runs = day_runs(p, s) if runs is None else runs
    if not runs or runs[-1].last != s:
        return False
    last = runs[-1]
    return last.status in CHOICES[p.family] and last.first <= s - MARGIN * DAY


def cf_ranges(p: Proposal, s: date = S, runs: list[Run] | None = None) -> list[Range]:
    """Eligible F1-style days at which the status is one that the permutation changes."""
    runs = day_runs(p, s) if runs is None else runs
    return [
        w for r in runs
        if r.status in CHOICES[p.family] and permute(p.family, r.status) != r.status
        and (w := _window(r.first, r.last, s))
    ]


# --- requires graphs (F3, F4) --------------------------------------------------------------------


def closure(graph: dict[int, Iterable[int]], start: int) -> dict[int, int]:
    """Every node reachable from ``start`` along ``graph``'s edges, with its shortest distance.

    Cycles are safe; ``start`` itself is never a member.
    """
    dist = {start: 0}
    queue = deque([start])
    while queue:
        x = queue.popleft()
        for y in graph.get(x, ()):
            if y not in dist:
                dist[y] = dist[x] + 1
                queue.append(y)
    del dist[start]
    return dist


def hubs(graph: dict[int, Iterable[int]], k: int = F3_HUBS) -> list[int]:
    """The ``k`` most-required nodes by direct in-degree, ties broken by the lower number."""
    indegree = Counter(y for x, ys in graph.items() for y in set(ys) if y != x)
    return [y for y, _ in sorted(indegree.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def size_stratum(size: int) -> str | None:
    if size < 2:
        return None
    if size <= 4:
        return "size-2-4"
    if size <= 14:
        return "size-5-14"
    return "size-15-plus"


@dataclass(frozen=True)
class Segment:
    first: date
    last: date
    closure: frozenset[int]
    gold: frozenset[int]


def _reach(start: int, requires: dict[int, tuple[int, ...]]):
    """The closure of ``start`` among present proposals, and every number its search read."""
    if start not in requires:
        return frozenset(), frozenset((start,))
    seen = {start}
    touched = {start}
    queue = deque([start])
    while queue:
        for y in requires[queue.popleft()]:
            touched.add(y)
            if y in requires and y not in seen:
                seen.add(y)
                queue.append(y)
    return frozenset(seen - {start}), frozenset(touched)


def f4_segments(corpus: Corpus, s: date = S) -> dict[str, list[Segment]]:
    """Each Ethereum proposal's closure and Final members over time, merged where equal.

    The requires graph and the members' statuses can change only on a day when some version is
    recorded, so the state is evaluated at the end of those days only. A proposal's closure is
    searched again only when a number its last search read changed its presence or its requires
    (``touched``), and its gold only when its closure or a member's status changed.
    """
    eth = {p.number: p for p in corpus.proposals.values() if p.family == "eth"}
    by_day: dict[date, set[int]] = defaultdict(set)
    for n, p in eth.items():
        for v in p.versions:
            if utc_date(v.recorded_ts) <= s:
                by_day[utc_date(v.recorded_ts)].add(n)
    requires: dict[int, tuple[int, ...]] = {}  # present proposals only
    status: dict[int, str] = {}
    closures: dict[int, frozenset[int]] = {}
    touched: dict[int, frozenset[int]] = {}
    touched_by: dict[int, set[int]] = defaultdict(set)
    members_of: dict[int, set[int]] = defaultdict(set)
    starts: dict[int, list[tuple[date, frozenset[int], frozenset[int]]]] = defaultdict(list)
    empty: frozenset[int] = frozenset()
    for d in sorted(by_day):
        moment = end_of_day_ts(d)
        graph_changed: set[int] = set()
        status_changed: set[int] = set()
        for n in sorted(by_day[d]):
            v = eth[n].version_at(moment)
            present = v is not None and v.status != ABSENT
            new_requires = tuple(number_list(v.header.get("requires"))) if present else None
            if requires.get(n) != new_requires:
                graph_changed.add(n)
                if new_requires is None:
                    requires.pop(n, None)
                else:
                    requires[n] = new_requires
            new_status = v.status if present else None
            if status.get(n) != new_status:
                status_changed.add(n)
                if new_status is None:
                    status.pop(n, None)
                else:
                    status[n] = new_status
        again = set(graph_changed)
        for x in graph_changed:
            again |= touched_by.get(x, set())
        regold = set()
        for n in again:
            found, read = _reach(n, requires)
            before = touched.get(n, empty)
            for x in before - read:
                touched_by[x].discard(n)
            for x in read - before:
                touched_by[x].add(n)
            touched[n] = read
            old = closures.get(n, empty)
            if found != old:
                for x in old - found:
                    members_of[x].discard(n)
                for x in found - old:
                    members_of[x].add(n)
                closures[n] = found
                regold.add(n)
        for x in status_changed:
            regold |= members_of.get(x, set())
        for n in regold:
            members = closures.get(n, empty)
            gold = frozenset(m for m in members if status.get(m) == "Final")
            previous = starts[n][-1][1:] if starts[n] else (empty, empty)
            if (members, gold) != previous:
                starts[n].append((d, members, gold))
    out: dict[str, list[Segment]] = {}
    for n, found in starts.items():
        segments = []
        for k, (first, members, gold) in enumerate(found):
            last = found[k + 1][0] - DAY if k + 1 < len(found) else s
            segments.append(Segment(first, last, members, gold))
        out[eth[n].pid] = segments
    return out


def f4_ranges(
    corpus: Corpus, s: date = S, segments: dict[str, list[Segment]] | None = None
) -> dict[str, list[Range]]:
    """F4's eligible days per proposal: closure and gold unchanged from D-7 to D+7, a closure of
    2 to 14 members, and a status of the proposal itself at D that a question could offer."""
    segments = f4_segments(corpus, s) if segments is None else segments
    lo, hi = F4_CLOSURE
    out: dict[str, list[Range]] = {}
    for pid, found in sorted(segments.items()):
        windows = [w for seg in found if lo <= len(seg.closure) <= hi
                   and (w := _window(seg.first, seg.last, s))]
        if not windows:
            continue
        p = corpus.proposals[pid]
        allowed = [(r.first, r.last) for r in day_runs(p, s) if r.status in CHOICES[p.family]]
        days = _intersect(windows, allowed)
        if days:
            out[pid] = days
    return out


def _segment_at(segments: list[Segment], d: date) -> Segment | None:
    for seg in segments:
        if seg.first <= d <= seg.last:
            return seg
    return None


# --- F5 ------------------------------------------------------------------------------------------


def f5_change_days(proposals: Iterable[Proposal], status: str, s: date = S) -> list[date]:
    """Days on which some proposal entered or left ``status`` (by its end-of-day status)."""
    days: set[date] = set()
    for p in proposals:
        member = False
        for r in day_runs(p, s):
            if (r.status == status) != member:
                days.add(r.first)
                member = r.status == status
    return sorted(days)


def f5_date(change_days: list[date], d: date, s: date = S, margin: int = F5_MARGIN) -> date | None:
    """The first day from ``d`` on with no change from D-3 to D+3 (from S-3 to S at S).

    A change on day c is one between the ends of c-1 and c. An item at S that is not stable is
    dropped (None), and so is one that would have to move past S-3.
    """
    while True:
        lo, hi = d - margin * DAY, min(d + margin * DAY, s)
        if d != s and d + margin * DAY > s:
            return None
        if not any(lo < c <= hi for c in change_days):
            return d
        if d == s:
            return None
        d += DAY


# --- snapshot at S -------------------------------------------------------------------------------


def _present_at(corpus: Corpus, moment: int) -> dict[str, dict[int, Proposal]]:
    """The proposals of each corpus present at ``moment`` (they have a file, so a status), by
    number in number order: the one presence rule the strata and the edges read at S."""
    return {family: {p.number: p for p in _proposals(corpus, family)
                     if p.status_at(moment) is not None} for family in CORPORA}


class _AtS:
    """What the strata read at S: presence, bodies, mentions and the requires graph."""

    def __init__(self, corpus: Corpus, readers, s: date = S) -> None:
        self.s = s
        self.moment = end_of_day_ts(s)
        self.present = _present_at(corpus, self.moment)
        self.bodies: dict[str, str] = {}
        self.body_starts: dict[str, int] = {}  # where the body begins in the file's text
        self.mentions: dict[str, set[int]] = {}  # present proposals of its corpus, never itself
        for family, present in self.present.items():
            for n, p in present.items():
                text = p.text_at(self.moment, readers)
                start = header_span(family, text)[1]
                body = text[start:]
                self.bodies[p.pid] = body
                self.body_starts[p.pid] = start
                self.mentions[p.pid] = {m for m in mentions(family, body, n) if m in present}
        self.requires: dict[int, list[int]] = {}  # Ethereum, present targets only
        self.dropped_requires: list[dict] = []
        eth = self.present["eth"]
        for n, p in eth.items():
            kept = []
            for m in p.requires_at(self.moment):
                if m in eth:
                    if m != n:
                        kept.append(m)
                else:
                    self.dropped_requires.append({"proposal": p.pid, "requires": m})
            self.requires[n] = kept


# --- the families --------------------------------------------------------------------------------


def header_cues(family: str, text: str) -> list[str]:
    """The header fields of a CF proposal's file, other than its status, that record the true
    status although the counterfactual edit, which rewrites only the status fields, leaves them
    alone (section 7): a last-call-deadline under any of its names (``headers.LAST_CALL_NAMES``),
    whatever its value; a Superseded-By, Replaced-By, withdrawal-reason or Accepted
    (``headers.CUE_WITH_A_VALUE``) with a value, whatever the value says; and a Resolution
    (``headers.RESOLUTION_NAMES``) whose case-folded value contains a decision word, since a
    resolution link alone records that a pronouncement was made, not which way it went. Their
    keys, in header order, each once.

    No other field is read: titles, descriptions, authors and discussion links name a proposal's
    subject, people and venue, and the relation headers do not record its status. Every field
    read is one that H- withholds (section 8). The fields are those the header reader finds,
    repeated keys included, since the CF stream shows every line.
    """
    found: dict[str, None] = {}
    for f in _scan(family, text).fields:
        if (f.name in LAST_CALL_NAMES or (f.name in CUE_WITH_A_VALUE and f.value)
                or (f.name in RESOLUTION_NAMES
                    and any(word in f.value.casefold() for word in DECISION_WORDS))):
            found[f.key] = None
    return list(found)


def header_only_after(p: Proposal, d: date, readers) -> bool:
    """Whether ``p`` has, in the H- stream, an episode recorded after the end of ``d`` whose diff
    is empty in H- and not empty in H+: a commit that changed only withheld header lines, whose
    episode section 8 keeps with its message (``stream.header_only_times``, which reads the
    stream's own episodes with its own withholding). A stream system can see from such an
    episode that something it cannot read changed after D; grep at D, which reads the files and
    the log as of D, cannot."""
    from .stream import header_only_times  # stream imports this module, so not at the top

    return bool(header_only_times(p, readers, after=end_of_day_ts(d)))


def _status_item(corpus: Corpus, p: Proposal, family: str, stratum: str, d: date, rank: int,
                 readers) -> Item:
    """An F1 or CF item: the status of ``p`` as of the end of ``d`` (permuted for CF)."""
    moment = end_of_day_ts(d)
    true = p.status_at(moment)
    if true not in CHOICES[p.family]:
        raise AssertionError(f"{p.pid} has status {true!r} at {d}, which no question offers")
    version = p.version_at(moment)
    interval = next(i for i in reversed(p.intervals()) if i.start_ts <= moment)
    commits = corpus.as_of_commits(d)
    meta = {"ref": ref(p), "interval_start": iso_ts(interval.start_ts)}
    if family == "CF":
        gold = permute(p.family, true)
        body = p.body_at(moment, readers).casefold()
        found = [word for word in DECISION_WORDS if word in body]
        cues = header_cues(p.family, p.text_at(moment, readers))
        meta.update(true_status=true, decision_word=bool(found), decision_words=found,
                    header_cue=bool(cues), header_cue_fields=cues)
    else:
        gold = true
        meta["status_at_s"] = p.status_at(end_of_day_ts(S))
        # Section 10: F1-recent is reported split by this, and the scorer reads no timelines.
        meta["gold_equals_status_at_s"] = gold == meta["status_at_s"]
        # Section 12: H- F1 is reported split by this, and so is the headline beside it.
        meta["h_minus_header_only_after_d"] = header_only_after(p, d, readers)
    return Item(
        id=f"{family}-{stratum}-{p.pid}-{d.isoformat()}",
        family=family,
        stratum=stratum,
        corpus=p.family,
        proposal=p.pid,
        as_of=d.isoformat(),
        arms=ARMS[family],
        question=_ask(p, TEMPLATES["F1"], D=d.isoformat(), choices=", ".join(CHOICES[p.family])),
        answer_type=ANSWER_TYPES[family],
        choices=CHOICES[p.family],
        gold=gold,
        gold_outcome=outcome_class(p.family, gold),
        source={"repo": version.repo, "path": version.path, "commit": commits[version.repo]},
        as_of_commits=commits,
        rank=rank,
        meta=meta,
    )


def _set_item(corpus: Corpus, p: Proposal, family: str, stratum: str, d: date, rank: int,
              gold: Iterable[int], meta: dict, **values: str) -> Item:
    return Item(
        id=f"{family}-{stratum}-{p.pid}-{d.isoformat()}",
        family=family,
        stratum=stratum,
        corpus=p.family,
        proposal=p.pid,
        as_of=d.isoformat(),
        arms=ARMS[family],
        question=_ask(p, TEMPLATES[family], **values),
        answer_type=ANSWER_TYPES[family],
        choices=None,
        gold=sorted(gold),
        gold_outcome=None,
        source=None,
        as_of_commits=corpus.as_of_commits(d),
        rank=rank,
        meta={"ref": ref(p), **meta},
    )


def _f1(corpus: Corpus, at_s: _AtS, runs: dict[str, list[Run]], readers):
    items: list[Item] = []
    pools: dict = {}
    for family in CORPORA:
        proposals = list(at_s.present[family].values())  # the proposals present at S
        for stratum in F1_STRATA:
            if stratum == "current":
                eligible = {p.pid: None for p in proposals if f1_current(p, runs=runs[p.pid])}
            else:
                eligible = {p.pid: found for p in proposals
                            if (found := f1_ranges(p, stratum, runs=runs[p.pid]))}
            pools[("F1", stratum, family)] = len(eligible)
            taken = order(eligible, "F1", stratum, str)[: SIZES[("F1", stratum)]]
            for rank, pid in enumerate(taken):
                p = corpus.proposals[pid]
                d = S if stratum == "current" else pick_day(expand(eligible[pid]), "F1", stratum,
                                                             pid)
                items.append(_status_item(corpus, p, "F1", stratum, d, rank, readers))
    return items, pools, {}


def f2_gold(present: dict[int, Proposal], moment: int) -> dict[int, list[int]]:
    """For each PEP n present at S: the PEPs m that n lists in Superseded-By (or Replaced-By), or
    that list n in Replaces. Self-references and numbers that are no PEP at S are dropped."""
    gold: dict[int, set[int]] = {n: set() for n in present}
    for n, p in present.items():
        for m in p.superseders_at(moment):
            if m != n and m in present:
                gold[n].add(m)
        for old in p.replaces_at(moment):
            if old != n and old in present:
                gold[old].add(n)
    return {n: sorted(found) for n, found in gold.items()}


def f2_recoverable(p: Proposal, gold: Iterable[int], readers) -> dict[str, bool]:
    """Whether the asked PEP's own episodes can show an F2 item's gold, in each arm (section 12).

    In H+, every gold member must be listed in the PEP's Superseded-By (or Replaced-By) header
    in some version of its file, or mentioned (section 7's mention rule) in the body of some
    version; in H-, which withholds those headers, every member must be mentioned. A negative,
    whose gold is empty, is recoverable in both arms. A version without text (a deletion, or a
    symlink) shows nothing.
    """
    members = set(gold)
    if not members:
        return {"H+": True, "H-": True}
    listed: set[int] = set()
    mentioned: set[int] = set()
    for v in p.versions:
        if v.status == ABSENT or v.blob is None:
            continue
        listed.update(superseders(v.header))
        text = readers[v.repo].read(v.blob)
        mentioned |= mentions(p.family, text[header_span(p.family, text)[1] :], p.number)
    return {"H+": members <= listed | mentioned, "H-": members <= mentioned}


def _f2(corpus: Corpus, at_s: _AtS, readers):
    peps = at_s.present["pep"]
    gold = f2_gold(peps, at_s.moment)
    positives = [peps[n] for n in sorted(gold) if gold[n]]
    negatives = [peps[n].pid for n in sorted(gold) if not gold[n] and at_s.mentions[peps[n].pid]]
    taken = order(negatives, "F2", F2_NEGATIVE, str)[: len(positives)]
    items = [
        _set_item(corpus, p, "F2", F2_POSITIVE, S, rank, gold[p.number],
                  {"recoverable": f2_recoverable(p, gold[p.number], readers)}, S=S.isoformat())
        for rank, p in enumerate(positives)
    ]
    for rank, pid in enumerate(taken):
        p = corpus.proposals[pid]
        meta = {"mentions": sorted(at_s.mentions[pid]),
                "recoverable": f2_recoverable(p, [], readers)}
        items.append(_set_item(corpus, p, "F2", F2_NEGATIVE, S, rank, [], meta, S=S.isoformat()))
    pools = {("F2", F2_POSITIVE, "pep"): len(positives), ("F2", F2_NEGATIVE, "pep"): len(negatives)}
    not_recoverable = {arm: [i.id for i in items if not i.meta["recoverable"][arm]]
                       for arm in ARMS["F2"]}
    return items, pools, {"positives": len(positives), "not_recoverable": not_recoverable}


def edge_records(corpus: Corpus, s: date = S) -> list[dict]:
    """The extraction track's relation gold (section 9): every header edge at S, one record each.

    PEP m ``supersedes`` PEP n when, at S, n lists m in Superseded-By (or Replaced-By) or m lists
    n in Replaces, as F2 reads them (``f2_gold``); proposal N ``requires`` M when N's requires
    header at S lists M, in either corpus. Self-references and numbers that are no proposal of
    the corpus at S are dropped. Records are sorted by relation, then source id, then target id,
    each compared as a string, and carry the canary.
    """
    moment = end_of_day_ts(s)
    present = _present_at(corpus, moment)  # as the strata read presence at S (``_AtS``)
    edges: set[tuple[str, str, str, str]] = set()
    peps = present["pep"]
    for n, superseded_by in f2_gold(peps, moment).items():
        edges.update(("supersedes", "pep", peps[m].pid, peps[n].pid) for m in superseded_by)
    for family, found in present.items():
        for n, p in found.items():
            edges.update(("requires", family, p.pid, found[m].pid)
                         for m in p.requires_at(moment) if m != n and m in found)
    return [{"canary": CANARY, "corpus": family, "relation": relation, "source": source,
             "target": target}
            for relation, family, source, target in sorted(edges, key=lambda e: (e[0], *e[2:]))]


def _f3(corpus: Corpus, at_s: _AtS):
    graph = at_s.requires
    top = hubs(graph)
    strata: dict[str, dict[str, dict[int, int]]] = {stratum: {} for stratum in F3_STRATA}
    hub_recall, one_hop = [], []
    eth = at_s.present["eth"]
    for n in sorted(graph):
        dist = closure(graph, n)
        if len(dist) < 2:
            continue
        if max(dist.values()) < 2:
            one_hop.append(eth[n].pid)
        elif set(dist) <= set(top):
            hub_recall.append(eth[n].pid)
        else:
            strata[size_stratum(len(dist))][eth[n].pid] = dist
    items: list[Item] = []
    pools = {}
    for stratum, eligible in strata.items():
        pools[("F3", stratum, "eth")] = len(eligible)
        for rank, pid in enumerate(order(eligible, "F3", stratum, str)[: SIZES[("F3", stratum)]]):
            dist = eligible[pid]
            meta = {"size": len(dist), "max_hops": max(dist.values())}
            items.append(_set_item(corpus, corpus.proposals[pid], "F3", stratum, S, rank, dist,
                                   meta, S=S.isoformat()))
    notes = {"hubs": top, "dropped_as_hub_recall": hub_recall, "no_member_two_hops_away": one_hop,
             "dropped_requires": at_s.dropped_requires}
    return items, pools, notes


def _f4(corpus: Corpus, at_s: _AtS):
    segments = f4_segments(corpus)  # every Ethereum proposal: members need not be present at S
    eligible = {pid: days for pid, days in f4_ranges(corpus, segments=segments).items()
                if corpus.proposals[pid].number in at_s.present["eth"]}
    items: list[Item] = []
    differs = []
    taken = order(eligible, "F4", F4_STRATUM, str)[: SIZES[("F4", F4_STRATUM)]]
    for rank, pid in enumerate(taken):
        p = corpus.proposals[pid]
        d = pick_day(expand(eligible[pid]), "F4", F4_STRATUM, pid)
        seg = _segment_at(segments[pid], d)
        seg_at_s = _segment_at(segments[pid], S)
        gold_at_s = sorted(seg_at_s.gold) if seg_at_s is not None else []
        if gold_at_s != sorted(seg.gold):
            differs.append(pid)
        meta = {"closure": sorted(seg.closure), "gold_at_s": gold_at_s,
                "status_at_d": p.status_at(end_of_day_ts(d))}
        items.append(_set_item(corpus, p, "F4", F4_STRATUM, d, rank, seg.gold, meta,
                               D=d.isoformat()))
    pools = {("F4", F4_STRATUM, "eth"): len(eligible)}
    return items, pools, {"gold_differs_from_s": len(differs), "differs": differs}


def _f5(corpus: Corpus):
    items: list[Item] = []
    pools = {}
    dropped, moved = [], []
    for family in CORPORA:
        proposals = _proposals(corpus, family)
        slots = [(d, status) for d in F5_DATES[family] for status in F5_STATUSES[family]]
        pools[("F5", F5_STRATUM, family)] = len(slots)
        changes = {status: f5_change_days(proposals, status) for status in F5_STATUSES[family]}
        for rank, (registered, status) in enumerate(slots):
            d = f5_date(changes[status], registered)
            if d is None:
                dropped.append(f"F5-{F5_STRATUM}-{family}-{status}-{registered.isoformat()}")
                continue
            if d != registered:
                moved.append({"registered": registered.isoformat(), "used": d.isoformat(),
                              "corpus": family, "status": status,
                              "days": (d - registered).days})
            moment = end_of_day_ts(d)
            count = sum(1 for p in proposals if p.status_at(moment) == status)
            items.append(Item(
                id=f"F5-{F5_STRATUM}-{family}-{status}-{d.isoformat()}",
                family="F5",
                stratum=F5_STRATUM,
                corpus=family,
                proposal=None,
                as_of=d.isoformat(),
                arms=ARMS["F5"],
                question=TEMPLATES["F5"].format(series=F5_SERIES[family], status=status,
                                                D=d.isoformat()),
                answer_type=ANSWER_TYPES["F5"],
                choices=None,
                gold=count,
                gold_outcome=None,
                source=None,
                as_of_commits=corpus.as_of_commits(d),
                rank=rank,
                meta={"series": F5_SERIES[family], "status": status,
                      "registered_date": registered.isoformat(),
                      "moved_days": (d - registered).days},
            ))
    return items, pools, {"dropped": sorted(dropped), "moved": moved}


def _prominence(at_s: _AtS, family: str) -> dict[int, int]:
    """PEPs: how many other PEPs mention them; Ethereum: how many proposals require or mention
    them; at S, among proposals present at S."""
    score: Counter[int] = Counter()
    for n, p in at_s.present[family].items():
        referrers = set(at_s.mentions[p.pid])
        if family == "eth":
            referrers |= set(at_s.requires[n])
        score.update(referrers - {n})
    return score


def _cf(corpus: Corpus, at_s: _AtS, readers, runs: dict[str, list[Run]]):
    items: list[Item] = []
    pools = {}
    pool_notes = {}
    for family in CORPORA:
        present = at_s.present[family]
        score = _prominence(at_s, family)
        ranked = sorted(present, key=lambda n: (-score[n], n))[:CF_POOL]
        eligible = {}
        for n in ranked:
            p = present[n]
            found = cf_ranges(p, runs=runs[p.pid])
            if found:
                eligible[p.pid] = found
        pool_notes[family] = [{"proposal": present[n].pid, "prominence": score[n],
                               "eligible": present[n].pid in eligible} for n in ranked]
        pools[("CF", CF_STRATUM, family)] = len(eligible)
        taken = order(eligible, "CF", CF_STRATUM, str)[: SIZES[("CF", CF_STRATUM)]]
        for rank, pid in enumerate(taken):
            d = pick_day(expand(eligible[pid]), "CF", CF_STRATUM, pid)
            items.append(_status_item(corpus, corpus.proposals[pid], "CF", CF_STRATUM, d, rank,
                                      readers))
    flagged = [i.id for i in items if i.meta["decision_word"]]
    cued = [i for i in items if i.meta["header_cue"]]
    return items, pools, {"pool": pool_notes, "decision_word": flagged,
                          "header_cue": [(i.id, i.meta["header_cue_fields"]) for i in cued]}


def _passage_bounds(size: int, start: int, end: int, width: int) -> tuple[int, int]:
    """The ``width`` characters of a text of ``size`` centred on ``[start, end)``, shifted to
    stay inside the text."""
    lo = max(0, (start + end) // 2 - width // 2)
    hi = min(size, lo + width)
    return max(0, hi - width), hi


def passage(body: str, start: int, end: int, width: int = SNIPPET_CHARS) -> str:
    """The passage the annotators read around the mention at ``[start, end)`` of ``body``: the
    ``width`` characters centred on it, kept inside the body, whitespace collapsed (section 9)."""
    lo, hi = _passage_bounds(len(body), start, end, width)
    return " ".join(body[lo:hi].split())


def passage_spans(family: str, body: str, number: int, width: int = SNIPPET_CHARS,
                  most: int = SNIPPETS_PER_SIDE) -> list[tuple[int, int]]:
    """The mentions of ``number`` in ``body`` that the sample shows a passage around: up to
    ``most``, in text order. Passages do not overlap: a mention whose passage would start inside
    the last one taken is skipped."""
    found: list[tuple[int, int]] = []
    taken_to = 0
    for start, end in mention_spans(family, body, number):
        lo, hi = _passage_bounds(len(body), start, end, width)
        if found and lo < taken_to:
            continue
        found.append((start, end))
        taken_to = hi
        if len(found) == most:
            break
    return found


def snippets(family: str, body: str, number: int, width: int = SNIPPET_CHARS,
             most: int = SNIPPETS_PER_SIDE) -> list[str]:
    """The passages of ``passage_spans``, as text."""
    return [passage(body, start, end, width)
            for start, end in passage_spans(family, body, number, width, most)]


def _source(corpus: Corpus, p: Proposal, moment: int) -> dict:
    v = p.version_at(moment)
    return {"repo": v.repo, "path": v.path, "commit": corpus.pins.get(v.repo)}


def _mention_locations(corpus: Corpus, at_s: _AtS, p: Proposal, number: int) -> list[dict]:
    """Where the passages of ``p``'s body at S around its mentions of ``number`` are: the file at
    the pinned commit and the character offsets of each mention in its text (section 9)."""
    source = _source(corpus, p, at_s.moment)
    base = at_s.body_starts[p.pid]
    return [{**source, "start": base + start, "end": base + end}
            for start, end in passage_spans(p.family, at_s.bodies[p.pid], number)]


def _labelling(corpus: Corpus, at_s: _AtS):
    """Section 9: unordered pairs in one corpus, both present at S, where one body at S mentions
    the other and no header relation at S joins them; 100 per corpus in hash order.

    Each record holds the pair's ids and titles, which side mentions which, and for each side up
    to two mentions as their file and character offsets (``_mention_locations``), but no proposal
    text: the passages go into the annotators' local sheet (``asofbench labelling-sheet``)."""
    records: list[dict] = []
    pools = {}
    for family in CORPORA:
        present = at_s.present[family]
        linked: set[frozenset[int]] = set()
        for n, p in present.items():
            related = list(p.requires_at(at_s.moment))
            if family == "pep":
                related += p.superseders_at(at_s.moment) + p.replaces_at(at_s.moment)
            linked.update(frozenset((n, m)) for m in related if m != n)
        pairs = {(min(n, m), max(n, m)) for n, p in present.items() for m in at_s.mentions[p.pid]}
        eligible = {f"{present[a].pid}~{present[b].pid}": (a, b)
                    for a, b in sorted(pairs) if frozenset((a, b)) not in linked}
        pools[(LABELLING, family, family)] = len(eligible)
        taken = order(eligible, LABELLING, family, str)[: SIZES[(LABELLING, family)]]
        for rank, pair_id in enumerate(taken):
            a, b = (present[n] for n in eligible[pair_id])
            a_to_b = b.number in at_s.mentions[a.pid]
            b_to_a = a.number in at_s.mentions[b.pid]
            records.append({
                "pair_id": pair_id,
                "corpus": family,
                "rank": rank,
                "a": a.pid,
                "a_title": title(a),
                "a_mentions_b": a_to_b,
                "a_mention_spans": _mention_locations(corpus, at_s, a, b.number) if a_to_b else [],
                "a_source": _source(corpus, a, at_s.moment),
                "b": b.pid,
                "b_title": title(b),
                "b_mentions_a": b_to_a,
                "b_mention_spans": _mention_locations(corpus, at_s, b, a.number) if b_to_a else [],
                "b_source": _source(corpus, b, at_s.moment),
                "canary": CANARY,
            })
    return records, pools


def _interval_runs(p: Proposal, s: date = S) -> list[Run]:
    """Like ``day_runs``, but each interval on its own: runs never join across a reversion."""
    runs = []
    for i in p.intervals():
        first = utc_date(i.start_ts)
        last = s if i.end_ts is None else min(s, utc_date(i.end_ts) - DAY)
        if first <= last:
            runs.append(Run(None if i.status == ABSENT else i.status, first, last))
    return runs


def _reversions(corpus: Corpus, items: list[Item]) -> dict:
    """Statuses recorded and reverted within one UTC day, which no end of day sees, and the F1
    and CF items whose eligible days would differ if each interval were read on its own."""
    found = []
    for family in CORPORA:
        for p in _proposals(corpus, family):
            intervals = p.intervals()
            for k in range(1, len(intervals) - 1):
                i = intervals[k]
                if utc_date(i.start_ts) == utc_date(i.end_ts) and (
                    intervals[k - 1].status == intervals[k + 1].status
                ):
                    found.append({"proposal": p.pid, "day": utc_date(i.start_ts).isoformat(),
                                  "status": intervals[k - 1].status, "reverted": i.status})
    affected = {entry["proposal"] for entry in found}
    differ = []
    for i in items:
        if i.family not in ("F1", "CF") or i.proposal not in affected:
            continue
        p = corpus.proposals[i.proposal]
        alone = _interval_runs(p)
        if i.stratum == "current":
            same = f1_current(p, runs=alone)
        elif i.family == "CF":
            same = cf_ranges(p, runs=alone) == cf_ranges(p)
        else:
            same = f1_ranges(p, i.stratum, runs=alone) == f1_ranges(p, i.stratum)
        if not same:
            differ.append(i.id)
    return {"reversions": found, "items_of_these_proposals": sorted(
        i.id for i in items if i.family in ("F1", "CF") and i.proposal in affected),
        "items_that_would_differ": differ}


def build_items(corpus: Corpus, readers) -> Build:
    """Every family of section 7 with its registered sizes, and the mention-pair sample.

    ``readers`` maps each repository name to a ``BlobReader`` (``Corpus.readers()``); bodies are
    read at S for the pools and pairs, and at D for the CF decision words.
    """
    runs = {p.pid: day_runs(p) for p in corpus.proposals.values()}
    at_s = _AtS(corpus, readers)
    items: list[Item] = []
    pools: dict = {}
    notes: dict = {}
    for family, (found, family_pools, family_notes) in (
        ("F1", _f1(corpus, at_s, runs, readers)),
        ("F2", _f2(corpus, at_s, readers)),
        ("F3", _f3(corpus, at_s)),
        ("F4", _f4(corpus, at_s)),
        ("F5", _f5(corpus)),
        ("CF", _cf(corpus, at_s, readers, runs)),
    ):
        items.extend(found)
        pools.update(family_pools)
        notes[family] = family_notes
    labelling, labelling_pools = _labelling(corpus, at_s)
    pools.update(labelling_pools)
    notes["reversions"] = _reversions(corpus, items)
    seen: dict[tuple[str, str], list[str]] = defaultdict(list)
    for i in items:
        seen[(i.question, i.as_of)].append(i.id)
    notes["same_question"] = [ids for ids in seen.values() if len(ids) > 1]
    ids = [i.id for i in items]
    if len(ids) != len(set(ids)):
        raise AssertionError("two items share an id")
    return Build(items, pools, labelling, notes)


# --- outputs -------------------------------------------------------------------------------------


def dumps(obj: object) -> str:
    """One JSON value on one line: sorted keys, UTF-8 text, no spaces."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def majority_class(golds: Iterable[object]) -> tuple[object, int, int]:
    """The most common of some items' gold values, how many items have it, and how many values
    have that many items (1 unless values tie).

    Gold values are compared as their JSON text (``dumps``), so a set gold, a sorted list, is one
    value; a tie goes to the value whose text sorts first.
    """
    counts = Counter(dumps(gold) for gold in golds)
    if not counts:
        raise ValueError("no gold values")
    text, n = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return json.loads(text), n, sum(1 for count in counts.values() if count == n)


def _write_lines(path: Path, records: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(dumps(record) + "\n")


def _question_record(i: Item) -> dict:
    return {
        "id": i.id, "family": i.family, "stratum": i.stratum, "corpus": i.corpus,
        "proposal": i.proposal, "as_of": i.as_of, "arms": list(i.arms), "question": i.question,
        "answer_type": i.answer_type, "choices": None if i.choices is None else list(i.choices),
        "canary": CANARY,
    }


def _gold_record(i: Item) -> dict:
    return {
        "id": i.id, "gold": i.gold, "gold_outcome": i.gold_outcome, "source": i.source,
        "as_of_commits": i.as_of_commits, "meta": {**i.meta, "rank": i.rank}, "canary": CANARY,
    }


def _pilot(items: Build) -> dict:
    """Section 13: the first 50 F1-change and first 25 F1-current items of each corpus in frozen
    order, and every CF item; and the M0 Graphiti pilot's 50 PEPs (F1-change's first 50)."""
    groups = []
    for (family, stratum), size in [*PILOT.items(), (("CF", CF_STRATUM), None)]:
        for corpus in CORPORA:
            ids = [i.id for i in items
                   if (i.family, i.stratum, i.corpus) == (family, stratum, corpus)
                   and (size is None or i.rank < size)]
            arms = ["H-", "H+"] if family == "F1" else ["CF"]
            groups.append({"family": family, "stratum": stratum, "corpus": corpus, "arms": arms,
                           "ids": ids})
    m0 = [i.proposal for i in items
          if (i.family, i.stratum, i.corpus) == ("F1", "change", "pep") and i.rank < 50]
    return {
        "canary": CANARY,
        "groups": groups,
        "ids": [i for group in groups for i in group["ids"]],
        "m0_graphiti_proposals": m0,
    }


def _timelines(corpus: Corpus) -> Iterator[dict]:
    moment = end_of_day_ts(S)
    for family in CORPORA:
        for p in _proposals(corpus, family):
            intervals = []
            for i in p.intervals():
                intervals.append({
                    "status": i.status,
                    "start": iso_ts(i.start_ts),
                    "end": None if i.end_ts is None else iso_ts(i.end_ts),
                    "commit": i.start_commit,
                    "repo": p.version_at(i.start_ts).repo,
                    "is_transition": i.is_transition,
                })
            yield {
                "pid": p.pid,
                "family": p.family,
                "series": p.series,
                "number": p.number,
                "title": title(p),
                "present_at_s": p.status_at(moment) is not None,
                "intervals": intervals,
                "canary": CANARY,
            }


def write_outputs(items: Build, corpus: Corpus, out: Path) -> None:
    """Write the question, gold, pilot, mention-pair, timeline, edge and report files into
    ``out``."""
    if not isinstance(items, Build):
        raise TypeError("write_outputs needs the Build that build_items returns")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    _write_lines(out / "questions.jsonl", (_question_record(i) for i in items))
    _write_lines(out / "gold.jsonl", (_gold_record(i) for i in items))
    _write_lines(out / "pilot.json", [_pilot(items)])
    _write_lines(out / "labelling" / "mention_pairs.jsonl", items.labelling)
    _write_lines(out / "timelines.jsonl", _timelines(corpus))
    _write_lines(out / "edges.jsonl", edge_records(corpus))
    with (out / "REPORT.md").open("w", encoding="utf-8", newline="\n") as f:
        f.write(report(items, corpus))


# --- the report ----------------------------------------------------------------------------------


def _registered(family: str, stratum: str, corpus: str) -> str:
    if family == "F2":
        return "all" if stratum == F2_POSITIVE else "as many as positives"
    if family == "F5":
        return f"{len(F5_DATES[corpus]) * len(F5_STATUSES[corpus])} (at most)"
    size = SIZES[(family, stratum)]
    return str(size) if (family, stratum) in FIXED else f"up to {size}"


_FAMILY_ORDER = {name: k for k, name in enumerate((*FAMILIES, LABELLING))}
_STRATUM_ORDER = {name: k for k, name in enumerate((*F1_STRATA, F2_POSITIVE, F2_NEGATIVE,
                                                    *F3_STRATA, F4_STRATUM, F5_STRATUM,
                                                    CF_STRATUM, *CORPORA))}


def _listed(key: tuple[str, str, str]) -> tuple[int, int, int]:
    """The order in which the report lists (family, stratum, corpus) groups."""
    family, stratum, corpus = key
    return _FAMILY_ORDER[family], _STRATUM_ORDER[stratum], CORPORA.index(corpus)


def _rows(items: Build) -> list[tuple[str, str, str, str, int, int, str]]:
    counted = Counter((i.family, i.stratum, i.corpus) for i in items)
    counted.update((LABELLING, r["corpus"], r["corpus"]) for r in items.labelling)
    positives = items.pools.get(("F2", F2_POSITIVE, "pep"), 0)
    rows = []
    for family, stratum, corpus in sorted(items.pools, key=_listed):
        pool = items.pools[(family, stratum, corpus)]
        n = counted.get((family, stratum, corpus), 0)
        note = ""
        if family == "F5":
            if n < pool:
                note = f"{pool - n} item(s) at S dropped as unstable"
        elif family == "F2" and stratum == F2_NEGATIVE:
            if n < positives:
                note = f"SHORTFALL: all eligible ({pool}), fewer than {positives} positives"
        elif family == "F2":
            pass
        else:
            size = SIZES[(family, stratum)]
            if n < size:
                note = "all eligible"
                if (family, stratum) in FIXED:
                    note = f"SHORTFALL: all eligible, below the registered {size}"
        rows.append((family, stratum, corpus, _registered(family, stratum, corpus), pool, n, note))
    return rows


def report(items: Build, corpus: Corpus) -> str:
    """REPORT.md: item counts and pools, each family's notes, and the data-quality report."""
    quality = quality_report(corpus)
    notes = items.notes
    lines = [
        "# asof-bench v0.1 build report",
        "",
        CANARY,
        "",
        f"Snapshot S = {S.isoformat()}; selection seed `{SEED}`.",
        "",
        "## Pins",
        "",
        "| Repository | Commit |",
        "|---|---|",
        *(f"| {repo} | `{sha}` |" for repo, sha in sorted(corpus.pins.items())),
        "",
        "## Items by family, stratum and corpus",
        "",
        "The eligible pool is the number of candidates that met every rule of the stratum; items "
        "are taken from it in hash order. For F5 the pool is the registered dates and statuses.",
        "",
        "| Family | Stratum | Corpus | Registered | Eligible pool | Items | Note |",
        "|---|---|---|---|---|---|---|",
        *(f"| {f} | {st} | {c} | {reg} | {pool} | {n} | {note} |"
          for f, st, c, reg, pool, n, note in _rows(items)),
        "",
        f"Items in total: {len(items)}; mention pairs: {len(items.labelling)}.",
        "",
        *_floor_lines(items),
        *_header_only_lines(items),
        *_f2_lines(notes["F2"]),
        "## F3, requirement closure",
        "",
        "The five most-required proposals at S (by direct in-degree, ties to the lower number): "
        + ", ".join(str(n) for n in notes["F3"]["hubs"]) + ".",
        "",
        f"- Dropped as hub recall (closure only of those five): "
        f"{len(notes['F3']['dropped_as_hub_recall'])}.",
        f"- Closure of two or more, but no member two or more hops away: "
        f"{len(notes['F3']['no_member_two_hops_away'])}.",
        f"- Requires numbers that are not proposals at S, dropped: "
        f"{len(notes['F3']['dropped_requires'])}"
        + "".join(f"\n  - {d['proposal']} requires {d['requires']}"
                  for d in notes["F3"]["dropped_requires"]),
        "",
        "## F4, requirement closure with status",
        "",
        f"Items whose gold differs from the same question asked at S: "
        f"{notes['F4']['gold_differs_from_s']} of {sum(1 for i in items if i.family == 'F4')}.",
        "",
        "## F5, counts",
        "",
        "| Corpus | Status | Registered date | Date used | Count |",
        "|---|---|---|---|---|",
        *(f"| {i.corpus} | {i.meta['status']} | {i.meta['registered_date']} | {i.as_of} | "
          f"{i.gold} |" for i in items if i.family == "F5"),
        "",
        "Dates moved forward because some proposal entered or left the status within 3 days "
        "(registered date to date used): "
        + ("; ".join(f"{m['corpus']} {m['status']} {m['registered']} to {m['used']} "
                     f"({m['days']} days)" for m in notes["F5"]["moved"]) or "none") + ".",
        "",
        "Dropped as unstable at S: " + (", ".join(notes["F5"]["dropped"]) or "none") + ".",
        "",
        "## CF, counterfactual edits",
        "",
        f"Items whose body at D contains a decision word ({', '.join(DECISION_WORDS)}): "
        f"{len(notes['CF']['decision_word'])} of {sum(1 for i in items if i.family == 'CF')}"
        + "".join(f"\n- {item_id}" for item_id in notes["CF"]["decision_word"]),
        "",
        "Items whose file at D has a header cue, a header field other than the status that the "
        f"edit leaves alone and that records the true status ({HEADER_CUE_RULE}): "
        f"{len(notes['CF']['header_cue'])} of "
        f"{sum(1 for i in items if i.family == 'CF')}"
        + "".join(f"\n- {item_id} ({', '.join(fields)})"
                  for item_id, fields in notes["CF"]["header_cue"]),
        "",
        *_cf_pool_lines(notes["CF"]["pool"]),
        "## Statuses reverted within one day",
        "",
        "These statuses were recorded and reverted within one UTC day, so no end of day sees them "
        "and they split no window. F1 and CF items about these proposals: "
        + (", ".join(ids) if (ids := notes["reversions"]["items_of_these_proposals"]) else "none")
        + "; of those, items whose eligible days would differ if each status interval were read "
        "on its own: "
        + (", ".join(ids) if (ids := notes["reversions"]["items_that_would_differ"]) else "none")
        + ".",
        "",
        "| Proposal | Day | Status | Reverted |",
        "|---|---|---|---|",
        *(f"| {r['proposal']} | {r['day']} | {r['status']} | {r['reverted']} |"
          for r in notes["reversions"]["reversions"]),
        "",
        f"Items that repeat another item's question and date: {len(notes['same_question'])}"
        + "".join(f"\n- {', '.join(ids)}" for ids in notes["same_question"]),
        "",
        *_quality_lines(quality),
    ]
    return "\n".join(lines) + "\n"


def _floor_lines(items: Build) -> list[str]:
    """The trivial-answer floors: per family, stratum and corpus, what always giving the most
    common gold value scores, and for F1 what answering the status at S scores."""
    groups: dict[tuple[str, str, str], list[Item]] = defaultdict(list)
    for i in items:
        groups[(i.family, i.stratum, i.corpus)].append(i)
    rows = []
    for key in sorted(groups, key=_listed):
        found = groups[key]
        value, n, tied = majority_class(i.gold for i in found)
        shown = value if isinstance(value, str) else json.dumps(value)
        if tied > 1:
            shown += f" (one of {tied} tied)"
        at_s = "- | -"
        if key[0] == "F1":
            same = sum(1 for i in found if i.meta["gold_equals_status_at_s"])
            at_s = f"{same} | {same / len(found):.3f}"
        rows.append(f"| {' | '.join(key)} | {len(found)} | {shown} | {n} | {n / len(found):.3f} | "
                    f"{at_s} |")
    return [
        "## Trivial-answer floors",
        "",
        "For each family, stratum and corpus: the items, the most common gold value and how many "
        "items have it (always giving that answer scores its share), and for F1 how many items "
        "have the proposal's status at S as their gold (answering the status at S scores that "
        "share).",
        "",
        "| Family | Stratum | Corpus | Items | Most common gold | With it | Share | Gold is the "
        "status at S | Share |",
        "|---|---|---|---|---|---|---|---|---|",
        *rows,
        "",
        "A tie shows the gold value whose JSON text sorts first, marked with the number of values "
        "tied for the most items; the share is the same whichever of them is shown.",
        "",
    ]


def _header_only_lines(items: Build) -> list[str]:
    """Per F1 stratum and corpus, how many items have a header-only H- episode after their date
    (``header_only_after``)."""
    rows = []
    for stratum in F1_STRATA:
        for corpus in CORPORA:
            found = [i for i in items if (i.family, i.stratum, i.corpus) == ("F1", stratum, corpus)]
            if found:
                flagged = sum(i.meta["h_minus_header_only_after_d"] for i in found)
                rows.append(f"| {stratum} | {corpus} | {len(found)} | {flagged} |")
    return [
        "## F1 items with a header-only H- episode after their date",
        "",
        "The H- stream keeps the episode of a commit that changed only withheld header lines, "
        "with an empty diff and its commit message (section 8). An F1 item is flagged when its "
        "proposal has such an episode recorded after the end of its date: a modify or rename "
        "whose diff is empty in H- and not in H+, which a system that reads the stream can see and "
        "grep at D, which reads the files and the log as of D, cannot. The scorer reports every "
        "system's H- F1 accuracy apart by the flag (gold `meta.h_minus_header_only_after_d`).",
        "",
        "| Stratum | Corpus | Items | Flagged |",
        "|---|---|---|---|",
        *rows,
        "",
    ]


def _f2_lines(notes: dict) -> list[str]:
    missing = notes["not_recoverable"]
    shown = {arm: ", ".join(ids) or "none" for arm, ids in missing.items()}
    positives = notes["positives"]
    return [
        "## F2, supersession",
        "",
        "Positives whose gold the asked PEP's own episodes can show (every gold member listed in "
        "its Superseded-By or Replaced-By header, or mentioned in its body, in some version of "
        "its file; in H-, which withholds those headers, only a mention counts): "
        f"H+ {positives - len(missing['H+'])} of {positives}, "
        f"H- {positives - len(missing['H-'])} of {positives}. Negatives, whose gold is empty, are "
        "recoverable in both arms.",
        "",
        f"Not recoverable in H+: {shown['H+']}.",
        "",
        f"Not recoverable in H-: {shown['H-']}.",
        "",
    ]


def _cf_pool_lines(pool: dict) -> list[str]:
    lines = []
    for corpus in CORPORA:
        lines += [
            f"The {CF_POOL} most prominent {corpus} proposals at S (prominence, and whether a "
            "date is eligible):",
            "",
            ", ".join(f"{e['proposal']} ({e['prominence']}{'' if e['eligible'] else ', none'})"
                      for e in pool[corpus]) + ".",
            "",
        ]
    return lines


def _quality_lines(quality: dict) -> list[str]:
    series = quality["proposals_by_series"]
    lines = [
        "## Data quality",
        "",
        "| Series | Proposals | Present at the pin | Versions | Status intervals | Absent "
        "intervals | Transitions | Missing-status versions |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in series:
        lines.append(
            f"| {name} | {series[name]['proposals']} | {series[name]['present_at_pin']} | "
            f"{quality['versions'][name]} | {quality['intervals'][name]['status']} | "
            f"{quality['intervals'][name]['absent']} | {quality['transitions'][name]} | "
            f"{quality['missing_values'][name]['versions']} |"
        )
    lines += [
        "",
        f"- Unknown recorded statuses: {len(quality['unknown_raw_values'])}.",
        f"- Collisions (two files of one proposal): {len(quality['collisions'])}.",
        f"- ERC stitching conflicts: {len(quality['stitch_conflicts'])}.",
        f"- Stagnant moves: {quality['stagnant_moves']['total']} "
        f"({quality['stagnant_moves']['by_eth_bot_account']} by the eth-bot account, "
        f"{quality['stagnant_moves']['by_people']} by people).",
        f"- Paths named like proposals that match no pattern: {len(quality['near_miss_paths'])}.",
        f"- Excluded: {', '.join(quality['excluded']) or 'none'}.",
        "",
        "The full data-quality report:",
        "",
        "```json",
        json.dumps(quality, sort_keys=True, ensure_ascii=False, indent=1),
        "```",
    ]
    return lines
