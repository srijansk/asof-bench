"""Every sampled stratum re-derived from the corpus timelines, independently (corpus-marked).

Set ASOFBENCH_CLONES to run. Nothing here imports the selection code (``asofbench.questions`` or
``asofbench.select``); it reads only the corpus model (``load_corpus``) and the registered tables
(statuses, mentions, header lists). Each proposal's end-of-day state is swept day by day from its
versions: the version in effect at D 23:59:59 UTC is the last one recorded at or before then. A
day D is eligible for an answer when the sweep shows that answer at the end of every day from D-7
to D+7. The order and the chosen date are recomputed from SHA-256 and the registration's strings.

A fresh ``asofbench build`` is compared with the re-derivation stratum by stratum: its pool size,
its hash-ordered prefix, each picked day (in the item id), and each gold value. The last test
compares the fresh build with ``data/v0.1``: the seven data files byte for byte, and the manifest
as parsed JSON with only ``generator.source_commit`` and ``generator.source_dirty`` left out, which
depend on the checkout rather than on the build. The fresh build is made as ``data/v0.1`` was, with
its streams and with the Graphiti budget its manifest records.
"""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pytest

from asofbench import cli
from asofbench.headers import number_list, superseders
from asofbench.mentions import mentions
from asofbench.statuses import CHOICES, permute
from asofbench.timeline import load_corpus

pytestmark = pytest.mark.corpus

SEED = "asof-bench/v0.1"
S = date(2026, 10, 27)
EPOCH = date(1970, 1, 1)
DAY = timedelta(days=1)
MARGIN, F5_MARGIN = 7, 3
ORIGIN = {"pep": date(2000, 1, 1), "eth": date(2015, 1, 1)}  # before each corpus's first version
RECENT_FROM = date(2026, 1, 1)
DATA = Path(__file__).resolve().parents[1] / "data" / "v0.1"
DATA_FILES = ("questions.jsonl", "gold.jsonl", "pilot.json", "labelling/mention_pairs.jsonl",
              "timelines.jsonl", "edges.jsonl", "REPORT.md")
FILES = (*DATA_FILES, "MANIFEST.json")
CHECKOUT_ONLY = ("source_commit", "source_dirty")  # generator fields that say where it was run


def eod(d: date) -> int:
    return (d - EPOCH).days * 86400 + 86399


def digest(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


def hash_order(ids, family: str, stratum: str) -> list[str]:
    return sorted(ids, key=lambda c: digest(f"{SEED}|{family}|{stratum}|{c}"))


def chosen(days: list[date], family: str, stratum: str, ident: str) -> date:
    k = int.from_bytes(digest(f"{SEED}|date|{family}|{stratum}|{ident}")[:8], "big")
    return days[k % len(days)]


# --- the end-of-day sweep ------------------------------------------------------------------------


@dataclass
class Swept:
    origin: date
    runs: list[tuple[str | None, int, int]]  # status (None while absent), first and last index
    versions: list | None  # Ethereum only: the version in effect at the end of each day

    def status(self, d: date) -> str | None:
        i = (d - self.origin).days
        if i < 0:
            return None
        k = bisect_right(self.runs, i, key=lambda run: run[1]) - 1
        return self.runs[k][0] if k >= 0 else None

    def windows(self, keep) -> list[tuple[str, int, int]]:
        """(status, first, last) day indices whose margins lie inside one run, for runs whose
        status ``keep`` accepts."""
        return [(s, a + MARGIN, b - MARGIN) for s, a, b in self.runs
                if s is not None and keep(s) and a + MARGIN <= b - MARGIN]

    def days(self, ranges) -> list[date]:
        return [self.origin + k * DAY for _, a, b in ranges for k in range(a, b + 1)]


def sweep(p, origin: date, keep_versions: bool) -> Swept:
    n = (S - origin).days + 1
    versions, j, current = p.versions, 0, None
    runs: list[list] = []
    per_day = [] if keep_versions else None
    moment = eod(origin)
    for i in range(n):
        while j < len(versions) and versions[j].recorded_ts <= moment:
            current = versions[j]
            j += 1
        present = current is not None and current.status != "absent"
        status = current.status if present else None
        if runs and runs[-1][0] == status:
            runs[-1][2] = i
        else:
            runs.append([status, i, i])
        if per_day is not None:
            per_day.append(current if present else None)
        moment += 86400
    return Swept(origin, [tuple(run) for run in runs], per_day)


# --- fixtures ------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus(clones):
    return load_corpus(clones)


@pytest.fixture(scope="module")
def fresh(clones, tmp_path_factory) -> Path:
    """A build made as the stored one was: with its streams, and its budget if it has one."""
    root = tmp_path_factory.mktemp("redraw")
    out = root / "v0.1"
    argv = ["build", "--clones", str(clones), "--out", str(out)]
    stored = DATA / "MANIFEST.json"
    manifest = json.loads(stored.read_text(encoding="utf-8")) if stored.is_file() else {}
    if manifest.get("streams") is not None:
        argv += ["--streams", str(root / "streams")]
    if manifest.get("graphiti_budget_usd") is not None:
        argv += ["--graphiti-budget-usd", repr(manifest["graphiti_budget_usd"]),
                 "--price-basis", manifest["graphiti_budget_basis"]]
    assert cli.main(argv) == 0
    return out


@pytest.fixture(scope="module")
def built(fresh):
    questions = [json.loads(line) for line in (fresh / "questions.jsonl").read_text().splitlines()]
    lines = (fresh / "gold.jsonl").read_text().splitlines()
    gold = {row["id"]: row for row in map(json.loads, lines)}
    pairs = [json.loads(line) for line in
             (fresh / "labelling" / "mention_pairs.jsonl").read_text().splitlines()]
    manifest = json.loads((fresh / "MANIFEST.json").read_text())
    by_stratum = defaultdict(list)
    for row in questions:
        by_stratum[(row["family"], row["stratum"], row["corpus"])].append(row["id"])
    return by_stratum, gold, pairs, manifest["eligible_pools"]


@pytest.fixture(scope="module")
def swept(corpus):
    return {pid: sweep(p, ORIGIN[p.family], keep_versions=p.family == "eth")
            for pid, p in corpus.proposals.items()}


@pytest.fixture(scope="module")
def present(corpus, swept):
    found = {"pep": {}, "eth": {}}
    for pid, p in corpus.proposals.items():
        if swept[pid].status(S) is not None:
            found[p.family][p.number] = p
    return found


@pytest.fixture(scope="module")
def at_s(corpus, present):
    """Bodies, mentions and requires at S, among the proposals present at S."""
    moment = eod(S)
    said, required = {}, {}
    with corpus.readers() as readers:
        for family, ps in present.items():
            for n, p in ps.items():
                body = p.body_at(moment, readers)
                said[p.pid] = {m for m in mentions(family, body, n) if m in ps}
    for n, p in present["eth"].items():
        required[n] = sorted({m for m in p.requires_at(moment) if m in present["eth"] and m != n})
    return said, required


def compare(built, family: str, stratum: str, corpus_key: str, pool: dict, size: int | None,
            picked) -> list[str]:
    """The built stratum against the re-derived pool, prefix and days; returns the ids."""
    by_stratum, _, _, pools = built
    assert pools[f"{family}/{stratum}/{corpus_key}"] == len(pool), (family, stratum, corpus_key)
    taken = hash_order(pool, family, stratum)
    taken = taken if size is None else taken[:size]
    expected = [f"{family}-{stratum}-{pid}-{picked(pid).isoformat()}" for pid in taken]
    assert by_stratum[(family, stratum, corpus_key)] == expected, (family, stratum, corpus_key)
    return expected


# --- F1 and CF -----------------------------------------------------------------------------------


def last_offered(s: Swept, family: str) -> str | None:
    """Section 7: the last status a question offers that the sweep shows at the end of a day up to
    S, so the status at S when a question offers it, otherwise the last offered one before S."""
    return next((status for status, _, _ in reversed(s.runs) if status in CHOICES[family]), None)


def test_f1_redrawn(built, corpus, swept, present):
    _, gold, _, _ = built
    for family in ("pep", "eth"):
        for stratum, size in (("change", 150), ("recent", 60)):
            pool = {}
            for p in present[family].values():
                s = swept[p.pid]
                if stratum == "change":
                    last = last_offered(s, family)
                    ranges = s.windows(lambda x, last=last, family=family:
                                       x in CHOICES[family] and x != last)
                else:
                    ranges = recent_ranges(p, s)
                if ranges:
                    pool[p.pid] = s.days(ranges)
            ids = compare(built, "F1", stratum, family, pool, size,
                          lambda pid, st=stratum, pool=pool: chosen(pool[pid], "F1", st, pid))
            for item_id in ids:
                pid, d = item_id.split("-", 2)[2].rsplit("-", 3)[0], date.fromisoformat(
                    item_id[-10:])
                assert gold[item_id]["gold"] == swept[pid].status(d), item_id
                if stratum == "change":
                    assert gold[item_id]["gold"] != last_offered(swept[pid], family), item_id
        pool = {}
        for p in present[family].values():
            s = swept[p.pid]
            status, first, last = s.runs[-1]
            if status in CHOICES[family] and first <= last - MARGIN:
                pool[p.pid] = None
        compare(built, "F1", "current", family, pool, 50, lambda pid: S)


def test_f1_change_reads_the_last_offered_status_at_s(built, swept, present):
    """Section 7: F1-change compares each window's status with the proposal's last offered status
    at S, so a Stagnant status at S, which records no new decision, is no change. At v0.1's pins
    396 of the 1,229 Ethereum proposals present at S are Stagnant there, and none of the 745 PEPs
    present at S holds a status that no question offers; the F1-change pools are 635 PEPs and 477
    Ethereum proposals, where comparing each window with the status at S itself gives 635 and 820.
    No F1-change item's gold is its proposal's last offered status at S."""
    by_stratum, gold, _, pools = built
    not_offered, pool, raw = Counter(), Counter(), Counter()
    for family in ("pep", "eth"):
        for p in present[family].values():
            s = swept[p.pid]
            at_s, last = s.status(S), last_offered(s, family)
            if at_s not in CHOICES[family]:
                not_offered[(family, at_s)] += 1
            offered = s.windows(lambda x, family=family: x in CHOICES[family])
            pool[family] += any(status != last for status, _, _ in offered)
            raw[family] += any(status != at_s for status, _, _ in offered)
        for item_id in by_stratum[("F1", "change", family)]:
            pid = item_id[len("F1-change-"):-len("-YYYY-MM-DD")]
            assert gold[item_id]["gold"] != last_offered(swept[pid], family), item_id
    assert {family: len(found) for family, found in present.items()} == {"pep": 745, "eth": 1229}
    assert dict(not_offered) == {("eth", "Stagnant"): 396}
    assert dict(pool) == {"pep": 635, "eth": 477} == {
        family: pools[f"F1/change/{family}"] for family in pool}
    assert dict(raw) == {"pep": 635, "eth": 820}


def recent_ranges(p, s: Swept) -> list[tuple[str, int, int]]:
    """Days whose margins lie in one status run and in intervals that are transitions starting
    on or after 2026-01-01 (end of every day from D-7 to D+7)."""
    intervals = p.intervals()
    starts = [iv.start_ts for iv in intervals]
    first = (RECENT_FROM - s.origin).days
    last = (S - s.origin).days
    qualifies = []
    for i in range(first, last + 1):
        k = bisect_right(starts, eod(s.origin + i * DAY)) - 1
        iv = intervals[k] if k >= 0 else None
        qualifies.append(iv is not None and iv.is_transition
                         and iv.start_ts >= eod(RECENT_FROM) - 86399)
    ranges = []
    for status, a, b in s.windows(lambda x: x in CHOICES[p.family]):
        days = [i for i in range(max(a, first + MARGIN), b + 1)
                if all(qualifies[i + k - first] for k in range(-MARGIN, MARGIN + 1))]
        ranges.extend((status, i, i) for i in days)
    return ranges


def test_cf_redrawn(built, corpus, swept, present, at_s):
    _, gold, _, _ = built
    said, required = at_s
    for family in ("pep", "eth"):
        score = Counter()
        for n, p in present[family].items():
            refs = set(said[p.pid]) | (set(required[n]) if family == "eth" else set())
            score.update(refs - {n})
        ranked = sorted(present[family], key=lambda n: (-score[n], n))[:50]
        pool = {}
        for n in ranked:
            s = swept[present[family][n].pid]
            ranges = s.windows(lambda x, family=family:
                               x in CHOICES[family] and permute(family, x) != x)
            if ranges:
                pool[present[family][n].pid] = s.days(ranges)
        ids = compare(built, "CF", "counterfactual", family, pool, 15,
                      lambda pid, pool=pool: chosen(pool[pid], "CF", "counterfactual", pid))
        for item_id in ids:
            pid = item_id[len("CF-counterfactual-"):-11]
            true = swept[pid].status(date.fromisoformat(item_id[-10:]))
            assert gold[item_id]["meta"]["true_status"] == true, item_id
            assert gold[item_id]["gold"] == permute(family, true) != true, item_id


# --- F2 and F3 -----------------------------------------------------------------------------------


def test_f2_redrawn(built, present, at_s):
    by_stratum, gold, _, _ = built
    said, _ = at_s
    peps = present["pep"]
    moment = eod(S)
    supersede = {n: set() for n in peps}
    for n, p in peps.items():
        header = p.version_at(moment).header
        for m in superseders(header):
            if m != n and m in peps:
                supersede[n].add(m)
        for old in number_list(header.get("replaces")):
            if old != n and old in peps:
                supersede[old].add(n)
    positives = sorted(n for n in peps if supersede[n])
    expected = [f"F2-positive-{peps[n].pid}-{S.isoformat()}" for n in positives]
    assert by_stratum[("F2", "positive", "pep")] == expected
    assert all(gold[i]["gold"] == sorted(supersede[n]) for i, n in zip(expected, positives,
                                                                         strict=True))
    pool = {peps[n].pid: None for n in peps if not supersede[n] and said[peps[n].pid]}
    compare(built, "F2", "negative", "pep", pool, len(positives), lambda pid: S)


def reach(graph: dict[int, list[int]], n: int) -> dict[int, int]:
    dist, queue = {n: 0}, deque([n])
    while queue:
        x = queue.popleft()
        for y in graph.get(x, ()):
            if y not in dist:
                dist[y] = dist[x] + 1
                queue.append(y)
    del dist[n]
    return dist


def test_f3_redrawn(built, present, at_s):
    _, gold, _, _ = built
    _, required = at_s
    indegree = Counter(m for ms in required.values() for m in ms)
    hubs = {m for m, _ in sorted(indegree.items(), key=lambda kv: (-kv[1], kv[0]))[:5]}
    strata = {"size-2-4": {}, "size-5-14": {}, "size-15-plus": {}}
    for n, p in present["eth"].items():
        dist = reach(required, n)
        if len(dist) < 2 or max(dist.values()) < 2 or set(dist) <= hubs:
            continue
        size = len(dist)
        name = "size-2-4" if size <= 4 else "size-5-14" if size <= 14 else "size-15-plus"
        strata[name][p.pid] = sorted(dist)
    for name, pool in strata.items():
        ids = compare(built, "F3", name, "eth", pool, 40, lambda pid: S)
        for item_id, pid in zip(ids, hash_order(pool, "F3", name), strict=False):
            assert gold[item_id]["gold"] == pool[pid], item_id


# --- F4 ------------------------------------------------------------------------------------------


def test_f4_redrawn(built, corpus, swept, present):
    """Brute force: every day's closure and Final members, for every proposal present at S."""
    _, gold, _, _ = built
    eth = {p.number: swept[p.pid] for p in corpus.proposals.values() if p.family == "eth"}
    days_total = (S - ORIGIN["eth"]).days + 1
    cache: dict[int, tuple[int, ...]] = {}

    def requires_on(n: int, i: int) -> tuple[int, ...] | None:
        v = eth[n].versions[i]
        if v is None:
            return None
        found = cache.get(id(v))
        if found is None:
            found = cache[id(v)] = tuple(number_list(v.header.get("requires")))
        return found

    def state(n: int, i: int):
        if requires_on(n, i) is None:
            return frozenset(), frozenset()
        seen, todo = {n}, [n]
        while todo:
            for y in requires_on(todo.pop(), i) or ():
                if y in eth and y not in seen and requires_on(y, i) is not None:
                    seen.add(y)
                    todo.append(y)
        seen.discard(n)
        finals = frozenset(m for m in seen if eth[m].versions[i].status == "Final")
        return frozenset(seen), finals

    pool, states_of = {}, {}
    for n, p in present["eth"].items():
        s = eth[n]
        if all(v is None or not requires_on(n, i) for i, v in enumerate(s.versions)):
            continue
        states = [state(n, i) for i in range(days_total)]
        days = []
        for i in range(MARGIN, days_total - MARGIN):
            members, _ = states[i]
            if 2 <= len(members) <= 14 and s.versions[i].status in CHOICES["eth"] and all(
                states[i + k] == states[i] for k in range(-MARGIN, MARGIN + 1)
            ):
                days.append(ORIGIN["eth"] + i * DAY)
        if days:
            pool[p.pid] = days
            states_of[p.pid] = states
    ids = compare(built, "F4", "closure-final", "eth", pool, 120,
                  lambda pid: chosen(pool[pid], "F4", "closure-final", pid))
    for item_id in ids:
        pid = item_id[len("F4-closure-final-"):-11]
        i = (date.fromisoformat(item_id[-10:]) - ORIGIN["eth"]).days
        members, finals = states_of[pid][i]
        assert gold[item_id]["gold"] == sorted(finals), item_id
        assert gold[item_id]["meta"]["closure"] == sorted(members), item_id
        assert gold[item_id]["meta"]["gold_at_s"] == sorted(states_of[pid][-1][1]), item_id


# --- F5 ------------------------------------------------------------------------------------------


F5_DATES = {
    "pep": (date(2004, 1, 1), date(2008, 1, 1), date(2012, 1, 1), date(2016, 1, 1),
            date(2020, 1, 1), date(2024, 1, 1), S),
    "eth": (date(2018, 1, 1), date(2020, 1, 1), date(2022, 1, 1), date(2024, 1, 1),
            date(2026, 1, 1), S),
}
F5_STATUSES = {"pep": ("Draft", "Rejected", "Withdrawn", "Final"),
               "eth": ("Draft", "Review", "Final", "Withdrawn")}


def test_f5_redrawn(built, corpus, swept):
    by_stratum, gold, _, _ = built
    expected = []
    for family in ("pep", "eth"):
        family_sweeps = [swept[p.pid] for p in corpus.proposals.values() if p.family == family]
        for registered in F5_DATES[family]:
            for status in F5_STATUSES[family]:
                d = registered
                while True:
                    span = range(-F5_MARGIN, 1) if d == S else range(-F5_MARGIN, F5_MARGIN + 1)
                    if d != S and d + F5_MARGIN * DAY > S:
                        d = None
                        break
                    if all(len({s.status(d + k * DAY) == status for k in span}) == 1
                           for s in family_sweeps):
                        break
                    if d == S:
                        d = None
                        break
                    d += DAY
                if d is None:
                    continue
                item_id = f"F5-count-{family}-{status}-{d.isoformat()}"
                expected.append(item_id)
                count = sum(1 for s in family_sweeps if s.status(d) == status)
                assert gold[item_id]["gold"] == count, item_id
                assert gold[item_id]["meta"]["registered_date"] == registered.isoformat()
        assert by_stratum[("F5", "count", family)] == [i for i in expected
                                                       if i.startswith(f"F5-count-{family}-")]


# --- mention pairs -------------------------------------------------------------------------------


def test_mention_pairs_redrawn(built, present, at_s):
    _, _, pairs, pools = built
    said, _ = at_s
    moment = eod(S)
    for family in ("pep", "eth"):
        ps = present[family]
        linked = set()
        for n, p in ps.items():
            header = p.version_at(moment).header
            related = number_list(header.get("requires"))
            if family == "pep":
                related += superseders(header) + number_list(header.get("replaces"))
            linked.update(frozenset((n, m)) for m in related if m != n)
        pool = set()
        for n, p in ps.items():
            for m in said[p.pid]:
                a, b = min(n, m), max(n, m)
                if frozenset((a, b)) not in linked:
                    pool.add(f"{ps[a].pid}~{ps[b].pid}")
        assert pools[f"labelling/{family}/{family}"] == len(pool)
        assert [r["pair_id"] for r in pairs if r["corpus"] == family] == hash_order(
            pool, "labelling", family)[:100]


# --- the stored build ----------------------------------------------------------------------------


def without_checkout(manifest: dict) -> dict:
    """The manifest without the generator fields that describe the checkout, not the build."""
    generator = {k: v for k, v in manifest["generator"].items() if k not in CHECKOUT_ONLY}
    return {**manifest, "generator": generator}


def test_fresh_build_equals_data_v0_1(fresh):
    """``data/v0.1`` is what the code builds today: its seven data files byte for byte, and its
    manifest (pins, file hashes, stream hashes, source digest, budget) as parsed JSON once the
    checkout's commit and dirty flag are left out, so the test holds before and after the tree is
    committed. The stored manifest's file hashes match the stored files."""
    if not (DATA / "MANIFEST.json").is_file():
        pytest.skip("data/v0.1 has not been built")
    stored = sorted(str(p.relative_to(DATA)) for p in DATA.rglob("*") if p.is_file())
    assert stored == sorted(FILES)
    for name in DATA_FILES:
        assert (fresh / name).read_bytes() == (DATA / name).read_bytes(), name
    stored_manifest = json.loads((DATA / "MANIFEST.json").read_text(encoding="utf-8"))
    fresh_manifest = json.loads((fresh / "MANIFEST.json").read_text(encoding="utf-8"))
    assert set(CHECKOUT_ONLY) <= set(stored_manifest["generator"])
    assert without_checkout(fresh_manifest) == without_checkout(stored_manifest)
    for name, entry in stored_manifest["files"].items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == entry["sha256"], name
