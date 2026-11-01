"""The build on the real clones (corpus-marked; set ASOFBENCH_CLONES to run them).

The build runs once for the module through the command line. Every gold check here reads git or
the timelines independently of the selection code: F1 and CF gold against ``git show`` of the
recorded file at the item's as-of commit, and F4 gold against a closure recomputed day by day.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from collections import Counter
from datetime import date, timedelta

import pytest

from asofbench import cli
from asofbench.corpora import CANARY, S
from asofbench.gitwalk import end_of_day_ts
from asofbench.headers import parse_header
from asofbench.mentions import mentions
from asofbench.statuses import CHOICES, normalize_status, permute
from asofbench.timeline import load_corpus

pytestmark = pytest.mark.corpus

DAY = timedelta(days=1)
EOD_S = end_of_day_ts(S)
FILES = ("questions.jsonl", "gold.jsonl", "pilot.json", "labelling/mention_pairs.jsonl",
         "timelines.jsonl", "edges.jsonl", "REPORT.md", "MANIFEST.json")


def git(repo, *args: str) -> bytes:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL="/dev/null")
    return subprocess.run(["git", "-C", str(repo), *args], env=env, capture_output=True,
                          check=True).stdout


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def built(clones, tmp_path_factory):
    out = tmp_path_factory.mktemp("asofbench") / "v0.1"
    start = time.monotonic()
    assert cli.main(["build", "--clones", str(clones), "--out", str(out)]) == 0
    return out, time.monotonic() - start


@pytest.fixture(scope="module")
def out(built):
    return built[0]


@pytest.fixture(scope="module")
def corpus(clones):
    return load_corpus(clones)


@pytest.fixture(scope="module")
def rows(out):
    questions = {row["id"]: row for row in read_jsonl(out / "questions.jsonl")}
    gold = {row["id"]: row for row in read_jsonl(out / "gold.jsonl")}
    assert list(questions) == list(gold)
    return questions, gold


def test_build_time(built):
    assert built[1] < 600


def test_gold_matches_git_show(rows, clones):
    questions, gold = rows
    checked = 0
    for item_id, row in gold.items():
        family = questions[item_id]["family"]
        if family not in ("F1", "CF"):
            assert row["source"] is None, item_id
            continue
        source = row["source"]
        assert source["commit"] == row["as_of_commits"][source["repo"]], item_id
        text = git(clones / source["repo"], "show", "--no-textconv",
                   f"{source['commit']}:{source['path']}").decode("utf-8", errors="replace")
        corpus_key = questions[item_id]["corpus"]
        status = normalize_status(corpus_key, parse_header(corpus_key, text).get("status"))
        if family == "CF":
            assert status == row["meta"]["true_status"], item_id
            assert row["gold"] == permute(corpus_key, status) != status, item_id
        else:
            assert row["gold"] == status, item_id
        assert row["gold"] in CHOICES[corpus_key], item_id
        checked += 1
    assert checked >= 2 * 150 + 2 * 50 + 30


def counts(questions):
    found: dict[tuple[str, str, str], int] = {}
    for row in questions.values():
        key = (row["family"], row["stratum"], row["corpus"])
        found[key] = found.get(key, 0) + 1
    return found


def test_registered_sizes(rows, out):
    questions, _ = rows
    n = counts(questions)
    manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    pools = {tuple(key.split("/")): value for key, value in manifest["eligible_pools"].items()}
    for corpus_key in ("pep", "eth"):
        assert n[("F1", "change", corpus_key)] == 150
        assert n[("F1", "current", corpus_key)] == 50
        recent = n.get(("F1", "recent", corpus_key), 0)
        assert recent == min(60, pools[("F1", "recent", corpus_key)])
        assert n[("CF", "counterfactual", corpus_key)] == 15
    assert n[("F2", "positive", "pep")] == n[("F2", "negative", "pep")] > 0
    for stratum in ("size-2-4", "size-5-14", "size-15-plus"):
        assert n.get(("F3", stratum, "eth"), 0) == min(40, pools[("F3", stratum, "eth")])
    assert n[("F4", "closure-final", "eth")] == 120
    assert n[("F5", "count", "pep")] <= 28 and n[("F5", "count", "eth")] <= 24
    pairs = read_jsonl(out / "labelling" / "mention_pairs.jsonl")
    assert sum(1 for p in pairs if p["corpus"] == "pep") == 100
    assert sum(1 for p in pairs if p["corpus"] == "eth") == 100
    assert len({p["pair_id"] for p in pairs}) == 200


def test_build_is_deterministic(out, clones, tmp_path):
    again = tmp_path / "v0.1"
    assert cli.main(["build", "--clones", str(clones), "--out", str(again)]) == 0
    for name in FILES:
        assert (again / name).read_bytes() == (out / name).read_bytes(), name


def test_every_item_is_about_a_proposal_present_at_s(rows, corpus):
    """Section 7: every item about one proposal concerns a proposal present at S; F4 closure
    members and the proposals F5 counts need not be."""
    questions, _ = rows
    about = [row for row in questions.values() if row["proposal"] is not None]
    assert len(about) == len(questions) - sum(1 for r in questions.values() if r["family"] == "F5")
    for row in about:
        assert corpus.proposals[row["proposal"]].status_at(EOD_S) is not None, row["id"]
    deleted = {p.pid for p in corpus.proposals.values() if p.status_at(EOD_S) is None}
    assert len(deleted) == 8 and not deleted & {row["proposal"] for row in about}


def test_sampled_dates_hold_over_their_margins(rows, corpus):
    questions, gold = rows
    for item_id, row in questions.items():
        if row["family"] not in ("F1", "CF") or row["stratum"] == "current":
            continue
        p = corpus.proposals[row["proposal"]]
        d = date.fromisoformat(row["as_of"])
        expected = gold[item_id]["meta"].get("true_status", gold[item_id]["gold"])
        assert d <= S - 7 * DAY, item_id
        for k in range(-7, 8):
            assert p.status_at(end_of_day_ts(d + k * DAY)) == expected, (item_id, k)


def closure_at(present: dict, number: int, moment: int) -> set[int]:
    seen, todo = set(), [number]
    while todo:
        n = todo.pop()
        for m in present[n].requires_at(moment) if n in present else ():
            if m != number and m in present and m not in seen:
                seen.add(m)
                todo.append(m)
    return seen


def test_f4_gold_recomputed_day_by_day(rows, corpus):
    questions, gold = rows
    eth = [p for p in corpus.proposals.values() if p.family == "eth"]
    checked = 0
    for item_id, row in questions.items():
        if row["family"] != "F4":
            continue
        d = date.fromisoformat(row["as_of"])
        number = corpus.proposals[row["proposal"]].number
        assert corpus.proposals[row["proposal"]].status_at(end_of_day_ts(d)) in CHOICES["eth"]
        reference = None
        for k in range(-7, 8):
            moment = end_of_day_ts(d + k * DAY)
            present = {p.number: p for p in eth if p.status_at(moment) is not None}
            members = closure_at(present, number, moment)
            finals = sorted(m for m in members if present[m].status_at(moment) == "Final")
            if reference is None:
                reference = (members, finals)
                assert 2 <= len(members) <= 14, item_id
                assert gold[item_id]["gold"] == finals, item_id
                assert gold[item_id]["meta"]["closure"] == sorted(members), item_id
            assert (members, finals) == reference, (item_id, k)
        checked += 1
    assert checked == 120


def test_canary_everywhere(out):
    for name in ("questions.jsonl", "gold.jsonl", "labelling/mention_pairs.jsonl",
                 "timelines.jsonl", "edges.jsonl"):
        assert all(record["canary"] == CANARY for record in read_jsonl(out / name)), name
    for name in ("pilot.json", "MANIFEST.json"):
        assert json.loads((out / name).read_text(encoding="utf-8"))["canary"] == CANARY
    assert CANARY in (out / "REPORT.md").read_text(encoding="utf-8")


def test_pilot(rows, out):
    questions, gold = rows
    pilot = json.loads((out / "pilot.json").read_text(encoding="utf-8"))
    sizes = {(g["family"], g["stratum"], g["corpus"]): len(g["ids"]) for g in pilot["groups"]}
    assert sizes == {
        ("F1", "change", "pep"): 50, ("F1", "change", "eth"): 50,
        ("F1", "current", "pep"): 25, ("F1", "current", "eth"): 25,
        ("CF", "counterfactual", "pep"): 15, ("CF", "counterfactual", "eth"): 15,
    }
    for group in pilot["groups"]:
        ranks = [gold[i]["meta"]["rank"] for i in group["ids"]]
        assert ranks == list(range(len(ranks)))
        assert all(questions[i]["family"] == group["family"] for i in group["ids"])
    assert len(pilot["m0_graphiti_proposals"]) == 50
    assert all(pid.startswith("PEP-") for pid in pilot["m0_graphiti_proposals"])


def pinned_headers(corpus, clones) -> dict[tuple[str, int], dict]:
    """The header of every proposal file present at the pins, read with ``git cat-file --batch``
    from ``<pin>:<path>``, independently of the timelines' blobs."""
    wanted: dict[str, list[tuple[tuple[str, int], str]]] = {}
    for p in corpus.proposals.values():
        v = p.version_at(EOD_S)
        if v is not None and v.status != "absent":
            wanted.setdefault(v.repo, []).append(((p.family, p.number), v.path))
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL="/dev/null")
    headers = {}
    for repo, entries in wanted.items():
        request = "".join(f"{corpus.pins[repo]}:{path}\n" for _, path in entries).encode()
        out = subprocess.run(["git", "-C", str(clones / repo), "cat-file", "--batch"], env=env,
                             input=request, capture_output=True, check=True).stdout
        pos = 0
        for key, _ in entries:
            line_end = out.index(b"\n", pos)
            _, kind, size = out[pos:line_end].split()
            assert kind == b"blob"
            text = out[line_end + 1 : line_end + 1 + int(size)].decode("utf-8", errors="replace")
            headers[key] = parse_header(key[0], text)
            pos = line_end + 1 + int(size) + 1
    return headers


def test_f2_and_f3_gold_from_the_pinned_files(rows, corpus, clones):
    from asofbench.headers import number_list, superseders

    questions, gold = rows
    headers = pinned_headers(corpus, clones)
    peps = {n for family, n in headers if family == "pep"}
    eth = {n for family, n in headers if family == "eth"}
    superseded_by: dict[int, set[int]] = {n: set() for n in peps}
    for n in peps:
        for m in superseders(headers[("pep", n)]):
            if m != n and m in peps:
                superseded_by[n].add(m)
        for old in number_list(headers[("pep", n)].get("replaces")):
            if old != n and old in peps:
                superseded_by[old].add(n)
    requires = {n: [m for m in number_list(headers[("eth", n)].get("requires")) if m in eth]
                for n in eth}
    checked = Counter()
    for item_id, row in questions.items():
        number = corpus.proposals[row["proposal"]].number if row["proposal"] else None
        if row["family"] == "F2":
            assert gold[item_id]["gold"] == sorted(superseded_by[number]), item_id
            assert bool(gold[item_id]["gold"]) == (row["stratum"] == "positive"), item_id
        elif row["family"] == "F3":
            seen, todo = set(), [number]
            while todo:
                for m in requires[todo.pop()]:
                    if m != number and m not in seen:
                        seen.add(m)
                        todo.append(m)
            assert gold[item_id]["gold"] == sorted(seen), item_id
        else:
            continue
        checked[row["family"]] += 1
    assert checked["F2"] == 2 * sum(1 for n in peps if superseded_by[n])
    assert checked["F3"] == 120


def test_the_header_edges_from_the_pinned_files(out, corpus, clones):
    """Section 9's relation gold, edges.jsonl, recomputed from the files at the pins read with
    ``git cat-file`` (independently of the timelines' blobs), and its size at the pins."""
    from asofbench.headers import number_list, superseders

    headers = pinned_headers(corpus, clones)
    pid = {(p.family, p.number): p.pid for p in corpus.proposals.values()}
    present = {family: {n for f, n in headers if f == family} for family in ("pep", "eth")}
    expected = set()
    for (family, n), header in headers.items():
        for m in number_list(header.get("requires")):
            if m != n and m in present[family]:
                expected.add(("requires", family, pid[(family, n)], pid[(family, m)]))
        if family == "pep":
            for m in superseders(header):
                if m != n and m in present["pep"]:
                    expected.add(("supersedes", "pep", pid[("pep", m)], pid[("pep", n)]))
            for old in number_list(header.get("replaces")):
                if old != n and old in present["pep"]:
                    expected.add(("supersedes", "pep", pid[("pep", n)], pid[("pep", old)]))
    records = read_jsonl(out / "edges.jsonl")
    assert all(r["canary"] == CANARY for r in records)
    found = [(r["relation"], r["corpus"], r["source"], r["target"]) for r in records]
    assert sorted(found, key=lambda e: (e[0], e[2], e[3])) == found  # sorted as registered
    assert len(found) == len(set(found)) and set(found) == expected
    assert Counter((relation, family) for relation, family, _, _ in found) == {
        ("requires", "eth"): 1798, ("requires", "pep"): 41, ("supersedes", "pep"): 46}


def test_f2_recoverability_and_cf_header_cues_at_the_pins(rows, corpus, clones):
    """The F2 items whose gold the asked PEP's own episodes can show, and the CF items whose file
    at D has a header cue, counted at the pins.

    The F2 half walks each asked PEP's versions with the package's own header and mention helpers
    (``superseders``, ``header_span``, ``mentions``), so it is no independent recomputation: its
    guard is the pinned counts, 32 and 15. The CF half reads each CF file with git show at its
    as-of commit and writes section 7's rule out on its own: a last-call-deadline under any of
    its names, whatever its value; a Superseded-By, Replaced-By, withdrawal-reason or Accepted
    with a value; a Resolution whose case-folded value holds a decision word; no other field."""
    from asofbench.headers import header_span, superseders

    questions, gold = rows
    recoverable = {"H+": [], "H-": []}
    with corpus.readers() as readers:
        for item_id, q in questions.items():
            if q["family"] != "F2":
                continue
            p, members = corpus.proposals[q["proposal"]], set(gold[item_id]["gold"])
            listed, said = set(), set()
            for v in p.versions:
                if v.status != "absent" and v.blob is not None:
                    listed |= set(superseders(v.header))
                    text = readers[v.repo].read(v.blob)
                    said |= mentions("pep", text[header_span("pep", text)[1]:], p.number)
            flags = {"H+": members <= listed | said, "H-": members <= said}
            assert gold[item_id]["meta"]["recoverable"] == flags, item_id
            assert q["stratum"] == "positive" or flags == {"H+": True, "H-": True}, item_id
            for arm, ok in flags.items():
                if ok and q["stratum"] == "positive":
                    recoverable[arm].append(q["proposal"])
    # Of the 41 positives, 32 are recoverable with the headers visible and 15 with them withheld.
    assert (len(recoverable["H+"]), len(recoverable["H-"])) == (32, 15)
    last_call = ("last-call-deadline", "review-period-end", "review-period-ends",
                 "review-end-date")
    with_a_value = ("superseded-by", "replaced-by", "withdrawal-reason", "accepted")
    words = ("accept", "reject", "withdraw", "final", "supersede")
    cued = {}
    for item_id, q in questions.items():
        if q["family"] != "CF":
            continue
        source = gold[item_id]["source"]
        text = git(clones / source["repo"], "show", "--no-textconv",
                   f"{source['commit']}:{source['path']}").decode("utf-8", errors="replace")
        fields = [key for key, value in parse_header(q["corpus"], text).items()
                  if key in last_call or (key in with_a_value and value)
                  or (key == "resolution" and any(w in value.casefold() for w in words))]
        assert gold[item_id]["meta"]["header_cue_fields"] == fields, item_id
        assert gold[item_id]["meta"]["header_cue"] is bool(fields), item_id
        if fields:
            cued[item_id] = fields
    assert cued == {
        "CF-counterfactual-EIP-5792-2024-06-23": ["last-call-deadline"],
        "CF-counterfactual-EIP-7732-2026-10-13": ["last-call-deadline"],
        "CF-counterfactual-EIP-7928-2026-10-13": ["last-call-deadline"],
        "CF-counterfactual-PEP-426-2026-08-21": ["superseded-by"],
    }


def test_the_labelling_sample_holds_locations_and_the_sheet_holds_the_passages(out, clones,
                                                                              tmp_path):
    """Section 9: each published mention is a file at its pinned commit and the character offsets
    of a mention of the other proposal (checked against ``git show``); the passages, which only
    the local sheet holds, appear nowhere in the published file."""
    records = read_jsonl(out / "labelling" / "mention_pairs.jsonl")
    spans = [(r, side, s) for r in records for side in ("a", "b")
             for s in r[f"{side}_mention_spans"]]
    assert len(records) == 200 and len(spans) == 263
    texts = {}
    for r, side, s in spans:
        key = (s["repo"], s["commit"], s["path"])
        if key not in texts:
            texts[key] = git(clones / s["repo"], "show", "--no-textconv",
                             f"{s['commit']}:{s['path']}").decode("utf-8", errors="replace")
        other = r["b" if side == "a" else "a"]
        assert s["commit"] == r[f"{side}_source"]["commit"], r["pair_id"]
        assert mentions(r["corpus"], texts[key][s["start"]:s["end"]], -1) == {
            int(other.split("-")[1])}, (r["pair_id"], s)
    sheet = tmp_path / "labelling-sheet.md"
    argv = ["labelling-sheet", "--clones", str(clones), "--data", str(out), "--out", str(sheet)]
    assert cli.main(argv) == 0
    lines = sheet.read_text(encoding="utf-8").split("\n")
    passages = [lines[i + 1] for i, line in enumerate(lines) if re.fullmatch(r"`{3,}text", line)]
    assert len(passages) == 263 and all(0 < len(p) <= 200 for p in passages)
    published = (out / "labelling" / "mention_pairs.jsonl").read_text(encoding="utf-8")
    assert not [p for p in passages if len(p) >= 40 and p[:40] in published]
