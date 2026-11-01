"""The three streams on the real clones (corpus-marked; set ASOFBENCH_CLONES to run them).

The questions are built and the three streams are written through the command line, once for the
module. The checks do not trust the builder. Each arm is walked in stream order and every
proposal's text is rebuilt by applying the diffs to the create text; that state must equal what
git holds for the version, after the arm's edit as ``streamcheck.py`` writes it again from the
registration. The H- arm is also searched for withheld header lines, and the CF arm is compared
with H+ episode by episode. Finally each stream's state as of the question dates must say what
the gold says.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from collections import Counter, defaultdict
from datetime import date

import pytest
from streamcheck import (
    apply_diff,
    diff_header_hits,
    leaked_lines,
    permute_expected,
    status_value,
    strip_expected,
)

from asofbench import cli
from asofbench.corpora import CANARY, S
from asofbench.gitwalk import end_of_day_ts
from asofbench.mentions import mentions
from asofbench.statuses import CANONICAL, normalize_status
from asofbench.stream import KINDS, unified_diff
from asofbench.timeline import load_corpus

pytestmark = pytest.mark.corpus

FILES = {"H+": "h-plus.jsonl", "H-": "h-minus.jsonl", "CF": "cf.jsonl"}
ABSENT = "absent"


def git(repo, *args: str) -> bytes:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL="/dev/null")
    return subprocess.run(["git", "-C", str(repo), *args], env=env, capture_output=True,
                          check=True).stdout


def read_jsonl(path):
    # Split at newlines only: a text may hold characters that str.splitlines would cut at.
    with open(path, encoding="utf-8", newline="\n") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture(scope="module")
def data(clones, tmp_path_factory):
    out = tmp_path_factory.mktemp("asofbench") / "v0.1"
    assert cli.main(["build", "--clones", str(clones), "--out", str(out)]) == 0
    return out


@pytest.fixture(scope="module")
def built(clones, data, tmp_path_factory):
    """Each arm's stream file and the seconds the command took."""
    root = tmp_path_factory.mktemp("streams")
    found = {}
    for arm, name in FILES.items():
        out = root / name
        start = time.monotonic()
        argv = ["stream", "--clones", str(clones), "--data", str(data), "--arm", arm,
                "--out", str(out)]
        assert cli.main(argv) == 0
        found[arm] = (out, time.monotonic() - start)
    return found


@pytest.fixture(scope="module")
def corpus(clones):
    return load_corpus(clones)


@pytest.fixture(scope="module")
def streams(built):
    return {arm: read_jsonl(path) for arm, (path, _) in built.items()}


@pytest.fixture(scope="module")
def questions(data):
    return {row["id"]: row for row in read_jsonl(data / "questions.jsonl")}


@pytest.fixture(scope="module")
def gold(data):
    return {row["id"]: row for row in read_jsonl(data / "gold.jsonl")}


@pytest.fixture(scope="module")
def cf_proposals(questions):
    """The CF proposals, read from the question records (the builder reads gold.jsonl's ids)."""
    return {row["proposal"] for row in questions.values() if row["family"] == "CF"}


# --- coverage and build time ---------------------------------------------------------------------


def test_streams_cover_question_proposals(streams, questions):
    for arm, rows in streams.items():
        named = {q["proposal"] for q in questions.values()
                 if arm in q["arms"] and q["proposal"] is not None}
        have = {row["proposal"] for row in rows}
        assert named and named <= have, (arm, sorted(named - have)[:10])
    # The H+ arm is the full revision stream; the other two are the same episodes, edited.
    assert {row["proposal"] for row in streams["H+"]} >= {
        q["proposal"] for q in questions.values() if q["proposal"] is not None
    }


def test_stream_time(built):
    for arm, (_, seconds) in built.items():
        assert seconds < 600, arm


# --- shape ---------------------------------------------------------------------------------------


def test_stats_describe_the_file(built, streams):
    sizes = {}
    for arm, (path, _) in built.items():
        rows = streams[arm]
        chars = sum(len(row["text"]) for row in rows)
        stats = json.loads(open(f"{path}.stats.json", encoding="utf-8").read())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert stats == {"episodes": len(rows), "chars": chars, "tokens_est": chars // 4,
                         "sha256": digest}, arm
        sizes[arm] = stats
    assert sizes["H+"]["episodes"] == sizes["H-"]["episodes"] == sizes["CF"]["episodes"]
    assert sizes["H-"]["chars"] < sizes["H+"]["chars"]


def test_episodes_are_one_per_presence_change(streams, corpus):
    """Every version is an episode except one that leaves the proposal without text (a symlink,
    or a deletion of what was a symlink); no episode appears twice."""
    expected = 0
    for p in corpus.proposals.values():
        for i, v in enumerate(p.versions):
            if v.status == ABSENT and (i == 0 or p.versions[i - 1].status == ABSENT):
                continue
            expected += 1
    assert expected > 26000
    for arm, rows in streams.items():
        assert len(rows) == expected, arm
        ids = [row["episode_id"] for row in rows]
        assert len(set(ids)) == len(ids), arm
        assert all(row["kind"] in KINDS for row in rows)
        assert all(row["episode_id"] == f"{row['commit'][:12]}:{row['proposal']}" for row in rows)
        assert all(row["text"] == "" for row in rows if row["kind"] == "delete")
        assert all(row["text"] != "" for row in rows if row["kind"] == "create")


def test_episodes_are_in_recorded_order(streams, corpus):
    position = {repo: {c.sha: i for i, c in enumerate(chain)}
                for repo, chain in corpus.commits.items()}
    rows = streams["H+"]
    keys = [(row["recorded_at"], row["repo"], position[row["repo"]][row["commit"]],
             row["proposal"]) for row in rows]
    assert keys == sorted(keys)
    assert rows[-1]["recorded_at"] <= f"{S.isoformat()}T23:59:59Z"
    assert [row["episode_id"] for row in streams["H-"]] == [row["episode_id"] for row in rows]
    assert [row["episode_id"] for row in streams["CF"]] == [row["episode_id"] for row in rows]


def test_proposals_that_are_not_in_the_corpus_have_no_episodes(streams):
    seen = {row["proposal"] for row in streams["H+"]}
    assert "PEP-0" not in seen and "PEP-401" not in seen
    assert {"PEP-1", "PEP-572", "EIP-1559", "ERC-20"} <= seen


def test_pep_495_is_created_by_the_regular_file_not_the_symlink(streams):
    # A file numbered 495 (the 3.6 release schedule) lived for seven minutes and was deleted;
    # the proposal came back as a symlink, which is no text, and then as a regular file.
    pep495 = [row for row in streams["H+"] if row["proposal"] == "PEP-495"]
    assert [row["kind"] for row in pep495[:3]] == ["create", "delete", "create"]
    assert pep495[2]["commit"].startswith("7acfbeaf9f")
    assert pep495[2]["text"].startswith("PEP: 495\nTitle: Local Time Disambiguation\n")
    assert not [row for row in pep495 if row["commit"].startswith("135c01161b")]


def test_ercs_keep_one_id_across_the_two_repositories(streams, corpus):
    by_proposal = defaultdict(list)
    for row in streams["H+"]:
        if row["proposal"].startswith("ERC-"):
            by_proposal[row["proposal"]].append(row)
    assert len(by_proposal) == sum(1 for p in corpus.proposals.values() if p.series == "ERC")
    moved = 0
    for pid, rows in by_proposal.items():
        repos = [row["repo"] for row in rows]
        assert repos == sorted(repos, key=["EIPs", "ERCs"].index), pid  # EIP repo, then ERCs
        if "EIPs" in repos:
            joined = rows[repos.index("ERCs")]
            assert joined["kind"] == "rename", pid  # the file moved: a diff, not a new create
            moved += joined["message"] == "complete erc move"
            assert rows[0]["kind"] == "create" and rows[0]["repo"] == "EIPs", pid
        else:
            assert rows[0]["kind"] == "create", pid
    assert moved == 365  # section 6: the one commit that created 365 ERC files
    assert {row["corpus"] for row in streams["H+"] if row["proposal"] == "ERC-20"} == {"eth"}


def test_refs_are_the_questions_refs(streams, questions, gold):
    ref_of = {row["proposal"]: row["ref"] for row in streams["H+"]}
    checked = 0
    for item_id, q in questions.items():
        if q["proposal"] is None:
            continue
        assert ref_of[q["proposal"]] == gold[item_id]["meta"]["ref"], item_id
        checked += 1
    assert checked > 800


def test_h_plus_matches_git_show_and_commit_messages(streams, corpus, clones):
    """A sample checked against git itself rather than the walker: the create text is the
    committed file and the message is the commit's."""
    by_id = {f"{v.commit[:12]}:{pid}": (p, v)
             for pid, p in corpus.proposals.items() for v in p.versions}
    checked = 0
    for i, row in enumerate(streams["H+"]):
        if i % 97:
            continue
        p, v = by_id[row["episode_id"]]
        message = git(clones / v.repo, "log", "-1", "--format=%B", v.commit)
        assert row["message"] == message.decode("utf-8", errors="replace").rstrip("\n")
        if row["kind"] == "create":
            text = git(clones / v.repo, "show", "--no-textconv", f"{v.commit}:{v.path}")
            assert row["text"] == text.decode("utf-8", errors="replace"), row["episode_id"]
        checked += 1
    assert checked > 200


# --- the state of each arm, rebuilt from its diffs -----------------------------------------------


def status_line_commits(corpus, cf):
    """The commits that change the status line of a CF proposal, read from the git texts: a
    present version whose status line differs from that of the version before it."""
    found = set()
    with corpus.readers() as readers:
        for pid in cf:
            before = None
            for v in corpus.proposals[pid].versions:
                if v.status == ABSENT:
                    before = None
                    continue
                value = status_value(readers[v.repo].read(v.blob))
                if before is not None and value != before[0]:
                    found.add(v.commit)
                before = (value,)
    return found


def walk(arm, rows, corpus, cf, cf_refs=None):
    """Rebuild every proposal's text from ``rows`` and compare it with git.

    Returns the lists of problems and, per proposal, the status it shows after each episode.
    In the CF arm the messages are checked against section 7: in a commit that changes a CF
    status line the CF proposal's episode says ``Update status of {ref}`` and every other episode
    ``Update {ref}``; in any other commit the message is the commit's.
    """
    by_id = {f"{v.commit[:12]}:{pid}": (p, v)
             for pid, p in corpus.proposals.items() for v in p.versions}
    messages = {(repo, c.sha): c.message for repo, chain in corpus.commits.items() for c in chain}
    touched = status_line_commits(corpus, cf) if arm == "CF" else set()

    def wanted(row, original, changed):
        if row["commit"] not in touched:
            return original
        return f"Update status of {cf_refs[row['proposal']]}" if changed else f"Update {row['ref']}"

    def view(p, text):
        if arm == "H-":
            return strip_expected(p.family, text)
        if arm == "CF" and p.pid in cf:
            return permute_expected(p.family, text)
        return text

    found = {"kind": [], "text": [], "state": [], "message": [], "leak": [], "rewritten": [],
             "status_changes": [], "siblings": [], "touched": touched, "diffs": 0, "checked": 0}
    shown: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
    state: dict[str, str] = {}
    last: dict[str, object] = {}
    raw_status: dict[str, str | None] = {}
    with corpus.readers() as readers:
        for row in rows:
            p, v = by_id[row["episode_id"]]
            pid = p.pid
            message = messages[(v.repo, v.commit)]
            if v.status == ABSENT:
                if pid not in state or row["kind"] != "delete" or row["text"] != "":
                    found["kind"].append(row["episode_id"])
                state.pop(pid, None)
                last.pop(pid, None)
                raw_status.pop(pid, None)
                shown[pid].append((row["recorded_at"], None))
                if row["message"] != wanted(row, message, False):
                    found["message"].append(row["episode_id"])
                if row["message"] != message:
                    found["rewritten"].append(row["episode_id"])
                    found["siblings"].append(row["episode_id"])
                continue
            raw = readers[v.repo].read(v.blob)
            after = view(p, raw)
            if pid not in state:
                kind = "create"
            elif (last[pid].repo, last[pid].path) != (v.repo, v.path):
                kind = "rename"
            else:
                kind = "modify"
            if row["kind"] != kind:
                found["kind"].append(row["episode_id"])
            if kind == "create":
                if row["text"] != after:
                    found["text"].append(row["episode_id"])
                rebuilt_ok = True
            else:
                before = state[pid]
                if row["text"] != unified_diff(before, after):
                    found["text"].append(row["episode_id"])
                try:
                    rebuilt_ok = apply_diff(before, row["text"]) == after
                except ValueError:
                    rebuilt_ok = False
                if arm == "H-":
                    found["leak"] += [(row["episode_id"], line)
                                      for line in diff_header_hits(row["text"], before, after)]
            if not rebuilt_ok:
                found["state"].append(row["episode_id"])
            found["diffs"] += kind != "create"
            found["checked"] += 1
            if arm == "H-":
                found["leak"] += [(row["episode_id"], line) for line in leaked_lines(after)]
                if kind == "create":
                    found["leak"] += [(row["episode_id"], line)
                                      for line in leaked_lines(row["text"])]
            # The message: the commit's, except in the CF arm (section 7).
            changed = arm == "CF" and pid in cf and kind != "create" and (
                status_value(raw) != raw_status[pid])
            if changed:
                found["status_changes"].append(row["episode_id"])
            if row["message"] != wanted(row, message, changed):
                found["message"].append(row["episode_id"])
            if row["message"] != message:
                found["rewritten"].append(row["episode_id"])
                if not changed:
                    found["siblings"].append(row["episode_id"])
            state[pid] = after
            last[pid] = v
            raw_status[pid] = status_value(raw)
            shown[pid].append((row["recorded_at"], status_value(after)))
    return found, shown


@pytest.fixture(scope="module")
def cf_refs(questions, gold):
    return {q["proposal"]: gold[i]["meta"]["ref"] for i, q in questions.items()
            if q["family"] == "CF"}


@pytest.fixture(scope="module")
def walked(streams, corpus, cf_proposals, cf_refs):
    return {arm: walk(arm, rows, corpus, cf_proposals, cf_refs) for arm, rows in streams.items()}


@pytest.mark.parametrize("arm", ["H+", "H-", "CF"])
def test_applying_the_diffs_rebuilds_the_files_git_holds(walked, streams, arm):
    found, _ = walked[arm]
    assert found["kind"] == []
    assert found["text"] == []
    # Every diff applied to the earlier text gives the later one exactly, final newline included:
    # none is cut at a form feed or left without a marker (section 8).
    assert found["state"] == []
    assert found["checked"] > 24000
    assert found["diffs"] == sum(1 for row in streams[arm] if row["kind"] in ("modify", "rename"))
    assert found["diffs"] > 24000


def test_hminus_leaks_no_withheld_header_line(walked, streams, corpus):
    found, shown = walked["H-"]
    assert found["leak"] == []
    # The check has something to find: the raw texts do carry such lines.
    with corpus.readers() as readers:
        raw = sum(len(leaked_lines(readers[v.repo].read(v.blob)))
                  for p in list(corpus.proposals.values())[::20]
                  for v in p.versions if v.status != ABSENT)
    assert raw > 500
    # No H- state shows a status at all.
    assert all(value is None for history in shown.values() for _, value in history)
    # And no commit message is touched.
    assert found["message"] == []


def test_cf_differs_from_hplus_only_where_section_7_says(walked, built, streams, cf_proposals,
                                                         cf_refs):
    plus, counterfactual = streams["H+"], streams["CF"]
    assert len(cf_proposals) == 30 and set(cf_refs) == cf_proposals
    found, _ = walked["CF"]
    touched = found["touched"]
    # Every message of the CF stream is the one section 7 gives (checked in the walk).
    assert found["message"] == []
    texts = Counter()
    with open(built["H+"][0], encoding="utf-8", newline="\n") as a, \
            open(built["CF"][0], encoding="utf-8", newline="\n") as b:
        for line_plus, line_cf, row, edited in zip(a, b, plus, counterfactual, strict=True):
            mine = row["proposal"] in cf_proposals
            # The episode id, repository, time, kind and ref never change.
            assert {k: v for k, v in row.items() if k not in ("text", "message")} == {
                k: v for k, v in edited.items() if k not in ("text", "message")
            }
            if row["commit"] not in touched:
                # A commit that changes no CF status line: the message is as in H+, and an
                # episode of a proposal that is not CF is the same line as in H+.
                assert edited["message"] == row["message"], row["episode_id"]
                if not mine:
                    assert line_plus == line_cf, row["episode_id"]
            else:
                # Every episode of such a commit says "Update ...", none keeps its message.
                assert edited["message"].startswith("Update "), row["episode_id"]
                assert edited["message"] != row["message"]
                if not mine:
                    assert edited["text"] == row["text"], row["episode_id"]
            if not mine and edited["text"] != row["text"]:
                texts["non-CF text differs"] += 1
            texts["CF text differs"] += mine and edited["text"] != row["text"]
    assert texts["non-CF text differs"] == 0 and texts["CF text differs"] > 0
    # The numbers section 7 implies: one own episode per status-line change, and every other
    # episode of those commits.
    in_touched = [e for e in counterfactual if e["commit"] in touched]
    assert len(found["rewritten"]) == len(in_touched)
    assert len(found["status_changes"]) + len(found["siblings"]) == len(in_touched)
    assert len(found["status_changes"]) >= len(cf_proposals)
    own = [e for e in in_touched if e["message"].startswith("Update status of ")]
    assert len(own) == len(found["status_changes"])
    assert all(e["message"] == f"Update status of {cf_refs[e['proposal']]}" for e in own)
    assert all(e["message"] == f"Update {e['ref']}" for e in in_touched if e not in own)


STATUS_WORD = re.compile(
    r"\b(draft|active|accept\w*|provisional|defer\w*|reject\w*|withdr\w+|final\w*|supersed\w*"
    r"|review\w*|last[ -]?call|stagnant|living|abandon\w*|replac\w*|approv\w*|postpon\w*"
    r"|complete\w*|incomplete)\b", re.IGNORECASE)


def names_a_cf_proposal(message, cf_proposals):
    numbers = {int(pid.split("-")[1]) for pid in cf_proposals}
    return {n for n in numbers
            if re.search(rf"\b(?:PEP|EIP|ERC)s?[ -]?0*{n}\b", message, re.IGNORECASE)}


def test_no_message_of_a_rewritten_commit_names_a_cf_proposal_with_a_status_word(
    walked, streams, cf_proposals
):
    """Section 7's reason for rewriting a whole commit: after it, no episode of that commit
    carries a message that could name the true status of a CF proposal.

    The property does not hold for the whole stream. Commits that change no CF status line keep
    their messages (section 7), and some of those still name a CF proposal and a status word;
    the report counts them. Here they are only required to lie outside the rewritten commits.
    """
    found, _ = walked["CF"]
    named = [row for row in streams["CF"]
             if STATUS_WORD.search(row["message"])
             and names_a_cf_proposal(row["message"], cf_proposals)]
    assert [row["episode_id"] for row in named if row["commit"] in found["touched"]] == []


def held_statuses(family, message):
    """The family's canonical status values that ``message`` holds as whole words, in any letter
    case. Scanned by hand: the builder uses a regular expression for the same rule."""
    text = " ".join(message.lower().split())
    found = set()
    for status in CANONICAL[family]:
        word = " ".join(status.lower().split())
        at = text.find(word)
        while at != -1:
            end = at + len(word)
            before = text[at - 1] if at else " "
            after = text[end] if end < len(text) else " "
            if not (before.isalnum() or before == "_" or after.isalnum() or after == "_"):
                found.add(status)
                break
            at = text.find(word, at + 1)
    return found


def holds_a_status(family, message):
    return bool(held_statuses(family, message))


def count_messages(rows, cf_proposals, skip, per_episode=False):
    """For each CF proposal, the commits outside ``skip`` (each once, unless ``per_episode``)
    whose message mentions it (the mention rule, in the proposal's own corpus) and holds one of
    that corpus's status values."""
    found = dict.fromkeys(cf_proposals, 0)
    seen = set()
    for row in rows:
        if row["commit"] in skip or (row["commit"] in seen and not per_episode):
            continue
        seen.add(row["commit"])
        for pid in cf_proposals:
            prefix, number = pid.split("-")
            family = "pep" if prefix == "PEP" else "eth"
            if (int(number) in mentions(family, row["message"], -1)
                    and holds_a_status(family, row["message"])):
                found[pid] += 1
    return found


def test_the_cf_message_counts_are_what_the_cf_stream_shows(built, streams, walked,
                                                            cf_proposals):
    """Section 7: for each CF proposal, the commits outside the rewritten ones whose message
    mentions it and holds one of its corpus's status values, each commit counted once. The count
    is recomputed here from the stream's own rows, with the rewritten commits taken from the
    independent walk."""
    with open(f"{built['CF'][0]}.cf_messages.json", encoding="utf-8") as f:
        counts = json.load(f)
    assert counts.pop("canary") == CANARY
    assert set(counts) == cf_proposals and len(counts) == 30
    touched = walked["CF"][0]["touched"]
    assert touched
    assert counts == count_messages(streams["CF"], cf_proposals, touched)
    # The registered numbers of this build: 33 commits in all; PEP 440's four and PEP 8016's one
    # were six and two when each changed file was counted.
    assert sum(counts.values()) == 33 and (counts["PEP-440"], counts["PEP-8016"]) == (4, 1)
    by_episode = count_messages(streams["CF"], cf_proposals, touched, per_episode=True)
    assert sum(by_episode.values()) == 36 and by_episode["PEP-440"] == 6
    # Not vacuous: the rewritten commits are what is left out. Their original messages would have
    # added more (they say what the status became).
    original = count_messages(
        [row for row in streams["H+"] if row["commit"] in touched], cf_proposals, set())
    assert sum(original.values()) > 0
    assert all(isinstance(n, int) and n >= 0 for n in counts.values())
    stats = json.loads(open(f"{built['CF'][0]}.stats.json", encoding="utf-8").read())
    assert set(stats) == {"episodes", "chars", "tokens_est", "sha256"}  # no counts in the stats


def test_the_hminus_message_cues_are_what_the_hminus_stream_shows(built, streams, corpus,
                                                                  questions, gold):
    """Section 12: per proposal, the [recorded_at, status] pairs of the messages the H- stream
    keeps that mention it, read in its own corpus, and hold a canonical status value; recomputed
    here from the H- rows, each commit once, with the status values scanned by hand."""
    with open(f"{built['H-'][0]}.message_cues.json", encoding="utf-8") as f:
        cues = json.load(f)
    assert cues.pop("canary") == CANARY
    pid_of = {(p.family, p.number): pid for pid, p in corpus.proposals.items()}
    expected = defaultdict(set)
    seen = set()
    for row in streams["H-"]:
        if row["commit"] in seen:
            continue
        seen.add(row["commit"])
        for family in ("pep", "eth"):
            statuses = held_statuses(family, row["message"])
            for number in mentions(family, row["message"], -1) if statuses else ():
                pid = pid_of.get((family, number))
                if pid is not None:
                    expected[pid] |= {(row["recorded_at"], status) for status in statuses}
    assert cues == {pid: [list(pair) for pair in sorted(pairs)]
                    for pid, pairs in sorted(expected.items())}
    # Not vacuous: some H- F1 items have a message at or before their date that names the
    # proposal with its gold status, and most do not.
    named = [i for i, q in questions.items() if q["family"] == "F1" and "H-" in q["arms"]
             and any(when <= f"{q['as_of']}T23:59:59Z" and status == gold[i]["gold"]
                     for when, status in cues.get(q["proposal"], []))]
    f1 = sum(1 for q in questions.values() if q["family"] == "F1")
    assert 0 < len(named) < f1


CHANGED_STATUS_LINE = re.compile(r"^[+-][ \t]*status[ \t]*:", re.IGNORECASE | re.MULTILINE)


def test_the_f1_header_only_flag_is_what_the_hminus_and_hplus_streams_show(streams, questions,
                                                                          gold):
    """Section 8 keeps an H- episode whose diff becomes empty, with its message. Every F1 item's
    gold records whether its proposal has one recorded after the end of its date
    (``meta.h_minus_header_only_after_d``); recomputed here from the two built streams alone,
    pairing each H- episode with the H+ episode of the same id: a modify or rename whose H- text
    is empty and whose H+ text is not. The counts at v0.1's pins are pinned too."""
    plus = {row["episode_id"]: row for row in streams["H+"]}
    later = defaultdict(list)
    episodes, status_lines = Counter(), Counter()
    for row in streams["H-"]:
        shown = plus[row["episode_id"]]["text"]
        if row["kind"] in ("modify", "rename") and row["text"] == "" and shown != "":
            later[row["proposal"]].append(row["recorded_at"])
            episodes[(row["corpus"], row["kind"])] += 1
            # Its H- diff is empty, so every line it changes is a withheld header line.
            status_lines[row["corpus"]] += CHANGED_STATUS_LINE.search(shown) is not None
    flagged, items = Counter(), Counter()
    for item_id, q in questions.items():
        if q["family"] != "F1":
            continue
        expected = any(when > f"{q['as_of']}T23:59:59Z" for when in later[q["proposal"]])
        assert gold[item_id]["meta"]["h_minus_header_only_after_d"] is expected, item_id
        items[(q["stratum"], q["corpus"])] += 1
        flagged[(q["stratum"], q["corpus"])] += expected
    assert dict(episodes) == {("pep", "modify"): 563, ("eth", "modify"): 1312}
    assert dict(status_lines) == {"pep": 476, "eth": 1215}
    assert {key: (flagged[key], n) for key, n in items.items()} == {
        ("change", "pep"): (71, 150), ("change", "eth"): (127, 150),
        ("current", "pep"): (0, 50), ("current", "eth"): (0, 50),
        ("recent", "pep"): (1, 43), ("recent", "eth"): (14, 60)}


def test_the_scorer_reads_the_cf_stream_companion_file(built, data, questions, gold, tmp_path):
    ids = sorted(i for i, q in questions.items() if q["family"] == "CF")
    rows = [{"id": i, "system": "dense", "arm": "CF", "run": run, "answer": gold[i]["gold"]}
            for i in ids for run in (1, 2, 3)]
    answers = tmp_path / "answers.jsonl"
    answers.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    listed = tmp_path / "cf.json"
    listed.write_text(json.dumps(ids), encoding="utf-8")
    out = tmp_path / "out"
    argv = ["score", "--data", str(data), "--answers", str(answers), "--out", str(out),
            "--items", str(listed), "--cf-messages", f"{built['CF'][0]}.cf_messages.json",
            "--answering-model", "claude-haiku-5-5", "--answering-settings",
            "effort medium, adaptive thinking, default temperature", "--construction-model",
            "none", "--embedding-model", "test-embedding-model"]  # dense embeds text
    assert cli.main(argv) == 0
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    with open(f"{built['CF'][0]}.cf_messages.json", encoding="utf-8") as f:
        counts = json.load(f)
    items = {r["id"]: r for r in results["cf_items"]}
    assert sorted(items) == ids and len(ids) == 30
    assert {i: r["cf_messages"] for i, r in items.items()} == {
        i: counts[questions[i]["proposal"]] for i in ids}
    assert all(r["systems"]["dense"] == {"accuracy_mean_over_runs": 1.0, "majority_correct": True}
               for r in items.values())
    # This build was made without --streams, so its manifest registers no file to check against.
    assert results["cf_messages"] == {"file": "cf.jsonl.cf_messages.json",
                                      "total": sum(v for k, v in counts.items() if k != "canary"),
                                      "checked_against_manifest": False}
    assert all(c["answered"] == c["asked"] for c in results["coverage"] if c["system"] == "dense")
    text = (out / "results.md").read_text(encoding="utf-8")
    assert "## Counterfactual items" in text


def test_the_cf_state_as_of_each_question_date_says_the_gold(walked, questions, gold):
    _, shown = walked["CF"]
    checked = 0
    for item_id, q in questions.items():
        if q["family"] != "CF":
            continue
        until = f"{q['as_of']}T23:59:59Z"
        value = [v for when, v in shown[q["proposal"]] if when <= until][-1]
        assert normalize_status(q["corpus"], value) == gold[item_id]["gold"], item_id
        # The edit really is an edit: the true status is another one.
        assert gold[item_id]["gold"] != gold[item_id]["meta"]["true_status"]
        checked += 1
    assert checked == 30


def test_the_hplus_state_as_of_each_f1_date_says_the_gold(walked, questions, gold):
    _, shown = walked["H+"]
    checked = 0
    for item_id, q in questions.items():
        if q["family"] != "F1":
            continue
        until = f"{q['as_of']}T23:59:59Z"
        value = [v for when, v in shown[q["proposal"]] if when <= until][-1]
        assert normalize_status(q["corpus"], value) == gold[item_id]["gold"], item_id
        checked += 1
    assert checked > 450
    assert date.fromisoformat(max(q["as_of"] for q in questions.values())) <= S
    assert end_of_day_ts(S) > 0


def test_the_materialised_trees_are_the_files_as_they_stood(built, streams, corpus, questions,
                                                           gold, clones, tmp_path):
    """Section 11: grep at D reads its arm's stream materialised as of the end of D.

    For 24 F1 items, the first four in frozen order of each stratum and corpus, the tree as of the
    item's date holds exactly the proposals present then, each as its version's text (H+) or that
    text with the withheld fields removed (H-); the item's own file is also checked against
    ``git show`` of its source at the as-of commit. The log written beside the tree lists the
    commits of the H- episodes recorded by the end of the date, newest first, with the messages
    the H- stream keeps.
    """
    from asofbench.headers import strip_withheld
    from asofbench.materialize import file_name, replay

    groups = defaultdict(list)
    for item_id, q in questions.items():
        if q["family"] == "F1":
            groups[(q["stratum"], q["corpus"])].append(item_id)
    sample = [i for key in sorted(groups)
              for i in sorted(groups[key], key=lambda i: gold[i]["meta"]["rank"])[:4]]
    assert len(sample) == 24
    days = sorted({date.fromisoformat(questions[i]["as_of"]) for i in sample})
    trees = {}
    for arm in ("H+", "H-"):
        with open(built[arm][0], encoding="utf-8", newline="\n") as f:
            trees[arm] = dict(replay(f, days))
    with corpus.readers() as readers:
        for day in days:
            moment = end_of_day_ts(day)
            plus = {}
            for pid, p in corpus.proposals.items():
                v = p.version_at(moment)
                if v is not None and v.status != ABSENT:
                    plus[(v.repo, pid)] = readers[v.repo].read(v.blob)
            assert trees["H+"][day] == plus, day
            assert trees["H-"][day] == {
                key: strip_withheld(corpus.proposals[key[1]].family, text)
                for key, text in plus.items()}, day
    for item_id in sample:
        q, source = questions[item_id], gold[item_id]["source"]
        day = date.fromisoformat(q["as_of"])
        shown = git(clones / source["repo"], "show", "--no-textconv",
                    f"{source['commit']}:{source['path']}").decode("utf-8", errors="replace")
        key = (source["repo"], q["proposal"])
        assert trees["H+"][day][key] == shown, item_id
        assert trees["H-"][day][key] == strip_withheld(q["corpus"], shown), item_id
        assert leaked_lines(trees["H-"][day][key]) == [], item_id
    # The command writes the same tree to disk, under the names the episodes give.
    day = days[len(days) // 2]
    out = tmp_path / "tree"
    argv = ["materialize", "--stream", str(built["H-"][0]), "--as-of", day.isoformat(),
            "--out", str(out)]
    assert cli.main(argv) == 0
    on_disk = {p.relative_to(out).as_posix(): p.read_bytes().decode("utf-8")
               for p in out.rglob("*") if p.is_file()}
    logged = on_disk.pop("git-log.txt")
    assert on_disk == {f"{repo}/{file_name(repo, pid)}": text
                       for (repo, pid), text in trees["H-"][day].items()}
    # The log: each commit once, newest first, its files, and its message indented four spaces.
    commits = {}
    for row in streams["H-"]:
        if row["recorded_at"] <= f"{day.isoformat()}T23:59:59Z":
            when, message, paths = commits.setdefault(
                row["commit"], (row["recorded_at"], row["message"], []))
            assert row["message"] == message  # H- keeps one message per commit
            paths.append(f"{row['repo']}/{file_name(row['repo'], row['proposal'])}")
    expected = "".join(
        f"commit {commit}\nDate: {when}\n" + "".join(f"File: {path}\n" for path in paths) + "\n"
        + "".join(f"    {line}\n" for line in message.split("\n")) + "\n"
        for commit, (when, message, paths) in reversed(commits.items()))
    assert logged == expected and len(commits) > 1000


def test_the_stream_is_deterministic(clones, data, built, tmp_path):
    again = tmp_path / "again.jsonl"
    argv = ["stream", "--clones", str(clones), "--data", str(data), "--arm", "H-", "--out",
            str(again)]
    assert cli.main(argv) == 0
    assert again.read_bytes() == built["H-"][0].read_bytes()
