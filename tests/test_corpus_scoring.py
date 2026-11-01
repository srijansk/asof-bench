"""The scorer on the real build (corpus-marked; set ASOFBENCH_CLONES to run them).

The questions and gold are built from the clones once for the module. Answers are made up from the
gold: a system that answers every item right (in spellings the normaliser must read) and one that
answers every item wrong, and six pilot systems at chosen accuracies. The scorer must give
exactly 1 and 0, and each pilot scenario must stop at the gate it was made for. Entries are
checked here too: long context is scored on F1, F2 and CF only, and a system that is entered in an
arm it never answered scores zero there with its rows present.
"""

from __future__ import annotations

import json

import pytest

from asofbench import cli
from asofbench.scoring import PILOT_SYSTEMS, wilson
from asofbench.statuses import outcome_class

pytestmark = pytest.mark.corpus


def read_jsonl(path):
    with open(path, encoding="utf-8", newline="\n") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture(scope="module")
def data(clones, tmp_path_factory):
    out = tmp_path_factory.mktemp("asofbench") / "v0.1"
    assert cli.main(["build", "--clones", str(clones), "--out", str(out)]) == 0
    return out


@pytest.fixture(scope="module")
def questions(data):
    return read_jsonl(data / "questions.jsonl")


@pytest.fixture(scope="module")
def gold(data):
    return {row["id"]: row for row in read_jsonl(data / "gold.jsonl")}


def score(data, answers, out, *extra):
    argv = ["score", "--data", str(data), "--answers", str(answers), "--out", str(out),
            "--answering-model", "claude-haiku-5-5", "--answering-settings",
            "effort medium, adaptive thinking, default temperature", "--construction-model",
            "claude-haiku-5-5", "--embedding-model",
            "voyageai/voyage-4-nano@5fb34e870f49f19e79c94d5e118f5ee730133279", *extra]
    assert cli.main(argv) == 0
    return json.loads((out / "results.json").read_text(encoding="utf-8"))


def write_json(path, content):
    path.write_text(json.dumps(content, sort_keys=True), encoding="utf-8")
    return path


def write_rows(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    return path


def right_answer(q, g, k):
    """The gold, in a spelling that varies with ``k``."""
    if q["answer_type"] == "status":
        text = g["gold"].lower() + "."
        return f"{chr(0x201C)}{text}{chr(0x201D)}" if k % 4 == 0 else text
    if q["answer_type"] == "set":
        prefix = "PEP " if q["corpus"] == "pep" else "EIP-"
        members = [f"{prefix}{n}" if (k + i) % 2 else n for i, n in enumerate(g["gold"])]
        return members + ([g["gold"][0]] if g["gold"] and k % 3 == 0 else [])
    return g["gold"]


def wrong_answer(q, g, k):
    if q["answer_type"] == "status":
        family = q["corpus"]
        other = [c for c in q["choices"]
                 if outcome_class(family, c) != outcome_class(family, g["gold"])]
        return (other or [c for c in q["choices"] if c != g["gold"]])[0]
    if q["answer_type"] == "set":
        return [10**6 + k]
    return g["gold"] + 1000


def rows_for(system, make, questions, gold):
    return [
        {"id": q["id"], "system": system, "arm": arm, "run": run,
         "answer": make(q, gold[q["id"]], k), "tokens_in": 1000, "tokens_out": 50,
         "wall_s": 2.5, "tool_calls": int(k % 3 == 0)}
        for k, q in enumerate(questions) for arm in q["arms"] for run in (1, 2, 3)
    ]


def test_a_perfect_system_scores_one_and_a_wrong_one_scores_zero(data, questions, gold, tmp_path):
    rows = rows_for("oracle", right_answer, questions, gold)
    rows += rows_for("dense", wrong_answer, questions, gold)
    results = score(data, write_rows(tmp_path / "answers.jsonl", rows), tmp_path / "out")
    assert results["unknown_ids"] == []
    assert results["answers"]["scored"] == len(rows) == results["answers"]["rows"]
    assert results["answers"]["missing_runs"] == [] and results["answers"]["odd_runs"] == []
    assert results["answers"]["arms_without_answers"] == []
    assert results["answers"]["out_of_scope_count"] == 0
    groups = {(arm, q["family"], q["stratum"], q["corpus"]) for q in questions for arm in q["arms"]}
    for system, want in (("oracle", 1.0), ("dense", 0.0)):
        mine = [r for r in results["results"] if r["system"] == system]
        assert {(r["arm"], r["family"], r["stratum"], r["corpus"]) for r in mine} == groups
        for r in mine:
            assert r["runs"] == 3
            assert (r["accuracy_mean_over_runs"], r["majority_accuracy"], r["pass_at_3"],
                    r["all_3"]) == (want,) * 4, r
            low, high = wilson(round(want * r["n_items"]), r["n_items"])
            assert (r["wilson_low"], r["wilson_high"]) == (low, high)
    # Every item the arm asks was answered, and every family and arm is one they are entered in.
    assert all(c["answered"] == c["asked"] for c in results["coverage"])
    for system in ("oracle", "dense"):
        assert results["scope"]["entries"][system] == {
            "arms": ["CF", "H+", "H-"], "families": sorted({q["family"] for q in questions})}
    # Secondary scores agree: right on the outcome class, F1 and the count, and wrong on all.
    for r in results["secondary"]:
        want = 1.0 if r["system"] == "oracle" else 0.0
        assert r["invalid_rate"] == 0.0
        for key in ("outcome_accuracy", "f1", "within_one_rate"):
            assert r[key] in (None, want), (key, r)
    # Decision-word CF items are scored apart: the build report's 25 of 30.
    flagged = {(r["corpus"], r["decision_word"]): r["n_items"]
               for r in results["cf_split"] if r["system"] == "oracle"}
    assert sum(n for (_, flag), n in flagged.items() if flag) == sum(
        1 for g in gold.values() if g["meta"].get("decision_word"))
    # So are the CF items with a header cue: 3 Ethereum and 1 PEP item at the pins.
    cued = {(r["corpus"], r["header_cue"]): r["n_items"]
            for r in results["cf_header_split"] if r["system"] == "oracle"}
    assert cued == {("eth", True): 3, ("eth", False): 12, ("pep", True): 1, ("pep", False): 14}
    # F2 over all items and over those the asked PEP's own episodes can show: the 41 negatives
    # and 32 positives in H+, the 41 negatives and 15 positives in H-.
    f2 = {r["arm"]: r for r in results["f2_split"] if r["system"] == "oracle"}
    assert {arm: (r["items"], r["recoverable_items"]) for arm, r in f2.items()} == {
        "H+": (82, 73), "H-": (82, 56)}
    assert all(r["accuracy"] == r["recoverable_accuracy"] == 1.0 for r in f2.values())


SCENARIOS = {
    # (accuracy on F1-change, accuracy on F1-current) per system, and the gate that should decide
    "ceiling": ({s: (0.92, 0.95) for s in PILOT_SYSTEMS}, 1, False),
    "loading": ({"closed-book": (0.30, 0.30), "bm25": (0.50, 0.50), "grep-at-d": (0.60, 0.90),
                 "long-context": (0.70, 0.90), "hipporag2": (0.50, 0.55),
                 "graphiti": (0.55, 0.50)}, 2, False),
    "signal": ({"closed-book": (0.30, 0.30), "bm25": (0.55, 0.90), "grep-at-d": (0.70, 0.90),
                "long-context": (0.80, 0.90), "hipporag2": (0.60, 0.90),
                "graphiti": (0.93, 0.95)}, None, True),
    "null": ({"closed-book": (0.50, 0.90), "bm25": (0.52, 0.90), "grep-at-d": (0.55, 0.90),
              "long-context": (0.58, 0.90), "hipporag2": (0.54, 0.90),
              "graphiti": (0.60, 0.90)}, None, False),
}


PILOT_ENTRIES = {system: {"arms": ["H-"], "families": ["F1"]} for system in PILOT_SYSTEMS}


def pilot_rows(accuracy, data, questions, gold):
    """H- answers, three runs, on the pilot's F1 items, at the accuracies given per system."""
    pilot = json.loads((data / "pilot.json").read_text(encoding="utf-8"))
    ids = {s: [i for g in pilot["groups"] if g["family"] == "F1" and g["stratum"] == s
               for i in g["ids"]] for s in ("change", "current")}
    assert (len(ids["change"]), len(ids["current"])) == (100, 50)  # section 13's pilot
    by_id = {q["id"]: q for q in questions}
    rows = []
    for system, (change, current) in accuracy.items():
        for stratum, share in (("change", change), ("current", current)):
            for k, item_id in enumerate(ids[stratum]):
                right = (k * 37 + len(system)) % 100 < share * 100
                wrong = next(c for c in by_id[item_id]["choices"] if c != gold[item_id]["gold"])
                for run in (1, 2, 3):
                    rows.append({"id": item_id, "system": system, "arm": "H-", "run": run,
                                 "answer": gold[item_id]["gold"] if right else wrong})
    return rows, ids


def read_gates(out):
    return json.loads((out / "gates.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_each_pilot_scenario_stops_at_its_gate(name, data, questions, gold, tmp_path):
    accuracy, stopped, signal = SCENARIOS[name]
    rows, _ = pilot_rows(accuracy, data, questions, gold)
    answers = write_rows(tmp_path / "answers.jsonl", rows)
    entries = write_json(tmp_path / "entries.json", PILOT_ENTRIES)
    out = tmp_path / "out"
    results = score(data, answers, out, "--items", str(data / "pilot.json"),
                    "--entries", str(entries))
    assert results["pilot"]["evaluable"] is True and results["pilot"]["missing_systems"] == []
    assert results["pilot"]["coverage"]["complete"] is True
    gates = read_gates(out)
    assert gates["evaluable"] is True
    assert (gates["stopped_at"], gates["signal"]) == (stopped, signal)
    assert gates["detail"]["by_majority_accuracy"]["stopped_at"] == stopped
    assert gates["coverage"]["complete"] is True and gates["coverage"]["pilot_items"] == 150
    # A pilot run is scored on the pilot list: 50 and 25 items per corpus, all of them answered.
    sizes = {(r["stratum"], r["corpus"]): r["n_items"] for r in results["results"]
             if r["system"] == "graphiti" and r["arm"] == "H-"}
    assert sizes == {("change", "pep"): 50, ("change", "eth"): 50, ("current", "pep"): 25,
                     ("current", "eth"): 25}
    assert {(r["arm"], r["family"]) for r in results["results"]} == {("H-", "F1")}
    assert all(c["answered"] == c["asked"] for c in results["coverage"])
    assert results["answers"]["arms_without_answers"] == []
    assert results["scope"]["items"] == 180 and results["scope"]["items_file"] == "pilot.json"
    assert results["scope"]["entries_file"] == "entries.json"
    # The same answers against the whole benchmark: the items they left out are wrong, so every
    # score is lower, and the gates, which read the pilot items, do not move.
    whole = tmp_path / "oracle"
    everything = score(data, answers, whole)
    full = {(r["system"], r["stratum"], r["corpus"]): r for r in everything["results"]
            if r["arm"] == "H-" and r["family"] == "F1"}
    mine = {(r["system"], r["stratum"], r["corpus"]): r for r in results["results"]
            if r["arm"] == "H-"}
    for key, small in mine.items():
        assert full[key]["n_items"] > small["n_items"]
        assert full[key]["accuracy_mean_over_runs"] < small["accuracy_mean_over_runs"] + 1e-12
    assert read_gates(whole) == gates
    # With the pilot list but the default entries, the six systems are also entered in the
    # arms they never answered (H+ and CF): zero there, rows present, and the arms are named.
    wide = tmp_path / "wide"
    listed = score(data, answers, wide, "--items", str(data / "pilot.json"))
    assert {(r["arm"], r["family"]) for r in listed["results"]} == {
        ("H-", "F1"), ("H+", "F1"), ("CF", "CF")}
    assert all(r["accuracy_mean_over_runs"] == 0.0 for r in listed["results"]
               if r["arm"] != "H-")
    assert listed["answers"]["arms_without_answers"] == [
        {"system": system, "arms": ["CF", "H+"]} for system in sorted(PILOT_SYSTEMS)]
    assert read_gates(wide) == gates


def test_a_pilot_missing_a_system_is_not_evaluable(data, questions, gold, tmp_path):
    accuracy = dict(SCENARIOS["signal"][0])
    del accuracy["graphiti"]
    rows, _ = pilot_rows(accuracy, data, questions, gold)
    out = tmp_path / "out"
    results = score(data, write_rows(tmp_path / "answers.jsonl", rows), out,
                    "--items", str(data / "pilot.json"))
    assert results["pilot"]["evaluable"] is False
    assert results["pilot"]["missing_systems"] == ["graphiti"]
    gates = read_gates(out)
    assert gates["evaluable"] is False
    assert "stopped_at" not in gates and "signal" not in gates
    assert gates["missing_systems"] == ["graphiti"]
    assert gates["decision"].startswith("not evaluable") and "graphiti" in gates["decision"]
    assert "graphiti" in (out / "results.md").read_text(encoding="utf-8")


def test_a_missing_pilot_answer_is_wrong_and_the_gates_are_still_decided(
    data, questions, gold, tmp_path
):
    accuracy = SCENARIOS["signal"][0]
    rows, ids = pilot_rows(accuracy, data, questions, gold)
    dropped = set(ids["change"][:10])  # ten items graphiti never answers, in any run
    thin = [r for r in rows if not (r["system"] == "graphiti" and r["id"] in dropped)]
    entries = write_json(tmp_path / "entries.json", PILOT_ENTRIES)
    argv = ("--items", str(data / "pilot.json"), "--entries", str(entries))
    full_out, thin_out = tmp_path / "full", tmp_path / "thin"
    score(data, write_rows(tmp_path / "full.jsonl", rows), full_out, *argv)
    results = score(data, write_rows(tmp_path / "thin.jsonl", thin), thin_out, *argv)
    gates = read_gates(thin_out)
    assert gates["evaluable"] is True and "stopped_at" in gates and "signal" in gates
    # The ten missing items are wrong, so graphiti scores lower than on the full answers.
    full_change = read_gates(full_out)["detail"]["accuracy_change"]["graphiti"]
    assert gates["detail"]["accuracy_change"]["graphiti"] < full_change
    # The coverage says what is missing, and the report names it.
    coverage = gates["coverage"]
    assert coverage["complete"] is False
    assert coverage["systems"]["graphiti"]["missing_count"] == 10
    assert set(coverage["systems"]["graphiti"]["missing"]) == dropped
    assert all(v["missing_count"] == 0 for k, v in coverage["systems"].items() if k != "graphiti")
    assert results["pilot"]["coverage"] == coverage and results["pilot"]["evaluable"] is True
    text = (thin_out / "results.md").read_text(encoding="utf-8")
    assert "graphiti has no answer to 10 of 150 items" in text


def test_leaving_items_out_lowers_the_score_on_the_real_build(data, questions, gold, tmp_path):
    rows = rows_for("oracle", right_answer, questions, gold)
    kept = {q["id"] for q in questions if q["family"] == "F1" and q["stratum"] == "change"
            and q["corpus"] == "pep"}
    dropped = sorted(kept)[::2]  # every other F1-change PEP item is left out
    skipping = [dict(r, system="hybrid") for r in rows if r["id"] not in set(dropped)]
    results = score(data, write_rows(tmp_path / "answers.jsonl", rows + skipping),
                    tmp_path / "out")
    group = {(r["system"], r["arm"]): r for r in results["results"]
             if (r["family"], r["stratum"], r["corpus"]) == ("F1", "change", "pep")}
    for arm in ("H+", "H-"):
        assert group[("oracle", arm)]["accuracy_mean_over_runs"] == 1.0
        skips = group[("hybrid", arm)]
        assert skips["n_items"] == 150  # the left-out items are in the denominator
        assert skips["accuracy_mean_over_runs"] == (150 - len(dropped)) / 150
    coverage = {(c["system"], c["arm"]): c for c in results["coverage"]
                if (c["family"], c["stratum"], c["corpus"]) == ("F1", "change", "pep")}
    assert coverage[("hybrid", "H+")]["answered"] == 150 - len(dropped)
    assert coverage[("hybrid", "H+")]["asked"] == 150
    # Entered in F1 only (by an entries file), it is scored on the F1 items alone.
    entries = write_json(tmp_path / "entries.json", {"oracle": {"families": ["F1"]}})
    only = score(data, write_rows(tmp_path / "f1.jsonl", [r for r in rows if r["id"].startswith(
        "F1-")]), tmp_path / "f1", "--entries", str(entries))
    assert {r["family"] for r in only["results"]} == {"F1"}
    assert all(r["accuracy_mean_over_runs"] == 1.0 for r in only["results"])
    assert only["scope"]["entries"]["oracle"] == {"arms": ["CF", "H+", "H-"], "families": ["F1"]}


def test_the_pilot_is_not_evaluable_without_all_six_systems(data, questions, gold, tmp_path):
    rows = rows_for("closed-book", right_answer, questions, gold)
    out = tmp_path / "out"
    results = score(data, write_rows(tmp_path / "answers.jsonl", rows), out)
    assert results["pilot"]["evaluable"] is False
    assert results["pilot"]["missing_systems"] == list(PILOT_SYSTEMS[1:])
    gates = read_gates(out)
    assert gates["evaluable"] is False and gates["missing_systems"] == list(PILOT_SYSTEMS[1:])
    assert "stopped_at" not in gates and "signal" not in gates
    # A closed book that is right on everything is far above both corpora's F1-change floors
    # (Draft, 91 of 150 PEP and 98 of 150 Ethereum items), so both are marked.
    flag = results["contamination"]
    assert set(flag) == {"pep", "eth"} and all(f["possibly_contaminated"] for f in flag.values())
    assert {c: (f["floor_gold"], f["floor_items"], f["items"]) for c, f in flag.items()} == {
        "pep": ("Draft", 91, 150), "eth": ("Draft", 98, 150)}
    assert all(f["closed_book_right"] == 150 for f in flag.values())


def test_a_combined_run_scores_long_context_only_where_it_is_entered(
    data, questions, gold, tmp_path
):
    """Long context is entered in F1, F2 and CF (section 11). It answers every item here, as the
    other system does; its answers outside those families are out of scope, so there are no F3,
    F4 or F5 rows for it. It never answers the CF arm, so that arm scores zero with rows present.
    """
    everyone = rows_for("oracle", right_answer, questions, gold)
    long_context = [r for r in rows_for("long-context", right_answer, questions, gold)
                    if r["arm"] != "CF"]
    out = tmp_path / "out"
    results = score(data, write_rows(tmp_path / "answers.jsonl", everyone + long_context), out)
    mine = [r for r in results["results"] if r["system"] == "long-context"]
    assert {r["family"] for r in mine} == {"F1", "F2", "CF"}
    assert results["scope"]["entries"]["long-context"] == {
        "arms": ["CF", "H+", "H-"], "families": ["CF", "F1", "F2"]}
    assert results["scope"]["entries"]["oracle"]["families"] == sorted(
        {q["family"] for q in questions})
    # The answers it gave outside its families were not scored.
    outside = [r for r in long_context if r["id"].split("-")[0] in ("F3", "F4", "F5")]
    assert results["answers"]["out_of_scope_count"] == len(outside) > 0
    # In its arms it is right on everything it was asked.
    for r in mine:
        if r["arm"] == "CF":
            assert (r["n_items"], r["accuracy_mean_over_runs"], r["runs"]) == (15, 0.0, 3)
        else:
            assert r["accuracy_mean_over_runs"] == 1.0, r
    assert {r["corpus"] for r in mine if r["arm"] == "CF"} == {"pep", "eth"}
    assert results["answers"]["arms_without_answers"] == [
        {"system": "long-context", "arms": ["CF"]}]
    # The other system is scored on everything.
    assert {r["family"] for r in results["results"] if r["system"] == "oracle"} == {
        q["family"] for q in questions}
    cf_rows = [c for c in results["coverage"] if (c["system"], c["arm"]) == ("long-context", "CF")]
    assert cf_rows and all(c["answered"] == 0 and c["asked"] == 15 for c in cf_rows)
