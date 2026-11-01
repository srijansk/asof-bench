"""The scorer: answers, per-item scores, intervals, aggregation and the pilot gates (12, 13).

The first group of tests is the unit level (``wilson``, ``mcnemar_exact``, ``score_answer``,
``aggregate``, ``secondary`` and ``pilot_gates`` on hand-made rows). The second runs
``asofbench score`` on small question, gold and answer files written by the test.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from datetime import date
from fractions import Fraction
from pathlib import Path

import pytest

from asofbench import cli
from asofbench.corpora import CANARY
from asofbench.questions import dumps
from asofbench.scoring import (
    aggregate,
    mcnemar_exact,
    pilot_gates,
    score_answer,
    secondary,
    wilson,
)

# --- wilson and mcnemar --------------------------------------------------------------------------


def test_wilson_known_value():
    lo, hi = wilson(8, 10)
    assert round(lo, 4) == 0.4902 and round(hi, 4) == 0.9433


def test_wilson_edges():
    assert wilson(0, 0) == (0.0, 1.0)
    low, high = wilson(0, 10)
    assert low == 0.0 and round(high, 4) == 0.2775
    low, high = wilson(10, 10)
    assert high == 1.0 and round(low, 4) == 0.7225
    low, high = wilson(5, 10)
    assert (round(low, 4), round(high, 4)) == (0.2366, 0.7634)
    # The interval is symmetric under swapping successes and failures.
    for k in range(0, 31):
        lo, hi = wilson(k, 30)
        mirror_lo, mirror_hi = wilson(30 - k, 30)
        assert lo == pytest.approx(1 - mirror_hi, abs=1e-12)
        assert hi == pytest.approx(1 - mirror_lo, abs=1e-12)
        assert 0.0 <= lo <= k / 30 <= hi <= 1.0
    # A custom z is honoured: z = 0 collapses the interval to the proportion.
    assert wilson(3, 12, z=0.0) == (0.25, 0.25)
    with pytest.raises(ValueError):
        wilson(11, 10)
    with pytest.raises(ValueError):
        wilson(-1, 10)


def test_mcnemar():
    assert mcnemar_exact(0, 10) == 2 * 0.5**10
    assert mcnemar_exact(5, 5) == 1.0 and mcnemar_exact(0, 0) == 1.0


def test_mcnemar_exact_values():
    assert mcnemar_exact(10, 0) == mcnemar_exact(0, 10)
    assert mcnemar_exact(1, 9) == 2 * 11 / 1024  # n = 10: 1 + 10 outcomes at or below 1
    assert mcnemar_exact(3, 9) == 598 / 4096  # n = 12: 2 * (1 + 12 + 66 + 220) / 2**12
    assert mcnemar_exact(4, 5) == 1.0  # the two-sided p is capped at 1
    assert mcnemar_exact(0, 1) == 1.0
    assert mcnemar_exact(0, 4) == 0.125
    assert mcnemar_exact(0, 5) == 0.0625 and mcnemar_exact(0, 6) == 0.03125  # the 5% boundary
    assert 0.0 <= mcnemar_exact(0, 5000) < 1e-300  # no overflow for a long list of items
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 3)


# --- answers -------------------------------------------------------------------------------------


def item(answer_type: str, corpus: str = "pep") -> dict:
    return {"id": "x", "answer_type": answer_type, "corpus": corpus}


status_item, set_item, count_item = item("status"), item("set"), item("count")


def test_bool_and_float_answers_score_zero():
    assert not score_answer(count_item, {"gold": 12}, True)["correct"]
    assert not score_answer(count_item, {"gold": 12}, 12.0)["correct"]
    assert not score_answer(set_item, {"gold": [20]}, [True])["correct"]
    assert not score_answer(set_item, {"gold": [20]}, [[20]])["correct"]


def test_a_boolean_is_not_the_integer_it_equals():
    # True == 1 in Python, so these are the answers that a careless comparison would accept.
    assert not score_answer(count_item, {"gold": 1}, True)["correct"]
    assert not score_answer(count_item, {"gold": 0}, False)["correct"]
    assert not score_answer(set_item, {"gold": [1]}, [True])["correct"]
    assert not score_answer(set_item, {"gold": [0]}, [False])["correct"]
    assert not score_answer(set_item, {"gold": [1]}, [1.0])["correct"]
    for answer in (True, 12.0, "12", None, [12], {"n": 12}):
        scored = score_answer(count_item, {"gold": 12}, answer)
        assert (scored["correct"], scored["valid"]) == (False, False), answer
        assert scored["within_one"] is False and scored["abs_error"] is None


@pytest.mark.parametrize("written", ["20.0", "2e1", "2E1", "2e+1", "20e0", "200e-1", "20.50"])
def test_numbers_written_with_a_fraction_or_an_exponent_are_invalid(written):
    # Section 12: "numbers written with a fraction or an exponent (20.0 included)". Parsed from
    # JSON text, as an answer file is, each of these is a float.
    as_set = score_answer(set_item, {"gold": [20]}, json.loads(f"[{written}]"))
    assert (as_set["valid"], as_set["correct"]) == (False, False), written
    assert (as_set["precision"], as_set["recall"], as_set["f1"]) == (0, 0, 0)
    as_count = score_answer(count_item, {"gold": 20}, json.loads(written))
    assert (as_count["valid"], as_count["correct"]) == (False, False), written
    assert as_count["abs_error"] is None and as_count["within_one"] is False
    # The same number written as an integer is fine.
    assert score_answer(set_item, {"gold": [20]}, json.loads("[20]"))["correct"]
    assert score_answer(count_item, {"gold": 20}, json.loads("20"))["correct"]


def test_status_answers():
    gold = {"gold": "Final", "gold_outcome": "adopted"}
    right = score_answer(status_item, gold, "final.")
    assert (right["correct"], right["outcome_correct"], right["valid"]) == (True, True, True)
    other = score_answer(status_item, gold, "Accepted")  # both are adopted
    assert (other["correct"], other["outcome_correct"], other["valid"]) == (False, True, True)
    declined = score_answer(status_item, gold, "Rejected")
    assert (declined["correct"], declined["outcome_correct"]) == (False, False)
    bad = score_answer(status_item, gold, 3)
    assert (bad["correct"], bad["valid"], bad["outcome_correct"]) == (False, False, False)
    # Set and count scores do not apply to a status.
    assert right["precision"] is right["recall"] is right["f1"] is None
    assert right["abs_error"] is None and right["within_one"] is None


QUOTED = [
    f"{chr(0x201C)}Final{chr(0x201D)}",  # typographic double quotes
    f"{chr(0x2018)}Final{chr(0x2019)}.",  # typographic single quotes, then a full stop
]


@pytest.mark.parametrize(
    "answer",
    ["final", "FINAL", "Final.", "  Final  ", "'Final'", '"Final".', *QUOTED, "'Final", "Final'"],
)
def test_status_answers_are_normalised(answer):
    assert score_answer(status_item, {"gold": "Final"}, answer)["correct"], answer


def test_status_outcome_comes_from_the_table_when_gold_has_no_outcome():
    scored = score_answer(status_item, {"gold": "Final"}, "Accepted")
    assert scored["outcome_correct"] is True
    scored = score_answer(status_item, {"gold": "Withdrawn"}, "Rejected")
    assert (scored["correct"], scored["outcome_correct"]) == (False, True)  # both declined


def test_status_legacy_names_and_ethereum_values():
    assert score_answer(status_item, {"gold": "Accepted"}, "Approved")["correct"]
    assert score_answer(status_item, {"gold": "Deferred"}, "postponed")["correct"]
    assert score_answer(status_item, {"gold": "Draft"}, "Incomplete")["correct"]
    eth = item("status", "eth")
    for spelling in ("last call", "Last-Call", "lastcall", "LAST  CALL."):
        assert score_answer(eth, {"gold": "Last Call"}, spelling)["correct"], spelling
    # The legacy table is the PEP one: Approved is no Ethereum status.
    assert not score_answer(eth, {"gold": "Accepted"}, "Approved")["correct"]
    assert score_answer(eth, {"gold": "Replaced"}, "Replaced")["correct"]
    # A string that names no status is wrong but not mistyped.
    named_nothing = score_answer(status_item, {"gold": "Final"}, "Moved")
    assert (named_nothing["correct"], named_nothing["outcome_correct"], named_nothing["valid"]) == (
        False, False, True
    )
    assert not score_answer(status_item, {"gold": "Final"}, "It was Final")["correct"]


def test_counterfactual_status_outcome_is_the_permuted_one():
    cf = {"gold": "Rejected", "gold_outcome": "declined"}  # the true status was Accepted
    assert score_answer(status_item, cf, "rejected")["correct"]
    true_answer = score_answer(status_item, cf, "Accepted")
    assert (true_answer["correct"], true_answer["outcome_correct"]) == (False, False)


def test_set_answers():
    gold = {"gold": [20, 165]}
    full = score_answer(set_item, gold, ["EIP-20", 165, 165])
    assert full["correct"] and full["valid"]
    assert (full["precision"], full["recall"], full["f1"]) == (1, 1, 1)
    part = score_answer(set_item, gold, [20])
    assert part["correct"] is False
    assert part["precision"] == 1 and part["recall"] == 0.5
    assert part["f1"] == pytest.approx(2 / 3)
    extra = score_answer(set_item, gold, [20, 165, 721, 1155])
    assert (extra["precision"], extra["recall"]) == (0.5, 1)
    assert extra["f1"] == pytest.approx(2 / 3)
    wrong = score_answer(set_item, gold, [1, 2])
    assert (wrong["precision"], wrong["recall"], wrong["f1"]) == (0, 0, 0)
    # An empty gold answered with an empty list scores 1; anything else scores 0.
    empty = score_answer(set_item, {"gold": []}, [])
    assert empty["correct"] and (empty["precision"], empty["recall"], empty["f1"]) == (1, 1, 1)
    spurious = score_answer(set_item, {"gold": []}, [5])
    assert not spurious["correct"]
    assert (spurious["precision"], spurious["recall"], spurious["f1"]) == (0, 0, 0)
    # A gold with members answered with an empty list: nothing right, F1 0.
    silent = score_answer(set_item, gold, [])
    assert not silent["correct"] and silent["valid"]
    assert (silent["precision"], silent["recall"], silent["f1"]) == (0, 0, 0)
    # The order does not matter.
    assert score_answer(set_item, gold, [165, 20])["correct"]
    # Status and count scores do not apply to a set.
    assert full["outcome_correct"] is None and full["abs_error"] is None
    assert full["within_one"] is None


@pytest.mark.parametrize(
    ("answer", "members"),
    [
        ([20, 165], {20, 165}),
        (["20", "165"], {20, 165}),
        (["EIP-20", "PEP 8", "pep-0008", "ERC 20", "erc20", "EIP  165"], {20, 8, 165}),
        ([" 20 ", "0020"], {20}),
        ([20, "20", "EIP-20"], {20}),
        ([-5, 0], {-5, 0}),
        ([], set()),
    ],
)
def test_set_answer_items_are_read_as_integers(answer, members):
    scored = score_answer(set_item, {"gold": sorted(members)}, answer)
    assert scored["valid"] and scored["correct"], answer


@pytest.mark.parametrize(
    "answer",
    [
        [True],
        [False],
        [20.0],
        [20.5],
        [None],
        [[20]],
        [{"n": 20}],
        ["20.0"],
        ["+20"],
        ["-20"],
        ["twenty"],
        ["20, 21"],
        ["EIP-"],
        ["PEP"],
        [""],
        ["0x14"],
        ["EIP-20x"],
        [chr(0x662) + chr(0x660)],  # Arabic-Indic digits: not ASCII digits
        [20, True],
        [20, [165]],
        20,
        "20",
        "[20]",
        None,
        {"20": 1},
        (20,),
    ],
)
def test_set_answers_with_an_unreadable_item_or_shape_are_invalid(answer):
    scored = score_answer(set_item, {"gold": [20]}, answer)
    assert scored["valid"] is False and scored["correct"] is False, answer
    assert (scored["precision"], scored["recall"], scored["f1"]) == (0, 0, 0)
    # Even when the readable part is the whole gold, one bad item voids the answer.
    assert not score_answer(set_item, {"gold": []}, answer)["correct"]


def test_count_answers():
    right = score_answer(count_item, {"gold": 12}, 12)
    assert (right["correct"], right["abs_error"], right["within_one"]) == (True, 0, True)
    near = score_answer(count_item, {"gold": 12}, 13)
    assert (near["correct"], near["abs_error"], near["within_one"]) == (False, 1, True)
    below = score_answer(count_item, {"gold": 12}, 11)
    assert (below["correct"], below["abs_error"], below["within_one"]) == (False, 1, True)
    far = score_answer(count_item, {"gold": 12}, 20)
    assert (far["correct"], far["abs_error"], far["within_one"]) == (False, 8, False)
    assert score_answer(count_item, {"gold": 0}, 0)["correct"]
    negative = score_answer(count_item, {"gold": 12}, -3)
    assert (negative["valid"], negative["abs_error"]) == (True, 15)
    assert right["outcome_correct"] is None and right["f1"] is None


def test_an_unknown_answer_type_is_an_error():
    with pytest.raises(ValueError, match="answer_type"):
        score_answer({"id": "x", "answer_type": "essay", "corpus": "pep"}, {"gold": 1}, 1)


# --- aggregation ---------------------------------------------------------------------------------


def row(item_id, run, correct, *, system="dense", arm="H+", family="F1", stratum="change",
        corpus="pep", tokens_in=100, tokens_out=10, tool_calls=0):
    return {"id": item_id, "system": system, "arm": arm, "run": run, "family": family,
            "stratum": stratum, "corpus": corpus, "correct": correct, "tokens_in": tokens_in,
            "tokens_out": tokens_out, "tool_calls": tool_calls}


def test_majority_pass_all():
    rows = [row("i1", 1, True), row("i1", 2, False), row("i1", 3, True)]
    (found,) = aggregate(rows)
    assert (found["system"], found["arm"], found["family"]) == ("dense", "H+", "F1")
    assert (found["stratum"], found["corpus"]) == ("change", "pep")
    assert found["n_items"] == 1 and found["runs"] == 3
    assert found["accuracy_mean_over_runs"] == pytest.approx(2 / 3)
    assert found["majority_accuracy"] == 1 and found["pass_at_3"] == 1 and found["all_3"] == 0
    assert (found["wilson_low"], found["wilson_high"]) == wilson(1, 1)


def test_aggregate_over_items_and_runs():
    flags = {"a": (True, True, True), "b": (True, False, True), "c": (False, False, True),
             "d": (False, False, False)}
    rows = [row(i, run, ok) for i, runs in flags.items() for run, ok in enumerate(runs, 1)]
    (found,) = aggregate(rows)
    assert found["n_items"] == 4 and found["runs"] == 3
    assert found["accuracy_mean_over_runs"] == 6 / 12
    assert found["majority_accuracy"] == 2 / 4  # a and b are correct in at least two runs
    assert found["pass_at_3"] == 3 / 4  # everything but d is correct in some run
    assert found["all_3"] == 1 / 4  # only a is correct in every run
    assert (found["wilson_low"], found["wilson_high"]) == wilson(2, 4)
    assert found["tokens_in"] == 12 * 100 and found["tokens_out"] == 12 * 10
    assert found["tool_use_rate"] == 0


def test_aggregate_tokens_and_tool_use():
    rows = [
        row("a", 1, True, tokens_in=10, tokens_out=1, tool_calls=3),
        row("a", 2, True, tokens_in=20, tokens_out=2, tool_calls=0),
        row("b", 1, False, tokens_in=30, tokens_out=3, tool_calls=1),
        row("b", 2, False, tokens_in=40, tokens_out=4, tool_calls=0),
    ]
    (found,) = aggregate(rows)
    assert (found["tokens_in"], found["tokens_out"]) == (100, 10)
    assert found["tool_use_rate"] == 0.5  # over the answers there are
    assert found["runs"] == 3  # section 12: always three runs
    # Rows that carry no usage fields still aggregate; with no row saying what it used, the token
    # totals and the tool-use rate are unknown, not zero (section 12).
    bare = [{k: v for k, v in r.items() if k not in ("tokens_in", "tokens_out", "tool_calls")}
            for r in rows]
    (plain,) = aggregate(bare)
    assert (plain["tokens_in"], plain["tokens_out"], plain["tool_use_rate"]) == (None, None, None)


def test_a_token_total_sums_the_rows_that_carry_it_and_is_unknown_when_none_does():
    rows = [row("a", 1, True, tokens_in=10, tokens_out=2), row("a", 2, True, tokens_in=None),
            row("b", 1, True, tokens_in="n/a", tokens_out=None)]
    (some,) = aggregate(rows)
    assert (some["tokens_in"], some["tokens_out"]) == (10, 12)  # tokens_out: 2 and row()'s 10
    none = [dict(r, tokens_in=None, tokens_out=True) for r in rows]  # a boolean is no count
    (found,) = aggregate(none)
    assert (found["tokens_in"], found["tokens_out"]) == (None, None)


def test_a_boolean_tool_flag_counts_as_a_tool_call():
    # The record has the number of tool calls; a system that can only say whether it used a tool
    # may give true or false, and true must not be read as zero. A row without the field is left
    # out of the rate rather than counted as no tool use (section 12).
    rows = [row("a", 1, True, tool_calls=True), row("a", 2, True, tool_calls=False),
            row("b", 1, True, tool_calls=2), row("b", 2, True, tool_calls=None)]
    (found,) = aggregate(rows)
    assert found["tool_use_rate"] == pytest.approx(2 / 3)


def test_a_missing_run_is_a_wrong_run():
    rows = [row("a", 1, True), row("a", 2, True), row("a", 3, True),
            row("b", 1, True), row("b", 2, True)]  # b has no third answer
    (found,) = aggregate(rows)
    assert found["n_items"] == 2 and found["runs"] == 3
    assert found["accuracy_mean_over_runs"] == 5 / 6
    assert found["majority_accuracy"] == 1 and found["pass_at_3"] == 1
    assert found["all_3"] == 0.5  # b did not answer every run
    only_first = [row("a", 1, True), row("a", 2, True), row("a", 3, True), row("b", 1, True)]
    (found,) = aggregate(only_first)
    assert found["majority_accuracy"] == 0.5  # b is correct in one run of three


def test_aggregate_keeps_groups_apart_and_in_order():
    rows = [
        row("a", 1, True, system="z", arm="H-"),
        row("a", 1, False, system="z", arm="H+"),
        row("a", 1, True, system="y", arm="H+", family="F2", stratum="positive"),
        row("a", 1, True, system="y", arm="H+", corpus="eth"),
        row("a", 1, True, system="y", arm="H+"),
        row("a", 2, True, system="y", arm="H+"),
    ]
    found = aggregate(rows)
    keys = [(r["system"], r["arm"], r["family"], r["stratum"], r["corpus"]) for r in found]
    assert keys == sorted(keys) and len(keys) == 5
    by = {k: r for k, r in zip(keys, found, strict=True)}
    assert by[("z", "H-", "F1", "change", "pep")]["runs"] == 3
    # Every group counts runs 1, 2 and 3 (section 12): the F2 item that has only a first answer
    # is wrong in the other two, and the F1 item answered in runs 1 and 2 is wrong in run 3.
    f2 = by[("y", "H+", "F2", "positive", "pep")]
    assert f2["runs"] == 3 and f2["accuracy_mean_over_runs"] == pytest.approx(1 / 3)
    assert by[("y", "H+", "F1", "change", "pep")]["accuracy_mean_over_runs"] == pytest.approx(2 / 3)


def test_aggregate_counts_a_repeated_answer_once():
    rows = [row("a", 1, True), row("a", 1, False), row("a", 2, True)]
    (found,) = aggregate(rows)
    # The first answer for a run stands, and run 3, which has none, is wrong.
    assert found["accuracy_mean_over_runs"] == pytest.approx(2 / 3)


def test_secondary_scores():
    def scored(item_id, run, **kw):
        base = {"id": item_id, "system": "dense", "arm": "H+", "run": run, "family": "F5",
                "stratum": "count", "corpus": "pep", "correct": False, "valid": True,
                "outcome_correct": None, "precision": None, "recall": None, "f1": None,
                "abs_error": None, "within_one": None, "wall_s": 2.0}
        return base | kw

    counts = [scored("a", 1, correct=True, abs_error=0, within_one=True),
              scored("a", 2, abs_error=2, within_one=False),
              scored("b", 1, abs_error=1, within_one=True),
              scored("b", 2, valid=False, within_one=False)]
    (found,) = secondary(counts)
    # Two items over three runs: six cells, of which two have no answer at all.
    assert found["mean_abs_error"] == pytest.approx(1.0)  # the invalid answer has no error
    assert found["within_one_rate"] == pytest.approx(2 / 6)
    assert found["invalid_rate"] == pytest.approx(3 / 6)
    assert found["mean_wall_s"] == 2.0
    sets = [scored("a", 1, family="F3", stratum="size-2-4", precision=1.0, recall=0.5, f1=2 / 3),
            scored("a", 2, family="F3", stratum="size-2-4", precision=0.0, recall=0.0, f1=0.0)]
    (found,) = secondary(sets)
    assert (found["precision"], found["recall"]) == (pytest.approx(1 / 3), pytest.approx(1 / 6))
    assert found["f1"] == pytest.approx(2 / 9)
    statuses = [scored("a", 1, family="F1", stratum="change", outcome_correct=True),
                scored("a", 2, family="F1", stratum="change", outcome_correct=False),
                scored("b", 1, family="F1", stratum="change", outcome_correct=True)]
    (found,) = secondary(statuses)  # every missing cell (b's second and both third) is wrong
    assert found["outcome_accuracy"] == pytest.approx(2 / 6)
    assert found["invalid_rate"] == pytest.approx(3 / 6)


# --- the pilot gates -----------------------------------------------------------------------------

SYSTEMS = ("closed-book", "bm25", "grep-at-d", "long-context", "hipporag2", "graphiti")


def pilot_rows(change: dict[str, int], current: dict[str, int], n: int = 100, m: int = 50):
    """Aggregate rows and per-item majorities where system s is right on its first change[s]
    change items and current[s] current items, in all three runs. Nested sets of right items
    make every discordant pair favour the higher scorer, so the McNemar tests follow from the
    score differences alone."""
    scored = []
    per_item: dict[str, dict[str, bool]] = {}
    for system in SYSTEMS:
        per_item[system] = {}
        for stratum, ok, size in (("change", change[system], n), ("current", current[system], m)):
            for i in range(size):
                for run in (1, 2, 3):
                    scored.append(row(f"{stratum}{i}", run, i < ok, system=system, arm="H-",
                                      stratum=stratum, corpus="all"))
                if stratum == "change":
                    per_item[system][f"change{i}"] = i < ok
    return aggregate(scored), per_item


def uniform(value: int) -> dict[str, int]:
    return {system: value for system in SYSTEMS}


def test_pilot_gate_1_stops_when_every_system_is_at_the_ceiling():
    from asofbench.scoring import DECISIONS

    rows, per_item = pilot_rows(uniform(90), uniform(50))
    gates = pilot_gates(rows, per_item)
    assert gates["stopped_at"] == 1 and gates["signal"] is False
    # The decision is a top-level field, as it is when the pilot is not evaluable.
    assert gates["decision"] == DECISIONS[1] and "decision" not in gates["detail"]
    # One system under 85% and the gate does not fire.
    rows, per_item = pilot_rows({**uniform(90), "closed-book": 84}, uniform(50))
    assert pilot_gates(rows, per_item)["stopped_at"] != 1


def test_pilot_gate_1_thresholds_are_inclusive():
    rows, per_item = pilot_rows(uniform(85), uniform(50))
    assert pilot_gates(rows, per_item)["stopped_at"] == 1  # exactly 85% is 85% or more


def test_pilot_gate_2_stops_when_the_engines_fail_alike_on_current_and_change():
    change = {"closed-book": 40, "bm25": 50, "grep-at-d": 60, "long-context": 70,
              "hipporag2": 60, "graphiti": 60}
    current = {**uniform(25), "hipporag2": 32, "graphiti": 34}  # failure 36% and 32% vs 40%
    rows, per_item = pilot_rows(change, current)
    gates = pilot_gates(rows, per_item)
    assert gates["stopped_at"] == 2 and gates["signal"] is False
    # If only one engine fails alike, the benchmark is not measuring loading.
    rows, per_item = pilot_rows(change, {**current, "graphiti": 50})
    assert pilot_gates(rows, per_item)["stopped_at"] != 2
    # The gap of 10 points counts as within; 11 does not.
    rows, per_item = pilot_rows(change, {**current, "hipporag2": 30, "graphiti": 30})  # 40% fails
    assert pilot_gates(rows, per_item)["stopped_at"] == 2
    edge = {**current, "hipporag2": 35, "graphiti": 35}  # failure 30%: exactly 10 points below
    rows, per_item = pilot_rows(change, edge)
    assert pilot_gates(rows, per_item)["stopped_at"] == 2
    over = {**current, "hipporag2": 36, "graphiti": 36}  # failure 28%: 12 points below
    rows, per_item = pilot_rows(change, over)
    assert pilot_gates(rows, per_item)["stopped_at"] != 2


def test_pilot_gate_3_stops_when_grep_and_long_context_saturate_and_graphiti_keeps_up():
    change = {"closed-book": 40, "bm25": 60, "grep-at-d": 90, "long-context": 90,
              "hipporag2": 70, "graphiti": 88}
    current = {**uniform(50)}  # no loading: current is easy
    rows, per_item = pilot_rows(change, current)
    gates = pilot_gates(rows, per_item)
    assert gates["stopped_at"] == 3 and gates["signal"] is False
    # Graphiti significantly worse than the better of the two: gate 3 does not fire, and the
    # pairs that differ by 15 points with p < 0.05 are a signal.
    rows, per_item = pilot_rows({**change, "graphiti": 60}, current)
    gates = pilot_gates(rows, per_item)
    assert gates["stopped_at"] is None and gates["signal"] is True
    # Only one of the two saturates: gate 3 does not fire.
    rows, per_item = pilot_rows({**change, "long-context": 84}, current)
    assert pilot_gates(rows, per_item)["stopped_at"] != 3


def test_pilot_gate_3_compares_graphiti_with_the_better_of_the_two():
    # Graphiti is 5 points under long context (p = 0.0625) but 6 under grep (p = 0.03125): the
    # better of the two is grep, and the difference is significant, so the gate does not fire.
    change = {"closed-book": 40, "bm25": 60, "grep-at-d": 91, "long-context": 90,
              "hipporag2": 70, "graphiti": 85}
    rows, per_item = pilot_rows(change, uniform(50))
    gates = pilot_gates(rows, per_item)
    assert gates["stopped_at"] is None
    # Against the better one by 5 points (p = 0.0625): not significantly worse, so it fires.
    rows, per_item = pilot_rows({**change, "grep-at-d": 90}, uniform(50))
    assert pilot_gates(rows, per_item)["stopped_at"] == 3


def pilot_rows_from_sets(right: dict[str, set[int]], n: int = 100, m: int = 50):
    """Like ``pilot_rows``, but each system is right on exactly the F1-change items in its set,
    so the sets can overlap in any way; every system is right on all of F1-current."""
    scored = []
    per_item: dict[str, dict[str, bool]] = {}
    for system in SYSTEMS:
        per_item[system] = {f"change{i}": i in right[system] for i in range(n)}
        for stratum, size in (("change", n), ("current", m)):
            for i in range(size):
                ok = i in right[system] if stratum == "change" else True
                for run in (1, 2, 3):
                    scored.append(row(f"{stratum}{i}", run, ok, system=system, arm="H-",
                                      stratum=stratum, corpus="all"))
    return aggregate(scored), per_item


def test_pilot_gate_3_looks_only_at_the_better_of_the_two_baselines():
    # Graphiti (79) against grep (86), the better: 21 items grep alone gets right and 14 Graphiti
    # alone does, p = 0.31, not significantly worse. Against long context (85) it is: 6 items
    # long context alone gets right and none the other way, p = 0.03. Only the better counts.
    graphiti = set(range(79))
    long_context = graphiti | set(range(79, 85))
    grep = (graphiti - set(range(14))) | set(range(79, 100))
    right = {"closed-book": set(range(40)), "bm25": set(range(60)), "grep-at-d": grep,
             "long-context": long_context, "hipporag2": set(range(70)), "graphiti": graphiti}
    rows, per_item = pilot_rows_from_sets(right)
    gates = pilot_gates(rows, per_item)
    tests = gates["detail"]["gate3"]["tests"]
    assert list(tests) == ["grep-at-d"]
    assert tests["grep-at-d"]["p"] > 0.05
    assert mcnemar_exact(6, 0) < 0.05
    assert gates["stopped_at"] == 3


def test_pilot_gate_3_does_not_fire_when_graphiti_is_significantly_better():
    change = {"closed-book": 40, "bm25": 60, "grep-at-d": 85, "long-context": 85,
              "hipporag2": 70, "graphiti": 100}
    rows, per_item = pilot_rows(change, uniform(50))
    gates = pilot_gates(rows, per_item)
    assert gates["stopped_at"] is None and gates["signal"] is True


def test_pilot_gate_4_signal_needs_15_points_and_p_below_05():
    change = {"closed-book": 40, "bm25": 50, "grep-at-d": 60, "long-context": 70,
              "hipporag2": 60, "graphiti": 95}
    current = {**uniform(25), "hipporag2": 45, "graphiti": 50}
    rows, per_item = pilot_rows(change, current)
    gates = pilot_gates(rows, per_item)
    assert gates["stopped_at"] is None and gates["signal"] is True
    # Two systems 20 points apart, everything else within 15 points of its neighbours.
    flat = {"closed-book": 50, "bm25": 52, "grep-at-d": 54, "long-context": 56,
            "hipporag2": 58, "graphiti": 70}
    rows, per_item = pilot_rows(flat, uniform(45))
    gates = pilot_gates(rows, per_item)
    assert gates["stopped_at"] is None and gates["signal"] is True
    # No pair 15 points apart: the null.
    close = {"closed-book": 50, "bm25": 52, "grep-at-d": 54, "long-context": 56,
             "hipporag2": 58, "graphiti": 60}
    rows, per_item = pilot_rows(close, uniform(45))
    gates = pilot_gates(rows, per_item)
    assert gates["stopped_at"] is None and gates["signal"] is False


def test_pilot_gate_4_needs_both_conditions():
    # 15 points apart on 100 items is significant, so the edge needs a small sample: 15 points
    # on 20 items (3 items, p = 0.25) is not significant.
    change = {"closed-book": 10, "bm25": 11, "grep-at-d": 12, "long-context": 13,
              "hipporag2": 11, "graphiti": 13}
    current = uniform(5)
    rows, per_item = pilot_rows(change, current, n=20, m=10)
    gates = pilot_gates(rows, per_item)
    assert gates["stopped_at"] is None and gates["signal"] is False
    # 15 points exactly and p < 0.05: 100 items, 15 discordant in one direction (p = 6e-5, the
    # smallest of the ten pairs, so 6e-4 after the Holm adjustment).
    change = {"closed-book": 50, "bm25": 50, "grep-at-d": 54, "long-context": 56,
              "hipporag2": 58, "graphiti": 65}
    rows, per_item = pilot_rows(change, uniform(45))
    gates = pilot_gates(rows, per_item)
    assert gates["signal"] is True  # bm25 vs graphiti: exactly 15 points
    change = {**change, "graphiti": 64}  # 14 points from bm25 (and from closed book)
    rows, per_item = pilot_rows(change, uniform(45))
    assert pilot_gates(rows, per_item)["signal"] is False


def test_holm_is_the_step_down_adjustment():
    from asofbench.scoring import holm
    # Ten pairs: the smallest p is multiplied by ten, the next by nine, and so on.
    assert holm([0.004] + [0.5] * 9)[0] == pytest.approx(0.04)  # passes at 0.05
    assert holm([0.04] + [0.5] * 9)[0] == pytest.approx(0.4)  # does not
    # Monotone in the order of the raw p-values, in the order given, and capped at 1.
    assert holm([0.03, 0.01, 0.02]) == pytest.approx([0.04, 0.03, 0.04])
    assert holm([0.5, 0.6]) == [1.0, 1.0]
    assert holm([]) == []


def test_pilot_gate_4_counts_only_the_ten_pairs_without_closed_book():
    # Only closed book is 15 points or more from anyone: no signal, though the unadjusted reading
    # over all fifteen pairs finds five pairs, which it reports and decides nothing with.
    change = {"closed-book": 30, "bm25": 50, "grep-at-d": 52, "long-context": 54,
              "hipporag2": 56, "graphiti": 60}
    rows, per_item = pilot_rows(change, uniform(50))
    gates = pilot_gates(rows, per_item)
    assert gates["stopped_at"] is None and gates["signal"] is False
    gate4 = gates["detail"]["gate4"]
    assert len(gate4["pairs"]) == 10
    assert not [p for p in gate4["pairs"] if "closed-book" in (p["a"], p["b"])]
    unadjusted = gate4["unadjusted"]
    assert len(unadjusted["pairs"]) == 15 and unadjusted["qualifying"] == 5
    assert all("closed-book" in (p["a"], p["b"]) for p in unadjusted["pairs"] if p["qualifies"])


def test_pilot_gate_4_holm_adjusts_the_ten_pairs():
    # 20 F1-change items. Graphiti against bm25 is 45 points apart with 9 discordant items, all its
    # own: raw p = 2/512, the smallest of the ten, so Holm multiplies it by ten: 0.039, a signal.
    base = {"closed-book": set(range(10)), "bm25": set(range(10)), "grep-at-d": set(range(11)),
            "long-context": set(range(12)), "hipporag2": set(range(11))}
    rows, per_item = pilot_rows_from_sets({**base, "graphiti": set(range(19))}, n=20, m=10)
    gates = pilot_gates(rows, per_item)
    assert gates["stopped_at"] is None and gates["signal"] is True
    pairs = {(p["a"], p["b"]): p for p in gates["detail"]["gate4"]["pairs"]}
    best = pairs[("bm25", "graphiti")]
    assert best["p"] == 2 / 512 and best["p_holm"] == pytest.approx(10 * 2 / 512)
    assert best["qualifies"] is True
    assert [k for k, p in pairs.items() if p["qualifies"]] == [("bm25", "graphiti")]
    # Graphiti right on 1 to 17: against bm25, 8 items for it and 1 against, raw p = 0.039,
    # adjusted 0.39. The unadjusted reading would call it (and the closed-book pair) a signal.
    rows, per_item = pilot_rows_from_sets({**base, "graphiti": set(range(1, 18))}, n=20, m=10)
    gates = pilot_gates(rows, per_item)
    assert gates["stopped_at"] is None and gates["signal"] is False
    gate4 = gates["detail"]["gate4"]
    pair = next(p for p in gate4["pairs"] if (p["a"], p["b"]) == ("bm25", "graphiti"))
    assert pair["p"] == mcnemar_exact(1, 8) == 2 * 10 / 512
    assert pair["p_holm"] == pytest.approx(10 * 2 * 10 / 512) and pair["qualifies"] is False
    assert gate4["unadjusted"]["qualifying"] == 2
    assert gates["decision"].startswith("report the null")


def test_pilot_gate_3_on_a_tie_compares_graphiti_with_each_baseline():
    # Grep at D and long context both score 90%: Graphiti (84%) does not differ significantly
    # from grep (16 items against 10, p = 0.33) but does from long context (6 against 0, p =
    # 0.03), so the gate does not stop; it stops when it differs from neither.
    graphiti = set(range(84))
    grep = set(range(74)) | set(range(84, 100))
    right = {"closed-book": set(range(40)), "bm25": set(range(60)), "grep-at-d": grep,
             "long-context": set(range(90)), "hipporag2": set(range(70)), "graphiti": graphiti}
    rows, per_item = pilot_rows_from_sets(right)
    gates = pilot_gates(rows, per_item)
    gate3 = gates["detail"]["gate3"]
    assert gate3["better"] == ["grep-at-d", "long-context"]
    assert gate3["tests"]["grep-at-d"]["p"] > 0.05 > gate3["tests"]["long-context"]["p"]
    assert gates["stopped_at"] != 3
    rows, per_item = pilot_rows_from_sets({**right, "long-context": grep})
    assert pilot_gates(rows, per_item)["stopped_at"] == 3


def test_gate_3_says_it_tests_for_no_significant_difference():
    from asofbench.scoring import DECISIONS
    assert "does not differ significantly" in DECISIONS[3] and "two-sided" in DECISIONS[3]
    assert "worse" not in DECISIONS[3]
    assert "Holm" in DECISIONS["signal"] and "closed book" in DECISIONS["signal"]


def test_pilot_gates_read_only_the_six_pilot_systems():
    rows, per_item = pilot_rows(uniform(90), uniform(50))
    extra = [dict(r, system="oracle", accuracy_mean_over_runs=0.1, majority_accuracy=0.1)
             for r in rows if r["system"] == "graphiti"]
    gates = pilot_gates(rows + extra, per_item)
    assert gates["stopped_at"] == 1  # a seventh, weak system is not one of the six


def test_pilot_gates_pool_the_two_corpora_and_use_only_h_minus_f1():
    # The same scenario as gate 1, with each system's rows split by corpus (50 items each),
    # plus H+ rows that would break the ceiling if they were read.
    scored = []
    per_item: dict[str, dict[str, bool]] = {s: {} for s in SYSTEMS}
    for system in SYSTEMS:
        for corpus in ("pep", "eth"):
            for i in range(50):
                for run in (1, 2, 3):
                    ok = i < 45
                    scored.append(row(f"{corpus}{i}", run, ok, system=system, arm="H-",
                                      corpus=corpus))
                    scored.append(row(f"{corpus}{i}", run, False, system=system, arm="H+",
                                      corpus=corpus))
                per_item[system][f"{corpus}{i}"] = i < 45
    gates = pilot_gates(aggregate(scored), per_item)
    assert gates["stopped_at"] == 1  # 90% pooled; the H+ rows decide nothing


def test_pilot_gates_weight_each_corpus_by_its_item_count():
    # 20 PEP items at 100% and 80 Ethereum items at 80%: pooled 84%, under the ceiling, although
    # the mean of the two corpora's scores (90%) is over it.
    scored = []
    per_item: dict[str, dict[str, bool]] = {s: {} for s in SYSTEMS}
    for system in SYSTEMS:
        for corpus, size, right in (("pep", 20, 20), ("eth", 80, 64)):
            for i in range(size):
                for run in (1, 2, 3):
                    scored.append(row(f"{corpus}{i}", run, i < right, system=system, arm="H-",
                                      corpus=corpus))
                per_item[system][f"{corpus}{i}"] = i < right
    gates = pilot_gates(aggregate(scored), per_item)
    assert gates["detail"]["accuracy_change"]["graphiti"] == pytest.approx(0.84)
    # Under the ceiling, so the ceiling gate does not stop (there are no F1-current rows here, so
    # the pilot cannot be decided past it, and no gate carries a decision flag).
    assert gates["detail"]["gate1"]["below"] == list(SYSTEMS) and gates["evaluable"] is False
    assert "stop" not in gates["detail"]["gate1"]


def test_an_item_is_a_majority_when_more_than_half_of_the_runs_are_right():
    def majority(flags):
        rows = [row("a", run, ok) for run, ok in enumerate(flags, 1)]
        return aggregate(rows)[0]["majority_accuracy"]

    # Correct in at least two of the three runs (section 12); a missing run is wrong, and a run
    # labelled 4 is not one of the three.
    assert majority([True]) == 0 and majority([False]) == 0
    assert majority([True, False]) == 0 and majority([True, True]) == 1
    assert majority([True, True, False]) == 1 and majority([True, False, False]) == 0
    assert majority([True, False, False, True]) == 0 and majority([True, True, False, False]) == 1


def test_pilot_gates_with_rows_that_carry_only_the_accuracy():
    rows = [{"system": s, "stratum": "change", "majority_accuracy": 0.9, "n_items": 100}
            for s in SYSTEMS]
    per_item = {s: {f"i{k}": k < 90 for k in range(100)} for s in SYSTEMS}
    assert pilot_gates(rows, per_item)["stopped_at"] == 1


def test_pilot_gates_are_not_evaluable_without_all_six_systems():
    rows, per_item = pilot_rows({**uniform(40), "long-context": 90, "grep-at-d": 95}, uniform(50))
    partial = [r for r in rows if r["system"] != "graphiti"]
    gates = pilot_gates(partial, {k: v for k, v in per_item.items() if k != "graphiti"})
    # A missing system makes the pilot not evaluable: no gate stop and no signal is reported.
    assert gates["evaluable"] is False
    assert gates["missing_systems"] == ["graphiti"]
    assert "stopped_at" not in gates and "signal" not in gates
    assert "not evaluable" in gates["decision"] and "graphiti" in gates["decision"]
    # Several systems missing, and none at all.
    two = [r for r in rows if r["system"] not in ("graphiti", "bm25")]
    assert pilot_gates(two, per_item)["missing_systems"] == ["bm25", "graphiti"]
    nothing = pilot_gates([], {})
    assert nothing["evaluable"] is False and nothing["missing_systems"] == list(SYSTEMS)
    assert "stopped_at" not in nothing and "signal" not in nothing
    # The others cannot stand in for a missing system: not even the ceiling gate decides.
    high, per_high = pilot_rows(uniform(90), uniform(50))
    gates = pilot_gates([r for r in high if r["system"] != "bm25"], per_high)
    assert gates["evaluable"] is False and gates["missing_systems"] == ["bm25"]


def test_pilot_gates_are_not_evaluable_without_the_inputs_a_gate_needs():
    rows, per_item = pilot_rows({**uniform(40), "grep-at-d": 95}, uniform(50))
    # Gate 2 reads the engines' F1-current rows, and the ceiling gate does not decide first.
    no_current = [r for r in rows if not (r["stratum"] == "current"
                                          and r["system"] in ("hipporag2", "graphiti"))]
    gates = pilot_gates(no_current, per_item)
    assert gates["evaluable"] is False and gates["missing_systems"] == []
    assert gates["missing_inputs"] == ["hipporag2 F1-current", "graphiti F1-current"]
    assert "stopped_at" not in gates and "signal" not in gates
    # Gates 3 and 4 read each item's majority result.
    gates = pilot_gates(rows, {})
    assert gates["evaluable"] is False
    assert gates["missing_inputs"] == [f"{s} per-item results" for s in SYSTEMS]
    # An earlier gate that decides makes the later inputs unnecessary.
    high, per_high = pilot_rows(uniform(90), uniform(50))
    only_change = [r for r in high if r["stratum"] == "change"]
    decided = pilot_gates(only_change, {})
    assert decided["evaluable"] is True and decided["stopped_at"] == 1


# --- asofbench score ----------------------------------------------------------------------------


def question(item_id, family, stratum, corpus, answer_type, arms, **extra):
    return {"id": item_id, "family": family, "stratum": stratum, "corpus": corpus,
            "proposal": None, "as_of": "2020-01-01", "arms": list(arms), "question": "?",
            "answer_type": answer_type, "choices": None, "canary": CANARY, **extra}


# The flags the build writes in every F1, F2 and CF item's gold meta, which the report reads; a
# test gives its own where it matters.
META_FLAGS = {"F1": {"h_minus_header_only_after_d": False},
              "F2": {"recoverable": {"H+": True, "H-": True}},
              "CF": {"decision_word": False, "header_cue": False}}


def gold_row(item_id, gold, outcome=None, **meta):
    flags = META_FLAGS.get(item_id.split("-")[0], {})
    return {"id": item_id, "gold": gold, "gold_outcome": outcome, "source": None,
            "as_of_commits": {}, "meta": {"rank": 0, **flags, **meta}, "canary": CANARY}


def small_data(root: Path) -> Path:
    items = [
        (question("F1-change-PEP-1-2010-01-01", "F1", "change", "pep", "status", ("H+", "H-")),
         gold_row("F1-change-PEP-1-2010-01-01", "Final", "adopted")),
        (question("F1-change-PEP-2-2011-01-01", "F1", "change", "pep", "status", ("H+", "H-")),
         gold_row("F1-change-PEP-2-2011-01-01", "Rejected", "declined")),
        (question("F2-positive-PEP-5-2026-10-27", "F2", "positive", "pep", "set", ("H+", "H-")),
         gold_row("F2-positive-PEP-5-2026-10-27", [387])),
        (question("F2-negative-PEP-6-2026-10-27", "F2", "negative", "pep", "set", ("H+", "H-")),
         gold_row("F2-negative-PEP-6-2026-10-27", [])),
        (question("F5-count-pep-Final-2020-01-01", "F5", "count", "pep", "count", ("H+",)),
         gold_row("F5-count-pep-Final-2020-01-01", 12)),
        (question("CF-counterfactual-PEP-7-2012-01-01", "CF", "counterfactual", "pep", "status",
                  ("CF",)),
         gold_row("CF-counterfactual-PEP-7-2012-01-01", "Rejected", "declined",
                  true_status="Accepted", decision_word=True)),
    ]
    data = root / "data"
    data.mkdir(parents=True)
    (data / "questions.jsonl").write_text("".join(dumps(q) + "\n" for q, _ in items),
                                          encoding="utf-8")
    (data / "gold.jsonl").write_text("".join(dumps(g) + "\n" for _, g in items), encoding="utf-8")
    return data


GOLD = {
    "F1-change-PEP-1-2010-01-01": "Final",
    "F1-change-PEP-2-2011-01-01": "Rejected",
    "F2-positive-PEP-5-2026-10-27": [387],
    "F2-negative-PEP-6-2026-10-27": [],
    "F5-count-pep-Final-2020-01-01": 12,
    "CF-counterfactual-PEP-7-2012-01-01": "Rejected",
}
ARMS_OF = {"F1": ("H+", "H-"), "F2": ("H+", "H-"), "F5": ("H+",), "CF": ("CF",)}


def answer(item_id, system, arm, run, value, *, tokens_in=100, tokens_out=10, tool_calls=0):
    return {"id": item_id, "system": system, "arm": arm, "run": run, "answer": value,
            "tokens_in": tokens_in, "tokens_out": tokens_out, "wall_s": 1.5,
            "tool_calls": tool_calls}


def perfect(system, runs=(1, 2, 3)):
    for item_id, value in GOLD.items():
        for arm in ARMS_OF[item_id.split("-")[0]]:
            for run in runs:
                yield answer(item_id, system, arm, run, value)


def write_answers(path: Path, rows, raw_lines=()):
    lines = [dumps(r) for r in rows] + list(raw_lines)
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return path


# The models every test run names, the registered ones (section 11): the runs hold systems that
# build their stores with a language model (hipporag2, graphiti) and systems that embed text
# (dense, hybrid), so neither the construction nor the embedding model is none.
ANSWERING, CONSTRUCTION = "claude-haiku-5-5", "claude-haiku-5-5"
SETTINGS = "effort medium, adaptive thinking, default temperature"  # the registered settings
EMBEDDING = "voyageai/voyage-4-nano@5fb34e870f49f19e79c94d5e118f5ee730133279"
MODELS = ["--answering-model", ANSWERING, "--answering-settings", SETTINGS,
          "--construction-model", CONSTRUCTION, "--embedding-model", EMBEDDING]


def run_score(data: Path, answers: list[Path], out: Path, *, entries=None, items=None,
              cf_messages=None, message_cues=None, training_cutoff=None) -> int:
    argv = ["score", "--data", str(data), "--answers", *map(str, answers), "--out", str(out),
            *MODELS]
    if training_cutoff is not None:
        argv += ["--training-cutoff", training_cutoff]
    if isinstance(entries, dict):  # written beside the output directory
        entries = write_items(out.parent / f"{out.name}-entries.json", entries)
    if entries is not None:
        argv += ["--entries", str(entries)]
    if items is not None:
        argv += ["--items", str(items)]
    if cf_messages is not None:
        argv += ["--cf-messages", str(cf_messages)]
    if message_cues is not None:
        argv += ["--message-cues", str(message_cues)]
    return cli.main(argv)


def write_items(path: Path, ids) -> Path:
    path.write_text(json.dumps(ids), encoding="utf-8")
    return path


def load(out: Path, name: str = "results.json"):
    return json.loads((out / name).read_text(encoding="utf-8"))


def find(results, **keys):
    (found,) = [r for r in results if all(r[k] == v for k, v in keys.items())]
    return found


@pytest.fixture
def registered(monkeypatch):
    """Marks a test's small build as the registered one (sections 12 and 13), for the tests that
    need a registered run: ``REGISTERED_DATA`` takes the SHA-256s of its questions, gold and pilot
    list, after an empty pilot list is written where the build has none, since the registered
    build has one. A build left unmarked is not the registered one."""
    from asofbench import scoring

    def mark(data: Path) -> Path:
        pilot = data / "pilot.json"
        if not pilot.exists():
            pilot.write_text(dumps({"canary": CANARY, "groups": [], "ids": []}) + "\n",
                             encoding="utf-8")
        monkeypatch.setattr(scoring, "REGISTERED_DATA", {
            name: hashlib.sha256((data / name).read_bytes()).hexdigest()
            for name in ("questions.jsonl", "gold.jsonl", "pilot.json")})
        return data

    return mark


def test_score_writes_results_for_perfect_and_wrong_systems(tmp_path):
    data = small_data(tmp_path)
    wrong = {
        "F1-change-PEP-1-2010-01-01": "Draft",
        "F1-change-PEP-2-2011-01-01": "Final",
        "F2-positive-PEP-5-2026-10-27": [1],
        "F2-negative-PEP-6-2026-10-27": [2],
        "F5-count-pep-Final-2020-01-01": 900,
        "CF-counterfactual-PEP-7-2012-01-01": "Accepted",
    }
    bad = [answer(i, "dense", arm, run, v) for i, v in wrong.items()
           for arm in ARMS_OF[i.split("-")[0]] for run in (1, 2, 3)]
    path = write_answers(tmp_path / "answers.jsonl", [*perfect("oracle"), *bad])
    out = tmp_path / "out"
    assert run_score(data, [path], out) == 0
    for name in ("results.json", "results.md"):
        assert (out / name).is_file()
    assert not (out / "gates.json").exists()  # the answers do not cover the pilot
    results = load(out)
    assert results["unknown_ids"] == []
    rows = results["results"]
    assert len(rows) == 2 * 8  # per system: F1 in two arms, F2 two strata in two arms, F5, CF
    for r in rows:
        want = 1.0 if r["system"] == "oracle" else 0.0
        assert r["accuracy_mean_over_runs"] == want
        assert r["majority_accuracy"] == want and r["pass_at_3"] == want and r["all_3"] == want
        assert r["runs"] == 3
        assert r["tokens_in"] == r["n_items"] * 3 * 100 and r["tokens_out"] == r["n_items"] * 3 * 10
    best = find(rows, system="oracle", arm="H+", family="F1", stratum="change")
    assert best["n_items"] == 2 and (best["wilson_low"], best["wilson_high"]) == wilson(2, 2)
    assert best["tokens_in"] == 2 * 3 * 100 and best["tokens_out"] == 2 * 3 * 10
    assert best["tool_use_rate"] == 0
    secondary_rows = results["secondary"]
    bad_status = find(secondary_rows, system="dense", arm="H+", family="F1")
    # Draft (open) for an adopted Final, and Final (adopted) for a declined Rejected.
    assert bad_status["outcome_accuracy"] == 0.0
    good_count = find(secondary_rows, system="oracle", family="F5")
    assert good_count["within_one_rate"] == 1.0 and good_count["mean_abs_error"] == 0
    bad_count = find(secondary_rows, system="dense", family="F5")
    assert bad_count["within_one_rate"] == 0.0 and bad_count["mean_abs_error"] == 888
    good_set = find(secondary_rows, system="oracle", arm="H+", family="F2", stratum="positive")
    assert good_set["f1"] == 1.0
    assert find(secondary_rows, system="dense", arm="H+", family="F2",
                stratum="negative")["f1"] == 0.0


def test_missing_answer_is_wrong_and_unknown_ids_reported(tmp_path):
    data = small_data(tmp_path)
    a, b = "F1-change-PEP-1-2010-01-01", "F1-change-PEP-2-2011-01-01"
    rows = [answer(a, "dense", "H+", run, "Final") for run in (1, 2, 3)]
    rows += [answer(b, "dense", "H+", 1, "Rejected")]  # no answer for b in runs 2 and 3
    rows += [answer("ghost-1", "dense", "H+", 1, "Final"),
             answer("ghost-1", "dense", "H+", 2, "Final"),
             answer("ghost-0", "dense", "H+", 1, "Final")]
    path = write_answers(tmp_path / "a.jsonl", rows)
    out = tmp_path / "out"
    assert run_score(data, [path], out, items=write_items(tmp_path / "i.json", [a, b])) == 0
    results = load(out)
    assert results["unknown_ids"] == ["ghost-0", "ghost-1"]
    found = find(results["results"], system="dense", arm="H+")
    assert found["n_items"] == 2 and found["runs"] == 3  # the ghosts are ignored
    assert found["accuracy_mean_over_runs"] == 4 / 6
    assert found["majority_accuracy"] == 0.5 and found["all_3"] == 0.5 and found["pass_at_3"] == 1.0
    coverage = find(results["coverage"], system="dense", arm="H+", family="F1")
    assert (coverage["answered"], coverage["asked"]) == (2, 2)


def test_omitting_an_item_lowers_accuracy(tmp_path):
    data = small_data(tmp_path)
    a, b = "F1-change-PEP-1-2010-01-01", "F1-change-PEP-2-2011-01-01"
    rows = [answer(i, "dense", "H+", run, GOLD[i]) for i in (a, b) for run in (1, 2, 3)]
    rows += [answer(a, "hybrid", "H+", run, GOLD[a]) for run in (1, 2, 3)]
    path = write_answers(tmp_path / "a.jsonl", rows)
    out = tmp_path / "out"
    assert run_score(data, [path], out, items=write_items(tmp_path / "i.json", [a, b])) == 0
    results = load(out)
    both = find(results["results"], system="dense", arm="H+")
    left_out = find(results["results"], system="hybrid", arm="H+")
    assert (both["n_items"], both["accuracy_mean_over_runs"]) == (2, 1.0)
    # The item it did not answer is wrong in every run, so the system that skipped it scores less.
    assert (left_out["n_items"], left_out["accuracy_mean_over_runs"]) == (2, 0.5)
    assert left_out["majority_accuracy"] == 0.5 and left_out["pass_at_3"] == 0.5
    assert left_out["all_3"] == 0.5
    assert (left_out["wilson_low"], left_out["wilson_high"]) == wilson(1, 2)
    # ... and the coverage table says how much it answered.
    coverage = find(results["coverage"], system="hybrid", arm="H+")
    assert (coverage["answered"], coverage["asked"]) == (1, 2)
    assert find(results["coverage"], system="dense", arm="H+")["answered"] == 2
    # Skipping the one hard item must not beat answering both and getting one wrong.
    wrong_b = [answer(a, "ppr", "H+", run, GOLD[a]) for run in (1, 2, 3)]
    wrong_b += [answer(b, "ppr", "H+", run, "Final") for run in (1, 2, 3)]
    path2 = write_answers(tmp_path / "b.jsonl", wrong_b)
    out2 = tmp_path / "out2"
    assert run_score(data, [path, path2], out2, items=tmp_path / "i.json") == 0
    again = load(out2)["results"]
    assert find(again, system="ppr", arm="H+")["accuracy_mean_over_runs"] == (
        find(again, system="hybrid", arm="H+")["accuracy_mean_over_runs"]
    )


def test_a_system_is_scored_in_every_arm_and_family_it_is_entered_in(tmp_path):
    data = small_data(tmp_path)
    rows = [answer("F1-change-PEP-1-2010-01-01", "dense", "H+", run, "Final") for run in (1, 2, 3)]
    rows += [answer("F1-change-PEP-2-2011-01-01", "dense", "H+", run, "Rejected")
             for run in (1, 2, 3)]
    path = write_answers(tmp_path / "a.jsonl", rows)
    out = tmp_path / "out"
    assert run_score(data, [path], out) == 0  # the registered default: entered everywhere
    results = load(out)
    groups = {(r["arm"], r["family"], r["stratum"]): r for r in results["results"]}
    # Every arm and family the system is entered in has rows, answered or not: it answered the
    # F1 items of arm H+, so everything else scores zero, with its rows present.
    assert set(groups) == {
        ("H+", "F1", "change"), ("H+", "F2", "positive"), ("H+", "F2", "negative"),
        ("H+", "F5", "count"), ("H-", "F1", "change"), ("H-", "F2", "positive"),
        ("H-", "F2", "negative"), ("CF", "CF", "counterfactual"),
    }
    assert groups[("H+", "F1", "change")]["accuracy_mean_over_runs"] == 1.0
    sizes = {"F1": 2, "F2": 1, "F5": 1, "CF": 1}
    for key, left in groups.items():
        if key == ("H+", "F1", "change"):
            continue
        assert left["n_items"] == sizes[key[1]] and left["runs"] == 3, key
        assert (left["accuracy_mean_over_runs"], left["majority_accuracy"]) == (0.0, 0.0), key
        assert (left["pass_at_3"], left["all_3"], left["tokens_in"]) == (0.0, 0.0, None), key
        assert left["tool_use_rate"] is None, key  # there is no answer to have used a tool
        assert (left["wilson_low"], left["wilson_high"]) == wilson(0, left["n_items"]), key
    coverage = {(c["arm"], c["family"], c["stratum"]): c for c in results["coverage"]}
    assert (coverage[("H+", "F1", "change")]["answered"],
            coverage[("H+", "F1", "change")]["asked"]) == (2, 2)
    assert (coverage[("H-", "F1", "change")]["answered"],
            coverage[("H-", "F1", "change")]["asked"]) == (0, 2)
    assert (coverage[("CF", "CF", "counterfactual")]["answered"],
            coverage[("CF", "CF", "counterfactual")]["asked"]) == (0, 1)
    second = {(r["arm"], r["family"], r["stratum"]): r for r in results["secondary"]}
    assert second[("H+", "F5", "count")]["invalid_rate"] == 1.0
    assert second[("H+", "F5", "count")]["within_one_rate"] == 0.0  # no answer is within one
    answers = results["answers"]
    assert answers["arms_without_answers"] == [{"system": "dense", "arms": ["CF", "H-"]}]


def test_entries_are_declared_per_system(tmp_path):
    data = small_data(tmp_path)
    f1 = "F1-change-PEP-1-2010-01-01"
    rows = [answer(i, "dense", "H+", run, v) for i, v in GOLD.items() if i.startswith(("F1", "F2"))
            for run in (1, 2, 3)]
    rows += [answer(f1, "hybrid", "H+", run, "Final") for run in (1, 2, 3)]
    rows += [answer("F2-positive-PEP-5-2026-10-27", "hybrid", "H+", 1, [387])]  # not entered
    path = write_answers(tmp_path / "a.jsonl", rows)
    out = tmp_path / "out"
    entries = {"hybrid": {"arms": ["H+"], "families": ["F1"]}}
    assert run_score(data, [path], out, entries=entries) == 0
    results = load(out)
    narrow = [r for r in results["results"] if r["system"] == "hybrid"]
    assert {(r["arm"], r["family"]) for r in narrow} == {("H+", "F1")}
    assert narrow[0]["n_items"] == 2 and narrow[0]["accuracy_mean_over_runs"] == 0.5
    # The system the file does not mention has the registered default: every arm and family.
    wide = {(r["arm"], r["family"]) for r in results["results"] if r["system"] == "dense"}
    assert wide == {("H+", "F1"), ("H+", "F2"), ("H+", "F5"), ("H-", "F1"), ("H-", "F2"),
                    ("CF", "CF")}
    # An answer outside the system's entries is listed and not scored.
    assert results["answers"]["out_of_scope_count"] == 1
    assert [(o["system"], o["id"]) for o in results["answers"]["out_of_scope"]] == [
        ("hybrid", "F2-positive-PEP-5-2026-10-27")
    ]
    # The entries each scored system had are recorded.
    assert results["scope"]["entries_file"] == "out-entries.json"
    assert results["scope"]["entries"]["hybrid"] == {"arms": ["H+"], "families": ["F1"]}
    assert results["scope"]["entries"]["dense"] == {
        "arms": ["CF", "H+", "H-"], "families": ["CF", "F1", "F2", "F5"]}
    assert "hybrid" in (out / "results.md").read_text(encoding="utf-8")


def test_an_entered_arm_with_no_answers_scores_zero_with_its_rows_present(tmp_path):
    data = small_data(tmp_path)
    a = "F1-change-PEP-1-2010-01-01"
    rows = [answer(a, "dense", "H+", run, "Final") for run in (1, 2, 3)]
    path = write_answers(tmp_path / "a.jsonl", rows)
    out = tmp_path / "out"
    # The file declares a system that has no answers at all, and an arm that s never answered.
    entries = {"dense": {"arms": ["H+", "H-"], "families": ["F1"]},
               "oracle": {"arms": ["H-"], "families": ["F1", "F2"]}}
    assert run_score(data, [path], out, entries=entries) == 0
    results = load(out)
    mine = {(r["system"], r["arm"], r["family"]): r for r in results["results"]}
    assert set(mine) == {("dense", "H+", "F1"), ("dense", "H-", "F1"), ("oracle", "H-", "F1"),
                         ("oracle", "H-", "F2")}
    assert mine[("dense", "H+", "F1")]["accuracy_mean_over_runs"] == 0.5
    for key in (("dense", "H-", "F1"), ("oracle", "H-", "F1"), ("oracle", "H-", "F2")):
        assert mine[key]["accuracy_mean_over_runs"] == 0.0 and mine[key]["runs"] == 3, key
        assert mine[key]["n_items"] >= 1 and mine[key]["tool_use_rate"] is None, key
    coverage = {(c["system"], c["arm"], c["family"], c["stratum"]): c for c in results["coverage"]}
    assert coverage[("oracle", "H-", "F2", "positive")]["answered"] == 0
    assert coverage[("oracle", "H-", "F2", "positive")]["asked"] == 1
    assert coverage[("oracle", "H-", "F1", "change")]["asked"] == 2
    assert results["answers"]["arms_without_answers"] == [
        {"system": "dense", "arms": ["H-"]}, {"system": "oracle", "arms": ["H-"]}]


def rich_data(root: Path) -> Path:
    """``small_data`` plus an F3 and an F4 item, which only some systems are entered in."""
    data = small_data(root)
    extra = [
        (question("F3-size-2-4-ERC-6384-2026-10-27", "F3", "size-2-4", "eth", "set", ("H+", "H-")),
         gold_row("F3-size-2-4-ERC-6384-2026-10-27", [155, 191, 712])),
        (question("F4-closure-final-EIP-2256-2020-12-23", "F4", "closure-final", "eth", "set",
                  ("H+",)),
         gold_row("F4-closure-final-EIP-2256-2020-12-23", [55, 155])),
    ]
    with (data / "questions.jsonl").open("a", encoding="utf-8") as f:
        f.write("".join(dumps(q) + "\n" for q, _ in extra))
    with (data / "gold.jsonl").open("a", encoding="utf-8") as f:
        f.write("".join(dumps(g) + "\n" for _, g in extra))
    return data


RICH_GOLD = {**GOLD, "F3-size-2-4-ERC-6384-2026-10-27": [155, 191, 712],
             "F4-closure-final-EIP-2256-2020-12-23": [55, 155]}
RICH_ARMS = {**ARMS_OF, "F3": ("H+", "H-"), "F4": ("H+",)}


def test_long_context_is_entered_in_f1_f2_and_cf_only_by_default(tmp_path):
    data = rich_data(tmp_path)

    def everything(system, arms=None):
        return [answer(i, system, arm, run, v) for i, v in RICH_GOLD.items()
                for arm in RICH_ARMS[i.split("-")[0]] if arms is None or arm in arms
                for run in (1, 2, 3)]

    # Both systems answer every item of every family, so long context's F3, F4 and F5 answers
    # are answers to families it is not entered in.
    rows = everything("closed-book") + everything("long-context", arms=("H+",))
    path = write_answers(tmp_path / "a.jsonl", rows)
    out = tmp_path / "out"
    assert run_score(data, [path], out) == 0
    results = load(out)
    by_system = {}
    for r in results["results"]:
        by_system.setdefault(r["system"], set()).add((r["arm"], r["family"]))
    # The other system is entered everywhere; long context only in F1, F2 and CF.
    assert {f for _, f in by_system["closed-book"]} == {"F1", "F2", "F3", "F4", "F5", "CF"}
    assert {f for _, f in by_system["long-context"]} == {"F1", "F2", "CF"}
    assert by_system["long-context"] == {("H+", "F1"), ("H+", "F2"), ("H-", "F1"), ("H-", "F2"),
                                         ("CF", "CF")}
    # It answered arm H+ only: its other entered arms (H- and CF) have rows, all zero.
    long_context = {(r["arm"], r["family"]): r for r in results["results"]
                    if r["system"] == "long-context"}
    assert long_context[("H+", "F1")]["accuracy_mean_over_runs"] == 1.0
    assert long_context[("H+", "F2")]["accuracy_mean_over_runs"] == 1.0
    for key in (("H-", "F1"), ("H-", "F2"), ("CF", "CF")):
        assert long_context[key]["accuracy_mean_over_runs"] == 0.0, key
    # Its answers to F3, F4 and F5 (3 runs each, in H+) are set aside, not scored.
    set_aside = results["answers"]["out_of_scope"]
    assert {(o["system"], o["id"].split("-")[0]) for o in set_aside} == {
        ("long-context", "F3"), ("long-context", "F4"), ("long-context", "F5")}
    assert results["answers"]["out_of_scope_count"] == 9
    assert results["scope"]["entries"]["long-context"]["families"] == ["CF", "F1", "F2"]
    # The default can be narrowed further for it, and not widened.
    narrower = tmp_path / "narrower"
    assert run_score(data, [path], narrower, entries={"long-context": {"families": ["F1"]}}) == 0
    assert {r["family"] for r in load(narrower)["results"]
            if r["system"] == "long-context"} == {"F1"}
    wider = tmp_path / "wider"
    assert run_score(data, [path], wider, entries={"long-context": {"families": ["F1", "F3"]}}) == 2
    assert not wider.exists()


def test_a_system_whose_answers_are_all_out_of_scope_still_has_its_rows(tmp_path):
    """Long context answers only an F3 item, which it is not entered in. It is still a system of
    the run, entered in F1, F2 and CF, so it has rows there, all zero, and the arms are named."""
    data = rich_data(tmp_path)
    rows = [answer("F3-size-2-4-ERC-6384-2026-10-27", "long-context", "H+", run, [155, 191, 712])
            for run in (1, 2, 3)]
    rows += [answer("F1-change-PEP-1-2010-01-01", "ppr", "H+", run, "Final")
             for run in (1, 2, 3)]
    out = tmp_path / "out"
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", rows)], out) == 0
    results = load(out)
    mine = [r for r in results["results"] if r["system"] == "long-context"]
    assert {(r["arm"], r["family"]) for r in mine} == {
        ("H+", "F1"), ("H-", "F1"), ("H+", "F2"), ("H-", "F2"), ("CF", "CF")}
    assert all(r["accuracy_mean_over_runs"] == 0.0 and r["n_items"] >= 1 for r in mine)
    assert results["answers"]["out_of_scope_count"] == 3
    assert {"system": "long-context", "arms": ["CF", "H+", "H-"]} in (
        results["answers"]["arms_without_answers"])
    # An id that is not in the questions names no system of the run.
    ghost = tmp_path / "oracle"
    rows = [answer("F1-change-PEP-1-2010-01-01", "ppr", "H+", 1, "Final"),
            answer("F9-nothing-PEP-1-2010-01-01", "oracle", "H+", 1, "Final")]
    assert run_score(data, [write_answers(tmp_path / "b.jsonl", rows)], ghost) == 0
    assert {r["system"] for r in load(ghost)["results"]} == {"ppr"}


@pytest.mark.parametrize(
    "entries",
    [
        [],
        "dense",
        {"dense": []},
        {"dense": {"arms": ["H*"]}},
        {"dense": {"families": ["F9"]}},
        {"dense": {"arm": ["H+"]}},  # a misspelt key
        {"dense": {"arms": []}},
        {"dense": {"families": []}},
        {"dense": {"arms": "H+"}},
        {"dense": {"families": [1]}},
        {"long-context": {"families": ["F1", "F3"]}},  # it is not entered in F3
        {"": {"arms": ["H+"]}},
    ],
)
def test_a_bad_entries_file_is_refused(tmp_path, entries, capsys):
    data = small_data(tmp_path)
    path = write_answers(tmp_path / "a.jsonl",
                         [answer("F1-change-PEP-1-2010-01-01", "dense", "H+", 1, "Final")])
    out = tmp_path / "out"
    given = write_items(tmp_path / "entries.json", entries)
    assert run_score(data, [path], out, entries=given) == 2
    assert not out.exists() and "asofbench score" in capsys.readouterr().err
    # Neither a missing file nor one that is not JSON.
    assert run_score(data, [path], out, entries=tmp_path / "missing.json") == 2
    (tmp_path / "bad.json").write_text("not json", encoding="utf-8")
    assert run_score(data, [path], out, entries=tmp_path / "bad.json") == 2
    assert not out.exists()


@pytest.mark.parametrize(
    ("entries", "says"),
    [
        ({"dense": {"arms": ["H*"]}}, "unknown arm H*"),
        ({"dense": {"families": ["F9"]}}, "unknown family F9"),
        ({"dense": {"arm": ["H+"]}}, "unknown key(s) arm"),
        ({"long-context": {"families": ["F1", "F5"]}}, "entered in CF, F1, F2 only"),
    ],
)
def test_a_refused_entries_file_says_what_is_wrong(tmp_path, entries, says, capsys):
    data = small_data(tmp_path)
    path = write_answers(tmp_path / "a.jsonl",
                         [answer("F1-change-PEP-1-2010-01-01", "dense", "H+", 1, "Final")])
    given = write_items(tmp_path / "entries.json", entries)
    assert run_score(data, [path], tmp_path / "out", entries=given) == 2
    assert says in capsys.readouterr().err


def test_entries_read_the_arm_names_the_answers_use(tmp_path):
    data = small_data(tmp_path)
    a = "F1-change-PEP-1-2010-01-01"
    path = write_answers(tmp_path / "a.jsonl", [answer(a, "dense", "H" + chr(0x2212), 1, "Final")])
    out = tmp_path / "out"
    entries = {"dense": {"arms": ["H" + chr(0x2212)], "families": ["F1"]}}
    assert run_score(data, [path], out, entries=entries) == 0
    assert {r["arm"] for r in load(out)["results"]} == {"H-"}


def test_items_list_limits_the_items_a_system_is_scored_on(tmp_path):
    data = small_data(tmp_path)
    a, b = "F1-change-PEP-1-2010-01-01", "F1-change-PEP-2-2011-01-01"
    rows = [answer(a, "dense", "H+", run, "Final") for run in (1, 2, 3)]
    rows += [answer(b, "dense", "H+", run, "Rejected") for run in (1, 2, 3)]
    path = write_answers(tmp_path / "a.jsonl", rows)
    h_plus_only = {"dense": {"arms": ["H+"]}}
    # A list of ids, and the structure of pilot.json (its "ids", or else its groups' ids).
    forms = {
        "list": [a],
        "ids": {"ids": [a], "groups": []},
        "groups": {"groups": [{"family": "F1", "stratum": "change", "corpus": "pep",
                               "arms": ["H-", "H+"], "ids": [a]}]},
    }
    for name, content in forms.items():
        out = tmp_path / f"out-{name}"
        listed = write_items(tmp_path / f"{name}.json", content)
        assert run_score(data, [path], out, items=listed, entries=h_plus_only) == 0, name
        results = load(out)
        (found,) = results["results"]  # only the listed item counts, in its own family
        assert (found["n_items"], found["accuracy_mean_over_runs"]) == (1, 1.0), name
        assert results["scope"]["items"] == 1 and results["scope"]["items_file"] == f"{name}.json"
        assert results["answers"]["out_of_scope_count"] == 3  # b's three answers
    # The list applies to the default entries too: both arms of the listed item.
    out = tmp_path / "default"
    assert run_score(data, [path], out, items=tmp_path / "list.json") == 0
    assert {(r["arm"], r["n_items"]) for r in load(out)["results"]} == {("H+", 1), ("H-", 1)}
    # A list that names an item the build does not have, or is no list at all, is refused.
    for bad_content in (["no-such-item"], {"nothing": []}, "a string", [1, 2]):
        refused = tmp_path / "refused"
        assert run_score(data, [path], refused, items=write_items(tmp_path / "x.json",
                                                                 bad_content)) == 2
        assert not refused.exists()
    (tmp_path / "x.json").write_text("not json", encoding="utf-8")
    assert run_score(data, [path], refused, items=tmp_path / "x.json") == 2
    assert run_score(data, [path], refused, items=tmp_path / "missing.json") == 2


def test_answers_outside_the_scope_are_listed_and_ignored(tmp_path):
    data = small_data(tmp_path)
    a = "F1-change-PEP-1-2010-01-01"
    rows = [answer(a, "dense", "H+", 1, "Final"),
            answer("F2-positive-PEP-5-2026-10-27", "dense", "H+", 1, [387])]
    path = write_answers(tmp_path / "a.jsonl", rows)
    out = tmp_path / "out"
    assert run_score(data, [path], out, entries={"dense": {"families": ["F1"]}}) == 0
    answers = load(out)["answers"]
    assert answers["scored"] == 1 and answers["rows"] == 2
    assert [(o["id"], o["system"], o["arm"]) for o in answers["out_of_scope"]] == [
        ("F2-positive-PEP-5-2026-10-27", "dense", "H+")
    ]
    assert "out of scope" in (out / "results.md").read_text(encoding="utf-8").lower()


def test_a_missing_run_is_flagged(tmp_path):
    data = small_data(tmp_path)
    a = "F1-change-PEP-1-2010-01-01"
    rows = [answer(a, "hybrid", "H+", run, "Final") for run in (1, 2, 3)]
    rows += [answer(a, "dense", "H+", run, "Final") for run in (1, 2)]  # run 3 never happened
    path = write_answers(tmp_path / "a.jsonl", rows)
    out = tmp_path / "out"
    assert run_score(data, [path], out, items=write_items(tmp_path / "i.json", [a])) == 0
    answers = load(out)["answers"]
    assert answers["missing_runs"] == [{"system": "dense", "arm": "H+", "runs": ["3"]}]
    assert find(load(out)["results"], system="dense", arm="H+")["accuracy_mean_over_runs"] == (
        2 / 3)
    text = (out / "results.md").read_text(encoding="utf-8")
    assert "## Systems with a run that has no answer" in text
    assert "dense in arm H+: no answer in run 3" in text


def test_arms_in_which_a_system_has_no_answers_are_listed(tmp_path):
    data = small_data(tmp_path)
    a = "F1-change-PEP-1-2010-01-01"
    rows = [answer(a, "dense", "H+", run, "Final") for run in (1, 2, 3)]
    rows += [answer(a, "hybrid", arm, run, "Final") for arm in ("H+", "H-") for run in (1, 2, 3)]
    path = write_answers(tmp_path / "a.jsonl", rows)
    out = tmp_path / "out"
    assert run_score(data, [path], out, items=write_items(tmp_path / "i.json", [a])) == 0
    results = load(out)
    # The item is asked in H+ and H-; a system that answered one only is entered in the other
    # too, scores zero there and the report says so.
    assert results["answers"]["arms_without_answers"] == [
        {"system": "dense", "arms": ["H-"]}]
    assert find(results["results"], system="dense", arm="H-")["accuracy_mean_over_runs"] == 0
    text = (out / "results.md").read_text(encoding="utf-8")
    assert "## Arms a system is entered in and has no answers in, so it scores zero there" in text
    assert "dense: H-" in text


def test_the_cli_takes_entries_items_and_cf_messages():
    parser = cli._parser()
    models = MODELS
    args = parser.parse_args(["score", "--data", "d", "--answers", "a", "--out", "o", *models])
    assert (args.entries, args.items, args.cf_messages) == (None, None, None)
    assert args.training_cutoff is None
    args = parser.parse_args(["score", "--data", "d", "--answers", "a", "b", "--out", "o",
                              "--entries", "e.json", "--items", "pilot.json",
                              "--cf-messages", "cf.jsonl.cf_messages.json",
                              "--training-cutoff", "2026-07-01", *models])
    assert args.entries == Path("e.json") and args.items == Path("pilot.json")
    assert args.cf_messages == Path("cf.jsonl.cf_messages.json")
    assert args.answers == [Path("a"), Path("b")]
    assert args.training_cutoff == date(2026, 7, 1)
    with pytest.raises(SystemExit):  # a cutoff is a day, written YYYY-MM-DD
        parser.parse_args(["score", "--data", "d", "--answers", "a", "--out", "o",
                           "--training-cutoff", "June 2026", *models])
    # Entries are declared per system: there is no option that narrows every system at once.
    with pytest.raises(SystemExit):
        parser.parse_args(["score", "--data", "d", "--answers", "a", "--out", "o",
                           "--families", "F1", *models])


def test_aggregate_and_secondary_take_the_denominators():
    rows = [row("a", 1, True), row("a", 2, True)]
    items = [("a", "F1", "change", "pep"), ("b", "F1", "change", "pep"),
             ("c", "F2", "positive", "pep")]
    denominators = {"dense": {"H+": items, "H-": [items[0]]}, "oracle": {"H+": [items[0]]}}
    found = aggregate(rows, denominators)
    by = {(r["system"], r["arm"], r["family"]): r for r in found}
    assert set(by) == {("dense", "H+", "F1"), ("dense", "H+", "F2"), ("dense", "H-", "F1"),
                       ("oracle", "H+", "F1")}
    f1 = by[("dense", "H+", "F1")]
    assert (f1["n_items"], f1["accuracy_mean_over_runs"]) == (2, pytest.approx(2 / 6))
    assert f1["all_3"] == 0.0 and f1["pass_at_3"] == 0.5
    assert by[("dense", "H+", "F1")]["tokens_in"] == 200  # only the answers that exist cost tokens
    f2 = by[("dense", "H+", "F2")]
    assert (f2["n_items"], f2["accuracy_mean_over_runs"]) == (1, 0.0)
    assert by[("dense", "H+", "F2")]["tool_use_rate"] is None
    # An arm the system is entered in but has not answered: its rows are there, over three runs,
    # as for a system with no answers at all.
    scored = by[("dense", "H-", "F1")]
    assert (scored["runs"], scored["accuracy_mean_over_runs"]) == (3, 0)
    assert (by[("oracle", "H+", "F1")]["runs"],
            by[("oracle", "H+", "F1")]["accuracy_mean_over_runs"]) == (3, 0)
    second = {(r["system"], r["arm"], r["family"]): r for r in secondary(rows, denominators)}
    assert second[("dense", "H+", "F2")]["invalid_rate"] == 1.0
    assert second[("dense", "H+", "F1")]["invalid_rate"] == pytest.approx(4 / 6)
    assert second[("oracle", "H+", "F1")]["invalid_rate"] == 1.0
    # A system that is not in the denominators gets no rows beyond its answers, as before.
    assert [r["n_items"] for r in aggregate(rows)] == [1]
    assert [(r["system"], r["n_items"]) for r in aggregate(rows, {"ppr": {"H+": items}})] == [
        ("dense", 1), ("ppr", 2), ("ppr", 1)]


def test_answers_the_items_do_not_ask_and_repeated_answers_are_listed_and_ignored(tmp_path):
    data = small_data(tmp_path)
    good = "F1-change-PEP-1-2010-01-01"
    rows = [
        answer(good, "dense", "H+", 1, "Final"),
        answer(good, "dense", "H+", 1, "Draft"),  # a repeated answer: the first stands
        answer(good, "dense", "H+", "1", "Draft"),  # 1 and "1" are one run
        answer(good, "dense", "H+", 2, "Final"),
        answer(good, "dense", "CF", 1, "Final"),  # CF is not an arm this item is asked in
        answer("F5-count-pep-Final-2020-01-01", "dense", "H-", 1, 12),  # nor is H- for F5
    ]
    path = write_answers(tmp_path / "a.jsonl", rows, raw_lines=[""])
    out = tmp_path / "out"
    f1 = ["F1-change-PEP-1-2010-01-01", "F1-change-PEP-2-2011-01-01"]
    assert run_score(data, [path], out, items=write_items(tmp_path / "i.json", f1)) == 0
    results = load(out)["answers"]
    assert "malformed" not in results  # a malformed row stops the run instead (section 12)
    assert [(d["id"], d["line"], d["run"]) for d in results["duplicates"]] == [
        (good, 2, "1"), (good, 3, "1")]
    assert sorted((u["id"], u["arm"]) for u in results["unasked"]) == [
        ("F1-change-PEP-1-2010-01-01", "CF"), ("F5-count-pep-Final-2020-01-01", "H-")]
    found = find(load(out)["results"], system="dense", arm="H+")
    # Two F1 items are asked in H+, each over three runs: two right answers for the first.
    assert found["runs"] == 3 and found["n_items"] == 2
    assert found["accuracy_mean_over_runs"] == 2 / 6


@pytest.mark.parametrize("field", ["id", "system", "arm", "run", "answer"])
def test_a_row_missing_a_registered_field_stops_the_score_with_its_line(tmp_path, field, capsys):
    """Section 12: a row is ``id``, ``system``, ``arm``, ``run`` and ``answer`` (and the accounting
    fields); one without any of the five is not scored as wrong but stops the run, exit 2."""
    data = small_data(tmp_path)
    good = answer("F1-change-PEP-1-2010-01-01", "dense", "H+", 1, "Final")
    bad = {k: v for k, v in good.items() if k != field}
    path = write_answers(tmp_path / "a.jsonl", [good, {**good, "run": 2}, bad])
    out = tmp_path / "out"
    assert run_score(data, [path], out) == 2
    err = capsys.readouterr().err
    assert err.startswith("asofbench score: ") and "Traceback" not in err
    assert "a.jsonl line 3" in err and field in err and "line 1" not in err
    assert not out.exists()


def test_every_malformed_row_is_named_and_nothing_is_scored(tmp_path, capsys):
    data = small_data(tmp_path)
    good = answer("F1-change-PEP-1-2010-01-01", "dense", "H+", 1, "Final")
    rows = [good, {**good, "arm": "H"}, {**good, "run": True}, {**good, "run": 1.5},
            {**good, "system": ""}, {**good, "id": 7}, {**good, "arm": None}]
    path = write_answers(tmp_path / "a.jsonl", rows, raw_lines=["not json", "[1, 2]", ""])
    other = write_answers(tmp_path / "b.jsonl", [good, {**good, "system": 3}])
    out = tmp_path / "out"
    assert run_score(data, [path, other], out) == 2
    err = capsys.readouterr().err
    for line in range(2, 10):
        assert f"a.jsonl line {line}:" in err, line
    assert "a.jsonl line 1:" not in err and "a.jsonl line 10" not in err
    assert "b.jsonl line 2:" in err and "b.jsonl line 1:" not in err
    assert "unknown arm 'H'" in err and not out.exists()


def test_an_answer_file_may_begin_with_a_byte_order_mark(tmp_path):
    """Some editors and Windows tools start a UTF-8 file with U+FEFF; its first row still counts."""
    data = small_data(tmp_path)
    good = "F1-change-PEP-1-2010-01-01"
    path = write_answers(tmp_path / "a.jsonl", [answer(good, "dense", "H+", run, "Final")
                                                 for run in (1, 2, 3)])
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    out = tmp_path / "out"
    assert run_score(data, [path], out, items=write_items(tmp_path / "i.json", [good]),
                     entries={"dense": {"arms": ["H+"]}}) == 0
    (found,) = load(out)["results"]
    assert found["accuracy_mean_over_runs"] == 1.0
    assert load(out)["answers"]["rows"] == 3


def test_a_null_answer_is_wrong_and_invalid(tmp_path):
    data = small_data(tmp_path)
    good = "F1-change-PEP-1-2010-01-01"
    rows = [answer(good, "dense", "H+", 1, "Final"), answer(good, "dense", "H+", 2, None),
            answer(good, "dense", "H+", 3, [])]
    path = write_answers(tmp_path / "a.jsonl", rows)
    out = tmp_path / "out"
    assert run_score(data, [path], out, items=write_items(tmp_path / "i.json", [good]),
                     entries={"dense": {"arms": ["H+"]}}) == 0
    (found,) = load(out)["results"]
    assert found["accuracy_mean_over_runs"] == pytest.approx(1 / 3)
    assert found["majority_accuracy"] == 0.0
    assert load(out)["secondary"][0]["invalid_rate"] == pytest.approx(2 / 3)


def test_scores_always_count_runs_1_2_3_and_set_other_labels_aside(tmp_path):
    """Section 12: run is read as a string; scores count runs 1, 2 and 3, a missing run is wrong,
    and any other label is reported and ignored."""
    data = small_data(tmp_path)
    a, b = "F1-change-PEP-1-2010-01-01", "F1-change-PEP-2-2011-01-01"
    rows = [answer(a, "dense", "H+", 1, "Final"), answer(a, "dense", "H+", "2", "Final"),
            answer(a, "dense", "H+", " 3 ", "Final"),
            answer(b, "dense", "H+", "1", "Rejected"), answer(b, "dense", "H+", 2, "Rejected"),
            answer(a, "dense", "H+", 4, "Final"), answer(b, "dense", "H+", "03", "Rejected"),
            answer(b, "dense", "H+", "three", "Rejected"),
            answer(a, "hybrid", "H+", 1, "Final"), answer(b, "hybrid", "H+", "1", "Rejected")]
    path = write_answers(tmp_path / "a.jsonl", rows)
    out = tmp_path / "out"
    assert run_score(data, [path], out, items=write_items(tmp_path / "i.json", [a, b]),
                     entries={"dense": {"arms": ["H+"]}, "hybrid": {"arms": ["H+"]}}) == 0
    results = load(out)
    dense = find(results["results"], system="dense")
    assert dense["runs"] == 3 and dense["accuracy_mean_over_runs"] == 5 / 6
    # A system that made only one run is scored over three: its other two are wrong.
    hybrid = find(results["results"], system="hybrid")
    assert hybrid["runs"] == 3 and hybrid["accuracy_mean_over_runs"] == 2 / 6
    assert hybrid["majority_accuracy"] == 0.0 and hybrid["pass_at_3"] == 1.0
    answers = results["answers"]
    assert [(r["line"], r["run"]) for r in answers["odd_runs"]] == [(6, "4"), (7, "03"),
                                                                    (8, "three")]
    assert answers["missing_runs"] == [{"system": "hybrid", "arm": "H+", "runs": ["2", "3"]}]
    text = (out / "results.md").read_text(encoding="utf-8")
    assert "## Runs other than 1, 2 and 3, set aside" in text
    assert "hybrid in arm H+: no answer in runs 2 and 3, so every item is wrong in it" in text
    assert 'a.jsonl line 6: dense H+ F1-change-PEP-1-2010-01-01 run "4"' in text


def test_a_repeated_answer_is_settled_by_file_name_then_line_whatever_the_order(tmp_path):
    data = small_data(tmp_path)
    good = "F1-change-PEP-1-2010-01-01"
    first = write_answers(tmp_path / "a.jsonl", [answer(good, "dense", "H+", 1, "Final")])
    second = write_answers(tmp_path / "b.jsonl", [answer(good, "dense", "H+", 1, "Draft"),
                                                  answer(good, "dense", "H+", 2, "Final"),
                                                  answer(good, "dense", "H+", "2", "Draft")])
    entries = write_items(tmp_path / "entries.json", {"dense": {"arms": ["H+"]}})
    outs = []
    for name, order in (("ab", [first, second]), ("ba", [second, first])):
        out = tmp_path / name
        assert run_score(data, order, out, items=write_items(tmp_path / "i.json", [good]),
                         entries=entries) == 0
        outs.append(out)
    one, two = (load(out) for out in outs)
    assert one == two
    # a.jsonl comes first by name, so its answer for run 1 stands; then b.jsonl in line order.
    assert find(one["results"], system="dense")["accuracy_mean_over_runs"] == 2 / 3
    assert [(d["file"], d["line"]) for d in one["answers"]["duplicates"]] == [
        ("b.jsonl", 1), ("b.jsonl", 3)]


def test_answers_from_several_files_are_pooled(tmp_path):
    data = small_data(tmp_path)
    good = "F1-change-PEP-1-2010-01-01"
    first = write_answers(tmp_path / "1.jsonl", [answer(good, "dense", "H+", 1, "Final"),
                                                 answer(good, "dense", "H+", 2, "Final")])
    second = write_answers(tmp_path / "2.jsonl", [answer(good, "dense", "H+", 3, "Draft")])
    out = tmp_path / "out"
    assert run_score(data, [first, second], out, items=write_items(tmp_path / "i.json", [good]),
                     entries={"dense": {"arms": ["H+"]}}) == 0
    (found,) = load(out)["results"]
    assert found["runs"] == 3 and found["accuracy_mean_over_runs"] == pytest.approx(2 / 3)
    assert found["majority_accuracy"] == 1.0


def test_unicode_minus_in_an_arm_name_is_read_as_the_hyphen(tmp_path):
    data = small_data(tmp_path)
    good = "F1-change-PEP-1-2010-01-01"
    minus = "H" + chr(0x2212)
    path = write_answers(tmp_path / "a.jsonl", [answer(good, "dense", minus, 1, "Final")])
    out = tmp_path / "out"
    assert run_score(data, [path], out, items=write_items(tmp_path / "i.json", [good]),
                     entries={"dense": {"arms": ["H-"]}}) == 0
    (found,) = load(out)["results"]
    assert found["arm"] == "H-" and found["runs"] == 3
    assert found["accuracy_mean_over_runs"] == pytest.approx(1 / 3)  # runs 2 and 3 are missing


def test_results_md(tmp_path):
    data = small_data(tmp_path)
    path = write_answers(tmp_path / "a.jsonl", list(perfect("oracle")))
    out = tmp_path / "out"
    assert run_score(data, [path], out) == 0
    text = (out / "results.md").read_text(encoding="utf-8")
    assert CANARY in text
    for heading in ("# asof-bench v0.1 results", "## Primary results", "## Secondary scores",
                    "## Coverage"):
        assert heading in text
    assert "oracle" in text and "Wilson" in text
    assert text.endswith("\n") and "\r" not in text


def test_usage_fields_are_read_from_the_answers(tmp_path):
    data = small_data(tmp_path)
    good = "F1-change-PEP-1-2010-01-01"
    rows = [
        answer(good, "dense", "H+", 1, "Final", tokens_in=7, tokens_out=3, tool_calls=4),
        answer(good, "dense", "H+", 2, "Final", tokens_in=5, tokens_out=1, tool_calls=True),
        answer(good, "dense", "H+", 3, "Final", tokens_in=0, tokens_out=0, tool_calls=False),
        {"id": good, "system": "hybrid", "arm": "H+", "run": 1, "answer": "Final",
         "tokens_in": "many", "tokens_out": None, "tool_calls": 2.0},
    ]
    path = write_answers(tmp_path / "a.jsonl", rows)
    out = tmp_path / "out"
    assert run_score(data, [path], out, items=write_items(tmp_path / "i.json", [good]),
                     entries={"dense": {"arms": ["H+"]}, "hybrid": {"arms": ["H+"]}}) == 0
    mine = find(load(out)["results"], system="dense")
    assert (mine["tokens_in"], mine["tokens_out"]) == (12, 4)
    assert mine["tool_use_rate"] == pytest.approx(2 / 3)
    # Unreadable token counts are no token counts: the totals are unknown, shown as "-".
    other = find(load(out)["results"], system="hybrid")
    assert (other["tokens_in"], other["tokens_out"], other["tool_use_rate"]) == (None, None, 1.0)
    text = (out / "results.md").read_text(encoding="utf-8")
    primary = text.split("\n## Primary results", 1)[1].split("\n## ", 1)[0]
    (line,) = [line for line in primary.splitlines() if line.startswith("| hybrid |")]
    assert line.endswith("| - | - | 1.000 |")


def test_a_row_without_accounting_is_scored_counted_and_left_out_of_the_averages(tmp_path):
    """Section 12: a row that lacks tokens_in, tokens_out, wall_s or tool_calls, or has a value
    there that is not a number, is still scored; the report counts such rows per system, and the
    cost figures leave each missing value out instead of counting it as zero."""
    data = small_data(tmp_path)
    good = "F1-change-PEP-1-2010-01-01"
    full = answer(good, "dense", "H+", 1, "Final", tool_calls=0) | {"wall_s": 1.0}
    no_wall = {k: v for k, v in answer(good, "dense", "H+", 2, "Final", tool_calls=2).items()
               if k != "wall_s"}
    odd = {k: v for k, v in answer(good, "dense", "H+", 3, "Final").items() if k != "tool_calls"}
    odd |= {"tokens_in": "many", "wall_s": 3.0}
    rows = [full, no_wall, odd, answer(good, "hybrid", "H+", 1, "Final")]
    out = tmp_path / "out"
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", rows)], out,
                     items=write_items(tmp_path / "i.json", [good]),
                     entries={"dense": {"arms": ["H+"]}, "hybrid": {"arms": ["H+"]}}) == 0
    results = load(out)
    dense = find(results["results"], system="dense")
    assert dense["accuracy_mean_over_runs"] == 1.0  # every row is scored
    assert dense["tokens_in"] == 200 and dense["tokens_out"] == 30  # over the values there are
    assert dense["tool_use_rate"] == 0.5  # of the two rows that say: not one in three
    assert find(results["secondary"], system="dense")["mean_wall_s"] == 2.0  # not 4/3
    assert results["accounting"] == {
        "dense": {"rows": 3, "without": 2,
                  "by_field": {"tokens_in": 1, "tokens_out": 0, "wall_s": 1, "tool_calls": 1}},
        "hybrid": {"rows": 1, "without": 0,
                   "by_field": {"tokens_in": 0, "tokens_out": 0, "wall_s": 0, "tool_calls": 0}}}
    text = (out / "results.md").read_text(encoding="utf-8")
    assert "## Accounting" in text and "| dense | 3 | 2 | 1 | 0 | 1 | 1 |" in text
    assert "| hybrid | 1 | 0 | 0 | 0 | 0 | 0 |" in text


MODEL_FLAGS = dict(zip(MODELS[::2], MODELS[1::2], strict=True))  # flag -> the value tests give


def test_score_needs_the_models_and_the_answering_settings(tmp_path):
    """Section 11: the answering model, its settings, the construction model and the embedding
    model are all required, so that none of them can be left to a quiet default."""
    base = ["score", "--data", "d", "--answers", "a.jsonl", "--out", "o"]
    for left_out in [*MODEL_FLAGS, None]:
        given = [word for flag, value in MODEL_FLAGS.items() if flag != left_out
                 for word in (flag, value)]
        if left_out is None:
            args = cli._parser().parse_args(base + given)
            assert (args.answering_model, args.answering_settings, args.construction_model,
                    args.embedding_model) == (ANSWERING, SETTINGS, CONSTRUCTION, EMBEDDING)
            continue
        with pytest.raises(SystemExit):
            cli._parser().parse_args(base + given)
    args = cli._parser().parse_args(base + [*MODELS, "--embedding-model", " an-embedder-v2 "])
    assert args.embedding_model == "an-embedder-v2"


@pytest.mark.parametrize("spelled", ["none", "None", " NONE "])
def test_none_names_no_answering_model_but_may_name_the_construction_and_embedding_models(
    spelled, capsys
):
    """Section 11: every scored system answers through a model, so the answering model is
    never none; none is the construction model of a run in which no store is LLM-built, and the
    embedding model of a run in which no system embeds text."""
    base = ["score", "--data", "d", "--answers", "a.jsonl", "--out", "o"]
    with pytest.raises(SystemExit) as refused:
        cli.main(base + [*MODELS, "--answering-model", spelled])
    assert refused.value.code == 2
    err = capsys.readouterr().err
    assert "--answering-model" in err and "every scored system answers through a model" in err
    args = cli._parser().parse_args(base + [*MODELS, "--construction-model", spelled,
                                            "--embedding-model", spelled])
    assert args.construction_model == args.embedding_model == "none"  # in any letter case


def test_none_in_any_letter_case_is_recorded_as_none(tmp_path):
    """The construction and embedding models' none is matched in any letter case and recorded
    as none in results.json, gates.json and every heading of results.md."""
    data, ids = pilot_data(tmp_path)  # a pilot list, so gates.json is written
    path = write_answers(tmp_path / "a.jsonl", pilot_answers(ids, {"bm25": (9, 6)},
                                                             systems=("bm25",)))
    out = tmp_path / "out"
    assert cli.main(["score", "--data", str(data), "--answers", str(path), "--out", str(out),
                     "--answering-model", ANSWERING, "--answering-settings", SETTINGS,
                     "--construction-model", "NONE", "--embedding-model", " None "]) == 0
    for name in ("results.json", "gates.json"):
        content = load(out, name)
        assert (content["construction_model"], content["embedding_model"]) == ("none", "none")
    text = (out / "results.md").read_text(encoding="utf-8")
    assert "; construction model: none; embedding model: none." in text
    headings = [line for line in text.splitlines() if line.startswith("## ") and "model" in line]
    assert headings and all("construction model none; embedding model none)" in heading
                            for heading in headings)
    assert not re.search(r"model:? (?:NONE|None)\b", text)  # never as typed


NEEDS = {"--answering-model": "give the answering model's id",
         "--answering-settings": "give the settings of every answering call (such as the "
                                 "effort, thinking and temperature)",
         "--construction-model": "give the construction model's id, or none when no store in the "
                                 "run is LLM-built",
         "--embedding-model": "give the embedding model's id, or none when no system of the run "
                              "embeds text"}


@pytest.mark.parametrize("blank", ["", " ", "\t"])
@pytest.mark.parametrize("flag", sorted(NEEDS))
def test_a_blank_model_is_refused_with_what_its_flag_needs(flag, blank, capsys):
    """A blank model id or setting says what the flag names, and for the construction and
    embedding models that none is the word for a run that has no such model (section 12)."""
    models = {**MODEL_FLAGS, flag: blank}
    argv = ["score", "--data", "d", "--answers", "a.jsonl", "--out", "o"]
    with pytest.raises(SystemExit) as refused:
        cli.main(argv + [word for pair in models.items() for word in pair])
    assert refused.value.code == 2
    assert f"argument {flag}: {NEEDS[flag]}" in capsys.readouterr().err


def test_the_models_are_recorded_and_named_at_the_top_and_in_every_table_heading(tmp_path):
    data, ids = pilot_data(tmp_path)
    path = write_answers(tmp_path / "a.jsonl", pilot_answers(ids, {s: (9, 6) for s in SYSTEMS}))
    out = tmp_path / "out"
    assert run_score(data, [path], out) == 0
    for name in ("results.json", "gates.json"):
        content = load(out, name)
        assert (content["answering_model"], content["answering_settings"],
                content["construction_model"], content["embedding_model"]) == (
            ANSWERING, SETTINGS, CONSTRUCTION, EMBEDDING), name
    lines = (out / "results.md").read_text(encoding="utf-8").splitlines()
    assert (f"Answering model: {ANSWERING}, with {SETTINGS}; construction model: {CONSTRUCTION}; "
            f"embedding model: {EMBEDDING}.") in lines[:8]
    named = (f"(answering model {ANSWERING} with {SETTINGS}; construction model {CONSTRUCTION}; "
             f"embedding model {EMBEDDING})")
    headings = [k for k, line in enumerate(lines) if line.startswith("## ")] + [len(lines)]
    with_tables = [lines[a] for a, b in zip(headings, headings[1:], strict=False)
                   if any(line.startswith("| ") for line in lines[a:b])]
    assert len(with_tables) >= 5  # primary, secondary, coverage, entries, accounting, pilot ...
    assert all(heading.endswith(named) for heading in with_tables), with_tables


def test_the_authors_own_retriever_is_labelled_in_every_results_table(tmp_path):
    """Section 11: every table of the results report labels the Personalized PageRank retriever
    as the author's own, and so does every other line of results.md that names it: the lines
    under the headline's tables and the lists at its end, such as the list of missing runs.
    results.json keeps its id, as the answer rows do."""
    data = small_data(tmp_path)
    rows = [*perfect("ppr", runs=(1, 2)), *perfect("dense")]  # ppr has no run 3, listed below
    path = write_answers(tmp_path / "a.jsonl", rows)
    out = tmp_path / "out"
    assert run_score(data, [path], out) == 0
    label = "ppr (author's own)"
    lines = (out / "results.md").read_text(encoding="utf-8").splitlines()
    tables = [line for line in lines if line.startswith("|")]
    cells = [[cell.strip() for cell in line.strip("|").split("|")] for line in tables]
    labelled = [line for line, row in zip(tables, cells, strict=True) if label in row]
    # The tables a system appears in here: primary, secondary, coverage, entries, accounting,
    # the CF items (a column per system), CF by decision word, by header cue, and F2.
    assert len(labelled) >= 9 and any(line.startswith("| Item |") for line in labelled)
    assert not any("ppr" in row for row in cells)  # no table cell is the bare id
    headline = "\n".join(lines).split("\n## Headline comparison", 1)[1].split("\n## ", 1)[0]
    # Under the headline ppr is not compared, since grep at D is not in the run, and the line
    # that says so names it as the tables do; so does the list of runs it left out.
    assert "hybrid, ppr (author's own), hipporag2" in headline
    assert "- ppr (author's own) in arm H+: no answer in run 3, so every item is wrong in it" in (
        lines)
    assert not re.search(r"\bppr\b(?! \(author's own\))", "\n".join(lines))  # never the bare id
    report = (out / "results.json").read_text(encoding="utf-8")
    assert "author's own" not in report and '"system":"ppr"' in report


def test_score_is_deterministic(tmp_path):
    data = small_data(tmp_path)
    path = write_answers(tmp_path / "a.jsonl", [*perfect("hybrid"), *perfect("dense")])
    one, two = tmp_path / "one", tmp_path / "two"
    assert run_score(data, [path], one) == 0 and run_score(data, [path], two) == 0
    for name in ("results.json", "results.md"):
        assert (one / name).read_bytes() == (two / name).read_bytes()
    # The row order in the answers file does not matter either (the file keeps its name, which
    # the results record).
    (tmp_path / "reversed").mkdir()
    reversed_path = write_answers(
        tmp_path / "reversed" / "a.jsonl", list(reversed([*perfect("hybrid"), *perfect("dense")]))
    )
    three = tmp_path / "three"
    assert run_score(data, [reversed_path], three) == 0
    assert (three / "results.json").read_bytes() == (one / "results.json").read_bytes()


def test_cf_items_are_scored_apart_by_decision_word(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    qs, gs = [], []
    for n, flag in enumerate((True, True, False, False)):
        item_id = f"CF-counterfactual-PEP-{n}-2012-01-01"
        qs.append(question(item_id, "CF", "counterfactual", "pep", "status", ("CF",)))
        gs.append(gold_row(item_id, "Rejected", "declined", true_status="Accepted",
                           decision_word=flag))
    (data / "questions.jsonl").write_text("".join(dumps(q) + "\n" for q in qs), encoding="utf-8")
    (data / "gold.jsonl").write_text("".join(dumps(g) + "\n" for g in gs), encoding="utf-8")
    # Right on the two items whose body has a decision word, wrong (the true status) on the rest.
    rows = [answer(q["id"], "dense", "CF", run, "Rejected" if i < 2 else "Accepted")
            for i, q in enumerate(qs) for run in (1, 2, 3)]
    path = write_answers(tmp_path / "a.jsonl", rows)
    out = tmp_path / "out"
    assert run_score(data, [path], out) == 0
    results = load(out)
    (together,) = results["results"]
    assert together["accuracy_mean_over_runs"] == 0.5 and together["n_items"] == 4
    split = {r["decision_word"]: r for r in results["cf_split"]}
    assert split[True]["n_items"] == 2 and split[True]["accuracy_mean_over_runs"] == 1.0
    assert split[False]["n_items"] == 2 and split[False]["accuracy_mean_over_runs"] == 0.0


def test_cf_items_are_scored_apart_by_header_cue(tmp_path):
    """Section 7: CF items whose file at D has a header field recording the true status are
    scored apart from the rest, as the decision-word items are."""
    data = tmp_path / "data"
    data.mkdir()
    qs, gs = [], []
    for n, cue in enumerate((True, False, False)):
        item_id = f"CF-counterfactual-PEP-{n}-2012-01-01"
        qs.append(question(item_id, "CF", "counterfactual", "pep", "status", ("CF",),
                           proposal=f"PEP-{n}"))
        gs.append(gold_row(item_id, "Rejected", "declined", true_status="Accepted",
                           decision_word=False, header_cue=cue,
                           header_cue_fields=["superseded-by"] if cue else []))
    (data / "questions.jsonl").write_text("".join(dumps(q) + "\n" for q in qs), encoding="utf-8")
    (data / "gold.jsonl").write_text("".join(dumps(g) + "\n" for g in gs), encoding="utf-8")
    # Wrong (the true status) on the item with a header cue, right on the other two.
    rows = [answer(q["id"], "dense", "CF", run, "Accepted" if i == 0 else "Rejected")
            for i, q in enumerate(qs) for run in (1, 2, 3)]
    out = tmp_path / "out"
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", rows)], out) == 0
    results = load(out)
    split = {r["header_cue"]: r for r in results["cf_header_split"]}
    assert (split[True]["n_items"], split[True]["accuracy_mean_over_runs"]) == (1, 0.0)
    assert (split[False]["n_items"], split[False]["accuracy_mean_over_runs"]) == (2, 1.0)
    assert [i["header_cue"] for i in results["cf_items"]] == [True, False, False]
    text = (out / "results.md").read_text(encoding="utf-8")
    assert "## Counterfactual items by header cue" in text
    assert "| Item | Decision word | Header cue |" in text
    # The section states the rule as section 7 and REPORT.md do, in the one wording the code keeps.
    from asofbench.questions import HEADER_CUE_RULE

    assert f"that records the true status ({HEADER_CUE_RULE}) are scored apart" in text


def test_f2_is_reported_over_all_items_and_over_the_recoverable_ones(tmp_path):
    """Section 12: each system's F2 accuracy over all items, and over the items whose gold the
    asked PEP's own episodes can show in the arm."""
    data = tmp_path / "data"
    data.mkdir()
    flags = {"F2-positive-PEP-1-2026-10-27": ([2], {"H+": True, "H-": False}),
             "F2-positive-PEP-3-2026-10-27": ([4], {"H+": False, "H-": False}),
             "F2-negative-PEP-5-2026-10-27": ([], {"H+": True, "H-": True})}
    qs = [question(i, "F2", i.split("-")[1], "pep", "set", ("H+", "H-")) for i in flags]
    gs = [gold_row(i, gold, recoverable=found) for i, (gold, found) in flags.items()]
    (data / "questions.jsonl").write_text("".join(dumps(q) + "\n" for q in qs), encoding="utf-8")
    (data / "gold.jsonl").write_text("".join(dumps(g) + "\n" for g in gs), encoding="utf-8")
    # Right on PEP 1 and PEP 5 in both arms, wrong on PEP 3.
    wrong = "F2-positive-PEP-3-2026-10-27"
    rows = [answer(i, "long-context", arm, run, [9] if i == wrong else gold)
            for i, (gold, _) in flags.items() for arm in ("H+", "H-") for run in (1, 2, 3)]
    out = tmp_path / "out"
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", rows)], out) == 0
    split = {r["arm"]: r for r in load(out)["f2_split"] if r["system"] == "long-context"}
    assert (split["H+"]["items"], split["H+"]["accuracy"]) == (3, pytest.approx(2 / 3))
    assert (split["H+"]["recoverable_items"], split["H+"]["recoverable_accuracy"]) == (2, 1.0)
    assert (split["H-"]["recoverable_items"], split["H-"]["recoverable_accuracy"]) == (1, 1.0)
    text = (out / "results.md").read_text(encoding="utf-8")
    assert "## F2 over all items and over the items the asked PEP's own episodes can show" in text


@pytest.mark.parametrize(("item_id", "key"), [
    ("F2-positive-PEP-5-2026-10-27", "recoverable"),
    ("CF-counterfactual-PEP-7-2012-01-01", "header_cue"),
    ("CF-counterfactual-PEP-7-2012-01-01", "decision_word"),
])
def test_an_item_without_a_flag_the_report_reads_stops_the_score(tmp_path, item_id, key, capsys):
    data = small_data(tmp_path)
    gold = data / "gold.jsonl"
    rows = [json.loads(line) for line in gold.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row["id"] == item_id:
            del row["meta"][key]
    gold.write_text("".join(dumps(row) + "\n" for row in rows), encoding="utf-8")
    answers = write_answers(tmp_path / "a.jsonl",
                            [answer("F1-change-PEP-1-2010-01-01", "dense", "H+", 1, "Final")])
    assert run_score(data, [answers], tmp_path / "out") == 2
    err = capsys.readouterr().err
    assert item_id in err and key in err and not (tmp_path / "out").exists()


def test_the_primary_table_gives_each_groups_majority_class_rate(tmp_path):
    data = small_data(tmp_path)
    path = write_answers(tmp_path / "a.jsonl", list(perfect("oracle")))
    out = tmp_path / "out"
    assert run_score(data, [path], out) == 0
    rates = {(r["arm"], r["family"], r["stratum"]): r["majority_class_rate"]
             for r in load(out)["results"]}
    # F1-change: Final and Rejected, one each; every other group has one item.
    assert rates[("H+", "F1", "change")] == 0.5 == rates[("H-", "F1", "change")]
    assert rates[("H+", "F2", "positive")] == 1.0 and rates[("CF", "CF", "counterfactual")] == 1.0
    text = (out / "results.md").read_text(encoding="utf-8")
    assert "| Accuracy | Majority-class rate |" in text


def cf_items_data(root: Path, flags=(True, True, False, False)):
    """Four CF items about PEP-0 to PEP-3; the flags say which have a decision word."""
    data = root / "data"
    data.mkdir()
    qs, gs = [], []
    for n, flag in enumerate(flags):
        item_id = f"CF-counterfactual-PEP-{n}-2012-01-01"
        qs.append(question(item_id, "CF", "counterfactual", "pep", "status", ("CF",),
                           proposal=f"PEP-{n}"))
        gs.append(gold_row(item_id, "Rejected", "declined", true_status="Accepted",
                           decision_word=flag))
    (data / "questions.jsonl").write_text("".join(dumps(q) + "\n" for q in qs), encoding="utf-8")
    (data / "gold.jsonl").write_text("".join(dumps(g) + "\n" for g in gs), encoding="utf-8")
    return data, qs


def test_the_cf_report_gives_each_items_message_count(tmp_path):
    data, qs = cf_items_data(tmp_path)
    counts = {"canary": CANARY, "PEP-0": 5, "PEP-1": 0, "PEP-2": 2, "PEP-3": 433}
    listed = write_items(tmp_path / "cf.json", counts)
    # Right on the first two items, wrong (the true status) on the other two.
    rows = [answer(q["id"], "dense", "CF", run, "Rejected" if i < 2 else "Accepted")
            for i, q in enumerate(qs) for run in (1, 2, 3)]
    # One run short on the first item, but a third run on the second.
    rows += [answer(qs[0]["id"], "hybrid", "CF", run, "Rejected") for run in (1, 2)]
    rows += [answer(qs[1]["id"], "hybrid", "CF", 3, "Accepted")]
    path = write_answers(tmp_path / "a.jsonl", rows)
    out = tmp_path / "out"
    assert run_score(data, [path], out, cf_messages=listed) == 0
    results = load(out)
    items = {r["id"]: r for r in results["cf_items"]}
    assert list(items) == sorted(q["id"] for q in qs)
    first = items[qs[0]["id"]]
    assert (first["proposal"], first["corpus"], first["decision_word"]) == ("PEP-0", "pep", True)
    assert [items[q["id"]]["cf_messages"] for q in qs] == [5, 0, 2, 433]
    assert [items[q["id"]]["decision_word"] for q in qs] == [True, True, False, False]
    # What each system made of each item, beside the count.
    assert first["systems"]["dense"] == {"accuracy_mean_over_runs": 1.0, "majority_correct": True}
    assert items[qs[3]["id"]]["systems"]["dense"] == {"accuracy_mean_over_runs": 0.0,
                                                  "majority_correct": False}
    assert first["systems"]["hybrid"]["accuracy_mean_over_runs"] == pytest.approx(2 / 3)
    assert items[qs[1]["id"]]["systems"]["hybrid"] == {"accuracy_mean_over_runs": 0.0,
                                                  "majority_correct": False}
    # The data have no manifest, so the file cannot be checked against one, and the report says so.
    assert results["cf_messages"] == {"file": "cf.json", "total": 440,
                                      "checked_against_manifest": False}
    text = (out / "results.md").read_text(encoding="utf-8")
    assert "## Counterfactual items" in text and "433" in text
    assert "CF message counts: cf.json, which the build's manifest does not register" in text
    # Section 7 counts commits, each once.
    assert "Commits whose message mentions it with a status value" in text
    assert "the number of commits outside the rewritten ones, each counted once" in text
    # The counts may also come under a "counts" key.
    bare = {key: value for key, value in counts.items() if key != "canary"}
    nested = write_items(tmp_path / "nested.json", {"canary": CANARY, "counts": bare})
    again = tmp_path / "again"
    assert run_score(data, [path], again, cf_messages=nested) == 0
    assert [i["cf_messages"] for i in load(again)["cf_items"]] == [
        i["cf_messages"] for i in results["cf_items"]]
    # Without the file the items are listed without counts.
    plain = tmp_path / "plain"
    assert run_score(data, [path], plain) == 0
    assert all(i["cf_messages"] is None for i in load(plain)["cf_items"])
    assert load(plain)["cf_messages"] is None
    assert {i["id"] for i in load(plain)["cf_items"]} == set(items)


@pytest.mark.parametrize(
    "content",
    [
        {"canary": CANARY, "PEP-0": 1, "PEP-1": 1, "PEP-2": 1},  # no count for PEP-3
        {"canary": CANARY, "PEP-0": -1, "PEP-1": 0, "PEP-2": 0, "PEP-3": 0},
        {"canary": CANARY, "PEP-0": True, "PEP-1": 0, "PEP-2": 0, "PEP-3": 0},
        {"canary": CANARY, "PEP-0": 1.5, "PEP-1": 0, "PEP-2": 0, "PEP-3": 0},
        {"canary": CANARY, "PEP-0": "3", "PEP-1": 0, "PEP-2": 0, "PEP-3": 0},
        [1, 2, 3],
        "text",
    ],
)
def test_a_bad_cf_messages_file_is_refused(tmp_path, content, capsys):
    data, qs = cf_items_data(tmp_path)
    path = write_answers(tmp_path / "a.jsonl", [answer(qs[0]["id"], "dense", "CF", 1, "Rejected")])
    out = tmp_path / "out"
    assert run_score(data, [path], out, cf_messages=write_items(tmp_path / "cf.json", content)) == 2
    assert not out.exists() and "asofbench score" in capsys.readouterr().err
    assert run_score(data, [path], out, cf_messages=tmp_path / "missing.json") == 2
    (tmp_path / "bad.json").write_text("not json", encoding="utf-8")
    assert run_score(data, [path], out, cf_messages=tmp_path / "bad.json") == 2
    assert not out.exists()


def test_binomial_tail_is_exact():
    from asofbench.scoring import binomial_tail
    # P(X >= k) for X ~ Binomial(n, p), by hand: n = 4, p = 1/2, k = 3 is (4 + 1) / 16.
    assert binomial_tail(3, 4, Fraction(1, 2)) == 5 / 16
    # n = 3, p = 3/5, k = 2: 3 * (3/5)^2 * (2/5) + (3/5)^3 = 54/125 + 27/125.
    assert binomial_tail(2, 3, Fraction(3, 5)) == 81 / 125
    assert binomial_tail(0, 7, Fraction(1, 3)) == 1.0
    assert binomial_tail(5, 5, Fraction(1, 2)) == 1 / 32
    assert binomial_tail(1, 10, Fraction(0)) == 0.0 and binomial_tail(10, 10, Fraction(1)) == 1.0
    # The registered floors' size: exact, and small without underflow.
    assert 0.0 < binomial_tail(150, 150, Fraction(91, 150)) < 1e-30
    with pytest.raises(ValueError):
        binomial_tail(4, 3, Fraction(1, 2))


FLOOR_PEP = ("Draft", "Draft", "Draft", "Final", "Rejected")  # most common: Draft, 3 of 5
FLOOR_ETH = ("Draft", "Draft", "Draft", "Review", "Final", "Last Call")  # Draft, 3 of 6


def floor_data(root: Path) -> tuple[Path, list[str], list[str], list[str]]:
    """F1-change items of both corpora with the golds above, and two PEP F1-recent items, one
    whose gold is the status at S and one whose gold is not."""
    data = root / "data"
    data.mkdir()
    qs, gs, pep, eth = [], [], [], []
    for corpus, golds, ids in (("pep", FLOOR_PEP, pep), ("eth", FLOOR_ETH, eth)):
        for n, value in enumerate(golds):
            item_id = f"F1-change-{corpus}{n}-2010-01-01"
            qs.append(question(item_id, "F1", "change", corpus, "status", ("H+", "H-")))
            gs.append(gold_row(item_id, value, status_at_s="Withdrawn",
                               gold_equals_status_at_s=False))
            ids.append(item_id)
    recent = []
    for n, (value, at_s) in enumerate((("Accepted", "Accepted"), ("Accepted", "Final"))):
        item_id = f"F1-recent-pep{n}-2026-06-01"
        qs.append(question(item_id, "F1", "recent", "pep", "status", ("H+", "H-")))
        gs.append(gold_row(item_id, value, status_at_s=at_s,
                           gold_equals_status_at_s=value == at_s))
        recent.append(item_id)
    (data / "questions.jsonl").write_text("".join(dumps(q) + "\n" for q in qs), encoding="utf-8")
    (data / "gold.jsonl").write_text("".join(dumps(g) + "\n" for g in gs), encoding="utf-8")
    return data, pep, eth, recent


def closed_book_rows(pep: list[str], eth: list[str]) -> list[dict]:
    """Closed book's answers: its result on each PEP item is the majority of every answer it gave
    the item, over the arms; on Ethereum it is right on every item in H- only."""
    def say(item_id, arm, runs, right):
        value = dict(zip(pep + eth, FLOOR_PEP + FLOOR_ETH, strict=True))[item_id]
        return [answer(item_id, "closed-book", arm, run, value if right else "Superseded")
                for run in runs]

    rows = say(pep[0], "H+", (1, 2, 3), True) + say(pep[0], "H-", (1, 2, 3), True)  # 6 of 6
    rows += say(pep[1], "H+", (1, 2, 3), True) + say(pep[1], "H-", (1, 2, 3), False)  # a tie
    rows += say(pep[2], "H+", (1,), True)  # one answer, right: a majority of what it gave
    rows += say(pep[3], "H+", (1,), True) + say(pep[3], "H+", (2,), False)
    rows += say(pep[3], "H-", (1,), False)  # one right of three
    # pep[4]: no answer at all, so wrong.
    return rows + [r for item_id in eth for r in say(item_id, "H-", (1, 2, 3), True)]


def test_contamination_is_judged_against_the_f1_change_floor(tmp_path):
    """Section 10: a corpus's floor is the share of its F1-change items whose gold is the most
    common gold value; closed book's result on an item is the majority of all its answers to it
    (a tie or no answer is wrong); an exact one-sided binomial test against the floor at p < 0.05
    marks the corpus's F1-change as possibly contaminated, and F1-change stays primary."""
    from asofbench.scoring import binomial_tail
    data, pep, eth, _ = floor_data(tmp_path)
    out = tmp_path / "out"
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", closed_book_rows(pep, eth))],
                     out) == 0
    flag = load(out)["contamination"]
    assert set(flag) == {"pep", "eth"}
    assert {k: flag["pep"][k] for k in ("items", "floor_gold", "floor_items")} == {
        "items": 5, "floor_gold": "Draft", "floor_items": 3}
    assert flag["pep"]["floor"] == pytest.approx(0.6)
    # Right on pep0 (six of six) and pep2 (one of one); the tie, one of three and no answer are
    # wrong.
    assert flag["pep"]["closed_book_right"] == 2 and flag["pep"]["arms"] == ["H+", "H-"]
    assert flag["pep"]["closed_book_right_ids"] == [pep[0], pep[2]]
    assert flag["pep"]["p"] == binomial_tail(2, 5, Fraction(3, 5)) == pytest.approx(0.91296)
    assert flag["pep"]["possibly_contaminated"] is False
    assert (flag["eth"]["floor_gold"], flag["eth"]["floor_items"], flag["eth"]["items"]) == (
        "Draft", 3, 6)
    assert flag["eth"]["closed_book_right"] == 6 and flag["eth"]["arms"] == ["H-"]
    assert flag["eth"]["p"] == 1 / 64 and flag["eth"]["possibly_contaminated"] is True
    assert not any("recent_is_primary" in f for f in flag.values())
    text = (out / "results.md").read_text(encoding="utf-8")
    pep_line = next(line for line in text.splitlines() if line.startswith("- pep: "))
    eth_line = next(line for line in text.splitlines() if line.startswith("- eth: "))
    assert "floor 0.600 (Draft, 3 of 5 items)" in pep_line and "right on 2 of 5" in pep_line
    assert "possibly contaminated" not in pep_line and "not marked" in pep_line
    assert "possibly contaminated" in eth_line and "p = 0.01562" in eth_line
    assert "F1-change is the primary as-of result" in text
    assert "F1-recent becomes" not in text and "primary as-of result for" not in text
    # Section 10: the floor is over the corpus's F1-change items in the run's scope, those closed
    # book is scored on, which in a pilot run are not the stratum's.
    assert ("A corpus's F1-change floor is the share of its F1-change items in the run's scope, "
            "those closed book is scored on, whose gold is their most common gold value.") in text
    assert "the stratum's most common gold value" not in text
    scoped = tmp_path / "scoped"
    assert run_score(data, [write_answers(tmp_path / "b.jsonl", closed_book_rows(pep, eth))],
                     scoped, items=write_items(tmp_path / "pep.json", pep[2:])) == 0
    assert {k: load(scoped)["contamination"]["pep"][k] for k in (
        "items", "floor_gold", "floor_items")} == {"items": 3, "floor_gold": "Draft",
                                                   "floor_items": 1}
    assert "pep: floor 0.333 (Draft, 1 of 3 items)" in (scoped / "results.md").read_text(
        encoding="utf-8")


def test_without_closed_book_answers_no_corpus_is_tested_and_the_report_says_so(tmp_path):
    data, pep, _, _ = floor_data(tmp_path)
    rows = [answer(pep[0], "bm25", "H+", run, "Draft") for run in (1, 2, 3)]
    out = tmp_path / "out"
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", rows)], out) == 0
    results = load(out)
    assert results["contamination"] == {} and results["closed_book_split"] == []
    text = (out / "results.md").read_text(encoding="utf-8")
    assert "## Closed-book check" in text
    assert "No closed-book answers in this run" in text


def test_a_corpus_whose_f1_change_items_closed_book_never_answered_is_not_tested(tmp_path):
    """Closed book is a system of the run, but no answer of its reaches a corpus's F1-change
    items: that corpus gets no p and no verdict, the report says it was not tested, and no
    system's F1-change accuracy is split by closed book's results there."""
    data, pep, eth, recent = floor_data(tmp_path)
    rows = [r for r in closed_book_rows(pep, eth) if r["id"] in pep]  # no Ethereum answer
    out = tmp_path / "out"
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", rows)], out) == 0
    flag = load(out)["contamination"]
    assert flag["pep"]["arms"] == ["H+", "H-"] and flag["pep"]["possibly_contaminated"] is False
    assert flag["eth"]["arms"] == [] and flag["eth"]["items"] == 6
    assert (flag["eth"]["p"], flag["eth"]["possibly_contaminated"]) == (None, None)
    assert {r["corpus"] for r in load(out)["closed_book_split"]} == {"pep"}
    text = (out / "results.md").read_text(encoding="utf-8")
    assert "- eth: not tested: closed book gave no answer to its F1-change items." in text
    assert "- pep: floor 0.600" in text
    # Declared in an entries file, with answers only outside F1-change: neither corpus is tested.
    only_recent = [answer(recent[0], "closed-book", "H+", run, "Accepted") for run in (1, 2, 3)]
    out2 = tmp_path / "out2"
    assert run_score(data, [write_answers(tmp_path / "b.jsonl", only_recent)], out2,
                     entries={"closed-book": {"families": ["F1"]}}) == 0
    flags = load(out2)["contamination"]
    assert set(flags) == {"pep", "eth"}
    assert all((f["p"], f["possibly_contaminated"]) == (None, None) for f in flags.values())
    assert load(out2)["closed_book_split"] == []
    text = (out2 / "results.md").read_text(encoding="utf-8")
    for corpus in ("pep", "eth"):
        assert f"- {corpus}: not tested: closed book gave no answer to its F1-change items." in text
    assert "not marked" not in text


def test_every_systems_f1_change_accuracy_is_split_by_closed_books_result(tmp_path):
    data, pep, eth, _ = floor_data(tmp_path)
    rows = closed_book_rows(pep, eth)
    # bm25 is right in H+ on the PEP items closed book gets wrong (1, 3, 4) and wrong elsewhere.
    for k, item_id in enumerate(pep):
        rows += [answer(item_id, "bm25", "H+", run, FLOOR_PEP[k] if k in (1, 3, 4) else "Withdrawn")
                 for run in (1, 2, 3)]
    out = tmp_path / "out"
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", rows)], out) == 0
    split = {(r["system"], r["arm"], r["corpus"]): r for r in load(out)["closed_book_split"]}
    bm25 = split[("bm25", "H+", "pep")]
    assert (bm25["closed_book_wrong_items"], bm25["closed_book_wrong_accuracy"]) == (3, 1.0)
    assert (bm25["closed_book_right_items"], bm25["closed_book_right_accuracy"]) == (2, 0.0)
    # Entered in H- too, where it never answered: every item is wrong there.
    assert split[("bm25", "H-", "pep")]["closed_book_wrong_accuracy"] == 0.0
    # Closed book itself is split too; on Ethereum it got every item right.
    eth_row = split[("closed-book", "H-", "eth")]
    assert (eth_row["closed_book_wrong_items"], eth_row["closed_book_right_items"]) == (0, 6)
    assert eth_row["closed_book_wrong_accuracy"] is None
    assert eth_row["closed_book_right_accuracy"] == 1.0
    text = (out / "results.md").read_text(encoding="utf-8")
    assert "on the items closed book gets wrong and on the rest" in text


def test_every_systems_f1_recent_accuracy_is_split_by_whether_the_gold_is_the_status_at_s(
    tmp_path
):
    data, _, _, recent = floor_data(tmp_path)
    same, differs = recent  # the first item's gold is the status at S
    rows = [answer(same, "dense", "H+", run, "Accepted") for run in (1, 2, 3)]
    rows += [answer(differs, "dense", "H+", run, "Final") for run in (1, 2, 3)]  # the status at S
    out = tmp_path / "out"
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", rows)], out) == 0
    split = {(r["system"], r["arm"], r["corpus"]): r for r in load(out)["recent_split"]}
    dense = split[("dense", "H+", "pep")]
    assert (dense["equals_s_items"], dense["equals_s_accuracy"]) == (1, 1.0)
    assert (dense["differs_items"], dense["differs_accuracy"]) == (1, 0.0)
    assert split[("dense", "H-", "pep")]["equals_s_accuracy"] == 0.0  # never answered there
    text = (out / "results.md").read_text(encoding="utf-8")
    assert "## F1-recent by whether the gold is the status at S" in text


def test_an_f1_recent_item_without_its_status_at_s_flag_stops_the_score(tmp_path, capsys):
    data, _, _, recent = floor_data(tmp_path)
    gold = data / "gold.jsonl"
    rows = [json.loads(line) for line in gold.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row["id"] == recent[1]:
            del row["meta"]["gold_equals_status_at_s"]
    gold.write_text("".join(dumps(row) + "\n" for row in rows), encoding="utf-8")
    answers = write_answers(tmp_path / "a.jsonl", [answer(recent[0], "dense", "H+", 1, "Final")])
    assert run_score(data, [answers], tmp_path / "out") == 2
    err = capsys.readouterr().err
    assert recent[1] in err and "gold_equals_status_at_s" in err
    assert not (tmp_path / "out").exists()


def test_score_needs_the_data(tmp_path, capsys):
    path = write_answers(tmp_path / "a.jsonl", [])
    assert run_score(tmp_path / "nowhere", [path], tmp_path / "out") == 2
    assert "asofbench score" in capsys.readouterr().err
    data = small_data(tmp_path)
    assert run_score(data, [tmp_path / "missing.jsonl"], tmp_path / "out") == 2
    assert "missing.jsonl" in capsys.readouterr().err


# --- system identifiers (section 11) -------------------------------------------------------------


def test_the_scorer_knows_the_registered_entrants():
    from asofbench.scoring import REGISTERED_SYSTEMS
    assert REGISTERED_SYSTEMS == ("closed-book", "bm25", "dense", "hybrid", "grep-at-d",
                                  "long-context", "hipporag2", "graphiti", "graphiti-reduced",
                                  "ppr", "oracle")


def test_every_registered_entrant_is_scored(tmp_path):
    """Every entrant is scored; graphiti and graphiti-reduced, which no run holds together
    (section 13), each in a run of its own."""
    from asofbench.scoring import REGISTERED_SYSTEMS
    data = small_data(tmp_path)
    for graphiti, other in (("graphiti", "graphiti-reduced"), ("graphiti-reduced", "graphiti")):
        systems = [system for system in REGISTERED_SYSTEMS if system != other]
        rows = [answer("F1-change-PEP-1-2010-01-01", system, "H+", 1, "Final")
                for system in systems]
        out = tmp_path / graphiti
        assert run_score(data, [write_answers(tmp_path / f"{graphiti}.jsonl", rows)], out) == 0
        assert {r["system"] for r in load(out)["results"]} == set(systems)


def test_an_unregistered_system_stops_the_score_unless_the_entries_declare_it(tmp_path, capsys):
    data = small_data(tmp_path)
    good = "F1-change-PEP-1-2010-01-01"
    rows = [answer(good, "dense", "H+", 1, "Final"), answer(good, "grep_at_d", "H+", 1, "Final"),
            answer(good, "hippo-rag", "H+", 1, "Final"), answer(good, "zzz", "H+", 1, "Final")]
    path = write_answers(tmp_path / "a.jsonl", rows)
    out = tmp_path / "out"
    assert run_score(data, [path], out) == 2
    err = capsys.readouterr().err
    assert "grep_at_d" in err and "did you mean grep-at-d?" in err and "a.jsonl line 2" in err
    assert "hippo-rag" in err and "did you mean hipporag2?" in err
    assert "zzz" in err and "--entries" in err and not out.exists()
    entries = {"grep_at_d": {}, "hippo-rag": {"arms": ["H+"]}, "zzz": {"families": ["F1"]}}
    assert run_score(data, [path], out, entries=entries) == 0
    assert {r["system"] for r in load(out)["results"]} == {"dense", "grep_at_d", "hippo-rag",
                                                            "zzz"}


def test_an_entries_name_that_never_answers_is_a_warning(tmp_path):
    data = small_data(tmp_path)
    rows = [answer("F1-change-PEP-1-2010-01-01", "dense", "H+", run, "Final") for run in (1, 2, 3)]
    path = write_answers(tmp_path / "a.jsonl", rows)
    out = tmp_path / "out"
    entries = {"dense": {}, "graphitti": {"arms": ["H+"]}, "bm25": {"families": ["F1"]}}
    assert run_score(data, [path], out, entries=entries) == 0
    warnings = load(out)["warnings"]
    assert len(warnings) == 2
    assert "bm25" in warnings[0] and "graphitti" in warnings[1]
    text = (out / "results.md").read_text(encoding="utf-8")
    assert "## Warnings" in text and "graphitti" in text
    quiet = tmp_path / "quiet"
    assert run_score(data, [path], quiet, entries={"dense": {}}) == 0
    assert load(quiet)["warnings"] == []
    assert "## Warnings" not in (quiet / "results.md").read_text(encoding="utf-8")


# --- H- message cues (section 12) ----------------------------------------------------------------


def cues_data(root: Path) -> Path:
    items = [
        (question("F1-change-PEP-1-2010-01-01", "F1", "change", "pep", "status", ("H+", "H-"),
                  proposal="PEP-1", as_of="2010-01-01"),
         gold_row("F1-change-PEP-1-2010-01-01", "Final", "adopted")),
        (question("F1-change-PEP-2-2011-01-01", "F1", "change", "pep", "status", ("H+", "H-"),
                  proposal="PEP-2", as_of="2011-01-01"),
         gold_row("F1-change-PEP-2-2011-01-01", "Rejected", "declined")),
        (question("F1-current-EIP-3-2026-10-27", "F1", "current", "eth", "status", ("H+", "H-"),
                  proposal="EIP-3", as_of="2026-10-27"),
         gold_row("F1-current-EIP-3-2026-10-27", "Draft", "open")),
        (question("F2-positive-PEP-5-2026-10-27", "F2", "positive", "pep", "set", ("H+", "H-"),
                  proposal="PEP-5", as_of="2026-10-27"),
         gold_row("F2-positive-PEP-5-2026-10-27", [387])),
    ]
    data = root / "data"
    data.mkdir(parents=True)
    (data / "questions.jsonl").write_text("".join(dumps(q) + "\n" for q, _ in items),
                                          encoding="utf-8")
    (data / "gold.jsonl").write_text("".join(dumps(g) + "\n" for _, g in items), encoding="utf-8")
    return data


CUES = {
    "canary": CANARY,
    "PEP-1": [["2009-12-31T10:00:00Z", "Final"]],  # before the date, the gold status: named
    # Draft is not the gold status, and Rejected comes after the date.
    "PEP-2": [["2010-06-01T00:00:00Z", "Draft"], ["2011-01-02T00:00:00Z", "Rejected"]],
    "EIP-3": [["2026-10-27T23:59:59Z", "Draft"]],  # the last second of the date counts
    "PEP-5": [["2020-01-01T00:00:00Z", "Final"]],  # an F2 item: not read
}


def test_the_message_cues_say_which_h_minus_f1_items_a_message_names(tmp_path):
    data = cues_data(tmp_path)
    rows = [answer("F1-change-PEP-1-2010-01-01", "dense", "H-", run, "Final") for run in (1, 2, 3)]
    out = tmp_path / "out"
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", rows)], out,
                     message_cues=write_items(tmp_path / "cues.json", CUES)) == 0
    results = load(out)
    assert {r["id"]: r["named"] for r in results["message_cue_items"]} == {
        "F1-change-PEP-1-2010-01-01": True, "F1-change-PEP-2-2011-01-01": False,
        "F1-current-EIP-3-2026-10-27": True}
    summary = results["message_cues"]
    assert (summary["file"], summary["items"], summary["named"]) == ("cues.json", 3, 2)
    text = (out / "results.md").read_text(encoding="utf-8")
    assert "## H- items whose gold status a message names" in text
    assert "2 of the 3 H- F1 items" in text
    plain = tmp_path / "plain"  # without the file, no such section, and a line says why
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", rows)], plain) == 0
    assert load(plain)["message_cues"] is None
    assert load(plain)["message_cue_split"] == []
    text = (plain / "results.md").read_text(encoding="utf-8")
    assert "message names" not in text
    assert "H- message cues: no file given (--message-cues)" in text


def test_each_systems_h_minus_f1_accuracy_is_split_by_whether_a_message_names_the_gold(tmp_path):
    """Section 12: H- results read against the messages: each system's F1 accuracy, averaged over
    the three runs, on the items a kept message names with their gold status and on the rest."""
    data = cues_data(tmp_path)
    rows = [answer("F1-change-PEP-1-2010-01-01", "dense", "H-", run, "Final") for run in (1, 2, 3)]
    rows += [answer("F1-current-EIP-3-2026-10-27", "dense", "H-", run, value)
             for run, value in ((1, "Draft"), (2, "Final"), (3, "Final"))]
    rows += [answer("F1-change-PEP-2-2011-01-01", "dense", "H-", run, value)
             for run, value in ((1, "Rejected"), (2, "Rejected"), (3, "Draft"))]
    rows += [answer("F1-change-PEP-2-2011-01-01", "bm25", "H-", run, "Rejected")
             for run in (1, 2, 3)]
    # Answers in another arm are not read.
    rows += [answer("F1-change-PEP-1-2010-01-01", "bm25", "H+", run, "Final") for run in (1, 2, 3)]
    out = tmp_path / "out"
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", rows)], out,
                     message_cues=write_items(tmp_path / "cues.json", CUES)) == 0
    split = {r["system"]: r for r in load(out)["message_cue_split"]}
    assert set(split) == {"bm25", "dense"}
    dense, bm25 = split["dense"], split["bm25"]
    assert (dense["named_items"], dense["other_items"]) == (2, 1)
    assert dense["named_accuracy"] == pytest.approx(2 / 3)  # PEP-1 3 of 3, EIP-3 1 of 3
    assert dense["other_accuracy"] == pytest.approx(2 / 3)  # PEP-2 2 of 3
    # bm25 never answered the named items: they count as wrong, as everywhere else.
    assert (bm25["named_items"], bm25["named_accuracy"]) == (2, 0.0)
    assert (bm25["other_items"], bm25["other_accuracy"]) == (1, 1.0)
    text = (out / "results.md").read_text(encoding="utf-8")
    assert "| System | Named items | Accuracy | Other items | Accuracy |" in text


@pytest.mark.parametrize("content", [
    [1, 2], "text", {"PEP-1": 3}, {"PEP-1": [["2009-12-31T10:00:00Z"]]},
    {"PEP-1": [[1, "Final"]]}, {"PEP-1": [["yesterday", "Final"]]},
])
def test_a_bad_message_cues_file_is_refused(tmp_path, content, capsys):
    data = cues_data(tmp_path)
    rows = [answer("F1-change-PEP-1-2010-01-01", "dense", "H-", 1, "Final")]
    out = tmp_path / "out"
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", rows)], out,
                     message_cues=write_items(tmp_path / "cues.json", content)) == 2
    assert "cues.json" in capsys.readouterr().err and not out.exists()


# --- secondary shares, gates and damaged inputs --------------------------------------------------


def test_a_group_with_no_answers_has_zero_shares_not_none(tmp_path):
    data = small_data(tmp_path)
    rows = [answer("F1-change-PEP-1-2010-01-01", "dense", "H+", run, "Final") for run in (1, 2, 3)]
    out = tmp_path / "out"
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", rows)], out) == 0
    second = {(r["arm"], r["family"], r["stratum"]): r for r in load(out)["secondary"]}
    # Every cell of a group with no answer is wrong, so its own shares are 0; the shares of other
    # answer types stay None.
    f1 = second[("H-", "F1", "change")]
    assert f1["outcome_accuracy"] == 0.0 and f1["invalid_rate"] == 1.0
    assert (f1["precision"], f1["within_one_rate"], f1["mean_abs_error"]) == (None, None, None)
    f2 = second[("H+", "F2", "positive")]
    assert (f2["precision"], f2["recall"], f2["f1"]) == (0.0, 0.0, 0.0)
    assert f2["outcome_accuracy"] is None
    f5 = second[("H+", "F5", "count")]
    assert f5["within_one_rate"] == 0.0 and f5["mean_abs_error"] is None
    assert second[("CF", "CF", "counterfactual")]["outcome_accuracy"] == 0.0
    assert second[("H+", "F1", "change")]["outcome_accuracy"] == 0.5  # answered: as before


def decision_keys(found: object) -> set[str]:
    if isinstance(found, dict):
        return set(found) | {k for v in found.values() for k in decision_keys(v)}
    if isinstance(found, list):
        return {k for v in found for k in decision_keys(v)}
    return set()


def test_a_pilot_that_is_not_evaluable_carries_no_decision_flag_anywhere():
    rows, per_item = pilot_rows(uniform(50), uniform(100))
    del per_item["graphiti"]  # gate 4 cannot be read; gates 1 to 3 do not stop
    result = pilot_gates(rows, per_item)
    assert result["evaluable"] is False and result["missing_inputs"] == [
        "graphiti per-item results"]
    assert not {"stop", "signal", "stopped_at"} & decision_keys(result)
    assert set(result["detail"]) >= {"gate1", "gate2", "gate3", "gate4"}  # the inputs stay


def test_the_majority_reading_of_a_decided_pilot_may_itself_not_be_evaluable(tmp_path,
                                                                             monkeypatch,
                                                                             registered):
    """The gates read the accuracy averaged over the runs; the majority reading beside it can
    differ in evaluability, and then it says so instead of failing."""
    from asofbench import scoring
    data, ids = pilot_data(tmp_path)
    registered(data)
    rows = []
    for system in SYSTEMS:  # 8 of 10 right in every run, 2 right in one run of three
        for k, item_id in enumerate(ids["change"]):
            rows += [answer(item_id, system, "H-", run, "Final" if k < 8 or run == 1 else "Draft")
                     for run in (1, 2, 3)]
        rows += [answer(item_id, system, "H-", run, "Draft") for item_id in ids["current"]
                 for run in (1, 2, 3)]
    real = scoring._majorities
    monkeypatch.setattr(scoring, "_majorities", lambda *args: {
        s: found for s, found in real(*args).items() if s != "graphiti"})
    out = tmp_path / "out"
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", rows)], out,
                     items=data / "pilot.json") == 0
    gates = load(out, "gates.json")
    # Averaged over the runs every system scores 26/30, so gate 1 stops the pilot; on majority
    # results they score 0.8 and gate 4 would need the missing per-item results.
    assert gates["evaluable"] is True and gates["stopped_at"] == 1
    other = gates["detail"]["by_majority_accuracy"]
    assert other["evaluable"] is False and "stopped_at" not in other
    assert "graphiti" in other["decision"]
    text = (out / "results.md").read_text(encoding="utf-8")
    assert "on each item's majority result instead" in text and "not evaluable" in text


@pytest.mark.parametrize(("name", "damage"), [
    ("questions.jsonl", lambda text: text[: len(text) // 2]),  # cut inside a line
    ("gold.jsonl", lambda text: text[: len(text) - 10]),
    ("gold.jsonl", lambda text: text + "[1, 2]\n"),  # a line that is no record
    ("questions.jsonl", lambda text: text + '{"id": "x"}\n'),  # a record without its fields
    ("gold.jsonl", lambda text: text + dumps(gold_row("F9-extra-PEP-9-2020-01-01", "Draft"))
     + "\n"),  # gold for an id that is no question
])
def test_a_damaged_data_file_stops_the_score_with_a_message(tmp_path, name, damage, capsys):
    data = small_data(tmp_path)
    path = data / name
    path.write_text(damage(path.read_text(encoding="utf-8")), encoding="utf-8")
    rows = [answer("F1-change-PEP-1-2010-01-01", "dense", "H+", 1, "Final")]
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", rows)], tmp_path / "out") == 2
    err = capsys.readouterr().err
    assert err.startswith("asofbench score: ") and name in err and "Traceback" not in err


@pytest.mark.parametrize("content", [
    '{"groups": [1, 2]}', '{"groups": [{"family": "F1"}]}',
    '{"groups": [{"family": "F1", "stratum": "change", "ids": "x"}]}', '{"groups": ', "[]",
])
def test_a_damaged_pilot_file_stops_the_score_with_a_message(tmp_path, content, capsys):
    data = small_data(tmp_path)
    (data / "pilot.json").write_text(content, encoding="utf-8")
    rows = [answer("F1-change-PEP-1-2010-01-01", "dense", "H+", 1, "Final")]
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", rows)], tmp_path / "out") == 2
    err = capsys.readouterr().err
    assert "pilot.json" in err and "Traceback" not in err


def test_the_data_files_must_be_the_ones_the_manifest_registers(tmp_path, capsys):
    data = small_data(tmp_path)
    files = {name: {"sha256": hashlib.sha256((data / name).read_bytes()).hexdigest()}
             for name in ("questions.jsonl", "gold.jsonl")}
    (data / "MANIFEST.json").write_text(json.dumps({"files": files}), encoding="utf-8")
    rows = [answer("F1-change-PEP-1-2010-01-01", "dense", "H+", 1, "Final")]
    answers_path = write_answers(tmp_path / "a.jsonl", rows)
    assert run_score(data, [answers_path], tmp_path / "ok") == 0
    gold = data / "gold.jsonl"  # still well formed, but not the registered file
    gold.write_text(gold.read_text(encoding="utf-8").replace('"Final"', '"Draft"', 1),
                    encoding="utf-8")
    assert run_score(data, [answers_path], tmp_path / "edited") == 2
    err = capsys.readouterr().err
    assert "gold.jsonl" in err and "MANIFEST.json" in err and not (tmp_path / "edited").exists()
    (data / "MANIFEST.json").write_text("{cut", encoding="utf-8")
    assert run_score(data, [answers_path], tmp_path / "bad") == 2
    assert "MANIFEST.json" in capsys.readouterr().err


# --- the pilot, end to end -----------------------------------------------------------------------

PILOT_SIZES = {("change", "pep"): 5, ("change", "eth"): 5, ("current", "pep"): 3,
               ("current", "eth"): 3}


def pilot_data(root: Path) -> tuple[Path, dict[str, list[str]]]:
    """A tiny pilot: 10 F1-change items and 6 F1-current items over two corpora."""
    data = root / "data"
    data.mkdir()
    qs, gs, groups = [], [], []
    ids = {"change": [], "current": []}
    for (stratum, corpus), size in PILOT_SIZES.items():
        group = []
        for n in range(size):
            item_id = f"F1-{stratum}-{corpus}{n}-2010-01-01"
            qs.append(question(item_id, "F1", stratum, corpus, "status", ("H+", "H-")))
            gs.append(gold_row(item_id, "Final", "adopted"))
            group.append(item_id)
            ids[stratum].append(item_id)
        groups.append({"family": "F1", "stratum": stratum, "corpus": corpus,
                       "arms": ["H-", "H+"], "ids": group})
    (data / "questions.jsonl").write_text("".join(dumps(q) + "\n" for q in qs), encoding="utf-8")
    (data / "gold.jsonl").write_text("".join(dumps(g) + "\n" for g in gs), encoding="utf-8")
    (data / "pilot.json").write_text(
        dumps({"canary": CANARY, "groups": groups, "ids": ids["change"] + ids["current"]}) + "\n",
        encoding="utf-8")
    return data, ids


def pilot_answers(ids, right: dict[str, tuple[int, int]], systems=SYSTEMS):
    """System s is right on the first right[s][0] change items and right[s][1] current items."""
    rows = []
    for system in systems:
        n_change, n_current = right[system]
        for stratum, count in (("change", n_change), ("current", n_current)):
            for k, item_id in enumerate(ids[stratum]):
                for run in (1, 2, 3):
                    rows.append(answer(item_id, system, "H-", run,
                                       "Final" if k < count else "Draft"))
    return rows


def test_gates_json_when_the_answers_cover_the_pilot(tmp_path, registered):
    data, ids = pilot_data(tmp_path)
    registered(data)
    right = {s: (9, 6) for s in SYSTEMS}
    path = write_answers(tmp_path / "a.jsonl", pilot_answers(ids, right))
    out = tmp_path / "out"
    assert run_score(data, [path], out, items=data / "pilot.json") == 0
    gates = load(out, "gates.json")
    assert gates["evaluable"] is True
    assert gates["stopped_at"] == 1 and gates["signal"] is False  # 9 of 10 everywhere
    coverage = gates["coverage"]
    assert coverage["complete"] is True and coverage["pilot_items"] == 16
    assert {s: c["answered"] for s, c in coverage["systems"].items()} == {s: 16 for s in SYSTEMS}
    assert all(c["missing_count"] == 0 and c["missing"] == [] for c in coverage["systems"].values())
    pilot = load(out)["pilot"]
    assert pilot["evaluable"] is True and pilot["missing_systems"] == []
    # A different scenario: graphiti far ahead of bm25, the rest in between.
    right = {"closed-book": (2, 3), "bm25": (0, 3), "grep-at-d": (5, 4), "long-context": (6, 3),
             "hipporag2": (5, 4), "graphiti": (10, 6)}
    path = write_answers(tmp_path / "b.jsonl", pilot_answers(ids, right))
    out2 = tmp_path / "out2"
    assert run_score(data, [path], out2, items=data / "pilot.json") == 0
    gates = load(out2, "gates.json")
    # 10 of 10 against 0 of 10: ten discordant items, p = 0.002, Holm-adjusted 0.02, 100 points.
    assert gates["stopped_at"] is None and gates["signal"] is True
    assert gates["canary"] == CANARY
    gate4 = gates["detail"]["gate4"]
    assert len(gate4["pairs"]) == 10 and len(gate4["unadjusted"]["pairs"]) == 15
    text = (out2 / "results.md").read_text(encoding="utf-8")
    assert "Holm-adjusted" in text and "Over all fifteen pairs" in text
    assert "decides nothing" in text


def test_the_pilot_section_gives_its_outcome_in_words(tmp_path, registered):
    """results.md says in words where the pilot stops, whether gate 4 finds a signal and the
    decision, never with a Python value, and so does the reading on majority results."""
    from asofbench.scoring import DECISIONS

    data, ids = pilot_data(tmp_path)
    registered(data)
    scenarios = {
        "stopped": ({s: (9, 6) for s in SYSTEMS},
                    "The pilot stops at gate 1, before gate 4, so there is no signal.",
                    DECISIONS[1]),
        "signal": ({"closed-book": (2, 3), "bm25": (0, 3), "grep-at-d": (5, 4),
                    "long-context": (6, 3), "hipporag2": (5, 4), "graphiti": (10, 6)},
                   "No gate stops the pilot, and gate 4 finds a signal.", DECISIONS["signal"]),
        "null": ({s: (6, 6) for s in SYSTEMS},
                 "No gate stops the pilot, and gate 4 finds no signal.", DECISIONS["null"]),
    }
    for name, (right, outcome, decision) in scenarios.items():
        out = tmp_path / name
        assert run_score(data, [write_answers(tmp_path / f"{name}.jsonl",
                                              pilot_answers(ids, right))], out,
                         items=data / "pilot.json") == 0
        section = pilot_section(out)
        assert f'{outcome} The decision is "{decision}".' in section, name
        reading = outcome[0].lower() + outcome[1:-1]
        assert ("Read on each item's majority result instead of the accuracy averaged over the "
                f"runs, {reading}.") in section, name
        prose = [line for line in section.splitlines() if not line.startswith("|")]
        assert not [line for line in prose if re.search(r"\b(?:True|False|None)\b", line)], name


def test_the_pilot_section_says_when_gate_4_was_not_reached(tmp_path, registered):
    """A pilot that stopped at gate 1, 2 or 3 still shows gate 4's reading, for reference only,
    and says that gate 4 was not reached; a pilot that reaches gate 4 says nothing of the kind."""
    data, ids = pilot_data(tmp_path)
    registered(data)
    stopped = tmp_path / "stopped"
    answers = write_answers(tmp_path / "a.jsonl", pilot_answers(ids, {s: (9, 6) for s in SYSTEMS}))
    assert run_score(data, [answers], stopped, items=data / "pilot.json") == 0
    assert load(stopped, "gates.json")["stopped_at"] == 1
    text = (stopped / "results.md").read_text(encoding="utf-8")
    assert ("Gate 4 was not reached (the pilot stopped at gate 1); for reference, gate 4 reads "
            "the ten pairs of pilot systems without closed book") in text
    assert ("| System A | System B | Gap | p | Holm-adjusted p | Gate 4 | Unadjusted reading |"
            in text)
    right = {"closed-book": (2, 3), "bm25": (0, 3), "grep-at-d": (5, 4), "long-context": (6, 3),
             "hipporag2": (5, 4), "graphiti": (10, 6)}
    reached = tmp_path / "reached"
    assert run_score(data, [write_answers(tmp_path / "b.jsonl", pilot_answers(ids, right))],
                     reached, items=data / "pilot.json") == 0
    assert load(reached, "gates.json")["stopped_at"] is None
    text = (reached / "results.md").read_text(encoding="utf-8")
    assert "Gate 4 reads the ten pairs of pilot systems without closed book" in text
    assert "was not reached" not in text


def test_the_pilot_is_not_evaluable_without_all_six_systems(tmp_path):
    data, ids = pilot_data(tmp_path)
    right = {s: (9, 6) for s in SYSTEMS}
    path = write_answers(tmp_path / "a.jsonl", pilot_answers(ids, right, systems=SYSTEMS[:5]))
    out = tmp_path / "out"
    assert run_score(data, [path], out) == 0
    gates = load(out, "gates.json")
    assert gates["evaluable"] is False and gates["missing_systems"] == ["graphiti"]
    assert "stopped_at" not in gates and "signal" not in gates
    assert "not evaluable" in gates["decision"]
    pilot = load(out)["pilot"]
    assert pilot["evaluable"] is False and pilot["missing_systems"] == ["graphiti"]
    text = (out / "results.md").read_text(encoding="utf-8")
    assert "not evaluable" in text and "graphiti" in text
    # A system that answered the pilot items in H+ only has no H- answers, which the gates read.
    rows = pilot_answers(ids, right, systems=SYSTEMS[:5])
    rows += [dict(r, arm="H+") for r in pilot_answers(ids, right, systems=("graphiti",))]
    path = write_answers(tmp_path / "b.jsonl", rows)
    out2 = tmp_path / "out2"
    assert run_score(data, [path], out2) == 0
    assert load(out2, "gates.json")["missing_systems"] == ["graphiti"]


def test_a_missing_pilot_answer_is_wrong_and_the_gates_are_still_decided(tmp_path, registered):
    data, ids = pilot_data(tmp_path)
    registered(data)
    right = {s: (9, 6) for s in SYSTEMS}
    full = pilot_answers(ids, right)
    path = write_answers(tmp_path / "full.jsonl", full)
    complete = tmp_path / "complete"
    assert run_score(data, [path], complete, items=data / "pilot.json") == 0
    assert load(complete, "gates.json")["stopped_at"] == 1  # 9 of 10 everywhere
    # bm25 leaves out one change item and one current item that it would have got right.
    gone = (ids["change"][0], ids["current"][0])
    rows = [r for r in full if not (r["system"] == "bm25" and r["id"] in gone)]
    path = write_answers(tmp_path / "short.jsonl", rows)
    out = tmp_path / "out"
    assert run_score(data, [path], out, items=data / "pilot.json") == 0
    gates = load(out, "gates.json")
    assert gates["evaluable"] is True
    # The omission is a wrong answer: bm25 falls to 8 of 10, under the ceiling, so the ceiling gate
    # no longer stops the pilot; the engines fail alike on current and change.
    assert gates["detail"]["accuracy_change"]["bm25"] == pytest.approx(0.8)
    assert gates["detail"]["accuracy_current"]["bm25"] == pytest.approx(5 / 6)
    assert gates["stopped_at"] == 2 and gates["signal"] is False
    # The coverage is in the file, naming the system and the items.
    coverage = gates["coverage"]
    assert coverage["complete"] is False and coverage["pilot_items"] == 16
    assert coverage["systems"]["bm25"] == {"answered": 14, "asked": 16, "missing_count": 2,
                                           "missing": sorted(gone)}
    assert coverage["systems"]["graphiti"]["missing"] == []
    assert load(out)["pilot"]["evaluable"] is True
    text = (out / "results.md").read_text(encoding="utf-8")
    assert "count as wrong" in text and "bm25" in text
    assert ("Pilot items in the run's scope with no answer count as wrong (section 12). Missing: "
            "bm25 has no answer to 2 of 16 items") in text
    assert all(item_id in text for item_id in gone)
    # A run a system never made is wrong on every item (section 12), and it is flagged.
    no_run = [r for r in full if not (r["system"] == "graphiti" and r["run"] == 3)]
    path = write_answers(tmp_path / "norun.jsonl", no_run)
    out2 = tmp_path / "out2"
    assert run_score(data, [path], out2, items=data / "pilot.json") == 0
    assert load(out2, "gates.json")["evaluable"] is True
    assert {"system": "graphiti", "arm": "H-", "runs": ["3"]} in load(out2)["answers"][
        "missing_runs"]


def test_gates_say_whether_the_measure_matters(tmp_path, registered):
    # Every system is right in runs 1 and 2 and wrong in run 3 on every item: the headline score
    # (accuracy averaged over the runs) is 67%, but each item's majority result is right.
    data, ids = pilot_data(tmp_path)
    registered(data)
    rows = []
    for system in SYSTEMS:
        for item_id in ids["change"] + ids["current"]:
            for run, value in ((1, "Final"), (2, "Final"), (3, "Draft")):
                rows.append(answer(item_id, system, "H-", run, value))
    path = write_answers(tmp_path / "a.jsonl", rows)
    out = tmp_path / "out"
    assert run_score(data, [path], out, items=data / "pilot.json") == 0
    gates = load(out, "gates.json")
    assert gates["detail"]["measure"] == "accuracy_mean_over_runs"
    assert gates["stopped_at"] == 2  # the engines fail alike on current and change
    assert gates["detail"]["by_majority_accuracy"] == {"evaluable": True, "stopped_at": 1,
                                                       "signal": False}
    assert "majority" in (out / "results.md").read_text(encoding="utf-8")


def test_gates_json_follows_the_answers(tmp_path, registered):
    data, ids = pilot_data(tmp_path)
    registered(data)
    right = {s: (9, 6) for s in SYSTEMS}
    out = tmp_path / "out"
    full = write_answers(tmp_path / "full.jsonl", pilot_answers(ids, right))
    assert run_score(data, [full], out, items=data / "pilot.json") == 0
    assert load(out, "gates.json")["evaluable"] is True
    # Scored again with fewer systems, the same directory says so rather than keeping the old file.
    partial = write_answers(tmp_path / "partial.jsonl", pilot_answers(ids, right, SYSTEMS[:3]))
    assert run_score(data, [partial], out, items=data / "pilot.json") == 0
    gates = load(out, "gates.json")
    assert gates["evaluable"] is False and gates["missing_systems"] == list(SYSTEMS[3:])
    # With no pilot in the data there are no gates, and an old file is removed.
    (data / "pilot.json").unlink()
    assert run_score(data, [full], out) == 0
    assert not (out / "gates.json").exists()
    assert load(out)["pilot"] == {"evaluable": False, "reasons": ["no pilot.json"]}


def test_gates_are_deterministic_and_ignore_h_plus_answers(tmp_path, registered):
    data, ids = pilot_data(tmp_path)
    registered(data)
    right = {s: (9, 6) for s in SYSTEMS}
    rows = pilot_answers(ids, right)
    rows += [dict(r, arm="H+", answer="Draft") for r in rows]  # H+ answers decide nothing
    path = write_answers(tmp_path / "a.jsonl", rows)
    one, two = tmp_path / "one", tmp_path / "two"
    assert run_score(data, [path], one, items=data / "pilot.json") == 0
    assert run_score(data, [path], two, items=data / "pilot.json") == 0
    assert (one / "gates.json").read_bytes() == (two / "gates.json").read_bytes()
    assert load(one, "gates.json")["stopped_at"] == 1


def pilot_section(out: Path) -> str:
    return (out / "results.md").read_text(encoding="utf-8").split("\n## Pilot", 1)[1].split(
        "\n## ", 1)[0]


def test_a_pilot_whose_scope_leaves_out_pilot_items_decides_nothing(tmp_path, registered):
    """Section 13: the gates decide only when the run's scope holds every F1 item of pilot.json,
    which they read in H-. A pilot scored on its PEP items alone (--items) gives every gate's
    computation on those items, not evaluable, with the reason and no decision; the
    coverage counts the items in the scope only, so an answer the scope sets aside is never also
    reported missing. An entries file that leaves a pilot system out of H- F1 is the same."""
    data, ids = pilot_data(tmp_path)
    registered(data)
    rows = pilot_answers(ids, {s: (9, 6) for s in SYSTEMS})  # all items, both corpora
    pep = [i for i in ids["change"] + ids["current"] if "-pep" in i]
    out = tmp_path / "pep"
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", rows)], out,
                     items=write_items(tmp_path / "pep.json", pep)) == 0
    gates = load(out, "gates.json")
    assert gates["evaluable"] is False and gates["reasons"] == [
        "the run's scope leaves out 8 pilot items"]
    assert not {"stop", "signal", "stopped_at"} & decision_keys(gates)
    assert gates["decision"].startswith("not evaluable: the run's scope leaves out 8 pilot items")
    assert gates["missing_systems"] == []
    # Every gate's computation is recorded, on the pilot items in the scope: the five PEP
    # F1-change items, all right for every system.
    assert set(gates["detail"]) >= {"gate1", "gate2", "gate3", "gate4"}
    assert gates["detail"]["accuracy_change"] == {s: 1.0 for s in SYSTEMS}
    coverage = gates["coverage"]
    assert coverage["pilot_items"] == 8 and coverage["complete"] is True
    assert {s: (c["answered"], c["asked"], c["missing_count"])
            for s, c in coverage["systems"].items()} == {s: (8, 8, 0) for s in SYSTEMS}
    pilot = load(out)["pilot"]
    assert (pilot["evaluable"], pilot["reasons"]) == (False, gates["reasons"])
    section = pilot_section(out)
    assert ("The pilot is not evaluable, so no gate decides: the run's scope leaves out 8 pilot "
            "items.") in section
    assert "has no answer to" not in section and "count as wrong" not in section
    assert ("Every system with pilot answers answered all 8 pilot items in the run's scope."
            in section)
    assert load(out)["answers"]["out_of_scope_count"] == 6 * 8 * 3  # set aside, listed, not missing
    # An entries file that leaves a pilot system out of H- F1: the gates do not decide, and the
    # system is named for that, not as missing.
    entries = {"bm25": {"arms": ["H+"]}}
    narrowed = tmp_path / "narrowed"
    assert run_score(data, [write_answers(tmp_path / "b.jsonl", rows)], narrowed,
                     entries=entries, items=data / "pilot.json") == 0
    gates = load(narrowed, "gates.json")
    assert gates["reasons"] == ["the entries file leaves bm25 out of H- F1, which the gates read"]
    assert gates["evaluable"] is False and gates["missing_systems"] == []
    assert not {"stop", "signal", "stopped_at"} & decision_keys(gates)
    assert "bm25" not in gates["coverage"]["systems"]
    assert "has no answer to" not in pilot_section(narrowed)


def test_a_pilot_answered_by_another_model_or_with_other_settings_decides_nothing(tmp_path,
                                                                                  registered):
    """Section 13: the gates decide only in the registered pilot run, answered by the registered
    answering model with the registered settings, over stores the registered construction model
    built, with text the registered embedding model embedded. The same pilot answered by
    claude-sonnet-5-5, the sensitivity check, with other settings, over stores another model
    built or with another embedding model records every gate's computation and no decision."""
    data, ids = pilot_data(tmp_path)
    registered(data)
    rows = pilot_answers(ids, {s: (9, 6) for s in SYSTEMS})
    for name, models, reasons in (
        ("sonnet", {"answering_model": "claude-sonnet-5-5"},
         ["answered by claude-sonnet-5-5, not the registered answering model"]),
        ("settings", {"answering_settings": "effort high, adaptive thinking"},
         ['answered with "effort high, adaptive thinking", not the registered answering '
          'settings']),
        ("both", {"answering_model": "claude-sonnet-5-5", "answering_settings": "effort high"},
         ["answered by claude-sonnet-5-5, not the registered answering model",
          'answered with "effort high", not the registered answering settings']),
        ("construction", {"construction_model": "claude-sonnet-5-5"},
         ["built by claude-sonnet-5-5, not the registered construction model"]),
        ("embedding", {"embedding_model": "voyageai/voyage-4-lite"},
         ["embedded with voyageai/voyage-4-lite, not the registered embedding model"]),
    ):
        out = tmp_path / name
        assert score_as(data, rows, out, "--items", str(data / "pilot.json"), **models) == 0, name
        gates = load(out, "gates.json")
        assert (gates["evaluable"], gates.get("stopped_at", "none")) == (False, "none"), name
        assert gates["reasons"] == reasons, name
        assert not {"stop", "signal", "stopped_at"} & decision_keys(gates), name
        assert gates["answering_model"] == models.get("answering_model", ANSWERING), name
        assert set(gates["detail"]) >= {"gate1", "gate2", "gate3", "gate4"}, name
        assert gates["detail"]["accuracy_change"] == {s: 0.9 for s in SYSTEMS}, name
        assert f"so no gate decides: {'; '.join(reasons)}." in pilot_section(out), name


def with_items_beside_the_pilot(data: Path, count: int) -> list[str]:
    """Adds ``count`` F1-change PEP items that pilot.json does not list to the build in ``data``,
    as the registered build holds items beside its pilot list, and returns their ids."""
    beside = [f"F1-change-pep{100 + n}-2010-01-01" for n in range(count)]
    with (data / "questions.jsonl").open("a", encoding="utf-8") as f:
        f.writelines(dumps(question(i, "F1", "change", "pep", "status", ("H+", "H-"))) + "\n"
                     for i in beside)
    with (data / "gold.jsonl").open("a", encoding="utf-8") as f:
        f.writelines(dumps(gold_row(i, "Final", "adopted")) + "\n" for i in beside)
    return beside


def test_only_a_run_of_exactly_the_pilot_items_is_the_registered_pilot_run(tmp_path, registered):
    """Section 13: the registered pilot run's scope is exactly the items of pilot.json, given with
    --items: --items pilot.json, or a list of the same ids in any order, decides, with no reason.
    On a build that holds items beside its pilot list, as the registered one does, a run of every
    item, a wider list and a list that swaps a pilot item for another are not the registered pilot
    run: each records every gate's computation, on the pilot items in its scope, with its reasons
    and no decision. A run with no --items is never the registered pilot run, even on a build whose
    every item is a pilot item, so a full run never decides the gates."""
    from asofbench.scoring import REGISTERED_PILOT

    data, ids = pilot_data(tmp_path)
    beside = with_items_beside_the_pilot(data, 3)
    registered(data)
    rows = pilot_answers(ids, {s: (9, 6) for s in SYSTEMS})  # the pilot items only
    pilot = ids["change"] + ids["current"]
    runs = {
        "listed": ("--items", str(data / "pilot.json")),
        "reordered": ("--items", str(write_items(tmp_path / "reordered.json", pilot[::-1]))),
        "every": (),
        "wider": ("--items", str(write_items(tmp_path / "wider.json", [*pilot, beside[0]]))),
        "swapped": ("--items", str(write_items(tmp_path / "swapped.json",
                                                [*pilot[1:], beside[0]]))),
    }
    for name, extra in runs.items():
        assert score_as(data, rows, tmp_path / name, *extra) == 0, name
    gates = {name: load(tmp_path / name, "gates.json") for name in runs}
    listed = gates["listed"]
    assert (listed["evaluable"], listed["stopped_at"], listed["signal"]) == (True, 1, False)
    assert listed["reasons"] == [] and load(tmp_path / "listed")["pilot"]["reasons"] == []
    assert "so no gate decides" not in pilot_section(tmp_path / "listed")
    assert gates["reordered"] == listed  # the same ids in another order: the same scope
    holds = "items that pilot.json does not list"
    reasons = {
        "every": [f"the run's scope is every item (no --items), which holds 3 {holds}"],
        "wider": ["the run's scope holds 1 item that pilot.json does not list"],
        "swapped": ["the run's scope leaves out 1 pilot item",
                    "the run's scope holds 1 item that pilot.json does not list"],
    }
    for name, why in reasons.items():
        found = gates[name]
        assert (found["evaluable"], found["reasons"]) == (False, why), name
        assert found["decision"] == f"not evaluable: {'; '.join(why)}; {REGISTERED_PILOT}", name
        assert not {"stop", "signal", "stopped_at"} & decision_keys(found), name
        assert load(tmp_path / name)["pilot"]["reasons"] == why, name
        assert f"so no gate decides: {'; '.join(why)}." in pilot_section(tmp_path / name), name
    # Every pilot item is in the scope of a run of every item and of the wider list, so each
    # records the computation of the registered pilot run on the same answers, deciding nothing.
    for name in ("every", "wider"):
        assert gates[name]["detail"]["accuracy_change"] == listed["detail"]["accuracy_change"]
    # A build whose every item is a pilot item: a run with no --items is still not the pilot run.
    (tmp_path / "alone").mkdir()
    alone, _ = pilot_data(tmp_path / "alone")
    registered(alone)
    assert score_as(alone, rows, tmp_path / "alone-every") == 0
    assert load(tmp_path / "alone-every", "gates.json")["reasons"] == [
        "the run's scope is every item (no --items)"]
    assert score_as(alone, rows, tmp_path / "alone-listed", "--items",
                    str(alone / "pilot.json")) == 0
    assert load(tmp_path / "alone-listed", "gates.json")["evaluable"] is True


def test_a_build_that_is_not_the_registered_one_neither_claims_nor_decides(tmp_path, registered):
    """Sections 12 and 13: a run is the registered full run, and a pilot run decides, only on the
    registered build, whose questions, gold and pilot list have the SHA-256s in REGISTERED_DATA
    (data/v0.1's). On a small build left unmarked, a full run makes every comparison and claims
    nothing, and a pilot run (--items pilot.json) has every pilot answer and decides nothing, each
    saying why; marked as the registered build, the same answers claim in the full run and decide
    in the pilot run. A build without one of the three files names it."""
    data, ids = pilot_data(tmp_path)
    rows = pilot_answers(ids, {s: (9, 6) for s in SYSTEMS})
    rows += like_grep(rows, ("dense", "hybrid", "ppr"))  # every compared system answers
    listed = ("--items", str(data / "pilot.json"))
    reason = ("the build is not the registered one: questions.jsonl, gold.jsonl and pilot.json "
              "differ from the registered files by SHA-256")
    plain, plain_pilot = tmp_path / "plain", tmp_path / "plain-pilot"
    assert score_as(data, rows, plain) == 0 and score_as(data, rows, plain_pilot, *listed) == 0
    headline, gates = load(plain)["headline"], load(plain_pilot, "gates.json")
    assert (headline["registered"], headline["not_registered"]) == (False, [reason])
    assert len(headline["comparisons"]) == 14
    assert not any("claim" in c for c in headline["comparisons"])
    assert reason in headline_intro(plain)
    assert (gates["evaluable"], gates["reasons"]) == (False, [reason])
    assert not {"stop", "signal", "stopped_at"} & decision_keys(gates)
    assert f"so no gate decides: {reason}." in pilot_section(plain_pilot)
    registered(data)
    marked, marked_pilot = tmp_path / "marked", tmp_path / "marked-pilot"
    assert score_as(data, rows, marked) == 0 and score_as(data, rows, marked_pilot, *listed) == 0
    assert (load(marked)["headline"]["registered"],
            load(marked_pilot, "gates.json")["stopped_at"]) == (True, 1)
    (data / "pilot.json").unlink()
    absent = tmp_path / "absent"
    assert score_as(data, rows, absent) == 0
    assert load(absent)["headline"]["not_registered"] == [
        "the build is not the registered one: pilot.json is absent"]


def test_a_pilot_that_cannot_be_decided_has_one_shape_whatever_the_cause(tmp_path, registered):
    """One constructor writes every pilot that cannot be decided: a pilot system without answers
    and a run that is not the registered pilot run give gates.json the same keys, each list of
    systems joined as a list is written, and no decision flag anywhere."""
    data, ids = pilot_data(tmp_path)
    registered(data)
    rows = pilot_answers(ids, {s: (9, 6) for s in SYSTEMS})
    missing, sonnet = tmp_path / "missing", tmp_path / "sonnet"
    listed = ("--items", str(data / "pilot.json"))
    assert score_as(data, [r for r in rows if r["system"] not in ("bm25", "graphiti")],
                    missing, *listed) == 0
    assert score_as(data, rows, sonnet, *listed, answering_model="claude-sonnet-5-5") == 0
    one, two = load(missing, "gates.json"), load(sonnet, "gates.json")
    assert set(one) == set(two)
    assert {"measure", "accuracy_change", "accuracy_current"} <= set(one["detail"]) & set(
        two["detail"])
    assert one["reasons"] == ["no H- F1 answers from bm25 and graphiti"]
    assert one["decision"] == ("not evaluable: no H- F1 answers from bm25 and graphiti; the gates "
                               "need all six pilot systems and their inputs")
    assert two["decision"].startswith("not evaluable: answered by claude-sonnet-5-5, not the "
                                      "registered answering model; the gates decide only in the "
                                      "registered pilot run")
    assert not {"stop", "signal", "stopped_at"} & (decision_keys(one) | decision_keys(two))


def test_gates_json_gives_its_decision_at_the_top_level_in_every_shape(tmp_path, registered):
    """Section 13: gates.json's ``decision`` is a top-level field whether the pilot is decided or
    not evaluable, so a script that reads ``gates["decision"]`` finds it in every run; a decided
    pilot keeps no second copy under ``detail``, and results.md quotes the same decision."""
    from asofbench.scoring import DECISIONS

    data, ids = pilot_data(tmp_path)
    registered(data)
    rows = pilot_answers(ids, {s: (9, 6) for s in SYSTEMS})  # 9 of 10 everywhere: gate 1 stops
    listed = ("--items", str(data / "pilot.json"))
    decided, sonnet, missing = tmp_path / "decided", tmp_path / "sonnet", tmp_path / "missing"
    assert score_as(data, rows, decided, *listed) == 0
    assert score_as(data, rows, sonnet, *listed, answering_model="claude-sonnet-5-5") == 0
    assert score_as(data, [r for r in rows if r["system"] != "bm25"], missing, *listed) == 0
    shapes = {out.name: load(out, "gates.json") for out in (decided, sonnet, missing)}
    assert (shapes["decided"]["evaluable"], shapes["decided"]["decision"]) == (True, DECISIONS[1])
    assert "decision" not in shapes["decided"]["detail"]
    assert f'The decision is "{DECISIONS[1]}".' in pilot_section(decided)
    for name in ("sonnet", "missing"):
        assert shapes[name]["evaluable"] is False, name
        assert shapes[name]["decision"].startswith("not evaluable: "), name
        assert "decision" not in shapes[name]["detail"], name


def test_the_pilot_record_gives_its_reasons_as_a_list_in_every_shape(tmp_path, registered):
    """results.json's ``pilot`` says why the pilot is not evaluable with ``reasons``, a list,
    whether the build has no pilot list, has one without F1 items, or has one that cannot be
    decided; results.md gives the reason."""
    data, ids = headline_data(tmp_path)  # no pilot list
    rows = headline_answers(ids)
    none = tmp_path / "none"
    assert score_as(data, rows, none) == 0
    assert load(none)["pilot"] == {"evaluable": False, "reasons": ["no pilot.json"]}
    assert "There is no pilot to decide (no pilot.json)." in pilot_section(none)
    registered(data)  # gives the build an empty pilot list
    empty = tmp_path / "empty"
    assert score_as(data, rows, empty) == 0
    assert load(empty)["pilot"] == {"evaluable": False, "reasons": ["no F1 items in pilot.json"]}
    assert "There is no pilot to decide (no F1 items in pilot.json)." in pilot_section(empty)


# --- the headline comparison (section 12) --------------------------------------------------------

HEADLINE_N = 12  # H- F1-change items per corpus
RIGHT, WRONG = (True, True, True), (False, False, False)


def headline_data(root: Path, flagged: int = 6,
                  sizes: dict[str, int] | None = None) -> tuple[Path, dict[str, list[str]]]:
    """H- F1-change items of both corpora (``HEADLINE_N`` each, or ``sizes``), all with gold
    Final; the first ``flagged`` of each corpus have a header-only H- episode after their date.
    One F1-current item, which the headline does not read, is there too."""
    data = root / "data"
    data.mkdir(parents=True)
    qs, gs, ids = [], [], {"pep": [], "eth": []}
    for corpus, series in (("pep", "PEP"), ("eth", "EIP")):
        for k in range((sizes or {}).get(corpus, HEADLINE_N)):
            item_id = f"F1-change-{series}-{k}-2010-01-01"
            qs.append(question(item_id, "F1", "change", corpus, "status", ("H+", "H-"),
                               proposal=f"{series}-{k}", as_of="2010-01-01"))
            gs.append(gold_row(item_id, "Final", "adopted",
                               h_minus_header_only_after_d=k < flagged))
            ids[corpus].append(item_id)
    current = "F1-current-PEP-99-2026-10-27"
    qs.append(question(current, "F1", "current", "pep", "status", ("H+", "H-"),
                       proposal="PEP-99", as_of="2026-10-27"))
    gs.append(gold_row(current, "Final", "adopted"))
    (data / "questions.jsonl").write_text("".join(dumps(q) + "\n" for q in qs), encoding="utf-8")
    (data / "gold.jsonl").write_text("".join(dumps(g) + "\n" for g in gs), encoding="utf-8")
    return data, ids


def say(system: str, item_id: str, runs) -> list[dict]:
    """``system``'s H- answers to ``item_id``: right (True), wrong (False) or no answer (None) in
    runs 1, 2 and 3."""
    return [answer(item_id, system, "H-", run, "Final" if right else "Draft")
            for run, right in zip((1, 2, 3), runs, strict=True) if right is not None]


# Graphiti on the PEP items: right in runs 1 and 2 with no run 3 (a majority, since a missing run is
# wrong), then right in runs 1 and 3 with no run 2 (a majority), then right in run 1 alone (not a
# majority, though it is every run answered), then wrong.
GRAPHITI_PEP = [(True, True, None)] * 4 + [(True, None, True), (True, None, None)] + [WRONG] * 6


def headline_answers(ids) -> list[dict]:
    """Grep at D is right on PEP items 0 to 3 and on every Ethereum item; bm25 is right on every
    PEP item and wrong on every Ethereum one; long context is right on PEP items 0 to 9 and on
    every Ethereum item; Graphiti answers the PEP items only. Closed book and the oracle answer
    too, and are never compared."""
    rows = []
    for k, item_id in enumerate(ids["pep"]):
        rows += say("grep-at-d", item_id, RIGHT if k < 4 else WRONG)
        rows += say("bm25", item_id, RIGHT)
        rows += say("long-context", item_id, RIGHT if k < 10 else WRONG)
        rows += say("graphiti", item_id, GRAPHITI_PEP[k])
        rows += say("closed-book", item_id, RIGHT) + say("oracle", item_id, RIGHT)
    for item_id in ids["eth"]:
        rows += say("grep-at-d", item_id, RIGHT) + say("bm25", item_id, WRONG)
        rows += say("long-context", item_id, RIGHT) + say("closed-book", item_id, WRONG)
    return rows


def comparisons_of(reading: dict) -> dict[tuple[str, str], dict]:
    return {(c["system"], c["corpus"]): c for c in reading["comparisons"]}


# Section 12: why a comparison is not made, as results.json and results.md write it. A comparison
# is made only when the compared system and grep at D both have a scored answer on the corpus's
# H- F1-change items in the scope (in a reading, on the corpus's items left in it).
NO_ANSWER = "no scored answer to the corpus's H- F1-change items"
NO_ANSWER_LEFT = "no scored answer to the corpus's items left in this reading"
LEFT_OUT = "left out of H- F1 by the entries file"
BASELINE_NO_ANSWER = f"grep-at-d, the baseline, has {NO_ANSWER}"


def test_the_headline_compares_each_context_system_with_grep_at_d_per_corpus(tmp_path, registered):
    """Section 12: each system that keeps or retrieves context, against grep at D, on the H-
    F1-change items of each corpus; accuracy averaged over the runs and its difference; the exact
    two-sided McNemar test on majority results (a missing run is wrong); Holm over the 14
    registered comparisons, a comparison the run does not make counting as p = 1; and the
    reading, in a run that is not the registered full run: above or below grep at D when the
    Holm-adjusted p is below 0.05, by which side is right where the other is wrong on more items,
    and otherwise no detectable difference."""
    data, ids = headline_data(tmp_path)
    registered(data)
    out = tmp_path / "out"
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", headline_answers(ids))], out) == 0
    headline = load(out)["headline"]
    # Four compared systems have no answer, and Graphiti none on Ethereum, so the run makes 5 of
    # the 14 comparisons and is not the registered full run; Graphiti also leaves runs out of the
    # PEP items it is compared on.
    assert headline["registered"] is False
    assert headline["not_registered"] == [
        f"dense, hybrid, ppr and hipporag2 are not compared on pep or eth ({NO_ANSWER})",
        f"graphiti is not compared on eth ({NO_ANSWER})",
        "graphiti answered 29 of the 36 runs on the 12 compared pep items"]
    assert headline["baseline"] == "grep-at-d"
    assert (headline["arm"], headline["family"], headline["stratum"]) == ("H-", "F1", "change")
    assert headline["systems"] == ["bm25", "dense", "hybrid", "ppr", "hipporag2", "graphiti",
                                   "long-context"]
    assert headline["items"] == {"pep": 12, "eth": 12}  # the F1-current item is not read
    found = comparisons_of(headline)
    # Closed book and the oracle answered but are never compared; a system with no scored answer
    # on a corpus's items is not compared there.
    assert list(found) == [("bm25", "pep"), ("bm25", "eth"), ("graphiti", "pep"),
                           ("long-context", "pep"), ("long-context", "eth")]
    bm25 = found[("bm25", "pep")]
    assert (bm25["items"], bm25["accuracy"], bm25["baseline_accuracy"]) == (12, 1.0,
                                                                           pytest.approx(1 / 3))
    assert bm25["difference"] == pytest.approx(2 / 3)
    assert (bm25["only_system_right"], bm25["only_baseline_right"]) == (8, 0)
    assert bm25["p"] == mcnemar_exact(8, 0) == 2 / 256
    # Graphiti: 11 right runs of 36, and a majority on PEP items 0 to 4 only, as grep at D's on
    # 0 to 3. It has no answer on Ethereum, where it is entered, so it is not compared there.
    graphiti = found[("graphiti", "pep")]
    assert graphiti["accuracy"] == pytest.approx(11 / 36)
    assert graphiti["difference"] == pytest.approx(11 / 36 - 12 / 36)
    assert (graphiti["only_system_right"], graphiti["only_baseline_right"], graphiti["p"]) == (
        1, 0, 1.0)
    long_pep = found[("long-context", "pep")]
    assert (long_pep["only_system_right"], long_pep["only_baseline_right"]) == (6, 0)
    assert long_pep["p"] == 1 / 32  # below 0.05 before the adjustment
    # Holm over the 14 registered comparisons, the nine this run does not make at p = 1,
    # smallest first: 14 times 2/4096, 13 times 2/256, 12 times 1/32, and the p-values of 1.
    holm_p = {key: c["p_holm"] for key, c in found.items()}
    assert holm_p == {("bm25", "pep"): 13 * 2 / 256,
                      ("bm25", "eth"): 14 * 2 / 4096,
                      ("graphiti", "pep"): 1.0,
                      ("long-context", "pep"): 12 / 32,
                      ("long-context", "eth"): 1.0}
    # Not the registered full run: each row gives a reading by the headline's rule, not a claim.
    assert all("claim" not in c for c in found.values())
    assert {key: c["reading"] for key, c in found.items()} == {
        ("bm25", "pep"): "no detectable difference", ("bm25", "eth"): "below grep at D",
        ("graphiti", "pep"): "no detectable difference",
        ("long-context", "pep"): "no detectable difference",
        ("long-context", "eth"): "no detectable difference"}
    # The two corpora pooled, descriptively: no test, no claim.
    pooled = {p["system"]: p for p in headline["pooled"]}
    assert set(pooled) == {"bm25", "graphiti", "long-context"}
    assert pooled["bm25"]["items"] == 24 and pooled["bm25"]["corpora"] == ["pep", "eth"]
    assert pooled["bm25"]["accuracy"] == 0.5
    assert pooled["bm25"]["baseline_accuracy"] == pytest.approx(2 / 3)
    assert pooled["bm25"]["difference"] == pytest.approx(0.5 - 2 / 3)
    assert (pooled["graphiti"]["corpora"], pooled["graphiti"]["items"]) == (["pep"], 12)
    assert pooled["graphiti"]["difference"] == pytest.approx(11 / 36 - 12 / 36)
    assert pooled["long-context"]["difference"] == pytest.approx(66 / 72 - 48 / 72)
    assert not any("p" in p or "claim" in p or "reading" in p for p in pooled.values())
    # The comparisons not made are said, with why.
    assert headline["not_made"] == [
        *({"system": system, "corpus": corpus, "reason": NO_ANSWER}
          for system in ("dense", "hybrid", "ppr", "hipporag2") for corpus in ("pep", "eth")),
        {"system": "graphiti", "corpus": "eth", "reason": NO_ANSWER}]


def test_the_headline_section_follows_the_primary_results(tmp_path):
    data, ids = headline_data(tmp_path)
    out = tmp_path / "out"
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", headline_answers(ids))], out) == 0
    text = (out / "results.md").read_text(encoding="utf-8")
    headings = [line for line in text.splitlines() if line.startswith("## ")]
    assert headings[0].startswith("## Primary results (")
    assert headings[1].startswith("## Headline comparison (")
    section = text.split("\n## Headline comparison", 1)[1].split("\n## ", 1)[0]
    intro = headline_intro(out)
    assert "in the run's scope (every item): 12 PEP items and 12 Ethereum items." in intro
    assert ("Holm-adjusted over the 14 registered comparisons (each compared system on each "
            "corpus), in which a comparison the run does not make counts as p = 1; this run "
            "makes 5 comparisons.") in intro
    # Not the registered full run (it makes 5 of the 14 comparisons): its rows are readings.
    assert ("| System | Corpus | Items | Accuracy | Grep at D's accuracy | Difference | System "
            "right, grep at D wrong | Grep at D right, system wrong | p | Holm-adjusted p | "
            "Reading |") in section
    assert "| Claim |" not in section
    assert ("| bm25 | pep | 12 | 1.000 | 0.333 | 0.667 | 8 | 0 | 0.007812 | 0.1016 | no "
            "detectable difference |") in section
    assert ("| long-context | pep | 12 | 0.833 | 0.333 | 0.500 | 6 | 0 | 0.03125 | 0.375 | no "
            "detectable difference |") in section
    assert "| System | Items | Accuracy | Grep at D's accuracy | Difference |" in section
    assert "| bm25 | 24 | 0.500 | 0.667 | -0.167 |" in section
    assert "not tested" in section
    # One not-made line per cause and set of corpora, naming the systems as the tables do.
    lines = section.splitlines()
    assert (f"- Not compared on pep or eth ({NO_ANSWER}): dense, hybrid, ppr (author's own) and "
            "hipporag2.") in lines
    assert f"- Not compared on eth ({NO_ANSWER}): graphiti." in lines
    # No direction line for a comparison with no detectable difference, whatever the sign of its
    # run-averaged difference: long context on PEP (+0.500) and on Ethereum (0.000) have none.
    assert ("| long-context | eth | 12 | 1.000 | 1.000 | 0.000 | 0 | 0 | 1 | 1 | no detectable "
            "difference |") in section
    assert not [line for line in section.splitlines()
                if line.startswith("- ") and "follows the majority results" in line]


def test_the_headline_reads_the_same_comparisons_without_cued_items_beside_it(tmp_path):
    """Section 12: beside the headline, deciding nothing, the same comparisons on the items
    without a header-only H- episode after the date, and, given --message-cues, on the items
    whose gold status no kept message names by the date."""
    data, ids = headline_data(tmp_path)
    answers = write_answers(tmp_path / "a.jsonl", headline_answers(ids))
    out = tmp_path / "out"
    assert run_score(data, [answers], out) == 0
    headline = load(out)["headline"]
    plain = headline["readings"]["without_header_only_episode"]
    assert plain["items"] == {"pep": 6, "eth": 6}  # items 6 to 11 of each corpus
    found = comparisons_of(plain)
    assert (found[("bm25", "pep")]["only_system_right"], found[("bm25", "pep")]["p"]) == (6, 1 / 32)
    # Holm within the reading, over its own 14 registered comparisons: three p-values of 1/32
    # come first, so the smallest is multiplied by 14.
    assert found[("bm25", "pep")]["p_holm"] == 14 / 32
    assert all(c["reading"] == "no detectable difference" and "claim" not in c
               for c in found.values())  # a reading decides nothing, so it makes no claim
    assert headline["readings"]["without_message_cue"] is None  # no --message-cues
    text = (out / "results.md").read_text(encoding="utf-8")
    section = text.split("\n## Headline comparison", 1)[1].split("\n## ", 1)[0]
    assert "decide nothing" in section
    assert "no --message-cues file was given" in section
    # With the cues: PEP-0 to PEP-2 have a kept message naming Final by their date; EIP-0's
    # message comes after its date.
    cues = {"canary": CANARY, "EIP-0": [["2010-01-02T00:00:00Z", "Final"]],
            **{f"PEP-{k}": [["2009-12-31T00:00:00Z", "Final"]] for k in range(3)}}
    cued = tmp_path / "cued"
    assert run_score(data, [answers], cued,
                     message_cues=write_items(tmp_path / "cues.json", cues)) == 0
    reading = load(cued)["headline"]["readings"]["without_message_cue"]
    assert reading["items"] == {"pep": 9, "eth": 12}
    found = comparisons_of(reading)
    assert (found[("bm25", "pep")]["items"], found[("bm25", "pep")]["only_system_right"]) == (9, 8)
    assert found[("graphiti", "pep")]["accuracy"] == pytest.approx(5 / 27)
    assert load(cued)["headline"]["comparisons"] == headline["comparisons"]  # the headline itself
    text = (cued / "results.md").read_text(encoding="utf-8")
    assert "no --message-cues file was given" not in text


def test_the_headline_says_which_comparisons_were_not_made(tmp_path):
    data, ids = headline_data(tmp_path)
    rows = headline_answers(ids)
    # Without grep at D, nothing is compared, and a visible line says why.
    out = tmp_path / "no-grep"
    without = [r for r in rows if r["system"] != "grep-at-d"]
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", without)], out) == 0
    headline = load(out)["headline"]
    assert headline["comparisons"] == [] and headline["pooled"] == []
    assert {n["reason"] for n in headline["not_made"]} == {BASELINE_NO_ANSWER}
    assert len(headline["not_made"]) == 14
    section = (out / "results.md").read_text(encoding="utf-8").split(
        "\n## Headline comparison", 1)[1].split("\n## ", 1)[0]
    # One line for the one cause, though it removes all 14 comparisons.
    assert [line for line in section.splitlines() if line.startswith("- Not compared")] == [
        f"- Not compared on pep or eth ({BASELINE_NO_ANSWER}): bm25, dense, hybrid, ppr (author's "
        "own), hipporag2, graphiti and long-context."]
    # The intro says that no comparison is made, not that 0 comparisons were.
    assert "; this run makes no comparison." in headline_intro(out)
    assert "0 comparisons" not in section and "over the 0" not in section
    # A system an entries file narrows out of H- F1 is not compared, and the line says why.
    narrowed = tmp_path / "narrowed"
    assert run_score(data, [write_answers(tmp_path / "b.jsonl", rows)], narrowed,
                     entries={"bm25": {"arms": ["H+"]}}) == 0
    headline = load(narrowed)["headline"]
    assert [(n["system"], n["reason"]) for n in headline["not_made"] if n["system"] == "bm25"] == [
        ("bm25", LEFT_OUT), ("bm25", LEFT_OUT)]
    assert ("bm25", "pep") not in comparisons_of(headline)
    # A scope with PEP items only: no Ethereum comparison.
    pep_only = tmp_path / "pep-only"
    assert run_score(data, [write_answers(tmp_path / "c.jsonl", rows)], pep_only,
                     items=write_items(tmp_path / "pep.json", ids["pep"])) == 0
    headline = load(pep_only)["headline"]
    assert headline["items"] == {"pep": 12, "eth": 0}
    assert {c["corpus"] for c in headline["comparisons"]} == {"pep"}
    assert {(n["corpus"], n["reason"]) for n in headline["not_made"] if n["corpus"] == "eth"} == {
        ("eth", "no H- F1-change item of the corpus is in the run's scope")}
    # A scope with no H- F1-change item at all.
    none = tmp_path / "none"
    assert run_score(data, [write_answers(tmp_path / "d.jsonl", rows)], none,
                     items=write_items(tmp_path / "current.json",
                                       ["F1-current-PEP-99-2026-10-27"])) == 0
    headline = load(none)["headline"]
    assert headline["comparisons"] == [] and headline["items"] == {"pep": 0, "eth": 0}
    section = (none / "results.md").read_text(encoding="utf-8").split(
        "\n## Headline comparison", 1)[1].split("\n## ", 1)[0]
    assert "no H- F1-change item is in the run's scope" in section


def test_results_json_gives_every_items_runs_so_that_any_comparison_can_be_recomputed(tmp_path):
    """Section 12: for every system, arm and scored item, the runs right, the runs answered and
    the majority result, answered or not."""
    data, ids = headline_data(tmp_path)
    out = tmp_path / "out"
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", headline_answers(ids))], out) == 0
    results = load(out)
    items = results["items"]
    assert set(items) == {"bm25", "closed-book", "graphiti", "grep-at-d", "long-context", "oracle"}
    graphiti = items["graphiti"]["H-"]
    assert graphiti["F1-change-PEP-0-2010-01-01"] == {"runs_right": 2, "runs_answered": 2,
                                                      "majority_correct": True}
    assert graphiti["F1-change-PEP-4-2010-01-01"] == {"runs_right": 2, "runs_answered": 2,
                                                      "majority_correct": True}
    assert graphiti["F1-change-PEP-5-2010-01-01"] == {"runs_right": 1, "runs_answered": 1,
                                                      "majority_correct": False}
    assert graphiti["F1-change-EIP-0-2010-01-01"] == {"runs_right": 0, "runs_answered": 0,
                                                      "majority_correct": False}
    # Every item the system is scored on is there, in every arm it is entered in.
    assert set(graphiti) == {*ids["pep"], *ids["eth"], "F1-current-PEP-99-2026-10-27"}
    assert set(items["graphiti"]) == {"H+", "H-"}
    # The headline's discordant counts follow from these records.
    mine, theirs = items["bm25"]["H-"], items["grep-at-d"]["H-"]
    only_bm25 = sum(1 for i in ids["pep"] if mine[i]["majority_correct"]
                    and not theirs[i]["majority_correct"])
    assert only_bm25 == comparisons_of(results["headline"])[("bm25", "pep")]["only_system_right"]


def test_every_systems_h_minus_f1_accuracy_is_split_by_a_header_only_episode_after_the_date(
    tmp_path
):
    """Section 12: an H- episode whose diff the withholding empties shows a stream system that a
    withheld header changed; every system's H- F1 accuracy is reported apart for the items whose
    proposal has one after their date (gold ``meta.h_minus_header_only_after_d``)."""
    data, ids = headline_data(tmp_path)  # the first six items of each corpus are flagged
    out = tmp_path / "out"
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", headline_answers(ids))], out) == 0
    split = {(r["system"], r["stratum"], r["corpus"]): r for r in load(out)["header_only_split"]}
    grep = split[("grep-at-d", "change", "pep")]  # right on items 0 to 3 only
    assert (grep["flagged_items"], grep["flagged_accuracy"]) == (6, pytest.approx(4 / 6))
    assert (grep["other_items"], grep["other_accuracy"]) == (6, 0.0)
    graphiti = split[("graphiti", "change", "pep")]  # 11 right runs of 18 on items 0 to 5
    assert graphiti["flagged_accuracy"] == pytest.approx(11 / 18)
    assert split[("bm25", "change", "eth")]["flagged_accuracy"] == 0.0
    current = split[("bm25", "current", "pep")]  # the F1-current item is never flagged
    assert (current["flagged_items"], current["flagged_accuracy"]) == (0, None)
    assert (current["other_items"], current["other_accuracy"]) == (1, 0.0)
    assert {key[0] for key in split} == {"bm25", "closed-book", "graphiti", "grep-at-d",
                                         "long-context", "oracle"}
    text = (out / "results.md").read_text(encoding="utf-8")
    section = text.split("\n## H- F1 by a header-only episode after the date", 1)[1].split(
        "\n## ", 1)[0]
    assert section.startswith(" (answering model ")
    assert ("| System | Stratum | Corpus | Header-only episode after the date: items | Accuracy | "
            "None: items | Accuracy |") in section
    assert "| grep-at-d | change | pep | 6 | 0.667 | 6 | 0.000 |" in section


# --- the answering model's training cutoff (section 10) ------------------------------------------


def cutoff_data(root: Path, starts=("2026-03-15T10:00:00Z", "2026-06-30T23:59:59Z",
                                    "2026-07-01T00:00:00Z", "2026-08-02T12:00:00Z")):
    """F1-recent PEP items whose transitions were recorded at ``starts`` (gold
    ``meta.interval_start``)."""
    data = root / "data"
    data.mkdir()
    qs, gs, ids = [], [], []
    for n, start in enumerate(starts):
        item_id = f"F1-recent-PEP-{n}-2026-09-01"
        qs.append(question(item_id, "F1", "recent", "pep", "status", ("H+", "H-"),
                           proposal=f"PEP-{n}", as_of="2026-09-01"))
        gs.append(gold_row(item_id, "Accepted", "adopted", interval_start=start,
                           status_at_s="Accepted", gold_equals_status_at_s=True))
        ids.append(item_id)
    (data / "questions.jsonl").write_text("".join(dumps(q) + "\n" for q in qs), encoding="utf-8")
    (data / "gold.jsonl").write_text("".join(dumps(g) + "\n" for g in gs), encoding="utf-8")
    return data, ids


def test_f1_recent_is_split_by_the_training_cutoff_when_one_is_given(tmp_path):
    """Section 10: given --training-cutoff, the first day after the answering model's training
    data, every system's F1-recent accuracy is reported apart by whether the item's transition
    (gold ``meta.interval_start``) was recorded on or after that day."""
    data, ids = cutoff_data(tmp_path)
    rows = [answer(item_id, "dense", "H+", run, "Accepted" if k < 2 else "Draft")
            for k, item_id in enumerate(ids) for run in (1, 2, 3)]
    out = tmp_path / "out"
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", rows)], out,
                     training_cutoff="2026-07-01") == 0
    results = load(out)
    assert results["training_cutoff"] == "2026-07-01"
    split = {(r["system"], r["arm"], r["corpus"]): r for r in results["recent_cutoff_split"]}
    plus = split[("dense", "H+", "pep")]  # 23:59:59 on 2026-06-30 is before; midnight is on
    assert (plus["before_items"], plus["before_accuracy"]) == (2, 1.0)
    assert (plus["on_or_after_items"], plus["on_or_after_accuracy"]) == (2, 0.0)
    minus = split[("dense", "H-", "pep")]  # entered there and never answered
    assert (minus["before_accuracy"], minus["on_or_after_accuracy"]) == (0.0, 0.0)
    text = (out / "results.md").read_text(encoding="utf-8")
    assert "- Training cutoff: 2026-07-01, the first day after the answering model's training " \
           "data" in text
    section = text.split("\n## F1-recent by the answering model's training cutoff", 1)[1].split(
        "\n## ", 1)[0]
    assert section.startswith(" (answering model ")
    assert ("| System | Arm | Corpus | Transition before 2026-07-01: items | Accuracy | On or "
            "after 2026-07-01: items | Accuracy |") in section
    assert "| dense | H+ | pep | 2 | 1.000 | 2 | 0.000 |" in section
    # A provider that gives the cutoff as a month leaves that month's items on either side.
    assert ("Only the second group's transitions are certainly beyond the training data: when "
            "the provider states the cutoff as a month, an item recorded in that month may fall "
            "on either side of it.") in section
    assert "transitions are beyond the training data" not in section


def test_without_a_training_cutoff_results_md_says_f1_recent_is_not_split_by_one(tmp_path):
    data, ids = cutoff_data(tmp_path)
    rows = [answer(ids[0], "dense", "H+", run, "Accepted") for run in (1, 2, 3)]
    out = tmp_path / "out"
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", rows)], out) == 0
    results = load(out)
    assert results["training_cutoff"] is None and results["recent_cutoff_split"] == []
    text = (out / "results.md").read_text(encoding="utf-8")
    assert ("- Training cutoff: not given (--training-cutoff), so F1-recent is not split by the "
            "answering model's training cutoff.") in text
    assert "## F1-recent by the answering model's training cutoff" not in text


def test_a_training_cutoff_needs_every_f1_recent_items_transition_time(tmp_path, capsys):
    data, ids = cutoff_data(tmp_path)
    gold = data / "gold.jsonl"
    rows = [json.loads(line) for line in gold.read_text(encoding="utf-8").splitlines()]
    del rows[1]["meta"]["interval_start"]
    gold.write_text("".join(dumps(row) + "\n" for row in rows), encoding="utf-8")
    answers = write_answers(tmp_path / "a.jsonl", [answer(ids[0], "dense", "H+", 1, "Accepted")])
    out = tmp_path / "out"
    assert run_score(data, [answers], out, training_cutoff="2026-07-01") == 2
    err = capsys.readouterr().err
    assert ids[1] in err and "interval_start" in err and not out.exists()
    assert run_score(data, [answers], out) == 0  # without a cutoff the time is not read


# --- graphiti-reduced, M0's fallback (section 13) ------------------------------------------------


def test_graphiti_reduced_is_registered_and_entered_in_f1_f2_and_cf_by_default(tmp_path):
    """Section 13: if the M0 projection exceeds the budget, Graphiti is entered as
    graphiti-reduced, in F1, F2 and CF only, since F3, F4 and F5 need proposals no question
    names."""
    from asofbench.scoring import REGISTERED_SYSTEMS

    assert "graphiti-reduced" in REGISTERED_SYSTEMS
    data = rich_data(tmp_path)
    rows = [answer(i, "graphiti-reduced", arm, run, v) for i, v in RICH_GOLD.items()
            for arm in RICH_ARMS[i.split("-")[0]] for run in (1, 2, 3)]
    path = write_answers(tmp_path / "a.jsonl", rows)
    out = tmp_path / "out"
    assert run_score(data, [path], out) == 0
    results = load(out)
    assert {r["family"] for r in results["results"]} == {"F1", "F2", "CF"}
    assert all(r["accuracy_mean_over_runs"] == 1.0 for r in results["results"])
    assert results["scope"]["entries"]["graphiti-reduced"]["families"] == ["CF", "F1", "F2"]
    outside = sum(3 * len(RICH_ARMS[family]) for family in ("F3", "F4", "F5"))
    assert results["answers"]["out_of_scope_count"] == outside
    wider = tmp_path / "wider"
    assert run_score(data, [path], wider,
                     entries={"graphiti-reduced": {"families": ["F1", "F3"]}}) == 2
    assert not wider.exists()


def test_a_run_that_holds_both_graphiti_and_graphiti_reduced_stops(tmp_path, capsys):
    data = small_data(tmp_path)
    good = "F1-change-PEP-1-2010-01-01"
    rows = [answer(good, "graphiti", "H+", 1, "Final"),
            answer(good, "graphiti-reduced", "H+", 1, "Final")]
    out = tmp_path / "out"
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", rows)], out) == 2
    err = capsys.readouterr().err
    assert err.startswith("asofbench score: ") and "graphiti-reduced" in err
    assert "both graphiti and graphiti-reduced" in err and not out.exists()
    # A file that declares the other one is the same run.
    assert run_score(data, [write_answers(tmp_path / "b.jsonl", rows[:1])], out,
                     entries={"graphiti-reduced": {}}) == 2
    assert "both graphiti and graphiti-reduced" in capsys.readouterr().err and not out.exists()


def test_graphiti_reduced_is_the_pilots_graphiti(tmp_path, registered):
    data, ids = pilot_data(tmp_path)
    registered(data)
    right = {"closed-book": (2, 3), "bm25": (0, 3), "grep-at-d": (5, 4), "long-context": (6, 3),
             "hipporag2": (5, 4), "graphiti": (10, 6)}
    reduced = {("graphiti-reduced" if s == "graphiti" else s): v for s, v in right.items()}
    out = tmp_path / "out"
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", pilot_answers(
        ids, reduced, systems=tuple(reduced)))], out, items=data / "pilot.json") == 0
    gates = load(out, "gates.json")
    assert (gates["evaluable"], gates["stopped_at"], gates["signal"]) == (True, None, True)
    assert gates["pilot_systems"] == ["closed-book", "bm25", "grep-at-d", "long-context",
                                      "hipporag2", "graphiti-reduced"]
    assert set(gates["detail"]["gate2"]["engines"]) == {"hipporag2", "graphiti-reduced"}
    pairs = {(p["a"], p["b"]) for p in gates["detail"]["gate4"]["pairs"]}
    assert len(pairs) == 10 and ("hipporag2", "graphiti-reduced") in pairs
    assert load(out)["pilot"]["systems"] == gates["pilot_systems"]
    text = (out / "results.md").read_text(encoding="utf-8")
    assert "| graphiti-reduced | 1.000 | 1.000 |" in text  # the pilot table
    # The same answers under the id graphiti decide the same.
    plain = tmp_path / "plain"
    assert run_score(data, [write_answers(tmp_path / "b.jsonl", pilot_answers(ids, right))],
                     plain, items=data / "pilot.json") == 0
    assert load(plain, "gates.json")["pilot_systems"][-1] == "graphiti"
    assert (load(plain, "gates.json")["stopped_at"], load(plain, "gates.json")["signal"]) == (
        None, True)


def test_the_headline_compares_graphiti_reduced_in_graphitis_place(tmp_path):
    data, ids = headline_data(tmp_path)
    rows = [dict(r, system="graphiti-reduced") if r["system"] == "graphiti" else r
            for r in headline_answers(ids)]
    out = tmp_path / "out"
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", rows)], out) == 0
    headline = load(out)["headline"]
    assert headline["systems"] == ["bm25", "dense", "hybrid", "ppr", "hipporag2",
                                   "graphiti-reduced", "long-context"]
    found = comparisons_of(headline)
    assert found[("graphiti-reduced", "pep")]["accuracy"] == pytest.approx(11 / 36)
    assert not any(n["system"] == "graphiti" for n in headline["not_made"])


# --- the F2 floors and outside entries -----------------------------------------------------------


def test_the_f2_table_gives_the_always_empty_answers_score_beside_each_accuracy(tmp_path):
    """Always answering the empty list is right on every negative: its score on the same items
    is given beside each F2 accuracy, over all items and over the recoverable ones."""
    data = tmp_path / "data"
    data.mkdir()
    flags = {"F2-positive-PEP-1-2026-10-27": ([2], {"H+": True, "H-": False}),
             "F2-positive-PEP-3-2026-10-27": ([4], {"H+": False, "H-": False}),
             "F2-negative-PEP-5-2026-10-27": ([], {"H+": True, "H-": True})}
    qs = [question(i, "F2", i.split("-")[1], "pep", "set", ("H+", "H-")) for i in flags]
    gs = [gold_row(i, gold, recoverable=found) for i, (gold, found) in flags.items()]
    (data / "questions.jsonl").write_text("".join(dumps(q) + "\n" for q in qs), encoding="utf-8")
    (data / "gold.jsonl").write_text("".join(dumps(g) + "\n" for g in gs), encoding="utf-8")
    rows = [answer(i, "long-context", arm, run, [9] if i.endswith("PEP-3-2026-10-27") else gold)
            for i, (gold, _) in flags.items() for arm in ("H+", "H-") for run in (1, 2, 3)]
    out = tmp_path / "out"
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", rows)], out) == 0
    split = {r["arm"]: r for r in load(out)["f2_split"] if r["system"] == "long-context"}
    assert split["H+"]["always_empty"] == split["H-"]["always_empty"] == pytest.approx(1 / 3)
    assert split["H+"]["recoverable_always_empty"] == 0.5  # PEP 1 and the negative
    assert split["H-"]["recoverable_always_empty"] == 1.0  # the negative alone
    text = (out / "results.md").read_text(encoding="utf-8")
    assert ("| System | Arm | Items | Accuracy | Always empty | Recoverable items | Accuracy | "
            "Always empty |") in text
    assert "| long-context | H+ | 3 | 0.667 | 0.333 | 2 | 1.000 | 0.500 |" in text
    assert "always answering the empty list" in text


def test_an_outside_entry_is_labelled_in_every_results_table(tmp_path):
    """Section 11: a system that an entries file declares and section 11 does not name is shown
    as an outside entry in every table of results.md, as ppr is shown as the author's own;
    results.json keeps its id."""
    data = small_data(tmp_path)
    rows = [*perfect("my-system", runs=(1, 2)), *perfect("dense")]
    out = tmp_path / "out"
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", rows)], out,
                     entries={"my-system": {}, "dense": {}}) == 0
    label = "my-system (outside entry)"
    lines = (out / "results.md").read_text(encoding="utf-8").splitlines()
    tables = [line for line in lines if line.startswith("|")]
    cells = [[cell.strip() for cell in line.strip("|").split("|")] for line in tables]
    labelled = [line for line, row in zip(tables, cells, strict=True) if label in row]
    assert len(labelled) >= 9 and any(line.startswith("| Item |") for line in labelled)
    assert not any("my-system" in row for row in cells)  # no table cell is the bare id
    assert not any("dense (outside entry)" in row for row in cells)  # declared, but registered
    # Every other line of results.md that names it labels it, the closing list of runs it left
    # out among them; the headline's intro names the run's outside entries by id alone, in a
    # clause that says they are outside entries.
    text = "\n".join(lines)
    clause = "; this run holds the outside entry my-system."
    assert text.count(clause) == 1
    assert not re.search(r"my-system(?! \(outside entry\))", text.replace(clause, ""))
    assert "- my-system (outside entry) in arm H+: no answer in run 3" in text
    report = (out / "results.json").read_text(encoding="utf-8")
    assert "(outside entry)" not in report and '"system":"my-system"' in report
    assert load(out)["scope"]["outside_entries"] == ["my-system"]


def test_the_headline_says_that_closed_book_the_oracle_and_outside_entries_are_not_compared(
    tmp_path
):
    """Section 12: closed book, the oracle and outside entries are not compared, and the
    headline's intro names the run's outside entries by their ids, since its clause says that
    they are outside entries."""
    data, ids = headline_data(tmp_path)
    rows = headline_answers(ids)
    rows += [dict(r, system="my-system") for r in rows if r["system"] == "bm25"]
    plain, one, two = tmp_path / "plain", tmp_path / "one", tmp_path / "two"
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", headline_answers(ids))],
                     plain) == 0
    assert headline_intro(plain).endswith(
        "Closed book, the oracle and outside entries are not compared.")
    assert run_score(data, [write_answers(tmp_path / "b.jsonl", rows)], one,
                     entries={"my-system": {}}) == 0
    assert headline_intro(one).endswith(
        "Closed book, the oracle and outside entries are not compared; this run holds the "
        "outside entry my-system.")
    assert not any(c["system"] == "my-system" for c in load(one)["headline"]["comparisons"])
    assert run_score(data, [write_answers(tmp_path / "c.jsonl", rows)], two,
                     entries={"my-system": {}, "another": {}}) == 0
    assert headline_intro(two).endswith(
        "Closed book, the oracle and outside entries are not compared; this run holds the "
        "outside entries another and my-system.")


# --- the wording of results.md -------------------------------------------------------------------


def test_the_gate_4_table_names_its_two_systems_a_and_b(tmp_path, registered):
    data, ids = pilot_data(tmp_path)
    registered(data)
    out = tmp_path / "out"
    rows = pilot_answers(ids, {s: (9, 6) for s in SYSTEMS})
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", rows)], out,
                     items=data / "pilot.json") == 0
    text = (out / "results.md").read_text(encoding="utf-8")
    assert ("| System A | System B | Gap | p | Holm-adjusted p | Gate 4 | Unadjusted reading |"
            in text)
    assert "| System | System |" not in text


def test_a_scope_of_one_item_is_one_item(tmp_path):
    data = small_data(tmp_path)
    good = "F1-change-PEP-1-2010-01-01"
    rows = [answer(good, "dense", "H+", run, "Final") for run in (1, 2, 3)]
    answers = write_answers(tmp_path / "a.jsonl", rows)
    one = tmp_path / "one"
    assert run_score(data, [answers], one, items=write_items(tmp_path / "one.json", [good])) == 0
    text = (one / "results.md").read_text(encoding="utf-8")
    assert "Scope: the 1 item of one.json;" in text and "the 1 items of" not in text
    two = tmp_path / "two"
    assert run_score(data, [answers], two, items=write_items(
        tmp_path / "two.json", [good, "F1-change-PEP-2-2011-01-01"])) == 0
    assert "Scope: the 2 items of two.json;" in (two / "results.md").read_text(encoding="utf-8")


def test_the_closing_lists_name_systems_as_the_tables_do(tmp_path):
    """The lists at the end of results.md name each system as the tables do: ppr as the
    author's own and a system an entries file declares, and section 11 does not name, as an
    outside entry; a list of runs reads "runs 2 and 3" and a run label is quoted as text.
    results.json keeps the ids."""
    data = small_data(tmp_path)
    good = "F1-change-PEP-1-2010-01-01"
    rows = []
    for system in ("ppr", "my-system"):
        rows += [answer(good, system, "H+", 1, "Final"),
                 answer(good, system, "H+", 1, "Draft"),  # a repeated answer
                 answer(good, system, "CF", 1, "Final"),  # an arm the item is not asked in
                 answer("F2-positive-PEP-5-2026-10-27", system, "H+", 1, [387]),  # not entered
                 answer(good, system, "H+", 4, "Final")]  # a run label other than 1, 2 and 3
    entered = {"arms": ["H+", "H-"], "families": ["F1"]}
    out = tmp_path / "out"
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", rows)], out,
                     entries={"ppr": entered, "my-system": entered, "quiet": {}}) == 0
    closing = (out / "results.md").read_text(encoding="utf-8").split("\n## Warnings\n", 1)[1]
    lines = closing.splitlines()
    for line in (
        "- quiet (outside entry) is declared in out-entries.json but has no answer rows; is its "
        "name as the answers write it?",
        "- a.jsonl line 2: ppr (author's own) H+ F1-change-PEP-1-2010-01-01 run 1",
        "- ppr (author's own) CF F1-change-PEP-1-2010-01-01",
        "- ppr (author's own) H+ F2-positive-PEP-5-2026-10-27",
        '- a.jsonl line 5: ppr (author\'s own) H+ F1-change-PEP-1-2010-01-01 run "4"',
        "- a.jsonl line 10: my-system (outside entry) H+ F1-change-PEP-1-2010-01-01 run \"4\"",
        "- ppr (author's own): H-",
        "- my-system (outside entry): H-",
        "- ppr (author's own) in arm H+: no answer in runs 2 and 3, so every item is wrong in it",
        "- my-system (outside entry) in arm H+: no answer in runs 2 and 3, so every item is "
        "wrong in it",
    ):
        assert line in lines, line
    assert not re.search(r"\bppr\b(?! \(author's own\))", closing)
    assert not re.search(r"\b(?:my-system|quiet)\b(?! \(outside entry\))", closing)
    report = load(out)
    assert report["warnings"] == ["quiet is declared in out-entries.json but has no answer rows; "
                                  "is its name as the answers write it?"]
    assert {d["system"] for d in report["answers"]["duplicates"]} == {"ppr", "my-system"}


def test_the_scorer_says_what_its_code_does():
    """The module's description gives the number of malformed lines the command names, and the
    comment above the labels says where results.md shows them."""
    from asofbench import scoring

    assert f"names up to {scoring._LISTED} of them" in scoring.__doc__
    assert "names every such line" not in scoring.__doc__
    source = Path(scoring.__file__).read_text(encoding="utf-8")
    block = source.split("\nTABLE_LABELS = ", 1)[0].rsplit("\n\n", 1)[1]
    comment = " ".join(line.lstrip("#").strip() for line in block.splitlines())
    assert all(place in comment for place in ("tables", "the headline's lines", "closing lists"))
    assert "Only the tables" not in comment


def test_the_registered_runs_sentences_are_written_from_the_constants(tmp_path, registered):
    """results.md's sentence on the registered full run counts its comparisons from
    HEADLINE_FAMILY, as the scorer counts them, not from a number written into the text; and the
    sentences that state the registered full run's, the registered pilot run's and gates.json's
    conditions name each of the four registered values (section 11): the answering model, the
    answering settings, the construction model and the embedding model, the settings set off from
    the stores by a comma in both sentences, as results.md and gates.json print them."""
    from asofbench import scoring

    source = Path(scoring.__file__).read_text(encoding="utf-8")
    block = source.split("\nREGISTERED_RUN = (", 1)[1].split("\n\n", 1)[0]
    assert "{HEADLINE_FAMILY} registered comparisons" in block, block
    assert not re.search(r"\d+ registered comparisons", block), block
    values = (scoring.REGISTERED_ANSWERING_MODEL, scoring.REGISTERED_ANSWERING_SETTINGS,
              scoring.REGISTERED_CONSTRUCTION_MODEL, scoring.REGISTERED_EMBEDDING_MODEL)
    for sentence in (scoring.REGISTERED_RUN, scoring.REGISTERED_PILOT):
        assert all(value in sentence for value in values), sentence
        assert f"{scoring.REGISTERED_ANSWERING_SETTINGS}, over stores " in sentence, sentence
    data, ids = pilot_data(tmp_path)
    registered(data)
    out = tmp_path / "sonnet"
    assert score_as(data, pilot_answers(ids, {s: (9, 6) for s in SYSTEMS}), out,
                    answering_model="claude-sonnet-5-5") == 0
    section = pilot_section(out)
    rule = section.split("The gates decide only in the registered pilot run", 1)[1].split("\n")[0]
    assert all(value in rule for value in values), rule
    assert all(value in load(out, "gates.json")["decision"] for value in values)


# --- float sums that do not depend on the Python version -----------------------------------------

# Python 3.12 changed the built-in sum of floats, so a mean summed with it can differ in its last
# digit between 3.11 and 3.12 or later; math.fsum rounds correctly on both. Ten tenths sum to
# 0.9999999999999999 left to right (3.11's sum) and to 1.0 exactly; SKEWED is summed wrongly by
# both versions' sum and by a left-to-right loop.
TENTHS = [0.1] * 10
SKEWED = [0.1, 0.2, 0.3, 1e16, 0.7, 0.3, 3.0, 1.0, 0.7, 0.7]


def test_a_mean_wall_time_is_summed_exactly_on_every_python_version(tmp_path):
    """The mean of ten wall times of 0.1 s is 0.1, math.fsum's 1.0 over 10, on Python 3.11 as
    on 3.12 or later; so is any mean the scorer writes, here with values whose every other sum
    differs."""
    data, ids = headline_data(tmp_path, flagged=0, sizes={"pep": 10, "eth": 1})
    assert math.fsum(TENTHS) / 10 == 0.1
    for name, values in (("tenths", TENTHS), ("skewed", SKEWED)):
        rows = [dict(answer(item_id, "dense", "H+", 1, "Final"), wall_s=value)
                for item_id, value in zip(ids["pep"], values, strict=True)]
        out = tmp_path / name
        assert run_score(data, [write_answers(tmp_path / f"{name}.jsonl", rows)], out) == 0
        found = find(load(out)["secondary"], system="dense", arm="H+", stratum="change",
                     corpus="pep")
        assert found["mean_wall_s"] == math.fsum(values) / len(values), name


def test_every_share_and_total_is_summed_exactly():
    """The secondary shares are math.fsum's sums over the grid of items and runs, and a token
    total of floats is math.fsum's sum, while a total of integers stays an integer."""
    def scored(k: int, value: float) -> dict:
        return {"id": f"i{k}", "system": "dense", "arm": "H+", "run": "1", "family": "F3",
                "stratum": "size-2-4", "corpus": "eth", "correct": False, "valid": True,
                "outcome_correct": None, "precision": value, "recall": value, "f1": value,
                "abs_error": None, "within_one": None, "wall_s": value, "tokens_in": value,
                "tokens_out": k, "tool_calls": 0}

    rows = [scored(k, value) for k, value in enumerate(SKEWED)]
    (second,) = secondary(rows)
    assert second["precision"] == second["recall"] == second["f1"] == math.fsum(SKEWED) / 30
    (first,) = aggregate(rows)
    assert first["tokens_in"] == math.fsum(SKEWED)
    assert first["tokens_out"] == sum(range(10)) and isinstance(first["tokens_out"], int)


def test_the_split_and_cue_accuracies_are_summed_exactly(tmp_path):
    """An accuracy averaged over runs and items in the report's splits is math.fsum's sum of the
    items' accuracies: ten items, each right in one run of three, give 1/3, not the
    0.33333333333333337 that adding them one by one gives."""
    data, ids = headline_data(tmp_path, flagged=10, sizes={"pep": 10, "eth": 1})
    rows = [r for item_id in ids["pep"] for r in say("bm25", item_id, (True, False, False))]
    out = tmp_path / "out"
    cues = write_items(tmp_path / "cues.json", {"canary": CANARY})  # no message names a status
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", rows)], out,
                     message_cues=cues) == 0
    third = math.fsum([1 / 3] * 10) / 10
    assert third == 1 / 3
    split = find(load(out)["header_only_split"], system="bm25", stratum="change", corpus="pep")
    assert (split["flagged_items"], split["flagged_accuracy"]) == (10, third)
    (cue,) = load(out)["message_cue_split"]
    assert (cue["other_items"], cue["other_accuracy"]) == (12, math.fsum([1 / 3] * 10) / 12)


def test_the_pilots_pooled_accuracy_is_summed_exactly():
    """The gates pool each system's rows by item count with math.fsum: rows at 0.1, 0.2 and 0.3,
    one item each, pool to math.fsum's 0.6 over 3."""
    rows = [{"system": s, "stratum": "change", "corpus": corpus, "accuracy_mean_over_runs": value,
             "n_items": 1} for s in SYSTEMS for corpus, value in (("a", 0.1), ("b", 0.2),
                                                                  ("c", 0.3))]
    gates = pilot_gates(rows, {})
    assert gates["detail"]["accuracy_change"] == {s: math.fsum([0.1, 0.2, 0.3]) / 3
                                                  for s in SYSTEMS}


def test_the_secondary_table_heads_the_set_score_f_score(tmp_path):
    """Section 12 calls the harmonic mean of a set answer's precision and recall its F-score, so
    that F1 names only the family."""
    data = small_data(tmp_path)
    out = tmp_path / "out"
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", list(perfect("oracle")))],
                     out) == 0
    text = (out / "results.md").read_text(encoding="utf-8")
    section = text.split("\n## Secondary scores", 1)[1].split("\n## ", 1)[0]
    assert ("| System | Arm | Family | Stratum | Corpus | Outcome class | Precision | Recall | "
            "F-score | Mean abs error | Within one | Invalid | Mean wall s |") in section
    assert "precision, recall and F-score (their harmonic mean) are for sets" in section


# --- the headline's direction follows its test ---------------------------------------------------

TWO_OF_THREE = (True, True, False)
DIRECTION = "majority results: items the system gets right and grep at D wrong, against the reverse"


def direction_answers(ids) -> list[dict]:
    """90 PEP and 30 Ethereum items. Grep at D is right in every run on PEP items 0 to 79 and
    Ethereum items 0 to 19, and wrong on the rest. bm25 is right in two runs of three on every
    item, so it is right where grep at D is wrong on 10 items of each corpus and never the
    reverse, while its accuracy averaged over the runs is lower on PEP and equal on Ethereum.
    Long context is right everywhere; Graphiti is right in two runs of three on PEP items 0 to
    81 and wrong elsewhere."""
    rows = []
    for corpus, right_until in (("pep", 80), ("eth", 20)):
        for k, item_id in enumerate(ids[corpus]):
            rows += say("grep-at-d", item_id, RIGHT if k < right_until else WRONG)
            rows += say("bm25", item_id, TWO_OF_THREE)
            rows += say("long-context", item_id, RIGHT)
            graphiti_right = corpus == "pep" and k < 82
            rows += say("graphiti", item_id, TWO_OF_THREE if graphiti_right else WRONG)
    return rows


def like_grep(rows: list[dict], systems) -> list[dict]:
    """Each of ``systems`` answering exactly as grep at D does in ``rows``: right on the same
    items in the same runs, so each of its comparisons has p = 1."""
    return [dict(r, system=system) for r in rows if r["system"] == "grep-at-d"
            for system in systems]


# The compared systems that direction_answers and headline_answers leave out: with rows from them
# answering as grep at D does, a run holds every compared system and is the registered full run.
NOT_IN_ANSWERS = ("dense", "hybrid", "ppr", "hipporag2")


def direction_line(system, corpus, word, value, b, c, difference):
    """The visible line under a table for a directional claim or reading whose accuracy
    averaged over the runs differs the other way, or not at all."""
    end = ("does not differ (0.000)" if difference == "0.000"
           else f"differs the other way, by {difference}")
    return (f'- {system} on {corpus}: the {word} "{value}" follows the majority results the test '
            f"compares ({b} items right only for {system}, {c} right only for grep at D); the "
            f"accuracy averaged over the runs {end}.")


def test_the_headline_claim_follows_the_majority_results_the_test_compares(tmp_path, registered):
    """Section 12: a system is above grep at D when its Holm-adjusted p is below 0.05 and it is
    right where grep at D is wrong on more items than the reverse, below when the reverse holds,
    and otherwise there is no detectable difference. The accuracy averaged over the runs does not
    decide the direction: when it differs the other way, or not at all, a visible line says so."""
    data, ids = headline_data(tmp_path, flagged=0, sizes={"pep": 90, "eth": 30})
    registered(data)
    out = tmp_path / "out"
    rows = direction_answers(ids)
    rows += like_grep(rows, NOT_IN_ANSWERS)  # every compared system: the registered full run
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", rows)], out) == 0
    headline = load(out)["headline"]
    assert (headline["registered"], headline["not_registered"]) == (True, [])
    assert headline["direction"] == DIRECTION
    found = comparisons_of(headline)
    bm25_pep, bm25_eth = found[("bm25", "pep")], found[("bm25", "eth")]
    assert (bm25_pep["only_system_right"], bm25_pep["only_baseline_right"]) == (10, 0)
    assert bm25_pep["difference"] == pytest.approx(2 / 3 - 80 / 90)  # about -0.222
    assert bm25_pep["claim"] == "above grep at D"  # the test's direction, not the average's
    assert bm25_eth["difference"] == 0.0 and bm25_eth["claim"] == "above grep at D"
    # Four p-values of 2/1024 follow the smallest (2/2**20), so Holm gives them 13 times 2/1024.
    assert bm25_pep["p_holm"] == 13 * 2 / 1024
    assert bm25_pep["majority_difference"] == pytest.approx(10 / 90)
    assert bm25_eth["majority_difference"] == pytest.approx(10 / 30)
    assert found[("graphiti", "eth")]["majority_difference"] == pytest.approx(-20 / 30)
    assert all("reading" not in c for c in found.values())
    assert {key: c["claim"] for key, c in found.items()} == {
        ("bm25", "pep"): "above grep at D", ("bm25", "eth"): "above grep at D",
        ("graphiti", "pep"): "no detectable difference",  # 2 to 0: p = 0.5
        ("graphiti", "eth"): "below grep at D",
        ("long-context", "pep"): "above grep at D", ("long-context", "eth"): "above grep at D",
        **{(system, corpus): "no detectable difference" for system in NOT_IN_ANSWERS
           for corpus in ("pep", "eth")}}
    text = (out / "results.md").read_text(encoding="utf-8")
    section = text.split("\n## Headline comparison", 1)[1].split("\n## ", 1)[0]
    main, beside = section.split("Beside the headline", 1)
    lines = [line for line in main.splitlines()
             if line.startswith("- ") and "follows the majority results" in line]
    # One line for each directional claim whose run-averaged difference has the other sign or
    # is zero; none where the signs agree (long context, Graphiti on Ethereum), and none for a
    # comparison with no detectable difference (Graphiti on PEP, whose signs disagree).
    assert lines == [direction_line("bm25", "pep", "claim", "above grep at D", 10, 0, "-0.222"),
                     direction_line("bm25", "eth", "claim", "above grep at D", 10, 0, "0.000")]
    assert "the claim follows the majority results the test compares" in main  # the intro


def test_a_reading_beside_the_headline_follows_the_same_rule_and_makes_no_claim(tmp_path,
                                                                                registered):
    data, ids = headline_data(tmp_path, flagged=0, sizes={"pep": 90, "eth": 30})
    registered(data)
    out = tmp_path / "out"
    rows = direction_answers(ids)
    rows += like_grep(rows, NOT_IN_ANSWERS)  # the registered full run, whose headline claims
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", rows)], out) == 0
    headline = load(out)["headline"]
    assert headline["registered"] is True
    reading = comparisons_of(headline["readings"]["without_header_only_episode"])
    assert all("claim" not in c and "majority_difference" in c for c in reading.values())
    assert {key: c["reading"] for key, c in reading.items()} == {
        key: c["claim"] for key, c in comparisons_of(headline).items()}  # no item is flagged
    text = (out / "results.md").read_text(encoding="utf-8")
    beside = text.split("\n## Headline comparison", 1)[1].split("\n## ", 1)[0].split(
        "Beside the headline", 1)[1]
    assert ("| System | Corpus | Items | Accuracy | Grep at D's accuracy | Difference | System "
            "right, grep at D wrong | Grep at D right, system wrong | p | Holm-adjusted p | "
            "Reading |") in beside
    assert "| Claim |" not in beside
    assert [line for line in beside.splitlines()
            if line.startswith("- ") and "follows the majority results" in line] == [
        direction_line("bm25", "pep", "reading", "above grep at D", 10, 0, "-0.222"),
        direction_line("bm25", "eth", "reading", "above grep at D", 10, 0, "0.000")]


def test_a_comparison_records_its_runs_answered_and_a_line_names_any_shortfall(tmp_path):
    """A run with no answer is wrong, so a system, or grep at D, that left runs out of the
    compared items is weaker by that much: each comparison records the runs each side answered,
    and the headline and each reading name every shortfall in a visible line."""
    data, ids = headline_data(tmp_path)  # 12 items per corpus, items 0 to 5 flagged
    rows = headline_answers(ids)
    # Grep at D leaves run 3 out on PEP items 0 and 1; bm25 never answers Ethereum items 10, 11.
    rows = [r for r in rows if not (r["system"] == "grep-at-d" and r["run"] == 3
                                    and r["id"] in ids["pep"][:2])]
    rows = [r for r in rows if not (r["system"] == "bm25" and r["id"] in ids["eth"][10:])]
    out = tmp_path / "out"
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", rows)], out) == 0
    headline = load(out)["headline"]
    found = comparisons_of(headline)
    assert (found[("bm25", "pep")]["runs_answered"],
            found[("bm25", "pep")]["baseline_runs_answered"]) == (36, 34)
    assert (found[("bm25", "eth")]["runs_answered"],
            found[("bm25", "eth")]["baseline_runs_answered"]) == (30, 36)
    # Graphiti, entered on Ethereum with no scored answer there, is not compared on it, rather
    # than compared on no answer.
    assert ("graphiti", "eth") not in found
    assert {"system": "graphiti", "corpus": "eth", "reason": NO_ANSWER} in headline["not_made"]
    reading = comparisons_of(headline["readings"]["without_header_only_episode"])
    assert (reading[("bm25", "eth")]["runs_answered"], reading[("bm25", "eth")]["items"]) == (12, 6)
    assert reading[("bm25", "pep")]["baseline_runs_answered"] == 18  # items 6 to 11 only
    text = (out / "results.md").read_text(encoding="utf-8")
    section = text.split("\n## Headline comparison", 1)[1].split("\n## ", 1)[0]
    main, beside = section.split("Beside the headline", 1)
    wrong = "a missing run counts as wrong"
    assert [line for line in main.splitlines() if wrong in line] == [
        f"- grep-at-d answered 34 of the 36 runs on the 12 compared pep items; {wrong}.",
        f"- graphiti answered 29 of the 36 runs on the 12 compared pep items; {wrong}.",
        f"- bm25 answered 30 of the 36 runs on the 12 compared eth items; {wrong}."]
    assert [line for line in beside.splitlines() if wrong in line] == [
        f"- bm25 answered 12 of the 18 runs on the 6 compared eth items; {wrong}."]


def test_the_headline_lines_count_one_item_as_one_and_name_systems_as_the_tables_do(tmp_path):
    """The lines under the headline's tables count one item as "1 item", and name each system as
    the tables do, so the author's own retriever is "ppr (author's own)"."""
    data, ids = headline_data(tmp_path, flagged=0, sizes={"pep": 22, "eth": 1})
    pep, (eth,) = ids["pep"], ids["eth"]
    rows = say("grep-at-d", pep[0], WRONG) + say("ppr", pep[0], RIGHT)  # right only for ppr
    for item_id in pep[1:13]:  # right only for grep at D, by majority
        rows += say("grep-at-d", item_id, TWO_OF_THREE) + say("ppr", item_id, (True, False, False))
    for item_id in pep[13:]:  # both right; ppr in every run, grep at D in two, so the
        rows += say("grep-at-d", item_id, TWO_OF_THREE) + say("ppr", item_id, RIGHT)  # averages tie
    rows += say("grep-at-d", eth, RIGHT) + say("ppr", eth, (True, True, None))  # ppr leaves run 3
    out = tmp_path / "out"
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", rows)], out) == 0
    ppr_pep = comparisons_of(load(out)["headline"])[("ppr", "pep")]
    # Only grep at D and ppr answer, so the run gives readings; 14 times p = 28/8192 is 0.048.
    assert (ppr_pep["only_system_right"], ppr_pep["only_baseline_right"], ppr_pep["difference"],
            ppr_pep["reading"]) == (1, 12, 0.0, "below grep at D")
    assert ppr_pep["p_holm"] == 14 * mcnemar_exact(1, 12) == 14 * 28 / 8192
    text = (out / "results.md").read_text(encoding="utf-8")
    section = text.split("\n## Headline comparison", 1)[1].split("\n## ", 1)[0]
    main = section.split("Beside the headline", 1)[0]
    lines = [line for line in main.splitlines() if line.startswith("- ")]
    assert lines == [
        "- ppr (author's own) on pep: the reading \"below grep at D\" follows the majority results "
        "the test compares (1 item right only for ppr (author's own), 12 right only for grep at "
        "D); the accuracy averaged over the runs does not differ (0.000).",
        "- ppr (author's own) answered 2 of the 3 runs on the 1 compared eth item; a missing run "
        "counts as wrong.",
        f"- Not compared on pep or eth ({NO_ANSWER}): bm25, dense, hybrid, hipporag2, graphiti "
        "and long-context."]
    assert "1 items" not in section and "1 compared eth items" not in section
    # The intros count with the noun's number too: the headline's and the reading's.
    assert "in the run's scope (every item): 22 PEP items and 1 Ethereum item." in (
        headline_intro(out))
    assert "; this run makes 2 comparisons." in headline_intro(out)
    assert not re.search(r"\b\d+ (?:systems|corpora|of them)\b", headline_intro(out))
    assert ("The items without a header-only H- episode after their date (a commit that changed "
            "only withheld header lines, whose H- diff is empty): 22 PEP items and 1 Ethereum "
            "item.") in section.splitlines()
    assert not re.search(r"\bppr\b(?! \(author's own\))", section)  # never the bare id


def test_the_headline_names_grep_at_d_left_out_of_h_minus_f1_by_an_entries_file(tmp_path):
    data, ids = headline_data(tmp_path)
    out = tmp_path / "out"
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", headline_answers(ids))], out,
                     entries={"grep-at-d": {"arms": ["H+"]}}) == 0
    headline = load(out)["headline"]
    assert headline["comparisons"] == []
    assert {n["reason"] for n in headline["not_made"]} == {
        f"grep-at-d, the baseline, is {LEFT_OUT}"}
    assert len(headline["not_made"]) == 14


# --- the registered full run and the headline's fixed family (section 12) -----------------------

COMPARED = ("bm25", "dense", "hybrid", "ppr", "hipporag2", "graphiti", "long-context")


def eight_ahead_rows(ids, systems) -> list[dict]:
    """On each corpus's H- F1-change items, every system of ``systems`` is right in all three
    runs on items 0 to 39 and grep at D on items 0 to 31, so each comparison has 8 items right
    only for the system and none the other way (p = 2/256)."""
    rows = []
    for corpus in ("pep", "eth"):
        for k, item_id in enumerate(ids[corpus]):
            rows += say("grep-at-d", item_id, RIGHT if k < 32 else WRONG)
            for system in systems:
                rows += say(system, item_id, RIGHT if k < 40 else WRONG)
    return rows


def test_the_headline_is_holm_adjusted_over_the_14_registered_comparisons_in_every_run(tmp_path,
                                                                                       registered):
    """Section 12: the headline's Holm family is the 14 registered comparisons, every compared
    system on each corpus, a comparison the run does not make counting as p = 1, so scoring a
    system with fewer others never shrinks the family. Here, where the six others have hipporag2's
    own p, its rows give 14 times 2/256 and no detectable difference alone beside grep at D as
    beside them (in general the family only keeps an adjusted p from falling when systems are
    left out). Scored alone, the run is not the registered full run, so its rows are readings."""
    data, ids = headline_data(tmp_path, flagged=0, sizes={"pep": 150, "eth": 150})
    registered(data)
    alone, full = tmp_path / "alone", tmp_path / "full"
    assert run_score(data, [write_answers(tmp_path / "alone.jsonl",
                                          eight_ahead_rows(ids, ("hipporag2",)))], alone) == 0
    assert run_score(data, [write_answers(tmp_path / "full.jsonl",
                                          eight_ahead_rows(ids, COMPARED))], full) == 0
    for out, key, made in ((alone, "reading", 2), (full, "claim", 14)):
        headline = load(out)["headline"]
        for corpus in ("pep", "eth"):
            found = comparisons_of(headline)[("hipporag2", corpus)]
            assert (found["only_system_right"], found["only_baseline_right"], found["p"]) == (
                8, 0, 2 / 256)
            verdict = found.get("claim", found.get("reading"))
            assert (found["p_holm"], verdict) == (0.109375, "no detectable difference"), out.name
            assert 14 * 2 / 256 == 0.109375 and found[key] == verdict
        assert headline["registered"] is (key == "claim") and len(headline["comparisons"]) == made
    assert load(alone)["headline"]["not_registered"] == [
        f"bm25, dense, hybrid, ppr, graphiti and long-context are not compared on pep or eth "
        f"({NO_ANSWER})"]
    assert {c["claim"] for c in load(full)["headline"]["comparisons"]} == {
        "no detectable difference"}


def test_leaving_comparisons_out_of_a_run_never_lowers_a_remaining_adjusted_p():
    """With every comparison the run does not make counted as p = 1 in the 14 registered ones,
    removing comparisons never lowers the Holm-adjusted p of one that remains, so a claim can only
    weaken, from above or below grep at D to no detectable difference, over many random sets of
    p-values. A run that makes all 14 is plain Holm."""
    from asofbench.scoring import HEADLINE_FAMILY, _claimed, holm

    assert HEADLINE_FAMILY == 14
    rng = random.Random(20261031)

    def comparison() -> dict:
        b, c = rng.randint(0, 15), rng.randint(0, 15)
        p = mcnemar_exact(b, c) if rng.random() < 0.7 else rng.random() / 10
        return {"only_system_right": b, "only_baseline_right": c, "p": p}

    for _ in range(3000):
        made = [comparison() for _ in range(rng.randint(1, HEADLINE_FAMILY))]
        before = _claimed([dict(c) for c in made])
        kept = sorted(rng.sample(range(len(made)), rng.randint(0, len(made))))
        after = _claimed([dict(made[i]) for i in kept])
        for found, i in zip(after, kept, strict=True):
            assert found["p_holm"] >= before[i]["p_holm"], (made, kept)
            assert found["claim"] in (before[i]["claim"], "no detectable difference"), (made, kept)
    every = [comparison() for _ in range(HEADLINE_FAMILY)]
    assert [c["p_holm"] for c in _claimed(every)] == holm([c["p"] for c in every])


def score_as(data: Path, rows: list[dict], out: Path, *extra: str, **models: str) -> int:
    """``asofbench score`` on ``rows`` with the test models (``MODEL_FLAGS``), each of ``models``
    (such as ``answering_model``) replacing its flag's value, and the flags ``extra``."""
    flags = {**MODEL_FLAGS, **{f"--{name.replace('_', '-')}": value
                               for name, value in models.items()}}
    answers = write_answers(out.parent / f"{out.name}.jsonl", rows)
    return cli.main(["score", "--data", str(data), "--answers", str(answers), "--out", str(out),
                     *(word for pair in flags.items() for word in pair), *extra])


def headline_section(out: Path) -> str:
    text = (out / "results.md").read_text(encoding="utf-8")
    return text.split("\n## Headline comparison", 1)[1].split("\n## ", 1)[0]


def headline_intro(out: Path) -> str:
    """The paragraph that opens results.md's headline section, under its heading."""
    return headline_section(out).split("\n")[2]


def full_answers(ids) -> list[dict]:
    """``headline_answers``, completed: Graphiti answers every run of the PEP items (wrong in each
    run that ``GRAPHITI_PEP`` leaves out, so its majority results and accuracy do not move) and
    every Ethereum item (wrong in every run), and the four compared systems it leaves out answer
    as grep at D does. Every compared system and grep at D then answer every H- F1-change item of
    each corpus in runs 1, 2 and 3, so the run makes all 14 comparisons on complete answers."""
    rows = [r for r in headline_answers(ids) if r["system"] != "graphiti"]
    for k, item_id in enumerate(ids["pep"]):
        rows += say("graphiti", item_id, tuple(bool(right) for right in GRAPHITI_PEP[k]))
    rows += [r for item_id in ids["eth"] for r in say("graphiti", item_id, WRONG)]
    return rows + like_grep(rows, NOT_IN_ANSWERS)


def test_only_the_registered_full_run_makes_claims(tmp_path, registered):
    """Section 12: the registered full run scores every item (no --items), makes all 14
    registered comparisons, and was answered by the registered answering model with the
    registered settings over stores the registered construction model built, with text the
    registered embedding model embedded, on the registered build. Only its rows carry a claim and
    only its section calls itself the registered headline. Any other run makes its comparisons as
    readings, and its intro names each condition it does not meet; results.json says which run it
    is."""
    data, ids = headline_data(tmp_path)
    registered(data)
    rows = full_answers(ids)
    full = tmp_path / "full"
    assert score_as(data, rows, full) == 0
    headline = load(full)["headline"]
    assert (headline["registered"], headline["not_registered"]) == (True, [])
    claims = {key: c["claim"] for key, c in comparisons_of(headline).items()}
    assert len(claims) == 14 and not any("reading" in c for c in headline["comparisons"])
    assert claims[("bm25", "eth")] == "below grep at D"
    assert headline_intro(full).startswith("The registered headline (section 12).")
    assert "| Claim |" in headline_section(full).split("Beside the headline", 1)[0]
    every = write_items(tmp_path / "every.json", [*ids["pep"], *ids["eth"],
                                                  "F1-current-PEP-99-2026-10-27"])
    variants = {
        "items": (rows, ("--items", str(every)), {},
                  ["the run's scope is the 25 items of every.json, given with --items"]),
        "system": ([r for r in rows if r["system"] != "ppr"], (), {},
                   [f"ppr is not compared on pep or eth ({NO_ANSWER})"]),
        "model": (rows, (), {"answering_model": "claude-sonnet-5-5"},
                  ["answered by claude-sonnet-5-5, not the registered answering model"]),
        "settings": (rows, (), {"answering_settings": "effort high, adaptive thinking"},
                     ['answered with "effort high, adaptive thinking", not the registered '
                      'answering settings']),
        "construction": (rows, (), {"construction_model": "claude-sonnet-5-5"},
                         ["built by claude-sonnet-5-5, not the registered construction model"]),
        "embedding": (rows, (), {"embedding_model": "voyageai/voyage-4-lite"},
                      ["embedded with voyageai/voyage-4-lite, not the registered embedding "
                       "model"]),
    }
    for name, (given, extra, models, reasons) in variants.items():
        out = tmp_path / name
        assert score_as(data, given, out, *extra, **models) == 0, name
        found = load(out)["headline"]
        assert (found["registered"], found["not_registered"]) == (False, reasons), name
        assert not any("claim" in c for c in found["comparisons"]), name
        # The same comparisons by the same rule: ppr's had p = 1, so the rest read as the
        # registered run's claims whether its rows are there or not.
        assert {key: c["reading"] for key, c in comparisons_of(found).items()} == {
            key: claim for key, claim in claims.items()
            if name != "system" or key[0] != "ppr"}, name
        intro, section = headline_intro(out), headline_section(out)
        assert intro.startswith("This run is not the registered full run"), name
        assert "registered headline" not in section, name
        assert "| Reading |" in section and "| Claim |" not in section, name
        # results.md names the systems as the tables do: the author's own retriever is labelled.
        words = [reason.replace("ppr", "ppr (author's own)") for reason in reasons]
        assert all(reason in intro for reason in words), name
    # Every condition unmet at once: each is named, in this order; grep at D's missing answers
    # remove all 14 comparisons, under one cause.
    (data / "pilot.json").unlink()
    out = tmp_path / "none"
    assert score_as(data, [r for r in rows if r["system"] not in ("ppr", "grep-at-d")], out,
                    "--items", str(every), answering_model="claude-sonnet-5-5",
                    answering_settings="effort high",
                    construction_model="claude-sonnet-5-5",
                    embedding_model="voyageai/voyage-4-lite") == 0
    assert load(out)["headline"]["not_registered"] == [
        "the run's scope is the 25 items of every.json, given with --items",
        "bm25, dense, hybrid, ppr, hipporag2, graphiti and long-context are not compared on pep "
        f"or eth ({BASELINE_NO_ANSWER})",
        "answered by claude-sonnet-5-5, not the registered answering model",
        'answered with "effort high", not the registered answering settings',
        "built by claude-sonnet-5-5, not the registered construction model",
        "embedded with voyageai/voyage-4-lite, not the registered embedding model",
        "the build is not the registered one: pilot.json is absent"]


def test_the_registered_full_run_needs_every_run_of_both_sides(tmp_path, registered):
    """Section 12: the registered full run also needs grep at D and each compared system to have
    a scored answer to every H- F1-change item of each corpus in runs 1, 2 and 3. A run that lacks
    one run of one item, of grep at D or of a compared system, still makes all 14 comparisons (one
    scored answer makes a comparison), but its rows give readings, not claims, and it names each
    incomplete side and corpus, one reason each, in results.json and in the opening paragraph of
    results.md's headline, before the table."""
    data, ids = headline_data(tmp_path)  # 12 H- F1-change items per corpus
    registered(data)
    rows = full_answers(ids)
    complete = tmp_path / "complete"
    assert score_as(data, rows, complete) == 0
    headline = load(complete)["headline"]
    assert (headline["registered"], headline["not_registered"]) == (True, [])
    grep_pep, grep_eth, ppr_eth = (("grep-at-d", ids["pep"][0], 3), ("grep-at-d", ids["eth"][0], 1),
                                   ("ppr", ids["eth"][5], 2))
    on = {"pep": "grep-at-d answered 35 of the 36 runs on the 12 compared pep items",
          "eth": "grep-at-d answered 35 of the 36 runs on the 12 compared eth items",
          "ppr": "ppr answered 35 of the 36 runs on the 12 compared eth items"}
    cases = {"grep": ([grep_pep], [on["pep"]]), "ppr": ([ppr_eth], [on["ppr"]]),
             "grep on both": ([grep_pep, grep_eth], [on["pep"], on["eth"]]),
             "both sides": ([grep_pep, ppr_eth], [on["pep"], on["ppr"]])}
    for name, (gaps, reasons) in cases.items():
        out = tmp_path / name.replace(" ", "-")
        assert score_as(data, [r for r in rows if (r["system"], r["id"], r["run"]) not in gaps],
                        out) == 0, name
        headline = load(out)["headline"]
        assert (headline["registered"], headline["not_registered"]) == (False, reasons), name
        assert len(headline["comparisons"]) == 14, name  # every comparison is still made
        assert not any("claim" in c for c in headline["comparisons"]), name
        intro, section = headline_intro(out), headline_section(out)
        assert intro.startswith("This run is not the registered full run"), name
        for reason in reasons:  # results.md names the author's retriever as its tables do
            said = reason.replace("ppr", "ppr (author's own)")
            assert said in intro and section.index(said) < section.index("| System |"), name
    # The registered full run's opening paragraph states the condition it meets.
    assert ("grep at D and each compared system having a scored answer to every H- F1-change item "
            "of each corpus in runs 1, 2 and 3") in headline_intro(complete)


def test_the_registered_full_run_uses_the_registered_entries(tmp_path, registered):
    """Section 12: the registered full run uses the registered entries: no entries file, or one
    that only declares outside entries (systems section 11 does not name), whatever it enters
    them in, or that declares a registered system with its registered entries. An entries file
    that narrows any registered system's arms or families, one the headline does not compare
    included, makes the run not the registered full run, with a reason that names the system and
    what it narrows it to; the comparisons are still made, as readings."""
    data, ids = headline_data(tmp_path)
    with (data / "questions.jsonl").open("a", encoding="utf-8") as f:  # a second family, F2
        f.write(dumps(question("F2-positive-PEP-5-2026-10-27", "F2", "positive", "pep", "set",
                               ("H+", "H-"))) + "\n")
    with (data / "gold.jsonl").open("a", encoding="utf-8") as f:
        f.write(dumps(gold_row("F2-positive-PEP-5-2026-10-27", [387])) + "\n")
    registered(data)
    rows = full_answers(ids)

    def entered(name: str, entries: dict) -> tuple[dict, Path]:
        out = tmp_path / name
        assert score_as(data, rows, out, "--entries",
                        str(write_items(tmp_path / f"{name}.json", entries))) == 0, name
        return load(out)["headline"], out

    for name, entries in (
        ("outside", {"my-system": {}}),
        ("outside-narrowed", {"my-system": {"arms": ["H-"], "families": ["F1"]}}),
        ("as-registered", {"bm25": {}, "dense": {"arms": ["CF", "H+", "H-"]},
                           "long-context": {"families": ["F1", "F2"]}, "my-system": {}}),
    ):
        headline, out = entered(name, entries)
        assert (headline["registered"], headline["not_registered"]) == (True, []), name
        assert ("with the registered entries (no entries file, or one that only declares outside "
                "entries)") in headline_intro(out), name
    narrows = {
        "closed-book": ({"closed-book": {"arms": ["H-"]}},
                        ["the entries file narrows closed-book to the arm H-"]),
        "dense": ({"dense": {"families": ["F1"]}},
                  ["the entries file narrows dense to the family F1"]),
        "ppr": ({"ppr": {"arms": ["H+", "H-"], "families": ["F1"]}, "my-system": {}},
                ["the entries file narrows ppr to the arms H+ and H- in the family F1"]),
        "two": ({"closed-book": {"arms": ["H-"]}, "dense": {"families": ["F1"]}},
                ["the entries file narrows closed-book to the arm H-",
                 "the entries file narrows dense to the family F1"]),
    }
    for name, (entries, reasons) in narrows.items():
        headline, out = entered(name, entries)
        assert (headline["registered"], headline["not_registered"]) == (False, reasons), name
        assert len(headline["comparisons"]) == 14, name  # every comparison is still made
        assert not any("claim" in c for c in headline["comparisons"]), name
        intro = headline_intro(out)  # results.md names the author's retriever as its tables do
        assert all(r.replace("ppr", "ppr (author's own)") in intro for r in reasons), name


def test_a_comparison_is_made_only_where_both_sides_have_a_scored_answer(tmp_path, registered):
    """Section 12: a comparison is made only when the compared system and grep at D both have a
    scored answer on the corpus's H- F1-change items in the scope; a row set aside, on another run
    label or in another arm, or outside the system's entries, is none. A comparison not made
    names its cause, one line per cause, and a run that does not make all 14 is not the
    registered full run: it claims nothing."""
    data, ids = headline_data(tmp_path)
    registered(data)
    rows = full_answers(ids)
    full = tmp_path / "full"
    assert score_as(data, rows, full) == 0
    assert load(full)["headline"]["registered"] is True
    # (a) Every row of grep at D on run label 4: no comparison is made, and no claim anywhere.
    out = tmp_path / "grep-run-4"
    assert score_as(data, [dict(r, run=4) if r["system"] == "grep-at-d" else r for r in rows],
                    out) == 0
    headline = load(out)["headline"]
    assert headline["comparisons"] == [] and len(headline["not_made"]) == 14
    assert headline["not_registered"] == [
        "bm25, dense, hybrid, ppr, hipporag2, graphiti and long-context are not compared on pep "
        f"or eth ({BASELINE_NO_ANSWER})"]
    assert '"claim"' not in (out / "results.json").read_text(encoding="utf-8")
    assert "registered headline" not in (out / "results.md").read_text(encoding="utf-8")
    assert (f"- Not compared on pep or eth ({BASELINE_NO_ANSWER}): bm25, dense, hybrid, ppr "
            "(author's own), hipporag2, graphiti and long-context.") in headline_section(
        out).splitlines()
    # (b) bm25's rows all in arm H+: its two comparisons are not made.
    out = tmp_path / "bm25-h-plus"
    assert score_as(data, [dict(r, arm="H+") if r["system"] == "bm25" else r for r in rows],
                    out) == 0
    headline = load(out)["headline"]
    assert headline["not_made"] == [{"system": "bm25", "corpus": corpus, "reason": NO_ANSWER}
                                    for corpus in ("pep", "eth")]
    assert headline["not_registered"] == [f"bm25 is not compared on pep or eth ({NO_ANSWER})"]
    assert not any("claim" in c for c in headline["comparisons"])
    # (c) An entries file that leaves bm25 out of H- F1: the reasons say that it narrows a
    # registered system and that the comparisons are not made.
    out = tmp_path / "bm25-entries"
    assert score_as(data, rows, out, "--entries", str(write_items(
        tmp_path / "entries.json", {"bm25": {"arms": ["H+"]}}))) == 0
    headline = load(out)["headline"]
    assert headline["not_registered"] == ["the entries file narrows bm25 to the arm H+",
                                          f"bm25 is not compared on pep or eth ({LEFT_OUT})"]
    assert f"- Not compared on pep or eth ({LEFT_OUT}): bm25." in headline_section(
        out).splitlines()


def test_graphiti_reduced_takes_graphitis_place_in_the_registered_full_run(tmp_path, registered):
    """Section 13's fallback: with graphiti-reduced in Graphiti's place, and no graphiti rows, the
    run is the registered full run and compares graphiti-reduced. Declared in an entries file with
    no rows, graphiti-reduced has no scored answer: its comparisons are not made, the run is not
    the registered full run, and the declaration is a warning."""
    data, ids = headline_data(tmp_path)
    registered(data)
    rows = full_answers(ids)
    out = tmp_path / "reduced"
    assert score_as(data, [dict(r, system="graphiti-reduced") if r["system"] == "graphiti"
                           else r for r in rows], out) == 0
    headline = load(out)["headline"]
    assert headline["registered"] is True and "graphiti-reduced" in headline["systems"]
    found = comparisons_of(headline)
    assert len(found) == 14 and {("graphiti-reduced", "pep"), ("graphiti-reduced", "eth")} <= set(
        found)
    assert found[("graphiti-reduced", "eth")]["claim"] == "below grep at D"
    # Declared, with no rows.
    declared = tmp_path / "declared"
    assert score_as(data, [r for r in rows if r["system"] != "graphiti"], declared, "--entries",
                    str(write_items(tmp_path / "reduced.json", {"graphiti-reduced": {}}))) == 0
    results = load(declared)
    assert results["headline"]["not_made"] == [
        {"system": "graphiti-reduced", "corpus": corpus, "reason": NO_ANSWER}
        for corpus in ("pep", "eth")]
    assert results["headline"]["not_registered"] == [
        f"graphiti-reduced is not compared on pep or eth ({NO_ANSWER})"]
    assert results["warnings"] == ["graphiti-reduced is declared in reduced.json but has no "
                                   "answer rows; is its name as the answers write it?"]


def test_a_reading_makes_a_comparison_only_on_scored_answers_to_its_own_items(tmp_path,
                                                                             registered):
    """Section 12: a reading makes a headline comparison only when both sides have a scored
    answer on the corpus's items left in it. bm25 answers only the PEP items with a header-only
    episode after their date: the headline compares it on PEP, since one scored answer makes a
    comparison, the reading without those items cannot, and its line says why. A side that leaves
    runs out keeps the run from being the registered full run, which needs every run."""
    data, ids = headline_data(tmp_path)  # PEP and Ethereum items 0 to 5 are flagged
    registered(data)
    rows = [r for r in full_answers(ids)
            if not (r["system"] == "bm25" and r["id"] in ids["pep"][6:])]
    out = tmp_path / "out"
    assert score_as(data, rows, out) == 0
    headline = load(out)["headline"]
    assert ("bm25", "pep") in comparisons_of(headline) and len(headline["comparisons"]) == 14
    assert (headline["registered"], headline["not_registered"]) == (
        False, ["bm25 answered 18 of the 36 runs on the 12 compared pep items"])
    plain = headline["readings"]["without_header_only_episode"]
    assert ("bm25", "pep") not in comparisons_of(plain) and ("bm25", "eth") in comparisons_of(
        plain)
    assert plain["not_made"] == [{"system": "bm25", "corpus": "pep", "reason": NO_ANSWER_LEFT}]
    assert f"- Not compared on pep ({NO_ANSWER_LEFT}): bm25." in headline_section(
        out).splitlines()


def test_a_training_cutoff_with_no_f1_recent_item_in_the_scope_splits_nothing(tmp_path):
    data = small_data(tmp_path)  # no F1-recent item
    out = tmp_path / "out"
    rows = [answer("F1-change-PEP-1-2010-01-01", "dense", "H+", run, "Final") for run in (1, 2, 3)]
    assert run_score(data, [write_answers(tmp_path / "a.jsonl", rows)], out,
                     training_cutoff="2026-07-01") == 0
    results = load(out)
    assert (results["training_cutoff"], results["recent_cutoff_split"]) == ("2026-07-01", [])
    text = (out / "results.md").read_text(encoding="utf-8")
    assert ("- Training cutoff: 2026-07-01, the first day after the answering model's training "
            "data (--training-cutoff); no F1-recent item is in the run's scope, so nothing is "
            "split by it.") in text
    assert "also split" not in text
    assert "## F1-recent by the answering model's training cutoff" not in text


def model_run(tmp_path, rows, **flags) -> int:
    """``asofbench score`` on small_data with the given model flags (the others as MODELS)."""
    data = small_data(tmp_path / "d")
    given = {**MODEL_FLAGS, **{f"--{k.replace('_', '-')}": v for k, v in flags.items()
                               if k != "entries"}}
    argv = ["score", "--data", str(data), "--answers",
            str(write_answers(tmp_path / "a.jsonl", rows)), "--out", str(tmp_path / "out"),
            *(word for pair in given.items() for word in pair)]
    if "entries" in flags:
        argv += ["--entries", str(write_items(tmp_path / "entries.json", flags["entries"]))]
    return cli.main(argv)


@pytest.mark.parametrize("system", ["dense", "hybrid", "hipporag2", "graphiti",
                                    "graphiti-reduced"])
def test_no_embedding_model_is_refused_when_the_run_holds_a_system_that_embeds(tmp_path, system,
                                                                             capsys):
    """Sections 11 and 12: --embedding-model none says that no system of the run embeds text, so
    a run whose rows or entries file hold one of the registered systems that do is refused."""
    rows = [answer("F1-change-PEP-1-2010-01-01", system, "H+", 1, "Final")]
    assert model_run(tmp_path, rows, embedding_model="none") == 2
    err = capsys.readouterr().err
    assert "--embedding-model" in err and system in err and "embed" in err
    assert not (tmp_path / "out").exists()
    plain = [answer("F1-change-PEP-1-2010-01-01", "bm25", "H+", 1, "Final")]
    assert model_run(tmp_path / "declared", plain, embedding_model="none",
                     entries={system: {}}) == 2
    assert "--embedding-model" in capsys.readouterr().err
    # bm25, long context, grep at D and ppr hold no embedding model the registration fixes.
    for other in ("bm25", "long-context", "grep-at-d", "ppr"):
        rows = [answer("F1-change-PEP-1-2010-01-01", other, "H+", 1, "Final")]
        assert model_run(tmp_path / other, rows, embedding_model="none",
                         construction_model="none") == 0, other


@pytest.mark.parametrize("system", ["hipporag2", "graphiti", "graphiti-reduced"])
def test_no_construction_model_is_refused_when_the_run_holds_an_llm_built_store(tmp_path, system,
                                                                              capsys):
    rows = [answer("F1-change-PEP-1-2010-01-01", system, "H+", 1, "Final")]
    assert model_run(tmp_path, rows, construction_model="none") == 2
    err = capsys.readouterr().err
    assert "--construction-model" in err and system in err and not (tmp_path / "out").exists()
    plain = [answer("F1-change-PEP-1-2010-01-01", "dense", "H+", 1, "Final")]
    assert model_run(tmp_path / "declared", plain, construction_model="none",
                     entries={system: {}}) == 2
    assert "--construction-model" in capsys.readouterr().err
    assert model_run(tmp_path / "dense", plain, construction_model="none") == 0


@pytest.mark.parametrize("flag", sorted(NEEDS))
@pytest.mark.parametrize("bad", ["claude\\nhaiku", "x\\ty", "model" + chr(0x202E) + "x",
                                 "a" + chr(0x200B) + "b", "a" + chr(0x2028) + "b",
                                 "a" + chr(0x2029) + "b"])
def test_a_model_or_setting_with_a_control_or_format_character_is_refused(flag, bad, capsys):
    """The models and the settings go into every heading of results.md, so a value holding a
    character of Unicode category Cc, Cf, Zl or Zp (a newline, a tab, a bidirectional control, a
    zero-width space, a line or a paragraph separator) is refused, naming its flag."""
    value = bad.replace("\\n", "\n").replace("\\t", "\t")
    models = {**MODEL_FLAGS, flag: value}
    argv = ["score", "--data", "d", "--answers", "a.jsonl", "--out", "o"]
    with pytest.raises(SystemExit) as refused:
        cli.main(argv + [word for pair in models.items() for word in pair])
    assert refused.value.code == 2
    err = capsys.readouterr().err
    assert f"argument {flag}:" in err and "Unicode category Cc, Cf, Zl or Zp" in err
