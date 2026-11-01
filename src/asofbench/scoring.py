"""The scorer: answers, per-item scores, intervals, aggregation and the pilot gates.

Spec section 12 (answers and scoring) and section 13 (the gates). ``asofbench score`` reads the
questions and gold of a build and one or more answer files, and writes ``results.json``,
``results.md`` and, when the build has a pilot list (``pilot.json``), ``gates.json``. With a
``MANIFEST.json`` in the build, the questions, gold and pilot list must be the files it registers.

**Answer rows** (section 12). Each row is one JSON object with ``id``, ``system``, ``arm``, ``run``
and ``answer``, and the accounting fields ``tokens_in``, ``tokens_out``, ``wall_s`` and
``tool_calls``. ``arm`` is ``H+``, ``H-`` or ``CF`` (the minus sign reads as the hyphen). A row
that lacks one of the five fields, or has a value no row can have (an arm that is not one of the
three, a system that is no name, a run that is neither an integer nor a string), stops the run:
the command exits 2, gives the number of such lines and names up to 50 of them, and nothing is
scored. ``system`` is one of ``REGISTERED_SYSTEMS`` (section 11) or a name an entries file
declares; any other name stops the run too, with the nearest registered name as a suggestion. An
entries-file name that no row uses is reported as a warning. A row that lacks an accounting field,
or has a value there that is not a number, is still scored: the report counts such rows per
system and field, and the token totals, tool-use rate and mean wall time leave the missing values
out instead of counting zeros (each is unknown, shown as "-", for a group in which no row carries
its value).

**Models** (section 11). Every run names the model that answered, the settings of every
answering call, the model that built the run's LLM-built stores and the one embedding model of
the run's systems that embed text (``--answering-model``, ``--answering-settings``,
``--construction-model`` and ``--embedding-model``; the last two are ``none``, in any letter case
and recorded as ``none``, when the run has no such store or no such system). All four are
required, refused when blank or when they hold a character of Unicode category Cc, Cf, Zl or Zp,
recorded in ``results.json`` and ``gates.json`` and named at the top of ``results.md`` and in the
heading of each of its tables; each answering model is scored in its own run. ``none`` is refused
for the embedding or the construction model when the run holds a registered system that embeds
text (``EMBEDDING_SYSTEMS``) or builds its store with a language model (``CONSTRUCTED_SYSTEMS``).
The Personalized PageRank retriever, ``ppr``, is the benchmark author's own, so every line of
``results.md`` that names it, in a table, under the headline or in the closing lists, shows it as
``ppr (author's own)``, and a system that an entries file declares and section 11 does not name
is shown there as ``<id> (outside entry)``; ``results.json`` and ``gates.json`` keep the ids
(section 11). The registered answering model, settings, construction model and embedding model
are ``REGISTERED_ANSWERING_MODEL``, ``REGISTERED_ANSWERING_SETTINGS``,
``REGISTERED_CONSTRUCTION_MODEL`` and ``REGISTERED_EMBEDDING_MODEL``, and the registered build is
data/v0.1, whose questions, gold and pilot list have the SHA-256s in ``REGISTERED_DATA``.

**Runs.** ``run`` is read as a string, so 1 and "1" are one run. Every score counts runs 1, 2 and
3: a run with no answer is wrong, and a row with any other run label is listed and ignored, as
are the runs in which a system that answered in an arm has no answer there. Answer files are read
in order of their names, each line by line; when one system, arm, item and run has several rows
the first stands, whatever order the files were given in, and the rest are listed.

**Answers.** A status answer is a string, normalised by ``statuses`` (stripped, case-folded,
quotes and a trailing full stop removed, ``last-call`` read as ``last call``, the PEP legacy table
applied). A set answer is a list; each item must be an integer that is not a boolean, or a string
that is all ASCII digits or a series prefix (``PEP``, ``EIP`` or ``ERC``, then spaces or a hyphen)
and digits; the list is de-duplicated. Anything else, including a boolean, a float (whether or not
it has a fractional part), a nested list, null or any other value, makes the whole answer invalid
and it scores zero. A count answer must be an ``int`` that is not a ``bool``. ``valid`` says
whether the answer had the right shape: a string that names no status is valid but wrong.

**Entries and denominators** (section 12). Each system is entered in some arms and families.
By default every system is entered in every arm and family, except long context and
graphiti-reduced, which are entered in F1, F2 and CF only (sections 11 and 13). graphiti-reduced
is Graphiti under M0's fallback, entered instead of graphiti, so a run that holds both
stops. ``--entries FILE`` declares them per system, as a JSON object mapping a system to
``{"arms": [...], "families": [...]}``; a key left out means the default, and a file may narrow
what a system is entered in but not widen it. A system's denominator in an arm is every item
of that arm in the families it is entered in, or the item ids of a given list for a pilot run
(``--items``). A system is in the run if the entries file names it or it answered any item of the
questions. A system entered in an arm or family in which it has no answers scores zero there, and
its rows are present. An item a system does not answer in a run is wrong in that run, so leaving
items out cannot raise a score. The coverage table gives answered against asked. Ids that are not
in the questions are listed in ``unknown_ids`` and ignored, as are answers in an arm the item is
not asked in and answers outside a system's entries or the items of the run.

**Counterfactual items** (section 7). CF items are scored apart by whether their body at D holds
a decision word and by whether their file at D has a header cue, a header field other than the
status that records the true status (``meta.header_cue``, ``questions.HEADER_CUE_RULE``).
``--cf-messages FILE`` is the companion file that ``asofbench stream --arm CF`` writes beside
the CF stream: for each CF proposal, the number of commits outside the rewritten ones, each
counted once, whose message mentions it and holds one of its corpus's canonical status values.
The CF items table gives that count beside each item and what each system made of it. When the
build's manifest registers the file beside its CF stream (``build --streams``), a file whose
SHA-256 differs stops the run; ``results.md`` says in a visible line which file was read and
whether it was checked, or that none was given. The same holds for the message cues below.

**Message cues** (section 12). ``--message-cues FILE`` is the companion file that ``asofbench
stream --arm H-`` writes: for each proposal, the ``[recorded_at, status]`` pairs of the kept
messages that mention it with a canonical status value. For each H- F1 item the report says
whether a cue at or before the end of its date names its gold status, and gives each system's
H- F1 accuracy on those items and on the rest, so H- results can be read against what the
messages alone reveal. Every system's H- F1 accuracy is also given apart by whether the item's
proposal has a header-only H- episode after its date (gold ``meta.h_minus_header_only_after_d``):
an episode whose diff the withholding empties, which shows that a withheld header changed.

**Headline** (section 12). Each system that keeps or retrieves context from the stream or its
proposals (``HEADLINE_SYSTEMS``) is compared with grep at D on each corpus's H- F1-change items
in the scope: accuracy averaged over the runs, its difference from grep at D's, and the exact
two-sided McNemar test on majority results, Holm-adjusted over the 14 registered comparisons
(``HEADLINE_FAMILY``, every compared system on each corpus), a comparison the run does not make
counting as p = 1. A comparison is made only when the system and grep at D both have a scored
answer (in the scope and the entries, on run 1, 2 or 3, not set aside) to the corpus's H-
F1-change items; otherwise a visible line names it and its cause, one line per cause. A system
is above grep at D when its adjusted p is below 0.05 and it is right where grep at D is wrong on
more items than the reverse, below when the adjusted p is below 0.05 and the reverse holds, and
otherwise shows no detectable difference: the verdict follows the majority results the test
compares, and a visible line says when the accuracy averaged over the runs differs the other way,
or not at all. The verdict is a ``claim`` only in the registered full run, which scores every item
(no ``--items``) of the registered build with the registered entries (no entries file, or one
that narrows no registered system), makes all 14 comparisons with grep at D and each compared
system answering every compared item in runs 1, 2 and 3, and is answered by the registered
answering model with the registered settings over stores the registered construction model
built, with text the registered embedding model embedded; any other run gives its comparisons as
a ``reading``, and ``results.json`` says which run it is (``headline.registered``) and why not
(``headline.not_registered``: each registered system an entries file narrows, each comparison
not made, and each side and corpus that left a run out, among the other conditions). Each
comparison also gives the share of items by which the majority results lead
(``majority_difference``) and the runs each side answered, and a visible line names every side
that left runs out, since a missing run is wrong. The corpora
pooled are reported descriptively. The same comparisons on the items without a header-only
episode after their date, and, given the message cues, on those that no kept message names, are
reported beside it, each made only where both sides have a scored answer to the items left in
it, each Holm-adjusted over its own 14, and decide nothing: their rows carry a ``reading``, by
the same rule, not a ``claim``. ``results.json`` also gives, for every system, arm and item it is
scored on, the runs right, the runs answered and the majority result (``items``), from which any
comparison can be recomputed.

**Contamination** (section 10). Each corpus's F1-change floor is the share of its F1-change items
in the run's scope, those closed book is scored on, whose gold is their most common gold value.
Closed book's result on an item is the majority of every answer it gave the item, in every run
and arm (a tie, or no answer, is wrong), and an exact one-sided binomial test of those results
against the floor at p < 0.05 marks the corpus's F1-change as possibly contaminated; F1-change
stays the primary as-of result either way. A corpus whose F1-change items closed book gave no
answer to is not tested, and the report says so. Every system's F1-change accuracy is also given
on the items closed book gets wrong and on the rest, and its F1-recent accuracy apart by whether
the gold is the status at S, which the build records in each F1 item's gold
(``gold_equals_status_at_s``). Given ``--training-cutoff DAY``, the first day after the answering
model's training data, F1-recent is also given apart by whether the item's transition
(``meta.interval_start``) was recorded before that day or on or after it; without it, a visible
line says that F1-recent is not split by the cutoff.

**F2** (section 12). Each system's F2 accuracy is also given over the items whose gold the asked
PEP's own episodes can show in the arm, which the build records per arm (``meta.recoverable``),
and beside each accuracy what always answering the empty list scores on the same items.

**Aggregation.** One row per (system, arm, family, stratum, corpus). ``accuracy_mean_over_runs``
is exact match averaged over the items and runs, and ``majority_class_rate`` is the share of the
group's items whose gold is their most common gold value. An item's majority result is correct
when it is correct in more than half of the runs (at least two of three), and the Wilson
interval is over those results; ``pass_at_3`` is the share of items correct in any run and
``all_3`` in every run. The secondary scores (outcome-class accuracy, set precision, recall and
F-score, absolute error and the within-one rate, the invalid share and wall time) are in a
separate table.

**Gates** (section 13). ``pilot_gates`` applies the four gates in order to the PEP and Ethereum
H- F1 pilot rows pooled by item count. It reads the headline score (accuracy averaged over the
runs) for the thresholds and each item's majority result for the exact McNemar tests. Gate 3's
test is two-sided, against the better of grep at D and long context (against each on a tie);
gate 4 counts the ten pairs of pilot systems without closed book, with their p-values
Holm-adjusted, and keeps the unadjusted reading over all fifteen pairs beside it. The pilot's
Graphiti is graphiti or, under M0's fallback, graphiti-reduced (``pilot_systems``). The gates
decide only in the registered pilot run, whose scope, given with ``--items``, is exactly the
items of ``pilot.json`` (so a run of every item never decides them), with every pilot system
entered in H- F1, on the registered build, and whose answers the registered
answering model gave with the registered settings over stores the registered construction model
built, with text the registered embedding model embedded, and only when all six pilot systems
have answers there; an item a present system did not answer is wrong, and the coverage of the
pilot items in the run's scope is reported in ``gates.json``. Otherwise the result is "not
evaluable" (``_not_evaluable``, its one constructor): ``evaluable`` is false, ``reasons`` says
why, every gate's computation is kept, and there is no ``stopped_at`` or ``signal``. In both
shapes ``gates.json`` gives its ``decision`` as a top-level field.
``results.json``'s ``pilot`` gives ``reasons`` in every shape, that of a build without a pilot
list included.
"""

from __future__ import annotations

import difflib
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from fractions import Fraction
from hashlib import sha256
from itertools import combinations
from pathlib import Path

from .corpora import CANARY
from .manifest import CF_MESSAGES_FILE, CUES_FILE, ManifestError, read_manifest, registered_stream
from .questions import ANSWER_TYPES, HEADER_CUE_RULE, dumps, majority_class
from .statuses import normalize_answer_status, outcome_class

Z95 = 1.959963984540054
GROUP = ("system", "arm", "family", "stratum", "corpus")
PILOT_SYSTEMS = ("closed-book", "bm25", "grep-at-d", "long-context", "hipporag2", "graphiti")
CLOSED_BOOK = "closed-book"
ENGINES = ("hipporag2", "graphiti")
SATURATING = ("grep-at-d", "long-context")  # gate 3: the two baselines that can saturate
# Gate 4 counts the ten pairs of pilot systems in which neither system is closed book.
SIGNAL_PAIRS = tuple(pair for pair in combinations(PILOT_SYSTEMS, 2) if CLOSED_BOOK not in pair)
CEILING = 0.85  # gates 1 and 3: "scores 85% or more"
LOADING_MARGIN = 0.10  # gate 2: "within 10 points"
SIGNAL_GAP = 0.15  # gate 4: "differs by 15 points or more"
ALPHA = 0.05  # significance: gates 3 and 4 (exact McNemar) and section 10 (exact binomial)
EPS = 1e-9  # proportions are compared with the registered thresholds to this tolerance
MAX_LISTED = 200  # entries of each problem list that results.json spells out
CORPORA = ("pep", "eth")
ARMS = ("H+", "H-", "CF")
RUNS = ("1", "2", "3")  # section 11: each system answers each item three times in each arm
# Section 11: the entrants' identifiers. Any other system name needs an entries file.
# graphiti-reduced is Graphiti under M0's fallback (section 13), entered instead of graphiti.
REGISTERED_SYSTEMS = ("closed-book", "bm25", "dense", "hybrid", "grep-at-d", "long-context",
                      "hipporag2", "graphiti", "graphiti-reduced", "ppr", "oracle")
# Section 11: the registered answering model, the settings of every answering call, the model
# that builds every LLM-built store, and the one embedding model of every system that embeds text
# (a Hugging Face model at its revision). Only a run answered by this model with these settings,
# over stores that model built, with text that embedding model embedded, can be the registered
# full run, whose headline makes claims (section 12), or the registered pilot run, whose gates
# decide (section 13).
REGISTERED_ANSWERING_MODEL = "claude-haiku-5-5"
REGISTERED_ANSWERING_SETTINGS = "effort medium, adaptive thinking, default temperature"
REGISTERED_CONSTRUCTION_MODEL = "claude-haiku-5-5"
REGISTERED_EMBEDDING_MODEL = "voyageai/voyage-4-nano@5fb34e870f49f19e79c94d5e118f5ee730133279"
# Sections 12 and 13: the registered build, data/v0.1, by the SHA-256s of its questions, gold and
# pilot list. Only a run on these files can be the registered full run or the registered pilot
# run; a test pins them to data/v0.1, so a rebuild that changes a file must change them too.
REGISTERED_DATA = {
    "questions.jsonl": "62b1ba39179be4890c2222e25628273a84722aa7f5218151502bc336e52b0d4c",
    "gold.jsonl": "fe497dd7fca89cd39e1afa6e2f56fb84554747d996a6d692d5a14e815f01da3a",
    "pilot.json": "0e7f02beef3dbb0f5bcf79ea8aeed302ad347d15c88a9b7e23ed222065a42514",
}
GRAPHITI, GRAPHITI_REDUCED = "graphiti", "graphiti-reduced"
# Section 11: the entrants that embed text, all with the one embedding model of the run, and those
# whose store a language model builds, the construction model. A run that holds one of them cannot
# name none for that model. ppr is in neither: section 11 does not fix whether its configuration
# embeds or uses a language model to build its graph.
EMBEDDING_SYSTEMS = ("dense", "hybrid", "hipporag2", "graphiti", "graphiti-reduced")
CONSTRUCTED_SYSTEMS = ("hipporag2", "graphiti", "graphiti-reduced")
ROW_FIELDS = ("id", "system", "arm", "run", "answer")  # section 12: every row has these
# Section 12: every row also records these; a row without one, or with no number there, is still
# scored and counted per system in the report, and the cost figures leave its value out.
ACCOUNTING = ("tokens_in", "tokens_out", "wall_s", "tool_calls")
LONG_CONTEXT = "long-context"
# Sections 11 and 13: the families a system is entered in when its entries are not declared.
# Long context reads the episodes of the proposal a question names, and graphiti-reduced ingests
# the proposals the F1, F2 and CF questions name; F3, F4 and F5 need proposals no question names.
# Every other system is entered in every arm and family.
REGISTERED_FAMILIES = {LONG_CONTEXT: frozenset({"F1", "F2", "CF"}),
                       GRAPHITI_REDUCED: frozenset({"F1", "F2", "CF"})}
_MINUS = chr(0x2212)  # the minus sign in the registration's "H-"

_SERIES_ITEM = re.compile(r"(?:(?:PEP|EIP|ERC)[ \t-]*)?([0-9]+)", re.IGNORECASE | re.ASCII)
_STAMP = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
# The secondary shares of each answer type (section 12); a group with no answer has 0 for these.
_OWN_SHARES = {"status": ("outcome_correct",), "set": ("precision", "recall", "f1"),
               "count": ("within_one",)}


class ScoringError(Exception):
    """An input problem that ``main`` reports instead of a traceback."""


# --- statistics ----------------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """The Wilson score interval for ``k`` successes in ``n`` trials; ``(0.0, 1.0)`` if n is 0."""
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    if n == 0:
        return 0.0, 1.0
    p = k / n
    z2 = z * z
    denominator = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator
    low = 0.0 if k == 0 else max(0.0, centre - half)
    high = 1.0 if k == n else min(1.0, centre + half)
    return low, high


def mcnemar_exact(b: int, c: int) -> float:
    """The two-sided exact McNemar p-value for ``b`` and ``c`` discordant pairs.

    It is the two-sided exact binomial p-value of the smaller count out of ``b + c`` at
    probability one half, capped at 1.0; with no discordant pair it is 1.0.
    """
    if b < 0 or c < 0:
        raise ValueError(f"discordant counts cannot be negative, got b={b}, c={c}")
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, 2 * tail / 2**n)


def binomial_tail(k: int, n: int, p: Fraction) -> float:
    """P(X >= k) for X ~ Binomial(n, p): the exact one-sided binomial p-value of ``k`` successes
    in ``n`` trials against the rate ``p`` (section 10).

    The sum is taken in rational arithmetic, with ``math.comb``, and rounded to a float once.
    """
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    p = Fraction(p)
    if not 0 <= p <= 1:
        raise ValueError(f"a rate is between 0 and 1, got {p}")
    return float(sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def holm(pvalues: list[float]) -> list[float]:
    """Holm's step-down adjustment of ``pvalues``, returned in the order given (gate 4).

    The i-th smallest of m p-values is multiplied by m - i + 1 (the smallest by m), capped at 1,
    and the results are made monotone: none is below the adjusted value of a smaller raw p-value.
    """
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(sorted(range(len(pvalues)), key=lambda i: pvalues[i])):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


# --- one answer ----------------------------------------------------------------------------------


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _member(value: object) -> int | None:
    """The proposal number a set-answer item stands for, or None if it is not readable."""
    if _is_int(value):
        return value  # type: ignore[return-value]
    if isinstance(value, str):
        match = _SERIES_ITEM.fullmatch(value.strip())
        if match is not None:
            try:
                return int(match.group(1))
            except ValueError:  # more digits than int() will read
                return None
    return None


def _result(correct, valid, *, outcome=None, prf=None, error=None, within=None) -> dict:
    precision, recall, f1 = prf if prf is not None else (None, None, None)
    return {
        "correct": correct,
        "outcome_correct": outcome,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "abs_error": error,
        "within_one": within,
        "valid": valid,
    }


def _score_status(item: dict, gold: dict, answer: object) -> dict:
    if not isinstance(answer, str):
        return _result(False, False, outcome=False)
    family = item["corpus"]
    named = normalize_answer_status(family, answer)
    expected = gold["gold"]
    expected_class = gold.get("gold_outcome") or outcome_class(family, expected)
    got_class = outcome_class(family, named) if named is not None else None
    return _result(
        named is not None and named == expected,
        True,
        outcome=got_class is not None and got_class == expected_class,
    )


def _score_set(gold: dict, answer: object) -> dict:
    members = None
    if isinstance(answer, list):
        found = [_member(x) for x in answer]
        if None not in found:
            members = set(found)
    if members is None:
        return _result(False, False, prf=(0.0, 0.0, 0.0))
    expected = set(gold["gold"])
    if not expected and not members:
        return _result(True, True, prf=(1.0, 1.0, 1.0))
    hits = len(members & expected)
    precision = hits / len(members) if members else 0.0
    recall = hits / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return _result(members == expected, True, prf=(precision, recall, f1))


def _score_count(gold: dict, answer: object) -> dict:
    if not _is_int(answer):
        return _result(False, False, within=False)
    error = abs(answer - gold["gold"])  # type: ignore[operator]
    return _result(answer == gold["gold"], True, error=error, within=error <= 1)


def score_answer(item: dict, gold: dict, answer: object) -> dict:
    """The scores of one answer to one item (section 12).

    ``item`` is the item's record in ``questions.jsonl`` (its ``answer_type`` and, for a status,
    its ``corpus``) and ``gold`` its record in ``gold.jsonl``.
    """
    kind = item["answer_type"]
    if kind == "status":
        return _score_status(item, gold, answer)
    if kind == "set":
        return _score_set(gold, answer)
    if kind == "count":
        return _score_count(gold, answer)
    raise ValueError(f"unknown answer_type {kind!r}: expected status, set or count")


# --- aggregation ---------------------------------------------------------------------------------


def run_label(value: object) -> str | None:
    """A run as a string, so that 1 and "1" are one run; None for a value that is no run label
    (a boolean, a float, anything that is neither an integer nor a string)."""
    if _is_int(value):
        return str(value)
    if isinstance(value, str):
        return value.strip()
    return None


def _grid(rows: list[dict], denominators: dict | None = None) -> dict[tuple, dict[str, dict]]:
    """The answers by group, item and run label.

    Every system and arm counts runs 1, 2 and 3 (``RUNS``, section 12); an answer with another run
    label is left out, and of several answers for one run the first stands. ``denominators`` maps a
    system to its arms, and each arm to the ``(id, family, stratum, corpus)`` of every item that
    counts for it there. Each system then has all of those items in its groups, answered or not,
    in every arm it is entered in, answered in or not.
    """
    cells: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        run = run_label(r["run"])
        if run not in RUNS:
            continue
        by_run = cells[tuple(r[k] for k in GROUP)].setdefault(r["id"], {})
        by_run.setdefault(run, r)  # the first answer for a run stands
    for system, arms in (denominators or {}).items():
        for arm, items in arms.items():
            for item_id, family, stratum, corpus in items:
                cells[(system, arm, family, stratum, corpus)].setdefault(item_id, {})
    return cells


def _missing_runs(rows: list[dict]) -> list[dict]:
    """For each system and arm it answered in, the runs of 1, 2 and 3 with no answer there: a run
    that was never made, whose answers all count as wrong."""
    labels: dict[tuple, set] = defaultdict(set)
    for r in rows:
        labels[(r["system"], r["arm"])].add(run_label(r["run"]))
    return [{"system": system, "arm": arm, "runs": [run for run in RUNS if run not in found]}
            for (system, arm), found in sorted(labels.items())
            if any(run not in found for run in RUNS)]


def _majority(flags: list[bool]) -> bool:
    return 2 * sum(flags) > len(flags)


def _number(value: object) -> int | float | None:
    """A finite number that is not a boolean; None for anything else, a missing value included."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return None if isinstance(value, float) and not math.isfinite(value) else value


def _calls(value: object) -> int | float | None:
    """An answer's ``tool_calls`` as a number: a count, or 1 or 0 for a true or false flag; None
    when it is missing or no number."""
    return int(value) if isinstance(value, bool) else _number(value)


def _values(rows: list[dict], key: str, read=_number) -> list[int | float]:
    """The values of ``key`` that ``rows`` carry as numbers: a row without one is left out of a
    cost figure, not counted as zero (section 12)."""
    return [v for r in rows if (v := read(r.get(key))) is not None]


def _total(values: list[int | float]) -> int | float:
    """A sum the scorer writes: exact for integers, and ``math.fsum``'s correctly rounded sum when
    any value is a float, so that results.json is the same on Python 3.11 and on 3.12 or later
    (3.12 changed how the built-in ``sum`` adds floats)."""
    return sum(values) if all(isinstance(v, int) for v in values) else math.fsum(values)


def _flags(by_run: dict[str, dict], runs: tuple[str, ...] = RUNS) -> list[bool]:
    """Whether the item was answered correctly in each run; a run without an answer is wrong."""
    return [bool(by_run[run]["correct"]) if run in by_run else False for run in runs]


def aggregate(rows: list[dict], denominators: dict | None = None) -> list[dict]:
    """One row per (system, arm, family, stratum, corpus) from scored per-run rows.

    Each row needs ``id``, ``system``, ``arm``, ``run``, ``family``, ``stratum``, ``corpus`` and
    ``correct``; ``tokens_in``, ``tokens_out`` and ``tool_calls`` are read if present. Every group
    counts runs 1, 2 and 3. With ``denominators`` (system -> arm -> items, see ``_grid``) an item
    that has no answer is in its group and wrong in every run, and a system entered in an arm it
    never answered has its rows there too, scored zero. The token totals and the tool-use rate are
    over the answers that carry each value as a number; each is None when none does.
    """
    cells = _grid(rows, denominators)
    out = []
    for group in sorted(cells):
        runs = RUNS
        items = cells[group]
        flags = [_flags(by_run, runs) for by_run in items.values()]
        n = len(items)
        present = [r for by_run in items.values() for r in by_run.values()]
        calls = _values(present, "tool_calls", _calls)
        tokens_in, tokens_out = _values(present, "tokens_in"), _values(present, "tokens_out")
        majority = sum(_majority(f) for f in flags)
        low, high = wilson(majority, n)
        out.append(
            dict(
                zip(GROUP, group, strict=True),
                n_items=n,
                runs=len(runs),
                accuracy_mean_over_runs=sum(sum(f) for f in flags) / (n * len(runs)),
                majority_accuracy=majority / n,
                wilson_low=low,
                wilson_high=high,
                pass_at_3=sum(any(f) for f in flags) / n,
                all_3=sum(all(f) for f in flags) / n,
                tokens_in=_total(tokens_in) if tokens_in else None,
                tokens_out=_total(tokens_out) if tokens_out else None,
                tool_use_rate=sum(1 for c in calls if c > 0) / len(calls) if calls else None,
            )
        )
    return out


def _mean(values: list[float]) -> float | None:
    """The mean of ``values``, summed with ``math.fsum`` (``_total``)."""
    return math.fsum(values) / len(values) if values else None


def secondary(rows: list[dict], denominators: dict | None = None) -> list[dict]:
    """The secondary scores of section 12 and the wall time, per group.

    The shares have the grid of items and runs as their denominator, so a cell with no answer
    counts as wrong and invalid, and a group with no answer at all has 0 for the shares of its
    family's answer type (None for the others); the mean absolute error is over the valid answers
    and the wall time over the answers that carry it as a number. ``denominators`` are those of
    ``aggregate``.
    """
    cells = _grid(rows, denominators)
    out = []
    for group in sorted(cells):
        items = cells[group]
        grid = len(items) * len(RUNS)
        present = [r for by_run in items.values() for r in by_run.values()]
        own = _OWN_SHARES.get(ANSWER_TYPES.get(group[2]), ())

        def share(key, present=present, grid=grid, own=own):
            kept = [r for r in present if r.get(key) is not None]
            if kept:
                return math.fsum(float(r[key]) for r in kept) / grid
            return 0.0 if key in own else None

        out.append(
            dict(
                zip(GROUP, group, strict=True),
                n_items=len(items),
                runs=len(RUNS),
                outcome_accuracy=share("outcome_correct"),
                precision=share("precision"),
                recall=share("recall"),
                f1=share("f1"),
                mean_abs_error=_mean([r["abs_error"] for r in present
                                      if r.get("abs_error") is not None]),
                within_one_rate=share("within_one"),
                invalid_rate=1 - sum(1 for r in present if r.get("valid", True)) / grid,
                mean_wall_s=_mean(_values(present, "wall_s")),
            )
        )
    return out


def _majorities(rows: list[dict], denominators: dict | None = None) -> dict[str, dict[str, bool]]:
    """Each system's majority result per item, from the scored rows of one arm; an item with no
    answer, if the denominators name it, is wrong."""
    cells = _grid(rows, denominators)
    found: dict[str, dict[str, bool]] = defaultdict(dict)
    for group in sorted(cells):
        for item_id, by_run in cells[group].items():
            found[group[0]][item_id] = _majority(_flags(by_run))
    return dict(found)


# --- the pilot gates -----------------------------------------------------------------------------


def _with_graphiti(names: tuple[str, ...], graphiti: str) -> tuple[str, ...]:
    """``names`` with Graphiti's id as the run enters it: graphiti, or graphiti-reduced under
    M0's fallback (section 13)."""
    return tuple(graphiti if name == GRAPHITI else name for name in names)


def pilot_systems(graphiti: str = GRAPHITI) -> tuple[str, ...]:
    """Section 13's six pilot systems, with ``graphiti`` as the pilot's Graphiti."""
    return _with_graphiti(PILOT_SYSTEMS, graphiti)


def _pooled(rows: list[dict], measure: str,
            systems: tuple[str, ...] = PILOT_SYSTEMS) -> dict[str, dict[str, float]]:
    """The H- F1 pilot rows' score per stratum and system, corpora pooled by their item counts
    (the weighted sums taken with ``math.fsum``)."""
    weighted: dict[tuple[str, str], tuple[list, list]] = defaultdict(lambda: ([], []))
    for r in rows:
        if r.get("arm", "H-") != "H-" or r.get("family", "F1") != "F1":
            continue
        if r["system"] not in systems or r.get("stratum") not in ("change", "current"):
            continue
        value = r.get(measure, r.get("majority_accuracy"))
        if value is None:
            raise ValueError(f"a pilot row for {r['system']} has no {measure}")
        weight = r.get("n_items", 1)
        totals, weights = weighted[(r["stratum"], r["system"])]
        totals.append(value * weight)
        weights.append(weight)
    pooled: dict[str, dict[str, float]] = {"change": {}, "current": {}}
    for (stratum, system), (totals, weights) in weighted.items():
        pooled[stratum][system] = math.fsum(totals) / math.fsum(weights)
    return pooled


def _discordant(per_item: dict, a: str, b: str) -> tuple[int, int, int]:
    """(a right and b wrong, a wrong and b right, items both answered)."""
    common = per_item.get(a, {}).keys() & per_item.get(b, {}).keys()
    only_a = sum(1 for i in common if per_item[a][i] and not per_item[b][i])
    only_b = sum(1 for i in common if per_item[b][i] and not per_item[a][i])
    return only_a, only_b, len(common)


def _gate1(change: dict[str, float], systems: tuple[str, ...] = PILOT_SYSTEMS) -> dict:
    below = [s for s in systems if change[s] < CEILING - EPS]
    return {"threshold": CEILING, "evaluable": True, "missing": [], "below": below,
            "stop": not below}


def _gate2(change: dict[str, float], current: dict[str, float], graphiti: str = GRAPHITI) -> dict:
    engines, missing = {}, []
    for s in _with_graphiti(ENGINES, graphiti):
        if s in current:
            fail_change, fail_current = 1 - change[s], 1 - current[s]
            gap = abs(fail_current - fail_change)
            engines[s] = {"fail_change": fail_change, "fail_current": fail_current, "gap": gap,
                          "within": gap <= LOADING_MARGIN + EPS}
        else:
            missing.append(f"{s} F1-current")
    return {"margin": LOADING_MARGIN, "evaluable": not missing, "missing": missing,
            "engines": engines,
            "stop": not missing and all(e["within"] for e in engines.values())}


def _gate3(change: dict[str, float], per_item: dict, graphiti: str = GRAPHITI) -> dict:
    """Saturation: grep at D and long context both at 85% or more, and Graphiti (``graphiti``,
    its id in the run) does not differ significantly from the better of the two (two-sided exact
    McNemar p of 0.05 or more).

    If the two score alike, Graphiti is compared with each, and the gate stops only if it differs
    significantly from neither. The test is two-sided, so a Graphiti that is significantly better
    than a saturated baseline differs from it and the gate does not stop. The McNemar tests are
    needed only if both baselines are saturated.
    """
    saturated = all(change[s] >= CEILING - EPS for s in SATURATING)
    gate = {"threshold": CEILING, "alpha": ALPHA, "stop": False, "evaluable": True,
            "missing": [], "saturated": saturated}
    if not saturated:
        return gate
    missing = [f"{s} per-item results" for s in (*SATURATING, graphiti) if s not in per_item]
    if missing:
        return {**gate, "evaluable": False, "missing": missing}
    best = max(change[s] for s in SATURATING)
    better = [s for s in SATURATING if change[s] >= best - EPS]
    tests = {}
    for s in better:
        only_s, only_graphiti, common = _discordant(per_item, s, graphiti)
        tests[s] = {"b": only_s, "c": only_graphiti, "items": common,
                    "p": mcnemar_exact(only_s, only_graphiti)}
    return {**gate, "better": better, "tests": tests,
            "stop": all(t["p"] >= ALPHA for t in tests.values())}


def _gate4(change: dict[str, float], per_item: dict,
           systems: tuple[str, ...] = PILOT_SYSTEMS) -> dict:
    """Signal: at least one of the ten pairs of pilot systems without closed book differs by 15
    points or more with an exact McNemar p below 0.05 once the ten p-values are Holm-adjusted.

    The unadjusted reading over all fifteen pairs, closed book's included, is kept beside it
    (``unadjusted``: each pair, and how many qualify) and decides nothing.
    """
    missing = [f"{s} per-item results" for s in systems if s not in per_item]
    gate = {"min_gap": SIGNAL_GAP, "alpha": ALPHA, "adjustment": "holm",
            "evaluable": not missing, "missing": missing, "pairs": [], "signal": False,
            "unadjusted": {"pairs": [], "qualifying": 0}}
    if missing:
        return gate
    tested = []
    for a, b in combinations(systems, 2):
        only_a, only_b, common = _discordant(per_item, a, b)
        gap = abs(change[a] - change[b])
        tested.append({"a": a, "b": b, "gap": gap, "b_count": only_a, "c_count": only_b,
                       "items": common, "p": mcnemar_exact(only_a, only_b)})
    counted = [t for t in tested if CLOSED_BOOK not in (t["a"], t["b"])]  # SIGNAL_PAIRS' ten
    for t, adjusted in zip(counted, holm([t["p"] for t in counted]), strict=True):
        gate["pairs"].append({**t, "p_holm": adjusted,
                              "qualifies": t["gap"] >= SIGNAL_GAP - EPS and adjusted < ALPHA})
    gate["signal"] = any(p["qualifies"] for p in gate["pairs"])
    unadjusted = [{**t, "qualifies": t["gap"] >= SIGNAL_GAP - EPS and t["p"] < ALPHA}
                  for t in tested]
    gate["unadjusted"] = {"pairs": unadjusted,
                          "qualifying": sum(1 for t in unadjusted if t["qualifies"])}
    return gate


DECISIONS = {
    1: "stop: every system scores 85% or more on H- F1-change, so the questions are too easy "
       "to show change",
    2: "stop: both engines fail on F1-current about as often as on F1-change, so the benchmark "
       "would be measuring loading rather than change",
    3: "stop: grep at D and long context both score 85% or more and Graphiti does not differ "
       "significantly from the better of the two (two-sided exact McNemar p of 0.05 or more; on a "
       "tie, from each), so the time-travel comparison stops and the project continues as the "
       "section 14 fallback",
    "signal": "continue to the full runs: at least one of the ten pairs of pilot systems without "
              "closed book differs by 15 points or more with a Holm-adjusted exact McNemar p "
              "below 0.05",
    "null": "report the null and continue only with the measurements in section 14",
}


def _not_evaluable(measure, change, current, missing_systems, missing_inputs, gates=None,
                   reasons=()) -> dict:
    """The answer when the pilot cannot be decided (section 13), the one place every such answer
    is written: ``evaluable`` false; ``reasons``, every reason why (the run's own when it is not
    the registered pilot run, ``reasons``, then the pilot systems without answers,
    ``missing_systems``, or else the inputs a gate lacks, ``missing_inputs``); a ``decision`` that
    says it is not evaluable and why, a top-level field as a decided pilot's is (``pilot_gates``);
    and each gate's computation (``gates``) without its
    ``stop`` or ``signal``. There is no ``stopped_at`` or ``signal``, at the top or in any gate,
    which would read as a decision."""
    why = list(reasons)
    if missing_systems:
        why.append(f"no H- F1 answers from {_listed_words(list(missing_systems))}")
    elif missing_inputs:
        why.append(f"no {_listed_words(list(missing_inputs))}")
    tails = [REGISTERED_PILOT] if reasons else []
    if missing_systems or missing_inputs:
        tails.append("the gates need all six pilot systems and their inputs")
    inputs = {name: {key: value for key, value in gate.items() if key not in ("stop", "signal")}
              for name, gate in (gates or {}).items()}
    return {
        "evaluable": False,
        "decision": f"not evaluable: {'; '.join([*why, *tails])}",
        "reasons": why,
        "missing_systems": list(missing_systems),
        "missing_inputs": list(missing_inputs),
        "detail": {"measure": measure, "accuracy_change": change, "accuracy_current": current,
                   **inputs},
    }


def pilot_gates(
    rows: list[dict],
    per_item: dict[str, dict[str, bool]],
    measure: str = "accuracy_mean_over_runs",
    graphiti: str = GRAPHITI,
) -> dict:
    """Apply section 13's four gates in order to the H- F1 pilot (gate 2 also reads F1-current).

    ``rows`` are aggregated rows (``aggregate``) of the pilot items; those of one system and
    stratum in several corpora are pooled by ``n_items``. ``per_item`` maps system to item id to
    that item's majority result on the F1-change items, for the exact McNemar tests. Only the six
    pilot systems are read, with ``graphiti`` as the pilot's Graphiti: graphiti, or
    graphiti-reduced under M0's fallback (section 13).

    The gates are evaluated only when all six have answers, and each gate only when its inputs
    are there (the engines' F1-current rows for gate 2, each item's majority result for gates 3
    and 4); otherwise the pilot is "not evaluable" and the result names what is missing and has
    no ``stopped_at`` or ``signal``. A decided pilot returns ``{"evaluable": True, "decision":
    str, "stopped_at": 1 | 2 | 3 | None, "signal": bool, "detail": {...}}``: in both shapes the
    ``decision`` is a top-level field.
    """
    systems = pilot_systems(graphiti)
    pooled = _pooled(rows, measure, systems)
    change, current = pooled["change"], pooled["current"]
    missing = [s for s in systems if s not in change]
    if missing:
        return _not_evaluable(measure, change, current, missing, [])
    gates = {"gate1": _gate1(change, systems), "gate2": _gate2(change, current, graphiti),
             "gate3": _gate3(change, per_item, graphiti),
             "gate4": _gate4(change, per_item, systems)}
    stopped = None
    for n in (1, 2, 3):
        gate = gates[f"gate{n}"]
        if not gate["evaluable"]:
            return _not_evaluable(measure, change, current, [], gate["missing"], gates)
        if gate["stop"]:
            stopped = n
            break
    if stopped is None and not gates["gate4"]["evaluable"]:
        return _not_evaluable(measure, change, current, [], gates["gate4"]["missing"], gates)
    if stopped is not None:
        decision = DECISIONS[stopped]
    else:
        decision = DECISIONS["signal" if gates["gate4"]["signal"] else "null"]
    return {
        "evaluable": True,
        "decision": decision,  # a top-level field, as in _not_evaluable's shape (section 13)
        "stopped_at": stopped,
        "signal": stopped is None and gates["gate4"]["signal"],
        "detail": {
            "measure": measure,
            "accuracy_change": change,
            "accuracy_current": current,
            **gates,
        },
    }


# --- asofbench score -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Scope:
    """What the run scores (section 12): the families the questions have, what each system that
    an entries file declares is entered in, and for a pilot run the items of a list.

    A system the file does not declare has the registered default: every arm and family, except
    that long context and graphiti-reduced are entered in F1, F2 and CF only (sections 11 and
    13, ``REGISTERED_FAMILIES``).
    """

    families: tuple[str, ...]
    declared: dict[str, tuple[frozenset[str], frozenset[str]]] = field(default_factory=dict)
    items: frozenset[str] | None = None
    items_file: str | None = None
    entries_file: str | None = None

    def default_families(self, system: str) -> frozenset[str]:
        everything = frozenset(self.families)
        narrower = REGISTERED_FAMILIES.get(system)
        return everything if narrower is None else everything & narrower

    def entry(self, system: str) -> tuple[frozenset[str], frozenset[str]]:
        """The arms and the families ``system`` is entered in."""
        if system in self.declared:
            return self.declared[system]
        return frozenset(ARMS), self.default_families(system)

    def contains(self, system: str, arm: str, question: dict) -> bool:
        arms, families = self.entry(system)
        return (arm in arms and question["family"] in families
                and (self.items is None or question["id"] in self.items))


def _arms_of(question: dict) -> tuple[str, ...]:
    return tuple(question.get("arms") or ARMS)


def _json_file(path: Path):
    if not path.is_file():
        raise ScoringError(f"{path} not found")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        raise ScoringError(f"{path} is not JSON") from None


def _names(path: Path, system: str, key: str, value: object) -> list[str]:
    if (not isinstance(value, list) or not value
            or not all(isinstance(v, str) and v.strip() for v in value)):
        raise ScoringError(f"{path}: {key} of {system} must be a list of names, not empty")
    return [_arm(v) if key == "arms" else v.strip() for v in value]


def _entries(path: Path | None, known: list[str]):
    """The entries file (section 12): system -> {"arms": [...], "families": [...]}.

    A key left out means the system's default for it. The file may narrow what a system is
    entered in and not widen it: long context and graphiti-reduced stay within F1, F2 and CF.
    """
    if path is None:
        return {}, None
    path = Path(path)
    content = _json_file(path)
    if not isinstance(content, dict):
        raise ScoringError(f"{path} is not a JSON object of system -> entries")
    base = Scope(tuple(known))
    declared = {}
    for system, entry in content.items():
        if not system or not isinstance(entry, dict):
            raise ScoringError(f"{path}: the entries of {system!r} must be an object with "
                               f"\"arms\" and \"families\"")
        if set(entry) - {"arms", "families"}:
            raise ScoringError(f"{path}: {system} has the unknown key(s) "
                               f"{', '.join(sorted(set(entry) - {'arms', 'families'}))}")
        arms, families = set(ARMS), set(base.default_families(system))
        if "arms" in entry:
            arms = set(_names(path, system, "arms", entry["arms"]))
            if arms - set(ARMS):
                raise ScoringError(f"{path}: unknown arm {', '.join(sorted(arms - set(ARMS)))} "
                                   f"for {system}; the arms are {', '.join(ARMS)}")
        if "families" in entry:
            chosen = set(_names(path, system, "families", entry["families"]))
            if chosen - set(known):
                raise ScoringError(f"{path}: unknown family "
                                   f"{', '.join(sorted(chosen - set(known)))} for {system}; the "
                                   f"questions have {', '.join(known)}")
            if chosen - families:  # ``families`` still holds the system's default
                raise ScoringError(
                    f"{path}: {system} is entered in {', '.join(sorted(families))} only "
                    f"(section 11), not {', '.join(sorted(chosen - families))}")
            families = chosen
        declared[system] = (frozenset(arms), frozenset(families))
    return declared, path.name


def _items(path: Path | None, questions: dict[str, dict]):
    """The items list (``--items``): ids, or pilot.json's structure (its ids, else its groups')."""
    if path is None:
        return None, None
    path = Path(path)
    content = _json_file(path)
    if isinstance(content, dict):
        if isinstance(content.get("ids"), list):
            content = content["ids"]
        elif isinstance(content.get("groups"), list):
            content = [i for g in content["groups"] if isinstance(g, dict)
                       for i in g.get("ids", [])]
        else:
            raise ScoringError(f"{path} is neither a list of item ids nor pilot.json's form")
    if not isinstance(content, list) or not all(isinstance(i, str) for i in content):
        raise ScoringError(f"{path} is not a list of item ids")
    unknown = sorted(set(content) - set(questions))
    if unknown:
        raise ScoringError(f"{path} names {len(unknown)} id(s) that are not in the questions, "
                           f"such as {unknown[0]}")
    return frozenset(content), path.name


def _scope(args, questions: dict[str, dict]) -> Scope:
    """The scope the command line asks for (``--entries``, ``--items``), checked against the
    questions."""
    known = sorted({q["family"] for q in questions.values()})
    declared, entries_file = _entries(getattr(args, "entries", None), known)
    items, items_file = _items(getattr(args, "items", None), questions)
    return Scope(tuple(known), declared, items, items_file, entries_file)


def _denominators(questions: dict[str, dict], scope: Scope, systems) -> dict:
    """For each system, every item of each arm it is entered in, in the families it is entered
    in and, for a pilot run, in the list (section 12)."""
    found: dict[str, dict[str, list]] = {}
    for system in systems:
        by_arm: dict[str, list] = defaultdict(list)
        for q in questions.values():
            for arm in _arms_of(q):
                if scope.contains(system, arm, q):
                    by_arm[arm].append((q["id"], q["family"], q["stratum"], q["corpus"]))
        found[system] = dict(by_arm)
    return found


_QUESTION_FIELDS = ("id", "family", "stratum", "corpus", "answer_type")
_GOLD_FIELDS = ("id", "gold")
_LISTED = 50  # malformed rows an error message spells out


def _read_records(path: Path, fields: tuple[str, ...]) -> dict[str, dict]:
    """A build's JSONL file as records by id; a missing, truncated or damaged file is an error."""
    if not path.is_file():
        raise ScoringError(f"{path} not found: run `asofbench build` first")
    found: dict[str, dict] = {}
    try:
        with path.open(encoding="utf-8", newline="\n") as f:
            for number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    raise ScoringError(f"{path} line {number}: not JSON (a truncated or damaged "
                                       f"file?)") from None
                if (not isinstance(record, dict) or any(k not in record for k in fields)
                        or not isinstance(record["id"], str)):
                    raise ScoringError(f"{path} line {number}: not a record with "
                                       f"{', '.join(fields)}")
                if record["id"] in found:
                    raise ScoringError(f"{path} line {number}: the id {record['id']} again")
                found[record["id"]] = record
    except UnicodeDecodeError:
        raise ScoringError(f"{path} is not UTF-8 text") from None
    return found


def _manifest(data: Path) -> dict | None:
    """The build's manifest, or None for a build that has none."""
    path = data / "MANIFEST.json"
    if not path.is_file():
        return None
    try:
        return read_manifest(path)
    except ManifestError as e:
        raise ScoringError(str(e)) from None


def _check_registered(data: Path, manifest: dict | None) -> None:
    """With a manifest in ``data``, the questions, gold and pilot list must be the files it
    registers (section 15): a truncated or edited file is not scored."""
    if manifest is None:
        return
    path = data / "MANIFEST.json"
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ScoringError(f"{path} lists no files")
    for name in ("questions.jsonl", "gold.jsonl", "pilot.json"):
        entry = files.get(name)
        if entry is None:
            continue
        if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str):
            raise ScoringError(f"{path} gives no SHA-256 for {name}")
        if not (data / name).is_file():
            raise ScoringError(f"{path} registers {name}, which is not in {data}")
        if sha256((data / name).read_bytes()).hexdigest() != entry["sha256"]:
            raise ScoringError(f"{data / name} is not the file {path} registers (its SHA-256 "
                               "differs): a truncated or edited file is not scored")


def _registered_companion(data: Path, manifest: dict | None, arm: str, suffix: str) -> str | None:
    """The SHA-256 the build's manifest registers for the file written beside ``arm``'s stream as
    ``<stream>.<suffix>``, or None when it registers none (no manifest, a build without
    ``--streams``, or a manifest written before those files were registered)."""
    try:
        entry = registered_stream(manifest, arm, data / "MANIFEST.json")
    except ManifestError as e:
        raise ScoringError(str(e)) from None
    sidecars = None if entry is None else entry.get("sidecars")
    return None if sidecars is None else sidecars[suffix]["sha256"]


def _companion(path: Path, registered: str | None, what: str, manifest: Path) -> tuple:
    """A companion file's JSON content, and whether it was checked against the manifest: when
    the build's manifest registers the file, one whose SHA-256 differs is not read (section
    15)."""
    if not path.is_file():
        raise ScoringError(f"{path} not found")
    raw = path.read_bytes()
    if registered is not None and sha256(raw).hexdigest() != registered:
        raise ScoringError(f"{path} is not the {what} that {manifest} registers (its SHA-256 "
                           "differs): give the file written beside the registered stream")
    try:
        return json.loads(raw.decode("utf-8")), registered is not None
    except ValueError:  # a decoding error is a ValueError too
        raise ScoringError(f"{path} is not JSON") from None


def _arm(value: object) -> str:
    return value.replace(_MINUS, "-").strip() if isinstance(value, str) else ""


def _row_problem(raw: object) -> str | None:
    """Why ``raw`` cannot be an answer row (section 12), or None if it can."""
    if not isinstance(raw, dict):
        return "not a JSON object"
    missing = [name for name in ROW_FIELDS if name not in raw]
    if missing:
        return f"missing {', '.join(missing)}"
    if not isinstance(raw["id"], str) or not raw["id"]:
        return f"the id {raw['id']!r} is not a string"
    if not isinstance(raw["system"], str) or not raw["system"].strip():
        return f"the system {raw['system']!r} is not a name"
    if not isinstance(raw["arm"], str) or _arm(raw["arm"]) not in ARMS:
        return f"unknown arm {raw['arm']!r}; the arms are {', '.join(ARMS)}"
    if run_label(raw["run"]) is None:
        return f"the run {raw['run']!r} is neither an integer nor a string"
    return None


def _rows(paths: list[Path]) -> list[tuple[str, int, dict]]:
    """``(file name, line, row)`` of every answer row, files in order of their names and each in
    line order, so that the first of several answers is the same whatever order the files were
    given in. Any malformed row stops the run, and every one is named."""
    rows, malformed = [], []
    for path in sorted(paths, key=lambda p: (p.name, str(p))):
        if not path.is_file():
            raise ScoringError(f"{path} not found")
        # utf-8-sig: a byte-order mark that some tools write before the first row is not part of it
        with path.open(encoding="utf-8-sig", errors="replace") as f:
            for number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except ValueError:
                    malformed.append(f"{path} line {number}: not JSON")
                    continue
                problem = _row_problem(raw)
                if problem is not None:
                    malformed.append(f"{path} line {number}: {problem}")
                    continue
                rows.append((path.name, number, raw))
    if malformed:
        more = len(malformed) - _LISTED
        raise ScoringError(
            f"{len(malformed)} answer row(s) cannot be scored, so nothing is (section 12: each "
            f"row has {', '.join(ROW_FIELDS)}): " + "; ".join(malformed[:_LISTED])
            + (f"; and {more} more" if more > 0 else ""))
    return rows


def _check_systems(rows: list[tuple[str, int, dict]], scope: Scope) -> None:
    """Every system name is a registered entrant (section 11) or one the entries file declares."""
    unknown: dict[str, str] = {}
    for name, number, raw in rows:
        system = raw["system"]
        if system not in REGISTERED_SYSTEMS and system not in scope.declared:
            unknown.setdefault(system, f"{name} line {number}")
    if not unknown:
        return
    named = []
    for system, where in sorted(unknown.items()):
        close = difflib.get_close_matches(system, REGISTERED_SYSTEMS, n=1)
        named.append(f"{system!r} ({where}" + (f"; did you mean {close[0]}?)" if close else ")"))
    raise ScoringError(f"unknown system name(s) {', '.join(named)}. The entrants are "
                       f"{', '.join(REGISTERED_SYSTEMS)} (section 11); declare any other system "
                       "in an entries file (--entries)")


def _graphiti_of(names) -> str:
    """The run's Graphiti, from the system names its rows use and its entries file declares:
    graphiti, or graphiti-reduced, which section 13 enters instead of it when M0's projection
    exceeds the budget. A run that holds both is refused."""
    held = {GRAPHITI, GRAPHITI_REDUCED} & set(names)
    if len(held) == 2:
        raise ScoringError("the run holds both graphiti and graphiti-reduced: Graphiti is entered "
                           "as graphiti-reduced instead of graphiti when M0's projection exceeds "
                           "the budget (section 13), never as both, so score each in a run of "
                           "its own")
    return GRAPHITI_REDUCED if GRAPHITI_REDUCED in held else GRAPHITI


def _check_models(names, embedding_model: str, construction_model: str) -> None:
    """``none`` for the embedding or the construction model says that no system of the run
    embeds text, or builds its store with a language model; a run whose rows or entries file
    name a registered system that does (``EMBEDDING_SYSTEMS``, ``CONSTRUCTED_SYSTEMS``) is
    refused, so that results.json cannot record a model as none that the run used."""
    problems = []
    for flag, value, registered, one, several in (
        ("--embedding-model", embedding_model, EMBEDDING_SYSTEMS, "embeds text", "embed text"),
        ("--construction-model", construction_model, CONSTRUCTED_SYSTEMS,
         "builds its store with a language model", "build their stores with a language model"),
    ):
        held = [system for system in registered if system in set(names)]
        if value.casefold() == "none" and held:
            does = one if len(held) == 1 else several
            problems.append(f"{flag} is none, but the run holds {_listed_words(held)}, which "
                            f"{does} (section 11): give that model's id")
    if problems:
        raise ScoringError("; ".join(problems))


def _answers(paths: list[Path], questions: dict[str, dict], gold: dict[str, dict], scope: Scope):
    """Score every answer row in ``paths``; return the scored rows and what was set aside.

    The report also names the systems of the run: every system that answered an item of the
    questions, whether or not the answer was in its entries, since a system that is entered in an
    arm or family and has no answer there scores zero there (section 12); and every system name
    that any row uses, for the warnings.
    """
    rows = _rows(paths)
    _check_systems(rows, scope)
    scored: list[dict] = []
    seen: set[tuple] = set()
    report = {"rows": len(rows), "duplicates": [], "unasked": [], "out_of_scope": [],
              "odd_runs": [], "unknown": set(), "systems": set(),
              "named": {raw["system"] for _, _, raw in rows}}
    for name, number, raw in rows:
        where = {"file": name, "line": number}
        item_id, system, arm = raw["id"], raw["system"], _arm(raw["arm"])
        run = run_label(raw["run"])
        if item_id not in questions:
            report["unknown"].add(item_id)
            continue
        report["systems"].add(system)
        question = questions[item_id]
        listed = {**where, "id": item_id, "system": system, "arm": arm}
        if arm not in _arms_of(question):
            report["unasked"].append(listed)
            continue
        if not scope.contains(system, arm, question):
            report["out_of_scope"].append(listed)
            continue
        if run not in RUNS:
            report["odd_runs"].append({**listed, "run": run})
            continue
        if (system, arm, item_id, run) in seen:
            report["duplicates"].append({**listed, "run": run})
            continue
        seen.add((system, arm, item_id, run))
        scored.append(
            {
                "id": item_id,
                "system": system,
                "arm": arm,
                "run": run,
                "family": question["family"],
                "stratum": question["stratum"],
                "corpus": question["corpus"],
                **score_answer(question, gold[item_id], raw["answer"]),
                # None where the row lacks the field or has no number there (section 12)
                "tokens_in": _number(raw.get("tokens_in")),
                "tokens_out": _number(raw.get("tokens_out")),
                "wall_s": _number(raw.get("wall_s")),
                "tool_calls": _calls(raw.get("tool_calls")),
            }
        )
    return scored, report


def _accounting(scored: list[dict], systems) -> dict[str, dict]:
    """Per system of the run, its scored rows and how many of them have no accounting: rows that
    lack one of ``ACCOUNTING`` or have a value there that is not a number (section 12), with the
    count per field. Such rows are scored; only the cost figures leave their missing values out."""
    found = {system: {"rows": 0, "without": 0, "by_field": dict.fromkeys(ACCOUNTING, 0)}
             for system in sorted(systems)}
    for r in scored:
        entry = found[r["system"]]
        entry["rows"] += 1
        lacking = [name for name in ACCOUNTING if r[name] is None]
        entry["without"] += bool(lacking)
        for name in lacking:
            entry["by_field"][name] += 1
    return found


def _arms_without_answers(observed: dict, denominators: dict) -> list[dict]:
    """The arms of its entries in which a system has no answer at all. Its rows are there, scored
    zero; this names the arms so that an arm left out entirely is not missed."""
    answered: dict[str, set] = defaultdict(set)
    for system, arm in observed:
        answered[system].add(arm)
    found = []
    for system in sorted(denominators):
        entered = {arm for arm, items in denominators[system].items() if items}
        missing = sorted(entered - answered[system])
        if missing:
            found.append({"system": system, "arms": missing})
    return found


def _coverage(results: list[dict], scored: list[dict]) -> list[dict]:
    """Per group, how many of the items in the denominator the system answered (in any run)."""
    answered: dict[tuple, set] = defaultdict(set)
    for r in scored:
        answered[tuple(r[k] for k in GROUP)].add(r["id"])
    return [
        {**{k: r[k] for k in GROUP},
         "answered": len(answered.get(tuple(r[k] for k in GROUP), ())),
         "asked": r["n_items"]}
        for r in results
    ]


def _counterfactual_split(
    scored: list[dict], gold: dict[str, dict], denominators: dict, key: str = "decision_word"
) -> list[dict]:
    """CF items scored apart by a flag in their gold meta (section 7): ``decision_word``, whether
    the body at D holds a decision word, or ``header_cue``, whether a header field at D other
    than the status records the true status."""

    def tag(item_id: str) -> str:
        return "flagged" if gold[item_id]["meta"][key] else "none"  # _check_flags: always there

    tagged = [dict(r, stratum=tag(r["id"])) for r in scored if r["family"] == "CF"]
    cf_items = {
        system: {arm: [(i, f, tag(i), c) for i, f, _, c in items if f == "CF"]
                 for arm, items in arms.items()}
        for system, arms in denominators.items()
    }
    cf_items = {system: {arm: v for arm, v in arms.items() if v}
                for system, arms in cf_items.items()}
    return [
        {**{k: v for k, v in row.items() if k != "stratum"}, key: row["stratum"] == "flagged"}
        for row in aggregate(tagged, cf_items)
    ]


def _cf_counts(path: Path | None, questions: dict[str, dict], scope: Scope,
               data: Path, manifest: dict | None):
    """The counts of messages left alone for each CF proposal (``--cf-messages``, written by
    ``asofbench stream --arm CF``): proposal id -> count, the file's name, and whether it was
    checked against the manifest (``_companion``). Every CF item's proposal needs a count."""
    if path is None:
        return None, None, False
    path = Path(path)
    registered = _registered_companion(data, manifest, "CF", CF_MESSAGES_FILE)
    content, checked = _companion(path, registered, "CF message-counts file",
                                  data / "MANIFEST.json")
    if not isinstance(content, dict):
        raise ScoringError(f"{path} is not a JSON object of proposal -> count")
    counts = (content["counts"] if isinstance(content.get("counts"), dict)
              else {k: v for k, v in content.items() if k != "canary"})
    bad = sorted(k for k, v in counts.items() if not _is_int(v) or v < 0)
    if bad:
        raise ScoringError(f"{path}: the count for {', '.join(bad[:3])} is not a non-negative "
                           f"integer")
    needed = {q["proposal"] for q in questions.values()
              if q["family"] == "CF" and q.get("proposal")
              and (scope.items is None or q["id"] in scope.items)}
    absent = sorted(needed - set(counts))
    if absent:
        raise ScoringError(f"{path} has no count for {', '.join(absent[:3])}"
                           f"{' and others' if len(absent) > 3 else ''}")
    return counts, path.name, checked


def _cf_items(questions, gold, cells, counts, scope) -> list[dict]:
    """One row per CF item: its proposal, whether its body holds a decision word and whether a
    header field points at its true status, the number of commits whose unedited message
    mentions the proposal and a status value (section 7), and what each system entered in CF
    made of it."""
    per_system: dict[str, dict[str, dict]] = defaultdict(dict)
    for (system, _, family, _, _), by_item in cells.items():
        if family != "CF":
            continue
        for item_id, by_run in by_item.items():
            flags = _flags(by_run)
            per_system[item_id][system] = {"accuracy_mean_over_runs": sum(flags) / len(flags),
                                           "majority_correct": _majority(flags)}
    rows = []
    for item_id in sorted(questions):
        q = questions[item_id]
        if q["family"] != "CF" or (scope.items is not None and item_id not in scope.items):
            continue
        rows.append({
            "id": item_id,
            "proposal": q.get("proposal"),
            "corpus": q["corpus"],
            "decision_word": gold[item_id]["meta"]["decision_word"],
            "header_cue": gold[item_id]["meta"]["header_cue"],
            "cf_messages": None if counts is None else counts.get(q.get("proposal")),
            "systems": dict(sorted(per_system.get(item_id, {}).items())),
        })
    return rows


def _contamination(gold: dict[str, dict], scored: list[dict], denominators: dict) -> dict:
    """Section 10: each corpus's F1-change floor, and closed book's results tested against it.

    A corpus's floor is the share of its F1-change items in the run's scope, those closed book is
    scored on, whose gold is their most common gold value (``majority_class``); in a pilot run
    those are not the stratum's. Closed book's result on an item is the majority of every
    answer it gave the item, over runs 1, 2 and 3 in every arm it answered; a tie, or no answer,
    is wrong. The test is the exact one-sided binomial p of that many right items out of the
    corpus's F1-change items at the floor's rate, and p below 0.05 marks the corpus's F1-change
    as possibly contaminated. F1-change stays the primary as-of result whatever the test says.
    The items are the F1-change items closed book is scored on, so a corpus has an entry only
    when closed book is a system of the run entered in F1. A corpus whose F1-change items closed
    book gave no answer to is not tested: its ``p`` and ``possibly_contaminated`` are None, since
    a verdict from no answer would say nothing.
    """
    items: dict[str, set[str]] = defaultdict(set)
    for entered in denominators.get(CLOSED_BOOK, {}).values():
        for item_id, family, stratum, corpus in entered:
            if (family, stratum) == ("F1", "change"):
                items[corpus].add(item_id)
    answers: dict[str, list[bool]] = defaultdict(list)
    arms: dict[str, set[str]] = defaultdict(set)
    for r in scored:
        if r["system"] == CLOSED_BOOK and (r["family"], r["stratum"]) == ("F1", "change"):
            answers[r["id"]].append(bool(r["correct"]))
            arms[r["corpus"]].add(r["arm"])
    found = {}
    for corpus in CORPORA:
        ids = sorted(items.get(corpus, ()))
        if not ids:
            continue
        value, with_it, _ = majority_class(gold[i]["gold"] for i in ids)
        right = [i for i in ids if 2 * sum(answers.get(i, ())) > len(answers.get(i, ()))]
        answered = sorted(arms.get(corpus, ()))
        p = (binomial_tail(len(right), len(ids), Fraction(with_it, len(ids))) if answered
             else None)
        found[corpus] = {
            "items": len(ids),
            "floor_gold": value,
            "floor_items": with_it,
            "floor": with_it / len(ids),
            "arms": answered,
            "closed_book_right": len(right),
            "closed_book_right_ids": right,
            "p": p,
            "possibly_contaminated": None if p is None else p < ALPHA,
        }
    return found


def _split_sums(cells: dict[tuple, dict[str, dict]], keep, flag_of, key=None,
                value=None) -> dict[tuple, dict]:
    """Per (system, arm, corpus), or per ``key(system, arm, stratum, corpus)``: the items of the
    groups that ``keep(arm, family, stratum, corpus)`` accepts, apart by ``flag_of(item id,
    arm)``, as ``[item count, each item's accuracy over runs 1, 2 and 3]`` for each flag; an
    unanswered item is wrong in every run. With ``value(item id)``, that is kept instead of
    each item's accuracy. ``_share`` sums them."""
    sums: dict[tuple, dict[bool, list]] = {}
    for (system, arm, family, stratum, corpus), by_item in sorted(cells.items()):
        if not keep(arm, family, stratum, corpus):
            continue
        name = (system, arm, corpus) if key is None else key(system, arm, stratum, corpus)
        group = sums.setdefault(name, {True: [0, []], False: [0, []]})
        for item_id, by_run in by_item.items():
            cell = group[bool(flag_of(item_id, arm))]
            cell[0] += 1
            cell[1].append(sum(_flags(by_run)) / len(RUNS) if value is None else value(item_id))
    return sums


def _share(cell: list) -> float | None:
    """``[item count, values]`` as the values' mean, summed with ``math.fsum``; None for no
    item."""
    return math.fsum(cell[1]) / cell[0] if cell[0] else None


def _closed_book_split(cells: dict[tuple, dict[str, dict]], contamination: dict) -> list[dict]:
    """Section 10: each system's F1-change accuracy, averaged over runs 1, 2 and 3, on the items
    closed book gets wrong and on the rest, per arm, in each corpus closed book was tested in (a
    corpus it gave no answer in is not split, since it would put every item on one side)."""
    right = {i for flag in contamination.values() for i in flag["closed_book_right_ids"]}
    tested = {corpus for corpus, flag in contamination.items() if flag["p"] is not None}
    sums = _split_sums(cells, lambda a, f, s, c: (f, s) == ("F1", "change") and c in tested,
                       lambda i, arm: i in right)
    return [{"system": system, "arm": arm, "corpus": corpus,
             "closed_book_wrong_items": g[False][0], "closed_book_wrong_accuracy": _share(g[False]),
             "closed_book_right_items": g[True][0], "closed_book_right_accuracy": _share(g[True])}
            for (system, arm, corpus), g in sums.items()]


def _recent_split(cells: dict[tuple, dict[str, dict]], gold: dict[str, dict]) -> list[dict]:
    """Section 10: each system's F1-recent accuracy, averaged over runs 1, 2 and 3, on the items
    whose gold is the proposal's status at S and on the rest, per arm and corpus."""
    sums = _split_sums(cells, lambda a, f, s, c: (f, s) == ("F1", "recent"),
                       lambda i, arm: gold[i]["meta"]["gold_equals_status_at_s"])
    return [{"system": system, "arm": arm, "corpus": corpus,
             "equals_s_items": g[True][0], "equals_s_accuracy": _share(g[True]),
             "differs_items": g[False][0], "differs_accuracy": _share(g[False])}
            for (system, arm, corpus), g in sums.items()]


def _cutoff_split(cells: dict[tuple, dict[str, dict]], gold: dict[str, dict],
                  cutoff: date) -> list[dict]:
    """Section 10: each system's F1-recent accuracy, averaged over runs 1, 2 and 3, on the items
    whose transition (gold ``meta.interval_start``) was recorded before ``cutoff``, the first day
    after the answering model's training data, and on those recorded on or after it, per arm and
    corpus. An unanswered item is wrong."""
    day = cutoff.isoformat()
    sums = _split_sums(cells, lambda a, f, s, c: (f, s) == ("F1", "recent"),
                       lambda i, arm: gold[i]["meta"]["interval_start"][:10] >= day)
    return [{"system": system, "arm": arm, "corpus": corpus,
             "before_items": g[False][0], "before_accuracy": _share(g[False]),
             "on_or_after_items": g[True][0], "on_or_after_accuracy": _share(g[True])}
            for (system, arm, corpus), g in sums.items()]


def _header_only_split(cells: dict[tuple, dict[str, dict]], gold: dict[str, dict]) -> list[dict]:
    """Section 12: each system's H- F1 accuracy, averaged over runs 1, 2 and 3, per stratum and
    corpus, on the items whose proposal has a header-only H- episode after their date (gold
    ``meta.h_minus_header_only_after_d``: an episode whose diff the withholding empties, which
    shows a stream system that a withheld header changed) and on the rest. An unanswered item is
    wrong."""
    sums = _split_sums(cells, lambda a, f, s, c: (a, f) == ("H-", "F1"),
                       lambda i, arm: gold[i]["meta"]["h_minus_header_only_after_d"],
                       key=lambda system, arm, stratum, corpus: (system, stratum, corpus))
    return [{"system": system, "stratum": stratum, "corpus": corpus,
             "flagged_items": g[True][0], "flagged_accuracy": _share(g[True]),
             "other_items": g[False][0], "other_accuracy": _share(g[False])}
            for (system, stratum, corpus), g in sums.items()]


def _f2_split(cells: dict[tuple, dict[str, dict]], gold: dict[str, dict]) -> list[dict]:
    """Section 12: each system's F2 accuracy, averaged over runs 1, 2 and 3, over all F2 items of
    an arm and over those whose gold the asked PEP's own episodes can show in that arm (the build
    records it per arm in ``meta.recoverable``), each beside what always answering the empty list
    scores on the same items: the share of them whose gold is empty, the negatives. An unanswered
    item is wrong."""
    def keep(arm, family, stratum, corpus):
        return family == "F2"

    def recoverable(item_id, arm):
        return gold[item_id]["meta"]["recoverable"][arm]

    sums = _split_sums(cells, keep, recoverable)
    floors = _split_sums(cells, keep, recoverable, value=lambda i: float(not gold[i]["gold"]))
    rows = []
    for (system, arm, corpus), g in sums.items():
        f = floors[(system, arm, corpus)]
        both = [g[True][0] + g[False][0], g[True][1] + g[False][1]]
        rows.append({"system": system, "arm": arm, "corpus": corpus,
                     "items": both[0], "accuracy": _share(both),
                     "always_empty": _share([both[0], f[True][1] + f[False][1]]),
                     "recoverable_items": g[True][0], "recoverable_accuracy": _share(g[True]),
                     "recoverable_always_empty": _share(f[True])})
    return rows


def _is_bool(value: object) -> bool:
    return isinstance(value, bool)


def _per_arm(value: object) -> bool:
    return isinstance(value, dict) and all(isinstance(value.get(arm), bool) for arm in ("H+", "H-"))


# The flags in an item's gold meta that the report reads, by family and stratum (None for every
# stratum of the family): the build writes them, and an item in the scope without one stops the
# run rather than being counted in either group.
_META_FLAGS = {("F1", None): {"h_minus_header_only_after_d": _is_bool},
               ("F1", "recent"): {"gold_equals_status_at_s": _is_bool},
               ("F2", None): {"recoverable": _per_arm},
               ("CF", None): {"decision_word": _is_bool, "header_cue": _is_bool}}


def _check_flags(path: Path, questions: dict[str, dict], gold: dict[str, dict], scope) -> None:
    lacking = []
    for item_id in sorted(questions):
        q = questions[item_id]
        if scope.items is not None and item_id not in scope.items:
            continue
        wanted = {**_META_FLAGS.get((q["family"], None), {}),
                  **_META_FLAGS.get((q["family"], q["stratum"]), {})}
        meta = gold[item_id].get("meta")
        meta = meta if isinstance(meta, dict) else {}
        lacking += [f"{item_id} ({key})" for key, valid in wanted.items()
                    if not valid(meta.get(key))]
    if lacking:
        raise ScoringError(f"{path}: {len(lacking)} item(s) lack a flag in their gold meta that "
                           f"the report reads, such as {', '.join(lacking[:3])}; build the data "
                           "with this version of asofbench")


def _check_transition_times(path: Path, questions: dict[str, dict], gold: dict[str, dict],
                            scope) -> None:
    """With --training-cutoff, every F1-recent item of the scope needs the time its transition
    was recorded (gold ``meta.interval_start``), which the split by the cutoff reads."""
    lacking = []
    for item_id in sorted(questions):
        q = questions[item_id]
        if (q["family"], q["stratum"]) != ("F1", "recent"):
            continue
        if scope.items is not None and item_id not in scope.items:
            continue
        meta = gold[item_id].get("meta")
        start = meta.get("interval_start") if isinstance(meta, dict) else None
        if not isinstance(start, str) or _STAMP.fullmatch(start) is None:
            lacking.append(item_id)
    if lacking:
        raise ScoringError(f"{path}: {len(lacking)} F1-recent item(s) lack the time their "
                           f"transition was recorded (meta.interval_start, written "
                           f"YYYY-MM-DDTHH:MM:SSZ), which --training-cutoff reads, such as "
                           f"{', '.join(lacking[:3])}")


def _message_cues(path: Path | None, data: Path, manifest: dict | None):
    """The H- message cues (``--message-cues``, written by ``asofbench stream --arm H-``):
    proposal id -> ``[recorded_at, status]`` pairs, the file's name, and whether it was checked
    against the manifest (``_companion``)."""
    if path is None:
        return None, None, False
    path = Path(path)
    registered = _registered_companion(data, manifest, "H-", CUES_FILE)
    content, checked = _companion(path, registered, "H- message-cues file",
                                  data / "MANIFEST.json")
    if not isinstance(content, dict):
        raise ScoringError(f"{path} is not a JSON object of proposal -> [recorded_at, status] "
                           "pairs")
    cues = {}
    for pid, pairs in content.items():
        if pid == "canary":
            continue
        if not isinstance(pairs, list) or not all(
            isinstance(pair, list) and len(pair) == 2 and all(isinstance(x, str) for x in pair)
            and _STAMP.fullmatch(pair[0]) for pair in pairs
        ):
            raise ScoringError(f"{path}: the cues of {pid} are not [recorded_at, status] pairs "
                               "with a time written YYYY-MM-DDTHH:MM:SSZ")
        cues[pid] = pairs
    return cues, path.name, checked


def _cue_items(questions, gold, cues, scope) -> list[dict]:
    """Section 12: for each H- F1 item, whether a message kept in the stream at or before the end
    of its date names its proposal together with its gold status."""
    rows = []
    for item_id in sorted(questions):
        q = questions[item_id]
        if (q["family"] != "F1" or "H-" not in _arms_of(q)
                or (scope.items is not None and item_id not in scope.items)):
            continue
        until = f"{q['as_of']}T23:59:59Z"
        expected = gold[item_id]["gold"]
        rows.append({
            "id": item_id, "proposal": q.get("proposal"), "corpus": q["corpus"],
            "stratum": q["stratum"], "as_of": q["as_of"], "gold": expected,
            "named": any(when <= until and status == expected
                         for when, status in cues.get(q.get("proposal"), [])),
        })
    return rows


def _cue_summary(items: list[dict], name: str) -> dict:
    groups: dict[str, dict[str, int]] = {}
    for item in items:
        group = groups.setdefault(f"{item['stratum']}/{item['corpus']}", {"items": 0, "named": 0})
        group["items"] += 1
        group["named"] += item["named"]
    return {"file": name, "items": len(items), "named": sum(i["named"] for i in items),
            "by_stratum": dict(sorted(groups.items()))}


def _cue_split(cells: dict[tuple, dict[str, dict]], items: list[dict]) -> list[dict]:
    """Section 12: each system's H- F1 accuracy, averaged over runs 1, 2 and 3, on the items whose
    gold status a kept message names by their date and on the other H- F1 items, so that an H-
    score can be read against what the messages alone reveal. Unanswered items are wrong."""
    named = {item["id"]: item["named"] for item in items}
    sums: dict[str, dict[bool, list]] = {}
    for (system, arm, family, _, _), by_item in sorted(cells.items()):
        if arm != "H-" or family != "F1":
            continue
        for item_id, by_run in by_item.items():
            if item_id in named:
                group = sums.setdefault(system, {True: [0, []], False: [0, []]})[named[item_id]]
                group[0] += 1
                group[1].append(sum(_flags(by_run)) / len(RUNS))
    return [{"system": system, "named_items": g[True][0], "named_accuracy": _share(g[True]),
             "other_items": g[False][0], "other_accuracy": _share(g[False])}
            for system, g in sorted(sums.items())]


# --- the headline comparison (section 12) --------------------------------------------------------

# Section 12: the headline compares each system that keeps or retrieves context from the stream or
# its proposals with grep at D, the version-control baseline, on the H- F1-change items of each
# corpus. Closed book and the oracle are not compared.
HEADLINE_BASELINE = "grep-at-d"
HEADLINE_SYSTEMS = ("bm25", "dense", "hybrid", "ppr", "hipporag2", "graphiti", "long-context")
# Holm's family, in every run and in each reading: the 14 registered comparisons, each compared
# system on each corpus. A comparison the run does not make counts as p = 1, so the family does
# not shrink with the systems a run holds, and leaving systems out of a run never lowers the
# adjusted p of a comparison that remains.
HEADLINE_FAMILY = len(HEADLINE_SYSTEMS) * len(CORPORA)
HEADLINE_GROUP = ("H-", "F1", "change")  # the arm, family and stratum it reads
ABOVE, BELOW, NO_DIFFERENCE = "above grep at D", "below grep at D", "no detectable difference"
# Why a comparison is not made. It is made only when the compared system and grep at D both have a
# scored answer (in the scope and the system's entries, on run 1, 2 or 3, and not set aside as a
# repeat) on the corpus's H- F1-change items, or, in a reading, on the corpus's items left in it.
NO_ITEMS = "no H- F1-change item of the corpus is in the run's scope"
NONE_LEFT = "no H- F1-change item of the corpus is left in this reading"
NOT_ENTERED = "left out of H- F1 by the entries file"
NO_ANSWER = "no scored answer to the corpus's H- F1-change items"
NO_ANSWER_LEFT = "no scored answer to the corpus's items left in this reading"
THE_BASELINE = f"{HEADLINE_BASELINE}, the baseline,"  # how a reason names grep at D's own cause
# What a claim's direction is read from: the majority results the McNemar test compares.
DIRECTION = ("majority results: items the system gets right and grep at D wrong, against the "
             "reverse")


def _runs_right(cells: dict, system: str, corpus: str, ids: list[str]) -> list[list[bool]]:
    """Whether ``system`` was right in each of runs 1, 2 and 3 on each of ``ids``, its H-
    F1-change items of ``corpus``; a run without an answer is wrong."""
    group = cells.get((system, *HEADLINE_GROUP, corpus), {})
    return [_flags(group.get(item_id, {})) for item_id in ids]


def _runs_answered(cells: dict, system: str, corpus: str, ids: list[str]) -> int:
    """How many of the ``3 * len(ids)`` runs ``system`` answered on ``ids``."""
    group = cells.get((system, *HEADLINE_GROUP, corpus), {})
    return sum(run in group.get(item_id, {}) for item_id in ids for run in RUNS)


def _comparison(cells: dict, system: str, corpus: str, ids: list[str]) -> dict:
    """One headline comparison: ``system`` against grep at D on the items ``ids`` of
    ``corpus``, by accuracy averaged over the runs and by the exact two-sided McNemar test on
    the items' majority results (right in at least two of the three runs), with the share of
    items by which the system's majority results lead (``majority_difference``) and the runs
    each side answered, out of three per item."""
    mine = _runs_right(cells, system, corpus, ids)
    theirs = _runs_right(cells, HEADLINE_BASELINE, corpus, ids)
    runs = len(ids) * len(RUNS)
    accuracy, baseline = sum(map(sum, mine)) / runs, sum(map(sum, theirs)) / runs
    pairs = [(_majority(a), _majority(b)) for a, b in zip(mine, theirs, strict=True)]
    only_system = sum(1 for a, b in pairs if a and not b)
    only_baseline = sum(1 for a, b in pairs if b and not a)
    return {"system": system, "corpus": corpus, "items": len(ids), "accuracy": accuracy,
            "baseline_accuracy": baseline, "difference": accuracy - baseline,
            "only_system_right": only_system, "only_baseline_right": only_baseline,
            "majority_difference": (only_system - only_baseline) / len(ids),
            "runs_answered": _runs_answered(cells, system, corpus, ids),
            "baseline_runs_answered": _runs_answered(cells, HEADLINE_BASELINE, corpus, ids),
            "p": mcnemar_exact(only_system, only_baseline)}


def _claimed(comparisons: list[dict], key: str = "claim") -> list[dict]:
    """Each comparison with its Holm-adjusted p and its claim (``key``; a run that is not the
    registered full run, and a reading beside the headline, which decide nothing, write
    ``reading``): above grep at D when the adjusted p is below 0.05 and the system is right where
    grep at D is wrong on more items than the reverse, below when the adjusted p is below 0.05
    and the reverse holds, and otherwise no detectable difference. The adjustment is over the
    ``HEADLINE_FAMILY`` registered comparisons, in which each one that is not among
    ``comparisons`` counts as p = 1. The direction is read from the majority results the McNemar
    test compares, not from the accuracy averaged over the runs; an adjusted p below 0.05 always
    has unequal counts, since equal counts give p = 1."""
    if len(comparisons) > HEADLINE_FAMILY:
        raise ValueError(f"{len(comparisons)} comparisons, but the registered family holds "
                         f"{HEADLINE_FAMILY}")
    family = [c["p"] for c in comparisons] + [1.0] * (HEADLINE_FAMILY - len(comparisons))
    for c, adjusted in zip(comparisons, holm(family)[:len(comparisons)], strict=True):
        direction = 0
        if adjusted < ALPHA:
            direction = (c["only_system_right"] > c["only_baseline_right"]) - (
                c["only_system_right"] < c["only_baseline_right"])
        c["p_holm"] = adjusted
        c[key] = {1: ABOVE, -1: BELOW, 0: NO_DIFFERENCE}[direction]
    return comparisons


def _has_scored_answer(cells: dict, system: str, corpus: str, ids: list[str]) -> bool:
    """Whether ``system`` has a scored answer, in any of runs 1, 2 and 3, to one of ``ids``, its
    H- F1-change items of ``corpus``: a row in the scope and the system's entries, on a registered
    run label, and the first for its run (``_answers`` sets the others aside), the runs that
    ``_runs_answered`` counts."""
    return _runs_answered(cells, system, corpus, ids) > 0


def _headline_plan(cells: dict, questions: dict[str, dict], scope: Scope,
                   graphiti: str = GRAPHITI) -> dict:
    """Which of section 12's comparisons the run makes: the compared systems, with ``graphiti`` in
    Graphiti's place (graphiti, or graphiti-reduced under M0's fallback), each corpus's H-
    F1-change items in the run's scope (``ids``), the comparisons ``made``, and those ``not_made``
    with why. A comparison is made only when the corpus has such items and both grep at D and the
    system are entered in H- F1 and have a scored answer to one of them (``_has_scored_answer``);
    grep at D's cause, found first, is the cause of every comparison on that corpus. ``incomplete``
    is each side of a comparison made, grep at D first on each corpus and then each compared
    system, whose scored answers leave out a run of the corpus's compared items (``_runs_answered``
    against three runs per item), with the runs it answered: the registered full run has none."""
    arm, family, stratum = HEADLINE_GROUP
    compared = _with_graphiti(HEADLINE_SYSTEMS, graphiti)
    ids = {corpus: sorted(i for i, q in questions.items()
                          if (q["family"], q["stratum"], q["corpus"]) == (family, stratum, corpus)
                          and arm in _arms_of(q) and (scope.items is None or i in scope.items))
           for corpus in CORPORA}

    def why_not(system: str, corpus: str) -> str | None:
        if not ids[corpus]:
            return NO_ITEMS
        for who, baseline in ((HEADLINE_BASELINE, True), (system, False)):
            if not _entered_in(scope, who, arm, family):
                return f"{THE_BASELINE} is {NOT_ENTERED}" if baseline else NOT_ENTERED
            if not _has_scored_answer(cells, who, corpus, ids[corpus]):
                return f"{THE_BASELINE} has {NO_ANSWER}" if baseline else NO_ANSWER
        return None

    made, not_made = [], []
    for system in compared:
        for corpus in CORPORA:
            reason = why_not(system, corpus)
            if reason is None:
                made.append((system, corpus))
            else:
                not_made.append({"system": system, "corpus": corpus, "reason": reason})
    incomplete = []
    for corpus in CORPORA:
        sides = [system for system, where in made if where == corpus]
        if not sides:
            continue  # no comparison is made on the corpus, so no side of one can be incomplete
        for system in (HEADLINE_BASELINE, *sides):
            answered = _runs_answered(cells, system, corpus, ids[corpus])
            if answered < len(RUNS) * len(ids[corpus]):
                incomplete.append({"system": system, "corpus": corpus, "runs_answered": answered,
                                   "items": len(ids[corpus])})
    return {"compared": compared, "ids": ids, "made": made, "not_made": not_made,
            "incomplete": incomplete}


def _headline(cells: dict, gold: dict[str, dict], plan: dict, cue_items: list[dict] | None,
              not_registered: list[str] | None = None) -> dict:
    """Section 12's headline comparison, of the comparisons ``plan`` makes (``_headline_plan``).

    Each is the system against grep at D on the corpus's H- F1-change items in the run's scope:
    accuracy averaged over runs 1, 2 and 3, its difference from grep at D's, and the exact
    two-sided McNemar test on majority results, Holm-adjusted over the registered family
    (``HEADLINE_FAMILY``, a comparison not made counting as p = 1), with a verdict whose direction
    follows those majority results (``_claimed``, ``DIRECTION``). The verdict is a ``claim`` only
    in the registered full run, when ``not_registered`` (the reasons the run is not that run,
    ``_registration_reasons``) is empty; otherwise each row carries a ``reading``. The two
    corpora pooled are reported descriptively and not tested, and ``not_made`` says why each
    comparison not made is not. Beside it, deciding nothing, ``readings`` makes the same
    comparisons on the items without a header-only H- episode after their date (gold
    ``meta.h_minus_header_only_after_d``) and, given the message cues (``cue_items``), on those
    whose gold status no kept message names by their date; a reading makes a comparison only when
    both sides have a scored answer to the corpus's items left in it, each reading is
    Holm-adjusted on its own over the registered family, and its rows carry a ``reading`` instead
    of a ``claim``.
    """
    not_registered = list(not_registered or [])
    arm, family, stratum = HEADLINE_GROUP
    compared, ids, made = plan["compared"], plan["ids"], plan["made"]

    def reading(keep, key: str = "reading") -> dict:
        kept = {corpus: [i for i in ids[corpus] if keep(i)] for corpus in CORPORA}
        comparisons, not_made = [], []
        for system, corpus in made:
            if not kept[corpus]:
                reason = NONE_LEFT
            elif not _has_scored_answer(cells, HEADLINE_BASELINE, corpus, kept[corpus]):
                reason = f"{THE_BASELINE} has {NO_ANSWER_LEFT}"
            elif not _has_scored_answer(cells, system, corpus, kept[corpus]):
                reason = NO_ANSWER_LEFT
            else:
                comparisons.append(_comparison(cells, system, corpus, kept[corpus]))
                continue
            not_made.append({"system": system, "corpus": corpus, "reason": reason})
        return {"items": {corpus: len(found) for corpus, found in kept.items()},
                "comparisons": _claimed(comparisons, key), "not_made": not_made}

    main = reading(lambda item_id: True, "reading" if not_registered else "claim")
    pooled = []
    for system in compared:
        corpora = [c["corpus"] for c in main["comparisons"] if c["system"] == system]
        if not corpora:
            continue
        mine = [f for corpus in corpora for f in _runs_right(cells, system, corpus, ids[corpus])]
        theirs = [f for corpus in corpora
                  for f in _runs_right(cells, HEADLINE_BASELINE, corpus, ids[corpus])]
        runs = len(mine) * len(RUNS)
        right, baseline = sum(map(sum, mine)), sum(map(sum, theirs))
        pooled.append({"system": system, "corpora": corpora, "items": len(mine),
                       "accuracy": right / runs, "baseline_accuracy": baseline / runs,
                       "difference": (right - baseline) / runs})
    named = None if cue_items is None else {item["id"]: item["named"] for item in cue_items}
    return {
        "registered": not not_registered,
        "not_registered": not_registered,
        "baseline": HEADLINE_BASELINE,
        "systems": list(compared),
        "arm": arm, "family": family, "stratum": stratum,
        "measure": "accuracy_mean_over_runs",
        "test": "exact two-sided McNemar on majority results",
        "direction": DIRECTION,
        "adjustment": "holm",
        "alpha": ALPHA,
        "items": main["items"],
        "comparisons": main["comparisons"],
        "pooled": pooled,
        "not_made": plan["not_made"],
        "readings": {
            "without_header_only_episode": reading(
                lambda item_id: not gold[item_id]["meta"]["h_minus_header_only_after_d"]),
            "without_message_cue": None if named is None else reading(
                lambda item_id: not named[item_id]),
        },
    }


def _not_made_groups(not_made: list[dict]) -> list[tuple[str, tuple[str, ...], list[str]]]:
    """The comparisons not made, one entry per cause and set of corpora: ``(reason, corpora,
    systems)``, so that one cause that removes several comparisons is said once."""
    by_system: dict[tuple[str, str], list[str]] = defaultdict(list)
    for entry in not_made:
        by_system[(entry["reason"], entry["system"])].append(entry["corpus"])
    grouped: dict[tuple[str, tuple[str, ...]], list[str]] = defaultdict(list)
    for (reason, system), corpora in by_system.items():
        grouped[(reason, tuple(corpora))].append(system)
    return [(reason, corpora, systems) for (reason, corpora), systems in grouped.items()]


def _model_reasons(models: dict) -> list[str]:
    """Each of the run's answering model, answering settings, construction model and embedding
    model that is not the registered one (section 11, ``REGISTERED_ANSWERING_MODEL``,
    ``REGISTERED_ANSWERING_SETTINGS``, ``REGISTERED_CONSTRUCTION_MODEL`` and
    ``REGISTERED_EMBEDDING_MODEL``), in words."""
    found = []
    if models["answering_model"] != REGISTERED_ANSWERING_MODEL:
        found.append(f"answered by {models['answering_model']}, not the registered answering "
                     "model")
    if models["answering_settings"] != REGISTERED_ANSWERING_SETTINGS:
        found.append(f'answered with "{models["answering_settings"]}", not the registered '
                     "answering settings")
    if models["construction_model"] != REGISTERED_CONSTRUCTION_MODEL:
        found.append(f"built by {models['construction_model']}, not the registered construction "
                     "model")
    if models["embedding_model"] != REGISTERED_EMBEDDING_MODEL:
        found.append(f"embedded with {models['embedding_model']}, not the registered embedding "
                     "model")
    return found


def _build_reasons(data: Path) -> list[str]:
    """Why the build in ``data`` is not the registered one (sections 12 and 13), in words, naming
    each of its questions, gold and pilot list that is absent or whose SHA-256 is not the one
    ``REGISTERED_DATA`` gives; empty for the registered build."""
    absent, differ = [], []
    for name, digest in REGISTERED_DATA.items():
        path = data / name
        if not path.is_file():
            absent.append(name)
        elif sha256(path.read_bytes()).hexdigest() != digest:
            differ.append(name)
    parts = []
    if differ:
        one = len(differ) == 1
        parts.append(f"{_listed_words(differ)} {'differs' if one else 'differ'} from the "
                     f"registered {'file' if one else 'files'} by SHA-256")
    if absent:
        parts.append(f"{_listed_words(absent)} {'is' if len(absent) == 1 else 'are'} absent")
    return [f"the build is not the registered one: {', and '.join(parts)}"] if parts else []


def _narrowed(scope: Scope) -> list[dict]:
    """Each registered system (section 11) that the entries file enters in fewer arms or families
    than its registered entries (``Scope.entry`` without the file), with the arms and the
    families it narrows it to (None for whichever it leaves as registered). The registered full
    run uses the registered entries (section 12): no entries file, or one that narrows no
    registered system. An entries file can only narrow and holds only arms and families, so a
    registered system it declares with its registered entries is entered as registered, and an
    outside entry never counts."""
    found = []
    for system in sorted(scope.declared):
        if system not in REGISTERED_SYSTEMS:
            continue  # an outside entry
        arms, families = scope.declared[system]
        default = scope.default_families(system)
        if arms != frozenset(ARMS) or families != default:
            found.append({"system": system,
                          "arms": None if arms == frozenset(ARMS) else sorted(arms),
                          "families": None if families == default else sorted(families)})
    return found


def _narrowed_words(system: str, arms: list[str] | None, families: list[str] | None) -> str:
    """What an entries file narrows ``system`` (as the caller names it) to, in words."""
    parts = [f"the {one if len(names) == 1 else several} {_listed_words(names)}"
             for names, one, several in ((arms, "arm", "arms"), (families, "family", "families"))
             if names]
    return f"the entries file narrows {system} to {' in '.join(parts)}"


def _registration_reasons(scope: dict, narrowed: list[dict], not_made: list[dict],
                          incomplete: list[dict], models: dict, build: list[str],
                          name=lambda system: system) -> list[str]:
    """Why the run is not the registered full run (section 12), each condition it does not meet
    in words; empty for the registered full run, which scores every item (no ``--items``) with the
    registered entries, makes all ``HEADLINE_FAMILY`` comparisons with grep at D and each compared
    system answering every compared item in runs 1, 2 and 3, and was answered by
    ``REGISTERED_ANSWERING_MODEL`` with ``REGISTERED_ANSWERING_SETTINGS`` over stores
    ``REGISTERED_CONSTRUCTION_MODEL`` built, with text ``REGISTERED_EMBEDDING_MODEL`` embedded, on
    the registered build. ``scope`` is the report's (its ``items`` and ``items_file``),
    ``narrowed`` the registered systems an entries file narrows (``_narrowed``), one reason each,
    ``not_made`` the comparisons not made, one reason per cause, and ``incomplete`` the sides of
    the comparisons made that leave a run out, one reason per side and corpus (both from
    ``_headline_plan``), ``models`` holds the models and settings, and ``build`` is
    ``_build_reasons``'s list. ``name`` writes a system's name, so that results.md can label it as
    its tables do while results.json keeps the id."""
    reasons = []
    if scope["items"] is not None:
        reasons.append(f"the run's scope is {_scope_words(scope)}, given with --items")
    reasons += [_narrowed_words(name(entry["system"]), entry["arms"], entry["families"])
                for entry in narrowed]
    for reason, corpora, systems in _not_made_groups(not_made):
        names = [name(system) for system in systems]
        reasons.append(f"{_listed_words(names)} {'is' if len(names) == 1 else 'are'} not compared "
                       f"on {_listed_words(list(corpora), 'or')} ({reason})")
    reasons += [_answered_words(name(side["system"]), side["runs_answered"], side["items"],
                                side["corpus"]) for side in incomplete]
    return reasons + _model_reasons(models) + list(build)


def _item_runs(cells: dict) -> dict[str, dict[str, dict[str, dict]]]:
    """For every system, arm and item the system is scored on there, answered or not: the runs
    right, the runs answered and the majority result (section 12), so that any comparison of two
    systems can be recomputed from results.json."""
    found: dict[str, dict[str, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for (system, arm, _, _, _), by_item in cells.items():
        for item_id, by_run in by_item.items():
            flags = _flags(by_run)
            found[system][arm][item_id] = {"runs_right": sum(flags),
                                           "runs_answered": sum(run in by_run for run in RUNS),
                                           "majority_correct": _majority(flags)}
    return {system: dict(arms) for system, arms in sorted(found.items())}


def _pilot_groups(path: Path, questions: dict[str, dict]) -> list[dict]:
    """``pilot.json``'s groups, each with a family, a stratum and a list of question ids."""
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ScoringError(f"{path} is not JSON (a truncated or damaged file?)") from None
    groups = content.get("groups") if isinstance(content, dict) else None
    if not isinstance(groups, list) or not all(
        isinstance(g, dict) and isinstance(g.get("family"), str)
        and isinstance(g.get("stratum"), str) and isinstance(g.get("ids"), list)
        and all(isinstance(i, str) for i in g["ids"]) for g in groups
    ):
        raise ScoringError(f"{path} is not a pilot file: each of its groups needs a family, a "
                           "stratum and a list of item ids")
    unknown = sorted({i for g in groups for i in g["ids"]} - set(questions))
    if unknown:
        raise ScoringError(f"{path} names {len(unknown)} id(s) that are not in the questions, "
                           f"such as {unknown[0]}")
    return groups


def _majority_reading(pooled: list[dict], per_item: dict, graphiti: str = GRAPHITI) -> dict:
    """What the gates would decide on each item's majority result instead of the accuracy
    averaged over the runs, which section 13 reads. It is reported beside the decision and
    decides nothing. The two readings can differ in whether they can be decided at all, and
    then this one says so."""
    other = pilot_gates(pooled, per_item, measure="majority_accuracy", graphiti=graphiti)
    if not other["evaluable"]:
        return {"evaluable": False, "decision": other["decision"]}
    return {"evaluable": True, "stopped_at": other["stopped_at"], "signal": other["signal"]}


def _entered_in(scope: Scope, system: str, arm: str, family: str) -> bool:
    """Whether ``system`` is entered in ``arm`` and ``family`` (section 12's entries)."""
    arms, families = scope.entry(system)
    return arm in arms and family in families


def _pilot_reasons(listed: frozenset[str], scope: Scope, questions: dict[str, dict],
                   systems: tuple[str, ...], models: dict, build: list[str]) -> list[str]:
    """Why the run is not the registered pilot run (section 13), each condition it does not meet
    in words; empty for the registered pilot run, whose scope, given with ``--items``, is exactly
    the items of pilot.json (``listed``, the ids ``--items`` reads from it, in any order), whose
    entries leave no pilot system out of H- F1, which the gates read, whose answers
    ``REGISTERED_ANSWERING_MODEL`` gave with ``REGISTERED_ANSWERING_SETTINGS`` over stores
    ``REGISTERED_CONSTRUCTION_MODEL`` built, with text ``REGISTERED_EMBEDDING_MODEL`` embedded,
    and whose build is the registered one (``build`` is ``_build_reasons``'s list). A run with no
    ``--items`` scores every item of ``questions`` and is never the registered pilot run, so a
    full run never decides the gates."""
    reasons = []
    holds = "that pilot.json does not list"
    if scope.items is None:
        beside = len(set(questions) - listed)
        reasons.append("the run's scope is every item (no --items)"
                       + (f", which holds {_counted(beside, 'item')} {holds}" if beside else ""))
    else:
        left = len(listed - scope.items)
        if left:
            reasons.append(f"the run's scope leaves out {_counted(left, 'pilot item')}")
        beside = len(scope.items - listed)
        if beside:
            reasons.append(f"the run's scope holds {_counted(beside, 'item')} {holds}")
    out = [s for s in systems if not _entered_in(scope, s, "H-", "F1")]
    if out:
        reasons.append(f"the entries file leaves {_listed_words(out)} out of H- F1, which the "
                       "gates read")
    return reasons + _model_reasons(models) + list(build)


REGISTERED_PILOT = ("the gates decide only in the registered pilot run, whose scope is exactly the "
                    "items of pilot.json, given with --items, for every pilot system, on the "
                    f"registered build, and whose answers {REGISTERED_ANSWERING_MODEL} gave with "
                    f"{REGISTERED_ANSWERING_SETTINGS}, over stores {REGISTERED_CONSTRUCTION_MODEL} "
                    f"built, with text embedded by {REGISTERED_EMBEDDING_MODEL}")


def _pilot(data: Path, scored: list[dict], questions: dict[str, dict], scope: Scope,
           models: dict, build: list[str], graphiti: str = GRAPHITI) -> tuple[dict, dict | None]:
    """The pilot's coverage by the answers and its gates (section 13).

    The pilot list is the F1 items of ``pilot.json``, read in arm H-. The pilot systems are the
    six of section 13, with ``graphiti`` as the pilot's Graphiti (graphiti, or graphiti-reduced
    under M0's fallback). Each system is scored on the pilot items in its scope (section 12), so
    an answer the scope sets aside is never counted as missing. The gates decide only in the
    registered pilot run (``_pilot_reasons``), and only when all six have answers there; an item
    a present system did not answer is wrong (section 12), and the coverage, with the systems and
    items that are missing, is in the result. Otherwise the gates say the pilot is not evaluable
    and why, with every gate's computation and no decision (``_not_evaluable``). Without a
    ``pilot.json`` (or with no F1 items in it) there are no gates.
    """
    path = data / "pilot.json"
    if not path.is_file():
        return {"evaluable": False, "reasons": ["no pilot.json"]}, None
    ids: dict[str, list[str]] = {"change": [], "current": []}
    for group in _pilot_groups(path, questions):
        if group["family"] == "F1" and group["stratum"] in ids:
            ids[group["stratum"]].extend(group["ids"])
    wanted = ids["change"] + ids["current"]
    if not wanted:
        return {"evaluable": False, "reasons": ["no F1 items in pilot.json"]}, None
    systems = pilot_systems(graphiti)
    stratum_of = {item_id: stratum for stratum, found in ids.items() for item_id in found}
    asked = {s: [i for i in wanted if scope.contains(s, "H-", questions[i])] for s in systems}
    answered: dict[str, set] = {s: set() for s in systems}
    for r in scored:  # the scored rows are those of the scope
        if r["arm"] == "H-" and r["id"] in stratum_of and r["system"] in answered:
            answered[r["system"]].add(r["id"])
    # A system that has pilot items in the scope and no answer to them; one with none in the
    # scope is named by the reasons instead.
    missing_systems = [s for s in systems if asked[s] and not answered[s]]
    present = [s for s in systems if answered[s]]
    coverage = {
        "pilot_items": sum(1 for i in wanted if scope.items is None or i in scope.items),
        "complete": all(asked[s] and answered[s] >= set(asked[s]) for s in systems),
        "systems": {
            s: {"answered": len(answered[s]), "asked": len(asked[s]),
                "missing_count": len(asked[s]) - len(answered[s]),
                "missing": [i for i in asked[s] if i not in answered[s]][:MAX_LISTED]}
            for s in present
        },
    }
    rows = [dict(r, corpus="all") for r in scored
            if r["arm"] == "H-" and r["id"] in stratum_of and r["system"] in present]
    items = {s: [(i, "F1", stratum_of[i], "all") for i in asked[s]] for s in present}
    # Every present system is scored on every pilot item in its scope; one it did not answer is
    # wrong.
    pooled = aggregate(rows, {s: {"H-": items[s]} for s in present})
    per_item = _majorities(
        [r for r in rows if r["stratum"] == "change"],
        {s: {"H-": [item for item in items[s] if item[2] == "change"]} for s in present},
    )
    gates = pilot_gates(pooled, per_item, graphiti=graphiti)
    # The ids that --items reads from pilot.json: the registered pilot run's scope is exactly them.
    listed, _ = _items(path, questions)
    reasons = _pilot_reasons(listed, scope, questions, systems, models, build)
    if reasons or not gates["evaluable"]:
        # Not the registered pilot run, or not decidable on its answers: the missing systems are
        # those of the scope, and a gate's missing inputs count only when every system answered.
        detail = gates["detail"]
        gates = _not_evaluable(
            detail["measure"], detail["accuracy_change"], detail["accuracy_current"],
            missing_systems, [] if missing_systems else gates.get("missing_inputs", []),
            {name: gate for name, gate in detail.items() if name.startswith("gate")}, reasons)
    else:
        # The gates read the headline measure (section 13); this records the other reading.
        gates["detail"]["by_majority_accuracy"] = _majority_reading(pooled, per_item, graphiti)
        gates["reasons"] = []
    summary = {"evaluable": gates["evaluable"], "reasons": gates["reasons"],
               "missing_systems": missing_systems, "systems": list(systems),
               "change_items": len(ids["change"]), "current_items": len(ids["current"]),
               "coverage": coverage}
    return summary, {**gates, "pilot_systems": list(systems), "coverage": coverage}


# --- the report ----------------------------------------------------------------------------------


def _f(value: object) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


# Section 11: every table of the results report labels the Personalized PageRank retriever as the
# author's own, and a system that an entries file declares and section 11 does not name as an
# outside entry. results.md shows the labels wherever it names a system: in its tables, in the
# headline's lines and in the closing lists, except in the headline's clause that lists the run's
# outside entries, which says that they are and so names them by id; results.json and gates.json
# keep the system's id, as the answer rows do.
TABLE_LABELS = {"ppr": "ppr (author's own)"}
OUTSIDE_ENTRY = "outside entry"


def _name(report: dict, system: str) -> str:
    """How a table of results.md names ``system``: with its label, if it has one."""
    if system in TABLE_LABELS:
        return TABLE_LABELS[system]
    if system in report["scope"]["outside_entries"]:
        return f"{system} ({OUTSIDE_ENTRY})"
    return system


def _cell(value: object) -> str:
    """A table cell: the value as ``_f`` writes it. A system's cell is written with ``_name``."""
    return value if isinstance(value, str) else _f(value)


def _table(headers: list[str], body: list[list[object]]) -> list[str]:
    lines = ["| " + " | ".join(_cell(h) for h in headers) + " |", "|" + "---|" * len(headers)]
    return lines + ["| " + " | ".join(_cell(c) for c in row) + " |" for row in body]


def _models(report: dict) -> str:
    return (f"answering model {report['answering_model']} with {report['answering_settings']}; "
            f"construction model {report['construction_model']}; embedding model "
            f"{report['embedding_model']}")


def _heading(title: str, report: dict) -> str:
    """A section heading that names the models and the answering settings (section 11: every
    table names them)."""
    return f"## {title} ({_models(report)})"


def _primary_section(report: dict) -> list[str]:
    head = ["System", "Arm", "Family", "Stratum", "Corpus", "Items", "Runs", "Accuracy",
            "Majority-class rate", "Majority accuracy", "Wilson 95%", "pass@3", "all-3",
            "Tokens in", "Tokens out", "Tool use"]
    body = [[_name(report, r["system"]), r["arm"], r["family"], r["stratum"], r["corpus"],
             r["n_items"],
             r["runs"], r["accuracy_mean_over_runs"], r["majority_class_rate"],
             r["majority_accuracy"], f"[{r['wilson_low']:.3f}, {r['wilson_high']:.3f}]",
             r["pass_at_3"], r["all_3"], r["tokens_in"], r["tokens_out"], r["tool_use_rate"]]
            for r in report["results"]]
    intro = ("Every item of the arm in the scope counts, answered or not, and an item with no "
             "answer in a run is wrong in that run. Accuracy is exact match averaged over the "
             "runs. The majority-class rate is the share of the group's items whose gold is "
             "their most common gold value: what always giving that one answer scores. The "
             "Wilson 95% interval is over the items, each taken at its majority result (correct "
             "in more than half of the runs). pass@3 is the share of items correct in any run "
             "and all-3 the share correct in every run.")
    return ["", _heading("Primary results", report), "", intro, "", *_table(head, body)]


def _listed_words(names: list[str], joined: str = "and") -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} {joined} {names[-1]}"


def _counted(count: int, noun: str) -> str:
    """``1 item``, ``2 items``."""
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _scope_words(scope: dict) -> str:
    """The items the run scores, in words: every item, or the items of a ``--items`` list."""
    if scope["items"] is None:
        return "every item"
    return f"the {_counted(scope['items'], 'item')} of {scope['items_file']}"


def _p(value: float) -> str:
    return f"{value:.4g}"


def _headline_rows(report: dict, comparisons: list[dict], key: str) -> list[str]:
    """The comparisons' table; its last column is the headline's claim, or a reading's."""
    head = ["System", "Corpus", "Items", "Accuracy", "Grep at D's accuracy", "Difference",
            "System right, grep at D wrong", "Grep at D right, system wrong", "p",
            "Holm-adjusted p", key.capitalize()]
    body = [[_name(report, c["system"]), c["corpus"], c["items"], c["accuracy"],
             c["baseline_accuracy"],
             c["difference"], c["only_system_right"], c["only_baseline_right"], _p(c["p"]),
             _p(c["p_holm"]), c[key]] for c in comparisons]
    return _table(head, body)


def _direction_lines(report: dict, comparisons: list[dict], key: str) -> list[str]:
    """For each comparison whose claim (or reading) is above or below grep at D while its
    accuracy averaged over the runs differs the other way, or not at all, a line that says so:
    the claim follows the majority results the test compares (``_claimed``). Systems are named
    as the tables name them (``_name``)."""
    out = []
    for c in comparisons:
        if c[key] == NO_DIFFERENCE:
            continue
        sign = 1 if c[key] == ABOVE else -1
        if sign * c["difference"] > EPS:  # the two agree
            continue
        end = ("does not differ (0.000)" if abs(c["difference"]) <= EPS
               else f"differs the other way, by {c['difference']:.3f}")
        name = _name(report, c["system"])
        out.append(f'- {name} on {c["corpus"]}: the {key} "{c[key]}" follows the majority '
                   f"results the test compares ({_counted(c['only_system_right'], 'item')} right "
                   f"only for {name}, {c['only_baseline_right']} right only for grep at D); the "
                   f"accuracy averaged over the runs {end}.")
    return out


def _answered_words(system: str, answered: int, items: int, corpus: str) -> str:
    """How many of the three runs of each of ``items`` compared items of ``corpus`` ``system``
    (as the caller names it) answered, in words: a reason why a run is not the registered full
    run (``_registration_reasons``) and a line under the headline's tables
    (``_shortfall_lines``)."""
    return (f"{system} answered {answered} of the {_counted(len(RUNS) * items, 'run')} on the "
            f"{_counted(items, f'compared {corpus} item')}")


def _shortfall_lines(report: dict, comparisons: list[dict]) -> list[str]:
    """For grep at D on each corpus, and for each compared system, that left runs out of the
    compared items, a line giving the runs it answered out of three per item: a missing run is
    wrong, so the comparison reads it as a wrong answer. Systems are named as the tables name
    them (``_name``)."""
    def line(system: str, answered: int, items: int, corpus: str) -> str:
        return (f"- {_answered_words(_name(report, system), answered, items, corpus)}; a missing "
                "run counts as wrong.")

    out = []
    for corpus in CORPORA:
        mine = [c for c in comparisons if c["corpus"] == corpus]
        if mine and mine[0]["baseline_runs_answered"] < len(RUNS) * mine[0]["items"]:
            out.append(line(HEADLINE_BASELINE, mine[0]["baseline_runs_answered"],
                            mine[0]["items"], corpus))
        out += [line(c["system"], c["runs_answered"], c["items"], corpus) for c in mine
                if c["runs_answered"] < len(RUNS) * c["items"]]
    return out


def _not_made_lines(report: dict, not_made: list[dict]) -> list[str]:
    """The comparisons not made, one line per cause and set of corpora (``_not_made_groups``),
    naming the systems as the tables name them (``_name``)."""
    return [f"- Not compared on {_listed_words(list(corpora), 'or')} ({reason}): "
            f"{_listed_words([_name(report, system) for system in systems])}."
            for reason, corpora, systems in _not_made_groups(not_made)]


REGISTERED_RUN = ("scores every item (no --items) of the registered build (data/v0.1's "
                  "questions.jsonl, gold.jsonl and pilot.json) with the registered entries (no "
                  "entries file, or one that narrows no registered system), makes all "
                  f"{HEADLINE_FAMILY} registered comparisons, grep at D and each compared system "
                  "having a scored answer to every H- F1-change item of each corpus in runs "
                  f"{_listed_words(list(RUNS))}, and is answered by "
                  f"{REGISTERED_ANSWERING_MODEL} with {REGISTERED_ANSWERING_SETTINGS}, over stores "
                  f"{REGISTERED_CONSTRUCTION_MODEL} built, with text embedded by "
                  f"{REGISTERED_EMBEDDING_MODEL}")


def _headline_lead(report: dict) -> tuple[str, str]:
    """The headline section's first sentences and the word of its rows: in the registered full
    run, that this is the registered headline and its rows' ``claim``; in any other run, that it
    is not the registered full run, each reason why (systems named as the tables name them), and
    its rows' ``reading``."""
    found = report["headline"]
    if found["registered"]:
        return ("The registered headline (section 12). This is the registered full run, which "
                f"{REGISTERED_RUN}.", "claim")
    reasons = _registration_reasons(report["scope"], report["narrowed"], found["not_made"],
                                    report["incomplete"], report, report["build_reasons"],
                                    lambda system: _name(report, system))
    return ("This run is not the registered full run, so its comparisons give readings, not "
            f"claims (section 12): {'; '.join(reasons)}. The registered full run "
            f"{REGISTERED_RUN}.", "reading")


def _headline_section(report: dict) -> list[str]:
    """Section 12's headline comparison, right after the primary results."""
    found, scope = report["headline"], report["scope"]
    items = found["items"]
    lead, key = _headline_lead(report)
    made = len(found["comparisons"])
    makes = "no comparison" if not made else _counted(made, "comparison")
    outside = scope["outside_entries"]  # by id: the clause itself says they are outside entries
    held = "" if not outside else (
        f"; this run holds the outside {'entry' if len(outside) == 1 else 'entries'} "
        f"{_listed_words(outside)}")
    intro = (f"{lead} Each system that keeps or retrieves context from the stream or its "
             f"proposals is compared with grep at D on the H- F1-change items of each corpus in "
             f"the run's scope ({_scope_words(scope)}): {_counted(items['pep'], 'PEP item')} and "
             f"{_counted(items['eth'], 'Ethereum item')}. Accuracy is averaged over runs 1, 2 and "
             f"3, and the difference is the system's accuracy minus grep at D's on the same items. "
             f"The test is the exact two-sided McNemar test on majority results (an item is right "
             f"when at least two of its three runs are right; a missing run is wrong), "
             f"Holm-adjusted over the {_counted(HEADLINE_FAMILY, 'registered comparison')} (each "
             f"compared system on each corpus), in which a comparison the run does not make "
             f"counts as p = 1; this run makes {makes}. A system "
             f"is above grep at D on a corpus when its Holm-adjusted p is below 0.05 and it is "
             f"right where grep at D is wrong on more items than the reverse, below grep at D "
             f"when its Holm-adjusted p is below 0.05 and the reverse holds, and otherwise there "
             f"is no detectable difference: the {key} follows the majority results the test "
             f"compares, and a line below the table says when the accuracy averaged over the runs "
             f"differs the other way, or not at all. Closed book, the oracle and outside entries "
             f"are not compared{held}.")
    out = ["", _heading("Headline comparison", report), "", intro, ""]
    if not any(items.values()):
        out.append("- No comparison is made: no H- F1-change item is in the run's scope.")
    else:
        if found["comparisons"]:
            out += [*_headline_rows(report, found["comparisons"], key), "",
                    *_direction_lines(report, found["comparisons"], key),
                    *_shortfall_lines(report, found["comparisons"])]
        out += _not_made_lines(report, found["not_made"])
    if found["pooled"]:
        body = [[_name(report, p["system"]), p["items"], p["accuracy"], p["baseline_accuracy"],
                 p["difference"]] for p in found["pooled"]]
        out += ["", "The corpora compared, pooled by item count: the accuracy difference, reported "
                    "descriptively and not tested.", "",
                *_table(["System", "Items", "Accuracy", "Grep at D's accuracy", "Difference"],
                        body)]
    family = _counted(HEADLINE_FAMILY, "registered comparison")
    out += ["", "Beside the headline, the same comparisons on two subsets of its items, each "
                f"Holm-adjusted on its own over the {family}, in which a comparison the reading "
                "does not make counts as p = 1. These readings decide nothing: each row gives a "
                "reading, not a claim, by the headline's rule."]
    for key, title in (
        ("without_header_only_episode", "The items without a header-only H- episode after their "
         "date (a commit that changed only withheld header lines, whose H- diff is empty)"),
        ("without_message_cue", "The items whose gold status no commit message kept in the H- "
         "stream names by their date"),
    ):
        reading = found["readings"][key]
        if reading is None:
            out += ["", f"- {title}: not read, since no --message-cues file was given."]
            continue
        out += ["", f"{title}: {_counted(reading['items']['pep'], 'PEP item')} and "
                    f"{_counted(reading['items']['eth'], 'Ethereum item')}."]
        if reading["comparisons"]:
            out += ["", *_headline_rows(report, reading["comparisons"], "reading")]
        notes = [*_direction_lines(report, reading["comparisons"], "reading"),
                 *_shortfall_lines(report, reading["comparisons"]),
                 *_not_made_lines(report, reading["not_made"])]
        if notes:
            out += ["", *notes]
    return out


def _secondary_section(report: dict) -> list[str]:
    head = ["System", "Arm", "Family", "Stratum", "Corpus", "Outcome class", "Precision", "Recall",
            "F-score", "Mean abs error", "Within one", "Invalid", "Mean wall s"]
    body = [[_name(report, r["system"]), r["arm"], r["family"], r["stratum"], r["corpus"],
             r["outcome_accuracy"], r["precision"], r["recall"], r["f1"], r["mean_abs_error"],
             r["within_one_rate"], r["invalid_rate"], r["mean_wall_s"]]
            for r in report["secondary"]]
    intro = ("Outcome-class accuracy is for statuses (adopted, declined, superseded, open); "
             "precision, recall and F-score (their harmonic mean) are for sets; error and the "
             "within-one rate are for counts. The invalid share counts missing answers.")
    return ["", _heading("Secondary scores", report), "", intro, "", *_table(head, body)]


def _coverage_section(report: dict) -> list[str]:
    body = [[_name(report, r["system"]), r["arm"], r["family"], r["stratum"], r["corpus"],
             r["answered"], r["asked"]] for r in report["coverage"]]
    intro = ("Answered is the number of the items in the denominator that the system answered in "
             "at least one run; asked is the number of items in the denominator. The items that "
             "were not answered are in the scores as wrong.")
    return ["", _heading("Coverage", report), "", intro, "",
            *_table(["System", "Arm", "Family", "Stratum", "Corpus", "Answered", "Asked"], body)]


def _entries_section(report: dict) -> list[str]:
    body = [[_name(report, system), ", ".join(e["arms"]), ", ".join(e["families"])]
            for system, e in report["scope"]["entries"].items()]
    intro = ("Each system is scored on the items of every arm it is entered in, in the families "
             "it is entered in (section 12).")
    return ["", _heading("Entries", report), "", intro, "",
            *_table(["System", "Arms", "Families"], body)]


def _accounting_section(report: dict) -> list[str]:
    body = [[_name(report, system), e["rows"], e["without"],
             *(e["by_field"][name] for name in ACCOUNTING)]
            for system, e in report["accounting"].items()]
    intro = ("Every answer row records tokens_in, tokens_out, wall_s and tool_calls (section 12). "
             "A row that lacks one of them, or has a value there that is not a number, is still "
             "scored and is counted here, per system and per field. The token totals, the "
             "tool-use rate and the mean wall time are over the rows that carry each value, so "
             "a missing value is left out of them rather than counted as zero.")
    return ["", _heading("Accounting", report), "", intro, "",
            *_table(["System", "Rows scored", "Rows without accounting",
                     *(f"No {name}" for name in ACCOUNTING)], body)]


def _cf_items_section(report: dict) -> list[str]:
    items, given = report["cf_items"], report["cf_messages"]
    systems = sorted({s for item in items for s in item["systems"]})
    head = ["Item", "Decision word", "Header cue",
            "Commits whose message mentions it with a status value",
            *(_name(report, s) for s in systems)]
    body = [[i["id"], i["decision_word"], i["header_cue"],
             "-" if i["cf_messages"] is None else i["cf_messages"],
             *[(i["systems"][s]["accuracy_mean_over_runs"] if s in i["systems"] else None)
               for s in systems]] for i in items]
    where = (f"The counts are from {given['file']}." if given is not None else
             "No file of counts was given (--cf-messages), so the column is empty.")
    intro = ("Messages of commits that change no CF status line are not edited. For each CF item "
             "the table gives the number of commits outside the rewritten ones, each counted "
             "once, whose message mentions the item's proposal and contains one of its corpus's "
             "status values as a whole word, in any letter case; the other columns are each "
             "system's accuracy on the item, averaged over its runs. " + where)
    return ["", _heading("Counterfactual items", report), "", intro, "", *_table(head, body)]


def _cues_section(report: dict) -> list[str]:
    summary = report["message_cues"]
    body = [[*key.split("/"), group["items"], group["named"]]
            for key, group in summary["by_stratum"].items()]
    intro = (f"For each H- F1 item, whether a commit message kept in the H- stream at or before "
             f"the end of the item's date names its proposal together with its gold status "
             f"(section 12), read from {summary['file']}: {summary['named']} of the "
             f"{summary['items']} H- F1 items. H- results can be read against what the messages "
             f"alone reveal; results.json lists each item.")
    split = [[_name(report, r["system"]), r["named_items"], r["named_accuracy"], r["other_items"],
              r["other_accuracy"]] for r in report["message_cue_split"]]
    lead = ("Each system's H- F1 accuracy, averaged over runs 1, 2 and 3, on the items a message "
            "names and on the other items (an unanswered item is wrong):")
    return ["", _heading("H- items whose gold status a message names", report), "", intro, "",
            *_table(["Stratum", "Corpus", "Items", "Named by a message"], body), "", lead, "",
            *_table(["System", "Named items", "Accuracy", "Other items", "Accuracy"], split)]


def _header_only_section(report: dict) -> list[str]:
    if not report["header_only_split"]:
        return []
    intro = ("Each system's H- F1 accuracy, averaged over runs 1, 2 and 3, on the items whose "
             "proposal has a header-only H- episode recorded after the end of their date and on "
             "the rest (section 12). Such an episode is a commit that changed only withheld "
             "header lines: the H- stream keeps it, with an empty diff and its message, so a "
             "system that reads the stream can see that a withheld header changed after the "
             "date, though not to what, and grep at D cannot. REPORT.md gives the counts. An "
             "unanswered item is wrong.")
    body = [[_name(report, r["system"]), r["stratum"], r["corpus"], r["flagged_items"],
             r["flagged_accuracy"], r["other_items"], r["other_accuracy"]]
            for r in report["header_only_split"]]
    return ["", _heading("H- F1 by a header-only episode after the date", report), "", intro, "",
            *_table(["System", "Stratum", "Corpus", "Header-only episode after the date: items",
                     "Accuracy", "None: items", "Accuracy"], body)]


def _contamination_section(report: dict) -> list[str]:
    intro = ("A corpus's F1-change floor is the share of its F1-change items in the run's scope, "
             "those closed book is scored on, whose gold is their most common gold value. Closed "
             "book's result on an item is the majority of "
             "every answer it gave the item, over all runs in every arm it answered; a tie, or no "
             "answer, is wrong. When an exact one-sided binomial test of closed book's results "
             "against the floor gives p below 0.05, the corpus's F1-change is marked possibly "
             "contaminated (section 10). F1-change is the primary as-of result whatever the test "
             "says.")
    out = ["", _heading("Closed-book check", report), "", intro, ""]
    flags = report["contamination"]
    if not flags:
        return out + ["No closed-book answers in this run reach an F1-change item: closed book is "
                      "not scored on any, so no corpus's F1-change is tested against its floor."]
    for corpus, f in flags.items():
        if f["p"] is None:
            out.append(f"- {corpus}: not tested: closed book gave no answer to its F1-change "
                       "items.")
            continue
        verdict = "possibly contaminated" if f["possibly_contaminated"] else "not marked"
        out.append(f"- {corpus}: floor {f['floor']:.3f} ({f['floor_gold']}, {f['floor_items']} of "
                   f"{f['items']} items); closed book right on {f['closed_book_right']} of "
                   f"{f['items']} items (arms it answered: {', '.join(f['arms'])}); one-sided "
                   f"exact binomial p = {f['p']:.4g}; {verdict}.")
    if not report["closed_book_split"]:
        return out
    body = [[_name(report, r["system"]), r["arm"], r["corpus"], r["closed_book_wrong_items"],
             r["closed_book_wrong_accuracy"], r["closed_book_right_items"],
             r["closed_book_right_accuracy"]] for r in report["closed_book_split"]]
    lead = ("Each system's F1-change accuracy, averaged over runs 1, 2 and 3, on the items closed "
            "book gets wrong and on the rest (an unanswered item is wrong):")
    return out + ["", lead, "", *_table(["System", "Arm", "Corpus", "Closed book wrong: items",
                                         "Accuracy", "Closed book right: items", "Accuracy"], body)]


def _recent_section(report: dict) -> list[str]:
    if not report["recent_split"]:
        return []
    intro = ("Each system's F1-recent accuracy, averaged over runs 1, 2 and 3, on the items whose "
             "gold is the proposal's status at S and on the rest (section 10): answering with the "
             "status at S is right on the first group only. An unanswered item is wrong.")
    body = [[_name(report, r["system"]), r["arm"], r["corpus"], r["equals_s_items"],
             r["equals_s_accuracy"],
             r["differs_items"], r["differs_accuracy"]] for r in report["recent_split"]]
    return ["", _heading("F1-recent by whether the gold is the status at S", report), "", intro,
            "", *_table(["System", "Arm", "Corpus", "Gold is the status at S: items", "Accuracy",
                         "Gold differs: items", "Accuracy"], body)]


def _cutoff_section(report: dict) -> list[str]:
    """Section 10: F1-recent apart by the answering model's training cutoff, when one is given
    (a line at the top of results.md says when it is not)."""
    day = report["training_cutoff"]
    if day is None or not report["recent_cutoff_split"]:
        return []
    intro = (f"Each system's F1-recent accuracy, averaged over runs 1, 2 and 3, on the items whose "
             f"transition (gold meta.interval_start) was recorded before {day}, the first day "
             f"after the answering model's training data (--training-cutoff), and on those "
             f"recorded on or after it (section 10). Only the second group's transitions are "
             f"certainly beyond the training data: when the provider states the cutoff as a "
             f"month, an item recorded in that month may fall on either side of it. An "
             f"unanswered item is wrong.")
    body = [[_name(report, r["system"]), r["arm"], r["corpus"], r["before_items"],
             r["before_accuracy"], r["on_or_after_items"], r["on_or_after_accuracy"]]
            for r in report["recent_cutoff_split"]]
    return ["", _heading("F1-recent by the answering model's training cutoff", report), "", intro,
            "", *_table(["System", "Arm", "Corpus", f"Transition before {day}: items",
                         "Accuracy", f"On or after {day}: items", "Accuracy"], body)]


def _cf_split_section(report: dict, key: str, flag: str, title: str, column: str,
                      intro: str) -> list[str]:
    """The CF items of ``report[key]`` (``_counterfactual_split``) scored apart by ``flag``."""
    rows = report[key]
    if not rows:
        return []
    body = [[_name(report, r["system"]), r["corpus"], r[flag], r["n_items"], r["runs"],
             r["accuracy_mean_over_runs"], r["majority_accuracy"]] for r in rows]
    return ["", _heading(title, report), "", intro, "",
            *_table(["System", "Corpus", column, "Items", "Runs", "Accuracy",
                     "Majority accuracy"], body)]


def _f2_section(report: dict) -> list[str]:
    if not report["f2_split"]:
        return []
    intro = ("Each system's F2 accuracy, averaged over runs 1, 2 and 3, over all F2 items of the "
             "arm and over the items whose gold the asked PEP's own episodes can show in that arm "
             "(section 12). In H+, every gold member is listed in the PEP's Superseded-By or "
             "Replaced-By header, or mentioned in its body, in some version of its file; in H-, "
             "which withholds those headers, every member is mentioned. A negative is "
             "recoverable in both arms. Beside each accuracy, always empty is what always "
             "answering the empty list scores on the same items: it is right on every negative "
             "and wrong on every positive. An unanswered item is wrong.")
    body = [[_name(report, r["system"]), r["arm"], r["items"], r["accuracy"], r["always_empty"],
             r["recoverable_items"], r["recoverable_accuracy"], r["recoverable_always_empty"]]
            for r in report["f2_split"]]
    title = "F2 over all items and over the items the asked PEP's own episodes can show"
    return ["", _heading(title, report), "", intro, "",
            *_table(["System", "Arm", "Items", "Accuracy", "Always empty", "Recoverable items",
                     "Accuracy", "Always empty"], body)]


def _optional_sections(report: dict) -> list[str]:
    out = _cf_split_section(
        report, "cf_split", "decision_word", "Counterfactual items by decision word",
        "Decision word",
        "Items whose body at the date holds a decision word (accept, reject, withdraw, final or "
        "supersede) are scored apart from the rest, and together in the primary results.")
    out += _cf_split_section(
        report, "cf_header_split", "header_cue", "Counterfactual items by header cue",
        "Header cue",
        "Items whose file at the date has a header field other than the status, which the edit "
        f"leaves alone, that records the true status ({HEADER_CUE_RULE}) are scored apart from "
        "the rest, and together in the primary results.")
    if report["cf_items"]:
        out += _cf_items_section(report)
    if report["message_cues"] is not None:
        out += _cues_section(report)
    out += _header_only_section(report)
    return (out + _f2_section(report) + _contamination_section(report) + _recent_section(report)
            + _cutoff_section(report))


def _outcome(stopped_at: int | None, signal: bool) -> str:
    """Where the pilot stops and whether gate 4 finds a signal, in words (a clause)."""
    if stopped_at is not None:
        return f"the pilot stops at gate {stopped_at}, before gate 4, so there is no signal"
    return f"no gate stops the pilot, and gate 4 finds {'a' if signal else 'no'} signal"


def _shown(ids: list[str], missing_count: int, limit: int = 8) -> str:
    more = f" and {missing_count - limit} more" if missing_count > limit else ""
    return ", ".join(ids[:limit]) + more


def _gate4_lines(report: dict, gate4: dict, stopped_at: int | None) -> list[str]:
    """Gate 4's two readings and the fifteen pairs; for reference only when an earlier gate
    stopped the pilot, so that its counts cannot be read as a signal."""
    counted = {(p["a"], p["b"]): p for p in gate4["pairs"]}
    unadjusted = gate4["unadjusted"]
    reading = (f"gate 4 reads the ten pairs of pilot systems without closed book, with "
               f"Holm-adjusted exact McNemar p-values: "
               f"{sum(1 for p in gate4['pairs'] if p['qualifies'])} of them differ by 15 points "
               f"or more with an adjusted p below 0.05. Over all fifteen pairs without the "
               f"adjustment, {unadjusted['qualifying']} pair(s) differ by 15 points or more with "
               f"p below 0.05; that reading is reported beside it and decides nothing.")
    lead = ("G" + reading[1:] if stopped_at is None else
            f"Gate 4 was not reached (the pilot stopped at gate {stopped_at}); for reference, "
            f"{reading}")
    body = []
    for p in unadjusted["pairs"]:
        mine = counted.get((p["a"], p["b"]))
        gate = "not counted" if mine is None else "qualifies" if mine["qualifies"] else "no"
        body.append([_name(report, p["a"]), _name(report, p["b"]), p["gap"], f"{p['p']:.4g}",
                     "-" if mine is None else f"{mine['p_holm']:.4g}", gate,
                     "qualifies" if p["qualifies"] else "no"])
    return ["", lead, "", *_table(["System A", "System B", "Gap", "p", "Holm-adjusted p",
                                   "Gate 4", "Unadjusted reading"], body)]


def _pilot_section(report: dict) -> list[str]:
    gates, pilot = report.get("gates"), report["pilot"]
    out = ["", _heading("Pilot", report), ""]
    if gates is None:
        return out + [f"There is no pilot to decide ({'; '.join(pilot['reasons'])})."]
    coverage, systems = gates["coverage"], gates["pilot_systems"]
    if not gates["evaluable"]:
        names = _listed_words([_name(report, s) for s in systems])
        out += [f"The pilot is not evaluable, so no gate decides: {'; '.join(gates['reasons'])}.",
                "", f"The gates decide only in the registered pilot run, and only when all six "
                    f"pilot systems ({names}; under M0's fallback, section 13, graphiti-reduced "
                    f"takes graphiti's place) have H- answers to the pilot items: the registered "
                    f"pilot run's scope is exactly the items of pilot.json, given with --items, "
                    f"whose F1 items the gates read in H-, with every pilot system entered in H- "
                    f"F1, so a run with no --items, which scores every item, never decides; its "
                    f"build is the registered one, data/v0.1; and its answers are "
                    f"{REGISTERED_ANSWERING_MODEL}'s, with {REGISTERED_ANSWERING_SETTINGS}, over "
                    f"stores {REGISTERED_CONSTRUCTION_MODEL} built, with text embedded by "
                    f"{REGISTERED_EMBEDDING_MODEL}."]
        if any(name.startswith("gate") for name in gates["detail"]):
            out += ["", "gates.json records each gate's computation on the pilot items in the "
                        "run's scope, which decides nothing."]
    else:
        detail = gates["detail"]
        outcome = _outcome(gates["stopped_at"], gates["signal"])
        out.append(f'{outcome[0].upper()}{outcome[1:]}. The decision is "{gates["decision"]}".')
        other = detail["by_majority_accuracy"]
        reading = (_outcome(other["stopped_at"], other["signal"]) if other["evaluable"]
                   else f"the pilot is {other['decision']}")
        out += ["", "Read on each item's majority result instead of the accuracy averaged over "
                    f"the runs, {reading}."]
        body = [[_name(report, s), detail["accuracy_change"].get(s),
                 detail["accuracy_current"].get(s)] for s in systems]
        out += ["", *_table(["System", "H- F1-change", "H- F1-current"], body)]
        if detail["gate4"]["evaluable"]:
            out += _gate4_lines(report, detail["gate4"], gates["stopped_at"])
    gaps = {s: c for s, c in coverage["systems"].items() if c["missing_count"]}
    out.append("")
    # The counts are of the pilot items in the run's scope (section 12): an answer the scope
    # sets aside is listed with the answers outside it, never as missing here.
    if gaps:
        missing = "; ".join(f"{_name(report, s)} has no answer to {c['missing_count']} of "
                            f"{c['asked']} items ({_shown(c['missing'], c['missing_count'])})"
                            for s, c in gaps.items())
        out.append("Pilot items in the run's scope with no answer count as wrong (section 12). "
                   f"Missing: {missing}.")
    elif coverage["systems"]:
        out.append(f"Every system with pilot answers answered all {coverage['pilot_items']} "
                   "pilot items in the run's scope.")
    return out


def _warning(system: str, entries_file: str | None) -> str:
    """The warning for a system an entries file declares and no answer row names."""
    return (f"{system} is declared in {entries_file} but has no answer rows; is its name as the "
            "answers write it?")


def _runs_words(runs: list[str]) -> str:
    """``run 3``, ``runs 2 and 3``, ``runs 1, 2 and 3``."""
    return f"run{'s' if len(runs) > 1 else ''} {_listed_words(runs)}"


def _problem_sections(report: dict) -> list[str]:
    """The lists at the end of results.md; each names a system as the tables do (``_name``)."""
    a = report["answers"]

    def name(system: str) -> str:
        return _name(report, system)

    lists = (
        ("Warnings", [_warning(name(system), report["scope"]["entries_file"])
                      for system in report["silent"]]),
        ("Ids that are not in the questions", report["unknown_ids"]),
        ("Repeated answers, of which the first stands (by file name, then line)",
         [f"{d['file']} line {d['line']}: {name(d['system'])} {d['arm']} {d['id']} run "
          f"{d['run']}" for d in a["duplicates"]]),
        ("Answers in an arm the item is not asked in",
         [f"{name(u['system'])} {u['arm']} {u['id']}" for u in a["unasked"]]),
        ("Answers outside a system's entries or the items of this run",
         [f"{name(o['system'])} {o['arm']} {o['id']}" for o in a["out_of_scope"]]),
        ("Runs other than 1, 2 and 3, set aside",  # the label as text, quoted as JSON writes it
         [f"{o['file']} line {o['line']}: {name(o['system'])} {o['arm']} {o['id']} run "
          f"{json.dumps(o['run'], ensure_ascii=False)}" for o in a["odd_runs"]]),
        ("Arms a system is entered in and has no answers in, so it scores zero there",
         [f"{name(e['system'])}: {', '.join(e['arms'])}" for e in a["arms_without_answers"]]),
        ("Systems with a run that has no answer",
         [f"{name(m['system'])} in arm {m['arm']}: no answer in {_runs_words(m['runs'])}, so "
          f"every item is wrong in it" for m in a["missing_runs"]]),
    )
    out: list[str] = []
    for title, entries in lists:
        if entries:
            out += ["", f"## {title}", ""] + [f"- {e}" for e in entries[:MAX_LISTED]]
    return out


def _markdown(report: dict) -> str:
    a = report["answers"]
    scope = report["scope"]
    which = _scope_words(scope)
    entries = (f"the entries declared in {scope['entries_file']}" if scope["entries_file"]
               else "the registered entries (every system in every arm and family, long context "
                    "and graphiti-reduced in F1, F2 and CF only)")
    what = f"Scope: {which}; entries: {entries}."
    summary = (f"Scored {a['scored']} answers of {a['rows']} read from {len(a['files'])} file(s), "
               f"each run counted as 1, 2 or 3. Set aside: {a['duplicate_count']} repeated, "
               f"{a['unasked_count']} in an arm the item is not asked in, "
               f"{a['out_of_scope_count']} out of scope, {a['odd_run_count']} with another run "
               f"label and {len(report['unknown_ids'])} id(s) that are not in the questions.")
    models = (f"Answering model: {report['answering_model']}, with "
              f"{report['answering_settings']}; construction model: "
              f"{report['construction_model']}; embedding model: {report['embedding_model']}.")
    lines = ["# asof-bench v0.1 results", "", CANARY, "", models, "", what, "", summary, "",
             *_companion_lines(report)]
    for section in (_primary_section, _headline_section, _secondary_section, _coverage_section,
                    _entries_section, _accounting_section, _optional_sections, _pilot_section,
                    _problem_sections):
        lines += section(report)
    return "\n".join(lines) + "\n"


def _companion_lines(report: dict) -> list[str]:
    """Which companion files of the streams the report read, and whether each is the file the
    build's manifest registers; a file not given is said so, never silently left out. So is the
    training cutoff (section 10)."""
    day = report["training_cutoff"]
    if day is None:
        cutoff = ("- Training cutoff: not given (--training-cutoff), so F1-recent is not split by "
                  "the answering model's training cutoff.")
    elif report["recent_cutoff_split"]:
        cutoff = (f"- Training cutoff: {day}, the first day after the answering model's training "
                  "data (--training-cutoff), so F1-recent is also split by whether each item's "
                  "transition was recorded before it or on or after it.")
    else:
        cutoff = (f"- Training cutoff: {day}, the first day after the answering model's training "
                  "data (--training-cutoff); no F1-recent item is in the run's scope, so nothing "
                  "is split by it.")
    out = [cutoff]
    for label, entry, option, without in (
        ("CF message counts", report["cf_messages"], "--cf-messages",
         "so the counterfactual items table has no counts"),
        ("H- message cues", report["message_cues"], "--message-cues",
         "so the H- F1 items are not read against the commit messages kept by their dates"),
    ):
        if entry is None:
            out.append(f"- {label}: no file given ({option}), {without}.")
        elif entry["checked_against_manifest"]:
            out.append(f"- {label}: {entry['file']}, the file the build's manifest registers (its "
                       "SHA-256 matches).")
        else:
            out.append(f"- {label}: {entry['file']}, which the build's manifest does not "
                       "register (the build has no manifest, or one that registers no files "
                       "beside its streams), so it is not checked.")
    return out


def _write(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _data(data: Path) -> tuple[dict[str, dict], dict[str, dict], dict | None]:
    """The build's questions and gold, checked against each other and its manifest, and the
    manifest (None for a build without one)."""
    manifest = _manifest(data)
    _check_registered(data, manifest)
    questions = _read_records(data / "questions.jsonl", _QUESTION_FIELDS)
    gold = _read_records(data / "gold.jsonl", _GOLD_FIELDS)
    absent = sorted(set(questions) - set(gold))
    if absent:
        raise ScoringError(f"no gold for {len(absent)} question(s), such as {absent[0]}")
    extra = sorted(set(gold) - set(questions))
    if extra:
        raise ScoringError(f"{data / 'gold.jsonl'} has gold for {len(extra)} id(s) that are not "
                           f"in the questions, such as {extra[0]}")
    return questions, gold, manifest


def main(args) -> int:
    """``asofbench score``: write the results and, when the data have a pilot, its gates."""
    data, out = Path(args.data), Path(args.out)
    paths = [Path(p) for p in args.answers]
    cutoff = getattr(args, "training_cutoff", None)
    # Section 11: every table names the models and the answering settings; a run is one
    # answering model's answers.
    models = {"answering_model": args.answering_model,
              "answering_settings": args.answering_settings,
              "construction_model": args.construction_model,
              "embedding_model": args.embedding_model}
    try:
        questions, gold, manifest = _data(data)
        build = _build_reasons(data)  # sections 12 and 13: whether this is the registered build
        scope = _scope(args, questions)
        _check_flags(data / "gold.jsonl", questions, gold, scope)
        if cutoff is not None:
            _check_transition_times(data / "gold.jsonl", questions, gold, scope)
        counts, counts_file, counts_checked = _cf_counts(
            getattr(args, "cf_messages", None), questions, scope, data, manifest)
        cues, cues_file, cues_checked = _message_cues(getattr(args, "message_cues", None), data,
                                                      manifest)
        scored, found = _answers(paths, questions, gold, scope)
        graphiti = _graphiti_of(found["named"] | set(scope.declared))
        _check_models(found["named"] | set(scope.declared), args.embedding_model,
                      args.construction_model)
        pilot, gates = _pilot(data, scored, questions, scope, models, build, graphiti)
    except ScoringError as e:
        print(f"asofbench score: {e}", file=sys.stderr)
        return 2
    systems = sorted(set(scope.declared) | found["systems"])
    denominators = _denominators(questions, scope, systems)
    results = aggregate(scored, denominators)
    cells = _grid(scored, denominators)
    for r in results:  # what always giving the group's most common gold value scores
        items = cells[tuple(r[k] for k in GROUP)]
        r["majority_class_rate"] = majority_class(gold[i]["gold"] for i in items)[1] / len(items)
    cf_items = _cf_items(questions, gold, cells, counts, scope)
    cue_items = [] if cues is None else _cue_items(questions, gold, cues, scope)
    scope_words = {"items": None if scope.items is None else len(scope.items),
                   "items_file": scope.items_file}
    # Section 12: only the registered full run makes claims; any other run says why it is not.
    plan = _headline_plan(cells, questions, scope, graphiti)
    narrowed = _narrowed(scope)
    headline = _headline(cells, gold, plan, None if cues is None else cue_items,
                         _registration_reasons(scope_words, narrowed, plan["not_made"],
                                               plan["incomplete"], models, build))
    coverage = _coverage(results, scored)
    answered = {(r["system"], r["arm"]) for r in scored}
    contamination = _contamination(gold, scored, denominators)
    # Declared in the entries file and named by no answer row: each is a warning.
    silent = [system for system in sorted(scope.declared) if system not in found["named"]]
    report = {
        "canary": CANARY,
        **models,
        "answers": {
            "files": sorted(p.name for p in paths),
            "rows": found["rows"],
            "scored": len(scored),
            "duplicates": found["duplicates"][:MAX_LISTED],
            "duplicate_count": len(found["duplicates"]),
            "unasked": found["unasked"][:MAX_LISTED],
            "unasked_count": len(found["unasked"]),
            "out_of_scope": found["out_of_scope"][:MAX_LISTED],
            "out_of_scope_count": len(found["out_of_scope"]),
            "odd_runs": found["odd_runs"][:MAX_LISTED],
            "odd_run_count": len(found["odd_runs"]),
            "missing_runs": _missing_runs(scored),
            "arms_without_answers": _arms_without_answers(answered, denominators),
        },
        "scope": {
            **scope_words,
            "entries_file": scope.entries_file,
            "entries": {system: {"arms": sorted(scope.entry(system)[0]),
                                 "families": sorted(scope.entry(system)[1])}
                        for system in systems},
            # Section 11: declared in the entries file and not a registered entrant.
            "outside_entries": sorted(s for s in scope.declared if s not in REGISTERED_SYSTEMS),
        },
        "warnings": [_warning(system, scope.entries_file) for system in silent],
        "unknown_ids": sorted(found["unknown"]),
        "results": results,
        "headline": headline,
        "items": _item_runs(cells),
        "secondary": secondary(scored, denominators),
        "coverage": coverage,
        "accounting": _accounting(scored, systems),
        "cf_split": _counterfactual_split(scored, gold, denominators),
        "cf_header_split": _counterfactual_split(scored, gold, denominators, "header_cue"),
        "cf_items": cf_items,
        "cf_messages": None if counts is None else {
            "file": counts_file,
            "total": sum(i["cf_messages"] or 0 for i in cf_items),
            "checked_against_manifest": counts_checked,
        },
        "message_cues": None if cues is None else {
            **_cue_summary(cue_items, cues_file), "checked_against_manifest": cues_checked},
        "message_cue_items": cue_items,
        "message_cue_split": _cue_split(cells, cue_items),
        "header_only_split": _header_only_split(cells, gold),
        "f2_split": _f2_split(cells, gold),
        "contamination": contamination,
        "closed_book_split": _closed_book_split(cells, contamination),
        "recent_split": _recent_split(cells, gold),
        "training_cutoff": None if cutoff is None else cutoff.isoformat(),
        "recent_cutoff_split": [] if cutoff is None else _cutoff_split(cells, gold, cutoff),
        "pilot": pilot,
    }
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "results.json", dumps(report) + "\n")
    # results.md also reads the gates and the system ids it names with their labels (_name),
    # which results.json keeps as ids.
    _write(out / "results.md", _markdown({**report, "gates": gates, "silent": silent,
                                          "build_reasons": build, "narrowed": narrowed,
                                          "incomplete": plan["incomplete"]}))
    gates_file = out / "gates.json"
    if gates is not None:
        _write(gates_file, dumps({"canary": CANARY, **models, **gates}) + "\n")
    elif gates_file.exists():
        gates_file.unlink()  # a gates file left by an earlier run would not match these answers
    print(f"asofbench score: {len(scored)} answers scored, {len(report['unknown_ids'])} unknown "
          f"id(s), results written to {out}")
    return 0
