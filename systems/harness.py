"""The answering harness, on the host (PREREGISTRATION.md sections 8, 11 and 12): it keeps each
item's id, arm, run and answer type to itself, writes the requests that a system's container
answers, and turns the container's responses into answer rows.

    uv run python systems/harness.py requests --system closed-book --data data/v0.1 \\
        --out-dir <run dir> [--items <file>] [--runs 1,2,3]
    uv run python systems/harness.py requests --system long-context|oracle --data data/v0.1 \\
        --streams-dir streams --out-dir <run dir> [--items <file>] [--runs 1,2,3]
    uv run python systems/harness.py requests --system grep-at-d --data data/v0.1 \\
        --out-dir <run dir> [--items <file>] [--runs 1,2,3]

reads the build's questions.jsonl, after checking it against the SHA-256 that the build's
MANIFEST.json registers, and writes a new run directory, outside the repository:

- ``input/requests.jsonl``: one request per item, arm and run that the system is asked, each a
  JSON line with ``key``, a random UUID4 that says nothing of the item, and ``question``, the
  item's question text, and nothing else (long context and the oracle add ``context``). The file
  is in ascending key order, which for random keys is a random permutation of the items' order,
  so that one item's requests are not next to one another: the reader answers a file in order
  and stops after a run of ``api_error`` responses, which must mean a cause that every request
  shares, and not one item whose context is over the model's window. The ``input`` directory
  holds nothing else: it is what the system's container is given, read-only at /input;
- ``output/``, empty, where the container writes its responses (mounted at /output);
- ``keys.jsonl``: one line per request, in the items' order (the questions', then the arms', then
  the runs'), with ``key``, ``id``, ``arm``, ``run`` and ``answer_type``, which never leave the
  host;
- ``plan.json``: the system, the runs, the number of requests and their counts per arm, family
  and run, and the name and SHA-256 of each file they come from (questions.jsonl, MANIFEST.json,
  the items file and the arms' streams that were read) and of requests.jsonl and keys.jsonl.

requests.jsonl and keys.jsonl are written under temporary names and renamed only after plan.json
is written, keys.jsonl first and requests.jsonl last, so that an interrupted invocation leaves no
requests.jsonl for a container to answer.

**Grep at D** reads its arm's tree as of the end of each item's date, which
grep-at-d/run_dates.py builds for one date at a time, so its requests are written one file for
each arm and date, ``input/<file>/requests.jsonl``, each in ascending key order and alone in its
directory, with an empty ``output/<file>/`` for its run; ``<file>`` is the arm's stream's name and
the date (``file_name``: ``h-minus-2021-09-03``). Beside them, ``dates.json`` maps each file to
its ``arm`` and ``as_of`` and holds nothing of the items, and plan.json adds each file's number of
requests and SHA-256 (``files``), and dates.json's SHA-256, with no single ``requests_sha256``.
dates.json is written last, after every requests file has its name, so that run_dates.py, which
reads it first, never runs a run directory that is not whole. Grep at D reads no stream here
(run_dates.py materialises the trees from them).

A system is asked the items of its entries as the scorer counts them (``asofbench.scoring.Scope``
with no entries file, section 12): closed book, the oracle and grep at D every item in every arm
its ``arms`` field lists, and long context only the F1, F2 and CF items. ``--items``, a list of
item ids or pilot.json's structure, read as ``asofbench score --items`` reads it, limits them to a
pilot's items, and ``--runs`` (by default 1,2,3) to some of the three runs.

**Contexts.** Closed book receives the question alone. Long context and the oracle receive a
``context`` as well, which the harness builds on the host from the arms' streams, ``--streams-dir``
(a directory holding the generator's files h-plus.jsonl, h-minus.jsonl and cf.jsonl; closed book
is given none). A stream's SHA-256 is checked against the one that the build's manifest registers
for the arm before the stream is read, and the bytes of the pass that parses it are hashed as
they are read (``checked_lines``) and must hash to the same, so that a stream that changes after
the check, or is not whole when it is read, is refused and the contexts built from it are never
used. Every stream is read and every context built before the run directory is made, so that a
stream the harness refuses leaves nothing behind. A context begins with a description of itself
(``LONG_CONTEXT_DESCRIPTION``, ``ORACLE_DESCRIPTION``), a blank line following it:

- long context (``long_contexts``): every episode of the proposal that the item names, in the
  item's arm's stream (H+, H- or CF), in stream order, those recorded after the item's date
  included, each as ``asofbench_systems.stream.render_episode`` renders it, joined by a line
  ``---``. Nothing is cut: a request over the model's window fails at the API as that request's
  own ``api_error`` and gets a null answer. Each stream is read once for all its proposals;
- the oracle (``oracle_contexts``): one line for every proposal of the item's corpus that has a
  file at the end of the item's date, in ascending number, ``{proposal}: status {status};
  requires {list or -}; replaces {list or -}; superseded-by {list or -}``, from the file's header
  as recorded (``statement``). The files come from ``asofbench.materialize.replay`` of the H+
  stream for H+ and H- items, which the oracle reads as recorded in both arms, and of the CF
  stream for CF items. Each stream is replayed once for all its dates, and each distinct
  (stream, corpus, date) is computed once.

    uv run python systems/harness.py rows --system closed-book --run-dir <run dir> \\
        --out <answers.jsonl>

reads the run directory's plan.json and keys.jsonl and the container's run.json and responses in
``output/`` (``asofbench_systems.answer.responses_by_key``; for grep at D, in each date's
``output/<file>/``, after checking dates.json and each date's requests file against plan.json and
each response against its date's keys), and writes, atomically, one answer
row per key, in keys.jsonl's order, with the nine fields of section 12 and no other: ``id``,
``system``, ``arm``, ``run``, ``answer``, ``tokens_in``, ``tokens_out``, ``wall_s`` and
``tool_calls``. The answer is shaped by the item's answer type and nothing else is done to it
(``shape``). A request that the system could not answer has a response with a null answer, and so
has its row. A key with no response at all, which a run that stopped and was not resumed leaves,
gets a row with a null answer and null cost fields, which scores zero, and rows says how many and
exits 1: after a stop with exit status 2 (or 4) resume the run first, and after one with exit
status 3 (``api_error`` responses in a row) discard it and run it again in a fresh run directory,
never resuming it; for grep at D, a date that has not run, or stopped, is resumed by running
run_dates.py again, and a date that stopped with status 3 means that the whole run is discarded.
``--system`` is the name the rows carry, the system that the requests were written for or an
outside entry's name, such as a toy run's, that no registered system has.

Exit status: 0 when the requests, or the rows, are written; 1 when rows wrote rows for keys with
no response; 2, with the reason, when an input is refused (one that cannot be read among them),
and then nothing is written; 4, with the traceback, for an error that no rule expects, so that no
failure exits with a status that means something else.

It uses the standard library and the asofbench package, and puts systems/common on the import
path, since nothing installs ``asofbench_systems`` on the host.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import sys
import traceback
import uuid
from collections.abc import Callable, Collection, Iterator
from datetime import date
from pathlib import Path
from typing import TypeVar

SYSTEMS = Path(__file__).resolve().parent
if str(SYSTEMS / "common") not in sys.path:
    sys.path.insert(0, str(SYSTEMS / "common"))

from asofbench_systems import answer  # noqa: E402
from asofbench_systems.run import (  # noqa: E402
    EXIT_CRASHED,
    EXIT_DONE,
    EXIT_FAILED,
    EXIT_FAILURES,
    EXIT_STOPPED,
    RUN,
    Refused,
    read_jsonl,
    sha256_file,
    temporary_path,
    write_json,
    write_text_atomically,
)
from asofbench_systems.stream import episodes_of, render_episode  # noqa: E402

from asofbench import headers, materialize, scoring  # noqa: E402
from asofbench.corpora import REPOS  # noqa: E402
from asofbench.manifest import ManifestError, registered_stream  # noqa: E402
from asofbench.stream import STREAM_FILES  # noqa: E402

INPUT, OUTPUT = "input", "output"
REQUESTS, KEYS, PLAN, DATES = "requests.jsonl", "keys.jsonl", "plan.json", "dates.json"
QUESTIONS, MANIFEST = "questions.jsonl", "MANIFEST.json"
KEY_FIELDS = ("key", "id", "arm", "run", "answer_type")
# Section 12: the nine fields of an answer row, of which the last four are its costs.
ROW_FIELDS = ("id", "system", "arm", "run", "answer", "tokens_in", "tokens_out", "wall_s",
              "tool_calls")
COSTS = ROW_FIELDS[5:]
# What a question record holds that the harness reads: the item's id, its question, its family and
# answer type, the arms that ask it, and the proposal it names (None for an F5 item), its corpus and
# its date, which only the context builders use.
QUESTION_FIELDS = ("id", "question", "family", "answer_type", "arms", "corpus", "proposal",
                   "as_of")
CORPORA = ("pep", "eth")
RUNS = (1, 2, 3)
# A count written as a string: ASCII digits, after an optional minus sign.
COUNT = re.compile(r"-?[0-9]+")
DAY = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
T = TypeVar("T")


def read_input(read: Callable[[Path], T], path: Path) -> T:
    """``read(path)`` for a file that the harness reads, with a file that cannot be read refused
    by its name and why, as ``load_json`` refuses one."""
    try:
        return read(path)
    except OSError as error:
        raise Refused(f"{path.name} cannot be read ({error.strerror or type(error).__name__})"
                      ) from None


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise Refused(f"{path.name} cannot be read as JSON: {error}") from None
    if not isinstance(value, dict):
        raise Refused(f"{path.name} is not a JSON object")
    return value


def inside_repository(path: Path) -> bool:
    """Whether ``path`` is in the checkout that this file is in."""
    root = SYSTEMS.parent.resolve()
    if not (root / "PREREGISTRATION.md").is_file():
        return False
    path = path.resolve()
    return path == root or root in path.parents


def is_integer(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def shape(answer_type: str, value):
    """The answer as its row carries it, shaped by the item's answer type and in no other way:
    for ``status``, a string is kept, an integer becomes its decimal string, and anything else is
    kept as it came; for ``set``, a list is kept, a string that parses as a JSON list is replaced
    by that list, and anything else is kept as it came; for ``count``, an integer is kept, a
    string of ASCII digits with an optional leading minus sign is replaced by its integer, and
    anything else is kept as it came. A boolean is no integer. The scorer normalises the rest."""
    if answer_type == "status":
        return str(value) if is_integer(value) else value
    if answer_type == "set":
        if not isinstance(value, str):
            return value
        # A string that json.loads cannot parse, for whatever reason, is kept as it came.
        try:
            parsed = json.loads(value)
        except (ValueError, RecursionError):
            return value
        return parsed if isinstance(parsed, list) else value
    if answer_type == "count":
        if isinstance(value, str) and COUNT.fullmatch(value):
            try:
                return int(value)
            except ValueError:  # more digits than int() converts
                return value
        return value
    raise ValueError(f"unknown answer_type {answer_type!r}")


def parse_runs(text: str) -> tuple[int, ...]:
    """The runs that ``--runs`` names, comma-separated, in ascending order; Refused unless each is
    1, 2 or 3, and named once."""
    words = text.split(",")
    if not all(word in ("1", "2", "3") for word in words) or len(set(words)) != len(words):
        raise Refused(f"--runs {text!r} is not a list of the runs 1, 2 and 3, each at most once")
    return tuple(sorted(int(word) for word in words))


def is_day(value) -> bool:
    """Whether ``value`` is a date written YYYY-MM-DD."""
    if not isinstance(value, str) or not DAY.fullmatch(value):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def read_questions(data: Path) -> tuple[list[dict], dict, dict, dict]:
    """The build's questions, in file order, after checking questions.jsonl against the SHA-256
    that the build's MANIFEST.json registers; with each file's name and SHA-256, and the
    manifest."""
    manifest = load_json(data / MANIFEST)
    entry = (manifest.get("files") or {}).get(QUESTIONS)
    registered = entry.get("sha256") if isinstance(entry, dict) else None
    if not (data / QUESTIONS).is_file():
        raise Refused(f"{data} holds no {QUESTIONS}")
    actual = read_input(sha256_file, data / QUESTIONS)
    if actual != registered:
        raise Refused(f"{QUESTIONS} is not the build's: its SHA-256 is {actual}, and the build's "
                      f"manifest ({MANIFEST}) registers {registered}")
    questions = read_input(read_jsonl, data / QUESTIONS)
    ids = set()
    for number, q in enumerate(questions, 1):
        if (not isinstance(q, dict) or any(name not in q for name in QUESTION_FIELDS)
                or not isinstance(q["question"], str) or not q["question"]
                or q["answer_type"] not in ("status", "set", "count")
                or q["corpus"] not in CORPORA or not is_day(q["as_of"])
                or not (q["proposal"] is None or isinstance(q["proposal"], str))):
            raise Refused(f"{QUESTIONS}: line {number} is not a question")
        if q["id"] in ids:
            raise Refused(f"{QUESTIONS}: line {number}'s id occurs twice")
        ids.add(q["id"])
    return (questions, {"file": QUESTIONS, "sha256": actual},
            {"file": MANIFEST, "sha256": read_input(sha256_file, data / MANIFEST)}, manifest)


def read_items(path: Path, questions: list[dict]) -> tuple[frozenset[str], dict]:
    """The item ids of an items file, read as ``asofbench score --items`` reads it (the scorer's
    own reading, so that a pilot run's requests are the items that its score counts), with the
    file's name, SHA-256 and number of ids; Refused for a file the scorer would refuse, or one
    that cannot be read."""
    by_id = {q["id"]: q for q in questions}
    try:
        items, _ = read_input(lambda listed: scoring._items(listed, by_id), path)
    except scoring.ScoringError as error:
        raise Refused(str(error)) from None
    return items, {"file": path.name, "sha256": read_input(sha256_file, path),
                   "ids": len(items)}


# --- the context builders of long context and the oracle (section 11) -------------------------

# The first line of each context: what the context holds, in the words the registration gives the
# entrant. A line of its own follows (a blank one), and then the context itself.
LONG_CONTEXT_DESCRIPTION = (
    "The context holds every recorded change to the file of the proposal the question names, "
    "oldest first: its full text when the file was created, then a unified diff for each later "
    "change, each with the time it was recorded and its commit message.")
ORACLE_DESCRIPTION = (
    "The context lists every proposal of the question's corpus present at the end of the "
    "question's date, with its status and its requires, replaces and superseded-by lists as its "
    "file's header recorded them then.")
# Joins the episodes of a long context, as a line of its own; and what stands for a status or a
# list that the header does not give.
EPISODE_SEPARATOR = "\n---\n"
NOTHING = "-"


@contextlib.contextmanager
def reading(path: Path) -> Iterator[None]:
    """The reading of the stream file at ``path``, in which every error that the file can cause
    (it cannot be read, is not UTF-8 text, or holds a line or an episode that is not one) is
    Refused, by the file's name. Nothing else belongs inside it: a mistake of the harness's own is
    no refused input."""
    try:
        yield
    except OSError as error:
        raise Refused(f"{path.name} cannot be read ({error.strerror or type(error).__name__})"
                      ) from None
    except UnicodeDecodeError:
        raise Refused(f"{path.name} is not UTF-8 text") from None
    except materialize.MaterializeError as error:
        raise Refused(str(error)) from None
    except KeyError as error:
        raise Refused(f"{path.name}: an episode lacks the field {error}") from None
    except ValueError as error:
        raise Refused(f"{path.name}: {str(error).removeprefix(f'{path.name}: ')}") from None


def checked_lines(path: Path, registered: str | None = None) -> Iterator[bytes]:
    """The lines of the stream file at ``path``, as bytes, split at newlines only, hashed as they
    are read: when the last line has been given, Refused unless the SHA-256 of the bytes that were
    read is ``registered`` (nothing is checked for None). A reader that parses these lines to the
    end has therefore parsed the registered stream and no other, whatever becomes of the file
    meanwhile; what it built from a pass that is refused is never used."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for raw in handle:
            digest.update(raw)
            yield raw
    if registered is not None and digest.hexdigest() != registered:
        raise Refused(f"{path.name} changed after it was checked: the bytes read from it have the "
                      f"SHA-256 {digest.hexdigest()}, not {registered}, which the build's "
                      f"manifest ({MANIFEST}) registers")


def long_contexts(stream_path: Path, proposals: Collection[str],
                  registered: str | None = None) -> dict[str, str]:
    """The long context of each of ``proposals`` (ids at S, such as ``PEP-8``) in the stream at
    ``stream_path``, by one reading of the stream (``checked_lines`` with ``registered``): the
    first line is ``LONG_CONTEXT_DESCRIPTION``, a blank line follows, and then every episode of
    the proposal that the stream holds, those recorded after the item's date included, in the
    stream's order, each as ``render_episode`` renders it, joined by a line ``---``. Nothing is
    cut, however long. Refused for a proposal with no episode in the stream."""
    rendered: dict[str, list[str]] = {proposal: [] for proposal in proposals}
    with reading(stream_path):
        for episode in episodes_of(checked_lines(stream_path, registered), stream_path.name):
            found = rendered.get(episode["proposal"])
            if found is not None:
                found.append(render_episode(episode))
    for proposal, pieces in rendered.items():
        if not pieces:
            raise Refused(f"{stream_path.name} holds no episode of {proposal}")
    return {proposal: f"{LONG_CONTEXT_DESCRIPTION}\n\n{EPISODE_SEPARATOR.join(pieces)}"
            for proposal, pieces in rendered.items()}


def number_of(proposal: str) -> int:
    """The number of a proposal id such as ``PEP-8`` or ``ERC-20``."""
    return int(proposal.rpartition("-")[2])


def listed(numbers: list[int]) -> str:
    return ", ".join(map(str, numbers)) or NOTHING


def statement(proposal: str, family: str, text: str) -> str:
    """The oracle's line for ``proposal``, whose file reads ``text``: its status as the header
    records it, and its requires, replaces and superseded-by lists, each as comma-separated
    numbers or ``-`` when the header gives none. The status is the header's own value, not
    normalised; requires is the requires field (``headers.number_list``), replaces the Replaces
    field, and superseded-by the Superseded-By field and, under its pre-2011 PEP name, Replaced-By
    (``headers.superseders``). Ethereum headers rarely have the last two, and then read ``-``."""
    header = headers.parse_header(family, text)
    return (f"{proposal}: status {header.get('status') or NOTHING}; "
            f"requires {listed(headers.number_list(header.get('requires')))}; "
            f"replaces {listed(headers.number_list(header.get('replaces')))}; "
            f"superseded-by {listed(headers.superseders(header))}")


def statements(files: dict[tuple[str, str], str], corpus: str) -> str:
    """The oracle's context for the files of one day (``(repo, proposal) -> text``, as
    ``materialize.replay`` gives them): the first line is ``ORACLE_DESCRIPTION``, a blank line
    follows, and then one ``statement`` for every proposal of ``corpus`` that has a file, in
    ascending number."""
    present = sorted((number_of(proposal), proposal, repo)
                     for repo, proposal in files if REPOS[repo].family == corpus)
    return f"{ORACLE_DESCRIPTION}\n\n" + "\n".join(
        statement(proposal, corpus, files[repo, proposal]) for _, proposal, repo in present)


def replayed(stream_path: Path, days: Collection[date], registered: str | None = None
             ) -> Iterator[tuple[date, dict[tuple[str, str], str]]]:
    """Each of ``days``, in date order, with the files as of its end, ``(repo, proposal) -> text``:
    ``asofbench.materialize.replay`` of the lines of the stream at ``stream_path`` as
    ``checked_lines`` gives them (with ``registered``), which Refused tells apart from the
    harness's own errors (``reading``)."""
    with reading(stream_path):
        lines = (raw.decode("utf-8") for raw in checked_lines(stream_path, registered))
        yield from materialize.replay(lines, days, where=stream_path.name)


def oracle_contexts(stream_path: Path, wanted: Collection[tuple[str, date]],
                    registered: str | None = None) -> dict[tuple[str, date], str]:
    """The oracle's context for each ``(corpus, day)`` of ``wanted``, in the stream at
    ``stream_path``, which one replay of it (``replayed``, with ``registered``) serves for all of
    them: the proposals that its files hold at the end of ``day`` (the episodes recorded by
    ``day`` 23:59:59 UTC, applied in stream order, a deleted proposal gone and one not yet created
    not there), as ``statements`` lists them."""
    corpora: dict[date, list[str]] = {}
    for corpus, day in sorted(wanted):
        corpora.setdefault(day, []).append(corpus)
    found: dict[tuple[str, date], str] = {}
    if not corpora:
        return found
    for day, files in replayed(stream_path, corpora, registered):
        for corpus in corpora[day]:
            found[corpus, day] = statements(files, corpus)
    return found


def stream_arm(system: str, arm: str) -> str:
    """The arm whose stream gives ``system`` its context for an item asked in ``arm``: long
    context reads the item's own arm's stream, and the oracle the headers as recorded (the H+
    stream) in H+ and H-, and the counterfactual stream's in CF (section 11)."""
    return "H+" if system == "oracle" and arm != "CF" else arm


def long_context_builder(planned: list[tuple[dict, str, int]],
                         streams: dict[str, tuple[Path, str]]) -> Callable[[dict, str], str]:
    """The function that gives a request of long context its context, from the planned items and
    the streams of their arms, each with the SHA-256 that its bytes must have: each stream is read
    once, for all of its proposals (``long_contexts``)."""
    wanted: dict[str, set[str]] = {}
    for q, arm, _ in planned:
        if q["proposal"] is None:
            raise Refused(f"{q['id']} names no proposal, so long context has nothing to read "
                          "for it")
        wanted.setdefault(arm, set()).add(q["proposal"])
    contexts = {}
    for arm, proposals in wanted.items():
        path, registered = streams[arm]
        contexts[arm] = long_contexts(path, proposals, registered)

    def context_of(q: dict, arm: str) -> str:
        return contexts[arm][q["proposal"]]

    return context_of


def oracle_builder(planned: list[tuple[dict, str, int]],
                   streams: dict[str, tuple[Path, str]]) -> Callable[[dict, str], str]:
    """The function that gives a request of the oracle its context, from the planned items and the
    streams it reads, each with the SHA-256 that its bytes must have: each stream is replayed once,
    and each distinct corpus and date that its items ask is computed once
    (``oracle_contexts``)."""
    wanted: dict[str, set[tuple[str, date]]] = {}
    for q, arm, _ in planned:
        wanted.setdefault(stream_arm("oracle", arm), set()).add(
            (q["corpus"], date.fromisoformat(q["as_of"])))
    contexts = {}
    for arm, days in wanted.items():
        path, registered = streams[arm]
        contexts[arm] = oracle_contexts(path, days, registered)

    def context_of(q: dict, arm: str) -> str:
        return contexts[stream_arm("oracle", arm)][q["corpus"], date.fromisoformat(q["as_of"])]

    return context_of


# The systems whose requests the harness writes, each with the function that builds, from the
# planned items and the streams it reads (arm to path and SHA-256), the function that gives a
# request its context; or None for a system whose requests hold the question alone (closed book,
# and grep at D, whose tree run_dates.py builds).
CONTEXTS = {"closed-book": None, "grep-at-d": None, "long-context": long_context_builder,
            "oracle": oracle_builder}
# The systems whose requests are written one file for each arm and date.
DATED = frozenset({"grep-at-d"})


def file_name(arm: str, day: str) -> str:
    """The name of grep at D's requests file for ``arm`` and ``day``, and of its input and output
    directories: the arm's stream's name without its suffix, and the day (``h-minus-2021-09-03``).
    """
    return f"{STREAM_FILES[arm].removesuffix('.jsonl')}-{day}"


def registered_streams(arms: Collection[str], streams_dir: Path, manifest: dict,
                       manifest_path: Path) -> dict[str, tuple[Path, str]]:
    """The stream of each of ``arms`` in ``streams_dir`` under the generator's name for it, with
    its SHA-256, after checking it against the SHA-256 that the build's manifest registers for the
    arm; Refused when it is not there, the manifest registers none, or they differ. This check
    refuses a wrong stream at once and by name; the pass that parses the stream hashes the bytes it
    parses again (``checked_lines``), which is what makes the digest recorded in plan.json the
    digest of what the contexts were built from."""
    found = {}
    for arm in sorted(arms):
        path = streams_dir / STREAM_FILES[arm]
        try:
            entry = registered_stream(manifest, arm, manifest_path)
        except ManifestError as error:
            raise Refused(str(error)) from None
        if entry is None:
            raise Refused(f"the build's manifest ({MANIFEST}) registers no {arm} stream")
        if not path.is_file():
            raise Refused(f"{streams_dir} holds no {path.name}, the {arm} stream")
        actual = read_input(sha256_file, path)
        if actual != entry["sha256"]:
            raise Refused(f"{path.name} is not the {arm} stream that the build registers: its "
                          f"SHA-256 is {actual}, and the build's manifest ({MANIFEST}) registers "
                          f"{entry['sha256']}")
        found[arm] = (path, actual)
    return found


def plan_requests(questions: list[dict], system: str, items: frozenset[str] | None,
                  runs: tuple[int, ...]) -> list[tuple[dict, str, int]]:
    """Each item, arm and run that ``system`` is asked, in the questions' order, then the arms'
    that each item's ``arms`` field lists, then the runs': those of the system's entries as the
    scorer counts them, and of ``items`` when it is given."""
    scope = scoring.Scope(tuple(sorted({q["family"] for q in questions})), items=items)
    return [(q, arm, run) for q in questions for arm in q.get("arms") or scoring.ARMS
            if scope.contains(system, arm, q) for run in runs]


def write_requests(system: str, data: Path, out_dir: Path, items_path: Path | None,
                   runs: tuple[int, ...], streams_dir: Path | None = None) -> int:
    """The ``requests`` command (the module's description)."""
    if inside_repository(out_dir):
        raise Refused(f"{out_dir} is inside the repository: a run directory holds the requests "
                      "and the system's responses, so give one outside it")
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise Refused(f"{out_dir} is not a new, empty directory: give each run a directory of "
                      "its own")
    builder = CONTEXTS[system]
    dated = system in DATED
    if dated and streams_dir is not None:
        raise Refused(f"{system} reads no stream here: run_dates.py materialises each date's tree "
                      "from the streams, so leave out --streams-dir")
    if builder is None and streams_dir is not None:
        raise Refused(f"{system} receives the question alone and reads no stream: leave out "
                      "--streams-dir")
    if builder is not None and streams_dir is None:
        raise Refused(f"{system} reads the arms' streams: give --streams-dir, the directory "
                      f"that holds {', '.join(STREAM_FILES.values())}")
    questions, questions_file, manifest_file, manifest = read_questions(data)
    items, items_file = (None, None) if items_path is None else read_items(items_path,
                                                                            questions)
    planned = plan_requests(questions, system, items, runs)
    if not planned:
        raise Refused(f"no item is asked: {system}'s entries hold none of the items given")
    # The streams are checked and the contexts built before anything is written, so that a stream
    # the harness refuses leaves no run directory behind.
    streams, context_of = {}, None
    if builder is not None:
        streams = registered_streams({stream_arm(system, arm) for _, arm, _ in planned},
                                     streams_dir, manifest, data / MANIFEST)
        context_of = builder(planned, streams)
    (out_dir / INPUT).mkdir(parents=True)
    (out_dir / OUTPUT).mkdir()
    # A key for each planned request, in the items' order; none is ever the same twice.
    used: set[str] = set()
    entries: list[tuple[str, dict, str, int]] = []
    for q, arm, run in planned:
        key = str(uuid.uuid4())
        while key in used:
            key = str(uuid.uuid4())
        used.add(key)
        entries.append((key, q, arm, run))
    # The requests files: the run's one (its name the empty one), or grep at D's, one for each arm
    # and date, each in a directory of its own.
    groups: dict[str, list[tuple[str, dict, str, int]]] = {}
    for entry in entries:
        _, q, arm, _ = entry
        groups.setdefault(file_name(arm, q["as_of"]) if dated else "", []).append(entry)
    names = sorted(groups)
    paths = {name: out_dir / INPUT / name / REQUESTS for name in names}
    pending = {name: temporary_path(path) for name, path in paths.items()}
    keys_path = out_dir / KEYS
    pending_keys = temporary_path(keys_path)
    for name in names if dated else ():
        (out_dir / INPUT / name).mkdir()
        (out_dir / OUTPUT / name).mkdir()
    counts: dict[str, dict[str, dict[str, int]]] = {}
    with open(pending_keys, "w", encoding="utf-8") as keys:
        for key, q, arm, run in entries:
            keys.write(json.dumps({"key": key, "id": q["id"], "arm": arm, "run": run,
                                   "answer_type": q["answer_type"]}) + "\n")
            by_run = counts.setdefault(arm, {}).setdefault(q["family"], {})
            by_run[str(run)] = by_run.get(str(run), 0) + 1
        keys.flush()
        os.fsync(keys.fileno())
    for name in names:
        with open(pending[name], "w", encoding="utf-8") as requests:
            # The requests in ascending key order, which for random keys is a random permutation
            # of the items' order: the reader answers a file in order, and stops after a run of
            # api_error responses, so that one item's requests, which would fail together if its
            # context were over the model's window, must not stand together.
            for key, q, arm, _ in sorted(groups[name], key=lambda entry: entry[0]):
                request = {"key": key, "question": q["question"]}
                if context_of is not None:
                    request["context"] = context_of(q, arm)
                requests.write(json.dumps(request, ensure_ascii=False) + "\n")
            requests.flush()
            os.fsync(requests.fileno())
    plan = {"system": system, "runs": list(runs), "questions": questions_file,
            "manifest": manifest_file, "items": items_file,
            "streams": {arm: {"file": path.name, "sha256": digest}
                        for arm, (path, digest) in streams.items()} or None,
            "requests": len(planned), "counts": counts,
            "requests_sha256": None if dated else sha256_file(pending[""]),
            "keys_sha256": sha256_file(pending_keys)}
    dates_text = None
    if dated:
        dates_text = json.dumps({name: {"arm": groups[name][0][2],
                                        "as_of": groups[name][0][1]["as_of"]}
                                 for name in names}, indent=2) + "\n"
        plan["files"] = {name: {"requests": len(groups[name]),
                                "requests_sha256": sha256_file(pending[name])} for name in names}
        plan["dates_sha256"] = hashlib.sha256(dates_text.encode("utf-8")).hexdigest()
    write_json(out_dir / PLAN, plan)
    # The keys first and the requests last: a run directory with a requests file is a whole one;
    # for grep at D, dates.json, which run_dates.py reads first, comes after every requests file.
    os.replace(pending_keys, keys_path)
    for name in names:
        os.replace(pending[name], paths[name])
    if dates_text is not None:
        write_text_atomically(out_dir / DATES, dates_text)
    described = (f"{len(planned)} requests for {system} "
                 f"({len({q['id'] for q, _, _ in planned})} items, runs "
                 f"{', '.join(map(str, runs))})")
    if dated:
        print(f"harness.py requests: {described} in {len(names)} files, one for each arm and "
              f"date, under {out_dir / INPUT}, mapped to their arms and dates in "
              f"{out_dir / DATES}; run_dates.py gives each date's run its input/<file>/ "
              "read-only and its output/<file>/, and nothing else of the run directory")
    else:
        print(f"harness.py requests: {described} in {paths['']}; the container is given "
              f"{out_dir / INPUT} read-only and {out_dir / OUTPUT}, and nothing else of the run "
              "directory")
    return EXIT_DONE


def read_responses(output: Path, requests_sha256: str | None, keys: set[str],
                   where: str = OUTPUT) -> tuple[dict[str, dict], dict]:
    """The responses of the container that wrote in ``output`` (the run directory's ``where``), by
    key, and its run.json; Refused unless it answered the requests file whose SHA-256 is
    ``requests_sha256`` and every line is the response to one of ``keys``."""
    record_path = output / RUN
    if not record_path.is_file():
        raise Refused(f"the system has not answered this run: {where}/{RUN} is missing")
    record = load_json(record_path)
    if record.get("requests_sha256") != requests_sha256:
        raise Refused(f"the responses in {where}/ answer another requests file: its SHA-256 is "
                      f"{record.get('requests_sha256')}, this run's {requests_sha256}")
    name = record.get("responses")
    if not isinstance(name, str) or Path(name).name != name or name in ("", ".", ".."):
        raise Refused(f"{where}/{RUN} names no responses file in {where}/")
    lines = read_input(read_jsonl, output / name)
    label = name if where == OUTPUT else f"{where}/{name}"
    return answer.responses_by_key(lines, keys, label), record


def read_dated_responses(run_dir: Path, plan: dict) -> tuple[dict[str, dict],
                                                             dict[str, dict | None]]:
    """Grep at D's responses, by key, gathered from each date's output directory, and each date's
    run.json (None for a date that has not run); Refused unless dates.json and each date's
    requests file are the ones that plan.json records, each date that has run answered its own
    file, every line is the response to one of that file's keys, and some date has run."""
    if read_input(sha256_file, run_dir / DATES) != plan.get("dates_sha256"):
        raise Refused(f"{DATES} is not the one that {PLAN} records: its SHA-256 differs")
    responses: dict[str, dict] = {}
    records: dict[str, dict | None] = {}
    for name, entry in plan["files"].items():
        requests_path = run_dir / INPUT / name / REQUESTS
        if read_input(sha256_file, requests_path) != entry["requests_sha256"]:
            raise Refused(f"{INPUT}/{name}/{REQUESTS} is not the one that {PLAN} records: its "
                          "SHA-256 differs")
        if not (run_dir / OUTPUT / name / RUN).is_file():
            records[name] = None
            continue
        keys = {request["key"] for request in read_input(read_jsonl, requests_path)}
        found, records[name] = read_responses(run_dir / OUTPUT / name, entry["requests_sha256"],
                                              keys, f"{OUTPUT}/{name}")
        responses.update(found)
    if all(record is None for record in records.values()):
        raise Refused(f"the system has not answered this run: no date's {OUTPUT}/<file>/ holds a "
                      f"{RUN}")
    return responses, records


def last_exit(record: dict) -> int | None:
    """The exit status of the run's last invocation, as run.json records it; None when it records
    none (an invocation killed outright, or a record with no invocation)."""
    invocations = record.get("invocations")
    last = invocations[-1] if isinstance(invocations, list) and invocations else None
    return last.get("exit") if isinstance(last, dict) else None


def write_rows(system: str, run_dir: Path, out: Path) -> int:
    """The ``rows`` command (the module's description)."""
    if not system or system != system.strip():
        raise Refused(f"--system {system!r} is no name")
    plan = load_json(run_dir / PLAN)
    if system != plan.get("system") and system in scoring.REGISTERED_SYSTEMS:
        raise Refused(f"the requests were written for {plan.get('system')}: their rows carry that "
                      f"name, or an outside entry's that no registered system has, not {system}")
    if out.exists():
        raise Refused(f"{out} exists already: give the rows a new file")
    if not out.parent.is_dir():
        raise Refused(f"{out.parent} is not a directory")
    if read_input(sha256_file, run_dir / KEYS) != plan.get("keys_sha256"):
        raise Refused(f"{KEYS} is not the one that {PLAN} records: its SHA-256 differs")
    keys = read_input(read_jsonl, run_dir / KEYS)
    dated = "files" in plan
    if dated:
        responses, records = read_dated_responses(run_dir, plan)
    else:
        responses, record = read_responses(run_dir / OUTPUT, plan.get("requests_sha256"),
                                           {key["key"] for key in keys})
        records = {"": record}
    rows, missing = [], 0
    for key in keys:
        row = {"id": key["id"], "system": system, "arm": key["arm"], "run": key["run"]}
        response = responses.get(key["key"])
        if response is None:
            missing += 1
            rows.append({**row, "answer": None, **dict.fromkeys(COSTS)})
        else:
            rows.append({**row, "answer": shape(key["answer_type"], response["answer"]),
                         **{name: response[name] for name in COSTS}})
    write_text_atomically(out, "".join(json.dumps(row, ensure_ascii=False) + "\n"
                                       for row in rows))
    unanswered = sum(1 for response in responses.values() if response["error"])
    models = sorted({str(record.get("model")) for record in records.values() if record})
    print(f"harness.py rows: {len(rows)} rows for {system} in {out}, answered by "
          f"{', '.join(models)}: {len(rows) - missing - unanswered} with an answer, "
          f"{unanswered} that the system could not answer, {missing} with no response")
    stopped = [name for name, record in records.items()
               if record is not None and last_exit(record) == EXIT_FAILURES]
    if missing:
        if dated and stopped:
            next_step = (f"{', '.join(stopped)} stopped with exit status {EXIT_FAILURES} "
                         "(api_error responses in a row, which point at a cause that every "
                         "request shares): discard the run and run it again in a fresh run "
                         "directory once the cause is fixed; never resume it (systems/README.md)")
        elif dated:
            next_step = ("a date has not run, or stopped before it answered them: run "
                         "grep-at-d/run_dates.py again, which resumes it, then write the rows "
                         "again")
        elif stopped:
            # Each api_error response is its request's own and stays, but api_error responses in
            # a row point at a cause that every request shares: the run is discarded, never
            # resumed under changed settings.
            next_step = (f"the run stopped with exit status {EXIT_FAILURES} (api_error responses "
                         "in a row, which point at a cause that every request shares): discard "
                         "the run and run it again in a fresh run directory once the cause is "
                         "fixed; never resume it (systems/README.md)")
        else:
            next_step = ("the run stopped before it answered them; resume it, then write the "
                         "rows again")
        print(f"harness.py rows: {missing} of the {len(rows)} keys have no response, so their "
              f"rows have a null answer, which scores zero: {next_step}", file=sys.stderr)
        return EXIT_FAILED
    return EXIT_DONE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="harness.py", description="The answering harness: "
                                     "a system's requests, and its answer rows.",
                                     allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")
    asked = commands.add_parser("requests", help="write a run's requests", allow_abbrev=False)
    asked.add_argument("--system", required=True, choices=sorted(CONTEXTS))
    asked.add_argument("--data", required=True, type=Path, help="the build, such as data/v0.1")
    asked.add_argument("--out-dir", required=True, type=Path, dest="out_dir",
                       help="a new run directory, outside the repository")
    asked.add_argument("--streams-dir", type=Path, dest="streams_dir",
                       help="the directory that holds the arms' streams (h-plus.jsonl, "
                            "h-minus.jsonl and cf.jsonl), which long-context and oracle read; "
                            "closed-book is given none")
    asked.add_argument("--items", type=Path, help="a list of item ids, or pilot.json")
    asked.add_argument("--runs", default="1,2,3", help="the runs to ask, of 1, 2 and 3")
    answered = commands.add_parser("rows", help="write a run's answer rows", allow_abbrev=False)
    answered.add_argument("--system", required=True, help="the name the rows carry")
    answered.add_argument("--run-dir", required=True, type=Path, dest="run_dir")
    answered.add_argument("--out", required=True, type=Path, help="the answer rows' new file")
    args = parser.parse_args(argv)
    try:
        if args.command == "requests":
            return write_requests(args.system, args.data, args.out_dir, args.items,
                                  parse_runs(args.runs), args.streams_dir)
        return write_rows(args.system, args.run_dir, args.out)
    except Refused as refusal:
        print(f"harness.py {args.command}: {refusal}", file=sys.stderr)
        return EXIT_STOPPED
    except Exception:  # noqa: BLE001 - an error that no rule expects: its own exit status
        print(f"harness.py {args.command}: crashed (exit {EXIT_CRASHED})\n"
              f"{traceback.format_exc()}", file=sys.stderr)
        return EXIT_CRASHED


if __name__ == "__main__":
    sys.exit(main())
