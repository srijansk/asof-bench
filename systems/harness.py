"""The answering harness, on the host (PREREGISTRATION.md sections 8, 11 and 12): it keeps each
item's id, arm, run and answer type to itself, writes the requests that a system's container
answers, and turns the container's responses into answer rows.

    uv run python systems/harness.py requests --system closed-book --data data/v0.1 \\
        --out-dir <run dir> [--items <file>] [--runs 1,2,3]

reads the build's questions.jsonl, after checking it against the SHA-256 that the build's
MANIFEST.json registers, and writes a new run directory, outside the repository:

- ``input/requests.jsonl``: one request per item, arm and run that the system is asked, each a
  JSON line with ``key``, a random UUID4 that says nothing of the item, and ``question``, the
  item's question text, and nothing else (long context and the oracle add ``context``). The
  ``input`` directory holds nothing else: it is what the system's container is given, read-only
  at /input;
- ``output/``, empty, where the container writes its responses (mounted at /output);
- ``keys.jsonl``: one line per request, in the same order, with ``key``, ``id``, ``arm``, ``run``
  and ``answer_type``, which never leave the host;
- ``plan.json``: the system, the runs, the number of requests and their counts per arm, family
  and run, and the name and SHA-256 of each file they come from (questions.jsonl, MANIFEST.json
  and the items file) and of requests.jsonl and keys.jsonl.

A system is asked the items of its entries as the scorer counts them (``asofbench.scoring.Scope``
with no entries file, section 12): closed book, the oracle and grep at D every item in every arm
its ``arms`` field lists, and long context only the F1, F2 and CF items. ``--items``, a list of
item ids or pilot.json's structure, read as ``asofbench score --items`` reads it, limits them to a
pilot's items, and ``--runs`` (by default 1,2,3) to some of the three runs.

    uv run python systems/harness.py rows --system closed-book --run-dir <run dir> \\
        --out <answers.jsonl>

reads the run directory's plan.json and keys.jsonl and the container's run.json and responses in
``output/``, and writes one answer row per key, in keys.jsonl's order, with the nine fields of
section 12 and no other: ``id``, ``system``, ``arm``, ``run``, ``answer``, ``tokens_in``,
``tokens_out``, ``wall_s`` and ``tool_calls``. The answer is shaped by the item's answer type and
nothing else is done to it (``shape``). A request that the system could not answer has a
response with a null answer, and so has its row. A key with no response at all, which a run that
stopped and was not resumed leaves, gets a row with a null answer and null cost fields, which
scores zero, and rows says how many and exits 1: resume the run first. ``--system`` is the name
the rows carry, the system that the requests were written for or an outside entry's name, such as
a toy run's, that no registered system has.

Exit status: 0 when the requests, or the rows, are written; 1 when rows wrote rows for keys with
no response; 2, with the reason, when an input is refused, and then nothing is written.

It uses the standard library and the asofbench package, and puts systems/common on the import
path, since nothing installs ``asofbench_systems`` on the host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import uuid
from pathlib import Path

SYSTEMS = Path(__file__).resolve().parent
if str(SYSTEMS / "common") not in sys.path:
    sys.path.insert(0, str(SYSTEMS / "common"))

from asofbench_systems import answer  # noqa: E402
from asofbench_systems.run import RUN, Refused, read_jsonl, write_json  # noqa: E402

from asofbench import scoring  # noqa: E402

INPUT, OUTPUT = "input", "output"
REQUESTS, KEYS, PLAN = "requests.jsonl", "keys.jsonl", "plan.json"
QUESTIONS, MANIFEST = "questions.jsonl", "MANIFEST.json"
KEY_FIELDS = ("key", "id", "arm", "run", "answer_type")
# Section 12: the nine fields of an answer row, of which the last four are its costs.
ROW_FIELDS = ("id", "system", "arm", "run", "answer", "tokens_in", "tokens_out", "wall_s",
              "tool_calls")
COSTS = ROW_FIELDS[5:]
# A response, as the reader writes it (reader/answer_requests.py): the key, then answer_once's.
RESPONSE_FIELDS = ("key", *answer.RESULT_FIELDS)
QUESTION_FIELDS = ("id", "question", "family", "answer_type", "arms")
RUNS = (1, 2, 3)
# The systems whose requests the harness writes, each with the function that gives a request its
# context, or None for a system that receives the question alone (closed book).
CONTEXTS = {"closed-book": None}
# A count written as a string: ASCII digits, after an optional minus sign.
COUNT = re.compile(r"-?[0-9]+")
CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


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


def read_questions(data: Path) -> tuple[list[dict], dict, dict]:
    """The build's questions, in file order, after checking questions.jsonl against the SHA-256
    that the build's MANIFEST.json registers; with each file's name and SHA-256."""
    manifest = load_json(data / MANIFEST)
    entry = (manifest.get("files") or {}).get(QUESTIONS)
    registered = entry.get("sha256") if isinstance(entry, dict) else None
    if not (data / QUESTIONS).is_file():
        raise Refused(f"{data} holds no {QUESTIONS}")
    actual = sha256_file(data / QUESTIONS)
    if actual != registered:
        raise Refused(f"{QUESTIONS} is not the build's: its SHA-256 is {actual}, and the build's "
                      f"manifest ({MANIFEST}) registers {registered}")
    questions = read_jsonl(data / QUESTIONS)
    ids = set()
    for number, q in enumerate(questions, 1):
        if (not isinstance(q, dict) or any(name not in q for name in QUESTION_FIELDS)
                or not isinstance(q["question"], str) or not q["question"]
                or q["answer_type"] not in ("status", "set", "count")):
            raise Refused(f"{QUESTIONS}: line {number} is not a question")
        if q["id"] in ids:
            raise Refused(f"{QUESTIONS}: line {number}'s id occurs twice")
        ids.add(q["id"])
    return (questions, {"file": QUESTIONS, "sha256": actual},
            {"file": MANIFEST, "sha256": sha256_file(data / MANIFEST)})


def read_items(path: Path, questions: list[dict]) -> tuple[frozenset[str], dict]:
    """The item ids of an items file, read as ``asofbench score --items`` reads it (the scorer's
    own reading, so that a pilot run's requests are the items that its score counts), with the
    file's name, SHA-256 and number of ids; Refused for a file the scorer would refuse."""
    try:
        items, _ = scoring._items(path, {q["id"]: q for q in questions})
    except scoring.ScoringError as error:
        raise Refused(str(error)) from None
    return items, {"file": path.name, "sha256": sha256_file(path), "ids": len(items)}


def plan_requests(questions: list[dict], system: str, items: frozenset[str] | None,
                  runs: tuple[int, ...]) -> list[tuple[dict, str, int]]:
    """Each item, arm and run that ``system`` is asked, in the questions' order, then the arms'
    that each item's ``arms`` field lists, then the runs': those of the system's entries as the
    scorer counts them, and of ``items`` when it is given."""
    scope = scoring.Scope(tuple(sorted({q["family"] for q in questions})), items=items)
    return [(q, arm, run) for q in questions for arm in q.get("arms") or scoring.ARMS
            if scope.contains(system, arm, q) for run in runs]


def write_requests(system: str, data: Path, out_dir: Path, items_path: Path | None,
                   runs: tuple[int, ...]) -> int:
    """The ``requests`` command (the module's description)."""
    if inside_repository(out_dir):
        raise Refused(f"{out_dir} is inside the repository: a run directory holds the requests "
                      "and the system's responses, so give one outside it")
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise Refused(f"{out_dir} is not a new, empty directory: give each run a directory of "
                      "its own")
    questions, questions_file, manifest_file = read_questions(data)
    items, items_file = (None, None) if items_path is None else read_items(items_path,
                                                                            questions)
    planned = plan_requests(questions, system, items, runs)
    if not planned:
        raise Refused(f"no item is asked: {system}'s entries hold none of the items given")
    context_of = CONTEXTS[system]
    (out_dir / INPUT).mkdir(parents=True)
    (out_dir / OUTPUT).mkdir()
    counts: dict[str, dict[str, dict[str, int]]] = {}
    used: set[str] = set()
    with open(out_dir / INPUT / REQUESTS, "w", encoding="utf-8") as requests, \
            open(out_dir / KEYS, "w", encoding="utf-8") as keys:
        for q, arm, run in planned:
            key = str(uuid.uuid4())
            while key in used:
                key = str(uuid.uuid4())
            used.add(key)
            request = {"key": key, "question": q["question"]}
            if context_of is not None:
                request["context"] = context_of(q, arm)
            requests.write(json.dumps(request, ensure_ascii=False) + "\n")
            keys.write(json.dumps({"key": key, "id": q["id"], "arm": arm, "run": run,
                                   "answer_type": q["answer_type"]}) + "\n")
            by_run = counts.setdefault(arm, {}).setdefault(q["family"], {})
            by_run[str(run)] = by_run.get(str(run), 0) + 1
    write_json(out_dir / PLAN, {
        "system": system, "runs": list(runs), "questions": questions_file,
        "manifest": manifest_file, "items": items_file, "requests": len(planned),
        "counts": counts, "requests_sha256": sha256_file(out_dir / INPUT / REQUESTS),
        "keys_sha256": sha256_file(out_dir / KEYS)})
    print(f"harness.py requests: {len(planned)} requests for {system} "
          f"({len({q['id'] for q, _, _ in planned})} items, runs "
          f"{', '.join(map(str, runs))}) in {out_dir / INPUT / REQUESTS}; the container is given "
          f"{out_dir / INPUT} read-only and {out_dir / OUTPUT}, and nothing else of the run "
          "directory")
    return 0


def read_responses(run_dir: Path, plan: dict, keys: set[str]) -> tuple[dict[str, dict], dict]:
    """The container's responses, by key, and its run.json; Refused unless the container answered
    this run's requests and every line is one key's response."""
    record_path = run_dir / OUTPUT / RUN
    if not record_path.is_file():
        raise Refused(f"the system has not answered this run: {OUTPUT}/{RUN} is missing")
    record = load_json(record_path)
    if record.get("requests_sha256") != plan["requests_sha256"]:
        raise Refused(f"the responses in {OUTPUT}/ answer another requests file: its SHA-256 is "
                      f"{record.get('requests_sha256')}, this run's {plan['requests_sha256']}")
    name = record.get("responses")
    if not isinstance(name, str) or Path(name).name != name or name in ("", ".", ".."):
        raise Refused(f"{OUTPUT}/{RUN} names no responses file in {OUTPUT}/")
    responses: dict[str, dict] = {}
    for number, response in enumerate(read_jsonl(run_dir / OUTPUT / name), 1):
        if not isinstance(response, dict) or set(response) != set(RESPONSE_FIELDS):
            raise Refused(f"{name}: line {number} is not a response")
        if response["key"] not in keys:
            raise Refused(f"{name}: line {number} answers a key that this run did not ask")
        if response["key"] in responses:
            raise Refused(f"{name}: line {number} answers a key that has a response already")
        responses[response["key"]] = response
    return responses, record


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
    if sha256_file(run_dir / KEYS) != plan.get("keys_sha256"):
        raise Refused(f"{KEYS} is not the one that {PLAN} records: its SHA-256 differs")
    keys = read_jsonl(run_dir / KEYS)
    responses, record = read_responses(run_dir, plan, {key["key"] for key in keys})
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
    temporary = out.with_name(f".{out.name}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, out)
    unanswered = sum(1 for response in responses.values() if response["error"])
    print(f"harness.py rows: {len(rows)} rows for {system} in {out}, answered by "
          f"{record.get('model')}: {len(rows) - missing - unanswered} with an answer, "
          f"{unanswered} that the system could not answer, {missing} with no response")
    if missing:
        print(f"harness.py rows: {missing} of the {len(rows)} keys have no response, so their "
              "rows have a null answer, which scores zero: the run stopped before it answered "
              "them; resume it, then write the rows again", file=sys.stderr)
        return 1
    return 0


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
                                  parse_runs(args.runs))
        return write_rows(args.system, args.run_dir, args.out)
    except Refused as refusal:
        print(f"harness.py {args.command}: {refusal}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
