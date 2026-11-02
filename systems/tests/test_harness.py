"""The answering harness on the host (systems/harness.py), on the toy world that
examples/toy_world.py builds: the requests it writes and what each holds, what never leaves the
host, the items and entries it asks, and the answer rows it writes from a run's responses, which
`asofbench score` scores as an outside entry. It needs git, which builds the toy world's
repositories, and the asofbench package, so the reader image skips it; it makes no model call
and reads no v0.1 file."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import uuid
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("asofbench")
if shutil.which("git") is None:
    pytest.skip("the toy world's repositories need git", allow_module_level=True)

import harness  # noqa: E402 - systems/ is on the path (conftest.py)

from asofbench import cli, scoring  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
NINE = ["id", "system", "arm", "run", "answer", "tokens_in", "tokens_out", "wall_s",
        "tool_calls"]


@pytest.fixture(scope="module")
def toy(tmp_path_factory) -> Path:
    """The toy world, built once: its data (toy-data/) and its H- stream."""
    spec = importlib.util.spec_from_file_location("toy_world", ROOT / "examples" / "toy_world.py")
    world = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(world)
    root = tmp_path_factory.mktemp("toy") / "world"
    assert world.main([str(root)]) == 0
    return root


@pytest.fixture(scope="module")
def questions(toy) -> list[dict]:
    return lines(toy / "toy-data" / "questions.jsonl")


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_requests(toy: Path, out: Path, *extra: str, system: str = "closed-book") -> int:
    return harness.main(["requests", "--system", system, "--data", str(toy / "toy-data"),
                         "--out-dir", str(out), *extra])


def make_rows(run_dir: Path, out: Path, system: str = "closed-book") -> int:
    return harness.main(["rows", "--system", system, "--run-dir", str(run_dir), "--out", str(out)])


def respond(run_dir: Path, answers: dict[str, object], **record) -> None:
    """Writes what the reader would have written for ``answers`` (key to answer): its run.json and
    one response per key, each with made-up costs."""
    plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
    output = run_dir / "output"
    (output / "run.json").write_text(json.dumps({
        "requests": "requests.jsonl", "requests_sha256": plan["requests_sha256"],
        "responses": "responses.jsonl", "model": "claude-haiku-5-5", "invocations": [],
        **record}), encoding="utf-8")
    with open(output / "responses.jsonl", "w", encoding="utf-8") as handle:
        for n, (key, value) in enumerate(answers.items()):
            handle.write(json.dumps({"key": key, "answer": value, "tokens_in": 100 + n,
                                     "tokens_out": 5, "wall_s": 1.5, "tool_calls": 0,
                                     "error": None if value is not None else "refusal",
                                     "thinking_blocks": 1}) + "\n")


def test_closed_book_is_asked_every_item_in_every_arm_it_lists_three_times(toy, questions,
                                                                           tmp_path):
    run_dir = tmp_path / "run"
    assert make_requests(toy, run_dir) == 0
    requests = lines(run_dir / "input" / "requests.jsonl")
    keys = lines(run_dir / "keys.jsonl")
    expected = [(q["id"], arm, run) for q in questions for arm in q["arms"] for run in (1, 2, 3)]
    assert [(k["id"], k["arm"], k["run"]) for k in keys] == expected
    assert len(requests) == len(expected) == sum(len(q["arms"]) for q in questions) * 3
    # keys.jsonl maps every key back, in the requests' order, with the item's answer type.
    by_id = {q["id"]: q for q in questions}
    assert [r["key"] for r in requests] == [k["key"] for k in keys]
    for request, key in zip(requests, keys, strict=True):
        assert list(key) == ["key", "id", "arm", "run", "answer_type"]
        assert key["answer_type"] == by_id[key["id"]]["answer_type"]
        assert request == {"key": key["key"], "question": by_id[key["id"]]["question"]}
    # The run's input directory holds the requests and nothing else; the output one is empty.
    assert sorted(p.name for p in (run_dir / "input").iterdir()) == ["requests.jsonl"]
    assert list((run_dir / "output").iterdir()) == []
    assert sorted(p.name for p in run_dir.iterdir()) == ["input", "keys.jsonl", "output",
                                                         "plan.json"]


def test_every_key_is_a_new_random_uuid4_and_no_request_carries_the_item(toy, questions,
                                                                          tmp_path):
    assert make_requests(toy, tmp_path / "a") == 0
    assert make_requests(toy, tmp_path / "b") == 0
    first = [r["key"] for r in lines(tmp_path / "a" / "input" / "requests.jsonl")]
    second = [r["key"] for r in lines(tmp_path / "b" / "input" / "requests.jsonl")]
    for key in first:
        assert uuid.UUID(key).version == 4 and str(uuid.UUID(key)) == key
    # Unique within a run, the three runs of an item and its two arms included, and never the
    # same twice, so that a key says nothing of the item it stands for.
    assert len(set(first)) == len(first)
    assert not set(first) & set(second)
    text = (tmp_path / "a" / "input" / "requests.jsonl").read_text(encoding="utf-8")
    for q in questions:
        assert q["id"] not in text
    for request in lines(tmp_path / "a" / "input" / "requests.jsonl"):
        assert set(request) == {"key", "question"}


def test_plan_json_counts_the_requests_and_records_the_files_they_come_from(toy, questions,
                                                                            tmp_path):
    run_dir = tmp_path / "run"
    assert make_requests(toy, run_dir) == 0
    plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
    data = toy / "toy-data"
    assert plan["system"] == "closed-book"
    assert plan["runs"] == [1, 2, 3]
    assert plan["questions"] == {"file": "questions.jsonl",
                                 "sha256": sha256(data / "questions.jsonl")}
    assert plan["manifest"] == {"file": "MANIFEST.json", "sha256": sha256(data / "MANIFEST.json")}
    assert plan["items"] is None
    assert plan["requests"] == sum(len(q["arms"]) for q in questions) * 3
    assert plan["requests_sha256"] == sha256(run_dir / "input" / "requests.jsonl")
    assert plan["keys_sha256"] == sha256(run_dir / "keys.jsonl")
    family = {q["id"]: q["family"] for q in questions}
    tally = Counter((k["arm"], family[k["id"]], str(k["run"]))
                    for k in lines(run_dir / "keys.jsonl"))
    assert {(arm, fam, run): n for arm, by_family in plan["counts"].items()
            for fam, by_run in by_family.items() for run, n in by_run.items()} == dict(tally)
    # File names only: the plan holds no local path.
    assert str(tmp_path) not in json.dumps(plan) and str(ROOT) not in json.dumps(plan)


def test_items_and_runs_limit_the_requests(toy, questions, tmp_path):
    pilot = toy / "toy-data" / "pilot.json"
    assert make_requests(toy, tmp_path / "pilot", "--items", str(pilot)) == 0
    keys = lines(tmp_path / "pilot" / "keys.jsonl")
    listed = set(json.loads(pilot.read_text(encoding="utf-8"))["ids"])
    assert {k["id"] for k in keys} == listed
    by_id = {q["id"]: q for q in questions}
    assert len(keys) == sum(len(by_id[i]["arms"]) for i in listed) * 3
    plan = json.loads((tmp_path / "pilot" / "plan.json").read_text(encoding="utf-8"))
    assert plan["items"] == {"file": "pilot.json", "sha256": sha256(pilot), "ids": len(listed)}
    one = tmp_path / "one.json"
    one.write_text(json.dumps(["F1-change-PEP-1-2003-08-24"]), encoding="utf-8")
    assert make_requests(toy, tmp_path / "one", "--items", str(one), "--runs", "2") == 0
    assert [(k["id"], k["arm"], k["run"]) for k in lines(tmp_path / "one" / "keys.jsonl")] == [
        ("F1-change-PEP-1-2003-08-24", "H+", 2), ("F1-change-PEP-1-2003-08-24", "H-", 2)]
    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps(["F1-change-PEP-1-2003-08-24", "not-an-item"]),
                       encoding="utf-8")
    assert make_requests(toy, tmp_path / "unknown", "--items", str(unknown)) == 2
    assert not (tmp_path / "unknown").exists()


def test_each_system_is_asked_exactly_what_the_scorer_counts(toy, questions):
    every = [(q["id"], arm) for q in questions for arm in q["arms"]]
    named = [(q["id"], arm) for q in questions for arm in q["arms"]
             if q["family"] in ("F1", "F2", "CF")]
    families = sorted({q["family"] for q in questions})
    pilot = frozenset(json.loads((toy / "toy-data" / "pilot.json").read_text(
        encoding="utf-8"))["ids"])
    for system, expected in (("closed-book", every), ("oracle", every), ("grep-at-d", every),
                             ("long-context", named)):
        for items in (None, pilot):
            planned = harness.plan_requests(questions, system, items, (1, 2, 3))
            asked = [(q["id"], arm) for q, arm, run in planned if run == 1]
            assert asked == [entry for entry in expected
                             if items is None or entry[0] in items], (system, items)
            assert [run for _, _, run in planned] == [1, 2, 3] * len(asked)
            # The scorer's own denominators for the system, so that every request is an answer
            # it counts and it counts no item that was not asked.
            scope = scoring.Scope(tuple(families), items=items)
            counted = scoring._denominators({q["id"]: q for q in questions}, scope, [system])
            assert sorted(asked) == sorted((entry[0], arm) for arm, entries in
                                           counted[system].items() for entry in entries)
    # Long context is asked no F3, F4 or F5 item: the toy world's F4 and F5 are left out.
    asked = {q["family"] for q, _, _ in harness.plan_requests(questions, "long-context", None,
                                                              (1,))}
    assert asked == {"F1", "CF"} and {"F4", "F5"} <= set(families)


def test_requests_refuses_what_it_cannot_ask_and_writes_nothing(toy, tmp_path, capsys):
    with pytest.raises(SystemExit) as caught:
        make_requests(toy, tmp_path / "x", system="bm25")
    assert caught.value.code == 2
    for runs in ("1,1", "0", "4", "1,x", ""):
        assert make_requests(toy, tmp_path / "x", "--runs", runs) == 2, runs
    assert not (tmp_path / "x").exists()
    # A run directory that holds anything already, and one inside the repository.
    full = tmp_path / "full"
    full.mkdir()
    (full / "notes.txt").write_text("an earlier run's file\n", encoding="utf-8")
    assert make_requests(toy, full) == 2
    assert sorted(p.name for p in full.iterdir()) == ["notes.txt"]
    inside = ROOT / "systems" / "tests" / "never-written-run"
    assert make_requests(toy, inside) == 2 and not inside.exists()
    # Questions that the build's manifest does not register.
    data = tmp_path / "edited-data"
    shutil.copytree(toy / "toy-data", data)
    with open(data / "questions.jsonl", "a", encoding="utf-8") as handle:
        handle.write("\n")
    assert harness.main(["requests", "--system", "closed-book", "--data", str(data),
                         "--out-dir", str(tmp_path / "y")]) == 2
    assert not (tmp_path / "y").exists()
    assert "manifest" in capsys.readouterr().err


@pytest.mark.parametrize(("answer_type", "came", "row"), [
    ("status", "Draft", "Draft"), ("status", " final. ", " final. "), ("status", 7, "7"),
    ("status", -2, "-2"), ("status", True, True), ("status", ["Draft"], ["Draft"]),
    ("status", None, None), ("status", 1.5, 1.5),
    ("set", [1, "PEP 8"], [1, "PEP 8"]), ("set", "[1, \"PEP 8\"]", [1, "PEP 8"]),
    ("set", " [] ", []), ("set", "1, 2", "1, 2"), ("set", "{\"a\": [1]}", "{\"a\": [1]}"),
    ("set", "\"[1]\"", "\"[1]\""), ("set", 5, 5), ("set", None, None),
    ("count", 5, 5), ("count", "5", 5), ("count", "-3", -3), ("count", "007", 7),
    ("count", "+3", "+3"), ("count", " 3", " 3"), ("count", "3.0", "3.0"), ("count", "-", "-"),
    ("count", "٣", "٣"), ("count", True, True), ("count", 3.0, 3.0),
    ("count", [3], [3]), ("count", None, None), ("count", "9" * 5000, "9" * 5000),
])
def test_an_answer_is_shaped_by_its_type_and_nothing_else(answer_type, came, row):
    shaped = harness.shape(answer_type, came)
    assert shaped == row and type(shaped) is type(row)


def test_rows_write_the_nine_fields_for_every_key_and_null_for_one_with_no_response(toy,
                                                                                    tmp_path):
    run_dir = tmp_path / "run"
    assert make_requests(toy, run_dir, "--items", str(toy / "toy-data" / "pilot.json")) == 0
    keys = lines(run_dir / "keys.jsonl")
    answered = {k["key"]: ("7" if n % 2 else 7) for n, k in enumerate(keys[:-2])}
    answered[keys[0]["key"]] = None  # a refusal's null answer
    respond(run_dir, answered)
    out = tmp_path / "answers.jsonl"
    assert make_rows(run_dir, out) == 1  # written, with two keys that have no response
    rows = lines(out)
    assert len(rows) == len(keys)
    for row, key in zip(rows, keys, strict=True):
        assert list(row) == NINE
        assert (row["id"], row["system"], row["arm"], row["run"]) == (
            key["id"], "closed-book", key["arm"], key["run"])
    # Every pilot item is an F1 or CF status item: 7 and "7" both become "7".
    assert {row["answer"] for row in rows[1:-2]} == {"7"}
    assert rows[0]["answer"] is None and rows[0]["tokens_in"] == 100
    assert (rows[1]["tokens_in"], rows[1]["tokens_out"], rows[1]["wall_s"],
            rows[1]["tool_calls"]) == (101, 5, 1.5, 0)
    for row in rows[-2:]:
        assert {name: row[name] for name in NINE[4:]} == dict.fromkeys(NINE[4:])


def test_the_rows_score_as_an_outside_entry_on_the_toy_data(toy, questions, tmp_path):
    run_dir = tmp_path / "run"
    assert make_requests(toy, run_dir) == 0
    # One answer of each item's type, as a model might write it: a string for every type.
    by_type = {"status": "Draft", "set": "[]", "count": "0"}
    respond(run_dir, {k["key"]: by_type[k["answer_type"]] for k in lines(run_dir / "keys.jsonl")})
    out = tmp_path / "answers.jsonl"
    assert make_rows(run_dir, out, system="closed-book-toy") == 0
    # The toy world has items of all three types (F1 and CF status, F4 set, F5 count).
    shaped = {"status": "Draft", "set": [], "count": 0}
    keys = lines(run_dir / "keys.jsonl")
    assert {k["answer_type"] for k in keys} == set(shaped)
    for row, key in zip(lines(out), keys, strict=True):
        expected = shaped[key["answer_type"]]
        assert row["answer"] == expected and type(row["answer"]) is type(expected)
        assert row["system"] == "closed-book-toy"
    entries = tmp_path / "entries.json"
    entries.write_text(json.dumps({"closed-book-toy": {}}), encoding="utf-8")
    results = tmp_path / "results"
    assert cli.main(["score", "--data", str(toy / "toy-data"), "--answers", str(out),
                     "--entries", str(entries), "--out", str(results), "--answering-model",
                     "toy-model", "--answering-settings", "toy settings",
                     "--construction-model", "none", "--embedding-model", "none"]) == 0
    report = json.loads((results / "results.json").read_text(encoding="utf-8"))
    total = sum(len(q["arms"]) for q in questions) * 3
    assert report["scope"]["outside_entries"] == ["closed-book-toy"]
    assert (report["answers"]["rows"], report["answers"]["scored"]) == (total, total)
    for listed in ("duplicate_count", "odd_run_count", "unasked_count", "out_of_scope_count"):
        assert report["answers"][listed] == 0, listed
    assert report["answers"]["missing_runs"] == [] and report["unknown_ids"] == []
    assert report["accounting"]["closed-book-toy"]["without"] == 0
    assert all(cell["answered"] == cell["asked"] for cell in report["coverage"])


def test_rows_refuses_responses_it_cannot_place(toy, tmp_path):
    run_dir = tmp_path / "run"
    assert make_requests(toy, run_dir, "--items", str(toy / "toy-data" / "pilot.json")) == 0
    keys = [k["key"] for k in lines(run_dir / "keys.jsonl")]
    out = tmp_path / "answers.jsonl"
    # The reader has not run: no run.json in the output directory.
    assert make_rows(run_dir, out) == 2 and not out.exists()
    # Responses to another requests file.
    respond(run_dir, dict.fromkeys(keys, "Draft"), requests_sha256="0" * 64)
    assert make_rows(run_dir, out) == 2 and not out.exists()
    # A response to a key that the run did not ask, and two responses to one key.
    respond(run_dir, {**dict.fromkeys(keys, "Draft"), str(uuid.uuid4()): "Draft"})
    assert make_rows(run_dir, out) == 2 and not out.exists()
    respond(run_dir, dict.fromkeys(keys, "Draft"))
    with open(run_dir / "output" / "responses.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"key": keys[0], "answer": "Final", "tokens_in": 1,
                                 "tokens_out": 1, "wall_s": 1.0, "tool_calls": 0, "error": None,
                                 "thinking_blocks": 0}) + "\n")
    assert make_rows(run_dir, out) == 2 and not out.exists()
    # Another registered system's name, an edited keys file, and an answers file already there.
    respond(run_dir, dict.fromkeys(keys, "Draft"))
    assert make_rows(run_dir, out, system="long-context") == 2 and not out.exists()
    keys_file = run_dir / "keys.jsonl"
    original = keys_file.read_bytes()
    keys_file.write_bytes(original.replace(b'"run": 1', b'"run": 3', 1))
    assert make_rows(run_dir, out) == 2 and not out.exists()
    keys_file.write_bytes(original)
    assert make_rows(run_dir, out) == 0
    assert make_rows(run_dir, out) == 2


def test_an_interrupted_requests_leaves_no_requests_file_for_a_container_to_answer(toy, tmp_path,
                                                                                    monkeypatch):
    from reader import answer_requests

    def interrupted(*args, **kwargs):
        raise OSError("the disk filled")

    real = uuid.uuid4
    made = []

    def uuid4_until_the_sixth():
        made.append(1)
        if len(made) > 5:
            raise OSError("the disk filled")
        return real()

    # Interrupted while it writes the lines, and then while it writes the plan, which comes
    # before either file has its name.
    monkeypatch.setattr(harness.uuid, "uuid4", uuid4_until_the_sixth)
    assert make_requests(toy, tmp_path / "lines") == 4
    monkeypatch.setattr(harness.uuid, "uuid4", real)
    monkeypatch.setattr(harness, "write_json", interrupted)
    assert make_requests(toy, tmp_path / "plan") == 4
    for run_dir in (tmp_path / "lines", tmp_path / "plan"):
        assert not (run_dir / "input" / "requests.jsonl").exists()
        assert not (run_dir / "keys.jsonl").exists() and not (run_dir / "plan.json").exists()
        # The reader finds no requests file there, and the harness refuses the directory again.
        with pytest.raises(answer_requests.Refused, match="is not a file"):
            answer_requests.scan_requests(run_dir / "input" / "requests.jsonl")
    monkeypatch.undo()
    assert make_requests(toy, tmp_path / "lines") == 2


def test_an_input_that_cannot_be_read_is_refused_with_exit_status_2(toy, tmp_path, capsys):
    run_dir = tmp_path / "run"
    assert make_requests(toy, run_dir, "--items", str(toy / "toy-data" / "pilot.json")) == 0
    keys = [k["key"] for k in lines(run_dir / "keys.jsonl")]
    respond(run_dir, dict.fromkeys(keys, "Draft"))
    out = tmp_path / "answers.jsonl"
    # The keys file gone, and an answers file in a directory that does not exist.
    kept = (run_dir / "keys.jsonl").read_bytes()
    (run_dir / "keys.jsonl").unlink()
    assert make_rows(run_dir, out) == 2 and not out.exists()
    assert "keys.jsonl cannot be read (No such file or directory)" in capsys.readouterr().err
    (run_dir / "keys.jsonl").write_bytes(kept)
    assert make_rows(run_dir, tmp_path / "missing" / "answers.jsonl") == 2
    # Files that the user cannot read: the responses, and the questions.
    if os.geteuid() != 0:  # root reads a file whatever its mode
        (run_dir / "output" / "responses.jsonl").chmod(0)
        assert make_rows(run_dir, out) == 2 and not out.exists()
        assert "responses.jsonl cannot be read (Permission denied)" in capsys.readouterr().err
        data = tmp_path / "unreadable-data"
        shutil.copytree(toy / "toy-data", data)
        (data / "questions.jsonl").chmod(0)
        assert harness.main(["requests", "--system", "closed-book", "--data", str(data),
                             "--out-dir", str(tmp_path / "never")]) == 2
        assert not (tmp_path / "never").exists()


def test_an_error_that_no_rule_expects_exits_4_and_never_1(toy, tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run"
    assert make_requests(toy, run_dir, "--items", str(toy / "toy-data" / "pilot.json")) == 0
    respond(run_dir, {k["key"]: "Draft" for k in lines(run_dir / "keys.jsonl")})

    def broken(answer_type, value):
        raise KeyError("a bug")

    monkeypatch.setattr(harness, "shape", broken)
    out = tmp_path / "answers.jsonl"
    assert make_rows(run_dir, out) == 4 and not out.exists()
    assert "crashed (exit 4)" in capsys.readouterr().err
