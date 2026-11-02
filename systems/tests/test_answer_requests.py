"""The reader's answering run (reader/answer_requests.py), with stand-ins for the answering call and
the client, so that it needs no SDK and makes no model call: the requests it accepts, the responses
it writes and their keys, how many calls are under way at once, the stop on an infrastructure
error and the resume that answers what the stop left, the stop after api_error responses in a row
and how a resume counts them, its run.json, its lock and its refusals. test_answer.py runs it once
through the real client. Standard library only."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import uuid

import pytest
from asofbench_systems import answer, usage
from asofbench_systems import run as plumbing
from reader import answer_requests as reader

MODEL = "claude-haiku-5-5"
IMAGES = {"reader": "sha256:" + "1" * 64, "egress": "sha256:" + "2" * 64}
INVOCATION_KEYS = ["command", "started_at", "resume", "images", "elapsed_s", "exit", "stop",
                   "crash"]


def failure(module: str, name: str) -> type[Exception]:
    """An exception class that stands in for one of a package's, by its module and name."""
    return type(name, (Exception,), {"__module__": module})


Unreachable = failure("anthropic._exceptions", "APIConnectionError")


class Client:
    """Stands in for the answering client, which the run closes at its end."""

    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class Asker:
    """Stands in for answer_once: answers each question after ``delay`` seconds (or
    ``delays[question]``), records each call with the usage log's attribution at the time, counts
    the calls under way, and raises ``raises[question]`` or gives ``errors[question]`` as the error
    code."""

    def __init__(self, *, raises=None, errors=None, delay: float = 0.0, delays=None) -> None:
        self.raises, self.errors, self.delay = raises or {}, errors or {}, delay
        self.delays = delays or {}
        self.calls: list[dict] = []
        self.under_way = self.most = 0

    async def __call__(self, client, model, question, context):
        self.calls.append({"model": model, "question": question, "context": context,
                           "attributed": usage.attribution()})
        self.under_way += 1
        self.most = max(self.most, self.under_way)
        try:
            await asyncio.sleep(self.delays.get(question, self.delay))
            if question in self.raises:
                raise self.raises[question]
            error = self.errors.get(question)
            return {"answer": None if error else f"answer to {question}", "tokens_in": 10,
                    "tokens_out": 2, "wall_s": 0.25, "tool_calls": 0, "error": error,
                    "thinking_blocks": 0 if error else 1}
        finally:
            self.under_way -= 1


@pytest.fixture
def layout(tmp_path):
    """A run's input and output directories, as the harness makes them, and a made-up key file."""
    (tmp_path / "input").mkdir()
    (tmp_path / "output").mkdir()
    (tmp_path / "key").write_text("sk-ant-api00-made-up-key\n", encoding="ascii")
    return tmp_path


def write_requests(layout, count: int, *, contexts: tuple[int, ...] = ()) -> list[dict]:
    requests = [{"key": str(uuid.uuid4()), "question": f"question {n}"} for n in range(count)]
    for n in contexts:
        requests[n]["context"] = f"context {n}"
    write_lines(layout / "input" / "requests.jsonl", requests)
    return requests


def write_lines(path, lines) -> None:
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")


def go(layout, asker, *, resume: bool = False, preflight=None, clients=None, model: str = MODEL,
       concurrency: int = reader.CONCURRENCY, progress=None, **counted) -> int:
    """One invocation on ``layout``, with ``asker`` for the answering call (and, in ``counted``,
    the run's ``api_errors_before``)."""
    opened = [] if clients is None else clients

    def open_client(log, key_file):
        opened.append(Client())
        return opened[-1]

    return asyncio.run(reader.run(
        layout / "input" / "requests.jsonl", layout / "output" / "responses.jsonl", model,
        resume=resume, images=IMAGES, open_client=open_client, ask=asker, preflight=preflight,
        key_file=str(layout / "key"), concurrency=concurrency,
        progress=progress or (lambda line: None), **counted))


def responses_of(layout) -> list[dict]:
    path = layout / "output" / "responses.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def record_of(layout) -> dict:
    return json.loads((layout / "output" / "run.json").read_text(encoding="utf-8"))


def test_every_request_gets_one_response_under_its_own_key(layout):
    requests = write_requests(layout, 5, contexts=(2,))
    asker, clients = Asker(), []
    assert go(layout, asker, clients=clients) == plumbing.EXIT_DONE
    responses = responses_of(layout)
    assert sorted(r["key"] for r in responses) == sorted(r["key"] for r in requests)
    by_key = {r["key"]: r for r in requests}
    for response in responses:
        assert list(response) == list(answer.RESPONSE_FIELDS)
        assert response["answer"] == f"answer to {by_key[response['key']]['question']}"
    # Each request's question and context, with the run's model, and each call attributed in the
    # usage log to the request's key, the run's first invocation and attempt 1.
    assert sorted((c["question"], c["context"], c["model"]) for c in asker.calls) == sorted(
        (r["question"], r.get("context"), MODEL) for r in requests)
    assert sorted(tuple(c["attributed"].values()) for c in asker.calls) == sorted(
        (r["key"], 1, 1) for r in requests)
    assert [client.closed for client in clients] == [True]
    record = record_of(layout)
    sha = hashlib.sha256((layout / "input" / "requests.jsonl").read_bytes()).hexdigest()
    assert {key: record[key] for key in ("requests", "requests_sha256", "responses", "model")} == {
        "requests": "requests.jsonl", "requests_sha256": sha, "responses": "responses.jsonl",
        "model": MODEL}
    (invocation,) = record["invocations"]
    assert list(invocation) == INVOCATION_KEYS
    assert (invocation["command"], invocation["resume"], invocation["images"],
            invocation["exit"], invocation["stop"], invocation["crash"]) == (
        "answer", False, IMAGES, 0, None, None)
    # Nothing in the output directory holds a local path.
    for path in (layout / "output").iterdir():
        assert str(layout) not in path.read_text(encoding="utf-8"), path.name


def test_at_most_eight_calls_are_under_way_at_once(layout):
    write_requests(layout, 20)
    asker = Asker(delay=0.02)
    assert go(layout, asker) == plumbing.EXIT_DONE
    assert reader.CONCURRENCY == 8 and asker.most == 8 and len(responses_of(layout)) == 20


def test_the_requests_are_read_one_at_a_time_in_the_file_s_order(layout):
    requests = write_requests(layout, 12)
    asker = Asker()
    assert go(layout, asker, concurrency=1) == plumbing.EXIT_DONE
    assert [c["question"] for c in asker.calls] == [r["question"] for r in requests]
    assert [r["key"] for r in responses_of(layout)] == [r["key"] for r in requests]


def test_an_infrastructure_error_stops_the_run_and_the_resume_answers_its_request(layout):
    requests = write_requests(layout, 10)
    down = Asker(raises={"question 3": Unreachable("connection error")})
    assert go(layout, down, concurrency=1) == plumbing.EXIT_STOPPED
    # No response for the request whose call raised, none for those after it, which were not
    # asked, and the earlier ones kept.
    assert [r["key"] for r in responses_of(layout)] == [r["key"] for r in requests[:3]]
    assert [c["question"] for c in down.calls] == [f"question {n}" for n in range(4)]
    (invocation,) = record_of(layout)["invocations"]
    assert (invocation["exit"], invocation["stop"]) == (
        2, {"key": requests[3]["key"], "error": "anthropic APIConnectionError"})
    up = Asker()
    assert go(layout, up, resume=True) == plumbing.EXIT_DONE
    assert sorted(c["question"] for c in up.calls) == sorted(f"question {n}" for n in range(3, 10))
    assert {(c["attributed"]["invocation"], c["attributed"]["attempt"])
            for c in up.calls} == {(2, 1)}
    assert sorted(r["key"] for r in responses_of(layout)) == sorted(r["key"] for r in requests)
    first, second = record_of(layout)["invocations"]
    assert (first["exit"], second["resume"], second["exit"], second["stop"]) == (2, True, 0, None)


def test_the_requests_under_way_at_a_stop_finish_and_no_other_starts(layout):
    requests = write_requests(layout, 12)
    # The first call raises once the next three are under way, which take longer.
    asker = Asker(raises={"question 0": Unreachable("connection error")}, delay=0.1,
                  delays={"question 0": 0.01})
    assert go(layout, asker, concurrency=4) == plumbing.EXIT_STOPPED
    assert sorted(c["question"] for c in asker.calls) == [f"question {n}" for n in range(4)]
    assert sorted(r["key"] for r in responses_of(layout)) == sorted(r["key"]
                                                                    for r in requests[1:4])


def test_a_request_that_was_not_answered_keeps_its_code_and_the_run_exits_1(layout):
    requests = write_requests(layout, 4)
    asker = Asker(errors={"question 1": "refusal", "question 2": "api_error"})
    assert go(layout, asker) == plumbing.EXIT_FAILED
    by_key = {r["key"]: r for r in responses_of(layout)}
    assert [(by_key[r["key"]]["answer"], by_key[r["key"]]["error"]) for r in requests] == [
        ("answer to question 0", None), (None, "refusal"), (None, "api_error"),
        ("answer to question 3", None)]
    # A resume has nothing left to answer, and the run's status stays 1.
    again = Asker()
    assert go(layout, again, resume=True) == plumbing.EXIT_FAILED and again.calls == []


def api_errors(*numbers: int) -> dict[str, str]:
    return {f"question {n}": "api_error" for n in numbers}


def test_five_api_errors_in_a_row_stop_the_run_with_exit_status_3(layout):
    requests = write_requests(layout, 10)
    asker = Asker(errors=api_errors(*range(10)))
    assert go(layout, asker, concurrency=1) == plumbing.EXIT_FAILURES
    # The five responses stay, each its request's own error, and nothing after them is asked.
    assert [(r["key"], r["answer"], r["error"]) for r in responses_of(layout)] == [
        (r["key"], None, "api_error") for r in requests[:5]]
    assert len(asker.calls) == 5 == reader.MAX_CONSECUTIVE_API_ERRORS
    (invocation,) = record_of(layout)["invocations"]
    assert (invocation["exit"], invocation["stop"]) == (
        3, {"error": "5 consecutive api_error responses"})


def test_four_api_errors_and_then_an_answer_or_a_refusal_do_not_stop_the_run(layout):
    write_requests(layout, 15)
    # Four in a row, an answer; four more, a refusal; four more, an answer.
    asker = Asker(errors={**api_errors(0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13),
                          "question 9": "refusal"})
    assert go(layout, asker, concurrency=1) == plumbing.EXIT_FAILED
    assert len(responses_of(layout)) == 15 and len(asker.calls) == 15
    (invocation,) = record_of(layout)["invocations"]
    assert (invocation["exit"], invocation["stop"]) == (1, None)


def test_after_five_api_errors_in_a_row_the_calls_under_way_finish_and_no_other_starts(layout):
    write_requests(layout, 40)
    asker = Asker(errors=api_errors(*range(40)), delay=0.01)
    assert go(layout, asker) == plumbing.EXIT_FAILURES
    responses = responses_of(layout)
    # Every call that began has its response, and the run stopped long before the file's end.
    assert len(asker.calls) == len(responses) < 40
    assert len(responses) >= reader.MAX_CONSECUTIVE_API_ERRORS


def test_a_resume_counts_on_from_the_api_errors_that_the_responses_end_with(layout):
    write_requests(layout, 12)
    # Three answers, then four api_error responses and an infrastructure stop.
    first = Asker(errors=api_errors(3, 4, 5, 6), raises={"question 7": Unreachable("down")})
    assert go(layout, first, concurrency=1) == plumbing.EXIT_STOPPED
    # The resumed run's first api_error response is the fifth in a row.
    second = Asker(errors=api_errors(7, 8, 9, 10, 11))
    assert go(layout, second, resume=True, concurrency=1) == plumbing.EXIT_FAILURES
    assert [c["question"] for c in second.calls] == ["question 7"]
    # And an answer first resets the count.
    third = Asker(errors=api_errors(9, 10, 11))
    assert go(layout, third, resume=True, concurrency=1) == plumbing.EXIT_FAILED
    assert [c["question"] for c in third.calls] == [f"question {n}" for n in range(8, 12)]
    assert reader.trailing_api_errors(responses_of(layout)) == 3


def test_a_run_counts_on_from_the_api_errors_that_the_runs_before_it_ended_with(layout):
    # Grep at D's dates are runs of their own, one after another: the count of api_error
    # responses in a row goes on from the dates before (api_errors_before, which run_dates.py
    # gives), so that a cause that every request shares stops the run wherever it falls.
    requests = write_requests(layout, 4)
    asker = Asker(errors=api_errors(0, 1, 2, 3))
    assert go(layout, asker, concurrency=1, api_errors_before=3) == plumbing.EXIT_FAILURES
    # The third run's fifth in a row is this one's second response.
    assert [r["key"] for r in responses_of(layout)] == [r["key"] for r in requests[:2]]
    (invocation,) = record_of(layout)["invocations"]
    assert (invocation["exit"], invocation["stop"]) == (
        3, {"error": "5 consecutive api_error responses", "api_errors_before": 3})


def test_an_answer_resets_the_count_carried_from_the_runs_before(layout):
    write_requests(layout, 6)
    asker = Asker(errors=api_errors(1, 2, 3, 4))
    assert go(layout, asker, concurrency=1, api_errors_before=4) == plumbing.EXIT_FAILED
    assert len(responses_of(layout)) == 6
    # And with nothing carried, the count is the reader's own.
    (invocation,) = record_of(layout)["invocations"]
    assert invocation["stop"] is None


def test_a_resumed_run_adds_the_carried_count_only_to_responses_that_are_all_api_errors(layout):
    write_requests(layout, 6)
    # Two api_error responses and a stop; resumed with two carried from the runs before, the
    # third api_error is the fifth in a row.
    first = Asker(errors=api_errors(0, 1), raises={"question 2": Unreachable("down")})
    assert go(layout, first, concurrency=1, api_errors_before=2) == plumbing.EXIT_STOPPED
    second = Asker(errors=api_errors(2, 3, 4, 5))
    assert go(layout, second, resume=True, concurrency=1,
              api_errors_before=2) == plumbing.EXIT_FAILURES
    assert [c["question"] for c in second.calls] == ["question 2"]
    assert reader.trailing_api_errors(responses_of(layout), 2) == 5
    # An answer among the responses: the carried count no longer joins them.
    assert reader.trailing_api_errors([{"error": None}, {"error": "api_error"}], 4) == 1
    assert reader.trailing_api_errors([], 4) == 4
    with pytest.raises(plumbing.Refused, match="api_errors_before"):
        go(layout, Asker(), resume=True, api_errors_before=-1)


def test_nothing_is_answered_unless_the_network_check_passes(layout):
    write_requests(layout, 3)
    asker, clients, checked = Asker(), [], []

    def failing(out_dir):
        checked.append(out_dir)
        return 1

    assert go(layout, asker, preflight=failing, clients=clients) == plumbing.EXIT_STOPPED
    assert checked == [layout / "output"]
    assert (asker.calls, clients) == ([], [])
    assert not (layout / "output" / "responses.jsonl").exists()
    (invocation,) = record_of(layout)["invocations"]
    assert (invocation["exit"], invocation["stop"]) == (2, {"error": "network check"})
    assert go(layout, asker, resume=True, preflight=lambda out_dir: 0) == plumbing.EXIT_DONE
    assert len(responses_of(layout)) == 3


def test_a_new_run_needs_a_new_directory_and_a_resume_its_own_requests_and_model(layout):
    write_requests(layout, 2)
    with pytest.raises(plumbing.Refused, match="holds no run to resume"):
        go(layout, Asker(), resume=True)
    assert go(layout, Asker()) == plumbing.EXIT_DONE
    with pytest.raises(plumbing.Refused, match="already holds a run"):
        go(layout, Asker())
    with pytest.raises(plumbing.Refused, match="the model claude-haiku-5-5, not with"):
        go(layout, Asker(), resume=True, model="claude-sonnet-5-5")
    write_requests(layout, 2)
    with pytest.raises(plumbing.Refused, match="the run began with the requests file"):
        go(layout, Asker(), resume=True)
    with pytest.raises(plumbing.Refused, match="not an answering model"):
        go(layout, Asker(), model="claude-opus-5-5")


@pytest.mark.parametrize(("line", "why"), [
    ({"key": "KEY", "question": "q", "id": "F1-change-PEP-1-2001-01-01"},
     "holds fields that a request never holds: id"),
    ({"key": "KEY", "question": "q", "answer_type": "status", "arm": "H+"},
     "holds fields that a request never holds: answer_type, arm"),
    ({"key": "KEY"}, "has no question"),
    ({"key": "KEY", "question": ""}, "has no question"),
    ({"key": "KEY", "question": "q", "context": 7}, "has a context that is not text"),
    ({"key": "F1-change-PEP-1-2001-01-01", "question": "q"}, "has no key that is a UUID4"),
    ({"key": "4B7E1F5A-0A3B-4C55-9D1E-2F3A4B5C6D7E", "question": "q"},
     "has no key that is a UUID4"),
    ({"key": str(uuid.uuid1()), "question": "q"}, "has no key that is a UUID4"),
    (["KEY", "q"], "is not a JSON object"),
])
def test_a_line_that_is_not_a_request_is_refused_before_anything_is_answered(layout, line, why):
    good = write_requests(layout, 2)
    if isinstance(line, dict) and line.get("key") == "KEY":
        line = {**line, "key": str(uuid.uuid4())}
    write_lines(layout / "input" / "requests.jsonl", [*good, line])
    asker = Asker()
    with pytest.raises(plumbing.Refused) as caught:
        go(layout, asker)
    assert str(caught.value) == f"requests.jsonl: line 3 {why}"
    assert asker.calls == [] and list((layout / "output").iterdir()) == []


def test_a_requests_file_that_is_no_list_of_requests_is_refused(layout):
    path = layout / "input" / "requests.jsonl"
    requests = write_requests(layout, 2)
    write_lines(path, [*requests, requests[0]])
    with pytest.raises(plumbing.Refused, match="line 3's key occurs twice"):
        go(layout, Asker())
    path.write_text(json.dumps(requests[0]) + "\n{\"key\": ", encoding="utf-8")
    with pytest.raises(plumbing.Refused, match="line 2 is not JSON"):
        go(layout, Asker())
    path.write_text("", encoding="utf-8")
    with pytest.raises(plumbing.Refused, match="holds no request"):
        go(layout, Asker())
    assert list((layout / "output").iterdir()) == []


def test_a_requests_file_that_is_not_alone_in_its_directory_is_refused(layout):
    write_requests(layout, 2)
    # The harness's host-side files, beside the requests as an operator might mount them.
    (layout / "input" / "keys.jsonl").write_text("{}\n", encoding="utf-8")
    (layout / "input" / "plan.json").write_text("{}\n", encoding="utf-8")
    asker = Asker()
    with pytest.raises(plumbing.Refused, match=r"holds other entries \(keys\.jsonl, plan\.json\)"):
        go(layout, asker)
    assert asker.calls == [] and list((layout / "output").iterdir()) == []


def test_a_second_invocation_is_refused_while_one_holds_the_lock(layout):
    write_requests(layout, 2)
    (layout / "output" / "run.lock").write_text(json.dumps({"command": "answer"}) + "\n",
                                                encoding="utf-8")
    asker = Asker()
    with pytest.raises(plumbing.Refused, match=r"systems/README\.md says how to check"):
        go(layout, asker)
    assert asker.calls == [] and sorted(p.name for p in (layout / "output").iterdir()) == [
        "run.lock"]


def test_a_resume_drops_an_unfinished_last_line_and_refuses_a_line_it_cannot_place(layout):
    requests = write_requests(layout, 4)
    assert go(layout, Asker(raises={"question 2": Unreachable("down")}),
              concurrency=1) == plumbing.EXIT_STOPPED
    responses = layout / "output" / "responses.jsonl"
    kept = responses.read_bytes()
    with open(responses, "ab") as handle:
        handle.write(b'{"key": "' + requests[2]["key"].encode())
    lines = []
    assert go(layout, Asker(), resume=True, progress=lines.append) == plumbing.EXIT_DONE
    assert any("dropped the unfinished last line of responses.jsonl" in line for line in lines)
    assert len(responses_of(layout)) == 4
    # A response line that answers no request of the file, or a key a second time.
    responses.write_bytes(kept + json.dumps({**responses_of(layout)[0],
                                             "key": str(uuid.uuid4())}).encode() + b"\n")
    with pytest.raises(plumbing.Refused, match="answers a key that no request of the run holds"):
        go(layout, Asker(), resume=True)
    responses.write_bytes(kept + kept.splitlines(keepends=True)[0])
    with pytest.raises(plumbing.Refused, match="answers a key that has a response already"):
        go(layout, Asker(), resume=True)


def test_a_crash_is_recorded_with_exit_status_4(layout):
    write_requests(layout, 3)
    with pytest.raises(ExceptionGroup):
        go(layout, Asker(raises={"question 1": ValueError("a bug")}))
    (invocation,) = record_of(layout)["invocations"]
    # The workers' TaskGroup raises the group, as Graphiti's ingestion's groups do.
    assert (invocation["exit"], invocation["crash"]) == (4, "ExceptionGroup")
    assert not (layout / "output" / "run.lock").exists()


def test_the_command_line_refuses_what_it_cannot_run(layout, monkeypatch, capsys):
    write_requests(layout, 1)
    args = ["--requests", str(layout / "input" / "requests.jsonl"),
            "--out", str(layout / "output" / "responses.jsonl")]
    for argv in ([], args, [*args, "--model", "claude-opus-5-5"],
                 ["--self-test", "--model", MODEL],
                 # The carried count is a run's, of which the self-test has none, even at 0.
                 ["--self-test", "--api-errors-before", "0"],
                 ["--self-test", "--api-errors-before", "3"],
                 [*args, "--model", MODEL, "--api-errors-before", "-1"],
                 [*args, "--model", MODEL, "--api-errors-before", "two"]):
        with pytest.raises(SystemExit) as caught:
            reader.main(argv)
        assert caught.value.code == 2, argv
    monkeypatch.delenv("ASOFBENCH_IMAGE_IDS", raising=False)
    assert reader.main([*args, "--model", MODEL]) == plumbing.EXIT_STOPPED
    assert "name=ID entries for reader, egress (systems/README.md)" in capsys.readouterr().err
    # The key is read from the Docker secret only, which a test's host has not mounted.
    monkeypatch.setenv("ASOFBENCH_IMAGE_IDS", ",".join(f"{k}={v}" for k, v in IMAGES.items()))
    if not os.path.exists(reader.llm.KEY_FILE):
        assert reader.main([*args, "--model", MODEL]) == plumbing.EXIT_STOPPED
        assert "the API key file /run/secrets/anthropic_api_key cannot be read" in (
            capsys.readouterr().err)
    assert list((layout / "output").iterdir()) == []
