"""The answering call of every single-call entrant (common/asofbench_systems/answer.py): the
request it sends, field by field, and how it reads the reply; the reader's run
(reader/answer_requests.py) through the real client; and the image's self-test
(reader/selftest.py) on this SDK. They need the anthropic package, so the root suite skips them.
CI runs them with the SDK and httpx that the reader image's lock pins:

    uv run --frozen --extra dev --with anthropic==0.78.0 --with httpx==0.28.1 \\
        python -m pytest systems/tests/test_answer.py systems/tests/test_llm_client.py

and they run inside the reader image, with no network at all:

    docker run --rm --network none -v "$PWD/systems:/systems:ro" asofbench/reader \\
        python -m pytest -p no:cacheprovider /systems/tests

Every request goes to an httpx MockTransport and the key is a made-up one, so no test reaches the
API or holds a real key.
"""

from __future__ import annotations

import asyncio
import json
import uuid

import pytest

anthropic = pytest.importorskip("anthropic")
httpx = pytest.importorskip("httpx")

from asofbench_systems import answer, llm, run, usage  # noqa: E402

KEY = "sk-ant-api00-made-up-for-tests-" + "9876543210" * 4
MODEL = "claude-haiku-5-5"
# A made-up question in the form of the toy world's, which no v0.1 item asks.
QUESTION = ("What was the status of PEP 1 (\"Proposal 1\") as of the end of 2001-06-30 (UTC)? "
            "Answer with exactly one value from: Draft, Final.")
USAGE = {"input_tokens": 11, "output_tokens": 7, "cache_creation_input_tokens": 3,
         "cache_read_input_tokens": 5}
# A thinking block as Claude Haiku 5.5 returns it by default: its text omitted, its signature kept.
THINKING = {"type": "thinking", "thinking": "", "signature": "bWFkZS11cA=="}
# The registered strings and schema, written out again here, so that an edit to answer.py that
# changes what the model is sent fails here and not only in a comparison with itself.
REGISTERED_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"anyOf": [
        {"type": "string"}, {"type": "integer"},
        {"type": "array", "items": {"anyOf": [{"type": "integer"}, {"type": "string"}]}}]}},
    "required": ["answer"],
    "additionalProperties": False,
}
REGISTERED_PROMPT = ("Answer the question that follows. Reply with a JSON object whose \"answer\" "
                     "field holds your answer in the form the question asks for.")
RESULT_FIELDS = ["answer", "tokens_in", "tokens_out", "wall_s", "tool_calls", "error",
                 "thinking_blocks"]
SAMPLING = {"temperature", "top_p", "top_k"}


@pytest.fixture
def environment(monkeypatch):
    """An environment that points the SDK elsewhere and offers it other credentials, none of which
    the client may use, and no proxy."""
    for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY",
                 "all_proxy", "NO_PROXY", "no_proxy", "ANTHROPIC_CUSTOM_HEADERS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://elsewhere.example.invalid")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-the-environment-not-the-file")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "a-token-from-the-environment")
    return monkeypatch


@pytest.fixture
def key_file(tmp_path):
    path = tmp_path / "anthropic_api_key"
    path.write_text(KEY + "\n", encoding="ascii")
    return str(path)


def message(content: list[dict], stop_reason: str = "end_turn", **counts: int) -> dict:
    return {"id": "msg_01TEST", "type": "message", "role": "assistant", "model": MODEL,
            "content": content, "stop_reason": stop_reason, "stop_sequence": None,
            "usage": {**USAGE, **counts}}


def text(value: str) -> dict:
    return {"type": "text", "text": value}


def error_body(kind: str) -> dict:
    return {"type": "error", "error": {"type": kind, "message": "a toy error"}}


class Recorder:
    """A MockTransport handler: records each request and answers it with ``answers`` in turn,
    repeating the last; an answer that is an exception class is raised instead."""

    def __init__(self, *answers) -> None:
        self.answers, self.requests = list(answers), []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answer_ = self.answers[min(len(self.requests), len(self.answers)) - 1]
        if isinstance(answer_, type) and issubclass(answer_, Exception):
            raise answer_("no route", request=request)
        status, body = answer_
        return httpx.Response(status, json=body)

    def bodies(self) -> list[dict]:
        return [json.loads(request.content) for request in self.requests]


def call(key_file: str, recorder: Recorder, question: str = QUESTION,
         context: str | None = None, *, max_retries: int = 0) -> dict:
    """``answer_once`` with a client whose requests go to ``recorder``, in one event loop."""
    client = llm.make_async_client(key_file, max_retries=max_retries,
                                   transport=httpx.MockTransport(recorder))

    async def go():
        try:
            return await answer.answer_once(client, MODEL, question, context)
        finally:
            await client.close()

    return asyncio.run(go())


def test_the_request_sends_the_registered_settings_field_by_field(environment, key_file):
    recorder = Recorder((200, message([THINKING, text('{"answer": "Draft"}')])))
    call(key_file, recorder)
    (request,) = recorder.requests
    assert str(request.url) == "https://api.anthropic.com/v1/messages"
    (body,) = recorder.bodies()
    assert body["model"] == MODEL
    assert body["max_tokens"] == 32000
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "medium",
                                     "format": {"type": "json_schema",
                                                "schema": answer.ANSWER_SCHEMA}}
    assert body["output_config"]["format"]["schema"] == REGISTERED_SCHEMA
    assert body["system"] == answer.SYSTEM_PROMPT == REGISTERED_PROMPT
    assert body["messages"] == [{"role": "user", "content": answer.user_message(QUESTION, None)}]
    assert body["messages"][0]["content"] == f"Question: {QUESTION}"
    assert not (SAMPLING | {"tools", "tool_choice"}) & set(body)
    # Nothing else: no field that the SDK adds by default, no streaming, no prefill (the one
    # message is the user's), and no beta feature switched on by a header.
    assert set(body) == {"model", "max_tokens", "thinking", "output_config", "system",
                         "messages"}
    assert "anthropic-beta" not in request.headers
    assert request.headers["anthropic-version"] == "2023-06-01"
    # The key from the file and no other credential: no bearer token from the environment.
    assert request.headers["x-api-key"] == KEY
    assert "authorization" not in request.headers


def test_a_request_with_a_context_sends_it_before_the_question(environment, key_file):
    context = "The context holds two toy lines, oldest first.\n\nPEP-1: first\n---\nPEP-1: second"
    recorder = Recorder((200, message([text('{"answer": "Final"}')])))
    assert call(key_file, recorder, QUESTION, context)["answer"] == "Final"
    (body,) = recorder.bodies()
    assert body["messages"] == [{"role": "user",
                                 "content": f"Context:\n\n{context}\n\nQuestion: {QUESTION}"}]
    assert body["system"] == REGISTERED_PROMPT and body["thinking"] == {"type": "adaptive"}


def test_the_registered_strings_are_exactly_those_of_the_constraints():
    assert answer.ANSWER_SCHEMA == REGISTERED_SCHEMA
    assert answer.SYSTEM_PROMPT == REGISTERED_PROMPT
    assert answer.user_message("Q?", None) == "Question: Q?"
    assert answer.user_message("Q?", "Ctx.") == "Context:\n\nCtx.\n\nQuestion: Q?"
    assert answer.MODELS == ("claude-haiku-5-5", "claude-sonnet-5-5")
    assert answer.ERRORS == ("refusal", "max_tokens", "api_error", "unparseable")
    assert list(answer.RESULT_FIELDS) == RESULT_FIELDS


def test_the_settings_are_one_definition_and_a_new_copy_for_every_call():
    settings = answer.answering_settings("claude-sonnet-5-5")
    assert settings == {"model": "claude-sonnet-5-5", "max_tokens": 32000,
                        "thinking": {"type": "adaptive"},
                        "output_config": {"effort": "medium",
                                          "format": {"type": "json_schema",
                                                     "schema": REGISTERED_SCHEMA}},
                        "system": REGISTERED_PROMPT}
    # An agent's loop sends the same settings with its own system prompt.
    assert answer.answering_settings(MODEL, system="An agent's prompt.")["system"] == (
        "An agent's prompt.")
    # A caller that changes what it was given changes neither the schema nor a later call.
    settings["output_config"]["format"]["schema"]["required"].append("other")
    settings["thinking"]["type"] = "disabled"
    assert answer.ANSWER_SCHEMA == REGISTERED_SCHEMA
    assert answer.answering_settings(MODEL)["output_config"]["format"]["schema"] == (
        REGISTERED_SCHEMA)
    assert answer.answering_settings(MODEL)["thinking"] == {"type": "adaptive"}


def test_a_text_block_after_a_thinking_block_is_the_answer(environment, key_file):
    result = call(key_file, Recorder((200, message([THINKING, text('{"answer": "Draft"}')]))))
    assert list(result) == RESULT_FIELDS
    assert {key: result[key] for key in RESULT_FIELDS if key != "wall_s"} == {
        "answer": "Draft", "tokens_in": 11 + 3 + 5, "tokens_out": 7, "tool_calls": 0,
        "error": None, "thinking_blocks": 1}
    assert isinstance(result["wall_s"], float) and result["wall_s"] >= 0


def test_an_answer_of_each_shape_is_kept_as_it_came(environment, key_file):
    for reply, expected in (('{"answer": 7}', 7), ('{"answer": [1, "PEP 8"]}', [1, "PEP 8"]),
                            ('{"answer": "  final. "}', "  final. "), ('{"answer": "7"}', "7")):
        result = call(key_file, Recorder((200, message([text(reply)]))))
        assert (result["answer"], result["error"], result["thinking_blocks"]) == (expected, None,
                                                                                  0)


def test_a_refusal_and_a_max_tokens_stop_give_a_null_answer(environment, key_file):
    for stop_reason in ("refusal", "max_tokens"):
        # Even a reply whose text would parse: the stop says the answer is not the model's own.
        reply = message([THINKING, text('{"answer": "Draft"}')], stop_reason, output_tokens=40)
        result = call(key_file, Recorder((200, reply)))
        assert (result["answer"], result["error"]) == (None, stop_reason)
        assert (result["tokens_in"], result["tokens_out"], result["tool_calls"]) == (19, 40, 0)
    cut = call(key_file, Recorder((200, message([text('{"answer": "Dra')], "max_tokens"))))
    assert (cut["answer"], cut["error"]) == (None, "max_tokens")


def test_a_reply_that_holds_no_json_answer_is_unparseable(environment, key_file):
    for content in ([text("Draft")], [text('{"other": "Draft"}')], [text('["Draft"]')],
                    [text('{"answer": "Draft"} and more')], [THINKING], []):
        result = call(key_file, Recorder((200, message(content))))
        assert (result["answer"], result["error"]) == (None, "unparseable"), content
        assert result["tokens_in"] == 19


def test_an_api_error_that_is_the_request_s_own_gives_a_null_answer(environment, key_file):
    for status, kind in ((400, "invalid_request_error"), (404, "not_found_error"),
                         (413, "request_too_large"), (422, "invalid_request_error")):
        result = call(key_file, Recorder((status, error_body(kind))))
        assert list(result) == RESULT_FIELDS
        assert {key: result[key] for key in RESULT_FIELDS if key != "wall_s"} == {
            "answer": None, "tokens_in": 0, "tokens_out": 0, "tool_calls": 0,
            "error": "api_error", "thinking_blocks": 0}, status


def test_an_infrastructure_error_is_raised_for_the_run_to_stop_on(environment, key_file):
    # After the SDK's retries (none here): the API refusing the key, rate-limited, failing,
    # unavailable or overloaded, or unreachable. The first-party async client raises its
    # InternalServerError for every 5xx, 503, 504 and 529 included: anthropic 0.78.0's has no
    # branch for 529, and the separate 503 and 504 classes are its Bedrock and Vertex clients'.
    answers = [(401, error_body("authentication_error")), (403, error_body("permission_error")),
               (429, error_body("rate_limit_error")), (500, error_body("api_error")),
               (503, error_body("api_error")), (504, error_body("api_error")),
               (529, error_body("overloaded_error")), httpx.ConnectError]
    for reply in answers:
        with pytest.raises(anthropic.APIError) as caught:
            call(key_file, Recorder(reply))
        assert run.infrastructure_error(caught.value, run.INFRASTRUCTURE) is not None, reply
        if isinstance(reply, tuple) and reply[0] >= 500:
            assert isinstance(caught.value, anthropic.InternalServerError), reply
        assert KEY not in str(caught.value)


def test_the_sdk_s_retries_come_before_any_answer(environment, key_file):
    recorder = Recorder((529, error_body("overloaded_error")),
                        (200, message([text('{"answer": "Draft"}')])))
    result = call(key_file, recorder, max_retries=1)
    assert (result["answer"], result["error"], len(recorder.requests)) == ("Draft", None, 2)


def test_the_reader_answers_each_request_through_the_client_and_logs_its_usage(environment,
                                                                               key_file,
                                                                               tmp_path):
    from reader import answer_requests

    requests = [{"key": str(uuid.uuid4()), "question": f"{QUESTION} ({n})"} for n in range(3)]
    requests[1]["context"] = "The context holds one toy line.\n\nPEP-1: Draft"
    (tmp_path / "input").mkdir()
    (tmp_path / "output").mkdir()
    path = tmp_path / "input" / "requests.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in requests), encoding="utf-8")
    recorder = Recorder((200, message([THINKING, text('{"answer": "Draft"}')])))

    def open_client(log, key):
        return usage.instrument(llm.make_async_client(
            key, transport=httpx.MockTransport(recorder)), log)

    images = {"reader": "sha256:" + "1" * 64, "egress": "sha256:" + "2" * 64}
    out = tmp_path / "output" / "responses.jsonl"
    code = asyncio.run(answer_requests.run(path, out, MODEL, resume=False, images=images,
                                           open_client=open_client, preflight=None,
                                           key_file=key_file, progress=lambda line: None))
    assert code == run.EXIT_DONE
    responses = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert sorted(r["key"] for r in responses) == sorted(r["key"] for r in requests)
    assert all((r["answer"], r["error"], r["thinking_blocks"]) == ("Draft", None, 1)
               for r in responses)
    sent = sorted(body["messages"][0]["content"] for body in recorder.bodies())
    assert sent == sorted(answer.user_message(r["question"], r.get("context")) for r in requests)
    assert all(not SAMPLING & set(body) and body["output_config"]["effort"] == "medium"
               for body in recorder.bodies())
    lines = [json.loads(line) for line in (tmp_path / "output" / "usage.jsonl").read_text(
        encoding="utf-8").splitlines()]
    # Each call is in the usage log under its request's key, the run's first invocation and the
    # key's one attempt, with the model it asked for.
    assert sorted((line["context"], line["invocation"], line["attempt"], line["model"])
                  for line in lines) == sorted((r["key"], 1, 1, MODEL) for r in requests)


def test_the_image_s_self_test_passes_on_this_sdk_and_fails_on_a_call_that_drifts(environment,
                                                                                 monkeypatch):
    from reader import selftest

    assert selftest.check_answering() == anthropic.__version__

    async def drifting(client, model, question, context):
        # The registered settings with a sampling parameter and a beta header beside them.
        await client.messages.create(
            **answer.answering_settings(model), temperature=1.0,
            messages=[{"role": "user", "content": answer.user_message(question, context)}],
            extra_headers={"anthropic-beta": "made-up-beta-2099-01-01"})
        return {"answer": "self-test", "tokens_in": 1, "tokens_out": 1, "wall_s": 0.0,
                "tool_calls": 0, "error": None, "thinking_blocks": 1}

    monkeypatch.setattr(answer, "answer_once", drifting)
    with pytest.raises(RuntimeError) as caught:
        selftest.check_answering()
    assert str(caught.value) == ("the call differs from the registered settings in temperature; "
                                 "the call sends an anthropic-beta header")


def test_the_self_test_passes_on_a_lock_of_what_is_installed_and_fails_on_another(environment,
                                                                                  tmp_path,
                                                                                  capsys):
    import importlib.metadata

    from reader import selftest

    versions = {name: importlib.metadata.version(name) for name in ("anthropic", "httpx")}
    good = tmp_path / "requirements.txt"
    good.write_text("".join(f"{name}=={version} \\\n    --hash=sha256:{'0' * 64}\n"
                            for name, version in versions.items()), encoding="utf-8")
    assert selftest.self_test(good) == 0
    assert capsys.readouterr().out.startswith(
        f"reader self-test: 2 packages as requirements.txt pins them, anthropic "
        f"{versions['anthropic']}; an answering call sends the registered settings")
    good.write_text("anthropic==0.0.1 \\\n    --hash=sha256:" + "0" * 64 + "\n", encoding="utf-8")
    assert selftest.self_test(good) == 1
    assert capsys.readouterr().err == (f"reader self-test: FAIL: anthropic is "
                                       f"{versions['anthropic']}, requirements.txt pins 0.0.1\n")
