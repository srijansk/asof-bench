"""The Anthropic client that every adapter builds (common/asofbench_systems/llm.py), the log of
each call's usage (common/asofbench_systems/usage.py), and Graphiti's AnthropicClient as the
Graphiti adapter builds it. They need the anthropic package, and the last ones graphiti-core, so
the root suite skips them. CI runs them with the SDK and httpx that the Graphiti image's lock pins,
which skips the ones that need graphiti-core:

    uv run --frozen --extra dev --with anthropic==0.111.0 --with httpx==0.28.1 \\
        python -m pytest systems/tests/test_llm_client.py

and all of them run inside the Graphiti image, with no network at all:

    docker run --rm --network none -v "$PWD/systems:/systems:ro" asofbench/graphiti \\
        python -m pytest -p no:cacheprovider /systems/tests/test_llm_client.py

Every request goes to an httpx MockTransport or to a stand-in proxy on 127.0.0.1, and the key is
a made-up one, so no test reaches the API or holds a real key.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import socketserver
import threading
import traceback

import pytest

anthropic = pytest.importorskip("anthropic")
httpx = pytest.importorskip("httpx")

from asofbench_systems import llm, usage  # noqa: E402

KEY = "sk-ant-api00-made-up-for-tests-" + "0123456789" * 4
MODEL = "claude-haiku-5-5"
# A made-up dated snapshot id, as a response may name for the model alias a call asked for.
SNAPSHOT = "claude-haiku-5-5-20990101"
USAGE = {"input_tokens": 11, "output_tokens": 7, "cache_creation_input_tokens": 3,
         "cache_read_input_tokens": 5}
ENVIRONMENT = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY",
               "all_proxy", "NO_PROXY", "no_proxy", "ANTHROPIC_CUSTOM_HEADERS")


@pytest.fixture
def environment(monkeypatch):
    """An environment that points the SDK elsewhere, which the client must not follow, and no
    proxy unless a test sets one."""
    for name in ENVIRONMENT:
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


def message(content: list[dict], stop_reason: str, **counts: int) -> dict:
    return {"id": "msg_01TEST", "type": "message", "role": "assistant", "model": MODEL,
            "content": content, "stop_reason": stop_reason, "stop_sequence": None,
            "usage": {**USAGE, **counts}}


TEXT = message([{"type": "text", "text": "ok"}], "end_turn")


class Recorder:
    """A MockTransport handler: records each request and answers it with ``answers`` in turn,
    repeating the last."""

    def __init__(self, *answers: tuple[int, dict]) -> None:
        self.answers, self.requests = list(answers), []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        status, body = self.answers[min(len(self.requests), len(self.answers)) - 1]
        return httpx.Response(status, json=body)

    def bodies(self) -> list[dict]:
        return [json.loads(request.content) for request in self.requests]


def run(client, make_calls):
    """Runs ``make_calls(client)`` and closes the client, in one event loop."""
    async def go():
        try:
            return await make_calls(client)
        finally:
            await client.close()

    return asyncio.run(go())


def ask(client, **extra):
    return client.messages.create(model=MODEL, max_tokens=16,
                                  messages=[{"role": "user", "content": "toy question"}], **extra)


def test_a_call_goes_to_the_api_with_the_key_from_the_file(environment, key_file):
    recorder = Recorder((200, TEXT))
    client = llm.make_async_client(key_file, transport=httpx.MockTransport(recorder))
    response = run(client, ask)
    assert response.content[0].text == "ok"
    (request,) = recorder.requests
    assert str(request.url) == "https://api.anthropic.com/v1/messages"
    assert request.headers["x-api-key"] == KEY
    assert "authorization" not in request.headers


def test_the_client_has_the_given_retries_and_timeout(environment, key_file):
    client = llm.make_async_client(key_file)
    assert (client.max_retries, client.timeout, str(client.base_url)) == (
        8, 600.0, "https://api.anthropic.com")
    other = llm.make_async_client(key_file, max_retries=2, timeout_s=30.0)
    assert (other.max_retries, other.timeout) == (2, 30.0)


def test_the_omit_sentinel_is_falsy_and_is_the_sdk_s_own():
    sentinel = llm.omit_sentinel()
    assert sentinel is getattr(anthropic, "omit", anthropic.NOT_GIVEN)
    assert not sentinel


class StubProxy(socketserver.ThreadingTCPServer):
    """Stands in for the egress proxy: records each request head it receives and refuses it."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self) -> None:
        self.heads: list[bytes] = []
        super().__init__(("127.0.0.1", 0), self.Handler)

    class Handler(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            head = b""
            while b"\r\n\r\n" not in head:
                chunk = self.request.recv(4096)
                if not chunk:
                    break
                head += chunk
            self.server.heads.append(head)
            self.request.sendall(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n"
                                 b"Connection: close\r\n\r\n")


@pytest.fixture
def stub_proxy():
    server = StubProxy()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def test_the_client_tunnels_through_the_proxy_that_https_proxy_names(environment, key_file,
                                                                       stub_proxy):
    environment.setenv("HTTPS_PROXY", f"http://127.0.0.1:{stub_proxy.server_address[1]}")
    client = llm.make_async_client(key_file, max_retries=0)
    with pytest.raises(anthropic.APIConnectionError) as caught:
        run(client, ask)
    (head,) = stub_proxy.heads
    assert head.startswith(b"CONNECT api.anthropic.com:443 HTTP/1.1\r\n")
    assert KEY.encode() not in head
    assert KEY not in str(caught.value) and KEY not in repr(caught.value)


def test_the_client_bypasses_the_proxy_for_a_host_that_no_proxy_names(environment, key_file,
                                                                        stub_proxy):
    environment.setenv("HTTPS_PROXY", f"http://127.0.0.1:{stub_proxy.server_address[1]}")
    environment.setenv("NO_PROXY", "api.anthropic.com")
    real = socket.getaddrinfo
    looked_up = []

    def getaddrinfo(host, *args, **kwargs):
        # The direct route is never taken: the name does not resolve here, so that the test
        # reaches no host even where there is a network. The name may come as IDNA bytes.
        name = host.decode("ascii") if isinstance(host, bytes) else host
        if name == "api.anthropic.com":
            looked_up.append(name)
            raise socket.gaierror(socket.EAI_NONAME, "not resolved in tests")
        return real(host, *args, **kwargs)

    environment.setattr(socket, "getaddrinfo", getaddrinfo)
    client = llm.make_async_client(key_file, max_retries=0)
    with pytest.raises(anthropic.APIConnectionError):
        run(client, ask)
    assert stub_proxy.heads == [] and looked_up == ["api.anthropic.com"]


def test_graphiti_s_anthropic_client_sends_no_sampling_parameter(environment, key_file):
    pytest.importorskip("graphiti_core")
    from graphiti import adapter
    from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS
    from graphiti_core.prompts.models import Message
    from pydantic import BaseModel

    class Answer(BaseModel):
        value: str

    answer = message([{"type": "tool_use", "id": "toolu_01", "name": "Answer",
                       "input": {"value": "toy"}}], "tool_use")
    recorder = Recorder((200, answer))
    client = llm.make_async_client(key_file, transport=httpx.MockTransport(recorder))
    graphiti_llm = adapter.make_llm_client(client)
    result = run(client, lambda _: graphiti_llm.generate_response(
        [Message(role="system", content="Extract."), Message(role="user", content="Toy.")],
        response_model=Answer))
    assert result == {"value": "toy"}
    (body,) = recorder.bodies()
    assert not {"temperature", "top_p", "top_k"} & set(body)
    assert body["model"] == MODEL
    # Graphiti's own code path: its forced tool call and its own max_tokens.
    assert body["tool_choice"] == {"type": "tool", "name": "Answer"}
    assert body["max_tokens"] == DEFAULT_MAX_TOKENS
    assert "thinking" not in body and "output_config" not in body


def test_graphiti_s_embedder_asks_for_the_model_s_prompt_and_keeps_2048_floats(environment):
    pytest.importorskip("graphiti_core")
    import base64
    import struct

    import openai
    from graphiti import adapter

    requests = []

    def embeddings(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = json.loads(request.content)
        texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
        data = []
        for index in range(len(texts)):
            values = [(index + 1) / 4096] * 2048
            embedding = (base64.b64encode(struct.pack("<2048f", *values)).decode("ascii")
                         if body.get("encoding_format") == "base64" else values)
            data.append({"object": "embedding", "index": index, "embedding": embedding})
        return httpx.Response(200, json={"object": "list", "data": data, "model": body["model"],
                                         "usage": {"prompt_tokens": 1, "total_tokens": 1}})

    def client():
        return openai.AsyncOpenAI(api_key="none", base_url=adapter.EMBEDDER_URL,
                                  http_client=httpx.AsyncClient(
                                      transport=httpx.MockTransport(embeddings)))

    document = adapter.make_embedder(client=client())
    query = adapter.make_embedder(adapter.QUERY_MODEL, client=client())

    async def calls():
        return (await document.create(input_data=["a toy name"]),
                await document.create_batch(["a toy fact", "another toy fact"]),
                await query.create(input_data=["a toy question"]))

    one, batch, asked = asyncio.run(calls())
    assert len(one) == 2048 and [len(vector) for vector in batch] == [2048, 2048]
    assert len(asked) == 2048 and batch[1][0] == pytest.approx(2 / 4096)
    bodies = [json.loads(request.content) for request in requests]
    assert [body["model"] for body in bodies] == ["voyage-4-nano-document"] * 2 + [
        "voyage-4-nano-query"]
    assert all(body.get("dimensions", 2048) == 2048 for body in bodies)
    assert {str(request.url) for request in requests} == {"http://embedder:8080/v1/embeddings"}


def test_the_usage_log_records_each_response_under_the_current_episode(environment, key_file,
                                                                       tmp_path):
    path = tmp_path / "usage.jsonl"
    # The response names a dated snapshot of the model asked for, as an alias's answer can.
    recorder = Recorder((200, {**TEXT, "model": SNAPSHOT}))
    client = usage.instrument(llm.make_async_client(
        key_file, transport=httpx.MockTransport(recorder)), usage.UsageLog(str(path)))

    async def calls(client):
        with usage.attributed("0123456789ab:PEP-1", 2, 3):
            await ask(client)
            # Tasks started while an episode is ingested make their calls under its id, in the
            # same invocation and attempt.
            await asyncio.gather(ask(client), ask(client))
        token = usage.usage_context.set("ba9876543210:EIP-2")
        try:
            await ask(client)
        finally:
            usage.usage_context.reset(token)
        await ask(client)

    run(client, calls)
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [(line["context"], line["invocation"], line["attempt"]) for line in lines] == [
        ("0123456789ab:PEP-1", 2, 3)] * 3 + [("ba9876543210:EIP-2", None, None),
                                             (None, None, None)]
    assert set(lines[0]) == {"ts", "context", "invocation", "attempt", "model", "response_model",
                             "input_tokens", "output_tokens", "cache_creation_input_tokens",
                             "cache_read_input_tokens", "stop_reason", "error"}
    assert {key: lines[0][key] for key in USAGE} == USAGE
    # Every line records the model the call asked for, and beside it the one the response named.
    assert all((line["model"], line["response_model"]) == (MODEL, SNAPSHOT) for line in lines)
    assert (lines[0]["stop_reason"], lines[0]["error"]) == ("end_turn", None)
    assert len(recorder.requests) == 5


def test_a_call_that_raises_is_recorded_with_its_class_only(environment, key_file, tmp_path):
    pytest.importorskip("graphiti_core")
    from graphiti import adapter
    from graphiti_core.prompts.models import Message
    from pydantic import BaseModel

    class Answer(BaseModel):
        value: str

    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    path = tmp_path / "usage.jsonl"
    client = usage.instrument(llm.make_async_client(
        key_file, max_retries=0, transport=httpx.MockTransport(unreachable)),
        usage.UsageLog(str(path)))
    with pytest.raises(anthropic.APIConnectionError):
        run(client, ask)
    # Graphiti catches a 400 and asks again: the failed call stays visible beside the answer.
    bad = {"type": "error", "error": {"type": "invalid_request_error",
                                      "message": f"a toy refusal quoting {KEY}"}}
    valid = message([{"type": "tool_use", "id": "toolu_01", "name": "Answer",
                      "input": {"value": "toy"}}], "tool_use")
    client = usage.instrument(llm.make_async_client(
        key_file, max_retries=0, transport=httpx.MockTransport(Recorder((400, bad),
                                                                        (200, valid)))),
        usage.UsageLog(str(path)))
    graphiti_llm = adapter.make_llm_client(client)

    async def calls(_):
        with usage.attributed("0123456789ab:PEP-1", 1, 1):
            return await graphiti_llm.generate_response(
                [Message(role="system", content="Extract."), Message(role="user", content="T.")],
                response_model=Answer)

    assert run(client, calls) == {"value": "toy"}
    text = path.read_text(encoding="utf-8")
    lines = [json.loads(line) for line in text.splitlines()]
    # The failure Graphiti caught is in the same attempt as the answer it carried on with.
    assert [(line["context"], line["invocation"], line["attempt"], line["error"])
            for line in lines] == [(None, None, None, "APIConnectionError"),
                                   ("0123456789ab:PEP-1", 1, 1, "BadRequestError"),
                                   ("0123456789ab:PEP-1", 1, 1, None)]
    assert all(line[count] == 0 for line in lines[:2] for count in USAGE)
    assert (lines[1]["model"], lines[1]["response_model"], lines[1]["stop_reason"]) == (
        MODEL, None, None)
    assert (lines[2]["model"], lines[2]["response_model"]) == (MODEL, MODEL)
    assert KEY not in text and "toy refusal" not in text


def test_the_usage_log_records_every_response_of_graphiti_s_own_retries(environment, key_file,
                                                                        tmp_path):
    pytest.importorskip("graphiti_core")
    from graphiti import adapter
    from graphiti_core.prompts.models import Message
    from pydantic import BaseModel

    class Answer(BaseModel):
        value: str

    invalid = message([{"type": "tool_use", "id": "toolu_01", "name": "Answer",
                        "input": {"other": 1}}], "tool_use", input_tokens=100, output_tokens=9)
    valid = message([{"type": "tool_use", "id": "toolu_02", "name": "Answer",
                      "input": {"value": "toy"}}], "tool_use", input_tokens=130, output_tokens=8)
    path = tmp_path / "usage.jsonl"
    recorder = Recorder((200, invalid), (200, valid))
    client = usage.instrument(llm.make_async_client(
        key_file, transport=httpx.MockTransport(recorder)), usage.UsageLog(str(path)))
    graphiti_llm = adapter.make_llm_client(client)

    async def calls(_):
        token = usage.usage_context.set("0123456789ab:PEP-1")
        try:
            return await graphiti_llm.generate_response(
                [Message(role="system", content="Extract."), Message(role="user", content="T.")],
                response_model=Answer)
        finally:
            usage.usage_context.reset(token)

    assert run(client, calls) == {"value": "toy"}
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [(line["input_tokens"], line["output_tokens"]) for line in lines] == [(100, 9),
                                                                                  (130, 8)]
    assert all(line["context"] == "0123456789ab:PEP-1" for line in lines)
    assert all(not {"temperature", "top_p", "top_k"} & set(body) for body in recorder.bodies())


def test_the_key_is_in_no_usage_line_and_no_error(environment, key_file, tmp_path):
    path = tmp_path / "usage.jsonl"
    refused = {"type": "error", "error": {"type": "authentication_error",
                                          "message": "invalid x-api-key"}}
    recorder = Recorder((200, TEXT), (401, refused))
    client = usage.instrument(llm.make_async_client(
        key_file, max_retries=0, transport=httpx.MockTransport(recorder)),
        usage.UsageLog(str(path)))

    async def calls(client):
        await ask(client)
        await ask(client)

    with pytest.raises(anthropic.AuthenticationError) as caught:
        run(client, calls)
    error = caught.value
    for text in (str(error), repr(error), "".join(traceback.format_exception(error)),
                 path.read_text(encoding="utf-8")):
        assert KEY not in text
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [line["error"] for line in lines] == [None, "AuthenticationError"]
    # A call that raised has the model it asked for and no response model.
    assert [(line["model"], line["response_model"]) for line in lines] == [(MODEL, MODEL),
                                                                          (MODEL, None)]


def test_a_key_file_that_holds_no_key_is_refused_without_showing_it(tmp_path):
    secret = "sk-ant-api00-would-be-secret"
    for content in ("", "\n", f"{secret}\nsecond line\n", f"{secret} with space\n"):
        path = tmp_path / "anthropic_api_key"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(llm.KeyFileError, match="does not hold an API key") as caught:
            llm.read_key(str(path))
        assert secret not in str(caught.value)
    path.write_bytes(b"sk-ant-\xe2\x80\x8bhidden\n")
    with pytest.raises(llm.KeyFileError) as caught:
        llm.read_key(str(path))
    assert "hidden" not in str(caught.value)
    # A file that cannot be read is refused by name and reason, never as an OSError's traceback.
    unreadable = tmp_path / "unreadable"
    unreadable.write_text(f"{secret}\n", encoding="ascii")
    unreadable.chmod(0)
    cases = [(tmp_path / "missing", "No such file or directory"), (tmp_path, "Is a directory")]
    if os.geteuid() != 0:  # root reads a file whatever its mode
        cases.append((unreadable, "Permission denied"))
    for path, why in cases:
        with pytest.raises(llm.KeyFileError) as caught:
            llm.read_key(str(path))
        assert str(caught.value) == f"{path} cannot be read ({why})"
        assert caught.value.__cause__ is None and secret not in str(caught.value)
    assert issubclass(llm.KeyFileError, ValueError)


def test_redact_removes_the_key_from_text(key_file):
    text = f"an error that quotes {KEY} twice: {KEY}."
    assert llm.redact(text, key_file) == "an error that quotes [redacted] twice: [redacted]."
    assert llm.redact("no key here", key_file) == "no key here"
    assert llm.redact("no key file", "/nonexistent/key") == "no key file"


def test_a_streaming_call_is_refused_by_the_usage_log(environment, key_file, tmp_path):
    client = usage.instrument(llm.make_async_client(
        key_file, transport=httpx.MockTransport(Recorder((200, TEXT)))),
        usage.UsageLog(str(tmp_path / "usage.jsonl")))
    with pytest.raises(NotImplementedError):
        run(client, lambda client: ask(client, stream=True))
