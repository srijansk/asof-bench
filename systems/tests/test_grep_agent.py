"""Grep at D's agent (grep-at-d/agent.py). With a mocked API, through the SDK that the reader
image's lock pins: every turn is an answering call with the registered settings and the one tool,
the transcript is append-only and every block of a reply goes back as it came, thinking blocks
included, each request marks its last block for the prompt cache, the commands are counted, the
cap's user turn is sent after the fortieth result and a further call is the request's own
``command_cap``, and the final reply's JSON is the answer; and the reader's runner answers through
the agent and the real command server, and stops with exit status 2, writing no response, when
the sandbox cannot be reached. Those tests need the anthropic package, so the root suite skips
them; CI runs them with the reader's SDK step:

    uv run --frozen --extra dev --with anthropic==0.78.0 --with httpx==0.28.1 \\
        python -m pytest systems/tests/test_answer.py systems/tests/test_llm_client.py \\
        systems/tests/test_grep_agent.py

The rest is standard library only: the registered strings, the text that a command's result
goes back as, the agent's sandbox client against the real command server, its network check, its
images and its command line. Every request goes to an httpx MockTransport and the key is a
made-up one, so no test reaches the API or holds a real key."""

from __future__ import annotations

import asyncio
import copy
import functools
import importlib.util
import json
import shutil
import socket
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import grep_modules
import pytest
from asofbench_systems import answer, llm, usage
from asofbench_systems import run as plumbing
from reader import answer_requests

agent = grep_modules.agent()
server = grep_modules.command_server()

SYSTEMS = Path(__file__).resolve().parents[1]
HAS_SDK = all(importlib.util.find_spec(name) is not None for name in ("anthropic", "httpx"))
needs_sdk = pytest.mark.skipif(not HAS_SDK, reason="needs the anthropic SDK and httpx, which the "
                               "reader image's lock pins (CI's reader SDK step)")
KEY = "sk-ant-api00-made-up-for-tests-" + "0123456789" * 4
MODEL = "claude-haiku-5-5"
# A made-up question in the form of the toy world's, which no v0.1 item asks.
QUESTION = ("What was the status of PEP 1 (\"Proposal 1\") as of the end of 2001-06-30 (UTC)? "
            "Answer with exactly one value from: Draft, Final.")
USAGE = {"input_tokens": 11, "output_tokens": 7, "cache_creation_input_tokens": 3,
         "cache_read_input_tokens": 5}
IMAGES = {"grep-at-d": "sha256:" + "4" * 64, "reader": "sha256:" + "1" * 64,
          "egress": "sha256:" + "2" * 64, "sandbox": "sha256:" + "3" * 64}
# The registered strings and tool, written out again here, so that an edit to agent.py that
# changes what the model is sent fails here and not only in a comparison with itself.
REGISTERED_PROMPT = (
    "You answer one question about a repository of proposal files by reading it with read-only "
    "shell commands. The directory /repo holds the proposals' files as they stood at the end of "
    "the question's date, each named <repo>/<series>-<number> (the repository, peps, EIPs or "
    "ERCs; the series in lower case, pep, eip or erc; and the number, in four digits for a PEP), "
    "and git-log.txt, the log of the commits up to that date, newest first, with each commit's "
    "time, the files it touched and its message. Use the run tool to look. When you know the "
    "answer, reply with a JSON object whose \"answer\" field holds it in the form the question "
    "asks for.")
REGISTERED_TOOL = {
    "name": "run",
    "description": ("Run a read-only shell command in /repo and return its output. Commands run "
                    "under bash with a 20 second limit; output beyond 20,000 characters is cut."),
    "input_schema": {"type": "object", "properties": {"command": {"type": "string"}},
                     "required": ["command"], "additionalProperties": False},
}
REGISTERED_CAP = ("You have used all 40 commands. Reply now with your final answer as the JSON "
                  "object.")
SAMPLING = {"temperature", "top_p", "top_k"}
# A thinking block as Claude Haiku 5.5 returns it by default, its text omitted; one with text, as a
# summarised display gives it; and a redacted one.
THINKING = {"type": "thinking", "thinking": "", "signature": "c2lnbmF0dXJlLTE="}
SUMMARISED = {"type": "thinking", "thinking": "The log first.", "signature": "c2lnbmF0dXJlLTI="}
REDACTED = {"type": "redacted_thinking", "data": "cmVkYWN0ZWQtdGhpbmtpbmc="}


def message(content: list[dict], stop_reason: str = "end_turn", **counts: int) -> dict:
    return {"id": "msg_01TEST", "type": "message", "role": "assistant", "model": MODEL,
            "content": content, "stop_reason": stop_reason, "stop_sequence": None,
            "usage": {**USAGE, **counts}}


def text(value: str, **extra) -> dict:
    return {"type": "text", "text": value, **extra}


def call(n: int, command, name: str = "run") -> dict:
    """A tool_use block, with ``command`` as its input's one field (or ``command`` as the whole
    input when it is not a string)."""
    tool_input = {"command": command} if isinstance(command, str) else command
    return {"type": "tool_use", "id": f"toolu_{n:03d}", "name": name, "input": tool_input}


def calling(*commands: str, start: int = 1) -> dict:
    """A reply that calls the tool once for each of ``commands``."""
    return message([THINKING, *(call(start + n, c) for n, c in enumerate(commands))], "tool_use")


def final(value) -> dict:
    return message([THINKING, text(json.dumps({"answer": value}))])


def error_body(kind: str) -> dict:
    return {"type": "error", "error": {"type": kind, "message": "a toy error"}}


class Script:
    """A MockTransport handler: records each request and answers the n-th with the n-th of
    ``answers`` (a status and body, a message, or an exception class to raise)."""

    def __init__(self, *answers) -> None:
        self.answers, self.requests = list(answers), []

    def __call__(self, request):
        import httpx

        self.requests.append(request)
        reply = self.answers[len(self.requests) - 1]
        if isinstance(reply, type) and issubclass(reply, Exception):
            raise reply("no route", request=request)
        status, body = reply if isinstance(reply, tuple) else (200, reply)
        return httpx.Response(status, json=body)

    def bodies(self) -> list[dict]:
        return [json.loads(request.content) for request in self.requests]


class Sandbox:
    """Stands in for the agent's sandbox client: records each command and answers it with
    ``results[command]``, or with a run that printed a line naming it."""

    def __init__(self, results: dict | None = None) -> None:
        self.results, self.commands = results or {}, []

    async def run(self, command: str) -> dict:
        self.commands.append(command)
        return self.results.get(command, {"exit_code": 0, "stdout": f"output of {command}\n",
                                          "stderr": "", "truncated": False, "timed_out": False})


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
def key_file(tmp_path) -> str:
    path = tmp_path / "anthropic_api_key"
    path.write_text(KEY + "\n", encoding="ascii")
    return str(path)


def ask(key_file: str, script: Script, sandbox, question: str = QUESTION,
        context: str | None = None) -> dict:
    """The agent's answer to ``question``, with a client whose requests go to ``script``."""
    import httpx

    client = llm.make_async_client(key_file, max_retries=0, transport=httpx.MockTransport(script))

    async def go():
        try:
            return await agent.answer_with_commands(client, MODEL, question, context,
                                                    sandbox=sandbox)
        finally:
            await client.close()

    return asyncio.run(go())


def unmarked(value):
    """``value`` without its cache_control markers, at any depth."""
    if isinstance(value, dict):
        return {key: unmarked(item) for key, item in value.items() if key != "cache_control"}
    if isinstance(value, list):
        return [unmarked(item) for item in value]
    return value


def markers(value, path: str = "") -> list[str]:
    """Where ``value`` holds a cache_control marker."""
    if isinstance(value, dict):
        return [*([path] if "cache_control" in value else []),
                *(found for key, item in value.items() for found in markers(item, f"{path}.{key}"))]
    if isinstance(value, list):
        return [found for n, item in enumerate(value) for found in markers(item, f"{path}[{n}]")]
    return []


def results_of(body: dict) -> list[dict]:
    """The blocks of a request's last user turn, markers left out."""
    return unmarked(body["messages"][-1]["content"])


# --- the registered strings, the turn's settings and a command's text (standard library) ------


def test_the_registered_strings_and_limits_are_exactly_these():
    assert agent.SYSTEM_PROMPT == REGISTERED_PROMPT
    assert agent.TOOL == REGISTERED_TOOL
    assert agent.TOOL_CHOICE == {"type": "auto"}
    assert agent.CAP_TURN == REGISTERED_CAP
    assert agent.MAX_TOOL_CALLS == 40
    assert agent.ERRORS == ("refusal", "max_tokens", "api_error", "unparseable", "command_cap")
    assert agent.CACHE == {"type": "ephemeral"}
    # The user turn is the single-call entrants' with no context.
    assert agent.first_turn(QUESTION) == {"role": "user", "content": [
        {"type": "text", "text": answer.user_message(QUESTION, None)}]}


def test_every_turn_s_settings_are_the_answering_settings_with_the_tool_added():
    settings = agent.settings("claude-sonnet-5-5")
    assert settings == {**answer.answering_settings("claude-sonnet-5-5", system=REGISTERED_PROMPT),
                        "tools": [REGISTERED_TOOL], "tool_choice": {"type": "auto"}}
    assert not SAMPLING & set(settings)
    # A new copy for every turn: a caller that changes what it was given changes no later turn.
    settings["tools"][0]["input_schema"]["required"].append("other")
    settings["tool_choice"]["type"] = "any"
    settings["output_config"]["effort"] = "high"
    assert agent.settings(MODEL)["tools"] == [REGISTERED_TOOL]
    assert agent.settings(MODEL)["tool_choice"] == {"type": "auto"}
    assert agent.settings(MODEL)["output_config"]["effort"] == "medium"
    assert agent.TOOL == REGISTERED_TOOL


def test_the_tool_s_description_states_the_sandbox_s_limits():
    assert agent.COMMAND_TIMEOUT_S == server.TIMEOUT_S == 20
    assert agent.OUTPUT_LIMIT == server.LIMIT == 20_000
    assert agent.MAX_COMMAND_BYTES == server.MAX_COMMAND_BYTES
    assert f"{agent.COMMAND_TIMEOUT_S} second limit" in agent.TOOL["description"]
    assert f"{agent.OUTPUT_LIMIT:,} characters" in agent.TOOL["description"]
    assert agent.SANDBOX == ("sandbox", server.PORT)


def test_a_command_s_result_goes_back_as_its_output_and_its_exit_code():
    def result(stdout="", stderr="", exit_code=0, timed_out=False, truncated=False) -> dict:
        return {"exit_code": exit_code, "stdout": stdout, "stderr": stderr,
                "truncated": truncated, "timed_out": timed_out}

    assert agent.render(result("peps\ngit-log.txt\n")) == "peps\ngit-log.txt\nexit code: 0"
    assert agent.render(result()) == "exit code: 0"
    assert agent.render(result(exit_code=1)) == "exit code: 1"
    assert agent.render(result("no newline")) == "no newline\nexit code: 0"
    assert agent.render(result("out\n", "cat: x: No such file or directory\n", 1)) == (
        "out\nstderr:\ncat: x: No such file or directory\nexit code: 1")
    assert agent.render(result(stderr="warn")) == "stderr:\nwarn\nexit code: 0"
    cut = "a" * 20_000 + "\n[cut: 5 characters omitted]\n"
    assert agent.render(result(cut, truncated=True)) == f"{cut}exit code: 0"
    assert agent.render(result("partial\n", exit_code=-9, timed_out=True)) == (
        "partial\nthe command ran past the 20 second limit and was killed")


def test_only_an_input_of_one_command_string_is_a_command():
    class Use:
        def __init__(self, tool_input, name="run"):
            self.input, self.name = tool_input, name

    assert agent.command_of(Use({"command": "ls -la peps"})) == "ls -la peps"
    assert agent.command_of(Use({"command": ""})) == ""
    longest = "x" * agent.MAX_COMMAND_BYTES
    assert agent.command_of(Use({"command": longest})) == longest
    for tool_input, name in (({}, "run"), ({"command": 5}, "run"), ({"cmd": "ls"}, "run"),
                             ({"command": "ls", "timeout": 1}, "run"), ("ls", "run"),
                             (None, "run"), ({"command": "ls"}, "bash"),
                             ({"command": "echo a\x00b"}, "run"),
                             ({"command": "é" * (agent.MAX_COMMAND_BYTES // 2 + 1)}, "run")):
        assert agent.command_of(Use(tool_input, name)) is None, (tool_input, name)


# --- the sandbox client, against the real command server (standard library) -------------------


@pytest.fixture
def sandbox_server(tmp_path):
    """The command server on a free local port, over a toy tree."""
    if shutil.which("bash") is None:
        pytest.skip("the command server runs its commands under bash")
    root = tmp_path / "repo"
    (root / "peps").mkdir(parents=True)
    (root / "peps" / "pep-0001").write_text("PEP: 1\nTitle: Proposal 1\nStatus: Draft\n",
                                            encoding="utf-8")
    (root / "git-log.txt").write_text("commit 0\n", encoding="utf-8")
    (tmp_path / "scratch").mkdir()
    httpd = server.CommandServer(("127.0.0.1", 0), root=root, scratch=tmp_path / "scratch")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()
    thread.join()


def closed_port() -> int:
    """A local port that nothing listens on (it was free a moment ago)."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_the_sandbox_client_runs_a_command_on_the_command_server(sandbox_server):
    client = agent.Sandbox(*sandbox_server.server_address[:2])
    result = asyncio.run(client.run("grep Status peps/pep-0001; echo 'é' >&2; exit 2"))
    assert result == {"exit_code": 2, "stdout": "Status: Draft\n", "stderr": "é\n",
                      "truncated": False, "timed_out": False}


def test_a_sandbox_that_cannot_be_reached_or_answers_otherwise_is_an_infrastructure_failure(
        sandbox_server):
    def failure(client: agent.Sandbox, command: str = "ls") -> agent.SandboxUnavailable:
        with pytest.raises(agent.SandboxUnavailable) as caught:
            asyncio.run(client.run(command))
        return caught.value

    refused = failure(agent.Sandbox("127.0.0.1", closed_port()))
    # The runner's stop rule names it: the standard library's ConnectionError, an infrastructure
    # error, so that the run stops and no request is given a null answer.
    assert plumbing.infrastructure_error(refused, plumbing.INFRASTRUCTURE) == (
        "builtins ConnectionError")
    # A server that answers with a failure, as the command server does when bash cannot start.
    shutil.rmtree(sandbox_server.root)
    assert "500" in str(failure(agent.Sandbox(*sandbox_server.server_address[:2])))
    # Something that is not the command server: no HTTP, or an HTTP answer that is not a result.
    for reply in (b"", b"not http at all", b"HTTP/1.0 200 OK\r\nContent-Length: 2\r\n\r\n{}",
                  b"HTTP/1.0 200 OK\r\nContent-Length: 99\r\n\r\n{\"cut\": ",
                  b"HTTP/1.0 200 OK\r\nContent-Length: 4\r\n\r\nnull"):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()

            def answer_once(listener=listener, reply=reply):
                connection, _ = listener.accept()
                with connection:
                    connection.recv(65536)
                    connection.sendall(reply)

            thread = threading.Thread(target=answer_once, daemon=True)
            thread.start()
            error = failure(agent.Sandbox(*listener.getsockname()))
            thread.join()
        assert plumbing.infrastructure_error(error, plumbing.INFRASTRUCTURE) is not None, reply
    # A sandbox that does not answer in time.
    with socket.socket() as silent:
        silent.bind(("127.0.0.1", 0))
        silent.listen()
        error = failure(agent.Sandbox(*silent.getsockname(), timeout_s=0.2))
    assert "TimeoutError" in str(error)


def test_the_sandbox_client_waits_long_enough_for_a_full_queue_of_commands():
    # Commands run one at a time, and the runner has CONCURRENCY requests under way, each with at
    # most one command at the sandbox: the last in the queue waits for every other to time out.
    worst = answer_requests.CONCURRENCY * (server.TIMEOUT_S + server.KILL_GRACE_S + 1)
    assert agent.Sandbox().timeout_s == agent.SANDBOX_TIMEOUT_S > worst


def test_grep_at_d_s_network_check_waits_for_the_proxy_and_the_sandbox(tmp_path, monkeypatch):
    waits, commands = [], []
    monkeypatch.setattr(plumbing, "wait_for_services", lambda *arguments: waits.append(arguments))

    def fake_run(command, check):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(plumbing.subprocess, "run", fake_run)
    assert agent.netcheck_preflight(tmp_path) == 0
    assert waits == [((("egress", 3128), ("sandbox", 7000)), 60.0, 1.0),
                     ((("sandbox", 7000),), 0.0, 1.0)]
    assert commands == [[sys.executable, str(SYSTEMS / "env" / "netcheck.py"),
                         *plumbing.NETCHECK_ARGS, "--services", "sandbox:7000",
                         "--mounts", "/input,/output,/run/secrets",
                         "--report", str(tmp_path / "netcheck.json")]]


def test_grep_at_d_s_images_are_its_own_the_reader_s_the_proxy_s_and_the_sandbox_s(tmp_path,
                                                                                  monkeypatch,
                                                                                  capsys):
    # Its own image, and the reader image that it is built from, beside the proxy's and the
    # sandbox's.
    assert agent.IMAGE_NAMES == ("grep-at-d", "reader", "egress", "sandbox")
    (tmp_path / "input").mkdir()
    (tmp_path / "output").mkdir()
    (tmp_path / "input" / "requests.jsonl").write_text(
        json.dumps({"key": str(uuid.uuid4()), "question": QUESTION}) + "\n", encoding="utf-8")
    args = ["--requests", str(tmp_path / "input" / "requests.jsonl"),
            "--out", str(tmp_path / "output" / "responses.jsonl"), "--model", MODEL]
    monkeypatch.setenv("ASOFBENCH_IMAGE_IDS", ",".join(f"{name}={IMAGES[name]}"
                                                       for name in ("reader", "egress", "sandbox")))
    assert agent.main(args) == plumbing.EXIT_STOPPED
    assert "name=ID entries for grep-at-d, reader, egress, sandbox (systems/README.md)" in (
        capsys.readouterr().err)
    with pytest.raises(SystemExit) as caught:
        agent.main(["--self-test", "--model", MODEL])
    assert caught.value.code == 2
    assert list((tmp_path / "output").iterdir()) == []


# --- the loop, through the pinned SDK with a mocked API ------------------------------------------


@needs_sdk
def test_every_turn_sends_the_registered_settings_and_the_one_tool(environment, key_file):
    script = Script(calling("ls"), calling("cat git-log.txt", start=2), final("Draft"))
    assert ask(key_file, script, Sandbox())["answer"] == "Draft"
    assert len(script.requests) == 3
    for request, body in zip(script.requests, script.bodies(), strict=True):
        assert str(request.url) == "https://api.anthropic.com/v1/messages"
        assert body["model"] == MODEL
        assert body["max_tokens"] == 32000
        assert body["thinking"] == {"type": "adaptive"}
        assert body["output_config"] == {"effort": "medium",
                                         "format": {"type": "json_schema",
                                                    "schema": answer.ANSWER_SCHEMA}}
        assert body["system"] == REGISTERED_PROMPT
        assert body["tools"] == [REGISTERED_TOOL]
        assert body["tool_choice"] == {"type": "auto"}
        # Nothing else: no sampling parameter, no streaming, no server tool, no beta header.
        assert set(body) == {"model", "max_tokens", "thinking", "output_config", "system",
                             "tools", "tool_choice", "messages"}
        assert not SAMPLING & set(body)
        assert "anthropic-beta" not in request.headers
        assert request.headers["x-api-key"] == KEY and "authorization" not in request.headers
    first = script.bodies()[0]
    assert unmarked(first["messages"]) == [{"role": "user", "content": [
        {"type": "text", "text": f"Question: {QUESTION}"}]}]


@needs_sdk
def test_the_transcript_is_append_only_and_every_block_goes_back_as_it_came(environment,
                                                                             key_file):
    turns = [
        message([SUMMARISED, text("The log first.", citations=None), call(1, "head git-log.txt")],
                "tool_use"),
        message([REDACTED, call(2, "ls peps"), call(3, "cat peps/pep-0001")], "tool_use"),
        message([THINKING, text('{"answer": "Draft"}')]),
    ]
    script = Script(*turns)
    result = ask(key_file, script, Sandbox())
    assert (result["answer"], result["error"], result["tool_calls"]) == ("Draft", None, 3)
    bodies = [unmarked(body["messages"]) for body in script.bodies()]
    for earlier, later, reply in zip(bodies, bodies[1:], turns, strict=False):
        # Each request repeats the one before it, then the reply to it, block for block as the
        # API sent it (thinking, redacted thinking and text alike), then the tool results.
        assert later[:len(earlier)] == earlier
        assert later[len(earlier)] == {"role": "assistant", "content": reply["content"]}
        assert later[len(earlier) + 1]["role"] == "user"
        assert len(later) == len(earlier) + 2
    assert [block["tool_use_id"] for block in bodies[2][-1]["content"]] == ["toolu_002",
                                                                           "toolu_003"]


@needs_sdk
def test_each_request_marks_its_last_block_for_the_prompt_cache_and_nothing_else(environment,
                                                                                 key_file):
    script = Script(calling("ls"), calling("ls peps", "wc -l git-log.txt", start=2),
                    final("Draft"))
    ask(key_file, script, Sandbox())
    for body in script.bodies():
        last = len(body["messages"]) - 1
        assert markers(body) == [f".messages[{last}].content"
                                 f"[{len(body['messages'][-1]['content']) - 1}]"]
        assert body["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}


@needs_sdk
def test_the_final_reply_s_json_is_the_answer_and_the_costs_are_summed(environment, key_file):
    script = Script(message([THINKING, call(1, "ls")], "tool_use", output_tokens=20),
                    message([call(2, "grep -c Status peps/*")], "tool_use", input_tokens=100),
                    message([THINKING, THINKING, text('{"answer": [1, "PEP 8"]}')]))
    sandbox = Sandbox()
    result = ask(key_file, script, sandbox)
    assert list(result) == list(answer.RESULT_FIELDS)
    assert {name: result[name] for name in answer.RESULT_FIELDS if name != "wall_s"} == {
        "answer": [1, "PEP 8"],
        "tokens_in": (11 + 3 + 5) * 2 + (100 + 3 + 5),
        "tokens_out": 20 + 7 + 7, "tool_calls": 2, "error": None, "thinking_blocks": 3}
    assert isinstance(result["wall_s"], float) and result["wall_s"] >= 0
    assert sandbox.commands == ["ls", "grep -c Status peps/*"]
    # An answer from memory, with no command run.
    assert ask(key_file, Script(final("Final")), Sandbox())["tool_calls"] == 0


@needs_sdk
def test_each_command_s_result_goes_back_as_its_output_text(environment, key_file):
    failing = {"exit_code": 2, "stdout": "", "stderr": "grep: x: No such file or directory\n",
               "truncated": False, "timed_out": False}
    script = Script(calling("ls", "grep Status x"), final("Draft"))
    ask(key_file, script, Sandbox({"grep Status x": failing}))
    assert results_of(script.bodies()[1]) == [
        {"type": "tool_result", "tool_use_id": "toolu_001",
         "content": "output of ls\nexit code: 0"},
        {"type": "tool_result", "tool_use_id": "toolu_002",
         "content": "stderr:\ngrep: x: No such file or directory\nexit code: 2"}]


def forty_calls_then(*last) -> Script:
    """Forty replies of one command each, then ``last``."""
    return Script(*(calling(f"echo {n}", start=n) for n in range(1, 41)), *last)


@needs_sdk
def test_after_the_fortieth_result_the_cap_turn_is_sent_and_a_further_call_is_command_cap(
        environment, key_file):
    sandbox = Sandbox()
    script = forty_calls_then(calling("echo 41", start=41))
    result = ask(key_file, script, sandbox)
    assert (result["answer"], result["error"], result["tool_calls"]) == (None, "command_cap", 40)
    assert len(script.requests) == 41 and len(sandbox.commands) == 40
    bodies = script.bodies()
    # Each turn before the fortieth result ends with that result; after the fortieth, the cap's
    # user turn follows it, as a text block after the tool result.
    for n, body in enumerate(bodies[1:40], 1):
        assert [block["type"] for block in results_of(body)] == ["tool_result"], n
    assert results_of(bodies[40]) == [
        {"type": "tool_result", "tool_use_id": "toolu_040", "content": "output of echo 40\n"
                                                                      "exit code: 0"},
        {"type": "text", "text": REGISTERED_CAP}]


@needs_sdk
def test_an_answer_after_the_cap_turn_is_kept(environment, key_file):
    result = ask(key_file, forty_calls_then(final("Final")), Sandbox())
    assert (result["answer"], result["error"], result["tool_calls"]) == ("Final", None, 40)


@needs_sdk
def test_calls_past_the_fortieth_in_one_turn_are_not_run(environment, key_file):
    sandbox = Sandbox()
    script = Script(*(calling(f"echo {n}", start=n) for n in range(1, 39)),
                    calling("echo 39", "echo 40", "echo 41", "echo 42", start=39),
                    final("Draft"))
    result = ask(key_file, script, sandbox)
    assert (result["answer"], result["tool_calls"]) == ("Draft", 40)
    assert sandbox.commands == [f"echo {n}" for n in range(1, 41)]
    assert results_of(script.bodies()[39]) == [
        {"type": "tool_result", "tool_use_id": "toolu_039",
         "content": "output of echo 39\nexit code: 0"},
        {"type": "tool_result", "tool_use_id": "toolu_040",
         "content": "output of echo 40\nexit code: 0"},
        {"type": "tool_result", "tool_use_id": "toolu_041", "content": REGISTERED_CAP,
         "is_error": True},
        {"type": "tool_result", "tool_use_id": "toolu_042", "content": REGISTERED_CAP,
         "is_error": True},
        {"type": "text", "text": REGISTERED_CAP}]


@needs_sdk
def test_a_call_that_is_not_one_command_is_not_run_and_counts_toward_the_cap(environment,
                                                                            key_file):
    sandbox = Sandbox()
    script = Script(message([call(1, {}), call(2, {"command": 5}), call(3, "ls", name="bash"),
                             call(4, "ls")], "tool_use"),
                    final("Draft"))
    result = ask(key_file, script, sandbox)
    assert (result["answer"], result["tool_calls"]) == ("Draft", 1)
    assert sandbox.commands == ["ls"]
    blocks = results_of(script.bodies()[1])
    assert [block.get("is_error", False) for block in blocks] == [True, True, True, False]
    assert {block["content"] for block in blocks[:3]} == {agent.NOT_RUN}
    # Forty calls that are not commands use up the cap as forty commands would.
    script = Script(*(message([call(n, {})], "tool_use") for n in range(1, 41)),
                    message([call(41, "ls")], "tool_use"))
    result = ask(key_file, script, Sandbox())
    assert (result["error"], result["tool_calls"]) == ("command_cap", 0)


@needs_sdk
def test_a_refusal_or_a_max_tokens_stop_on_any_turn_gives_a_null_answer(environment, key_file):
    for stop_reason in ("refusal", "max_tokens"):
        # Even a reply whose text would parse, or that calls the tool.
        last = message([THINKING, call(2, "ls"), text('{"answer": "Draft"}')], stop_reason)
        result = ask(key_file, Script(calling("ls"), last), Sandbox())
        assert (result["answer"], result["error"], result["tool_calls"]) == (None, stop_reason, 1)
        assert (result["tokens_in"], result["tokens_out"]) == (19 * 2, 7 * 2)


@needs_sdk
def test_a_final_reply_that_is_not_one_json_answer_is_unparseable(environment, key_file):
    for content in ([text("Draft")], [text('{"other": 1}')], [THINKING], []):
        result = ask(key_file, Script(calling("ls"), message(content)), Sandbox())
        assert (result["answer"], result["error"], result["tool_calls"]) == (None, "unparseable",
                                                                             1), content


@needs_sdk
def test_an_api_error_that_is_the_request_s_own_keeps_the_tokens_spent_so_far(environment,
                                                                              key_file):
    for status, kind in ((400, "invalid_request_error"), (413, "request_too_large")):
        result = ask(key_file, Script(calling("ls"), (status, error_body(kind))), Sandbox())
        assert {name: result[name] for name in answer.RESULT_FIELDS if name != "wall_s"} == {
            "answer": None, "tokens_in": 19, "tokens_out": 7, "tool_calls": 1,
            "error": "api_error", "thinking_blocks": 1}, status


@needs_sdk
def test_an_infrastructure_error_on_any_turn_is_raised(environment, key_file):
    import anthropic
    import httpx

    for reply in ((529, error_body("overloaded_error")), (429, error_body("rate_limit_error")),
                  (401, error_body("authentication_error")), httpx.ConnectError):
        with pytest.raises(anthropic.APIError) as caught:
            ask(key_file, Script(calling("ls"), reply), Sandbox())
        assert plumbing.infrastructure_error(caught.value, plumbing.INFRASTRUCTURE) is not None
        assert KEY not in str(caught.value)


@needs_sdk
def test_a_request_that_holds_a_context_is_not_grep_at_d_s(environment, key_file):
    script = Script(final("Draft"))
    with pytest.raises(ValueError, match="context"):
        ask(key_file, script, Sandbox(), context="Another entrant's context.")
    assert script.requests == []


def layout(tmp_path, count: int) -> list[dict]:
    """A run's input and output directories for one date, with ``count`` requests."""
    (tmp_path / "input").mkdir()
    (tmp_path / "output").mkdir()
    requests = [{"key": str(uuid.uuid4()), "question": f"{QUESTION} ({n})"} for n in range(count)]
    (tmp_path / "input" / "requests.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in sorted(requests, key=lambda r: r["key"])),
        encoding="utf-8")
    return requests


def run_through_the_runner(tmp_path, key_file, handler, sandbox, *, resume=False) -> int:
    import httpx

    def open_client(log, key):
        return usage.instrument(llm.make_async_client(
            key, max_retries=0, transport=httpx.MockTransport(handler)), log)

    return asyncio.run(answer_requests.run(
        tmp_path / "input" / "requests.jsonl", tmp_path / "output" / "responses.jsonl", MODEL,
        resume=resume, images=IMAGES, open_client=open_client,
        ask=functools.partial(agent.answer_with_commands, sandbox=sandbox), preflight=None,
        key_file=key_file, progress=lambda line: None))


def by_turn(request):
    """A MockTransport handler for conversations under way at once: a request whose last turn is
    the question calls the tool, and one whose last turn holds a tool result answers with the
    result's first line."""
    import httpx

    body = json.loads(request.content)
    last = body["messages"][-1]["content"][0]
    if last["type"] == "text":
        reply = message([THINKING, call(1, "grep Status peps/pep-0001")], "tool_use")
    else:
        reply = final(last["content"].splitlines()[0])
    return httpx.Response(200, json=reply)


@needs_sdk
def test_the_reader_s_runner_answers_through_the_agent_and_the_real_sandbox(environment,
                                                                            key_file, tmp_path,
                                                                            sandbox_server):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    requests = layout(run_dir, 3)
    sandbox = agent.Sandbox(*sandbox_server.server_address[:2])
    assert run_through_the_runner(run_dir, key_file, by_turn, sandbox) == plumbing.EXIT_DONE
    responses = [json.loads(line) for line in
                 (run_dir / "output" / "responses.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sorted(r["key"] for r in responses) == sorted(r["key"] for r in requests)
    for response in responses:
        assert list(response) == list(answer.RESPONSE_FIELDS)
        # The command ran in the tree, and its output reached the model.
        assert (response["answer"], response["error"], response["tool_calls"]) == (
            "Status: Draft", None, 1)
        assert response["thinking_blocks"] == 2
    lines = [json.loads(line) for line in
             (run_dir / "output" / "usage.jsonl").read_text(encoding="utf-8").splitlines()]
    # Both of each request's calls are in the usage log under its key.
    assert sorted((line["context"], line["invocation"], line["attempt"]) for line in lines) == (
        sorted((r["key"], 1, 1) for r in requests for _ in range(2)))


@needs_sdk
def test_a_sandbox_that_cannot_be_reached_stops_the_run_with_exit_status_2(environment,
                                                                          key_file, tmp_path):
    requests = layout(tmp_path, 2)
    sandbox = agent.Sandbox("127.0.0.1", closed_port())
    assert run_through_the_runner(tmp_path, key_file, by_turn,
                                  sandbox) == plumbing.EXIT_STOPPED
    # No request was given a null answer: the stop leaves each unanswered, for a resume.
    responses = tmp_path / "output" / "responses.jsonl"
    assert not responses.exists() or responses.read_text(encoding="utf-8") == ""
    record = json.loads((tmp_path / "output" / "run.json").read_text(encoding="utf-8"))
    (invocation,) = record["invocations"]
    assert invocation["exit"] == 2
    assert invocation["stop"]["error"] == "builtins ConnectionError"
    assert invocation["stop"]["key"] in {r["key"] for r in requests}


@needs_sdk
def test_the_self_test_passes_on_this_sdk_and_fails_on_a_loop_that_drifts(environment,
                                                                         monkeypatch, capsys):
    # The image's self-test is a module of its own, which the agent's --self-test runs.
    selftest = grep_modules.selftest()
    assert selftest.check_loop() == importlib.import_module("anthropic").__version__
    assert agent.self_test() == 0
    assert capsys.readouterr().out.startswith("grep at D self-test: ")
    registered = agent.settings

    def drifting(model):
        return {**registered(model), "temperature": 1.0}

    monkeypatch.setattr(agent, "settings", drifting)
    assert agent.self_test() == 1
    assert "temperature" in capsys.readouterr().err
    monkeypatch.setattr(agent, "settings", registered)

    def unmarked_requests(transcript):
        return copy.deepcopy(transcript)

    monkeypatch.setattr(agent, "marked", unmarked_requests)
    assert agent.self_test() == 1
    assert "cache" in capsys.readouterr().err
