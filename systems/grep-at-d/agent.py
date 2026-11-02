"""Grep at D's agent (PREREGISTRATION.md sections 8 and 11): the version-control baseline, which
answers a question by reading its arm's tree as of the end of the question's date and the log
beside it (the files at D and the log, without diffs or earlier versions) with read-only shell
commands, which run in a sandbox that has no route out. It runs in an image of its own,
asofbench/grep-at-d, built FROM the reader image with this module and its self-test added
(grep-at-d/Dockerfile), through the reader's runner (reader/answer_requests.py), as the answering
function that the runner calls for each request, so that the runner's concurrency, its resume by
key, its usage log and its stops hold for it as for the single-call entrants:

    grep_at_d --requests /input/requests.jsonl --out /output/responses.jsonl \\
        --model claude-haiku-5-5|claude-sonnet-5-5 [--resume] [--api-errors-before N]

One run answers the requests of one arm and date, whose tree run_dates.py builds on the host and
the sandbox holds, read-only, at /repo (systems/README.md); run_dates.py gives each run the count
of ``api_error`` responses in a row that the dates before it ended with (``--api-errors-before``),
so that the runner's stop after five in a row spans the dates. A request holds the question
alone, and the runner refuses a requests file whose lines hold a context, another entrant's,
before it answers anything, with exit status 2 (``main``).

**The loop** (``answer_with_commands``). The first turn is the question, ``Question: {question}``,
as a single-call entrant's is when it has no context. Every turn is an answering call with the
registered settings (``asofbench_systems.answer.answering_settings``, with ``SYSTEM_PROMPT``) and
the one tool, ``run`` (``TOOL``), with ``tool_choice`` auto (``settings``). The transcript is
append-only: each reply goes back as the API sent it, block for block, thinking and redacted
thinking blocks included, followed by a user turn with one tool result for each of its calls, in
their order. A turn's commands run one after another in the sandbox, and each result is the
command's output as text (``render``): its stdout, then its stderr after a line ``stderr:`` when
it has any, then a last line with its exit code, or, for a command that ran past the limit, a last
line that says it was killed. A call whose input is not one command (``command_of``) is not run,
and its result is an error, ``NOT_RUN``.

**The cap.** At most MAX_TOOL_CALLS calls are answered per request: after the 40th result the user
turn ends with ``CAP_TURN``, a text block after the results; a call of that turn past the 40th is
not run, and its result is an error that holds ``CAP_TURN``. If the model calls the tool again
after the cap's turn, the answer is null with the error ``command_cap``, the request's own
outcome and never an API error. A call that is not run counts toward the cap too, so that every
loop ends.

**The answer.** A reply that calls no tool ends the loop, and its text is read as a single-call
entrant's is (``asofbench_systems.answer.read_answer``): a JSON object with an ``answer`` field,
or ``unparseable``. ``refusal`` and ``max_tokens`` on any turn give a null answer with that error,
an API error that is the request's own gives ``api_error``, and an infrastructure error is raised
for the runner to stop on, as ``answer_once`` does. The result is ``answer_once``'s fields, over
the whole answer: ``tokens_in`` and ``tokens_out`` summed over its turns (the input with the
cache's tokens, ``asofbench_systems.answer.counts``), ``wall_s`` the answer's, ``tool_calls`` the
commands run, ``thinking_blocks`` the thinking blocks of every turn.

**Prompt caching.** Every turn sends the whole transcript again, so each request marks its last
block, and no other, with ``cache_control`` ``{"type": "ephemeral"}`` (``marked``): the
multi-turn pattern, with a marker on a block, since the pinned SDK (anthropic 0.78.0) takes block
markers and has no automatic caching. Each turn reads the prefix that the one before it wrote. A
marker that moves is no edit of the history, so thinking blocks stay valid; it changes no output
and no setting.

**The sandbox** (``Sandbox``): ``POST /run`` to the command server at ``sandbox:7000``
(sandbox/command_server.py), on a TCP connection of its own for each command, never through the
proxy. A sandbox that cannot be reached, does not answer within SANDBOX_TIMEOUT_S or answers with
anything but a command's result is a failure of the environment (``SandboxUnavailable``, a
``ConnectionError``, which the runner's stop rule names): the run stops with exit status 2 and the
request has no response, so that a resumed run answers it again; it never becomes a null answer.

The network check waits for the sandbox as for the proxy and checks it as the environment's one
service (``netcheck_preflight``), and each run records the IDs of four images (``IMAGE_NAMES``):
grep at D's own, the reader's that it is built from, the proxy's and the sandbox's.
``--self-test``, which the image's build runs with no network, is the image's self-test
(``grep_at_d.selftest``): one answer of two turns through the installed SDK to an httpx
MockTransport, with a stand-in for the sandbox.

``anthropic`` and ``httpx`` are imported only inside the functions that need them, so the module
imports where neither is installed.
"""

from __future__ import annotations

import asyncio
import copy
import functools
import json
import sys
import time
from pathlib import Path

from asofbench_systems import answer
from asofbench_systems.run import INFRASTRUCTURE, infrastructure_error
from reader import answer_requests

# What the model is sent besides the question and the commands' results (section 11's grep at D):
# the system prompt, the one tool and its choice, and the cap's user turn.
SYSTEM_PROMPT = (
    "You answer one question about a repository of proposal files by reading it with read-only "
    "shell commands. The directory /repo holds the proposals' files as they stood at the end of "
    "the question's date, each named <repo>/<series>-<number> (the repository, peps, EIPs or "
    "ERCs; the series in lower case, pep, eip or erc; and the number, in four digits for a PEP), "
    "and git-log.txt, the log of the commits up to that date, newest first, with each commit's "
    "time, the files it touched and its message. Use the run tool to look. When you know the "
    "answer, reply with a JSON object whose \"answer\" field holds it in the form the question "
    "asks for.")
TOOL = {
    "name": "run",
    "description": ("Run a read-only shell command in /repo and return its output. Commands run "
                    "under bash with a 20 second limit; output beyond 20,000 characters is cut."),
    "input_schema": {"type": "object", "properties": {"command": {"type": "string"}},
                     "required": ["command"], "additionalProperties": False},
}
TOOL_CHOICE = {"type": "auto"}
MAX_TOOL_CALLS = 40
CAP_TURN = "You have used all 40 commands. Reply now with your final answer as the JSON object."
# The sandbox's limits, which the tool's description states and the command server enforces:
# seconds for a command, characters kept of each of its outputs, and bytes of a command.
COMMAND_TIMEOUT_S = 20
OUTPUT_LIMIT = 20_000
MAX_COMMAND_BYTES = 65_536
# How a command's result reads in its tool result (``render``), and the result of a call that is
# not one command.
STDERR_LABEL = "stderr:"
EXIT_LINE = "exit code: {}"
TIMED_OUT_LINE = f"the command ran past the {COMMAND_TIMEOUT_S} second limit and was killed"
NOT_RUN = ('Not run: the run tool takes one field, "command", a shell command of at most '
           f"{MAX_COMMAND_BYTES:,} bytes.")
# The codes of a request that was not answered: the single call's, and the cap's.
ERRORS = (*answer.ERRORS, "command_cap")
# The prompt cache's marker on the last block of each request.
CACHE = {"type": "ephemeral"}
# The command server, as the compose file names it; and how long a command may take to come back.
# Commands run one at a time there, and the runner has CONCURRENCY (8) requests under way, each
# with at most one command at the sandbox, so the last command in the queue comes back within
# eight commands' limits, its own included, of 20 s and a second each: 300 s is well past that.
SANDBOX = ("sandbox", 7000)
SANDBOX_TIMEOUT_S = 300.0
# The fields of a command's result, the largest answer that the sandbox may send, and the size of
# each read of it.
RESULT_FIELDS = ("exit_code", "stdout", "stderr", "truncated", "timed_out")
MAX_ANSWER_BYTES = 1 << 22
CHUNK = 1 << 16
# The images a run records, as the launcher gives them in ASOFBENCH_IMAGE_IDS: grep at D's own,
# the reader's that it is built from, the proxy's and the sandbox's.
IMAGE_NAMES = ("grep-at-d", "reader", "egress", "sandbox")


class SandboxUnavailable(ConnectionError):
    """The sandbox cannot be reached, or does not answer as the command server does: a failure of
    the environment, which stops the run and gives no request a null answer (the module's
    description)."""


def first_turn(question: str) -> dict:
    """The transcript's first message: the question, as a single-call entrant asks it."""
    return {"role": "user",
            "content": [{"type": "text", "text": answer.user_message(question, None)}]}


def settings(model: str) -> dict:
    """The keyword arguments of every turn besides its messages: the registered answering
    settings with grep at D's system prompt, and the one tool with tool_choice auto, a new copy
    for every turn."""
    return {**answer.answering_settings(model, system=SYSTEM_PROMPT),
            "tools": [copy.deepcopy(TOOL)], "tool_choice": dict(TOOL_CHOICE)}


def marked(transcript: list[dict]) -> list[dict]:
    """The messages of a turn's request: the transcript, with its last block marked for the
    prompt cache and the transcript itself left as it was."""
    *earlier, last = transcript
    blocks = list(last["content"])
    blocks[-1] = {**blocks[-1], "cache_control": dict(CACHE)}
    return [*earlier, {**last, "content": blocks}]


def command_of(use) -> str | None:
    """The command of a tool call: the input's one field, ``command``, a string of at most
    MAX_COMMAND_BYTES bytes in UTF-8 with no NUL character, of a call of ``run``; None for any
    other call, which is not run."""
    tool_input = getattr(use, "input", None)
    if (getattr(use, "name", None) != TOOL["name"] or not isinstance(tool_input, dict)
            or set(tool_input) != {"command"}):
        return None
    command = tool_input["command"]
    if not isinstance(command, str) or "\x00" in command:
        return None
    try:
        size = len(command.encode("utf-8"))
    except UnicodeEncodeError:  # a lone surrogate, which no process can take
        return None
    return command if size <= MAX_COMMAND_BYTES else None


def render(result: dict) -> str:
    """A command's result as its tool result reads (the module's description)."""
    def ended(text: str) -> str:
        return text if not text or text.endswith("\n") else text + "\n"

    stderr = f"{STDERR_LABEL}\n{ended(result['stderr'])}" if result["stderr"] else ""
    last = TIMED_OUT_LINE if result["timed_out"] else EXIT_LINE.format(result["exit_code"])
    return f"{ended(result['stdout'])}{stderr}{last}"


def tool_result(use_id: str, content: str, *, error: bool = False) -> dict:
    block = {"type": "tool_result", "tool_use_id": use_id, "content": content}
    if error:
        block["is_error"] = True
    return block


async def answer_with_commands(client, model: str, question: str, context: str | None, *,
                               sandbox) -> dict:
    """One request's answer (the module's description), through ``client`` (as the runner opens
    it) and ``sandbox`` (a ``Sandbox``); what it returns is ``answer_once``'s fields, and it raises
    what ``answer_once`` raises and ``SandboxUnavailable``. ValueError for a request with a
    context, which is another entrant's; ``main``'s runner refuses a requests file that holds one
    before it answers anything, so this guards only a call made otherwise."""
    import anthropic

    if context is not None:
        raise ValueError("a grep at D request holds no context: this requests file is another "
                         "entrant's")
    begun = time.monotonic()
    transcript = [first_turn(question)]
    tokens_in = tokens_out = thinking = calls = ran = 0
    capped = False
    while True:
        try:
            response = await client.messages.create(**settings(model),
                                                    messages=marked(transcript))
        except anthropic.APIError as error:
            if infrastructure_error(error, INFRASTRUCTURE):
                raise
            return answer.result(None, tokens_in, tokens_out, begun, "api_error", thinking, ran)
        counted_in, counted_out, counted_thinking = answer.counts(response)
        tokens_in, tokens_out = tokens_in + counted_in, tokens_out + counted_out
        thinking += counted_thinking
        if response.stop_reason in ("refusal", "max_tokens"):
            return answer.result(None, tokens_in, tokens_out, begun, response.stop_reason,
                                 thinking, ran)
        uses = [block for block in response.content if getattr(block, "type", None) == "tool_use"]
        if response.stop_reason != "tool_use" or not uses:
            found, error = answer.read_answer(response.content)
            return answer.result(found, tokens_in, tokens_out, begun, error, thinking, ran)
        if capped:
            return answer.result(None, tokens_in, tokens_out, begun, "command_cap", thinking, ran)
        transcript.append({"role": "assistant", "content": response.content})
        results = []
        for use in uses:
            calls += 1
            command = command_of(use)
            if calls > MAX_TOOL_CALLS:
                results.append(tool_result(use.id, CAP_TURN, error=True))
            elif command is None:
                results.append(tool_result(use.id, NOT_RUN, error=True))
            else:
                results.append(tool_result(use.id, render(await sandbox.run(command))))
                ran += 1
        if calls >= MAX_TOOL_CALLS:
            results.append({"type": "text", "text": CAP_TURN})
            capped = True
        transcript.append({"role": "user", "content": results})


def is_result(value) -> bool:
    """Whether ``value`` is a command's result as the command server answers it."""
    return (isinstance(value, dict) and set(value) == set(RESULT_FIELDS)
            and isinstance(value["exit_code"], int) and not isinstance(value["exit_code"], bool)
            and isinstance(value["stdout"], str) and isinstance(value["stderr"], str)
            and isinstance(value["truncated"], bool) and isinstance(value["timed_out"], bool))


def parsed(raw: bytes, where: str) -> dict:
    """The command's result that the sandbox's whole answer ``raw`` holds; SandboxUnavailable
    for anything else."""
    head, separator, body = raw.partition(b"\r\n\r\n")
    status_line, *header_lines = head.decode("latin-1").split("\r\n")
    words = status_line.split(" ", 2)
    if (not separator or len(words) < 2 or not words[0].startswith("HTTP/")
            or not words[1].isdigit()):
        raise SandboxUnavailable(f"the sandbox at {where} sent no HTTP answer")
    headers = {}
    for line in header_lines:
        name, _, value = line.partition(":")
        headers[name.strip().lower()] = value.strip()
    length = headers.get("content-length", "")
    if not length.isdigit() or int(length) != len(body):
        raise SandboxUnavailable(f"the sandbox at {where} sent an answer cut short")
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise SandboxUnavailable(f"the sandbox at {where} answered with no JSON") from None
    if words[1] != "200":
        reason = value.get("error") if isinstance(value, dict) else None
        raise SandboxUnavailable(f"the sandbox at {where} answered {words[1]}: {reason}")
    if not is_result(value):
        raise SandboxUnavailable(f"the sandbox at {where} answered with no command's result")
    return value


class Sandbox:
    """The agent's client of the command server at ``host``:``port``: one TCP connection for each
    command, with ``timeout_s`` for its answer (the module's description)."""

    def __init__(self, host: str = SANDBOX[0], port: int = SANDBOX[1],
                 timeout_s: float = SANDBOX_TIMEOUT_S) -> None:
        self.host, self.port, self.timeout_s = host, port, timeout_s

    async def run(self, command: str) -> dict:
        """The result of ``command``, run in the sandbox; SandboxUnavailable when there is none."""
        where = f"{self.host}:{self.port}"
        body = json.dumps({"command": command}, ensure_ascii=False).encode("utf-8")
        head = (f"POST /run HTTP/1.1\r\nHost: {where}\r\nContent-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n").encode("ascii")
        raw = bytearray()
        try:
            async with asyncio.timeout(self.timeout_s):
                reader, writer = await asyncio.open_connection(self.host, self.port)
                try:
                    writer.write(head + body)
                    await writer.drain()
                    while chunk := await reader.read(CHUNK):
                        raw += chunk
                        if len(raw) > MAX_ANSWER_BYTES:
                            raise SandboxUnavailable(f"the sandbox at {where} sent more than "
                                                     f"{MAX_ANSWER_BYTES} bytes")
                finally:
                    writer.close()
        except SandboxUnavailable:
            raise
        except OSError as error:  # refused, reset, unresolvable, or TimeoutError
            raise SandboxUnavailable(f"the sandbox at {where} cannot be reached "
                                     f"({type(error).__name__})") from error
        return parsed(bytes(raw), where)


def netcheck_preflight(out_dir: Path) -> int:
    """The reader's network check with the sandbox as the environment's one service: it is waited
    for, as the proxy is, and must accept a connection."""
    return answer_requests.netcheck_preflight(out_dir, services=(SANDBOX,))


def self_test() -> int:
    """The image's self-test (``grep_at_d.selftest``), imported only when it runs."""
    from grep_at_d import selftest

    return selftest.self_test()


def main(argv: list[str] | None = None) -> int:
    """``grep_at_d``: the reader's runner, with grep at D's answering function, images, network
    check and self-test, and its requests without a context: a requests file whose lines hold one
    is another entrant's, and the runner refuses it before anything is answered."""
    ask = functools.partial(answer_with_commands, sandbox=Sandbox())
    return answer_requests.main(argv, prog="grep_at_d", ask=ask, image_names=IMAGE_NAMES,
                                preflight=netcheck_preflight, self_test=self_test,
                                takes_context=False)


if __name__ == "__main__":
    sys.exit(main())
