"""The reader image's self-test, which its build runs with no network at all through the command
that every run uses (``answer_requests --self-test``), so that an image that builds is one that
answers as registered:

- ``check_installation``: every package that the lock (requirements.txt, at /app in the image)
  pins for this platform is installed at its pinned version, the pins' environment markers read
  with ``packaging``, and the network check is where the runner looks for it;
- ``check_answering``: one answering call (``asofbench_systems.answer.answer_once``) through the
  installed SDK, to an httpx MockTransport, goes to https://api.anthropic.com/v1/messages, sends
  the registered settings and the user message and nothing else, and no beta header, and its
  reply's answer is read.

It imports the runner, ``reader.answer_requests``, so that every module that the runner imports
is imported too; ``packaging``, ``anthropic`` and ``httpx`` are imported inside the checks that
use them.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import re
import sys
import tempfile
from pathlib import Path

from asofbench_systems import answer, llm
from reader import answer_requests

# The lock that the image installs (reader/requirements.txt, copied to /app/requirements.txt), and
# a pin of it: the package, its version, and the environment marker after a semicolon, if any.
REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements.txt"
PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)[ \t]*(?:;([^\\\n]*))?", re.M)
QUESTION = "Is this the self-test?"


def check_installation(requirements: Path = REQUIREMENTS) -> dict[str, str]:
    """Checks that every package that ``requirements`` pins for this platform is installed at its
    pinned version, and that the network check is where the runner looks for it; returns the
    installed versions. RuntimeError naming each difference otherwise."""
    from packaging.markers import Marker

    problems, installed = [], {}
    if not answer_requests.NETCHECK_SCRIPT.is_file():
        problems.append(f"the network check is not at {answer_requests.NETCHECK_SCRIPT}")
    for name, version, marker in PIN.findall(requirements.read_text(encoding="utf-8")):
        if marker.strip() and not Marker(marker.strip()).evaluate():
            continue
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed[name] = None
        if installed[name] != version:
            problems.append(f"{name} is {installed[name]}, {requirements.name} pins {version}")
    if not installed:
        problems.append(f"{requirements.name} pins no package")
    if problems:
        raise RuntimeError("; ".join(problems))
    return installed


def check_answering() -> str:
    """Checks, with no network, that an answering call through the installed SDK goes to the
    API's own URL, sends the registered settings and nothing else, with no beta header, and reads
    the answer of the reply; returns the SDK's version. RuntimeError naming each difference
    otherwise."""
    import anthropic
    import httpx

    sent = []
    reply = {"id": "msg_self_test", "type": "message", "role": "assistant",
             "model": answer.MODELS[0], "stop_reason": "end_turn", "stop_sequence": None,
             "content": [{"type": "thinking", "thinking": "", "signature": "c2VsZi10ZXN0"},
                         {"type": "text", "text": '{"answer": "self-test"}'}],
             "usage": {"input_tokens": 1, "output_tokens": 1}}

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json=reply)

    with tempfile.TemporaryDirectory() as scratch:
        key_file = Path(scratch) / "key"
        key_file.write_text("self-test-key\n", encoding="ascii")
        client = llm.make_async_client(str(key_file), max_retries=0,
                                       transport=httpx.MockTransport(respond))

        async def call() -> dict:
            try:
                return await answer.answer_once(client, answer.MODELS[0], QUESTION, None)
            finally:
                await client.close()

        result = asyncio.run(call())
    problems = []
    (request,) = sent
    if str(request.url) != f"{llm.BASE_URL}/v1/messages":
        problems.append(f"the call goes to {request.url}")
    body = json.loads(request.content)
    expected = {**answer.answering_settings(answer.MODELS[0]),
                "messages": [{"role": "user", "content": answer.user_message(QUESTION, None)}]}
    if body != expected:
        differing = sorted(name for name in set(body) | set(expected)
                           if body.get(name) != expected.get(name))
        problems.append(f"the call differs from the registered settings in {', '.join(differing)}")
    if "anthropic-beta" in request.headers:
        problems.append("the call sends an anthropic-beta header")
    if (result["answer"], result["error"], result["thinking_blocks"]) != ("self-test", None, 1):
        problems.append(f"the reply was read as {result!r}")
    if problems:
        raise RuntimeError("; ".join(problems))
    return anthropic.__version__


def self_test(requirements: Path = REQUIREMENTS) -> int:
    """Both checks: prints what they found and returns 0, or prints the differences and returns
    1."""
    try:
        installed = check_installation(requirements)
        sdk = check_answering()
    except RuntimeError as error:
        print(f"reader self-test: FAIL: {error}", file=sys.stderr)
        return 1
    print(f"reader self-test: {len(installed)} packages as {requirements.name} pins them, "
          f"anthropic {sdk}; an answering call sends the registered settings and nothing else: "
          f"{answer.MAX_TOKENS} max tokens, adaptive thinking, effort {answer.EFFORT}, the answer "
          "schema, no temperature, top_p, top_k or tool")
    return 0
