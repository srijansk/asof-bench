"""Grep at D's self-test, which its image's build runs with no network at all through the command
that every run uses (``grep_at_d --self-test``), so that an image that builds is one whose agent
answers as registered: ``check_loop`` makes one answer of two turns through the installed SDK to an
httpx MockTransport, with a stand-in for the sandbox, and checks that each turn goes to
https://api.anthropic.com/v1/messages with the registered settings, the one tool and tool_choice
auto and no beta header, that the second turn sends the first reply back as it came, and that each
request marks its last block, and it alone, for the prompt cache. The installed packages are the
reader image's, which its own self-test checks against its lock.

It imports the agent, ``grep_at_d.agent``, so that every module the agent imports is imported too;
``anthropic`` and ``httpx`` are imported inside the check that uses them.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

from asofbench_systems import answer, llm
from grep_at_d import agent

QUESTION = "Is this the self-test?"


def unmarked(value):
    """``value`` without its prompt cache markers, at any depth."""
    if isinstance(value, dict):
        return {key: unmarked(item) for key, item in value.items() if key != "cache_control"}
    if isinstance(value, list):
        return [unmarked(item) for item in value]
    return value


def marker_count(value) -> int:
    """How many prompt cache markers ``value`` holds, at any depth."""
    if isinstance(value, dict):
        return ("cache_control" in value) + sum(marker_count(item) for item in value.values())
    if isinstance(value, list):
        return sum(marker_count(item) for item in value)
    return 0


def check_loop() -> str:
    """Checks, with no network, that an answer of two turns through the installed SDK sends the
    registered settings with the tool on each turn, sends the first reply back as it came and
    marks each request's last block alone for the prompt cache; returns the SDK's version.
    RuntimeError naming each difference otherwise."""
    import anthropic
    import httpx

    model = answer.MODELS[0]
    replies = [
        {"id": "msg_self_test_1", "type": "message", "role": "assistant", "model": model,
         "stop_reason": "tool_use", "stop_sequence": None,
         "content": [{"type": "thinking", "thinking": "", "signature": "c2VsZi10ZXN0LTE="},
                     {"type": "tool_use", "id": "toolu_self_test", "name": agent.TOOL["name"],
                      "input": {"command": "head -n 3 git-log.txt"}}],
         "usage": {"input_tokens": 1, "output_tokens": 1}},
        {"id": "msg_self_test_2", "type": "message", "role": "assistant", "model": model,
         "stop_reason": "end_turn", "stop_sequence": None,
         "content": [{"type": "thinking", "thinking": "", "signature": "c2VsZi10ZXN0LTI="},
                     {"type": "text", "text": '{"answer": "self-test"}'}],
         "usage": {"input_tokens": 1, "output_tokens": 1}},
    ]
    printed = {"exit_code": 0, "stdout": "self-test\n", "stderr": "", "truncated": False,
               "timed_out": False}
    sent, commands = [], []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json=replies[min(len(sent), len(replies)) - 1])

    class StandIn:
        async def run(self, command: str) -> dict:
            commands.append(command)
            return dict(printed)

    with tempfile.TemporaryDirectory() as scratch:
        key_file = Path(scratch) / "key"
        key_file.write_text("self-test-key\n", encoding="ascii")
        client = llm.make_async_client(str(key_file), max_retries=0,
                                       transport=httpx.MockTransport(respond))

        async def go() -> dict:
            try:
                return await agent.answer_with_commands(client, model, QUESTION, None,
                                                        sandbox=StandIn())
            finally:
                await client.close()

        try:
            result = asyncio.run(go())
        except Exception as error:  # noqa: BLE001 - the self-test reports it as a failure
            raise RuntimeError(f"the answer raised {type(error).__name__}: {error}") from None
    # The registered settings and the tool, written from their definitions and not from
    # agent.settings, so that a turn that drifts from them is found.
    registered = {**answer.answering_settings(model, system=agent.SYSTEM_PROMPT),
                  "tools": [agent.TOOL], "tool_choice": agent.TOOL_CHOICE}
    question = {"role": "user",
                "content": [{"type": "text", "text": answer.user_message(QUESTION, None)}]}
    transcripts = [[question],
                   [question, {"role": "assistant", "content": replies[0]["content"]},
                    {"role": "user", "content": [{"type": "tool_result",
                                                  "tool_use_id": "toolu_self_test",
                                                  "content": agent.render(printed)}]}]]
    problems = [] if len(sent) == 2 else [f"the answer made {len(sent)} calls, not 2"]
    for n, (request, transcript) in enumerate(zip(sent, transcripts, strict=False), 1):
        body = json.loads(request.content)
        if str(request.url) != f"{llm.BASE_URL}/v1/messages":
            problems.append(f"turn {n} goes to {request.url}")
        if "anthropic-beta" in request.headers:
            problems.append(f"turn {n} sends an anthropic-beta header")
        differing = sorted(name for name in (set(body) | set(registered)) - {"messages"}
                           if body.get(name) != registered.get(name))
        if differing:
            problems.append(f"turn {n} differs from the registered settings and tool in "
                            f"{', '.join(differing)}")
        messages = body.get("messages") or [{"content": [{}]}]
        if (marker_count(body) != 1
                or messages[-1]["content"][-1].get("cache_control") != agent.CACHE):
            problems.append(f"turn {n} does not mark its last block, and it alone, for the "
                            "prompt cache")
        if unmarked(messages) != transcript:
            problems.append(f"turn {n}'s messages are not the question, then each reply as it "
                            "came and its results")
    if commands != ["head -n 3 git-log.txt"]:
        problems.append(f"the sandbox was given {commands!r}")
    if (result["answer"], result["error"], result["tool_calls"], result["thinking_blocks"]) != (
            "self-test", None, 1, 2):
        problems.append(f"the answer was read as {result!r}")
    if problems:
        raise RuntimeError("; ".join(problems))
    return anthropic.__version__


def self_test() -> int:
    """``check_loop``: prints what it found and returns 0, or the differences and 1."""
    try:
        sdk = check_loop()
    except RuntimeError as error:
        print(f"grep at D self-test: FAIL: {error}", file=sys.stderr)
        return 1
    print(f"grep at D self-test: anthropic {sdk}; each turn of an answer sends the registered "
          "settings with the run tool and tool_choice auto, the second sends the first reply back "
          "as it came, and each marks its last block alone for the prompt cache")
    return 0
