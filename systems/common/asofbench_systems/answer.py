"""The answering call of every single-call entrant (PREREGISTRATION.md section 11): closed book,
long context and the oracle each answer a request with one call to the answering model, which
receives the question and, for long context and the oracle, the context that section 11 gives
them, and replies with the answer in a JSON object.

``answering_settings(model)`` is the one definition of what every answering call sends besides its
messages: the model (``MODELS``: claude-haiku-5-5 for every run, claude-sonnet-5-5 in the
sensitivity check), ``max_tokens`` 32000, ``thinking={"type": "adaptive"}``, ``output_config``
with effort ``medium`` and the ``json_schema`` format of ``ANSWER_SCHEMA``, and the system prompt,
``SYSTEM_PROMPT``; it sends no ``temperature``, ``top_p`` or ``top_k``, no prefill and no tool.
An agent's loop, such as grep at D's (grep-at-d/agent.py), sends the same settings on every turn
with its own system prompt and its tool added, reads each response's costs with ``counts`` and its
final reply with ``read_answer``, and returns its ``result`` with the commands it ran.
``user_message`` is the call's one user turn: ``Question: {question}``, or with a context
``Context:\\n\\n{context}\\n\\nQuestion: {question}``. These strings are model-facing and hold no
benchmark-derived value: no item id, gold value, family name or stratum.

``answer_once(client, model, question, context)`` makes the call and returns, in this order:

- ``answer``: the ``answer`` field of the JSON object that the reply's text holds, as it came
  (the harness shapes it by the item's answer type), or null when the request was not answered;
- ``tokens_in``: the response's input tokens, with those written to and read from the prompt
  cache; ``tokens_out``: its output tokens (0 and 0 when the call raised);
- ``wall_s``: the call's wall time in seconds, the SDK's retries included;
- ``tool_calls``: 0, since a single call runs no tool (an agent gives the commands it ran);
- ``error``: null, or the short code of a request that was not answered (``ERRORS``):
  ``refusal`` and ``max_tokens`` for a response with that stop reason, whatever its text holds,
  ``unparseable`` for a reply whose text is not a JSON object with an ``answer`` field, and
  ``api_error`` for an API error, after the SDK's retries, that is the request's own (such as a
  400 for a prompt over the context window);
- ``thinking_blocks``: the number of thinking blocks in the response. Adaptive thinking lets the
  model answer without thinking, so this is a record of what it did and not a check.

An infrastructure error (``asofbench_systems.run.INFRASTRUCTURE``: after the SDK's retries, the
API unreachable, rate-limited, overloaded or failing, or refusing the key) is raised, so that the
run stops and a resumed run answers the request again: an outage never gives a request a null
answer.

The reader writes one response per request, ``RESPONSE_FIELDS``: the request's key, then what
``answer_once`` returns; ``responses_by_key`` reads a run's responses back, for the reader when it
resumes and for the harness when it writes the rows, and ``responses_name`` checks the name of
the responses file that a run's run.json gives, for the harness and for grep at D's run_dates.py,
which open that file.

``anthropic`` is imported only inside ``answer_once``, so the module imports where the SDK is not
installed.
"""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Collection
from pathlib import Path

from asofbench_systems.run import INFRASTRUCTURE, RUN, Refused, infrastructure_error

# The answering models (section 11): claude-haiku-5-5 answers in every run, and claude-sonnet-5-5
# again on the pilot items as the sensitivity check, each run with one of them.
MODELS = ("claude-haiku-5-5", "claude-sonnet-5-5")
MAX_TOKENS = 32000
EFFORT = "medium"
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"anyOf": [
        {"type": "string"}, {"type": "integer"},
        {"type": "array", "items": {"anyOf": [{"type": "integer"}, {"type": "string"}]}}]}},
    "required": ["answer"],
    "additionalProperties": False,
}
SYSTEM_PROMPT = ("Answer the question that follows. Reply with a JSON object whose \"answer\" "
                 "field holds your answer in the form the question asks for.")
# What answer_once returns, in this order, and the codes of a request that was not answered,
# which gets a null answer and scores zero.
RESULT_FIELDS = ("answer", "tokens_in", "tokens_out", "wall_s", "tool_calls", "error",
                 "thinking_blocks")
ERRORS = ("refusal", "max_tokens", "api_error", "unparseable")
# A response, one JSON line per request: the request's key, then what answer_once returns.
RESPONSE_FIELDS = ("key", *RESULT_FIELDS)
THINKING_TYPES = ("thinking", "redacted_thinking")
CACHE_COUNTS = ("cache_creation_input_tokens", "cache_read_input_tokens")


def answering_settings(model: str, system: str = SYSTEM_PROMPT) -> dict:
    """The keyword arguments of ``messages.create`` that every answering call sends besides its
    messages, a new copy for every call, so that no caller can change what a later one sends."""
    return {"model": model, "max_tokens": MAX_TOKENS, "thinking": {"type": "adaptive"},
            "output_config": {"effort": EFFORT,
                              "format": {"type": "json_schema",
                                         "schema": copy.deepcopy(ANSWER_SCHEMA)}},
            "system": system}


def user_message(question: str, context: str | None) -> str:
    """The one user turn of a single call: the question, after the context when there is one."""
    if context is None:
        return f"Question: {question}"
    return f"Context:\n\n{context}\n\nQuestion: {question}"


def _count(response_usage, name: str) -> int:
    return int(getattr(response_usage, name, None) or 0)


def counts(response) -> tuple[int, int, int]:
    """A response's input tokens, with those written to and read from the prompt cache; its output
    tokens; and its thinking blocks."""
    response_usage = getattr(response, "usage", None)
    tokens_in = _count(response_usage, "input_tokens") + sum(_count(response_usage, name)
                                                             for name in CACHE_COUNTS)
    thinking = sum(getattr(block, "type", None) in THINKING_TYPES for block in response.content)
    return tokens_in, _count(response_usage, "output_tokens"), thinking


def read_answer(content) -> tuple[object, str | None]:
    """The answer that a response's text blocks hold, read as one JSON object with an ``answer``
    field, and None; or None and ``unparseable``."""
    text = "".join(block.text for block in content if getattr(block, "type", None) == "text")
    # Text that json.loads cannot parse, for whatever reason, is unparseable. Whether a reply
    # nested deeply enough parses depends on the Python that reads it, but no reply that conforms
    # to ANSWER_SCHEMA nests more than two levels, so the reading of one never depends on that.
    try:
        reply = json.loads(text)
    except (ValueError, RecursionError):
        return None, "unparseable"
    if not isinstance(reply, dict) or "answer" not in reply:
        return None, "unparseable"
    return reply["answer"], None


def result(answer, tokens_in: int, tokens_out: int, begun: float, error: str | None,
           thinking_blocks: int, tool_calls: int = 0) -> dict:
    """What an answering function returns (``RESULT_FIELDS``, in that order), its wall time
    measured from ``begun``."""
    return {"answer": answer, "tokens_in": tokens_in, "tokens_out": tokens_out,
            "wall_s": round(time.monotonic() - begun, 3), "tool_calls": tool_calls,
            "error": error, "thinking_blocks": thinking_blocks}


async def answer_once(client, model: str, question: str, context: str | None) -> dict:
    """One answering call with the registered settings, through ``client`` (an
    ``anthropic.AsyncAnthropic``, as ``asofbench_systems.llm.make_async_client`` builds it); the
    module's description gives what it returns and what it raises."""
    import anthropic

    begun = time.monotonic()
    try:
        response = await client.messages.create(
            **answering_settings(model),
            messages=[{"role": "user", "content": user_message(question, context)}])
    except anthropic.APIError as error:
        if infrastructure_error(error, INFRASTRUCTURE):
            raise
        return result(None, 0, 0, begun, "api_error", 0)
    tokens_in, tokens_out, thinking = counts(response)
    if response.stop_reason in ("refusal", "max_tokens"):
        return result(None, tokens_in, tokens_out, begun, response.stop_reason, thinking)
    answer, error = read_answer(response.content)
    return result(answer, tokens_in, tokens_out, begun, error, thinking)


def responses_by_key(responses: list[dict], keys: Collection[str], name: str) -> dict[str, dict]:
    """The responses of a run by key; Refused unless every line of ``responses`` (the file
    ``name``'s, which the refusal names) is the response to a request whose key ``keys`` holds and
    each request has at most one."""
    found: dict[str, dict] = {}
    for number, response in enumerate(responses, 1):
        if (not isinstance(response, dict) or set(response) != set(RESPONSE_FIELDS)
                or not isinstance(response["key"], str)):
            raise Refused(f"{name}: line {number} is not a response")
        if response["key"] not in keys:
            raise Refused(f"{name}: line {number} answers a key that no request of the run holds")
        if response["key"] in found:
            raise Refused(f"{name}: line {number} answers a key that has a response already")
        found[response["key"]] = response
    return found


def responses_name(record: dict, where: str) -> str:
    """The name of the responses file that a run's run.json, ``record``, gives, a file of the
    run's output directory, ``where`` (which the refusal names); Refused for anything else, a
    path or ``..`` among them, so that a reader of the responses never opens a file outside that
    directory."""
    name = record.get("responses")
    if not isinstance(name, str) or Path(name).name != name or name in ("", ".", ".."):
        raise Refused(f"{where}/{RUN} names no responses file in {where}/")
    return name
