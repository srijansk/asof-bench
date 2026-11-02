"""The log of every model call's usage: one JSON line per call, attributed to what the system was
doing when it made the call, so that a run's tokens can be counted per episode or per item, and a
call that failed placed in the attempt it fell in.

``usage_context``, a ``contextvars.ContextVar``, carries that attribution: an adapter sets it to the
episode id while the episode is ingested (or to the item while it is answered), and every call
made meanwhile, from tasks the system starts too, since a task copies the context it starts in, is
recorded under it. ``usage_attempt`` carries, for a system that attempts an episode more than
once, the invocation of the run that makes the call (its place among the run's invocations, from
1) and the attempt (from 1, within that invocation; a resumed invocation numbers its attempts
afresh, so the invocation is part of the attempt's name). ``attributed`` sets both for a block.
``instrument(client, log)`` wraps ``client.messages.create`` of an Anthropic client so that each
call is recorded in ``log``, a response with ``UsageLog.record`` and a call that raises with
``UsageLog.record_error``:

    {"ts", "context", "invocation", "attempt", "model", "input_tokens", "output_tokens",
     "cache_creation_input_tokens", "cache_read_input_tokens", "stop_reason", "error"}

``invocation`` and ``attempt`` are null where nothing set them. For a response, ``model`` is the
model that answered, the four counts are its ``usage`` (a missing count written as 0) and
``error`` is null. For a call that raises (after the SDK's own retries, which are inside one
call), ``model`` is the model asked for, the counts are 0, ``stop_reason`` is null and ``error``
is the exception's class name, nothing more: the system may catch the exception and carry on, and
the line keeps the failure visible. Only ``messages.create`` without streaming is recorded: a
streaming call through the wrapper is refused, since its usage is not known when it returns.

It uses only the standard library.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from contextvars import ContextVar
from datetime import UTC, datetime

usage_context: ContextVar[str | None] = ContextVar("usage_context", default=None)
usage_attempt: ContextVar[tuple[int | None, int | None]] = ContextVar("usage_attempt",
                                                                      default=(None, None))

COUNTS = ("input_tokens", "output_tokens", "cache_creation_input_tokens",
          "cache_read_input_tokens")


@contextlib.contextmanager
def attributed(context: str | None, invocation: int | None = None,
               attempt: int | None = None) -> Iterator[None]:
    """Attributes the calls made inside the block, from the tasks it starts too, to ``context``
    and to the invocation and the attempt that make them."""
    context_token = usage_context.set(context)
    attempt_token = usage_attempt.set((invocation, attempt))
    try:
        yield
    finally:
        usage_attempt.reset(attempt_token)
        usage_context.reset(context_token)


def attribution() -> dict:
    """The current attribution of a call: its ``context``, ``invocation`` and ``attempt``."""
    invocation, attempt = usage_attempt.get()
    return {"context": usage_context.get(), "invocation": invocation, "attempt": attempt}


def _count(response_usage, name: str) -> int:
    value = (response_usage.get(name) if isinstance(response_usage, dict)
             else getattr(response_usage, name, None))
    return int(value or 0)


class UsageLog:
    """Appends one JSON line per call to the file at ``path``."""

    def __init__(self, path: str) -> None:
        self.path = path

    def _append(self, line: dict) -> None:
        # One write of one whole line, so that lines from concurrent tasks never interleave.
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(line) + "\n")

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    @staticmethod
    def _attributed(attribution: dict) -> dict:
        return {name: attribution[name] for name in ("context", "invocation", "attempt")}

    def record(self, attribution: dict, model: str, response_usage,
               stop_reason: str | None) -> None:
        self._append({"ts": self._now(), **self._attributed(attribution), "model": model,
                      **{name: _count(response_usage, name) for name in COUNTS},
                      "stop_reason": stop_reason, "error": None})

    def record_error(self, attribution: dict, model: str | None, error: str) -> None:
        self._append({"ts": self._now(), **self._attributed(attribution), "model": model,
                      **{name: 0 for name in COUNTS}, "stop_reason": None, "error": error})


def instrument(client, log: UsageLog):
    """Wraps ``client.messages.create`` so that every call is recorded in ``log`` under the
    current attribution; returns ``client``."""
    create = client.messages.create

    async def recorded(*args, **kwargs):
        if kwargs.get("stream"):
            raise NotImplementedError("the usage log records calls without streaming only")
        try:
            response = await create(*args, **kwargs)
        except Exception as error:
            log.record_error(attribution(), kwargs.get("model"), type(error).__name__)
            raise
        log.record(attribution(), getattr(response, "model", None) or kwargs.get("model"),
                   getattr(response, "usage", None), getattr(response, "stop_reason", None))
        return response

    client.messages.create = recorded
    return client
