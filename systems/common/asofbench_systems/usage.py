"""The log of every model call's usage: one JSON line per call, attributed to what the system was
doing when it made the call, so that a run's tokens can be counted per episode or per item.

``usage_context``, a ``contextvars.ContextVar``, carries that attribution: an adapter sets it to the
episode id while the episode is ingested (or to the item while it is answered), and every call
made meanwhile, from tasks the system starts too, since a task copies the context it starts in, is
recorded under it. ``instrument(client, log)`` wraps ``client.messages.create`` of an Anthropic
client so that each call is recorded in ``log``, a response with ``UsageLog.record`` and a call
that raises with ``UsageLog.record_error``:

    {"ts", "context", "model", "input_tokens", "output_tokens", "cache_creation_input_tokens",
     "cache_read_input_tokens", "stop_reason", "error"}

For a response, ``model`` is the model that answered, the four counts are its ``usage`` (a
missing count written as 0) and ``error`` is null. For a call that raises (after the SDK's own
retries, which are inside one call), ``model`` is the model asked for, the counts are 0,
``stop_reason`` is null and ``error`` is the exception's class name, nothing more: the system may
catch the exception and carry on, and the line keeps the failure visible. Only
``messages.create`` without streaming is recorded: a streaming call through the wrapper is
refused, since its usage is not known when it returns.

It uses only the standard library.
"""

from __future__ import annotations

import json
from contextvars import ContextVar
from datetime import UTC, datetime

usage_context: ContextVar[str | None] = ContextVar("usage_context", default=None)

COUNTS = ("input_tokens", "output_tokens", "cache_creation_input_tokens",
          "cache_read_input_tokens")


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

    def record(self, context: str | None, model: str, response_usage,
               stop_reason: str | None) -> None:
        self._append({"ts": self._now(), "context": context, "model": model,
                      **{name: _count(response_usage, name) for name in COUNTS},
                      "stop_reason": stop_reason, "error": None})

    def record_error(self, context: str | None, model: str | None, error: str) -> None:
        self._append({"ts": self._now(), "context": context, "model": model,
                      **{name: 0 for name in COUNTS}, "stop_reason": None, "error": error})


def instrument(client, log: UsageLog):
    """Wraps ``client.messages.create`` so that every call is recorded in ``log`` under the
    current ``usage_context``; returns ``client``."""
    create = client.messages.create

    async def recorded(*args, **kwargs):
        if kwargs.get("stream"):
            raise NotImplementedError("the usage log records calls without streaming only")
        try:
            response = await create(*args, **kwargs)
        except Exception as error:
            log.record_error(usage_context.get(), kwargs.get("model"), type(error).__name__)
            raise
        log.record(usage_context.get(), getattr(response, "model", None) or kwargs.get("model"),
                   getattr(response, "usage", None), getattr(response, "stop_reason", None))
        return response

    client.messages.create = recorded
    return client
