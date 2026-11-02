"""The Anthropic client that every adapter builds, for the construction model and the answering
model alike (PREREGISTRATION.md section 11).

``make_async_client`` returns an ``anthropic.AsyncAnthropic`` that:

- sends every request to ``https://api.anthropic.com``, given in code, so that it never reads
  ``ANTHROPIC_BASE_URL`` (the SDK reads that variable only when no base URL is given);
- authenticates with the key read from one file, by default the Docker secret
  ``/run/secrets/anthropic_api_key``; given a key, the SDK reads no credential from the
  environment, so neither ``ANTHROPIC_API_KEY`` nor ``ANTHROPIC_AUTH_TOKEN`` is ever sent;
- reaches the API through the SDK's own httpx client, which honours ``HTTPS_PROXY`` and
  ``NO_PROXY`` from the container's environment: inside the run environment that is the egress
  proxy, the one route out;
- retries what the SDK retries (connection errors, 408, 409, 429 and 5xx) up to ``max_retries``
  times, and gives each request ``timeout_s`` seconds.

The key is never printed, logged or put in an error message: ``read_key`` refuses a file that
cannot be read, or does not hold one line of printable ASCII, with a ``KeyFileError`` that names
the file and why and never what it holds, and ``redact`` takes the key out of any text an adapter
writes, such as an exception's message.

``anthropic`` is imported only inside the functions that need it, so the module imports where the
SDK is not installed.
"""

from __future__ import annotations

import re
from pathlib import Path

KEY_FILE = "/run/secrets/anthropic_api_key"
BASE_URL = "https://api.anthropic.com"
# An API key: one run of printable ASCII, with no space or control character.
_KEY = re.compile(r"[\x21-\x7e]+")
REDACTED = "[redacted]"


class KeyFileError(ValueError):
    """The key file cannot be read, or does not hold one key: the message names the file and why,
    and never what the file holds."""


def read_key(key_file: str = KEY_FILE) -> str:
    """The API key that ``key_file`` holds, without the line break after it. KeyFileError when the
    file cannot be read (missing, a directory, or not readable by the user that reads it) or does
    not hold exactly one line of printable ASCII."""
    try:
        raw = Path(key_file).read_bytes()
    except OSError as error:
        raise KeyFileError(f"{key_file} cannot be read "
                           f"({error.strerror or type(error).__name__})") from None
    try:
        key = raw.decode("ascii").strip()
    except UnicodeDecodeError:
        key = ""
    if not _KEY.fullmatch(key):
        raise KeyFileError(f"{key_file} does not hold an API key (one line of printable ASCII "
                           "with no space)")
    return key


def redact(text: str, key_file: str = KEY_FILE) -> str:
    """``text`` with the key that ``key_file`` holds replaced by ``[redacted]``; ``text`` as it is
    when the file is missing or holds no key."""
    try:
        key = read_key(key_file)
    except (OSError, ValueError):
        return text
    return text.replace(key, REDACTED)


def omit_sentinel():
    """The SDK's sentinel for a parameter that a request leaves out: ``anthropic.omit`` where the
    installed SDK has it, else ``anthropic.NOT_GIVEN``. Both are falsy, so test for them by
    identity, never by truth."""
    import anthropic

    return anthropic.omit if hasattr(anthropic, "omit") else anthropic.NOT_GIVEN


def make_async_client(key_file: str = KEY_FILE, *, max_retries: int = 8, timeout_s: float = 600.0,
                      transport=None):
    """The ``anthropic.AsyncAnthropic`` that every adapter uses (the module's description).

    ``transport`` is for tests only: an httpx transport, such as an ``httpx.MockTransport``,
    that receives every request instead of the network. With it, httpx applies no proxy."""
    import anthropic

    key = read_key(key_file)
    options: dict = {"timeout": timeout_s}
    if transport is not None:
        options["transport"] = transport
    # The SDK's own client keeps its connection limits and keep-alive settings, and builds its
    # proxy mounts from HTTPS_PROXY and NO_PROXY.
    http_client = anthropic.DefaultAsyncHttpxClient(**options)
    return anthropic.AsyncAnthropic(api_key=key, base_url=BASE_URL, max_retries=max_retries,
                                    timeout=timeout_s, http_client=http_client)
