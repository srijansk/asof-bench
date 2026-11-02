"""The reader's answering run, inside the run environment (systems/README.md): one answering call
(``asofbench_systems.answer``) for each request of a requests file, and one response for each,
keyed as the request is. The reader image runs the single-call entrants, closed book first.

    answer_requests --requests /input/requests.jsonl --out /output/responses.jsonl \\
        --model claude-haiku-5-5|claude-sonnet-5-5 [--resume]

A request is one JSON line that the harness wrote (systems/harness.py): ``key``, a random UUID4,
``question``, the item's question text, and, for long context and the oracle, ``context``. The
reader refuses a line with any other field or without those two, a key that is not a UUID4 in its
canonical form or that occurs twice, and a requests file whose directory holds anything else,
since the harness's keys and plan, which hold each item's id, stay on the host. It reads the
requests file one line at a time, so that a file of long contexts is never held whole.

Every invocation first checks that it can read the API key file, then takes the output
directory's lock (``asofbench_systems.run.hold_lock``: ``run.lock``, created with O_EXCL) and holds
it to its end, so that a second invocation on the same directory stops at once; a SIGTERM cancels
the invocation, which then records itself and releases the lock (``cancelled_on_sigterm``). Then
it waits for the egress proxy to accept connections and runs the network check (env/netcheck.py,
with the flags that every system gives it, ``asofbench_systems.run.NETCHECK_ARGS``, no services,
since the reader runs none, and ``--mounts`` its input, output and secret paths), which writes
its report beside the responses, and it answers nothing unless every check passes. Then it answers
each request that has no response, in the file's order and CONCURRENCY at a time, every model call
recorded in the usage log under the request's key, the invocation (its place in run.json's
``invocations``, from 1) and attempt 1.

A response is one JSON line: ``key``, then what ``answer_once`` returns (``answer``,
``tokens_in``, ``tokens_out``, ``wall_s``, ``tool_calls``, ``error`` and ``thinking_blocks``). A
request that was not answered (a refusal, a max_tokens stop, an unparseable reply, or an API error
after the SDK's retries that is the request's own) has a response with a null answer and its
error code, and the harness writes its row with a null answer. An infrastructure error (after the
SDK's retries, the API unreachable, rate-limited, overloaded or failing, or refusing the key)
stops the run with exit status 2 and writes no response for its request: the requests under way
finish and keep their responses, no other starts, and a resumed run answers every request that has
none, so that an outage never gives a request a null answer. ``--resume`` checks that the requests
file, the responses file and the model are those that the run began with, and drops the unfinished
last line that an interrupted invocation can leave in the responses or in the usage log.

It writes, beside the responses file, and none of these files holds an item's id or a local path:

- ``run.json``, written atomically: the requests file's name and SHA-256, the responses file's
  name and the model, recorded when the run begins, and each invocation's command (``answer``),
  start, whether it resumed, the image IDs it ran with (``ASOFBENCH_IMAGE_IDS``, given by the
  launcher), its elapsed time, its exit status, what stopped it and the class of the exception
  that crashed it;
- ``netcheck.json``, the network check's report (an earlier invocation's is kept as
  ``netcheck-<n>.json``);
- ``usage.jsonl``, one line per model call (``asofbench_systems.usage``), under the request's key;
- the responses file.

Exit status (``asofbench_systems.run``): 0 when every request has a response with an answer; 1
when every request has a response and some of them have an error code; 2 when it is refused
before it starts (another invocation holds the lock, a key file it cannot read, a bad argument or
requests file, an output directory that holds a run without ``--resume``, a resume with another
requests file, responses file or model), when the network check fails, or when an infrastructure
error stops it; 4 when it crashes, which run.json records with the exception's class.

``answer_requests --self-test``, which the image's build runs with no network, checks that the
installed packages are those that the lock pins and that an answering call through the installed
SDK sends the registered settings and nothing else, so that an image that builds is one that
answers as registered.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import traceback
from collections import Counter
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path

from asofbench_systems import answer, llm, usage
from asofbench_systems import run as plumbing
from asofbench_systems.run import (
    EXIT_CRASHED,
    EXIT_DONE,
    EXIT_FAILED,
    EXIT_STOPPED,
    NETCHECK,
    NETCHECK_ARGS,
    RUN,
    Refused,
    Stop,
    cancelled_on_sigterm,
    drop_unfinished_line,
    hold_lock,
    new_invocation,
    read_jsonl,
    recorded,
)

USAGE = "usage.jsonl"
# A request's fields, of which the first two are required: the context is long context's and the
# oracle's. A response's fields, in the order the reader writes them: the key, then answer_once's.
REQUEST_FIELDS = ("key", "question", "context")
RESPONSE_FIELDS = ("key", *answer.RESULT_FIELDS)
# A request's key, as the harness writes it: a random UUID4, in canonical form.
KEY = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")
# The requests under way at once.
CONCURRENCY = 8
# The client's retries of a call (the SDK's own: connection errors, 408, 409, 429 and 5xx) and its
# timeout for each request, in seconds, which the egress proxy's idle timeout (660 s) outlasts.
MAX_RETRIES, TIMEOUT_S = 8, 600.0
# The images a run records, as the launcher gives them in ASOFBENCH_IMAGE_IDS.
IMAGE_NAMES = ("reader", "egress")
# Where the reader's refusals point for what to do next.
README = "systems/README.md"
# The network check, as systems/README.md runs it: the shared flags, no services, since the reader
# runs none, and the reader's mounts. The proxy is waited for first: it starts with the reader.
NETCHECK_SCRIPT = Path(__file__).resolve().parents[1] / "env" / "netcheck.py"
SERVICES: tuple[tuple[str, int], ...] = ()
MOUNTS = ("/input", "/output", "/run/secrets")
PROXY = ("egress", 3128)
PROXY_WAIT_S, PROXY_RETRY_S = 60.0, 1.0
# The lock the image installs (reader/requirements.txt), which the self-test checks.
REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements.txt"
PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)[ \t]*(?:;([^\\\n]*))?", re.M)
# How much of an exception's message a stop's reason keeps.
ERROR_CHARS = 300
CHUNK = 1 << 20


def parse_image_ids(text: str | None) -> dict[str, str]:
    """The image IDs of the reader and the egress proxy, as ``name=sha256:...`` entries;
    Refused otherwise (``asofbench_systems.run.parse_image_ids``)."""
    return plumbing.parse_image_ids(text, IMAGE_NAMES, README)


def check_key_file(key_file: str) -> None:
    """Refused unless the API key file can be read and holds a key. The refusal names the file and
    why, never what it holds: on a Linux host the secret keeps the host file's owner and mode,
    which the container's user, uid 10001, must be able to read."""
    try:
        llm.read_key(key_file)
    except llm.KeyFileError as error:
        raise Refused(f"the API key file {error}: it reaches the container as the Docker secret, "
                      "which the container's user, uid 10001, must be able to read "
                      f"({README})") from None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def append_line(path: Path, line: dict) -> None:
    """Appends ``line`` with one write of the whole line, so that lines never interleave."""
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, ensure_ascii=False) + "\n")


def error_text(error: BaseException, key_file: str) -> str:
    """An exception's class and the first ERROR_CHARS characters of its message, the key taken
    out first."""
    message = llm.redact(str(error), key_file)[:ERROR_CHARS]
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


def request_problem(request) -> str | None:
    """Why ``request`` is not a request (the module's description), or None."""
    if not isinstance(request, dict):
        return "is not a JSON object"
    if extra := sorted(set(request) - set(REQUEST_FIELDS)):
        return f"holds fields that a request never holds: {', '.join(extra)}"
    if not isinstance(request.get("key"), str) or not KEY.fullmatch(request["key"]):
        return "has no key that is a UUID4"
    if not isinstance(request.get("question"), str) or not request["question"]:
        return "has no question"
    if "context" in request and not isinstance(request["context"], str):
        return "has a context that is not text"
    return None


def requests_of(path: Path) -> Iterator[dict]:
    """The requests of the file at ``path``, one at a time, in its order; Refused for a line that
    is not a request."""
    with open(path, "rb") as handle:
        for number, raw in enumerate(handle, 1):
            try:
                request = json.loads(raw.decode("utf-8"))
            except ValueError:
                raise Refused(f"{path.name}: line {number} is not JSON") from None
            if problem := request_problem(request):
                raise Refused(f"{path.name}: line {number} {problem}")
            yield request


def scan_requests(path: Path) -> list[str]:
    """The keys of the requests file at ``path``, in its order, after checking that the file is
    alone in its directory and that every line is a request with a key of its own; Refused
    otherwise."""
    if not path.is_file():
        raise Refused(f"{path} is not a file")
    if others := sorted(entry.name for entry in path.parent.iterdir() if entry.name != path.name):
        shown = ", ".join(others[:5]) + (f" and {len(others) - 5} more" if len(others) > 5 else "")
        raise Refused(f"the requests file's directory holds other entries ({shown}): the reader "
                      "is given its requests and nothing else, so give it a directory that holds "
                      f"the requests file alone, as the harness writes it ({README})")
    keys, seen = [], set()
    for number, request in enumerate(requests_of(path), 1):
        if request["key"] in seen:
            raise Refused(f"{path.name}: line {number}'s key occurs twice")
        seen.add(request["key"])
        keys.append(request["key"])
    if not keys:
        raise Refused(f"{path.name} holds no request")
    return keys


def check_responses(responses: list[dict], keys: set[str], name: str) -> None:
    """Refused unless every line of ``responses`` is the response to a request of ``keys`` and
    each request has at most one."""
    seen = set()
    for number, response in enumerate(responses, 1):
        if not isinstance(response, dict) or set(response) != set(RESPONSE_FIELDS):
            raise Refused(f"{name}: line {number} is not a response")
        if response["key"] not in keys:
            raise Refused(f"{name}: line {number} answers a key that the requests file does not "
                          "hold")
        if response["key"] in seen:
            raise Refused(f"{name}: line {number} answers a key that has a response already")
        seen.add(response["key"])


def open_client(usage_log: usage.UsageLog, key_file: str = llm.KEY_FILE):
    """The answering client: the shared Anthropic client, with the reader's retries and timeout,
    every call recorded in ``usage_log``."""
    return usage.instrument(llm.make_async_client(key_file, max_retries=MAX_RETRIES,
                                                  timeout_s=TIMEOUT_S), usage_log)


def netcheck_preflight(out_dir: Path) -> int:
    """Waits for the egress proxy to accept connections, then runs the network check, which writes
    its report to ``out_dir/netcheck.json``; returns the check's exit status
    (``asofbench_systems.run.netcheck_preflight`` with the reader's mounts and no services)."""
    plumbing.wait_for_services((PROXY,), PROXY_WAIT_S, PROXY_RETRY_S)
    return plumbing.netcheck_preflight(out_dir, script=NETCHECK_SCRIPT, args=NETCHECK_ARGS,
                                       services=SERVICES, mounts=MOUNTS, wait_s=0.0,
                                       retry_s=PROXY_RETRY_S)


async def answer_pending(client, model: str, requests: Path, answered: set[str], stop: Stop,
                         out: Path, ask: Callable[..., Awaitable[dict]], key_file: str,
                         progress: Callable[[str], None], *, invocation: int, total: int,
                         concurrency: int) -> None:
    """Answers each request of ``requests`` that ``answered`` does not hold, in the file's order,
    ``concurrency`` at a time, and appends its response to ``out``, until an infrastructure error
    requests a stop (the module's description)."""
    pending = (request for request in requests_of(requests) if request["key"] not in answered)

    async def worker() -> None:
        for request in pending:
            if stop.code is not None:
                return
            key = request["key"]
            with usage.attributed(key, invocation, 1):
                try:
                    response = await ask(client, model, request["question"],
                                         request.get("context"))
                except Exception as raised:  # noqa: BLE001 - classified below, and never lost
                    if infrastructure := plumbing.infrastructure_error(raised,
                                                                       plumbing.INFRASTRUCTURE):
                        stop.request(EXIT_STOPPED, f"{key}: {infrastructure}: "
                                     f"{error_text(raised, key_file)}",
                                     {"key": key, "error": infrastructure})
                        progress(f"answer {key}: stopped by an infrastructure error "
                                 f"({infrastructure}); no response written")
                        return
                    raise
            append_line(out, {"key": key, **response})
            answered.add(key)
            progress(f"answer {len(answered)}/{total} {key}: "
                     f"{response['error'] or 'answered'} in {response['wall_s']:.1f} s")

    try:
        async with asyncio.TaskGroup() as tasks:
            for _ in range(concurrency):
                tasks.create_task(worker())
    finally:
        pending.close()


async def run(requests: Path, out: Path, model: str, *, resume: bool, images: dict[str, str],
              open_client: Callable = open_client,
              ask: Callable[..., Awaitable[dict]] = answer.answer_once,
              preflight: Callable[[Path], int] | None = netcheck_preflight,
              key_file: str = llm.KEY_FILE, concurrency: int = CONCURRENCY,
              progress: Callable[[str], None] = print) -> int:
    """One invocation of an answering run (the module's description); Refused when it cannot
    start."""
    if model not in answer.MODELS:
        raise Refused(f"--model {model!r} is not an answering model ({', '.join(answer.MODELS)})")
    if not out.parent.is_dir():
        raise Refused(f"{out.parent} is not a directory")
    check_key_file(key_file)
    with hold_lock(out.parent, "answer", readme=README), cancelled_on_sigterm():
        return await _run(requests, out, model, resume=resume, images=images,
                          open_client=open_client, ask=ask, preflight=preflight,
                          key_file=key_file, concurrency=concurrency, progress=progress)


async def _run(requests: Path, out: Path, model: str, *, resume: bool, images: dict[str, str],
               open_client: Callable, ask: Callable[..., Awaitable[dict]], preflight,
               key_file: str, concurrency: int, progress: Callable[[str], None]) -> int:
    begun = time.monotonic()
    out_dir = out.parent
    usage_path, run_path = out_dir / USAGE, out_dir / RUN
    keys = scan_requests(requests)
    requests_sha256 = sha256_file(requests)
    if resume:
        if not run_path.exists():
            raise Refused(f"{out_dir} holds no run to resume (no {RUN}): start one without "
                          "--resume")
        record = json.loads(run_path.read_text(encoding="utf-8"))
        if (record["requests_sha256"], record["responses"], record["model"]) != (
                requests_sha256, out.name, model):
            raise Refused(f"the run began with the requests file {record['requests']} (SHA-256 "
                          f"{record['requests_sha256']}), the responses file {record['responses']} "
                          f"and the model {record['model']}, not with {requests.name} (SHA-256 "
                          f"{requests_sha256}), {out.name} and {model}")
        for path in (out, usage_path):
            if dropped := drop_unfinished_line(path):
                progress(f"answer: dropped the unfinished last line of {path.name} ({dropped} "
                         "bytes), left by an interrupted invocation")
    else:
        if any(path.exists() for path in (out, usage_path, run_path)):
            raise Refused(f"{out_dir} already holds a run: resume it with --resume, or give a new "
                          "output directory")
        record = {"requests": requests.name, "requests_sha256": requests_sha256,
                  "responses": out.name, "model": model, "invocations": []}
    responses = read_jsonl(out)
    check_responses(responses, set(keys), out.name)
    answered = {response["key"] for response in responses}
    # This invocation's place in run.json, which the lock keeps from changing meanwhile.
    number = len(record["invocations"]) + 1
    invocation = new_invocation("answer", images, resume=resume)
    with recorded(record, invocation, run_path, begun):
        if preflight is not None and preflight(out_dir) != 0:
            progress(f"answer: the network check failed (see {NETCHECK}); nothing was answered")
            invocation["exit"], invocation["stop"] = EXIT_STOPPED, {"error": "network check"}
            return EXIT_STOPPED
        stop = Stop()
        progress(f"answer: {len(keys) - len(answered)} of the {len(keys)} requests to answer with "
                 f"{model}, {concurrency} at a time")
        client = open_client(usage.UsageLog(str(usage_path)), key_file)
        try:
            await answer_pending(client, model, requests, answered, stop, out, ask, key_file,
                                 progress, invocation=number, total=len(keys),
                                 concurrency=concurrency)
        finally:
            await client.close()
        responses = read_jsonl(out)
        errors = Counter(response["error"] for response in responses if response["error"])
        code = (stop.code if stop.code is not None else EXIT_FAILED if errors else EXIT_DONE)
        invocation["exit"], invocation["stop"] = code, stop.record
        if stop.reason is not None:
            progress(f"answer: stopped (exit {code}): {stop.reason}")
        thinking = sum(1 for response in responses if response["thinking_blocks"])
        progress(f"answer: {len(responses)} of the {len(keys)} requests have a response, "
                 f"{sum(errors.values())} with an error code"
                 + "".join(f", {name} {n}" for name, n in sorted(errors.items()))
                 + f"; {thinking} held a thinking block; "
                 f"{sum(response['tokens_in'] for response in responses)} input and "
                 f"{sum(response['tokens_out'] for response in responses)} output tokens")
        return code


def check_installation(requirements: Path = REQUIREMENTS) -> dict[str, str]:
    """Checks that every package that ``requirements`` pins for this platform is installed at its
    pinned version, and that the network check is where the run looks for it; returns the
    installed versions. RuntimeError naming each difference otherwise."""
    import importlib.metadata

    from packaging.markers import Marker

    problems, installed = [], {}
    if not NETCHECK_SCRIPT.is_file():
        problems.append(f"the network check is not at {NETCHECK_SCRIPT}")
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

    question = "Is this the self-test?"
    with tempfile.TemporaryDirectory() as scratch:
        key_file = Path(scratch) / "key"
        key_file.write_text("self-test-key\n", encoding="ascii")
        client = llm.make_async_client(str(key_file), max_retries=0,
                                       transport=httpx.MockTransport(respond))

        async def call() -> dict:
            try:
                return await answer.answer_once(client, answer.MODELS[0], question, None)
            finally:
                await client.close()

        result = asyncio.run(call())
    problems = []
    (request,) = sent
    if str(request.url) != f"{llm.BASE_URL}/v1/messages":
        problems.append(f"the call goes to {request.url}")
    body = json.loads(request.content)
    expected = {**answer.answering_settings(answer.MODELS[0]),
                "messages": [{"role": "user", "content": answer.user_message(question, None)}]}
    if body != expected:
        differing = sorted(name for name in set(body) | set(expected)
                           if body.get(name) != expected.get(name))
        problems.append(f"the call's {', '.join(differing)} differ from the registered settings")
    if "anthropic-beta" in request.headers:
        problems.append("the call sends an anthropic-beta header")
    if (result["answer"], result["error"], result["thinking_blocks"]) != ("self-test", None, 1):
        problems.append(f"the reply was read as {result!r}")
    if problems:
        raise RuntimeError("; ".join(problems))
    return anthropic.__version__


def self_test(requirements: Path = REQUIREMENTS) -> int:
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


def crashed(error: BaseException, key_file: str) -> int:
    """Prints a crash, the key taken out, and returns its exit status."""
    text = "".join(traceback.format_exception(error))
    print(f"answer_requests: crashed ({type(error).__name__}); run.json records exit "
          f"{EXIT_CRASHED}\n{llm.redact(text, key_file)}", file=sys.stderr)
    return EXIT_CRASHED


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="answer_requests", description="Answers a requests "
                                     "file with the answering model, inside the run environment.",
                                     allow_abbrev=False)
    parser.add_argument("--requests", type=Path, help="the run's requests file")
    parser.add_argument("--out", type=Path, help="the responses file, in the run's output "
                        "directory")
    parser.add_argument("--model", choices=answer.MODELS, help="the answering model")
    parser.add_argument("--resume", action="store_true",
                        help="answer the requests that have no response yet")
    parser.add_argument("--self-test", action="store_true",
                        help="check the installation and the answering call, with no network")
    args = parser.parse_args(argv)
    if args.self_test:
        if args.requests or args.out or args.model or args.resume:
            parser.error("--self-test takes no other argument")
        return self_test()
    if missing := [flag for flag, value in (("--requests", args.requests), ("--out", args.out),
                                            ("--model", args.model)) if value is None]:
        parser.error(f"the following arguments are required: {', '.join(missing)}")
    try:
        images = parse_image_ids(os.environ.get("ASOFBENCH_IMAGE_IDS"))
        return asyncio.run(run(args.requests, args.out, args.model, resume=args.resume,
                               images=images))
    except Refused as refusal:
        print(f"answer_requests: {refusal}", file=sys.stderr)
        return EXIT_STOPPED
    except SystemExit:
        raise
    except BaseException as error:  # noqa: BLE001 - a crash: recorded in run.json, exit 4
        return crashed(error, llm.KEY_FILE)


if __name__ == "__main__":
    sys.exit(main())
