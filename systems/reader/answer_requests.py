"""The reader's answering run, inside the run environment (systems/README.md): one answering call
(``asofbench_systems.answer``) for each request of a requests file, and one response for each,
keyed as the request is. The reader image runs the single-call entrants, closed book first.

    answer_requests --requests /input/requests.jsonl --out /output/responses.jsonl \\
        --model claude-haiku-5-5|claude-sonnet-5-5 [--resume] [--api-errors-before N]

Grep at D's agent (grep-at-d/agent.py) answers through this runner in an image of its own,
asofbench/grep-at-d, built FROM this one (grep-at-d/Dockerfile), which holds nothing of grep at
D's: its command, ``grep_at_d``, is ``main`` with the agent's answering function, which runs a
request's commands in the sandbox, its images (its own and the sandbox's beside the reader's and
the proxy's) and its network check (the sandbox as a service); everything below holds for it as
written.

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

A response is one JSON line (``asofbench_systems.answer.RESPONSE_FIELDS``): ``key``, then what
``answer_once`` returns (``answer``, ``tokens_in``, ``tokens_out``, ``wall_s``, ``tool_calls``,
``error`` and ``thinking_blocks``). A request that was not answered (a refusal, a max_tokens stop,
an unparseable reply, or an API error after the SDK's retries that is the request's own) has a
response with a null answer and its error code, and the harness writes its row with a null
answer. Two causes stop the run, and in each the requests under way finish and keep their
responses and no other starts:

- an infrastructure error (after the SDK's retries, the API unreachable, rate-limited, overloaded
  or failing, or refusing the key), with exit status 2 and no response for its request, so that a
  resumed run answers it: an outage never gives a request a null answer;
- MAX_CONSECUTIVE_API_ERRORS ``api_error`` responses in a row, in the order they are written, with
  exit status 3. Each of them is its request's own error and stays, but so many in a row point at
  a cause that every request shares, such as a setting that the API rejects, which would turn
  every request left into one more. A resume answers only the requests that have no response, so
  a run that stopped this way for a cause of the run is discarded and run again in a new run
  directory once the cause is fixed, never resumed under changed settings. A resumed run counts on
  from the ``api_error`` responses at the end of the responses file, as Graphiti's ingestion
  counts on from its log. ``--api-errors-before N`` (0 by default) is the count of ``api_error``
  responses in a row that the runs before this one ended with, for an entrant whose run is made
  of several, as grep at D's is of one run per date (grep-at-d/run_dates.py gives it): the count
  goes on from it, through this run's own responses while every one of them is an ``api_error``,
  so that the stop fires wherever the fifth in a row falls, and run.json's ``stop`` then records
  it as ``api_errors_before``.

``--resume`` checks that the requests file, the responses file and the model are those that the
run began with, drops the unfinished last line that an interrupted invocation can leave in the
responses or in the usage log, and answers every request that has no response.

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
error stops it; 3 when consecutive ``api_error`` responses stop it; 4 when it crashes, which
run.json records with the exception's class.

``answer_requests --self-test``, which the image's build runs with no network, is the image's
self-test (``reader.selftest``): the installed packages are those that the lock pins, and an
answering call through the installed SDK sends the registered settings and nothing else.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Iterator, Sequence
from pathlib import Path

from asofbench_systems import answer, llm, usage
from asofbench_systems import run as plumbing
from asofbench_systems.run import (
    EXIT_DONE,
    EXIT_FAILED,
    EXIT_FAILURES,
    EXIT_STOPPED,
    NETCHECK,
    NETCHECK_ARGS,
    RUN,
    Refused,
    Stop,
    append_line,
    cancelled_on_sigterm,
    crashed,
    drop_unfinished_line,
    error_text,
    hold_lock,
    new_invocation,
    read_jsonl,
    recorded,
    sha256_file,
)

USAGE = "usage.jsonl"
# A request's fields, of which the first two are required: the context is long context's and the
# oracle's.
REQUEST_FIELDS = ("key", "question", "context")
# A request's key, as the harness writes it: a random UUID4, in canonical form.
KEY = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")
# The requests under way at once.
CONCURRENCY = 8
# The api_error responses in a row that stop the run, as Graphiti stops after five consecutive
# failures in a group.
MAX_CONSECUTIVE_API_ERRORS = 5
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


def parse_image_ids(text: str | None, names: Sequence[str] = IMAGE_NAMES) -> dict[str, str]:
    """The image IDs of ``names``, by default the reader's and the egress proxy's, as
    ``name=sha256:...`` entries; Refused otherwise (``asofbench_systems.run.parse_image_ids``)."""
    return plumbing.parse_image_ids(text, names, README)


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


def trailing_api_errors(responses: list[dict], before: int = 0) -> int:
    """The ``api_error`` responses in a row at the end of ``responses``, in their order, counting
    on from ``before``, the ``api_error`` responses in a row that came before them, while every one
    of ``responses`` is an ``api_error``."""
    count = 0
    for response in reversed(responses):
        if response["error"] != "api_error":
            return count
        count += 1
    return count + before


def open_client(usage_log: usage.UsageLog, key_file: str = llm.KEY_FILE):
    """The answering client: the shared Anthropic client, with the reader's retries and timeout,
    every call recorded in ``usage_log``."""
    return usage.instrument(llm.make_async_client(key_file, max_retries=MAX_RETRIES,
                                                  timeout_s=TIMEOUT_S), usage_log)


def netcheck_preflight(out_dir: Path,
                       services: Sequence[tuple[str, int]] = SERVICES) -> int:
    """Waits for the egress proxy and ``services`` to accept connections, then runs the network
    check, which writes its report to ``out_dir/netcheck.json``; returns the check's exit status
    (``asofbench_systems.run.netcheck_preflight`` with the reader's mounts and ``services``: none
    for the reader, which runs none, and the sandbox for grep at D)."""
    plumbing.wait_for_services((PROXY, *services), PROXY_WAIT_S, PROXY_RETRY_S)
    return plumbing.netcheck_preflight(out_dir, script=NETCHECK_SCRIPT, args=NETCHECK_ARGS,
                                       services=services, mounts=MOUNTS, wait_s=0.0,
                                       retry_s=PROXY_RETRY_S)


async def answer_pending(client, model: str, requests: Path, answered: set[str], stop: Stop,
                         out: Path, ask: Callable[..., Awaitable[dict]], key_file: str,
                         progress: Callable[[str], None], *, invocation: int, total: int,
                         concurrency: int, api_errors: int, api_errors_before: int = 0) -> None:
    """Answers each request of ``requests`` that ``answered`` does not hold, in the file's order,
    ``concurrency`` at a time, and appends its response to ``out``, until a stop is requested (the
    module's description); ``api_errors`` is the count of ``api_error`` responses in a row that
    the responses file ends with, the runs before this one's (``api_errors_before``) included."""
    pending = (request for request in requests_of(requests) if request["key"] not in answered)
    in_a_row = api_errors

    async def worker() -> None:
        nonlocal in_a_row
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
            in_a_row = in_a_row + 1 if response["error"] == "api_error" else 0
            progress(f"answer {len(answered)}/{total} {key}: "
                     f"{response['error'] or 'answered'} in {response['wall_s']:.1f} s")
            if in_a_row >= MAX_CONSECUTIVE_API_ERRORS:
                stop.request(EXIT_FAILURES, f"{in_a_row} api_error responses in a row: a cause "
                             "that every request shares, such as a setting that the API "
                             "rejects, would give one to every request left; discard the run "
                             f"and run it again in a new run directory once it is fixed ({README})",
                             {"error": f"{in_a_row} consecutive api_error responses",
                              **({"api_errors_before": api_errors_before}
                                 if api_errors_before else {})})

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
              progress: Callable[[str], None] = print, api_errors_before: int = 0) -> int:
    """One invocation of an answering run (the module's description); Refused when it cannot
    start."""
    if model not in answer.MODELS:
        raise Refused(f"--model {model!r} is not an answering model ({', '.join(answer.MODELS)})")
    if api_errors_before < 0:
        raise Refused(f"api_errors_before {api_errors_before} is not a count")
    if not out.parent.is_dir():
        raise Refused(f"{out.parent} is not a directory")
    plumbing.check_key_file(key_file, README)
    with hold_lock(out.parent, "answer", readme=README), cancelled_on_sigterm():
        return await _run(requests, out, model, resume=resume, images=images,
                          open_client=open_client, ask=ask, preflight=preflight,
                          key_file=key_file, concurrency=concurrency, progress=progress,
                          api_errors_before=api_errors_before)


async def _run(requests: Path, out: Path, model: str, *, resume: bool, images: dict[str, str],
               open_client: Callable, ask: Callable[..., Awaitable[dict]], preflight,
               key_file: str, concurrency: int, progress: Callable[[str], None],
               api_errors_before: int) -> int:
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
    answered = set(answer.responses_by_key(responses, set(keys), out.name))
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
                 f"{model}, {concurrency} at a time"
                 + (f", after {api_errors_before} api_error responses in a row in the runs "
                    "before this one" if api_errors_before else ""))
        client = open_client(usage.UsageLog(str(usage_path)), key_file)
        try:
            await answer_pending(client, model, requests, answered, stop, out, ask, key_file,
                                 progress, invocation=number, total=len(keys),
                                 concurrency=concurrency,
                                 api_errors=trailing_api_errors(responses, api_errors_before),
                                 api_errors_before=api_errors_before)
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


def count(text: str) -> int:
    """A count given on the command line: a whole number, 0 or more."""
    if not text.isdigit():
        raise argparse.ArgumentTypeError(f"{text!r} is not a count of 0 or more")
    return int(text)


def main(argv: list[str] | None = None, *, prog: str = "answer_requests",
         ask: Callable[..., Awaitable[dict]] = answer.answer_once,
         image_names: Sequence[str] = IMAGE_NAMES,
         preflight: Callable[[Path], int] = netcheck_preflight,
         self_test: Callable[[], int] | None = None) -> int:
    """The command line of an answering run: the reader's own, or that of an entrant whose image is
    built FROM the reader image and answers through this runner with an answering function
    (``ask``), images, a network check and a self-test of its own, as grep at D's does
    (grep-at-d/agent.py, in asofbench/grep-at-d)."""
    parser = argparse.ArgumentParser(prog=prog, description="Answers a requests file with the "
                                     "answering model, inside the run environment.",
                                     allow_abbrev=False)
    parser.add_argument("--requests", type=Path, help="the run's requests file")
    parser.add_argument("--out", type=Path, help="the responses file, in the run's output "
                        "directory")
    parser.add_argument("--model", choices=answer.MODELS, help="the answering model")
    parser.add_argument("--resume", action="store_true",
                        help="answer the requests that have no response yet")
    parser.add_argument("--api-errors-before", type=count, dest="api_errors_before",
                        metavar="N", help="the api_error responses in a row that the runs before "
                        "this one ended with, which this run counts on from (0 when it is not "
                        "given; grep at D's run_dates.py gives it)")
    parser.add_argument("--self-test", action="store_true",
                        help="check the installation and the answering call, with no network")
    args = parser.parse_args(argv)
    if args.self_test:
        # Not given is None, so that even --api-errors-before 0 is refused with it.
        if (args.requests or args.out or args.model or args.resume
                or args.api_errors_before is not None):
            parser.error("--self-test takes no other argument")
        if self_test is not None:
            return self_test()
        from reader import selftest

        return selftest.self_test()
    if missing := [flag for flag, value in (("--requests", args.requests), ("--out", args.out),
                                            ("--model", args.model)) if value is None]:
        parser.error(f"the following arguments are required: {', '.join(missing)}")
    try:
        images = parse_image_ids(os.environ.get("ASOFBENCH_IMAGE_IDS"), image_names)
        return asyncio.run(run(args.requests, args.out, args.model, resume=args.resume,
                               images=images, ask=ask, preflight=preflight,
                               api_errors_before=args.api_errors_before or 0))
    except Refused as refusal:
        print(f"{prog}: {refusal}", file=sys.stderr)
        return EXIT_STOPPED
    except SystemExit:
        raise
    except BaseException as error:  # noqa: BLE001 - a crash: recorded in run.json, exit 4
        return crashed(error, llm.KEY_FILE, prog)


if __name__ == "__main__":
    sys.exit(main())
