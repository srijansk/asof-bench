"""Grep at D's runs, one for each arm and date, on the host (PREREGISTRATION.md section 11,
docs/entering-a-system.md's trees for grep at D, systems/README.md):

    uv run python systems/grep-at-d/run_dates.py --run-dir <run dir> --data data/v0.1 \\
        --streams-dir streams --trees-dir <a directory outside the repository> \\
        --model claude-haiku-5-5|claude-sonnet-5-5 [--project <compose project>]

with ASOFBENCH_ANTHROPIC_KEY_FILE and ASOFBENCH_IMAGE_IDS (grep at D's image, the reader image
that it is built from, the proxy's and the sandbox's) in its environment, as for the reader's
runs. Before anything else it checks, with ``docker image inspect``, that grep at D's image is
built on that reader image (its layers begin with the reader's), so that the reader's ID that each
run records is the one its agent ran on.

It reads the run directory that ``harness.py requests --system grep-at-d`` wrote: dates.json,
which maps each requests file to its arm and date, and plan.json, against which it checks
dates.json, each date's requests file and the build's MANIFEST.json, so that every tree comes from
the build whose questions the requests hold. It holds the run directory's lock (``run.lock``,
``asofbench_systems.run.hold_lock``) to its end, so that a second invocation on the same run
directory stops at once. Then, for each file in dates.json's order:

1. a date whose run is done (its last invocation exited 0 or 1) is skipped; one that stopped with
   exit status 3 (api_error responses in a row) stops this invocation with exit status 3, since
   such a run is discarded, with its whole run directory, and never resumed; any other date that
   has begun (stopped with 2, crashed with 4, or killed) is resumed;
2. its tree is built: ``asofbench materialize --stream <the arm's stream> --as-of <date> --arm
   <arm> --data <build>``, run in-process as the command-line tool runs it, writes the proposal
   files as they stood at the end of the date and git-log.txt, from the arm's registered stream
   and nothing else, into a fresh directory under --trees-dir, which must be outside the
   repository. The directory's name, ``tree-`` and random letters, says nothing of the arm or the
   date, since a bind mount's source shows in the sandbox's mount table; and every directory of
   the tree is made 0755 and every file 0644, whatever the umask, so that the sandbox's user, uid
   10001, reads it on any Docker host;
3. grep at D's environment (compose.yaml) answers the date's requests: ``docker compose run`` of
   the agent, with ASOFBENCH_RUN_INPUT the date's input/<file>/, ASOFBENCH_RUN_OUTPUT its
   output/<file>/ and ASOFBENCH_RUN_TREE the tree, which the sandbox holds read-only at /repo; no
   other file of the run directory, no other date's tree and no stream is mounted. The run is
   given ``--api-errors-before``, the ``api_error`` responses in a row that the dates before it
   ended with, in the order they were answered (dates.json's): a date whose responses are all
   ``api_error`` adds its count to the one before it, and any other date's count starts again
   from the ``api_error`` responses that it ends with. The runner counts on from it, so that its
   stop after five ``api_error`` responses in a row (exit status 3) fires wherever the fifth
   falls, across the dates as within one;
4. whatever the run did, the proxy's log is saved beside the date's responses (egress.log, and
   egress-<n>.log for a later attempt), the environment is taken down (``docker compose down``,
   which runs even when the log cannot be saved, and whose failure stops this invocation, since
   the environment may still be up) and the tree is deleted.

A SIGTERM (``terminated_on_sigterm``, the synchronous counterpart of the runner's
``asofbench_systems.run.cancelled_on_sigterm``, since this script is not async) raises
``Terminated`` wherever the script is, so that the date under way is taken down, its tree
deleted and the lock released, as at any other stop; the date's runner, stopped by the
environment's removal, records itself, and the next invocation resumes the date.

Exit status: 0 when every date is done and every request has an answer; 1 when every date is done
and some requests hold an error code; 2 when it is refused before it starts (another invocation
holds the lock among the reasons, and nothing is built), when a tree cannot be built, when a
date's run stops with exit status 2 (an infrastructure error or a network check that failed) or
is killed, when the environment did not start the run (docker compose itself failed, and no
invocation was recorded) or cannot be taken down, or on a SIGTERM; 3 when a date's run stopped
with exit status 3; 4 when a date's run crashed, or for an error that no rule expects. A stop
leaves the dates after it as they were, and running it again goes on from the date that stopped.

It uses the standard library, the asofbench package, harness.py, the reader's runner (for the
rule that counts api_error responses) and the shared run plumbing, which it puts on the import
path as harness.py does, and docker.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path

GREP_AT_D = Path(__file__).resolve().parent
if str(GREP_AT_D.parent) not in sys.path:
    sys.path.insert(0, str(GREP_AT_D.parent))

import harness  # noqa: E402 - systems/ is on the path; harness.py puts systems/common there
from asofbench_systems import answer  # noqa: E402
from asofbench_systems import run as plumbing  # noqa: E402
from asofbench_systems.run import (  # noqa: E402
    EXIT_CRASHED,
    EXIT_DONE,
    EXIT_FAILED,
    EXIT_FAILURES,
    EXIT_STOPPED,
    RUN,
    Refused,
    hold_lock,
    read_jsonl,
    sha256_file,
)
from reader.answer_requests import trailing_api_errors  # noqa: E402

from asofbench import cli  # noqa: E402
from asofbench.stream import STREAM_FILES  # noqa: E402

COMPOSE = GREP_AT_D / "compose.yaml"
# The images a run records (grep_at_d's IMAGE_NAMES), and the variables that give the key file's
# path and their IDs.
IMAGE_NAMES = ("grep-at-d", "reader", "egress", "sandbox")
KEY_VARIABLE, IMAGES_VARIABLE = "ASOFBENCH_ANTHROPIC_KEY_FILE", "ASOFBENCH_IMAGE_IDS"
# Where grep at D's documentation is, which the lock's refusal points at.
README = "systems/README.md"
# A compose project's name, as docker compose takes one.
PROJECT = re.compile(r"[a-z0-9][a-z0-9_-]*")
# The modes of a tree's directories and files: readable by every user, the sandbox's included.
TREE_DIRECTORY_MODE, TREE_FILE_MODE = 0o755, 0o644


class Terminated(BaseException):
    """A SIGTERM, raised where the script is so that every ``finally`` block runs."""


@contextlib.contextmanager
def terminated_on_sigterm() -> Iterator[None]:
    """Turns a SIGTERM into ``Terminated`` while the block runs, and puts the earlier handler back
    at its end."""
    def terminate(signum, frame):
        raise Terminated()

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def docker_compose(project: str, arguments: Sequence[str], env: Mapping[str, str],
                   stdout=None) -> int:
    """``docker compose`` on grep at D's compose file, in the project ``project``, with ``env``;
    its exit status."""
    return subprocess.run(["docker", "compose", "-f", str(COMPOSE), "-p", project, *arguments],
                          env=dict(env), stdout=stdout, check=False).returncode


def image_layers(image: str) -> list[str]:
    """The layers of the image ``image`` (an ID), as ``docker image inspect`` reports them;
    Refused when docker cannot inspect it."""
    done = subprocess.run(["docker", "image", "inspect", "--format", "{{json .RootFS.Layers}}",
                           image], capture_output=True, text=True, check=False)
    try:
        layers = json.loads(done.stdout) if done.returncode == 0 else None
    except ValueError:
        layers = None
    if not isinstance(layers, list) or not all(isinstance(layer, str) for layer in layers):
        raise Refused(f"docker cannot inspect the image {image}: {done.stderr.strip()[:200]}")
    return layers


def check_built_on_reader(images: Mapping[str, str],
                          inspect_layers: Callable[[str], list[str]]) -> None:
    """Refused unless grep at D's image is built on the reader image whose ID the run records:
    its layers are the reader's, followed by its own."""
    reader, built = inspect_layers(images["reader"]), inspect_layers(images["grep-at-d"])
    if not reader or len(built) <= len(reader) or built[:len(reader)] != reader:
        raise Refused(f"grep at D's image ({images['grep-at-d']}) is not built on the reader "
                      f"image whose ID {IMAGES_VARIABLE} gives ({images['reader']}): build "
                      "asofbench/grep-at-d again after asofbench/reader, and give the IDs of "
                      f"both ({README})")


def checked_run_dir(run_dir: Path, data: Path) -> tuple[dict, dict]:
    """The run directory's plan.json and dates.json, after checking that it is a whole grep at D
    run's and that ``data`` is the build whose questions its requests hold; Refused otherwise."""
    plan = harness.load_json(run_dir / harness.PLAN)
    if plan.get("system") != "grep-at-d" or not isinstance(plan.get("files"), dict):
        raise Refused(f"{run_dir} holds requests for {plan.get('system')}, not grep-at-d's: write "
                      "them with harness.py requests --system grep-at-d")
    dates_path = run_dir / harness.DATES
    if not dates_path.is_file():
        raise Refused(f"{run_dir} holds no {harness.DATES}, which harness.py requests writes "
                      "last: it was interrupted, so write the requests again in a new run "
                      "directory")
    if harness.read_input(sha256_file, dates_path) != plan.get("dates_sha256"):
        raise Refused(f"{harness.DATES} is not the one that {harness.PLAN} records: its SHA-256 "
                      "differs")
    dates = harness.load_json(dates_path)
    if list(dates) != list(plan["files"]):
        raise Refused(f"{harness.DATES} and {harness.PLAN} list different requests files")
    for name, entry in dates.items():
        if (not isinstance(entry, dict) or entry.get("arm") not in STREAM_FILES
                or not harness.is_day(entry.get("as_of"))
                or name != harness.file_name(entry["arm"], entry["as_of"])):
            raise Refused(f"{harness.DATES}: {name} is not a file of an arm and a date")
        path = run_dir / harness.INPUT / name / harness.REQUESTS
        if harness.read_input(sha256_file, path) != plan["files"][name]["requests_sha256"]:
            raise Refused(f"{harness.INPUT}/{name}/{harness.REQUESTS} is not the one that "
                          f"{harness.PLAN} records: its SHA-256 differs")
        if not (run_dir / harness.OUTPUT / name).is_dir():
            raise Refused(f"{run_dir} has no {harness.OUTPUT}/{name}/ for its run")
    manifest = harness.read_input(sha256_file, data / harness.MANIFEST)
    if manifest != plan["manifest"]["sha256"]:
        raise Refused(f"{data}'s {harness.MANIFEST} is not the build's whose questions the "
                      f"requests hold: its SHA-256 is {manifest}, and {harness.PLAN} records "
                      f"{plan['manifest']['sha256']}")
    return plan, dates


def date_record(output: Path) -> dict | None:
    """A date's run.json, or None when the date has not begun."""
    record_path = output / RUN
    if not record_path.is_file():
        return None
    record = harness.load_json(record_path)
    if not isinstance(record.get("invocations"), list):
        raise Refused(f"{output.name}/{RUN} records no invocations")
    return record


def date_responses(output: Path, record: dict) -> list[dict]:
    """The responses that a date's run wrote, in the order it wrote them."""
    name = record.get("responses")
    if not isinstance(name, str) or Path(name).name != name or name in ("", ".", ".."):
        raise Refused(f"{output.name}/{RUN} names no responses file")
    return read_jsonl(output / name)


def log_path(output: Path) -> Path:
    """Where a date's proxy log goes: egress.log, or egress-<n>.log beside earlier ones."""
    path, n = output / "egress.log", 0
    while path.exists():
        n += 1
        path = output / f"egress-{n}.log"
    return path


def build_tree(entry: dict, streams_dir: Path, data: Path, trees_dir: Path) -> Path | None:
    """The date's tree, in a fresh directory under ``trees_dir`` whose name says nothing of the
    date, readable by every user; None when ``asofbench materialize`` refuses (the directory is
    removed again)."""
    tree = Path(tempfile.mkdtemp(prefix="tree-", dir=trees_dir))
    built = cli.main(["materialize", "--stream", str(streams_dir / STREAM_FILES[entry["arm"]]),
                      "--as-of", entry["as_of"], "--out", str(tree), "--arm", entry["arm"],
                      "--data", str(data)])
    if built != 0:
        shutil.rmtree(tree)
        return None
    for path in [tree, *tree.rglob("*")]:
        os.chmod(path, TREE_DIRECTORY_MODE if path.is_dir() else TREE_FILE_MODE)
    return tree


def run_date(name: str, entry: dict, run_dir: Path, data: Path, streams_dir: Path,
             trees_dir: Path, model: str, *, resume: bool, api_errors_before: int, project: str,
             compose: Callable, environ: Mapping[str, str],
             progress: Callable[[str], None]) -> int:
    """One date's run: its tree built, its requests answered, the environment taken down and the
    tree deleted (the module's description); returns the run's exit status, or 2 when no run
    began or the environment could not be taken down."""
    output = run_dir / harness.OUTPUT / name
    before = date_record(output)
    tree = build_tree(entry, streams_dir, data, trees_dir)
    if tree is None:
        progress(f"run_dates.py: {name}: its tree could not be built (asofbench materialize "
                 "refused it), so nothing was run")
        return EXIT_STOPPED
    down = None
    try:
        env = {**environ,
               "ASOFBENCH_RUN_INPUT": str((run_dir / harness.INPUT / name).resolve()),
               "ASOFBENCH_RUN_OUTPUT": str(output.resolve()),
               "ASOFBENCH_RUN_TREE": str(tree.resolve())}
        arguments = ["run", "--rm", "agent", "grep_at_d", "--requests", "/input/requests.jsonl",
                     "--out", "/output/responses.jsonl", "--model", model,
                     "--api-errors-before", str(api_errors_before),
                     *(["--resume"] if resume else [])]
        progress(f"run_dates.py: {name} ({entry['arm']}, {entry['as_of']}): "
                 f"{'resuming' if resume else 'answering'} its requests, after "
                 f"{api_errors_before} api_error responses in a row")
        try:
            code = compose(project, arguments, env)
        finally:
            try:
                with open(log_path(output), "wb") as handle:
                    compose(project, ["logs", "--no-log-prefix", "egress"], env, stdout=handle)
            finally:
                down = compose(project, ["down"], env)
    finally:
        shutil.rmtree(tree)
    if down != 0:
        progress(f"run_dates.py: {name}: the environment could not be taken down (docker compose "
                 f"down exited {down}): check `docker compose -p {project} ps -a` before running "
                 "this again")
        return EXIT_STOPPED
    after = date_record(output)
    if after is None or (before is not None
                         and len(after["invocations"]) == len(before["invocations"])):
        progress(f"run_dates.py: {name}: the environment did not start the run (docker compose "
                 f"exited {code}), so nothing was answered")
        return EXIT_STOPPED
    last = harness.last_exit(after)
    return EXIT_STOPPED if last is None else last


def discard(name: str) -> str:
    return (f"run_dates.py: {name} stopped with exit status {EXIT_FAILURES} (api_error responses "
            "in a row, which point at a cause that every request shares): discard the run and "
            "run it again in a fresh run directory once the cause is fixed; never resume it "
            f"({README})")


def run(run_dir: Path, data: Path, streams_dir: Path, trees_dir: Path, model: str, *,
        project: str | None = None, compose: Callable | None = None,
        environ: Mapping[str, str] | None = None,
        inspect_layers: Callable[[str], list[str]] | None = None,
        progress: Callable[[str], None] = print) -> int:
    """Every date's run (the module's description); Refused when it cannot start."""
    compose = docker_compose if compose is None else compose
    environ = os.environ if environ is None else environ
    inspect_layers = image_layers if inspect_layers is None else inspect_layers
    if model not in answer.MODELS:
        raise Refused(f"--model {model!r} is not an answering model ({', '.join(answer.MODELS)})")
    if harness.inside_repository(trees_dir):
        raise Refused(f"{trees_dir} is inside the repository: give --trees-dir a directory "
                      "outside it, so that the trees and the checkout never share a directory")
    if not trees_dir.is_dir():
        raise Refused(f"{trees_dir} is not a directory: give --trees-dir a directory outside the "
                      "repository")
    if not run_dir.is_dir():
        raise Refused(f"{run_dir} is not a run directory")
    if project is None:
        project = "asofbench-grep-at-d-" + (re.sub(r"[^a-z0-9_-]+", "-",
                                                   run_dir.resolve().name.lower()).strip("-")
                                            or "run")
    if not PROJECT.fullmatch(project):
        raise Refused(f"--project {project!r} is not a compose project's name")
    images = plumbing.parse_image_ids(environ.get(IMAGES_VARIABLE), IMAGE_NAMES, README)
    key_file = environ.get(KEY_VARIABLE)
    if not key_file or not Path(key_file).is_file():
        raise Refused(f"{KEY_VARIABLE} must name the file that holds the API key, which reaches "
                      f"the agent only as a Docker secret ({README})")
    check_built_on_reader(images, inspect_layers)
    with hold_lock(run_dir, "run_dates", readme=README), terminated_on_sigterm():
        try:
            _, dates = checked_run_dir(run_dir, data)
            for arm in sorted({entry["arm"] for entry in dates.values()}):
                if not (streams_dir / STREAM_FILES[arm]).is_file():
                    raise Refused(f"{streams_dir} holds no {STREAM_FILES[arm]}, the {arm} "
                                  "stream, from which the trees of its dates are built")
            return run_all(dates, run_dir, data, streams_dir, trees_dir, model, project=project,
                           compose=compose, environ=environ, progress=progress)
        except Terminated:
            progress("run_dates.py: stopped by a SIGTERM: the date under way, if any, was taken "
                     "down and its tree deleted; run this again to resume it")
            return EXIT_STOPPED


def run_all(dates: dict, run_dir: Path, data: Path, streams_dir: Path, trees_dir: Path,
            model: str, *, project: str, compose: Callable, environ: Mapping[str, str],
            progress: Callable[[str], None]) -> int:
    """The dates in turn, with the count of api_error responses in a row carried from each to the
    next (the module's description)."""
    unanswered, carried = False, 0
    for name, entry in dates.items():
        output = run_dir / harness.OUTPUT / name
        record = date_record(output)
        last = None if record is None else harness.last_exit(record)
        if record is not None and last in (EXIT_DONE, EXIT_FAILED):
            unanswered |= last == EXIT_FAILED
            carried = trailing_api_errors(date_responses(output, record), carried)
            continue
        if record is not None and last == EXIT_FAILURES:
            progress(discard(name))
            return EXIT_FAILURES
        code = run_date(name, entry, run_dir, data, streams_dir, trees_dir, model,
                        resume=record is not None, api_errors_before=carried, project=project,
                        compose=compose, environ=environ, progress=progress)
        if code == EXIT_FAILURES:
            progress(discard(name))
            return code
        if code not in (EXIT_DONE, EXIT_FAILED):
            progress(f"run_dates.py: {name} stopped with exit status {code}; run this again to "
                     "resume it once the cause is fixed")
            return code
        unanswered |= code == EXIT_FAILED
        carried = trailing_api_errors(date_responses(output, date_record(output)), carried)
    progress(f"run_dates.py: the {len(dates)} dates are done"
             + ("; some requests hold an error code" if unanswered else ""))
    return EXIT_FAILED if unanswered else EXIT_DONE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_dates.py", description="Grep at D's runs, one for "
                                     "each arm and date, each with its own tree.",
                                     allow_abbrev=False)
    parser.add_argument("--run-dir", required=True, type=Path, dest="run_dir",
                        help="the run directory that harness.py requests --system grep-at-d wrote")
    parser.add_argument("--data", required=True, type=Path, help="the build, such as data/v0.1")
    parser.add_argument("--streams-dir", required=True, type=Path, dest="streams_dir",
                        help="the directory that holds the arms' streams")
    parser.add_argument("--trees-dir", required=True, type=Path, dest="trees_dir",
                        help="a directory outside the repository, for each date's tree")
    parser.add_argument("--model", required=True, choices=answer.MODELS)
    parser.add_argument("--project", help="the compose project (by default named after the run "
                                          "directory)")
    args = parser.parse_args(argv)
    try:
        return run(args.run_dir, args.data, args.streams_dir, args.trees_dir, args.model,
                   project=args.project, compose=docker_compose, inspect_layers=image_layers)
    except Refused as refusal:
        print(f"run_dates.py: {refusal}", file=sys.stderr)
        return EXIT_STOPPED
    except Exception:  # noqa: BLE001 - an error that no rule expects: its own exit status
        print(f"run_dates.py: crashed (exit {EXIT_CRASHED})\n{traceback.format_exc()}",
              file=sys.stderr)
        return EXIT_CRASHED


if __name__ == "__main__":
    sys.exit(main())
