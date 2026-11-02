"""The plumbing that every system's run shares (common/asofbench_systems/run.py): the output
directory's lock and the SIGTERM that cancels an invocation, run.json's invocation records and
their atomic writes, JSON lines, the image IDs, the wait for the services and the network check,
the classification of infrastructure errors, the stop, and the key check, the error text, the
crash report, the hash of a file and the atomic write of any text. What each piece does with
Graphiti's own values is checked in test_ingest_resume.py and test_graphiti_wiring.py. Standard
library only."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shlex
import signal
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from asofbench_systems import run

# Any names would do; these are the images of a run that has a store, an embedder and a proxy.
NAMES = ("graphiti", "egress", "embedder", "neo4j")
IMAGES = {name: "sha256:" + str(n) * 64 for n, name in enumerate(NAMES, 1)}


def failure(module: str, name: str) -> type[Exception]:
    """An exception class that stands in for one of a package's, by its module and name."""
    return type(name, (Exception,), {"__module__": module})


ServiceUnavailable = failure("neo4j.exceptions", "ServiceUnavailable")
SessionExpired = failure("neo4j.exceptions", "SessionExpired")
EmbedderDown = failure("openai._exceptions", "APIConnectionError")
Overloaded = failure("anthropic._exceptions", "OverloadedError")
BadRequest = failure("anthropic._exceptions", "BadRequestError")
# The table of a system that runs a graph database and an embedder besides calling the API: the
# shared entries and its own.
TABLE = {**run.INFRASTRUCTURE,
         "neo4j": {"ServiceUnavailable", "SessionExpired", "SessionError", "DatabaseUnavailable"},
         "openai": {"APIConnectionError", "InternalServerError"}}


def classified(error: BaseException) -> str | None:
    return run.infrastructure_error(error, TABLE)


@contextmanager
def listening() -> Iterator[int]:
    """The port of a service on 127.0.0.1 that accepts connections."""
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        yield server.getsockname()[1]


def refused_port() -> int:
    """A port on 127.0.0.1 that nothing listens on."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_infrastructure_errors_are_told_from_the_episode_s_own():
    assert classified(ServiceUnavailable()) == "neo4j ServiceUnavailable"
    assert classified(ConnectionRefusedError()) == "builtins ConnectionError"
    assert classified(Overloaded()) == "anthropic OverloadedError"
    assert classified(ValueError("bad output")) is None
    assert classified(BadRequest("prompt is too long")) is None
    # Through a cause, as Graphiti raises its RateLimitError from the SDK's, and inside a group.
    try:
        try:
            raise failure("anthropic._exceptions", "RateLimitError")("429")
        except Exception as cause:
            raise RuntimeError("wrapped") from cause
    except RuntimeError as wrapped:
        assert classified(wrapped) == "anthropic RateLimitError"
    group = ExceptionGroup("several", [ValueError("x"), SessionExpired("lost")])
    assert classified(group) == "neo4j SessionExpired"
    subclass = type("APITimeoutError", (EmbedderDown,), {"__module__": "openai._exceptions"})
    assert classified(subclass()) == "openai APIConnectionError"


def test_the_shared_table_holds_the_api_and_the_connection_errors_and_nothing_else():
    assert run.INFRASTRUCTURE == {
        "anthropic": {"APIConnectionError", "RateLimitError", "OverloadedError",
                      "InternalServerError", "AuthenticationError", "PermissionDeniedError"},
        "builtins": {"ConnectionError"}}


def test_a_table_built_on_the_shared_one_cannot_change_what_the_shared_one_holds():
    own = {**run.INFRASTRUCTURE, "mine": frozenset({"Boom"})}
    assert all(isinstance(classes, frozenset) for classes in run.INFRASTRUCTURE.values())
    with pytest.raises(AttributeError):
        own["anthropic"].add("Boom")
    assert "Boom" not in run.INFRASTRUCTURE["anthropic"]
    assert run.infrastructure_error(failure("mine.errors", "Boom")(), own) == "mine Boom"


def test_the_table_that_the_caller_gives_decides_which_errors_stop_the_run():
    mine = {"mine": {"Boom"}}
    assert run.infrastructure_error(failure("mine.errors", "Boom")(), mine) == "mine Boom"
    # The shared entries apply only where the caller's table has them.
    assert run.infrastructure_error(Overloaded(), mine) is None
    assert run.infrastructure_error(ConnectionRefusedError(), mine) is None
    assert run.infrastructure_error(Overloaded(), {}) is None
    assert run.infrastructure_error(Overloaded(), run.INFRASTRUCTURE) == "anthropic OverloadedError"
    # The first match in the walk is the one named: an error before its cause, a group's members
    # in order.
    try:
        try:
            raise ServiceUnavailable("the cause")
        except Exception as cause:
            raise Overloaded("the error") from cause
    except Overloaded as error:
        assert classified(error) == "anthropic OverloadedError"
    assert classified(ExceptionGroup("g", [ServiceUnavailable(), Overloaded()])) == (
        "neo4j ServiceUnavailable")


def test_the_image_ids_are_parsed_and_checked():
    text = ",".join(f"{name}={value}" for name, value in IMAGES.items())
    assert run.parse_image_ids(text, NAMES, "README.md") == IMAGES
    for bad in (None, "", text.replace("egress=", "proxy="), text + ",neo4j=" + IMAGES["neo4j"],
                text.replace(IMAGES["graphiti"], "asofbench/graphiti")):
        with pytest.raises(run.Refused, match="ASOFBENCH_IMAGE_IDS"):
            run.parse_image_ids(bad, NAMES, "README.md")


def test_the_image_ids_follow_the_names_given_and_the_refusal_points_at_the_readme():
    names = ("reader", "egress")
    ids = {"reader": "sha256:" + "a" * 64, "egress": "sha256:" + "b" * 64}
    # The result follows the names, not the text.
    assert list(run.parse_image_ids(f"egress={ids['egress']},reader={ids['reader']}", names,
                                    "systems/reader/README.md")) == ["reader", "egress"]
    # Another system's names are refused, and so is a missing image.
    for bad in (",".join(f"{name}={value}" for name, value in IMAGES.items()),
                f"reader={ids['reader']}"):
        with pytest.raises(run.Refused) as caught:
            run.parse_image_ids(bad, names, "systems/reader/README.md")
        assert str(caught.value) == (
            "ASOFBENCH_IMAGE_IDS must give the ID (sha256:...) of each image the run uses, as "
            "name=ID entries for reader, egress (systems/reader/README.md)")


def test_a_second_invocation_is_refused_while_one_holds_the_lock(tmp_path):
    with run.hold_lock(tmp_path, "answer"):
        holder = json.loads((tmp_path / "run.lock").read_text(encoding="utf-8"))
        assert set(holder) == {"token", "command", "host", "pid", "since"}
        assert (holder["command"], holder["host"], holder["pid"]) == (
            "answer", socket.gethostname(), os.getpid())
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", holder["since"])
        with pytest.raises(run.Refused, match=r"another invocation holds the lock on this output "
                           r"directory \(run\.lock: answer in container .*, pid \d+, since .*; "
                           r"its heartbeat is \d+ s old\).*remove run\.lock and run again"):
            with run.hold_lock(tmp_path, "answer"):
                pytest.fail("a second invocation took the lock")
        # The refused invocation leaves the holder's lock as it was.
        assert json.loads((tmp_path / "run.lock").read_text(encoding="utf-8")) == holder
    assert list(tmp_path.iterdir()) == []


def test_the_refusal_points_at_the_readme_that_the_caller_gives(tmp_path):
    with run.hold_lock(tmp_path, "answer"):
        # With none given, it is the README.md that Graphiti's refusals have always named.
        for arguments, readme in (({}, "README.md"),
                                  ({"readme": "systems/reader/README.md"},
                                   "systems/reader/README.md")):
            with pytest.raises(run.Refused) as caught:
                with run.hold_lock(tmp_path, "answer", **arguments):
                    pytest.fail("a second invocation took the lock")
            assert str(caught.value).endswith(f"gone ({readme} says how to check), remove "
                                              "run.lock and run again")


def test_the_lock_is_released_when_the_invocation_raises(tmp_path):
    with pytest.raises(RuntimeError, match="boom"), run.hold_lock(tmp_path, "answer"):
        raise RuntimeError("boom")
    assert list(tmp_path.iterdir()) == []
    with run.hold_lock(tmp_path, "answer"):
        assert (tmp_path / "run.lock").exists()


def test_a_lock_whose_holder_cannot_be_read_is_refused_all_the_same_and_left_alone(tmp_path):
    lock = tmp_path / "run.lock"
    lock.write_text("not json", encoding="utf-8")
    with pytest.raises(run.Refused, match=r"run\.lock: its holder cannot be read\)"):
        with run.hold_lock(tmp_path, "answer"):
            pytest.fail("a lock that someone holds was taken")
    assert lock.read_text(encoding="utf-8") == "not json"


def test_a_lock_is_touched_while_held_and_removed_only_by_its_holder(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "LOCK_HEARTBEAT_S", 0.05)
    lock = tmp_path / "run.lock"
    with run.hold_lock(tmp_path, "ingest"):
        os.utime(lock, (time.time() - 3600, time.time() - 3600))
        time.sleep(0.5)
        assert time.time() - lock.stat().st_mtime < 60
    assert not lock.exists()
    with run.hold_lock(tmp_path, "ingest"):
        lock.write_text(json.dumps({"token": "someone else's"}), encoding="utf-8")
    assert json.loads(lock.read_text(encoding="utf-8")) == {"token": "someone else's"}


def test_a_sigterm_cancels_the_task_that_runs_the_invocation():
    async def invocation():
        with run.cancelled_on_sigterm():
            os.kill(os.getpid(), signal.SIGTERM)
            await asyncio.sleep(5)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(invocation())


def test_off_the_main_thread_the_invocation_runs_with_no_handler_for_the_signal():
    ran = []

    async def invocation():
        with run.cancelled_on_sigterm():
            ran.append("ran")

    # An event loop off the main thread cannot take a signal handler (asyncio raises
    # RuntimeError): the body runs without one.
    thread = threading.Thread(target=lambda: asyncio.run(invocation()))
    thread.start()
    thread.join()
    assert ran == ["ran"]


def test_an_invocation_record_has_its_fields_in_one_order():
    images = dict(IMAGES)
    ingest = run.new_invocation("ingest", images, resume=True, skipped=[])
    removal = run.new_invocation("remove-episodes", images, uuids=["u1"], removed=[])
    assert list(ingest) == ["command", "started_at", "resume", "skipped", "images", "elapsed_s",
                            "exit", "stop", "crash"]
    assert list(removal) == ["command", "started_at", "uuids", "removed", "images", "elapsed_s",
                             "exit", "stop", "crash"]
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", ingest["started_at"])
    assert (ingest["command"], ingest["resume"], ingest["skipped"]) == ("ingest", True, [])
    assert [ingest[name] for name in ("elapsed_s", "exit", "stop", "crash")] == [None] * 4
    # The record holds a copy of the images; a command with no fields of its own has none between
    # started_at and images.
    images["graphiti"] = "changed"
    assert ingest["images"] == IMAGES
    assert list(run.new_invocation("answer", IMAGES)) == [
        "command", "started_at", "images", "elapsed_s", "exit", "stop", "crash"]


def test_recorded_writes_the_invocation_at_once_and_again_with_its_elapsed_time(tmp_path):
    path = tmp_path / "run.json"
    record = {"arm": "H+", "invocations": []}
    invocation = run.new_invocation("answer", IMAGES, resume=False)
    with run.recorded(record, invocation, path, time.monotonic() - 2.0):
        # While the body runs, run.json holds the invocation with nothing known of its end.
        started = json.loads(path.read_text(encoding="utf-8"))
        assert started["invocations"] == [invocation]
        assert (started["invocations"][0]["elapsed_s"], started["invocations"][0]["exit"]) == (
            None, None)
        invocation["exit"] = run.EXIT_FAILED
    done = json.loads(path.read_text(encoding="utf-8"))
    assert done == record and done["arm"] == "H+"
    (written,) = done["invocations"]
    assert (written["exit"], written["crash"]) == (run.EXIT_FAILED, None)
    assert 2.0 <= written["elapsed_s"] < 60.0
    # A second invocation is appended after the first.
    again = run.new_invocation("answer", IMAGES, resume=True)
    with run.recorded(record, again, path, time.monotonic()):
        pass
    assert [i["resume"] for i in json.loads(path.read_text(encoding="utf-8"))["invocations"]] == [
        False, True]


def test_a_refusal_is_recorded_as_exit_2_unless_the_body_set_another_status(tmp_path):
    path = tmp_path / "run.json"
    record = {"invocations": []}
    refused = run.new_invocation("answer", IMAGES)
    with pytest.raises(run.Refused), run.recorded(record, refused, path, time.monotonic()):
        raise run.Refused("not started")
    assert (refused["exit"], refused["crash"]) == (run.EXIT_STOPPED, None)
    assert refused["elapsed_s"] is not None
    set_first = run.new_invocation("answer", IMAGES)
    with pytest.raises(run.Refused), run.recorded(record, set_first, path, time.monotonic()):
        set_first["exit"] = run.EXIT_FAILURES
        raise run.Refused("stopped later")
    assert set_first["exit"] == run.EXIT_FAILURES
    assert json.loads(path.read_text(encoding="utf-8")) == record


def test_a_crash_is_recorded_as_exit_4_with_the_class_of_the_exception(tmp_path):
    path = tmp_path / "run.json"
    record = {"invocations": []}
    for error in (KeyError("x"), asyncio.CancelledError()):
        invocation = run.new_invocation("answer", IMAGES)
        with pytest.raises(type(error)), run.recorded(record, invocation, path, time.monotonic()):
            raise error
        assert (invocation["exit"], invocation["crash"]) == (run.EXIT_CRASHED,
                                                              type(error).__name__)
        assert invocation["elapsed_s"] is not None
    assert [(i["exit"], i["crash"]) for i in json.loads(
        path.read_text(encoding="utf-8"))["invocations"]] == [(4, "KeyError"),
                                                              (4, "CancelledError")]


def test_an_elapsed_time_that_the_body_set_is_kept(tmp_path):
    path = tmp_path / "run.json"
    invocation = run.new_invocation("answer", IMAGES)
    with run.recorded({"invocations": []}, invocation, path, time.monotonic()):
        invocation["elapsed_s"] = 12.5
    assert json.loads(path.read_text(encoding="utf-8"))["invocations"][0]["elapsed_s"] == 12.5


def test_run_json_is_written_whole_or_not_at_all(tmp_path, monkeypatch):
    path = tmp_path / "run.json"
    run.write_json(path, {"version": 1})
    run.write_json(path, {"version": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 2}
    assert [entry.name for entry in tmp_path.iterdir()] == ["run.json"]

    def interrupted(source, target):
        raise OSError("interrupted before the rename")

    monkeypatch.setattr(run.os, "replace", interrupted)
    with pytest.raises(OSError):
        run.write_json(path, {"version": 3})
    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 2}


def test_json_lines_are_split_at_newlines_only_and_a_bad_line_is_refused_by_its_number(tmp_path):
    path = tmp_path / "log.jsonl"
    assert run.read_jsonl(path) == []
    # A line separator inside a string stays in it: only a newline ends a line.
    path.write_bytes('{"text": "a\u2028b"}\n{"n": 2}\n'.encode())
    assert run.read_jsonl(path) == [{"text": "a\u2028b"}, {"n": 2}]
    path.write_text('{"n": 1}\nnot json\n{"n": 3}\n', encoding="utf-8")
    with pytest.raises(run.Refused, match=r"log\.jsonl: line 2 is not JSON"):
        run.read_jsonl(path)


def test_an_unfinished_last_line_is_cut_and_its_bytes_are_counted(tmp_path):
    path = tmp_path / "log.jsonl"
    assert run.drop_unfinished_line(path) == 0 and not path.exists()
    for whole in (b"", b'{"a": 1}\n{"b": 2}\n'):
        path.write_bytes(whole)
        assert run.drop_unfinished_line(path) == 0 and path.read_bytes() == whole
    unfinished = b'{"b": 2'
    path.write_bytes(b'{"a": 1}\n' + unfinished)
    assert run.drop_unfinished_line(path) == len(unfinished)
    assert path.read_bytes() == b'{"a": 1}\n'
    path.write_bytes(unfinished)
    assert run.drop_unfinished_line(path) == len(unfinished) and path.read_bytes() == b""


def test_the_wait_for_services_ends_when_every_service_accepts_a_connection():
    with listening() as first, listening() as second:
        started = time.monotonic()
        run.wait_for_services((("127.0.0.1", first), ("127.0.0.1", second)), 30.0, 0.05)
        assert time.monotonic() - started < 10.0


def test_a_service_that_comes_up_later_is_tried_again_after_the_retry_interval(monkeypatch):
    attempts, sleeps = [], []

    def create_connection(target, timeout):
        attempts.append((target, timeout))
        if len(attempts) < 3:
            raise ConnectionRefusedError
        return socket.socket()

    monkeypatch.setattr(run.socket, "create_connection", create_connection)
    monkeypatch.setattr(run.time, "sleep", sleeps.append)
    run.wait_for_services((("embedder", 8080), ("neo4j", 7687)), 600.0, 0.5)
    # Each try waits the retry interval for its connection, and the services are taken in order.
    assert attempts == [(("embedder", 8080), 0.5)] * 3 + [(("neo4j", 7687), 0.5)]
    assert sleeps == [0.5, 0.5]


def test_the_wait_for_a_service_that_never_comes_ends_at_the_deadline_and_raises_nothing():
    started = time.monotonic()
    assert run.wait_for_services((("127.0.0.1", refused_port()),), 0.3, 0.05) is None
    assert 0.3 <= time.monotonic() - started < 30.0


def stub_check(directory: Path) -> Path:
    """A stand-in for env/netcheck.py: it writes the arguments it was given to its ``--report``
    file, and exits with the status that follows ``--exit``."""
    script = directory / "stub_netcheck.py"
    script.write_text("import json, sys\n"
                      "argv = sys.argv[1:]\n"
                      "report = argv[argv.index('--report') + 1]\n"
                      "open(report, 'w').write(json.dumps(argv))\n"
                      "sys.exit(int(argv[argv.index('--exit') + 1]))\n", encoding="utf-8")
    return script


def test_the_network_check_gets_the_flags_services_and_mounts_the_system_gives(tmp_path,
                                                                                monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    waits = []
    monkeypatch.setattr(run, "wait_for_services", lambda *arguments: waits.append(arguments))
    services = (("neo4j", 7687), ("embedder", 8080))
    flags = ("--proxy", "http://egress:3128", "--exit", "3")
    status = run.netcheck_preflight(out, script=stub_check(tmp_path), args=flags,
                                    services=services, mounts=("/input", "/output"),
                                    wait_s=600.0, retry_s=2.0)
    # The check's own exit status comes back, after the wait the system asks for, for the
    # services in the order it gives.
    assert status == 3
    assert waits == [(services, 600.0, 2.0)]
    report = out / "netcheck.json"
    assert json.loads(report.read_text(encoding="utf-8")) == [
        *flags, "--services", "embedder:8080,neo4j:7687", "--mounts", "/input,/output",
        "--report", str(report)]


def test_a_system_with_no_services_or_mounts_passes_the_flags_with_nothing_after_them(tmp_path):
    script = stub_check(tmp_path)
    assert run.netcheck_preflight(tmp_path, script=script, args=("--exit", "0"), services=(),
                                  mounts=(), wait_s=0.0, retry_s=0.0) == 0
    assert json.loads((tmp_path / "netcheck.json").read_text(encoding="utf-8")) == [
        "--exit", "0", "--services", "", "--mounts", "", "--report",
        str(tmp_path / "netcheck.json")]


def readme_netcheck_words() -> list[str]:
    """The words of the network check's command line in systems/README.md, its continuation lines
    joined, as the shell splits them."""
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    commands = [line for block in re.findall(r"```bash\n(.*?)```", readme, re.S)
                for line in block.replace("\\\n", " ").splitlines()
                if "netcheck.py" in line and "--proxy" in line]
    (command,) = commands
    return shlex.split(command)


def test_the_flags_every_system_gives_the_check_are_those_of_the_readme_s_command():
    words = readme_netcheck_words()
    # The README runs the check by hand, with the embedder as its one service; the flags before
    # --services are the ones that every system's run gives it.
    shared = words[words.index("/tmp/netcheck.py") + 1:words.index("--services")]
    assert tuple(shared) == run.NETCHECK_ARGS
    assert run.NETCHECK_ARGS[:4] == ("--proxy", "http://egress:3128", "--allow",
                                     "api.anthropic.com:443")
    assert not {"--services", "--mounts", "--report"} & set(run.NETCHECK_ARGS)


def test_the_network_check_waits_for_a_service_that_is_not_up_yet(tmp_path):
    with socket.socket() as server:
        # Bound, so that the port is the service's, but not yet listening: a connection is refused.
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]
        threading.Timer(0.4, server.listen).start()
        started = time.monotonic()
        status = run.netcheck_preflight(tmp_path, script=stub_check(tmp_path),
                                        args=("--exit", "0"), services=(("127.0.0.1", port),),
                                        mounts=("/output",), wait_s=30.0, retry_s=0.05)
        assert time.monotonic() - started >= 0.3
    assert status == 0
    assert f"127.0.0.1:{port}" in json.loads(
        (tmp_path / "netcheck.json").read_text(encoding="utf-8"))


def test_an_earlier_network_report_is_kept_under_the_next_free_number(tmp_path):
    script = stub_check(tmp_path)

    def check(exit_status: int) -> int:
        return run.netcheck_preflight(tmp_path, script=script, args=("--exit", str(exit_status)),
                                      services=(), mounts=(), wait_s=0.0, retry_s=0.0)

    reports = []
    for exit_status in (1, 0, 2):
        assert check(exit_status) == exit_status
        reports.append((tmp_path / "netcheck.json").read_text(encoding="utf-8"))
    assert (tmp_path / "netcheck-1.json").read_text(encoding="utf-8") == reports[0]
    assert (tmp_path / "netcheck-2.json").read_text(encoding="utf-8") == reports[1]
    assert not (tmp_path / "netcheck-3.json").exists()
    assert json.loads(reports[2])[:2] == ["--exit", "2"]


def test_the_first_reason_to_stop_the_run_is_the_one_kept():
    async def scenario():
        stop = run.Stop()
        assert (stop.code, stop.reason, stop.record) == (None, None, None)
        assert await stop.wait(0.01) is False
        stop.request(run.EXIT_STOPPED, "first", {"error": "one"})
        stop.request(run.EXIT_FAILURES, "second", {"error": "two"})
        assert (stop.code, stop.reason, stop.record) == (2, "first", {"error": "one"})
        assert await stop.wait(5.0) is True

    asyncio.run(scenario())


def test_the_exit_statuses_and_the_names_of_the_files_every_run_shares():
    assert (run.EXIT_DONE, run.EXIT_FAILED, run.EXIT_STOPPED, run.EXIT_FAILURES,
            run.EXIT_CRASHED) == (0, 1, 2, 3, 4)
    # The status 0 is every runner's: EXIT_INGESTED is Graphiti's name for it, in ingest.py.
    assert not hasattr(run, "EXIT_INGESTED")
    assert (run.NETCHECK, run.RUN, run.LOCK) == ("netcheck.json", "run.json", "run.lock")
    assert run.LOCK_HEARTBEAT_S == 30.0


KEY = "sk-ant-api00-made-up-for-tests-" + "5" * 40


def test_the_key_check_refuses_a_file_without_a_key_and_points_at_the_readme_given(tmp_path):
    key_file = tmp_path / "key"
    key_file.write_text(KEY + "\n", encoding="ascii")
    run.check_key_file(str(key_file), "systems/reader/README.md")
    key_file.write_text(f"{KEY}\n{KEY}\n", encoding="ascii")
    for path, why in ((tmp_path / "missing", "cannot be read (No such file or directory)"),
                      (key_file, "does not hold an API key (one line of printable ASCII with no "
                                 "space)")):
        with pytest.raises(run.Refused) as caught:
            run.check_key_file(str(path), "systems/reader/README.md")
        assert str(caught.value) == (
            f"the API key file {path} {why}: it reaches the container as the Docker secret, which "
            "the container's user, uid 10001, must be able to read (systems/reader/README.md)")
        assert KEY not in str(caught.value)


def test_an_error_s_text_keeps_its_class_and_300_characters_without_the_key(tmp_path):
    key_file = tmp_path / "key"
    key_file.write_text(KEY + "\n", encoding="ascii")
    assert run.error_text(RuntimeError(f"rejected {KEY}"), str(key_file)) == (
        "RuntimeError: rejected [redacted]")
    assert run.error_text(ValueError("x" * 400), str(key_file)) == "ValueError: " + "x" * 300
    assert run.error_text(KeyError(), str(key_file)) == "KeyError"
    assert run.ERROR_CHARS == 300


def test_a_crash_is_printed_without_the_key_and_gives_exit_status_4(tmp_path, capsys):
    key_file = tmp_path / "key"
    key_file.write_text(KEY + "\n", encoding="ascii")
    try:
        raise RuntimeError(f"a bug that quotes {KEY}")
    except RuntimeError as error:
        assert run.crashed(error, str(key_file), "answer_requests") == run.EXIT_CRASHED == 4
    printed = capsys.readouterr().err
    assert printed.startswith("answer_requests: crashed (RuntimeError); run.json records exit 4\n"
                              "Traceback (most recent call last):\n")
    assert "a bug that quotes [redacted]" in printed and KEY not in printed


def test_a_line_is_appended_whole_and_a_file_is_hashed_in_chunks(tmp_path, monkeypatch):
    path = tmp_path / "log.jsonl"
    run.append_line(path, {"key": "a", "text": "café "})
    run.append_line(path, {"key": "b"})
    assert path.read_text(encoding="utf-8") == ('{"key": "a", "text": "café "}\n'
                                                '{"key": "b"}\n')
    monkeypatch.setattr(run, "CHUNK", 3)
    assert run.sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_text_is_written_whole_or_not_at_all_beside_its_file(tmp_path, monkeypatch):
    path = tmp_path / "answers.jsonl"
    assert run.temporary_path(path) == tmp_path / ".answers.jsonl.tmp"
    run.write_text_atomically(path, "one\n")
    run.write_text_atomically(path, "two\n")
    assert path.read_text(encoding="utf-8") == "two\n"
    assert [entry.name for entry in tmp_path.iterdir()] == ["answers.jsonl"]

    def interrupted(source, target):
        raise OSError("interrupted before the rename")

    monkeypatch.setattr(run.os, "replace", interrupted)
    with pytest.raises(OSError):
        run.write_text_atomically(path, "three\n")
    assert path.read_text(encoding="utf-8") == "two\n"
