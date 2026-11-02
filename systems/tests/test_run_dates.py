"""Grep at D's runner of the dates, on the host (grep-at-d/run_dates.py), on the toy world with its
streams registered, with a stand-in for `docker compose` whose runs answer with the reader's own
runner, in-process, and a stand-in for the answering call: for each arm and date it builds that
date's tree with `asofbench materialize`, in a fresh directory with a name that says nothing of
the date, outside the repository and readable by the sandbox's user, gives that tree and that
date's input and output directories alone to the date's run, and deletes the tree whatever the run
did; the count of api_error responses in a row goes on across the dates, so that five in a row
stop the run wherever they fall; it resumes a date that stopped, never one that api_error
responses stopped, skips a date that is done, stops on a run that the environment did not start
or a SIGTERM, takes the environment down whatever happened, holds the run directory's lock, checks
that grep at D's image is built on the reader image whose ID the run records, and refuses what it
cannot run before it builds anything. It needs git, which builds the toy world's repositories, and
the asofbench package, so the images skip it; it makes no model call and reads no v0.1 file."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import signal
import stat
import time
from pathlib import Path

import grep_modules
import pytest

pytest.importorskip("asofbench")
if shutil.which("git") is None:
    pytest.skip("the toy world's repositories need git", allow_module_level=True)

import harness  # noqa: E402 - systems/ is on the path (conftest.py)
from asofbench_systems import run as plumbing  # noqa: E402
from reader import answer_requests  # noqa: E402

from asofbench import cli  # noqa: E402

run_dates = grep_modules.run_dates()

ROOT = Path(__file__).resolve().parents[2]
MODEL = "claude-haiku-5-5"
IDS = {"grep-at-d": "sha256:" + "4" * 64, "reader": "sha256:" + "1" * 64,
       "egress": "sha256:" + "2" * 64, "sandbox": "sha256:" + "3" * 64}
IMAGES = ",".join(f"{name}={image}" for name, image in IDS.items())
# The images' layers as docker reports them: grep at D's begin with the reader's.
LAYERS = {IDS["reader"]: ["sha256:" + "a" * 64, "sha256:" + "b" * 64],
          IDS["grep-at-d"]: ["sha256:" + "a" * 64, "sha256:" + "b" * 64, "sha256:" + "c" * 64]}
EGRESS_LINE = (b'{"ts": "2026-11-02T00:00:00Z", "client": "172.20.0.3", "method": "CONNECT", '
               b'"target": "api.anthropic.com:443", "allowed": true, "reason": "allowlisted"}\n')
# A toy layout with three requests in each date's file: four items, one date each, in every arm
# that they list (the CF item in CF, the F1 items in H+ and H-), each three times.
ITEMS = ["F1-change-PEP-1-2003-08-24", "F1-change-PEP-2-2002-06-20", "F1-change-EIP-1-2017-11-25",
         "CF-counterfactual-PEP-1-2013-09-02"]


def failure(module: str, name: str) -> type[Exception]:
    """An exception class that stands in for one of a package's, by its module and name."""
    return type(name, (Exception,), {"__module__": module})


Unreachable = failure("anthropic._exceptions", "APIConnectionError")


def files_of(directory: Path) -> dict[str, bytes]:
    return {str(path.relative_to(directory)): path.read_bytes()
            for path in sorted(directory.rglob("*")) if path.is_file()}


def modes_of(directory: Path) -> dict[str, int]:
    return {str(path.relative_to(directory)): stat.S_IMODE(path.stat().st_mode)
            for path in [directory, *sorted(directory.rglob("*"))]}


class Client:
    async def close(self) -> None:
        pass


class Asker:
    """Stands in for the answering call: answers the date's requests one at a time, in turn, with
    ``outcomes`` (None for an answer, an error code, or an exception to raise), then answers."""

    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)

    async def __call__(self, client, model, question, context):
        outcome = self.outcomes.pop(0) if self.outcomes else None
        if isinstance(outcome, BaseException):
            raise outcome
        return {"answer": None if outcome else "Draft", "tokens_in": 10, "tokens_out": 2,
                "wall_s": 1.0, "tool_calls": 2, "error": outcome, "thinking_blocks": 1}


class Environment:
    """Stands in for `docker compose` on grep at D's compose file: records each command with the
    environment it was given, and the tree that each run was given as it stood then (its files
    and their modes); answers a run with the reader's runner, in-process, one request at a time,
    each request of a date getting the next of ``outcomes[date]``; starts no run for a date in
    ``unstarted``; and calls ``logs`` and answers ``down`` with ``down_status``."""

    def __init__(self, key_file: Path, outcomes: dict | None = None,
                 unstarted: tuple[str, ...] = (), logs=None, down_status: int = 0) -> None:
        self.key_file, self.outcomes, self.unstarted = key_file, outcomes or {}, unstarted
        self.logs, self.down_status = logs, down_status
        self.commands: list[tuple[str, str, list[str]]] = []
        self.environments: list[dict] = []
        self.trees: dict[str, tuple[Path, dict[str, bytes], dict[str, int]]] = {}

    def __call__(self, project, arguments, env, stdout=None) -> int:
        name = Path(env["ASOFBENCH_RUN_INPUT"]).name
        self.commands.append((name, arguments[0], list(arguments)))
        self.environments.append(dict(env))
        if arguments[0] == "logs":
            if self.logs is not None:
                return self.logs(stdout)
            stdout.write(EGRESS_LINE)
            return 0
        if arguments[0] == "down":
            return self.down_status
        if name in self.unstarted:
            return 1  # docker compose's own failure: no run began
        tree = Path(env["ASOFBENCH_RUN_TREE"])
        self.trees[name] = (tree, files_of(tree), modes_of(tree))
        flags = dict(zip(arguments[4::2], arguments[5::2], strict=False))
        images = plumbing.parse_image_ids(env["ASOFBENCH_IMAGE_IDS"], run_dates.IMAGE_NAMES,
                                          "README.md")
        try:
            return asyncio.run(answer_requests.run(
                Path(env["ASOFBENCH_RUN_INPUT"]) / "requests.jsonl",
                Path(env["ASOFBENCH_RUN_OUTPUT"]) / "responses.jsonl", flags["--model"],
                resume="--resume" in arguments, images=images,
                open_client=lambda log, key: Client(), ask=Asker(self.outcomes.get(name, [])),
                preflight=None, key_file=str(self.key_file), concurrency=1,
                progress=lambda line: None,
                api_errors_before=int(flags["--api-errors-before"])))
        except plumbing.Refused:
            return plumbing.EXIT_STOPPED
        except BaseException:  # noqa: BLE001 - a crash, which the runner recorded in run.json
            return plumbing.EXIT_CRASHED

    def runs(self) -> list[str]:
        return [name for name, command, _ in self.commands if command == "run"]

    def carried(self, name: str) -> list[str]:
        return [arguments[arguments.index("--api-errors-before") + 1]
                for date, command, arguments in self.commands
                if command == "run" and date == name]


@pytest.fixture
def setup(world, tmp_path):
    """Grep at D's requests on the toy world's four items, an empty directory for the trees, and
    the environment that run_dates.py reads: the key file's path and the images' IDs."""
    run_dir = tmp_path / "run"
    items = tmp_path / "items.json"
    items.write_text(json.dumps(ITEMS), encoding="utf-8")
    assert harness.main(["requests", "--system", "grep-at-d", "--data",
                         str(world / "toy-data"), "--out-dir", str(run_dir), "--items",
                         str(items)]) == 0
    trees = tmp_path / "trees"
    trees.mkdir()
    key_file = tmp_path / "key"
    key_file.write_text("sk-ant-api00-made-up-key\n", encoding="ascii")
    environ = {"PATH": "/usr/bin:/bin", "ASOFBENCH_ANTHROPIC_KEY_FILE": str(key_file),
               "ASOFBENCH_IMAGE_IDS": IMAGES}
    dates = json.loads((run_dir / "dates.json").read_text(encoding="utf-8"))
    # The toy layout's files hold at most four requests each.
    assert len(dates) == 7
    for name in dates:
        assert len((run_dir / "input" / name / "requests.jsonl").read_text(
            encoding="utf-8").splitlines()) <= 4
    return {"run_dir": run_dir, "trees": trees, "environ": environ, "world": world,
            "key_file": key_file, "dates": dates}


def environment(setup, **settings) -> Environment:
    return Environment(setup["key_file"], **settings)


def go(setup, compose, **changes) -> int:
    arguments = {"run_dir": setup["run_dir"], "data": setup["world"] / "toy-data",
                 "streams_dir": setup["world"] / "streams", "trees_dir": setup["trees"],
                 "model": MODEL, "project": "asofbench-grep-at-d-test", "compose": compose,
                 "environ": setup["environ"], "inspect_layers": LAYERS.__getitem__,
                 "progress": lambda line: None, **changes}
    return run_dates.run(arguments.pop("run_dir"), arguments.pop("data"),
                         arguments.pop("streams_dir"), arguments.pop("trees_dir"),
                         arguments.pop("model"), **arguments)


def responses_of(setup, name: str) -> list[dict]:
    path = setup["run_dir"] / "output" / name / "responses.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_each_date_s_tree_is_built_for_it_used_for_its_requests_alone_and_deleted(setup,
                                                                                tmp_path):
    compose = environment(setup)
    previous = os.umask(0o077)  # a strict umask: the tree must still be readable by uid 10001
    try:
        assert go(setup, compose) == 0
    finally:
        os.umask(previous)
    dates = setup["dates"]
    # Every date ran once, in dates.json's order, and its commands were the run, the saving of
    # the proxy's log and the environment's removal, in that order.
    assert compose.runs() == list(dates)
    assert [(name, command) for name, command, _ in compose.commands] == [
        (name, command) for name in dates for command in ("run", "logs", "down")]
    for name, entry in dates.items():
        tree, held, modes = compose.trees[name]
        # A fresh directory under the trees directory, outside the repository, whose name says
        # nothing of the arm or the date, since its path shows in the sandbox's mount table.
        assert tree.parent == setup["trees"] and not harness.inside_repository(tree)
        assert re.fullmatch(r"tree-[a-z0-9_]+", tree.name), tree.name
        for word in (name, entry["as_of"], entry["arm"]):
            assert word not in str(tree), word
        # It held what `asofbench materialize` writes for the date's arm and date, and nothing
        # else, every directory 0755 and every file 0644, so that uid 10001 can read it.
        expected = tmp_path / "expected" / name
        stream = setup["world"] / "streams" / harness.STREAM_FILES[entry["arm"]]
        assert cli.main(["materialize", "--stream", str(stream), "--as-of", entry["as_of"],
                         "--out", str(expected)]) == 0
        assert held == files_of(expected), name
        assert "git-log.txt" in held
        assert modes == {path: (0o644 if path in held else 0o755) for path in modes}
        assert not tree.exists()
    assert list(setup["trees"].iterdir()) == []
    # Each run was given its own date's input and output directories, absolute, and the tree.
    runs = [env for env, (_, command, _) in zip(compose.environments, compose.commands,
                                                 strict=True) if command == "run"]
    for env, name in zip(runs, dates, strict=True):
        assert env["ASOFBENCH_RUN_INPUT"] == str((setup["run_dir"] / "input" / name).resolve())
        assert env["ASOFBENCH_RUN_OUTPUT"] == str((setup["run_dir"] / "output" / name).resolve())
        assert Path(env["ASOFBENCH_RUN_TREE"]).is_absolute()
        assert env["ASOFBENCH_IMAGE_IDS"] == IMAGES
        assert env["ASOFBENCH_ANTHROPIC_KEY_FILE"] == (
            setup["environ"]["ASOFBENCH_ANTHROPIC_KEY_FILE"])
    (first,) = [arguments for name, command, arguments in compose.commands
                if command == "run" and name == next(iter(dates))]
    assert first == ["run", "--rm", "agent", "grep_at_d", "--requests", "/input/requests.jsonl",
                     "--out", "/output/responses.jsonl", "--model", MODEL,
                     "--api-errors-before", "0"]
    for name in dates:
        assert (setup["run_dir"] / "output" / name / "egress.log").read_bytes() == EGRESS_LINE
    # The run directory's lock is gone, and the rows of every date are whole.
    assert not (setup["run_dir"] / "run.lock").exists()
    out = tmp_path / "answers.jsonl"
    assert harness.main(["rows", "--system", "grep-at-d", "--run-dir", str(setup["run_dir"]),
                         "--out", str(out)]) == 0


def test_five_api_errors_in_a_row_across_dates_stop_the_run_with_exit_status_3(setup):
    names = list(setup["dates"])
    lines = []
    compose = environment(setup, outcomes={names[0]: ["api_error"] * 3,
                                           names[1]: ["api_error"] * 3})
    assert go(setup, compose, progress=lines.append) == plumbing.EXIT_FAILURES
    # The first date's three, then the second date's first two: the fifth in a row stops it.
    assert compose.runs() == names[:2]
    assert compose.carried(names[1]) == ["3"]
    assert [r["error"] for r in responses_of(setup, names[1])] == ["api_error", "api_error"]
    record = json.loads((setup["run_dir"] / "output" / names[1] / "run.json").read_text(
        encoding="utf-8"))
    assert record["invocations"][-1]["stop"] == {"error": "5 consecutive api_error responses",
                                                "api_errors_before": 3}
    assert any("discard the run" in line for line in lines)
    assert list(setup["trees"].iterdir()) == []
    # It is never resumed: the next run stops at once.
    again = environment(setup)
    assert go(setup, again) == plumbing.EXIT_FAILURES and again.commands == []


def test_four_api_errors_in_a_row_and_then_an_answer_do_not_stop_the_run(setup):
    names = list(setup["dates"])
    compose = environment(setup, outcomes={names[0]: ["api_error"] * 3,
                                           names[1]: ["api_error", None, "api_error"],
                                           names[2]: ["api_error"] * 3})
    assert go(setup, compose) == plumbing.EXIT_FAILED
    assert compose.runs() == names
    # Each date's runner is given the api_error responses in a row that the dates before it
    # ended with: 0, then 3 (the first date's), 1 (the second date's last), then 4 (that one and
    # the third date's three).
    assert [compose.carried(name) for name in names[:4]] == [["0"], ["3"], ["1"], ["4"]]


def test_a_resumed_run_carries_the_count_from_the_dates_already_done(setup):
    names = list(setup["dates"])
    first = environment(setup, outcomes={names[0]: ["api_error"] * 3,
                                         names[1]: ["api_error", Unreachable("down")]})
    assert go(setup, first) == plumbing.EXIT_STOPPED
    assert [r["error"] for r in responses_of(setup, names[1])] == ["api_error"]
    # Resumed: the first date is done and skipped, and its three api_error responses, with the
    # second date's one, make the second date's next api_error the fifth in a row.
    second = environment(setup, outcomes={names[1]: ["api_error"]})
    assert go(setup, second) == plumbing.EXIT_FAILURES
    assert second.runs() == [names[1]] and second.carried(names[1]) == ["3"]
    assert [r["error"] for r in responses_of(setup, names[1])] == ["api_error", "api_error"]


def test_a_stopped_date_stops_the_run_and_the_next_run_resumes_it(setup):
    names = list(setup["dates"])
    stopped = names[2]
    first = environment(setup, outcomes={stopped: [None, Unreachable("down")]})
    assert go(setup, first) == plumbing.EXIT_STOPPED
    assert first.runs() == names[:3]
    assert list(setup["trees"].iterdir()) == []
    second = environment(setup)
    assert go(setup, second) == 0
    # The dates that were done are skipped; the stopped one resumes; the rest begin.
    assert second.runs() == names[2:]
    (resumed,) = [arguments for name, command, arguments in second.commands
                  if command == "run" and name == stopped]
    assert resumed[-1] == "--resume"
    # The resumed date's second proxy log is kept beside its first.
    output = setup["run_dir"] / "output" / stopped
    assert (output / "egress.log").exists() and (output / "egress-1.log").exists()
    # A third run has nothing left to do.
    third = environment(setup)
    assert go(setup, third) == 0 and third.commands == []


def test_a_crash_stops_the_run_and_the_date_is_resumed(setup):
    names = list(setup["dates"])
    assert go(setup, environment(setup, outcomes={names[0]: [ValueError("a bug")]})) == 4
    resumed = environment(setup)
    assert go(setup, resumed) == 0 and resumed.runs() == names


def test_a_run_that_the_environment_did_not_start_stops_the_run(setup):
    names = list(setup["dates"])
    compose, lines = environment(setup, unstarted=(names[0],)), []
    assert go(setup, compose, progress=lines.append) == 2
    assert compose.runs() == [names[0]]
    assert [command for _, command, _ in compose.commands] == ["run", "logs", "down"]
    assert any("did not start" in line for line in lines)
    assert list(setup["trees"].iterdir()) == []


def test_a_date_with_requests_that_were_not_answered_is_done_and_the_run_exits_1(setup):
    names = list(setup["dates"])
    compose = environment(setup, outcomes={names[0]: ["refusal"]})
    assert go(setup, compose) == 1 and compose.runs() == names
    assert go(setup, environment(setup)) == 1  # done, and still with that request unanswered


def test_the_environment_is_taken_down_and_the_tree_deleted_when_the_log_cannot_be_saved(setup):
    def failing(stdout) -> int:
        raise OSError("the disk filled")

    compose = environment(setup, logs=failing)
    with pytest.raises(OSError, match="the disk filled"):
        go(setup, compose)
    assert [command for _, command, _ in compose.commands] == ["run", "logs", "down"]
    assert list(setup["trees"].iterdir()) == []
    assert not (setup["run_dir"] / "run.lock").exists()


def test_an_environment_that_cannot_be_taken_down_stops_the_run(setup):
    lines = []
    compose = environment(setup, down_status=1)
    assert go(setup, compose, progress=lines.append) == plumbing.EXIT_STOPPED
    assert compose.runs() == [next(iter(setup["dates"]))]
    assert any("taken down" in line for line in lines)
    assert list(setup["trees"].iterdir()) == []


def test_a_sigterm_stops_the_run_after_the_environment_is_down_and_the_tree_deleted(setup):
    names = list(setup["dates"])
    compose = environment(setup)

    def terminated(project, arguments, env, stdout=None):
        code = compose(project, arguments, env, stdout)
        if arguments[0] == "run" and Path(env["ASOFBENCH_RUN_INPUT"]).name == names[1]:
            raise run_dates.Terminated()
        return code

    lines = []
    assert go(setup, terminated, progress=lines.append) == plumbing.EXIT_STOPPED
    assert compose.runs() == names[:2]
    assert [command for name, command, _ in compose.commands if name == names[1]] == [
        "run", "logs", "down"]
    assert list(setup["trees"].iterdir()) == [] and not (setup["run_dir"] / "run.lock").exists()
    assert any("SIGTERM" in line for line in lines)
    # The handler turns the signal into Terminated, so that every finally block runs, and puts
    # the earlier handler back.
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(run_dates.Terminated), run_dates.terminated_on_sigterm():
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(5)
    assert signal.getsignal(signal.SIGTERM) == before


def test_a_second_invocation_is_refused_while_one_holds_the_run_directory_s_lock(setup):
    (setup["run_dir"] / "run.lock").write_text(json.dumps({"command": "run_dates"}) + "\n",
                                                encoding="utf-8")
    compose = environment(setup)
    with pytest.raises(run_dates.Refused, match=r"systems/README\.md says how to check"):
        go(setup, compose)
    assert compose.commands == [] and list(setup["trees"].iterdir()) == []


def test_a_tree_that_cannot_be_built_stops_the_run_before_its_date_runs(setup, tmp_path):
    # An H- stream that is not the registered one, which `asofbench materialize --arm --data`
    # refuses, so that no tree is built from it.
    streams = tmp_path / "streams-with-another-h-minus"
    shutil.copytree(setup["world"] / "streams", streams)
    with open(streams / "h-minus.jsonl", "a", encoding="utf-8") as handle:
        handle.write("\n")
    names = list(setup["dates"])
    first_h_minus = next(n for n in names if setup["dates"][n]["arm"] == "H-")
    compose = environment(setup)
    assert go(setup, compose, streams_dir=streams) == 2
    assert compose.runs() == names[:names.index(first_h_minus)]
    assert list(setup["trees"].iterdir()) == []


def test_what_it_cannot_run_is_refused_before_any_tree_is_built(setup, tmp_path):
    def refused(**changes) -> str:
        compose = environment(setup)
        with pytest.raises(run_dates.Refused) as caught:
            go(setup, compose, **changes)
        assert compose.commands == [] and list(setup["trees"].iterdir()) == []
        assert not (setup["run_dir"] / "run.lock").exists()
        return str(caught.value)

    inside = ROOT / "systems" / "tests"
    assert "inside the repository" in refused(trees_dir=inside)
    assert "is not a directory" in refused(trees_dir=tmp_path / "missing")
    assert "not an answering model" in refused(model="claude-opus-5-5")
    streams = tmp_path / "streams-without-h-minus"
    shutil.copytree(setup["world"] / "streams", streams)
    (streams / "h-minus.jsonl").unlink()
    assert "h-minus.jsonl" in refused(streams_dir=streams)
    environ = dict(setup["environ"])
    environ["ASOFBENCH_IMAGE_IDS"] = IMAGES.rsplit(",", 1)[0]
    assert "grep-at-d, reader, egress, sandbox" in refused(environ=environ)
    environ = {k: v for k, v in setup["environ"].items() if k != "ASOFBENCH_ANTHROPIC_KEY_FILE"}
    assert "ASOFBENCH_ANTHROPIC_KEY_FILE" in refused(environ=environ)
    # Grep at D's image not built on the reader image whose ID the run records, and an image
    # that docker cannot inspect.
    other = {**LAYERS, IDS["grep-at-d"]: ["sha256:" + "d" * 64, "sha256:" + "c" * 64]}
    assert "built on" in refused(inspect_layers=other.__getitem__)

    def uninspectable(image: str) -> list[str]:
        raise run_dates.Refused(f"docker cannot inspect the image {image}")

    assert "cannot inspect" in refused(inspect_layers=uninspectable)
    # A build that is not the one whose questions the requests hold.
    other_data = tmp_path / "other-data"
    shutil.copytree(setup["world"] / "toy-data", other_data)
    with open(other_data / "MANIFEST.json", "a", encoding="utf-8") as handle:
        handle.write("\n")
    assert "MANIFEST.json" in refused(data=other_data)
    run_dir = setup["run_dir"]
    # A requests file, or dates.json, that is not the one plan.json records.
    path = run_dir / "input" / next(iter(setup["dates"])) / "requests.jsonl"
    kept = path.read_bytes()
    path.write_bytes(kept + b"\n")
    assert "requests.jsonl" in refused()
    path.write_bytes(kept)
    dates = run_dir / "dates.json"
    kept = dates.read_bytes()
    dates.write_bytes(kept + b"\n")
    assert "dates.json" in refused()
    dates.unlink()
    assert "dates.json" in refused()
    dates.write_bytes(kept)
    # Another system's run directory.
    closed = tmp_path / "closed"
    assert harness.main(["requests", "--system", "closed-book", "--data",
                         str(setup["world"] / "toy-data"), "--out-dir", str(closed)]) == 0
    assert "grep-at-d" in refused(run_dir=closed)
    assert go(setup, environment(setup)) == 0  # and the run itself was fine all along


def test_the_command_line_runs_the_dates_and_reports_a_refusal(setup, monkeypatch, capsys):
    # The images that it checks for are those that the agent records.
    assert run_dates.IMAGE_NAMES == grep_modules.agent().IMAGE_NAMES
    monkeypatch.setattr(run_dates, "docker_compose", environment(setup))
    monkeypatch.setattr(run_dates, "image_layers", LAYERS.__getitem__)
    for name, value in setup["environ"].items():
        monkeypatch.setenv(name, value)
    argv = ["--run-dir", str(setup["run_dir"]), "--data", str(setup["world"] / "toy-data"),
            "--streams-dir", str(setup["world"] / "streams"), "--trees-dir", str(setup["trees"]),
            "--model", MODEL]
    assert run_dates.main(argv) == 0
    monkeypatch.delenv("ASOFBENCH_IMAGE_IDS")
    assert run_dates.main(argv) == 2
    assert "ASOFBENCH_IMAGE_IDS" in capsys.readouterr().err
    with pytest.raises(SystemExit) as caught:
        run_dates.main(argv[:-2])
    assert caught.value.code == 2
