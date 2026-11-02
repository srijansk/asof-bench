"""Grep at D's runner of the dates, on the host (grep-at-d/run_dates.py), on the toy world with its
streams registered, with a stand-in for `docker compose` that answers as the reader would: for
each arm and date it builds that date's tree with `asofbench materialize` in a fresh directory
outside the repository, gives that tree, and that date's input and output directories alone, to
the date's run, and deletes the tree whatever the run did; it resumes a date that stopped, never
one that five api_error responses in a row stopped, skips a date that is done, stops on a run that
the environment did not start, and refuses what it cannot run before it builds anything. It needs
git, which builds the toy world's repositories, and the asofbench package, so the reader image
skips it; it makes no model call and reads no v0.1 file."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import grep_modules
import pytest

pytest.importorskip("asofbench")
if shutil.which("git") is None:
    pytest.skip("the toy world's repositories need git", allow_module_level=True)

import harness  # noqa: E402 - systems/ is on the path (conftest.py)

from asofbench import cli  # noqa: E402

run_dates = grep_modules.run_dates()

ROOT = Path(__file__).resolve().parents[2]
MODEL = "claude-haiku-5-5"
IMAGES = "reader=sha256:" + "1" * 64 + ",egress=sha256:" + "2" * 64 + ",sandbox=sha256:" + "3" * 64
EGRESS_LINE = (b'{"ts": "2026-11-02T00:00:00Z", "client": "172.20.0.3", "method": "CONNECT", '
               b'"target": "api.anthropic.com:443", "allowed": true, "reason": "allowlisted"}\n')


def files_of(directory: Path) -> dict[str, bytes]:
    return {str(path.relative_to(directory)): path.read_bytes()
            for path in sorted(directory.rglob("*")) if path.is_file()}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Environment:
    """Stands in for `docker compose` on grep at D's compose file: records each command with the
    environment it was given, records the tree that each run was given as it stood then, and
    answers the run's requests as the reader would, with the exit status of ``exits[date]``
    (0 by default; for 2, 3 and 4 it answers none of them), or starts no run at all for a date
    in ``unstarted``."""

    def __init__(self, exits: dict[str, int] | None = None, unstarted: tuple[str, ...] = ()):
        self.exits, self.unstarted = exits or {}, unstarted
        self.commands: list[tuple[str, str, list[str]]] = []
        self.trees: dict[str, tuple[Path, dict[str, bytes]]] = {}
        self.environments: list[dict] = []

    def __call__(self, project, arguments, env, stdout=None) -> int:
        name = Path(env["ASOFBENCH_RUN_INPUT"]).name
        self.commands.append((name, arguments[0], list(arguments)))
        self.environments.append(dict(env))
        if arguments[0] == "logs":
            stdout.write(EGRESS_LINE)
            return 0
        if arguments[0] != "run":
            return 0
        if name in self.unstarted:
            return 1  # docker compose's own failure: no run began
        tree = Path(env["ASOFBENCH_RUN_TREE"])
        self.trees[name] = (tree, files_of(tree))
        inputs, output = Path(env["ASOFBENCH_RUN_INPUT"]), Path(env["ASOFBENCH_RUN_OUTPUT"])
        record_path = output / "run.json"
        record = (json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists()
                  else {"requests": "requests.jsonl",
                        "requests_sha256": sha256(inputs / "requests.jsonl"),
                        "responses": "responses.jsonl", "model": MODEL, "invocations": []})
        code = self.exits.get(name, 0)
        record["invocations"].append({"command": "answer", "resume": "--resume" in arguments,
                                      "exit": code})
        record_path.write_text(json.dumps(record), encoding="utf-8")
        if code in (0, 1):
            keys = [json.loads(line)["key"] for line in
                    (inputs / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
            with open(output / "responses.jsonl", "w", encoding="utf-8") as handle:
                for key in keys:
                    handle.write(json.dumps({"key": key, "answer": "Draft", "tokens_in": 10,
                                             "tokens_out": 2, "wall_s": 1.0, "tool_calls": 2,
                                             "error": "refusal" if code else None,
                                             "thinking_blocks": 1}) + "\n")
        return code

    def runs(self) -> list[str]:
        return [name for name, command, _ in self.commands if command == "run"]


@pytest.fixture
def setup(world, tmp_path, monkeypatch):
    """Grep at D's requests on the toy world, an empty directory for the trees, and the
    environment that run_dates.py reads: the key file's path and the images' IDs."""
    run_dir = tmp_path / "run"
    assert harness.main(["requests", "--system", "grep-at-d", "--data",
                         str(world / "toy-data"), "--out-dir", str(run_dir)]) == 0
    trees = tmp_path / "trees"
    trees.mkdir()
    key_file = tmp_path / "key"
    key_file.write_text("sk-ant-api00-made-up-key\n", encoding="ascii")
    environ = {"PATH": "/usr/bin:/bin", "ASOFBENCH_ANTHROPIC_KEY_FILE": str(key_file),
               "ASOFBENCH_IMAGE_IDS": IMAGES}
    return {"run_dir": run_dir, "trees": trees, "environ": environ, "world": world,
            "dates": json.loads((run_dir / "dates.json").read_text(encoding="utf-8"))}


def go(setup, environment: Environment, **changes) -> int:
    arguments = {"run_dir": setup["run_dir"], "data": setup["world"] / "toy-data",
                 "streams_dir": setup["world"] / "streams", "trees_dir": setup["trees"],
                 "model": MODEL, "project": "asofbench-grep-at-d-test",
                 "compose": environment, "environ": setup["environ"],
                 "progress": lambda line: None, **changes}
    return run_dates.run(arguments.pop("run_dir"), arguments.pop("data"),
                         arguments.pop("streams_dir"), arguments.pop("trees_dir"),
                         arguments.pop("model"), **arguments)


def test_each_date_s_tree_is_built_for_it_used_for_its_requests_alone_and_deleted(setup,
                                                                                tmp_path):
    environment = Environment()
    assert go(setup, environment) == 0
    dates = setup["dates"]
    # Every date ran once, in dates.json's order, and its commands were the run, the saving of
    # the proxy's log and the environment's removal, in that order.
    assert environment.runs() == list(dates)
    assert [(name, command) for name, command, _ in environment.commands] == [
        (name, command) for name in dates for command in ("run", "logs", "down")]
    for name, entry in dates.items():
        tree, held = environment.trees[name]
        # The tree was a fresh directory under the trees directory, outside the repository, and
        # held what `asofbench materialize` writes for the date's arm and date, and nothing else.
        assert tree.parent == setup["trees"] and not harness.inside_repository(tree)
        expected = tmp_path / "expected" / name
        stream = setup["world"] / "streams" / harness.STREAM_FILES[entry["arm"]]
        assert cli.main(["materialize", "--stream", str(stream), "--as-of", entry["as_of"],
                         "--out", str(expected)]) == 0
        assert held == files_of(expected), name
        assert "git-log.txt" in held
        assert not tree.exists()
    assert list(setup["trees"].iterdir()) == []
    # Each run was given its own date's input and output directories, absolute, and the tree.
    runs = [env for env, (_, command, _) in zip(environment.environments, environment.commands,
                                                 strict=True) if command == "run"]
    for env, name in zip(runs, dates, strict=True):
        assert env["ASOFBENCH_RUN_INPUT"] == str((setup["run_dir"] / "input" / name).resolve())
        assert env["ASOFBENCH_RUN_OUTPUT"] == str((setup["run_dir"] / "output" / name).resolve())
        assert Path(env["ASOFBENCH_RUN_TREE"]).is_absolute()
        assert env["ASOFBENCH_IMAGE_IDS"] == IMAGES
        assert env["ASOFBENCH_ANTHROPIC_KEY_FILE"] == (
            setup["environ"]["ASOFBENCH_ANTHROPIC_KEY_FILE"])
    (first,) = [arguments for name, command, arguments in environment.commands
                if command == "run" and name == next(iter(dates))]
    assert first == ["run", "--rm", "agent", "grep_at_d", "--requests", "/input/requests.jsonl",
                     "--out", "/output/responses.jsonl", "--model", MODEL]
    # The proxy's log is saved beside each date's responses.
    for name in dates:
        assert (setup["run_dir"] / "output" / name / "egress.log").read_bytes() == EGRESS_LINE
    # And the rows of every date are whole.
    out = tmp_path / "answers.jsonl"
    assert harness.main(["rows", "--system", "grep-at-d", "--run-dir", str(setup["run_dir"]),
                         "--out", str(out)]) == 0


def test_a_stopped_date_stops_the_run_and_the_next_run_resumes_it(setup):
    names = list(setup["dates"])
    stopped = names[2]
    first = Environment(exits={stopped: 2})
    assert go(setup, first) == 2
    assert first.runs() == names[:3]
    assert list(setup["trees"].iterdir()) == []
    second = Environment()
    assert go(setup, second) == 0
    # The dates that were done are skipped; the stopped one resumes; the rest begin.
    assert second.runs() == names[2:]
    resumed = [arguments for name, command, arguments in second.commands
               if command == "run" and name == stopped]
    assert resumed == [["run", "--rm", "agent", "grep_at_d", "--requests",
                        "/input/requests.jsonl", "--out", "/output/responses.jsonl", "--model",
                        MODEL, "--resume"]]
    # The resumed date's second proxy log is kept beside its first.
    output = setup["run_dir"] / "output" / stopped
    assert (output / "egress.log").exists() and (output / "egress-1.log").exists()
    # A third run has nothing left to do.
    third = Environment()
    assert go(setup, third) == 0 and third.commands == []


def test_a_date_stopped_by_api_errors_in_a_row_stops_the_run_and_is_never_resumed(setup):
    names = list(setup["dates"])
    lines = []
    assert go(setup, Environment(exits={names[1]: 3}), progress=lines.append) == 3
    assert any("discard" in line for line in lines)
    again, lines = Environment(), []
    assert go(setup, again, progress=lines.append) == 3
    assert again.commands == [] and list(setup["trees"].iterdir()) == []
    assert any("discard" in line and names[1] in line for line in lines)


def test_a_crash_or_a_kill_stops_the_run_and_the_date_is_resumed(setup):
    names = list(setup["dates"])
    assert go(setup, Environment(exits={names[0]: 4})) == 4
    resumed = Environment()
    assert go(setup, resumed) == 0 and resumed.runs() == names


def test_a_run_that_the_environment_did_not_start_stops_the_run(setup):
    names = list(setup["dates"])
    environment, lines = Environment(unstarted=(names[0],)), []
    assert go(setup, environment, progress=lines.append) == 2
    assert environment.runs() == [names[0]]
    assert [command for _, command, _ in environment.commands] == ["run", "logs", "down"]
    assert any("did not start" in line for line in lines)
    assert list(setup["trees"].iterdir()) == []


def test_a_date_with_requests_that_were_not_answered_is_done_and_the_run_exits_1(setup):
    names = list(setup["dates"])
    environment = Environment(exits={names[0]: 1})
    assert go(setup, environment) == 1 and environment.runs() == names
    assert go(setup, Environment()) == 1  # done, and still with those requests unanswered


def test_a_tree_that_cannot_be_built_stops_the_run_before_its_date_runs(setup, tmp_path):
    # An H- stream that is not the registered one, which `asofbench materialize --arm --data`
    # refuses, so that no tree is built from it.
    streams = tmp_path / "streams-with-another-h-minus"
    shutil.copytree(setup["world"] / "streams", streams)
    with open(streams / "h-minus.jsonl", "a", encoding="utf-8") as handle:
        handle.write("\n")
    names = list(setup["dates"])
    first_h_minus = next(n for n in names if setup["dates"][n]["arm"] == "H-")
    environment = Environment()
    assert go(setup, environment, streams_dir=streams) == 2
    assert environment.runs() == names[:names.index(first_h_minus)]
    assert list(setup["trees"].iterdir()) == []


def test_what_it_cannot_run_is_refused_before_any_tree_is_built(setup, tmp_path):
    def refused(**changes) -> str:
        environment = Environment()
        with pytest.raises(run_dates.Refused) as caught:
            go(setup, environment, **changes)
        assert environment.commands == [] and list(setup["trees"].iterdir()) == []
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
    assert "reader, egress, sandbox" in refused(environ=environ)
    environ = {k: v for k, v in setup["environ"].items() if k != "ASOFBENCH_ANTHROPIC_KEY_FILE"}
    assert "ASOFBENCH_ANTHROPIC_KEY_FILE" in refused(environ=environ)
    # A build that is not the one whose questions the requests hold.
    other = tmp_path / "other-data"
    shutil.copytree(setup["world"] / "toy-data", other)
    with open(other / "MANIFEST.json", "a", encoding="utf-8") as handle:
        handle.write("\n")
    assert "MANIFEST.json" in refused(data=other)
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
    assert go(setup, Environment()) == 0  # and the run itself was fine all along


def test_the_command_line_runs_the_dates_and_reports_a_refusal(setup, monkeypatch, capsys):
    # The images that it checks for are those that the agent records.
    assert run_dates.IMAGE_NAMES == grep_modules.agent().IMAGE_NAMES
    monkeypatch.setattr(run_dates, "docker_compose", Environment())
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
