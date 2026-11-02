"""The sandbox's command server (grep-at-d/sandbox/command_server.py), on a temporary tree and
scratch directory: each command runs under bash in the tree with none of the server's environment;
its exit status, stdout and stderr are reported apart, each cut at LIMIT characters with the exact
final line; a command that runs past the limit is killed with its whole process group, and so is
every background job that a command leaves; the scratch directory is empty when each command
starts; commands run one at a time; and the HTTP API answers POST /run with a JSON body and
refuses every other request, running nothing. Standard library only, with bash, which the sandbox
has too."""

from __future__ import annotations

import http.client
import json
import os
import shutil
import threading
import time
from pathlib import Path

import grep_modules
import pytest

if shutil.which("bash") is None:
    pytest.skip("the command server runs its commands under bash", allow_module_level=True)

server = grep_modules.command_server()

FIELDS = ["exit_code", "stdout", "stderr", "truncated", "timed_out"]


@pytest.fixture
def tree(tmp_path) -> tuple[Path, Path]:
    """A tree as `asofbench materialize` lays one out, and an empty scratch directory."""
    root = tmp_path / "repo"
    (root / "peps").mkdir(parents=True)
    (root / "peps" / "pep-0001").write_text("PEP: 1\nTitle: Proposal 1\nStatus: Draft\n",
                                            encoding="utf-8")
    (root / "git-log.txt").write_text("commit 0\nDate: 2001-01-01T12:00:00Z\n", encoding="utf-8")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    return root, scratch


def run(tree, command: str, **limits) -> dict:
    root, scratch = tree
    return server.run(command, root=root, scratch=scratch, **limits)


def gone(pid: int, within_s: float = 5.0) -> bool:
    """Whether process ``pid`` has ended: it no longer exists, or it is a zombie, dead but not yet
    reaped by the parent it was left to (inside a container whose first process is pytest, which
    reaps nothing, a killed orphan stays one)."""
    deadline = time.monotonic() + within_s
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        try:
            state = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            state = []
        if state[:1] == ["Z"]:
            return True
        time.sleep(0.05)
    return False


def test_the_sandbox_s_limits_are_twenty_seconds_and_twenty_thousand_characters():
    assert (server.TIMEOUT_S, server.LIMIT, server.PORT) == (20, 20_000, 7000)
    assert tuple(FIELDS) == server.FIELDS


def test_a_command_runs_under_bash_in_the_tree_and_reports_its_exit_code_and_both_streams(tree):
    root, _ = tree
    result = run(tree, 'test -n "$BASH_VERSION" && pwd -P; ls; grep -c Status peps/pep-0001; '
                       "echo 'an error' >&2; exit 3")
    assert list(result) == FIELDS
    assert result == {"exit_code": 3,
                      "stdout": f"{root.resolve()}\ngit-log.txt\npeps\n1\n",
                      "stderr": "an error\n", "truncated": False, "timed_out": False}
    assert run(tree, "false")["exit_code"] == 1
    assert run(tree, "true") == {"exit_code": 0, "stdout": "", "stderr": "", "truncated": False,
                                 "timed_out": False}


def test_a_command_sees_none_of_the_server_s_environment(tree, monkeypatch):
    _, scratch = tree
    monkeypatch.setenv("ASOFBENCH_PROBE_FROM_THE_SERVER", "not for the command")
    lines = run(tree, "env")["stdout"].splitlines()
    variables = dict(line.split("=", 1) for line in lines)
    assert "ASOFBENCH_PROBE_FROM_THE_SERVER" not in variables
    assert (variables["HOME"], variables["TMPDIR"]) == (str(scratch), str(scratch))
    assert variables["PATH"] == server.PATH
    assert variables["LANG"] == variables["LC_ALL"] == "C.UTF-8"
    # What bash itself adds, and nothing else.
    assert set(variables) - {"PATH", "LANG", "LC_ALL", "HOME", "TMPDIR"} <= {"PWD", "SHLVL", "_",
                                                                             "OLDPWD"}
    # And its stdin is empty.
    assert run(tree, "cat")["stdout"] == ""


def test_output_beyond_the_limit_is_cut_with_its_final_line(tree):
    root, _ = tree
    over = run(tree, "head -c 25000 /dev/zero | tr '\\0' a")
    assert over["stdout"] == "a" * 20_000 + "\n[cut: 5000 characters omitted]\n"
    assert (over["truncated"], over["timed_out"], over["exit_code"]) == (True, False, 0)
    # A cut at the end of a line adds no line break before the marker.
    lines = run(tree, "yes abcd | head -n 5000")
    assert lines["stdout"] == "abcd\n" * 4000 + "[cut: 5000 characters omitted]\n"
    # Exactly the limit is not cut, and stderr is cut on its own.
    exact = run(tree, "head -c 20000 /dev/zero | tr '\\0' a")
    assert (exact["stdout"], exact["truncated"]) == ("a" * 20_000, False)
    err = run(tree, "head -c 20001 /dev/zero | tr '\\0' b >&2")
    assert err["stdout"] == "" and err["stderr"] == "b" * 20_000 + "\n[cut: 1 characters omitted]\n"
    assert err["truncated"] is True
    # Characters are counted, not bytes, and a byte that is not UTF-8 is replaced.
    (root / "wide").write_text("é" * 20_005, encoding="utf-8")
    wide = run(tree, "cat wide")
    assert wide["stdout"] == "é" * 20_000 + "\n[cut: 5 characters omitted]\n"
    (root / "binary").write_bytes(b"ok\xff\xfe\n")
    assert run(tree, "cat binary")["stdout"] == "ok��\n"


def test_megabytes_of_output_are_cut_and_counted(tree):
    big = run(tree, "head -c 5000000 /dev/zero | tr '\\0' a")
    assert big["stdout"] == "a" * 20_000 + "\n[cut: 4980000 characters omitted]\n"
    assert big["truncated"] is True and big["exit_code"] == 0


def test_a_command_that_runs_past_the_limit_is_killed_with_its_process_group(tree):
    _, scratch = tree
    begun = time.monotonic()
    result = run(tree, 'sleep 60 & echo "$!" > "$TMPDIR/child"; echo started; sleep 60',
                 timeout_s=0.5)
    assert time.monotonic() - begun < 5
    assert (result["timed_out"], result["stdout"], result["stderr"]) == (True, "started\n", "")
    assert result["exit_code"] < 0  # bash, killed by the signal
    # The background job, in the command's process group, is killed with it.
    assert gone(int((scratch / "child").read_text(encoding="utf-8")))
    begun = time.monotonic()
    loop = run(tree, "yes > /dev/null", timeout_s=0.5)
    assert loop["timed_out"] is True and time.monotonic() - begun < 5


def test_a_background_job_that_a_command_leaves_is_killed_when_it_ends(tree):
    begun = time.monotonic()
    # The job holds the command's stdout open, and would keep the pipe from closing.
    held = run(tree, "sleep 60 & echo done")
    assert (held["stdout"], held["exit_code"], held["timed_out"]) == ("done\n", 0, False)
    detached = run(tree, 'sleep 60 > /dev/null 2>&1 & echo "$!"')
    assert time.monotonic() - begun < 5
    assert detached["timed_out"] is False and gone(int(detached["stdout"]))


def test_a_command_that_closes_its_output_still_runs_to_its_end_or_its_limit(tree):
    _, scratch = tree
    finished = run(tree, 'exec >&- 2>&-; sleep 0.3; touch "$TMPDIR/finished"')
    assert (finished["exit_code"], finished["timed_out"]) == (0, False)
    assert (scratch / "finished").exists()
    stopped = run(tree, "exec >&- 2>&-; sleep 60", timeout_s=0.5)
    assert stopped["timed_out"] is True


def test_the_scratch_directory_is_empty_when_each_command_starts(tree):
    first = run(tree, 'echo note > "$TMPDIR/note"; mkdir -p "$TMPDIR/d/e"; '
                      'touch "$TMPDIR/d/e/f" "$TMPDIR/.hidden"; chmod 000 "$TMPDIR/d"; '
                      'ls -A "$TMPDIR"')
    assert first["stdout"] == ".hidden\nd\nnote\n"
    second = run(tree, 'ls -A "$TMPDIR"; cat "$TMPDIR/note"')
    assert second["stdout"] == "" and second["exit_code"] == 1


@pytest.fixture
def live(tree):
    """The command server on a free local port, serving ``tree``."""
    root, scratch = tree
    httpd = server.CommandServer(("127.0.0.1", 0), root=root, scratch=scratch)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()
    thread.join()


def ask(httpd, method: str, path: str, body: bytes | None = None,
        headers: dict | None = None) -> tuple[int, dict]:
    connection = http.client.HTTPConnection(*httpd.server_address[:2], timeout=30)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, json.loads(response.read() or b"null")
    finally:
        connection.close()


def post(httpd, value) -> tuple[int, dict]:
    body = value if isinstance(value, bytes) else json.dumps(value).encode()
    return ask(httpd, "POST", "/run", body, {"Content-Type": "application/json"})


def test_post_run_with_a_json_body_runs_the_command(live):
    status, result = post(live, {"command": "cat peps/pep-0001 | grep Status"})
    assert (status, result) == (200, {"exit_code": 0, "stdout": "Status: Draft\n", "stderr": "",
                                      "truncated": False, "timed_out": False})


def raw(httpd, request: bytes) -> bytes:
    """Sends ``request`` as it is and returns what came back."""
    import socket

    with socket.create_connection(httpd.server_address[:2], timeout=30) as sock:
        sock.sendall(request)
        chunks = []
        while chunk := sock.recv(65536):
            chunks.append(chunk)
    return b"".join(chunks)


def test_every_other_request_is_refused_and_runs_nothing(live, tree):
    root, _ = tree
    marker = 'touch "$TMPDIR/../ran"'
    refused = [
        ask(live, "GET", "/run"),
        ask(live, "PUT", "/run", json.dumps({"command": marker}).encode()),
        ask(live, "DELETE", "/run"),
        ask(live, "POST", "/other", json.dumps({"command": marker}).encode()),
        ask(live, "POST", "/run?command=ls", json.dumps({"command": marker}).encode()),
        post(live, b"not json"),
        post(live, b"\xff\xfe"),
        post(live, []),
        post(live, {"cmd": marker}),
        post(live, {"command": 7}),
        post(live, {"command": marker, "timeout": 100}),
        post(live, {"command": "echo a\u0000b"}),
        post(live, {"command": "x" * (server.MAX_COMMAND_BYTES + 1)}),
    ]
    assert [status for status, _ in refused] == [405, 405, 405, 404, 404] + [400] * 8
    assert all(isinstance(body, dict) and set(body) == {"error"} for _, body in refused)
    # No Content-Length, and a body over the limit.
    head = raw(live, b"POST /run HTTP/1.1\r\nHost: sandbox\r\nConnection: close\r\n\r\n")
    assert head.startswith(b"HTTP/1.0 411 ")
    over = raw(live, b"POST /run HTTP/1.1\r\nHost: sandbox\r\nContent-Length: "
               + str(server.MAX_BODY + 1).encode() + b"\r\nConnection: close\r\n\r\n")
    assert over.startswith(b"HTTP/1.0 413 ")
    assert not (root.parent / "ran").exists()
    # A command of the longest length is run.
    longest = "true #" + "x" * (server.MAX_COMMAND_BYTES - 6)
    assert post(live, {"command": longest})[0] == 200


def test_commands_run_one_at_a_time(live, tree):
    root, _ = tree
    log = root.parent / "order"
    command = f'echo begin >> "{log}"; sleep 0.2; echo end >> "{log}"'
    results = []
    threads = [threading.Thread(target=lambda: results.append(post(live, {"command": command})))
               for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert [status for status, _ in results] == [200, 200, 200]
    assert log.read_text(encoding="utf-8").split() == ["begin", "end"] * 3


def test_a_command_that_cannot_start_is_a_failure_of_the_sandbox_not_a_result(live, tree):
    root, scratch = tree
    shutil.rmtree(root)  # the tree's directory is gone: bash cannot start in it
    status, body = post(live, {"command": "ls"})
    assert status == 500 and set(body) == {"error"}


def test_the_self_test_checks_the_tools_and_runs_a_command(monkeypatch, capsys):
    monkeypatch.setattr(server.shutil, "which", lambda tool: f"/usr/bin/{tool}")
    assert server.self_test() == 0
    assert "sandbox self-test: " in capsys.readouterr().out
    monkeypatch.setattr(server.shutil, "which",
                        lambda tool: None if tool == "gawk" else f"/usr/bin/{tool}")
    assert server.self_test() == 1
    assert "gawk" in capsys.readouterr().err
