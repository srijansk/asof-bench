"""The sandbox of grep at D: a server that runs the agent's read-only shell commands in the
materialised tree, one at a time, and answers each with what it printed (PREREGISTRATION.md
sections 8 and 11, systems/README.md).

    python3 command_server.py --root /repo --scratch /tmp [--host 0.0.0.0] [--port 7000]
    python3 command_server.py --self-test

It runs in the sandbox container, which holds its arm's tree as of the end of the date and the log
beside it (read-only at /repo), its own software and nothing else, on a network that only the
agent's container shares, which leads nowhere else. Its one request is ``POST /run`` with a JSON
body ``{"command": "<a shell command>"}``, answered with 200 and ``{"exit_code", "stdout",
"stderr", "truncated", "timed_out"}``. Every other request is refused with a JSON body
``{"error": ...}`` and runs nothing: another path with 404, another method with 405 (501 for one
that HTTP does not define), a body without a Content-Length with 411, one over MAX_BODY bytes with
413, and a body that is not a JSON object whose one field, ``command``, is a string of at most
MAX_COMMAND_BYTES bytes in UTF-8 with no NUL character with 400. A command that cannot be started
(the tree is gone, or no process can be made) is a failure of the sandbox, answered with 500.

Each command:

- runs as ``bash -c <command>`` in the root, in a session and process group of its own, with stdin
  from /dev/null and an environment of PATH, LANG, LC_ALL, HOME and TMPDIR alone (HOME and TMPDIR
  the scratch directory), so that nothing of the server's environment reaches it;
- ends when bash exits, or is stopped at TIMEOUT_S seconds; either way its whole process group is
  killed then, so that no background job it started outlives it, and ``timed_out`` says whether
  the limit stopped it (``exit_code`` is bash's status, negative for the signal that killed it);
- has its stdout and its stderr each kept up to LIMIT characters, read as UTF-8 with each byte
  that is not UTF-8 replaced by U+FFFD; an output that is longer is cut there and ends with a
  final line ``[cut: N characters omitted]``, N the characters left out, and ``truncated`` says
  whether either output was cut;
- starts with the scratch directory empty, whatever an earlier command left in it.

Commands run one at a time: a request that arrives while one runs waits for it. So no command sees
another's processes or scratch files, and the requests of a date, whose agents share one sandbox,
never see one another's work.

``--self-test``, which the image's build runs with no network, checks that the tools the sandbox
is built with are installed and that a command runs. It uses only the standard library.
"""

from __future__ import annotations

import argparse
import codecs
import contextlib
import json
import os
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# The limits that the agent's tool description states: seconds for a command, and the characters
# kept of each of its outputs.
TIMEOUT_S = 20
LIMIT = 20_000
# Seconds to read what a killed command's pipes still hold, and between looks at a command that
# runs while its pipes are quiet.
KILL_GRACE_S = 1.0
POLL_S = 0.05
CHUNK = 1 << 16
# The largest command, in bytes of UTF-8, and the largest request body, which holds one command
# however its JSON escapes it.
MAX_COMMAND_BYTES = 65_536
MAX_BODY = 1 << 20
PORT = 7000
FIELDS = ("exit_code", "stdout", "stderr", "truncated", "timed_out")
PATH = "/usr/local/bin:/usr/bin:/bin"
LOCALE = "C.UTF-8"
# The tools the sandbox is built with (bash, coreutils, grep, findutils, sed and gawk), by a
# command of each that the self-test looks for.
TOOLS = ("bash", "cat", "head", "tail", "ls", "sort", "uniq", "wc", "cut", "tr", "grep", "find",
         "xargs", "sed", "gawk")
BAD_BODY = ('the body must be a JSON object whose one field, "command", is a string of at most '
            f"{MAX_COMMAND_BYTES} bytes with no NUL character")


class Capture:
    """One of a command's outputs as it is read: its text up to ``limit`` characters, and how
    many characters it has in all."""

    def __init__(self, limit: int) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._kept: list[str] = []
        self._room = limit
        self.characters = 0

    def feed(self, data: bytes, final: bool = False) -> None:
        text = self._decoder.decode(data, final)
        self.characters += len(text)
        if self._room > 0 and text:
            self._kept.append(text[:self._room])
            self._room -= len(self._kept[-1])

    def result(self) -> tuple[str, bool]:
        """The output as it is answered, and whether it was cut."""
        self.feed(b"", final=True)
        kept = "".join(self._kept)
        omitted = self.characters - len(kept)
        if not omitted:
            return kept, False
        return (f"{kept}{'' if kept.endswith(chr(10)) else chr(10)}"
                f"[cut: {omitted} characters omitted]\n"), True


def environment(scratch: Path) -> dict[str, str]:
    """A command's whole environment."""
    return {"PATH": PATH, "LANG": LOCALE, "LC_ALL": LOCALE, "HOME": str(scratch),
            "TMPDIR": str(scratch)}


def empty(directory: Path) -> None:
    """Removes everything in ``directory`` and leaves it, whatever permissions a command gave what
    it made there: each directory is opened up before it is read, and removed after its contents.
    """
    pending, directories = [str(directory)], []
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    os.chmod(entry.path, 0o700)
                    pending.append(entry.path)
                    directories.append(entry.path)
                else:
                    os.unlink(entry.path)
    for path in reversed(directories):
        os.rmdir(path)


def kill_group(pgid: int) -> None:
    """Kills every process left in the process group ``pgid``. In the sandbox nothing else starts
    a process while a command's group is killed, so no other group can have taken its number."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGKILL)


def run(command: str, *, root: Path, scratch: Path, timeout_s: float = TIMEOUT_S,
        limit: int = LIMIT, grace_s: float = KILL_GRACE_S) -> dict:
    """Runs ``command`` as the module's description says and returns its result, ``FIELDS`` in
    that order. OSError when it cannot be started."""
    empty(scratch)
    process = subprocess.Popen(["bash", "-c", command], cwd=root, env=environment(scratch),
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, start_new_session=True)
    captures = {process.stdout: Capture(limit), process.stderr: Capture(limit)}
    deadline = time.monotonic() + timeout_s
    timed_out, stop_reading = False, None
    try:
        with selectors.DefaultSelector() as selector:
            for pipe in captures:
                os.set_blocking(pipe.fileno(), False)
                selector.register(pipe, selectors.EVENT_READ)
            while selector.get_map():
                now = time.monotonic()
                if stop_reading is None:
                    if process.poll() is not None or now >= deadline:
                        # Bash has ended, or the limit has come: nothing of the group goes on.
                        timed_out = process.returncode is None
                        kill_group(process.pid)
                        stop_reading = now + grace_s
                elif now >= stop_reading:
                    break  # a process that left the group holds a pipe open: stop reading it
                for key, _ in selector.select(POLL_S):
                    try:
                        chunk = os.read(key.fd, CHUNK)
                    except BlockingIOError:
                        continue
                    if chunk:
                        captures[key.fileobj].feed(chunk)
                    else:
                        selector.unregister(key.fileobj)
        if stop_reading is None:
            # Both outputs closed before bash ended: it still runs to its end or to the limit.
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                timed_out = True
    finally:
        kill_group(process.pid)
        process.wait()
        for pipe in captures:
            pipe.close()
    stdout, cut_out = captures[process.stdout].result()
    stderr, cut_err = captures[process.stderr].result()
    return {"exit_code": process.returncode, "stdout": stdout, "stderr": stderr,
            "truncated": cut_out or cut_err, "timed_out": timed_out}


def command_of(body: bytes) -> str | None:
    """The command of a request's body, or None when the body is not one (``BAD_BODY``)."""
    try:
        request = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None
    if not isinstance(request, dict) or set(request) != {"command"}:
        return None
    command = request["command"]
    if not isinstance(command, str) or "\x00" in command:
        return None
    try:
        size = len(command.encode("utf-8"))
    except UnicodeEncodeError:  # a lone surrogate, which JSON can carry and no process can take
        return None
    return command if size <= MAX_COMMAND_BYTES else None


class CommandServer(ThreadingHTTPServer):
    """The HTTP server of the sandbox, over the tree at ``root``, with ``scratch`` the directory
    that each command may write in; it runs one command at a time (``lock``)."""

    daemon_threads = True
    request_queue_size = 64

    def __init__(self, address: tuple[str, int], *, root: Path, scratch: Path,
                 timeout_s: float = TIMEOUT_S, limit: int = LIMIT) -> None:
        super().__init__(address, Handler)
        self.root, self.scratch = Path(root), Path(scratch)
        self.timeout_s, self.limit = timeout_s, limit
        self.lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    """``POST /run``, and a refusal for every other request (the module's description)."""

    server_version = "asofbench-sandbox"
    sys_version = ""
    # Seconds that a client has to send its request.
    timeout = 30

    def reply(self, status: int, body: dict, **headers: str) -> None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for name, value in headers.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802 - the name http.server calls
        if self.path != "/run":
            self.reply(404, {"error": "the one path is /run"})
            return
        length = self.headers.get("Content-Length")
        if length is None:
            self.reply(411, {"error": "a request needs a Content-Length"})
            return
        if not length.isdigit():
            self.reply(400, {"error": "the Content-Length is not a number"})
            return
        if int(length) > MAX_BODY:
            self.reply(413, {"error": f"a body holds at most {MAX_BODY} bytes"})
            return
        command = command_of(self.rfile.read(int(length)))
        if command is None:
            self.reply(400, {"error": BAD_BODY})
            return
        try:
            with self.server.lock:
                result = run(command, root=self.server.root, scratch=self.server.scratch,
                             timeout_s=self.server.timeout_s, limit=self.server.limit)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            self.reply(500, {"error": f"the command could not be run: {type(error).__name__}"})
            return
        self.reply(200, result)

    def refuse_method(self) -> None:
        self.reply(405, {"error": "the one request is POST /run"}, Allow="POST")

    do_GET = do_HEAD = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_TRACE = do_CONNECT = (
        refuse_method)


def self_test() -> int:
    """Checks that the sandbox's tools are installed and that a command runs; prints what it found
    and returns 0, or the problems and 1."""
    problems = [f"{tool} is not installed" for tool in TOOLS if shutil.which(tool) is None]
    with tempfile.TemporaryDirectory() as directory:
        root, scratch = Path(directory) / "repo", Path(directory) / "scratch"
        root.mkdir()
        scratch.mkdir()
        (root / "file").write_text("self-test\n", encoding="utf-8")
        result = run("cat file; head -c 30 /dev/zero | tr '\\0' a", root=root, scratch=scratch,
                     limit=20)
    expected = {"exit_code": 0, "stdout": "self-test\naaaaaaaaaa\n[cut: 20 characters omitted]\n",
                "stderr": "", "truncated": True, "timed_out": False}
    if result != expected:
        problems.append(f"a command ran as {result!r}")
    if problems:
        print(f"sandbox self-test: FAIL: {'; '.join(problems)}", file=sys.stderr)
        return 1
    print(f"sandbox self-test: {', '.join(TOOLS)} installed; a command runs, its output cut at "
          "the limit with its final line")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="command_server", description="Runs the agent's "
                                     "read-only shell commands in the tree, one at a time.",
                                     allow_abbrev=False)
    parser.add_argument("--root", type=Path, help="the tree, where each command runs")
    parser.add_argument("--scratch", type=Path, help="the directory each command may write in, "
                        "emptied before each command")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--self-test", action="store_true",
                        help="check the tools and run a command, with no network")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.root is None or args.scratch is None:
        parser.error("--root and --scratch are required")
    if not args.root.is_dir() or not args.scratch.is_dir():
        parser.error("--root and --scratch must be directories")
    httpd = CommandServer((args.host, args.port), root=args.root, scratch=args.scratch)
    print(f"command_server: serving {args.root} on port {args.port}, one command at a time",
          flush=True)
    with httpd:
        httpd.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
