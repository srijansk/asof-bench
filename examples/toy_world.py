"""A toy world to develop a system against, without asking it any v0.1 question.

The registration forbids a configuration to answer a v0.1 item before the commit that fixes it
(PREREGISTRATION.md section 11), so an adapter is developed on something else. This script builds
three small synthetic repositories laid out like python/peps, ethereum/EIPs and ethereum/ERCs,
then runs every command on them: ``build`` writes the toy questions and gold, ``stream`` the H-
stream, ``materialize`` the files and log that grep at D reads on one date, and ``score`` scores
an answers file of one constant answer per item for an outside entry, ``my-system``. None of its
items is a v0.1 item, and its results are a toy run's, never a reading of v0.1.

Usage: ``uv run python examples/toy_world.py [DIR]``; DIR, which must not exist yet, defaults to
a new temporary directory. Replace the constant answers with your system's to try its rows.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from asofbench import cli

# Git runs with no system or user configuration and no variable that could point it at another
# repository, so the script only ever touches its own directory.
_REDIRECTING = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
                "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR", "GIT_NAMESPACE",
                "GIT_CONFIG_PARAMETERS", "GIT_CONFIG_COUNT")
_IDENTITY = {"GIT_AUTHOR_NAME": "asofbench toy world", "GIT_AUTHOR_EMAIL": "toy@example.invalid",
             "GIT_COMMITTER_NAME": "asofbench toy world",
             "GIT_COMMITTER_EMAIL": "toy@example.invalid"}


def _git(repo: Path, *args: str, when: str | None = None) -> None:
    env = {k: v for k, v in os.environ.items() if k not in _REDIRECTING}
    env.update(_IDENTITY, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
    if when is not None:
        env.update(GIT_AUTHOR_DATE=when, GIT_COMMITTER_DATE=when)
    done = subprocess.run(["git", "-C", str(repo), "-c", "commit.gpgsign=false", *args],
                          env=env, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {done.stderr.strip()}")


def _commit(repo: Path, files: dict[str, str], message: str, when: str) -> None:
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message, when=when)


def pep(number: int, status: str, body: str = "Plain text.\n") -> str:
    return f"PEP: {number}\nTitle: Proposal {number}\nStatus: {status}\n\n{body}"


def eip(number: int, status: str, requires: str = "", body: str = "Plain text.\n") -> str:
    extra = f"requires: {requires}\n" if requires else ""
    return f"---\neip: {number}\ntitle: Proposal {number}\nstatus: {status}\n{extra}---\n{body}"


def toy_clones(root: Path) -> Path:
    """Three synthetic repositories with dated histories, in ``root``, which it returns."""
    for name in ("peps", "EIPs", "ERCs"):
        (root / name).mkdir(parents=True)
        _git(root / name, "init", "-q", "-b", "main")
        _commit(root / name, {"README.md": "Proposals.\n"}, "start", "2000-01-01T00:00:00Z")
    for day, files in [
        ("2001-01-01", {"pep-0001.txt": pep(1, "Draft", "Compare PEP 2.\n")}),
        ("2002-01-01", {"pep-0002.txt": pep(2, "Draft")}),
        ("2005-01-01", {"pep-0001.txt": pep(1, "Final", "Compare PEP 2.\n")}),
        ("2006-01-01", {"pep-0002.txt": pep(2, "Rejected")}),
    ]:
        _commit(root / "peps", files, f"Update on {day}", f"{day}T12:00:00Z")
    for day, files in [
        ("2015-11-19", {"EIPS/eip-20.md": eip(20, "Draft")}),
        ("2016-01-01", {"EIPS/eip-1.md": eip(1, "Draft", body="See EIP-2.\n")}),
        ("2016-02-01", {"EIPS/eip-2.md": eip(2, "Draft", requires="1")}),
        ("2016-03-01", {"EIPS/eip-3.md": eip(3, "Draft", requires="2")}),
        ("2017-01-01", {"EIPS/eip-2.md": eip(2, "Final", requires="1")}),
        ("2017-09-11", {"EIPS/eip-20.md": eip(20, "Final")}),
        ("2018-01-01", {"EIPS/eip-1.md": eip(1, "Living", body="See EIP-2.\n")}),
        ("2023-10-26", {"EIPS/eip-20.md": eip(20, "Moved")}),
    ]:
        _commit(root / "EIPs", files, f"Update on {day}", f"{day}T12:00:00Z")
    _commit(root / "ERCs", {"ERCS/erc-20.md": eip(20, "Final")}, "complete erc move",
            "2023-10-25T15:11:00Z")
    return root


def constant_answers(questions: list[dict], system: str) -> list[dict]:
    """One row per item, arm and run, each giving the same answer of the item's type."""
    answer = {"status": "Draft", "set": [], "count": 0}
    return [{"id": q["id"], "system": system, "arm": arm, "run": run,
             "answer": answer[q["answer_type"]], "tokens_in": 0, "tokens_out": 0, "wall_s": 0.0,
             "tool_calls": 0}
            for q in questions for arm in q["arms"] for run in (1, 2, 3)]


def run(root: Path) -> int:
    """Build the toy world in ``root`` and run every command on it; 0 when every step succeeds."""
    clones, data = toy_clones(root / "clones"), root / "toy-data"
    stream = root / "streams" / "h-minus.jsonl"
    steps = [["build", "--clones", str(clones), "--out", str(data)],
             ["stream", "--clones", str(clones), "--data", str(data), "--arm", "H-",
              "--out", str(stream)]]
    for argv in steps:
        if cli.main(argv) != 0:
            return 1
    questions = [json.loads(line) for line in (data / "questions.jsonl").read_text(
        encoding="utf-8").splitlines()]
    dated = next(q for q in questions if "H-" in q["arms"] and q["as_of"] != questions[0]["as_of"])
    if cli.main(["materialize", "--stream", str(stream), "--as-of", dated["as_of"],
                 "--out", str(root / "tree")]) != 0:
        return 1
    answers = root / "answers.jsonl"
    answers.write_text("".join(json.dumps(row) + "\n"
                               for row in constant_answers(questions, "my-system")),
                       encoding="utf-8")
    (root / "entries.json").write_text('{"my-system": {}}\n', encoding="utf-8")
    code = cli.main(["score", "--data", str(data), "--answers", str(answers),
                     "--entries", str(root / "entries.json"), "--out", str(root / "results"),
                     "--answering-model", "toy-model", "--answering-settings", "toy settings",
                     "--construction-model", "none", "--embedding-model", "none"])
    if code == 0:
        print(f"{len(questions)} toy items in {data}; the tree as of {dated['as_of']} in "
              f"{root / 'tree'}; results in {root / 'results' / 'results.md'}")
    return code


def main(argv: list[str]) -> int:
    root = Path(argv[0]) if argv else Path(tempfile.mkdtemp(prefix="asofbench-toy-"))
    if argv and root.exists():
        print(f"{root} already exists; give a new directory", file=sys.stderr)
        return 2
    return run(root.resolve())


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
