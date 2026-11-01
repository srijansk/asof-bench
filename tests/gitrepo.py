"""Throwaway git repositories with controlled dates, for tests.

Every git call here runs with no system or user configuration and without any
variable that could point git at some other repository, so a test can only ever
touch its own directory. Tests import this module as ``from gitrepo import TempRepo``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

# Variables that would redirect git to a repository, index or configuration other than the
# one under test (they are set, for example, while a git hook or a review script is running).
_REDIRECTING = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_NAMESPACE",
    "GIT_CONFIG_PARAMETERS",
    "GIT_CONFIG_COUNT",
)

_IDENTITY = {
    "GIT_AUTHOR_NAME": "asofbench tests",
    "GIT_AUTHOR_EMAIL": "tests@example.invalid",
    "GIT_COMMITTER_NAME": "asofbench tests",
    "GIT_COMMITTER_EMAIL": "tests@example.invalid",
}


def _env(when: str | None = None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _REDIRECTING}
    env.update(_IDENTITY)
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_CONFIG_GLOBAL"] = "/dev/null"
    if when is not None:
        env["GIT_AUTHOR_DATE"] = when
        env["GIT_COMMITTER_DATE"] = when
    return env


class TempRepo:
    """A repository on branch ``main`` that is built up with explicit commit dates."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._git("init", "-q", "-b", "main")

    def _git(self, *args: str, when: str | None = None) -> str:
        cmd = ["git", "-C", str(self.path), "-c", "commit.gpgsign=false", *args]
        done = subprocess.run(cmd, env=_env(when), capture_output=True, text=True, check=False)
        if done.returncode != 0:
            raise RuntimeError(f"{' '.join(args)} failed: {done.stderr.strip()}")
        return done.stdout.strip()

    def config(self, key: str, value: str) -> None:
        """Set a repository-local git configuration value."""
        self._git("config", key, value)

    def write(self, rel: str, text: str | bytes) -> None:
        target = self.path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(text.encode("utf-8") if isinstance(text, str) else text)

    def rm(self, rel: str) -> None:
        (self.path / rel).unlink()

    def mv(self, old: str, new: str) -> None:
        target = self.path / new
        target.parent.mkdir(parents=True, exist_ok=True)
        (self.path / old).rename(target)

    def commit(self, msg: str, when: str, author: str | None = None) -> str:
        """Commit everything in the working tree, authored and committed at ``when``.

        ``author`` (``"Name <email>"``) replaces the default author; the committer stays the same.
        """
        self._git("add", "-A")
        by = ("--author", author) if author is not None else ()
        self._git("commit", "-q", "-m", msg, *by, when=when)
        return self._git("rev-parse", "HEAD")

    def branch(self, name: str) -> None:
        """Create ``name`` at the current commit without switching to it."""
        self._git("branch", name)

    def checkout(self, name: str) -> None:
        self._git("checkout", "-q", name)

    def merge(self, name: str, msg: str, when: str) -> str:
        """Merge ``name`` into the current branch; always a merge commit, never a fast-forward."""
        self._git("merge", "-q", "--no-ff", "-m", msg, name, when=when)
        return self._git("rev-parse", "HEAD")


def clone(source: Path, target: Path, *args: str) -> Path:
    """``git clone`` of ``source`` into ``target`` over ``file://`` (so ``--depth`` is honoured),
    with the same clean environment as ``TempRepo``; returns ``target``."""
    cmd = ["git", "clone", "-q", *args, Path(source).resolve().as_uri(), str(target)]
    done = subprocess.run(cmd, env=_env(), capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"clone failed: {done.stderr.strip()}")
    return Path(target)
