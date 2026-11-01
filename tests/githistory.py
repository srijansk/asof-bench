"""Read-only git on the pinned corpus clones: the first-parent chain and section 3's pull merges.

Shared by the corpus tests that check the registration's history numbers (tests/
test_corpus_history.py and tests/test_build.py). Nothing here reads the generator's walker: the
chains and merges are read with ``git log``, ``git rev-list``, ``git merge-base`` and ``git diff``,
with no configuration and no optional lock taken, at a pin the caller gives (data/v0.1's
MANIFEST.json). A recorded time is the larger of a commit's committer time and its first parent's
recorded time (section 3), and a span is dated in UTC from the earliest commit a pull merge brings
in to the merge's recorded time.
"""

from __future__ import annotations

import os
import re
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path

# Section 2's proposal files of each repository.
FILES = {"peps": re.compile(r"^(?:peps/)?pep-(\d{4})\.(?:txt|rst)$"),
         "EIPs": re.compile(r"^(?:EIPS/)?eip-(\d+)\.(?:md|mediawiki)$"),
         "ERCs": re.compile(r"^ERCS/erc-(\d+)\.md$")}
# Section 3's pull merges, by the names each repository gives them.
PULL_MERGES = {"peps": ("Merge branch 'master' of github.com:python/peps",),
               "EIPs": ("Merge branch 'master' of github.com:ethereum/EIPs",
                        "Merge branch 'master' of https://github.com/ethereum/EIPs")}
PULL_MERGES["ERCs"] = PULL_MERGES["EIPs"]  # the ERC repository shares the EIP one's early history
DAY = 86400


def git_env() -> dict[str, str]:
    """No configuration, no optional lock and no variable that could point git elsewhere."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return {**env, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_OPTIONAL_LOCKS": "0"}


def git(clones: Path, repo: str, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(clones / repo), *args], env=git_env(),
                          capture_output=True, check=True)
    return done.stdout.decode("utf-8", errors="replace")


def utc_day(seconds: int) -> date:
    return datetime.fromtimestamp(seconds, UTC).date()


def chain(clones: Path, repo: str, pin: str) -> list[dict]:
    """The first-parent chain from the root to ``pin``: each commit's id, parents, committer
    time, recorded time, committer date in its own time zone, and subject."""
    out = git(clones, repo, "log", "--first-parent", "--reverse",
              "--format=%H%x00%P%x00%ct%x00%cd%x00%s", "--date=short", pin)
    found, recorded = [], 0
    for line in out.splitlines():
        sha, parents, committed, local, subject = line.split("\x00")
        recorded = max(recorded, int(committed))
        found.append({"sha": sha, "parents": parents.split(), "committed": int(committed),
                      "recorded": recorded, "local": local, "subject": subject})
    return found


def pull_merges(clones: Path, repo: str, pin: str) -> list[dict]:
    """Section 3's pull merges of ``repo`` up to ``pin``: the first-parent merges named as section
    3 names them, each with the commits it brings in from its second parent (reachable from it
    and not from the first) and the proposal numbers whose files those commits change."""
    found = []
    for c in chain(clones, repo, pin):
        if len(c["parents"]) != 2 or c["subject"] not in PULL_MERGES[repo]:
            continue
        first, second = c["parents"]
        brought = [int(t) for t in git(clones, repo, "rev-list", "--format=%ct",
                                       "--no-commit-header", f"^{first}", second).split()]
        base = git(clones, repo, "merge-base", first, second).strip()
        paths = git(clones, repo, "diff", "--name-only", base, second).splitlines()
        numbers = {int(m.group(1)) for path in paths if (m := FILES[repo].match(path))}
        found.append({**c, "earliest": min(brought), "numbers": numbers,
                      "start": utc_day(min(brought)), "end": utc_day(c["recorded"])})
    return found
