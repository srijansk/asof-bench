"""Hand-built corpora for tests: proposals made directly from ``Proposal`` and ``Version`` objects.

No git is involved. ``Hand.add`` records one version per step, with a header rendered as the
corpus would write it and parsed by the real header reader, so statuses, titles, requires lists
and bodies come out exactly as they would from a clone. Tests import this module as
``from handbuilt import DELETE, Hand``.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import datetime

from asofbench.gitwalk import Commit
from asofbench.headers import parse_header
from asofbench.statuses import normalize_status
from asofbench.timeline import Corpus, Proposal, Version

DELETE = object()  # a step that deletes the proposal's file
PATHS = {"peps": "peps/pep-{n:04d}.rst", "EIPs": "EIPS/eip-{n}.md", "ERCs": "ERCS/erc-{n}.md"}


def ts(when: str) -> int:
    """Epoch seconds of ``YYYY-MM-DD`` (noon UTC) or of a full ``...Z`` timestamp."""
    text = when if "T" in when else f"{when}T12:00:00Z"
    return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())


def render(family: str, number: int, title: str, status: str | None, headers, body: str) -> str:
    keys = ("PEP", "Title", "Status") if family == "pep" else ("eip", "title", "status")
    lines = [f"{keys[0]}: {number}", f"{keys[1]}: {title}"]
    if status is not None:
        lines.append(f"{keys[2]}: {status}")
    lines.extend(f"{key}: {value}" for key, value in headers)
    if family == "pep":
        return "\n".join(lines) + "\n\n" + body
    return "---\n" + "\n".join(lines) + "\n---\n" + body


class FakeReader:
    def __init__(self) -> None:
        self.blobs: dict[str, str] = {}

    def read(self, blob: str) -> str:
        return self.blobs[blob]


class Hand:
    """Proposals built version by version, with readers and per-repository commit lists."""

    def __init__(self) -> None:
        self.readers = {repo: FakeReader() for repo in PATHS}
        self.proposals: dict[str, Proposal] = {}
        self.commits: dict[str, list[Commit]] = defaultdict(list)

    def add(self, family, series, number, title, steps, *, headers=(), body="Plain text.\n"):
        """``steps`` are ``(when, status)`` or ``(when, status, repo)``; DELETE deletes."""
        default_repo = "peps" if family == "pep" else "EIPs"
        p = Proposal(family, series, number)
        present = False
        for step in steps:
            when, status = step[0], step[1]
            repo = step[2] if len(step) > 2 else default_repo
            path = PATHS[repo].format(n=number)
            sha = hashlib.sha1(f"{repo}|{when}|{series}-{number}".encode()).hexdigest()
            self.commits[repo].append(Commit(sha, ts(when), ts(when), f"Update {series}-{number}"))
            if status is DELETE:
                p.versions.append(
                    Version(repo, ts(when), sha, path, None, "delete", {}, None, "absent")
                )
                present = False
                continue
            text = render(family, number, title, status, headers, body)
            blob = hashlib.sha1(text.encode()).hexdigest()
            self.readers[repo].blobs[blob] = text
            header = parse_header(family, text)
            raw = header.get("status")
            kind = "modify" if present else "create"
            p.versions.append(
                Version(repo, ts(when), sha, path, blob, kind, header, raw,
                        normalize_status(family, raw))
            )
            present = True
        self.proposals[p.pid] = p
        return p

    def corpus(self) -> Corpus:
        commits = {repo: sorted(self.commits[repo], key=lambda c: c.recorded_ts) for repo in PATHS}
        pins = {repo: chain[-1].sha for repo, chain in commits.items() if chain}
        return Corpus(proposals=dict(self.proposals), commits=commits, pins=pins)
