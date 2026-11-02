"""The dated revision stream's episodes (PREREGISTRATION.md section 8), as a system reads them, and
the one template that renders an episode into the text a store system ingests.

Every store system renders an episode the same way, with ``render_episode``:

    {ref}: {kind} recorded at {recorded_at} in the {repo} repository, commit {commit}.

    Commit message:
    {message}

    {section}:
    {text}

where the section names what ``text`` holds: the full text of the file for a create, its unified
diff for a modify, and for a rename its diff against the file it moved from. A modify or rename
whose diff is empty (in H- a change to withheld header lines only, or a change of the file's mode
alone) renders the section line and then a line that says so. A delete has no text and no
section: its body ends with the line ``The file was deleted.`` The rendering adds nothing else and
leaves nothing out, so the same episode always renders to the same text, however long.

It uses only the standard library.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator

# The section line of each kind that has a text, and what an empty diff renders as.
SECTIONS = {
    "create": "Full text of the file",
    "modify": "Unified diff of the file",
    "rename": "Unified diff of the file, which moved here from another repository or path",
}
EMPTY_DIFF = "(empty: the change touched only lines this arm withholds, or the file's mode)"
DELETED = "The file was deleted."
KINDS = (*SECTIONS, "delete")


def read_episodes(path: str) -> Iterator[dict]:
    """The episodes of the stream file at ``path``, one JSON object per line, in file order: the
    episodes of its lines (``episodes_of``), which name ``path`` in a ValueError."""
    with open(path, "rb") as handle:
        yield from episodes_of(handle, path)


def episodes_of(lines: Iterable[bytes], where: str) -> Iterator[dict]:
    """The episodes of a stream's ``lines``, each a line of bytes as a binary file gives it, one
    JSON object per line, in order. A line ends at a newline only, so a line or paragraph
    separator inside a string stays in it; ValueError, naming the stream ``where``, for a line that
    is not a JSON object. A reader that must know the bytes it parsed, to hash them as it goes,
    passes its own lines."""
    for number, raw in enumerate(lines, 1):
        try:
            episode = json.loads(raw.decode("utf-8"))
        except ValueError as error:
            raise ValueError(f"{where}: line {number} is not JSON: {error}") from None
        if not isinstance(episode, dict):
            raise ValueError(f"{where}: line {number} is not a JSON object")
        yield episode


def render_episode(e: dict) -> str:
    """The text that a store system ingests for the episode ``e`` (the module's template)."""
    kind = e["kind"]
    if kind not in KINDS:
        raise ValueError(f"episode {e.get('episode_id')!r}: unknown kind {kind!r}")
    head = (f"{e['ref']}: {kind} recorded at {e['recorded_at']} in the {e['repo']} repository, "
            f"commit {e['commit']}.\n\nCommit message:\n{e['message']}\n\n")
    if kind == "delete":
        if e["text"]:
            raise ValueError(f"episode {e.get('episode_id')!r}: a delete has no text")
        return head + DELETED
    text = e["text"]
    if not text and kind != "create":
        text = EMPTY_DIFF
    return f"{head}{SECTIONS[kind]}:\n{text}"
