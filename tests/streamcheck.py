"""Independent checks of a built stream, for the corpus tests.

Nothing here imports ``stream``, ``headers`` or ``statuses``: the withheld keys, the header zone,
the stripping and the counterfactual swaps are written out again from the registration (sections
5, 7 and 8) by simpler and deliberately more suspicious rules, so that a mistake in the builder or
in ``headers.py`` shows up as a disagreement. The main check is by state: a system that applies a
stream's diffs to its create texts must end up with exactly the text that git holds for each
version, after the arm's edit.

The header zone is the block of ``Key: value`` lines at the top of a file: the lines between the
``---`` lines of front matter, or the run of lines that starts at the first key-like line within
the first 40 and ends before the next blank line, ``</pre>`` line, code fence or heading.
"""

from __future__ import annotations

import re

_NAMES = (
    r"status|resolution|accepted|superseded[-_ ]?by|replaced[-_ ]?by|replaces|requires?"
    r"|dependencies|last[-_ ]?call[-_ ]?deadline|review[-_ ]?period[-_ ]?ends?"
    r"|review[-_ ]?end[-_ ]?date|withdrawal[-_ ]?reason"
)
# Section 8's list, with separators other than the hyphen allowed, and a qualifier or a space
# before the colon: wider than anything headers.py would treat as a withheld field.
WITHHELD_LINE = re.compile(rf"[ \t]*(?:{_NAMES})[ \t]*(?:\([^()]*\))?[ \t]*:", re.IGNORECASE)
# A key line may have an empty value (as in a PEP's "Type:" line), but a header block is only
# taken to start at a key line that has a value.
_KEY_LINE = re.compile(r"[ \t]*[A-Za-z][A-Za-z0-9-]*[ \t]*(?:\([^()]*\))?[ \t]*:")
_START_LINE = re.compile(_KEY_LINE.pattern + r"[ \t]*\S")
_BOM = "\ufeff"


def header_zone(lines: list[str]) -> tuple[int, int]:
    """The ``[first, end)`` range of ``lines`` that holds the header fields, or ``(0, 0)``."""
    first = next((i for i, line in enumerate(lines) if line.strip(_BOM + " \t\r")), None)
    if first is None:
        return 0, 0
    if lines[first].strip(_BOM + " \t\r") == "---":
        for j in range(first + 1, len(lines)):
            if lines[j].strip() == "---":
                return first + 1, j
        return first + 1, len(lines)
    start = next(
        (i for i, line in enumerate(lines[:40]) if _START_LINE.match(line.lstrip(_BOM))), None
    )
    if start is None:
        return 0, 0
    for j in range(start + 1, len(lines)):
        stripped = lines[j].strip()
        if not stripped or stripped.startswith(("</pre>", "```", "#")):
            return start, j
    return start, len(lines)


def zone_lines(text: str) -> list[str]:
    lines = text.split("\n")
    first, end = header_zone(lines)
    return lines[first:end]


def leaked_lines(text: str) -> list[str]:
    """The lines of the header zone of ``text`` that look like a withheld field."""
    return [line for line in zone_lines(text) if WITHHELD_LINE.match(line.lstrip(_BOM))]


def diff_header_hits(diff: str, before: str, after: str) -> list[str]:
    """The added and removed lines of ``diff`` that fall in a header zone and look withheld.

    A removed line is placed by its number in ``before`` and an added line by its number in
    ``after``; each is compared with the header zone of that text. (Line numbers are those of
    the hunk headers, so a diff cut at form feeds is only roughly placed.)
    """
    zone = {"-": header_zone(before.split("\n")), "+": header_zone(after.split("\n"))}
    number = {"-": 0, "+": 0}
    hits = []
    started = False
    for line in diff.split("\n"):
        hunk = _HUNK.fullmatch(line)
        if hunk is not None:
            started = True
            for sign, start, count in (("-", hunk[1], hunk[2]), ("+", hunk[3], hunk[4])):
                number[sign] = int(start) + (1 if count == "0" else 0)
            continue
        if not started or not line:
            continue
        sign, content = line[0], line[1:]
        if sign == " ":
            number["-"] += 1
            number["+"] += 1
        elif sign in zone:
            first, end = zone[sign]
            if first <= number[sign] - 1 < end and WITHHELD_LINE.match(content.lstrip(_BOM)):
                hits.append(line)
            number[sign] += 1
    return hits


def status_value(text: str) -> str | None:
    """The value on the first status line of the header zone of ``text``, or None."""
    for line in zone_lines(text):
        match = _STATUS_LINE.fullmatch(line.lstrip(_BOM))
        if match is not None:
            return match.group(2).strip()
    return None


def _is_key_line(family: str, line: str, indented_keys: bool) -> bool:
    """Whether ``line`` starts a field. A PEP's continuation line is one that starts with
    whitespace; in an Ethereum file any line that is no ``key: value`` line continues the field
    above, and a key starts in the first column except in the run form, where blocks are indented.
    """
    if line[:1] in (" ", "\t") and not (family == "eth" and indented_keys):
        return False
    return _KEY_LINE.match(line) is not None


def strip_expected(family: str, text: str) -> str:
    """``text`` as H- should show it: the withheld field lines and their continuations removed."""
    lines = text.split("\n")
    first, end = header_zone(lines)
    front_matter = first > 0 and lines[first - 1].strip(_BOM + " \t\r") == "---"
    kept = lines[:first]
    dropping = False
    for line in lines[first:end]:
        body = line.lstrip(_BOM)
        if WITHHELD_LINE.match(body) and _is_key_line(family, body, not front_matter):
            dropping = True
            continue
        if _is_key_line(family, body, not front_matter):
            dropping = False
        elif dropping:
            continue  # a line that is no key line continues the field above it
        kept.append(line)
    return "\n".join(kept + lines[end:])


# --- the counterfactual edit, from section 7 -------------------------------------------------

_SWAPS = {
    "pep": [("Accepted", "Rejected"), ("Final", "Withdrawn"), ("Active", "Deferred"),
            ("Provisional", "Superseded")],
    "eth": [("Final", "Withdrawn"), ("Review", "Last Call")],
}
_SWAP = {
    family: {k: v for a, b in pairs for k, v in ((a, b), (b, a))}
    for family, pairs in _SWAPS.items()
}
_LEGACY = {  # section 5, PEPs only; each name read as the canonical status it stands for
    "incomplete": "Draft", "draught": "Draft", "approved": "Accepted", "finished": "Final",
    "complete": "Final", "postponed": "Deferred", "withdrawal": "Withdrawn",
    "replaced": "Superseded",
}
_STATUS_LINE = re.compile(r"([ \t]*status[ \t]*:[ \t]*)(.*?)(\r?)$", re.IGNORECASE)


def _canonical(family: str, value: str) -> str:
    value = re.sub(r"(?:^|\s)#.*", "", value.strip(), count=1).strip().strip("\"'").strip()
    key = " ".join(value.casefold().split())
    if family == "pep" and key in _LEGACY:
        return _LEGACY[key]
    for status in {s for pair in _SWAPS[family] for s in pair}:
        if status.casefold() == key:
            return status
    return value


def permute_expected(family: str, text: str) -> str:
    """``text`` as CF should show it: swapped statuses in the header zone, all else as written."""
    lines = text.split("\n")
    first, end = header_zone(lines)
    for i in range(first, end):
        match = _STATUS_LINE.fullmatch(lines[i].lstrip(_BOM))
        if match is None:
            continue
        swapped = _SWAP[family].get(_canonical(family, match.group(2)))
        if swapped is not None:
            lines[i] = match.group(1) + swapped + match.group(3)
    return "\n".join(lines)


# --- applying a diff -----------------------------------------------------------------------------

_HUNK = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
NO_NEWLINE = "\\ No newline at end of file"


def lines_of(text: str) -> list[str]:
    """``text`` as lines split at newline characters only, each with its newline; the last line
    has none if the text does not end with one."""
    parts = text.split("\n")
    return [part + "\n" for part in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def apply_diff(before: str, diff: str) -> str:
    """``before`` with a ``unified_diff`` of it applied; raises ValueError if it does not fit.

    The application is exact, final newline included: a context or removed line must be the line
    of ``before`` as it stands, with or without its newline according to the marker line that
    follows, the counts of every hunk header must be those of its lines, and a line left without
    a newline must be the last of the result.
    """
    if not diff:
        return before
    old = lines_of(before)
    rows = diff.split("\n")
    if rows.pop() != "":
        raise ValueError("the diff does not end with a newline")
    if rows[:2] != ["--- a", "+++ b"]:
        raise ValueError("the diff has no header")
    rows = rows[2:]
    out: list[str] = []
    pos = 0  # the next line of ``before`` that no hunk has read
    want = (0, 0)  # the counts the open hunk's header promises
    seen = [0, 0]
    in_hunk = False
    i = 0
    while i < len(rows):
        row = rows[i]
        i += 1
        if row.startswith("@@"):
            if in_hunk and tuple(seen) != want:
                raise ValueError(f"a hunk has {tuple(seen)} lines, its header says {want}")
            hunk = _HUNK.fullmatch(row)
            if hunk is None:
                raise ValueError(f"bad hunk header {row!r}")
            want = (1 if hunk[2] is None else int(hunk[2]), 1 if hunk[4] is None else int(hunk[4]))
            start = int(hunk[1]) if want[0] == 0 else int(hunk[1]) - 1
            if start < pos or start > len(old):
                raise ValueError("hunks overlap or run past the text")
            out.extend(old[pos:start])
            pos, seen, in_hunk = start, [0, 0], True
            continue
        sign, content = row[:1], row[1:]
        if not in_hunk or sign not in (" ", "-", "+"):
            raise ValueError(f"unexpected diff line {row!r}")
        open_end = i < len(rows) and rows[i] == NO_NEWLINE
        i += open_end
        line = content + ("" if open_end else "\n")
        if sign in (" ", "-"):
            if pos >= len(old) or old[pos] != line:
                raise ValueError(f"the diff does not fit the text at line {pos + 1}")
            pos += 1
            seen[0] += 1
        if sign in (" ", "+"):
            out.append(line)
            seen[1] += 1
    if in_hunk and tuple(seen) != want:
        raise ValueError(f"a hunk has {tuple(seen)} lines, its header says {want}")
    out.extend(old[pos:])
    if any(not line.endswith("\n") for line in out[:-1]):
        raise ValueError("a line without a newline is not the last line")
    return "".join(out)
