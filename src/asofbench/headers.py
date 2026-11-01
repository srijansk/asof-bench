"""Reading proposal headers (spec section 4) and withholding fields from them (section 8).

``family`` is ``"pep"`` for the Python Enhancement Proposals and ``"eth"`` for the Ethereum
EIPs and ERCs. One scanner finds the header block and its fields, with the character offsets
of each, and ``parse_header``, ``header_span`` and ``strip_withheld`` are all read off it, so
they cannot disagree about where the header is.

* PEP: the RFC 822-style block from the first non-blank line to the next blank line (two real
  PEP versions begin with a blank line); a line that starts with whitespace continues the field
  above it.
* EIP and ERC: if the first non-empty line is ``---``, the front matter up to the next ``---``
  line; otherwise the block that starts at the first ``Key: value`` line within the first 40
  lines and ends at the next blank line, ``</pre>`` line, code-fence line (one starting with
  three backticks) or Markdown heading line (one starting with ``#``), which covers the indented
  blocks and the mediawiki ``<pre>`` blocks of 2015-2017. In both forms a line inside the block
  that is not a ``key: value`` line continues the field above it, as a wrapped title does.

A block ends with the line that closes it (the blank line, the closing ``---`` or the ``</pre>``
line), which is part of the block, or with the end of the text if nothing closes it. A code
fence or a heading ends a block before itself: that line starts the body. A leading byte-order
mark is skipped and carriage returns at the end of a line are ignored.

A key may carry a parenthetical qualifier, as in ``requires (*optional): 2718``. It is a key line
in every form and is recorded under its full name, case-folded with its spaces single
(``requires (*optional)``), so it is never the unqualified field and never feeds a gold list.
``strip_withheld`` still removes it, with its continuation lines, whenever the name without the
qualifier is withheld.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

# Superseded-By and its pre-2011 PEP name (section 4), last-call-deadline with its earlier names
# (the review period's three), the reason a proposal was withdrawn, a PEP's dated acceptance link
# and a PEP's Resolution: named once here, for the reader, H- and the CF header cues.
SUPERSEDED_NAMES = ("superseded-by", "replaced-by")
LAST_CALL_NAMES = ("last-call-deadline", "review-period-end", "review-period-ends",
                   "review-end-date")
WITHDRAWAL_REASON_NAMES = ("withdrawal-reason",)
ACCEPTED_NAMES = ("accepted",)
RESOLUTION_NAMES = ("resolution",)
# Section 7's CF header cues read only these fields: the last-call names whatever their value,
# these names whenever they have a value, and a Resolution only when its value holds a decision
# word, since a resolution link alone records that a pronouncement was made, not which way it went.
CUE_WITH_A_VALUE = (*SUPERSEDED_NAMES, *WITHDRAWAL_REASON_NAMES, *ACCEPTED_NAMES)
CUE_NAMES = (*LAST_CALL_NAMES, *CUE_WITH_A_VALUE, *RESOLUTION_NAMES)
# The fields the H- arm withholds (section 8), in its order; every field a cue reads is among
# them. Require and dependencies are other names of requires. A corpus test classifies every key
# the corpora use.
WITHHELD = (
    "status",
    *RESOLUTION_NAMES,
    *ACCEPTED_NAMES,
    *SUPERSEDED_NAMES,
    "replaces",
    "requires",
    "require",
    "dependencies",
    *LAST_CALL_NAMES,
    *WITHDRAWAL_REASON_NAMES,
)

_WINDOW = 40  # a block without front matter starts within this many lines
_KEY = r"[A-Za-z][A-Za-z0-9-]*"
_QUALIFIER = r"\([^()]*\)"  # as in "requires (*optional)"
_FIELD = re.compile(rf"({_KEY})(?:[ \t]*({_QUALIFIER}))?[ \t]*:(.*)")
_RUN_LINE = re.compile(rf"\s*{_KEY}(?:[ \t]*{_QUALIFIER})?:\s*\S")
_BOM = "\ufeff"
_BODY_STARTS = ("```", "#")  # a code fence or a Markdown heading: the body begins at this line


@dataclass
class _Field:
    name: str  # the key without any qualifier, case-folded: "requires"
    key: str  # the recorded name, case-folded: "requires" or "requires (*optional)"
    start: int  # offset of the field's first line
    end: int  # offset just past its last line, continuation lines included
    parts: list[str]  # the value on the first line, then each continuation line, stripped

    def extend(self, end: int, line: str) -> None:
        self.end = end
        self.parts.append(line.strip())

    @property
    def value(self) -> str:
        return _unquote(" ".join(part for part in self.parts if part))


@dataclass
class _Block:
    start: int
    end: int
    fields: list[_Field]


def _unquote(value: str) -> str:
    """Strip the value and remove one pair of matching quotes around all of it."""
    value = value.strip()
    quote = value[:1]
    if quote in ("'", '"') and len(value) >= 2 and value.endswith(quote):
        if quote not in value[1:-1]:  # '"a" and "b"' is not one quoted string
            return value[1:-1]
    return value


def _lines(text: str, pos: int) -> Iterator[tuple[int, int, str]]:
    """(start, end, content) of each line from ``pos``; ``end`` includes the line ending."""
    size = len(text)
    while pos < size:
        newline = text.find("\n", pos)
        end = size if newline < 0 else newline + 1
        yield pos, end, text[pos:end].rstrip("\r\n")
        pos = end


def _new_field(start: int, end: int, line: str) -> _Field | None:
    match = _FIELD.match(line)
    if match is None:
        return None
    name = match.group(1).casefold()
    qualifier = match.group(2)
    key = name if qualifier is None else f"{name} ({' '.join(qualifier[1:-1].casefold().split())})"
    return _Field(name, key, start, end, [match.group(3).strip()])


def _scan_pep(text: str) -> _Block:
    """The block from the first non-blank line to the next blank line, which it includes."""
    pos = len(_BOM) if text.startswith(_BOM) else 0
    start: int | None = None
    fields: list[_Field] = []
    current: _Field | None = None
    for line_start, line_end, content in _lines(text, pos):
        if not content.strip():
            if start is None:
                continue  # a blank line before the header: two real PEP versions begin with one
            return _Block(start, line_end, fields)
        if start is None:
            start = line_start
        if content[0] in " \t":
            if current is not None:
                current.extend(line_end, content)
            continue
        current = _new_field(line_start, line_end, content)
        if current is not None:
            fields.append(current)
    if start is None:
        return _Block(pos, pos, [])  # no non-blank line, so no header: the body is all of it
    return _Block(start, len(text), fields)


def _scan_front_matter(text: str, start: int, pos: int) -> _Block:
    """Front matter from the opening ``---`` line (at ``start``) to the next ``---`` line."""
    fields: list[_Field] = []
    current: _Field | None = None
    for line_start, line_end, content in _lines(text, pos):
        if content.rstrip() == "---":
            return _Block(start, line_end, fields)
        field = _new_field(line_start, line_end, content)  # a key starts in column 0
        if field is not None:
            fields.append(field)
            current = field
        elif current is not None and content.strip():
            current.extend(line_end, content)
    return _Block(start, len(text), fields)  # never closed: it runs to the end of the text


def _scan_run(text: str, pos: int) -> _Block:
    """The block that starts at the first ``Key: value`` line within the first 40 lines.

    It ends at the next blank line or ``</pre>`` line (both inside the block), or before the next
    code-fence or Markdown-heading line (which starts the body). A key line may have an empty
    value; any other line continues the field above it.
    """
    fields: list[_Field] = []
    current: _Field | None = None
    start: int | None = None
    for index, (line_start, line_end, content) in enumerate(_lines(text, pos)):
        if start is None:
            if index >= _WINDOW:
                break
            if not _RUN_LINE.match(content):
                continue
            start = line_start
        else:
            line = content.strip()
            if not line or line.startswith("</pre>"):
                return _Block(start, line_end, fields)
            if line.startswith(_BODY_STARTS):
                return _Block(start, line_start, fields)
        field = _new_field(line_start, line_end, content.lstrip())
        if field is not None:
            fields.append(field)
            current = field
        elif current is not None:
            current.extend(line_end, content)
    if start is None:
        return _Block(pos, pos, [])
    return _Block(start, len(text), fields)


def _scan_eth(text: str) -> _Block:
    pos = len(_BOM) if text.startswith(_BOM) else 0
    for line_start, line_end, content in _lines(text, pos):
        if content.strip():
            if content.rstrip() == "---":
                return _scan_front_matter(text, line_start, line_end)
            break
    return _scan_run(text, pos)


_SCANNERS = {"pep": _scan_pep, "eth": _scan_eth}


def _scan(family: str, text: str) -> _Block:
    try:
        scan = _SCANNERS[family]
    except KeyError:
        raise ValueError(f"unknown family {family!r}: expected 'pep' or 'eth'") from None
    return scan(text)


def parse_header(family: str, text: str) -> dict[str, str]:
    """The header's fields: keys case-folded, values stripped and unquoted, continuations joined.

    A qualified key such as ``requires (*optional)`` is recorded under its full name, never as the
    plain field. If a key is repeated, its first occurrence is kept.
    """
    header: dict[str, str] = {}
    for field in _scan(family, text).fields:
        header.setdefault(field.key, field.value)
    return header


def header_span(family: str, text: str) -> tuple[int, int]:
    """The character offsets of the header block; the body is ``text[end:]``.

    ``end`` is just past the line that closes the block (the blank line that ends a PEP header
    or an indented block, the closing ``---`` of front matter, or the ``</pre>`` line of a
    mediawiki block), or the end of the text if nothing closes it. The one exception is a
    code-fence or Markdown-heading line directly under an indented or ``<pre>`` block: it
    belongs to the body, so ``end`` is the start of that line. ``start`` is the first line of the
    block (a PEP's first non-blank line, or the opening ``---`` of front matter), after any
    byte-order mark. A text with no header (a PEP with no non-blank line; an EIP or ERC with
    neither front matter nor a ``Key: value`` line in its first 40 lines) has an empty span just
    after any byte-order mark, so its body is the whole text after that mark.
    """
    block = _scan(family, text)
    return block.start, block.end


def number_list(value: str | None) -> list[int]:
    """Proposal numbers from a requires, replaces or superseded-by value.

    The value is split on commas and whitespace and only the digits of each item are kept; empty
    items are dropped, the order is kept and duplicates are removed.
    """
    numbers: dict[int, None] = {}
    for item in re.split(r"[,\s]+", value or ""):
        digits = re.sub(r"[^0-9]", "", item)
        if digits:
            numbers[int(digits)] = None
    return list(numbers)


def superseders(header: dict[str, str]) -> list[int]:
    """The proposals named by superseded-by and, under its pre-2011 PEP name, replaced-by
    (``SUPERSEDED_NAMES``), in that order, each once."""
    named = [n for name in SUPERSEDED_NAMES for n in number_list(header.get(name))]
    return list(dict.fromkeys(named))


def strip_withheld(family: str, text: str) -> str:
    """``text`` without the lines of each withheld field, continuation lines included.

    A field is withheld when its name without any qualifier is in ``WITHHELD``, so
    ``requires (*optional)`` goes too. Only the header block is touched; everything else is
    byte-identical.
    """
    kept: list[str] = []
    pos = 0
    for field in _scan(family, text).fields:
        if field.name in WITHHELD:
            kept.append(text[pos : field.start])
            pos = field.end
    kept.append(text[pos:])
    return "".join(kept)
