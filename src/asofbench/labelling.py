"""``asofbench labelling-sheet``: the annotators' local sheet of passages (spec section 9).

The published mention-pair sample, ``labelling/mention_pairs.jsonl``, holds for each pair the two
proposals' ids and titles, which side mentions which, and where each mention is: the repository,
path and commit of the file at the pinned commit, and the character offsets of the mention in its
text as ``gitwalk.BlobReader`` reads it (UTF-8 with undecodable bytes replaced, a byte-order mark
and carriage returns kept). It holds no proposal text.

This command reads each mention's file from the clones, checks that the offsets still hold a
mention of the other proposal, and writes a Markdown sheet for the annotators with the passage
around each mention: the 200 characters of the body centred on it, with whitespace collapsed
(``questions.passage``). The sheet holds proposal text, so it stays on the annotators' machines:
``labelling-sheet*.md`` is git-ignored and the sheet is never published.
"""

from __future__ import annotations

import json
import re
import sys
from contextlib import ExitStack
from pathlib import Path

from .corpora import CANARY, REPOS
from .gitwalk import BlobReader, CloneError, blob_at, check_clone
from .headers import header_span
from .mentions import mentions
from .questions import passage

_FIELDS = ("pair_id", "corpus", "rank", "a", "a_title", "a_mentions_b", "a_mention_spans",
           "a_source", "b", "b_title", "b_mentions_a", "b_mention_spans", "b_source")
_PID = re.compile(r"(PEP|EIP|ERC)-([0-9]+)")


class SheetError(Exception):
    """A sample or clone the command cannot use; it reports it and exits 2."""


def _span_ok(span: object) -> bool:
    return (isinstance(span, dict) and span.get("repo") in REPOS
            and all(isinstance(span.get(k), str) for k in ("path", "commit"))
            and all(isinstance(span.get(k), int) and not isinstance(span.get(k), bool)
                    for k in ("start", "end"))
            and 0 <= span["start"] < span["end"])


def read_sample(path: Path) -> list[dict]:
    """The mention-pair records of ``path``, each checked for the fields the sheet needs."""
    if not path.is_file():
        raise SheetError(f"{path} not found: run `asofbench build` first")
    records = []
    with path.open(encoding="utf-8", newline="\n") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                raise SheetError(f"{path} line {number}: not JSON") from None
            if (not isinstance(record, dict) or any(k not in record for k in _FIELDS)
                    or not all(_PID.fullmatch(str(record[side])) for side in ("a", "b"))
                    or record["corpus"] not in ("pep", "eth")
                    or not all(isinstance(record[f"{side}_mention_spans"], list)
                               and all(_span_ok(s) for s in record[f"{side}_mention_spans"])
                               for side in ("a", "b"))):
                raise SheetError(f"{path} line {number}: not a mention-pair record")
            records.append(record)
    return records


def _passages(clones: Path, records: list[dict]) -> dict[tuple, str]:
    """The passage of every mention of the sample: (pair id, side, index) -> text."""
    repos = sorted({span["repo"] for r in records for side in ("a", "b")
                    for span in r[f"{side}_mention_spans"]})
    for repo in repos:
        check_clone(clones / repo)
    texts: dict[tuple[str, str, str], str] = {}
    found = {}
    with ExitStack() as stack:
        readers = {repo: stack.enter_context(BlobReader(clones / repo)) for repo in repos}
        for record in records:
            family = record["corpus"]
            for side, other in (("a", "b"), ("b", "a")):
                number = int(_PID.fullmatch(record[other]).group(2))
                for k, span in enumerate(record[f"{side}_mention_spans"]):
                    key = (span["repo"], span["commit"], span["path"])
                    if key not in texts:
                        blob = blob_at(clones / span["repo"], span["commit"], span["path"])
                        if blob is None:
                            raise SheetError(f"the {span['repo']} clone has no {span['path']} at "
                                             f"{span['commit']}: is it the pinned clone?")
                        try:
                            texts[key] = readers[span["repo"]].read(blob)
                        except ValueError:
                            raise SheetError(f"{span['path']} at {span['commit']} is not a "
                                             "file") from None
                    text = texts[key]
                    start = header_span(family, text)[1]
                    mention = text[span["start"]:span["end"]]
                    if span["start"] < start or mentions(family, mention, -1) != {number}:
                        raise SheetError(f"{record['pair_id']}: {span['path']} at "
                                         f"{span['commit']} holds no mention of {record[other]} "
                                         f"at {span['start']}-{span['end']}")
                    found[(record["pair_id"], side, k)] = passage(
                        text[start:], span["start"] - start, span["end"] - start)
    return found


def _fenced(text: str) -> list[str]:
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return [f"{fence}text", text, fence]


def render(records: list[dict], passages: dict[tuple, str]) -> str:
    lines = [
        "# asof-bench v0.1 labelling sheet", "", CANARY, "",
        "This sheet is for the annotators of the mention-pair sample (labelling/GUIDELINES.md) "
        "and is not to be published: it holds proposal text, read from the pinned clones by "
        "`asofbench labelling-sheet`. Each passage is the 200 characters of a body centred on "
        "one mention, with whitespace collapsed.",
    ]
    for record in records:
        lines += ["", f"## {record['pair_id']}", "", f"Corpus {record['corpus']}, rank "
                  f"{record['rank']}.", ""]
        for side, other in (("a", "b"), ("b", "a")):
            source = record[f"{side}_source"] or {}
            said = "yes" if record[f"{side}_mentions_{other}"] else "no"
            lines.append(f"- {side.upper()}: {record[side]} (\"{record[f'{side}_title']}\"), "
                         f"{source.get('path')} at {str(source.get('commit'))[:12]} in "
                         f"{source.get('repo')}; mentions {other.upper()}: {said}.")
        for side, other in (("a", "b"), ("b", "a")):
            for k in range(len(record[f"{side}_mention_spans"])):
                lines += ["", f"{side.upper()} mentions {other.upper()}, passage {k + 1}:", "",
                          *_fenced(passages[(record["pair_id"], side, k)])]
    return "\n".join(lines) + "\n"


def main(args) -> int:
    """``asofbench labelling-sheet --clones DIR --data DIR --out FILE``."""
    out = Path(args.out)
    try:
        records = read_sample(Path(args.data) / "labelling" / "mention_pairs.jsonl")
        text = render(records, _passages(Path(args.clones), records))
    except UnicodeDecodeError:
        print("asofbench labelling-sheet: the sample is not UTF-8 text", file=sys.stderr)
        return 2
    except (SheetError, CloneError) as e:
        print(f"asofbench labelling-sheet: {e}", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"asofbench labelling-sheet: {len(records)} pairs, {len(text)} characters, written to "
          f"{out} (keep it local)")
    return 0
