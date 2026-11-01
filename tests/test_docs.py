"""The documents as a reader meets them: every link between the Markdown files reaches a file and
a heading, docs/data-format.md names exactly the fields the data files and the code hold, the toy
world runs, and the source distribution holds every tracked top-level file and directory."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import re
import subprocess
import tomllib
from pathlib import Path

import pytest

from asofbench.scoring import ACCOUNTING, ROW_FIELDS
from asofbench.stream import Episode

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "v0.1"

LINK = re.compile(r"\[[^\]]*\]\(<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\)")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
HTML_ANCHOR = re.compile(r"<a\s+(?:id|name)=\"([^\"]+)\"")
EXTERNAL = ("http://", "https://", "mailto:")


def tracked(pattern: str = "") -> list[str] | None:
    """The paths git tracks under the root (matching ``pattern``), or None outside a checkout."""
    command = ["git", "-C", str(ROOT), "ls-files", "-z", *([pattern] if pattern else [])]
    try:
        listed = subprocess.run(command, capture_output=True, check=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return [name for name in listed.split("\0") if name]


def markdown_files() -> list[Path]:
    """The Markdown files git tracks, or, outside a checkout, every one under the root but those
    in build directories or hidden ones other than .github."""
    names = tracked("*.md")
    if names is not None:
        return sorted(ROOT / name for name in names)
    return sorted(path for path in ROOT.rglob("*.md")
                  if not any(skipped(part) for part in path.relative_to(ROOT).parts[:-1]))


def skipped(part: str) -> bool:
    """Whether a directory holds no document of the repository: a build or hidden one, .github
    apart."""
    return part in ("build", "dist") or (part.startswith(".") and part != ".github")


def prose_lines(text: str) -> list[tuple[int, str]]:
    """The lines outside fenced code blocks, numbered from 1, with inline code spans removed."""
    lines, fenced = [], False
    for number, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
        elif not fenced:
            lines.append((number, re.sub(r"`[^`]*`", "", line)))
    return lines


def slug(heading: str) -> str:
    """GitHub's anchor for a heading: lower case, without punctuation, spaces as hyphens."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower().replace("`", "")).replace(" ", "-")


def anchors(path: Path) -> set[str]:
    """Every fragment a link may name in ``path``: its headings' slugs, with GitHub's ``-1``,
    ``-2`` suffixes on repeats, and its explicit HTML anchors."""
    found, seen = set(), {}
    for _, line in prose_lines(path.read_text(encoding="utf-8")):
        match = HEADING.match(line)
        if match:
            base = slug(match.group(2))
            n = seen.get(base, 0)
            found.add(base if n == 0 else f"{base}-{n}")
            seen[base] = n + 1
        found.update(HTML_ANCHOR.findall(line))
    return found


def link_gaps(files: list[Path], root: Path = ROOT) -> list[str]:
    """Each relative link in ``files`` whose file is not there, or whose fragment is not a heading
    of the Markdown file it names, as ``file:line: target (why)`` with the file under ``root``."""
    gaps = []
    for path in files:
        for number, line in prose_lines(path.read_text(encoding="utf-8")):
            for target in LINK.findall(line):
                if target.startswith(EXTERNAL):
                    continue
                name, _, fragment = target.partition("#")
                linked = (path.parent / name).resolve() if name else path
                where = f"{path.relative_to(root)}:{number}"
                if not linked.exists():
                    gaps.append(f"{where}: {target} (no such file)")
                elif fragment and linked.suffix == ".md" and fragment not in anchors(linked):
                    gaps.append(f"{where}: {target} (no such heading)")
    return gaps


def test_the_slug_is_githubs():
    assert slug("1. The question") == "1-the-question"
    assert slug('3. Recorded time and "as of"') == "3-recorded-time-and-as-of"
    assert slug("9. Extraction track (headers withheld)") == "9-extraction-track-headers-withheld"
    pilot = "the-pilot-the-gates-and-the-cost-pilot"
    assert slug("The pilot, the gates and the cost pilot") == pilot


def test_a_link_to_a_missing_file_or_heading_is_a_gap(tmp_path):
    (tmp_path / "a.md").write_text("# A\n\n## Kept heading\n\n[ok](#kept-heading) [ok](b.md#b)\n"
                                   "[file](missing.md) [heading](b.md#nowhere) `[code](x.md)`\n"
                                   "[web](https://example.invalid/x.md)\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("# B\n", encoding="utf-8")
    assert link_gaps([tmp_path / "a.md"], tmp_path) == ["a.md:6: missing.md (no such file)",
                                              "a.md:6: b.md#nowhere (no such heading)"]


def test_every_relative_link_reaches_a_file_and_a_heading_of_the_repository():
    assert link_gaps(markdown_files()) == []


def documented(page: str, section: str) -> list[set[str]]:
    """The field names the tables under ``## section`` of ``page`` give: one set per table, from
    each row's first cell, or from its second when the table's first column is the family."""
    text = (ROOT / page).read_text(encoding="utf-8").split(f"\n## {section}\n", 1)[1]
    text = text.split("\n## ", 1)[0]
    tables, current, key = [], None, 0
    for line in text.splitlines():
        if not line.startswith("|"):
            current = None
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if current is None:
            current, key = set(), 1 if cells[0] == "Family" else 0
            tables.append(current)
        elif not set(cells[0]) <= set("-:"):
            current.update(re.findall(r"`([^`]+)`", cells[key]))
    return tables


def records(name: str) -> list[dict]:
    path = DATA / name
    if name.endswith(".jsonl"):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [json.loads(path.read_text(encoding="utf-8"))]


def keys(found: list[dict]) -> set[str]:
    return {key for record in found for key in record}


def test_the_data_format_page_names_every_field_the_data_files_hold():
    page = "docs/data-format.md"
    for section, name in (("questions.jsonl", "questions.jsonl"), ("pilot.json", "pilot.json"),
                          ("timelines.jsonl", "timelines.jsonl"), ("edges.jsonl", "edges.jsonl"),
                          ("labelling/mention_pairs.jsonl", "labelling/mention_pairs.jsonl")):
        (table,) = documented(page, section)
        assert table == keys(records(name)), name
    fields, meta = documented(page, "gold.jsonl")
    gold = records("gold.jsonl")
    assert fields == keys(gold)
    assert meta | {"rank"} == {key for record in gold for key in record["meta"]}
    (manifest,) = documented(page, "REPORT.md and MANIFEST.json")
    assert manifest == keys(records("MANIFEST.json"))


def test_the_data_format_page_names_the_stream_and_answer_row_fields_the_code_holds():
    page = "docs/data-format.md"
    (stream,) = documented(page, "Streams")
    assert stream == {field.name for field in dataclasses.fields(Episode)}
    (rows,) = documented(page, "Answer rows, entries and items")
    assert rows == set(ROW_FIELDS) | set(ACCOUNTING)


def test_the_toy_world_runs_every_command(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("toy_world", ROOT / "examples" / "toy_world.py")
    toy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(toy)
    assert toy.main([str(tmp_path / "toy")]) == 0
    assert (tmp_path / "toy" / "results" / "results.md").is_file()
    assert "toy items in" in capsys.readouterr().out
    assert toy.main([str(tmp_path / "toy")]) == 2  # an existing directory is refused


def test_the_source_distribution_holds_every_tracked_top_level_entry():
    names = tracked()
    if names is None:
        pytest.skip("not a git checkout")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    listed = project["tool"]["hatch"]["build"]["targets"]["sdist"]["only-include"]
    assert sorted(listed) == sorted({name.split("/", 1)[0] for name in names})


def test_the_readmes_example_is_what_git_records_and_holds_no_answer_key():
    """The README's example asks PEP 695's status in the F1 template's own words, gives the
    commits, days and statuses that timelines.jsonl records for it, and names as its answer the
    status at the end of its date; and PEP 695 is in no item, so the front page holds no answer
    key: no question names it, and no answer list of a PEP item holds its number."""
    from asofbench.questions import TEMPLATES
    from asofbench.statuses import CHOICES

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    example = readme.split("\n## An example\n", 1)[1].split("\n## ", 1)[0]
    timeline = next(t for t in records("timelines.jsonl") if t["pid"] == "PEP-695")
    date = re.search(r"as of the end of (\d{4}-\d{2}-\d{2}) \(UTC\)\?", example).group(1)
    question = TEMPLATES["F1"].format(ref=f'PEP 695 ("{timeline["title"]}")', D=date,
                                      choices=", ".join(CHOICES["pep"]))
    assert f"*{question}*" in example
    rows = re.findall(r"^\| (\d{4}-\d{2}-\d{2}) \| `([0-9a-f]{8})` [^|]* \| ([A-Za-z ]+) \|$",
                      example, re.MULTILINE)
    assert rows == [(i["start"][:10], i["commit"][:8], i["status"]) for i in timeline["intervals"]]
    end = f"{date}T23:59:59Z"
    held = [i["status"] for i in timeline["intervals"]
            if i["start"] <= end and (i["end"] is None or i["end"] > end)]
    assert held == [re.search(r"So the answer is \*\*([A-Za-z]+)\*\*", example).group(1)]
    questions = records("questions.jsonl")
    assert [q["id"] for q in questions if q["proposal"] == "PEP-695" or "PEP 695 " in q["question"]
            ] == []
    peps = {q["id"] for q in questions if q["corpus"] == "pep"}
    assert [g["id"] for g in records("gold.jsonl")
            if g["id"] in peps and isinstance(g["gold"], list) and 695 in g["gold"]] == []
