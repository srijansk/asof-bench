"""The mention-pair sample and ``asofbench labelling-sheet`` (spec section 9).

The published sample records where each mention is (repository, path, commit and character
offsets into the blob at the pinned commit) and no proposal text; the annotators' passages come
from a local sheet that the command builds from the clones. A small history is committed to three
throwaway repositories, built, and the sheet's passages are checked against ``git show`` of each
source with the window written out here again: the 200 characters centred on the mention, kept
inside the body and with whitespace collapsed.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest
from gitrepo import TempRepo

from asofbench import cli
from asofbench.corpora import CANARY

ROOT = Path(__file__).resolve().parents[1]
FILLER = " ".join(f"word{i}" for i in range(60))  # about 450 characters of prose


def pep(number: int, title: str, body: str, headers=()) -> str:
    lines = [f"PEP: {number}", f"Title: {title}", "Status: Draft", *headers]
    return "\n".join(lines) + "\n\n" + body


def eip(number: int, title: str, body: str) -> str:
    return f"---\neip: {number}\ntitle: {title}\nstatus: Draft\n---\n{body}"


def git_show(repo: Path, spec: str) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL="/dev/null")
    done = subprocess.run(["git", "-C", str(repo), "show", "--no-textconv", spec], env=env,
                          capture_output=True, check=True)
    return done.stdout.decode("utf-8", errors="replace")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("labelling")
    clones = root / "clones"
    repos = {name: TempRepo(clones / name) for name in ("peps", "EIPs", "ERCs")}
    for repo in repos.values():
        repo.write("README.md", "Proposals.\n")
        repo.commit("start", "2000-01-01T00:00:00Z")
    # PEP 1 mentions PEP 2 three times: near its start, far into the text, and again just after,
    # inside the second passage, which therefore gives no third one.
    body = (f"Compare PEP 2 here. {FILLER}\n\n{FILLER} The PEP 2 idea, and PEP-2 again.\n"
            f"{FILLER}\n")
    repos["peps"].write("pep-0001.txt", pep(1, "First", body))
    repos["peps"].write("pep-0002.txt", pep(2, "Second", f"Nothing to see.\n{FILLER}\n"))
    repos["peps"].commit("Add PEP 1 and PEP 2", "2001-01-01T12:00:00Z")
    repos["EIPs"].write("EIPS/eip-5.md", eip(5, "Five", f"{FILLER}\nUses ERC-20 tokens.\n"))
    repos["EIPs"].commit("Add EIP 5", "2016-01-01T12:00:00Z")
    repos["ERCs"].write("ERCS/erc-20.md", eip(20, "Token", "A standard interface.\n"))
    repos["ERCs"].commit("Add ERC 20", "2024-01-01T12:00:00Z")
    data = root / "data"
    assert cli.main(["build", "--clones", str(clones), "--out", str(data)]) == 0
    sheet = root / "labelling-sheet.md"
    argv = ["labelling-sheet", "--clones", str(clones), "--data", str(data), "--out", str(sheet)]
    assert cli.main(argv) == 0
    return clones, data, sheet


def records(data: Path) -> list[dict]:
    text = (data / "labelling" / "mention_pairs.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def window(text: str, start: int, end: int) -> str:
    """The passage, written out again: the body starts after the header (the first blank line
    of a PEP, the closing ``---`` of front matter); 200 characters centred on the mention, moved
    to stay inside the body; whitespace collapsed."""
    if text.startswith("---\n"):
        body_start = text.index("\n---\n") + 5
    else:
        body_start = text.index("\n\n") + 2
    body = text[body_start:]
    lo = max(0, (start + end) // 2 - body_start - 100)
    hi = min(len(body), lo + 200)
    lo = max(0, hi - 200)
    return " ".join(body[lo:hi].split())


def test_the_published_sample_holds_locations_and_no_text(built):
    clones, data, _ = built
    found = {r["pair_id"]: r for r in records(data)}
    assert set(found) == {"PEP-1~PEP-2", "EIP-5~ERC-20"}
    pep_pair = found["PEP-1~PEP-2"]
    assert (pep_pair["a_mentions_b"], pep_pair["b_mentions_a"]) == (True, False)
    assert pep_pair["b_mention_spans"] == []
    spans = pep_pair["a_mention_spans"]
    assert len(spans) == 2  # at most two per side, and passages do not overlap
    for span in spans:
        assert set(span) == {"repo", "path", "commit", "start", "end"}
        text = git_show(clones / span["repo"], f"{span['commit']}:{span['path']}")
        assert re.fullmatch(r"PEP[ -]2", text[span["start"]:span["end"]]), span
    published = (data / "labelling" / "mention_pairs.jsonl").read_text(encoding="utf-8")
    assert "Compare" not in published and "word1 " not in published and "tokens" not in published


def test_the_sheet_holds_each_pairs_passages(built):
    clones, data, sheet = built
    text = sheet.read_text(encoding="utf-8")
    assert text.startswith("# asof-bench v0.1 labelling sheet\n") and CANARY in text
    assert "not to be published" in text
    for record in records(data):
        assert f"## {record['pair_id']}" in text
        assert record["a_title"] in text and record["b_title"] in text
        for side in ("a", "b"):
            for span in record[f"{side}_mention_spans"]:
                source = git_show(clones / span["repo"], f"{span['commit']}:{span['path']}")
                passage = window(source, span["start"], span["end"])
                assert len(passage) <= 200 and f"\n{passage}\n" in text, passage
    assert "Compare PEP 2 here." in text and "Uses ERC-20 tokens." in text


def test_the_sheet_refuses_inputs_it_cannot_use(built, tmp_path, capsys):
    clones, data, _ = built

    def sheet(clones_dir: Path, data_dir: Path) -> int:
        return cli.main(["labelling-sheet", "--clones", str(clones_dir), "--data", str(data_dir),
                         "--out", str(tmp_path / "sheet.md")])

    assert sheet(clones, tmp_path / "no-data") == 2
    assert "mention_pairs.jsonl" in capsys.readouterr().err
    assert sheet(tmp_path / "no-clones", data) == 2
    assert "not a directory" in capsys.readouterr().err
    damaged = tmp_path / "damaged"
    (damaged / "labelling").mkdir(parents=True)
    rows = records(data)
    rows[0]["a_mention_spans"] = [{**rows[0]["a_mention_spans"][0], "commit": "0" * 40}]
    (damaged / "labelling" / "mention_pairs.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert sheet(clones, damaged) == 2
    assert "0" * 40 in capsys.readouterr().err
    (damaged / "labelling" / "mention_pairs.jsonl").write_text("{cut", encoding="utf-8")
    assert sheet(clones, damaged) == 2
    err = capsys.readouterr().err
    assert "line 1" in err and "Traceback" not in err
    assert not (tmp_path / "sheet.md").exists()


def test_the_sheet_is_kept_out_of_the_repository_and_the_guidelines_point_at_it():
    assert "labelling-sheet*.md" in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    guidelines = (ROOT / "labelling" / "GUIDELINES.md").read_text(encoding="utf-8")
    assert "asofbench labelling-sheet" in guidelines and "short passages" not in guidelines
