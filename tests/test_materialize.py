"""``asofbench materialize``: an arm's stream rebuilt as the proposal files as of a date (11).

A small history is committed to three throwaway repositories laid out like the clones; the H+ and
H- streams are written through the command line and materialised as of several dates. The
expected trees are written out by hand from that history: each present proposal's file is
``git show <as-of commit>:<path>`` (H+) or ``strip_withheld`` of it (H-), under
``<repo>/<file name>``, and nothing else is written.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import date
from pathlib import Path

import pytest
from gitrepo import TempRepo

from asofbench import cli
from asofbench.headers import strip_withheld
from asofbench.materialize import file_name

BODY = "".join(f"Body line {i} of the proposal text.\n" for i in range(24))
MINUS = chr(0x2212)


def pep(number: int, status: str, *, headers=(), body=BODY) -> str:
    lines = [f"PEP: {number}", "Title: Example", f"Status: {status}", *headers]
    return "\n".join(lines) + "\n\n" + body


def eip(number: int, status: str, *, headers=(), body=BODY) -> str:
    lines = ["---", f"eip: {number}", "title: Example", f"status: {status}", *headers, "---"]
    return "\n".join(lines) + "\n" + body


def git_show(repo: Path, spec: str) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL="/dev/null")
    done = subprocess.run(["git", "-C", str(repo), "show", "--no-textconv", spec], env=env,
                          capture_output=True, check=True)
    return done.stdout.decode("utf-8", errors="replace")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """The history, for each checked date the expected files (name -> (repo, commit, path)), and
    every commit that gives the streams an episode, in stream order: (commit, recorded time, the
    files it touched as the tree names them, message)."""
    root = tmp_path_factory.mktemp("clones")
    repos = {name: TempRepo(root / name) for name in ("peps", "EIPs", "ERCs")}
    for repo in repos.values():
        repo.write("README.md", "Proposals.\n")
        repo.commit("start", "2000-01-01T00:00:00Z")
    peps, eips, ercs = repos["peps"], repos["EIPs"], repos["ERCs"]
    peps.write("pep-0001.txt", pep(1, "Draft", headers=["Resolution: https://example.invalid/1"]))
    peps.write("pep-0002.txt", pep(2, "Draft"))
    c1 = peps.commit("Add PEP 1 and PEP 2", "2020-01-01T12:00:00Z")
    eips.write("EIPS/eip-5.md", eip(5, "Draft", headers=["requires: 1",
                                                         "review-period-end: 2020-03-01"]))
    eips.write("EIPS/eip-20.md", eip(20, "Draft"))
    e1 = eips.commit("Add EIP 5 and EIP 20", "2020-01-15T12:00:00Z")
    peps.write("pep-0001.txt", pep(1, "Accepted", headers=["Resolution: https://example.invalid/1"]))
    c2 = peps.commit("Accept PEP 1", "2020-02-01T12:00:00Z")
    eips.write("EIPS/eip-5.md", eip(5, "Last Call", headers=["requires: 1, 2"]))
    eips.write("EIPS/eip-20.md", eip(20, "Final"))
    e2 = eips.commit("Move EIP 5 on; EIP 20 is Final", "2020-02-15T12:00:00Z")
    peps.mv("pep-0001.txt", "peps/pep-0001.rst")
    peps.write("peps/pep-0001.rst", pep(1, "Final", headers=["Resolution: https://example.invalid/1"]))
    peps.rm("pep-0002.txt")
    c3 = peps.commit("Move PEP 1 into peps/ as Final; delete PEP 2", "2020-03-01T12:00:00Z")
    # Re-added with a form feed inside a line and no newline at the end.
    peps.write("pep-0002.txt", pep(2, "Rejected", body="Page one\x0c\nPage two"))
    c4 = peps.commit("Re-add PEP 2", "2020-04-01T12:00:00Z")
    ercs.write("ERCS/erc-20.md", eip(20, "Final"))
    r1 = ercs.commit("complete erc move", "2023-10-25T15:11:00Z")
    eips.write("EIPS/eip-20.md", eip(20, "Moved"))
    eips.commit("Mark moved ERCs", "2023-10-26T12:00:00Z")
    first = {"peps/pep-0001": ("peps", c1, "pep-0001.txt"),
             "peps/pep-0002": ("peps", c1, "pep-0002.txt")}
    eips_1 = {"EIPs/eip-5": ("EIPs", e1, "EIPS/eip-5.md"),
              "EIPs/eip-20": ("EIPs", e1, "EIPS/eip-20.md")}
    eips_2 = {"EIPs/eip-5": ("EIPs", e2, "EIPS/eip-5.md"),
              "EIPs/eip-20": ("EIPs", e2, "EIPS/eip-20.md")}
    after_c3 = {"peps/pep-0001": ("peps", c3, "peps/pep-0001.rst")}
    after_c4 = {**after_c3, "peps/pep-0002": ("peps", c4, "pep-0002.txt")}
    moved = {"EIPs/eip-5": ("EIPs", e2, "EIPS/eip-5.md"),
             "ERCs/erc-20": ("ERCs", r1, "ERCS/erc-20.md")}
    expected = {
        date(2019, 12, 31): {},
        date(2020, 1, 1): first,
        date(2020, 1, 20): {**first, **eips_1},
        date(2020, 2, 1): {**first, "peps/pep-0001": ("peps", c2, "pep-0001.txt"), **eips_1},
        date(2020, 2, 29): {**first, "peps/pep-0001": ("peps", c2, "pep-0001.txt"), **eips_2},
        date(2020, 3, 1): {**after_c3, **eips_2},
        date(2020, 4, 1): {**after_c4, **eips_2},
        date(2023, 10, 24): {**after_c4, **eips_2},
        date(2023, 10, 25): {**after_c4, **moved},
        date(2026, 10, 27): {**after_c4, **moved},
    }
    # The EIP repository's Moved marker gives no episode (section 6), so it is in no log.
    history = [
        (c1, "2020-01-01T12:00:00Z", ["peps/pep-0001", "peps/pep-0002"], "Add PEP 1 and PEP 2"),
        (e1, "2020-01-15T12:00:00Z", ["EIPs/eip-5", "EIPs/eip-20"], "Add EIP 5 and EIP 20"),
        (c2, "2020-02-01T12:00:00Z", ["peps/pep-0001"], "Accept PEP 1"),
        (e2, "2020-02-15T12:00:00Z", ["EIPs/eip-5", "EIPs/eip-20"],
         "Move EIP 5 on; EIP 20 is Final"),
        (c3, "2020-03-01T12:00:00Z", ["peps/pep-0001", "peps/pep-0002"],
         "Move PEP 1 into peps/ as Final; delete PEP 2"),
        (c4, "2020-04-01T12:00:00Z", ["peps/pep-0002"], "Re-add PEP 2"),
        (r1, "2023-10-25T15:11:00Z", ["ERCs/erc-20"], "complete erc move"),
    ]
    return root, expected, history


@pytest.fixture(scope="module")
def streams(world, tmp_path_factory):
    root = world[0]
    out = tmp_path_factory.mktemp("streams")
    # The CF arm reads its proposals from the build's gold: PEP 1 is the one counterfactual here.
    data = out / "data"
    data.mkdir()
    cf_item = {"id": "CF-counterfactual-PEP-1-2020-01-20", "meta": {"ref": 'PEP 1 ("Example")'}}
    (data / "gold.jsonl").write_text(json.dumps(cf_item) + "\n", encoding="utf-8")
    found = {}
    for arm, name in (("H+", "h-plus.jsonl"), ("H-", "h-minus.jsonl"), ("CF", "cf.jsonl")):
        argv = ["stream", "--clones", str(root), "--data", str(data), "--arm", arm,
                "--out", str(out / name)]
        assert cli.main(argv) == 0
        found[arm] = out / name
    return found


def materialize(stream: Path, day: date, out: Path, *extra: str) -> int:
    return cli.main(["materialize", "--stream", str(stream), "--as-of", day.isoformat(),
                     "--out", str(out), *extra])


def tree(out: Path) -> dict[str, str]:
    """The proposal files written under ``out``; the log beside them is read with ``log``."""
    return {p.relative_to(out).as_posix(): p.read_bytes().decode("utf-8")
            for p in sorted(out.rglob("*")) if p.is_file() and p.name != "git-log.txt"}


def log(out: Path) -> str:
    return (out / "git-log.txt").read_bytes().decode("utf-8")


def entry(commit: str, recorded_at: str, groups: list[tuple[list[str], str]]) -> str:
    """One commit of git-log.txt: its id, its recorded time and, for each message its episodes
    carry, the files that carry it and then the message, each line indented by four spaces."""
    lines = [f"commit {commit}", f"Date: {recorded_at}"]
    for paths, message in groups:
        lines += [*(f"File: {path}" for path in paths), "",
                  *(f"    {line}" for line in message.split("\n")), ""]
    return "".join(line + "\n" for line in lines)


def test_file_names_follow_the_repositories_naming():
    assert file_name("peps", "PEP-8") == "pep-0008"
    assert file_name("peps", "PEP-8016") == "pep-8016"
    assert file_name("EIPs", "EIP-1559") == "eip-1559"
    assert file_name("EIPs", "ERC-20") == "eip-20"  # an ERC's EIP-repository file
    assert file_name("ERCs", "ERC-20") == "erc-20"


@pytest.mark.parametrize("arm", ["H+", "H-"])
def test_the_tree_as_of_each_date_is_the_files_as_they_stood(world, streams, tmp_path, arm):
    root, expected, _ = world
    for day, files in expected.items():
        out = tmp_path / f"{arm}-{day.isoformat()}"
        assert materialize(streams[arm], day, out) == 0, day
        want = {}
        for name, (repo, commit, path) in files.items():
            text = git_show(root / repo, f"{commit}:{path}")
            want[name] = text if arm == "H+" else strip_withheld(
                "pep" if repo == "peps" else "eth", text)
        assert tree(out) == want, (arm, day)
        assert out.is_dir()  # an empty tree is still a directory


def test_the_hminus_tree_holds_no_withheld_header(world, streams, tmp_path):
    out = tmp_path / "h-minus"
    assert materialize(streams["H-"], date(2020, 2, 29), out) == 0
    text = "".join(tree(out).values())
    assert "Status:" not in text and "status:" not in text and "requires:" not in text
    assert "review-period-end" not in text and "Resolution:" not in text
    assert "Body line 0" in text


LOG_AT_2020_03_01 = """\
commit {c3}
Date: 2020-03-01T12:00:00Z
File: peps/pep-0001
File: peps/pep-0002

    Move PEP 1 into peps/ as Final; delete PEP 2

commit {e2}
Date: 2020-02-15T12:00:00Z
File: EIPs/eip-5
File: EIPs/eip-20

    Move EIP 5 on; EIP 20 is Final

commit {c2}
Date: 2020-02-01T12:00:00Z
File: peps/pep-0001

    Accept PEP 1

commit {e1}
Date: 2020-01-15T12:00:00Z
File: EIPs/eip-5
File: EIPs/eip-20

    Add EIP 5 and EIP 20

commit {c1}
Date: 2020-01-01T12:00:00Z
File: peps/pep-0001
File: peps/pep-0002

    Add PEP 1 and PEP 2

"""


@pytest.mark.parametrize("arm", ["H+", "H-"])
def test_the_log_beside_the_tree_is_the_streams_commits_up_to_the_date(world, streams, tmp_path,
                                                                       arm):
    """Section 11: grep at D also reads ``git-log.txt``, one entry per commit that gives the
    arm's stream an episode recorded by the end of D, newest first: the commit, its recorded
    time, the files it touched as the tree names them, and its message as the stream keeps it;
    no diff and no text. Nothing recorded after D is in it."""
    _, expected, history = world
    c1, e1, c2, e2, c3 = (commit for commit, *_ in history[:5])
    out = tmp_path / f"{arm}-written-out"
    assert materialize(streams[arm], date(2020, 3, 1), out) == 0
    assert log(out) == LOG_AT_2020_03_01.format(c1=c1, e1=e1, c2=c2, e2=e2, c3=c3)
    for day in expected:
        out = tmp_path / f"{arm}-{day.isoformat()}"
        assert materialize(streams[arm], day, out) == 0, day
        until = f"{day.isoformat()}T23:59:59Z"
        shown = [entry(commit, when, [(paths, message)])
                 for commit, when, paths, message in reversed(history) if when <= until]
        assert log(out) == "".join(shown), (arm, day)  # empty before the first episode
    assert "Body line" not in log(tmp_path / f"{arm}-2026-10-27")  # no text, no diff


def test_the_cf_log_keeps_the_messages_the_cf_stream_gives(world, streams, tmp_path):
    """Section 7: in a commit that changes a CF status line, the CF stream gives the CF
    proposal's episode the message "Update status of {ref}" and every other episode "Update
    {ref}"; the log keeps each message with the files whose episodes carry it."""
    _, _, history = world
    c1, e1, c2, e2, c3 = (commit for commit, *_ in history[:5])
    out = tmp_path / "cf"
    assert materialize(streams["CF"], date(2020, 3, 1), out) == 0
    status = 'Update status of PEP 1 ("Example")'
    assert log(out) == "".join([
        entry(c3, "2020-03-01T12:00:00Z", [(["peps/pep-0001"], status),
                                           (["peps/pep-0002"], 'Update PEP 2 ("Example")')]),
        entry(e2, "2020-02-15T12:00:00Z", [(["EIPs/eip-5", "EIPs/eip-20"],
                                            "Move EIP 5 on; EIP 20 is Final")]),
        entry(c2, "2020-02-01T12:00:00Z", [(["peps/pep-0001"], status)]),
        entry(e1, "2020-01-15T12:00:00Z", [(["EIPs/eip-5", "EIPs/eip-20"],
                                            "Add EIP 5 and EIP 20")]),
        entry(c1, "2020-01-01T12:00:00Z", [(["peps/pep-0001", "peps/pep-0002"],
                                            "Add PEP 1 and PEP 2")]),
    ])


def test_the_log_keeps_every_message_exactly(streams, tmp_path):
    """A message's blank lines, carriage returns, tabs and outer spaces stay as the stream keeps
    them: each of its lines, split at newlines only, is the log's line after four spaces."""
    rows = [json.loads(line) for line in streams["H+"].read_text(encoding="utf-8").splitlines()]
    message = "  Subject\twith a tab \r\n\nBody line\r\n  \n"
    stream = tmp_path / "odd.jsonl"
    stream.write_text(json.dumps({**rows[0], "message": message}) + "\n", encoding="utf-8")
    out = tmp_path / "out"
    assert materialize(stream, date(2026, 10, 27), out) == 0
    written = log(out).split("\n\n", 1)[1]  # after the commit, its date and its files
    lines = written.split("\n")[: message.count("\n") + 1]
    assert all(line.startswith("    ") for line in lines)
    assert "\n".join(line[4:] for line in lines) == message


def test_materialize_refuses_an_out_directory_that_holds_files(streams, tmp_path, capsys):
    out = tmp_path / "out"
    out.mkdir()
    (out / "stray.txt").write_text("x\n")
    assert materialize(streams["H+"], date(2020, 3, 1), out) == 2
    assert "asofbench materialize: " in capsys.readouterr().err
    assert sorted(p.name for p in out.iterdir()) == ["stray.txt"]
    empty = tmp_path / "empty"
    empty.mkdir()
    assert materialize(streams["H+"], date(2020, 3, 1), empty) == 0


def test_materialize_refuses_a_stream_it_cannot_replay(streams, tmp_path, capsys):
    rows = [json.loads(line) for line in streams["H+"].read_text(encoding="utf-8").splitlines()]
    modify = next(i for i, row in enumerate(rows) if row["kind"] == "modify")
    later = next(row for row in rows if row["recorded_at"] > rows[0]["recorded_at"])
    cases = {
        "not-json": "{\n",
        "no-text": json.dumps({k: v for k, v in rows[0].items() if k != "text"}) + "\n",
        "no-message": json.dumps({k: v for k, v in rows[0].items() if k != "message"}) + "\n",
        "bad-commit": json.dumps({**rows[0], "commit": "HEAD\ncommit 0"}) + "\n",
        "bad-kind": json.dumps({**rows[0], "kind": "copy"}) + "\n",
        "bad-diff": "".join(json.dumps(r) + "\n" for r in rows[:modify])
        + json.dumps({**rows[modify], "text": "--- a\n+++ b\n@@ -1 +1 @@\n-no such line\n+x\n"})
        + "\n",
        "out-of-order": "".join(json.dumps(r) + "\n" for r in [later, rows[0]]),
        "orphan-diff": json.dumps(rows[modify]) + "\n",
    }
    for name, text in cases.items():
        stream = tmp_path / f"{name}.jsonl"
        stream.write_text(text, encoding="utf-8")
        assert materialize(stream, date(2026, 10, 27), tmp_path / f"out-{name}") == 2, name
        err = capsys.readouterr().err
        assert err.startswith("asofbench materialize: ") and "Traceback" not in err, name
    assert materialize(tmp_path / "missing.jsonl", date(2020, 1, 1), tmp_path / "o") == 2
    with pytest.raises(SystemExit):
        cli.main(["materialize", "--stream", str(streams["H+"]), "--as-of", "2020-13-01",
                  "--out", str(tmp_path / "bad-date")])


def test_materialize_checks_the_registered_stream_when_asked(streams, tmp_path, capsys):
    data = tmp_path / "data"
    data.mkdir()
    registered = {}
    for arm, path in streams.items():
        stats = json.loads(Path(f"{path}.stats.json").read_text(encoding="utf-8"))
        assert stats["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        registered[arm] = stats
    registered["H+"] = {**registered["H+"], "sha256": "0" * 64}  # not the H+ stream written
    (data / "MANIFEST.json").write_text(json.dumps({"streams": registered}), encoding="utf-8")
    # The registration writes the arm H-, with a minus sign; both spellings are one arm.
    for arm in ("H-", f"H{MINUS}"):
        out = tmp_path / f"ok-{len(arm.encode())}"
        assert materialize(streams["H-"], date(2020, 3, 1), out, "--arm", arm, "--data",
                           str(data)) == 0
    assert materialize(streams["H+"], date(2020, 3, 1), tmp_path / "wrong", "--arm", "H+",
                       "--data", str(data)) == 2
    assert "registered" in capsys.readouterr().err
    assert not (tmp_path / "wrong").exists()
    with pytest.raises(SystemExit):  # an arm needs the build whose manifest registers it
        cli.main(["materialize", "--stream", str(streams["H-"]), "--as-of", "2020-03-01",
                  "--out", str(tmp_path / "x"), "--arm", "H-"])
