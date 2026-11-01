"""The read-only git walker: recorded time, first-parent changes, blob reads (spec section 3)."""

from __future__ import annotations

import os
import re
from datetime import UTC, date, datetime

import pytest
from gitrepo import TempRepo, clone

from asofbench import gitwalk
from asofbench.gitwalk import (
    BlobReader,
    as_of_commit,
    author_names,
    end_of_day_ts,
    file_changes,
    first_parent_commits,
)

PEP_RE = re.compile(r"^(?:peps/)?pep-(\d{4})\.(?:txt|rst)$")
ANY_RE = re.compile(r".*")

# Long enough that moving the file and editing one header line still reads as a rename.
BODY = "".join(f"Body line {i} of the proposal text.\n" for i in range(24))


def pep_text(status: str) -> str:
    return f"PEP: 1\nTitle: Example\nStatus: {status}\n\n{BODY}"


def ts(iso: str) -> int:
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())


def utc_day(seconds: int) -> str:
    return datetime.fromtimestamp(seconds, tz=UTC).date().isoformat()


def changes_of(repo: TempRepo, tip: str = "main", path_re: re.Pattern[str] = PEP_RE):
    commits = first_parent_commits(repo.path, tip)
    return commits, file_changes(repo.path, tip, path_re, commits)


INVERTED_DATES = (
    "2020-01-01T12:00:00Z",
    "2020-01-05T12:00:00Z",
    "2020-01-03T12:00:00Z",  # earlier than its parent
    "2020-01-10T12:00:00Z",
)


@pytest.fixture
def inverted(tmp_path):
    """Four commits whose third one is dated before the second (a committer-date inversion)."""
    repo = TempRepo(tmp_path / "repo")
    shas = []
    for i, when in enumerate(INVERTED_DATES):
        repo.write("notes.txt", f"revision {i}\n")
        shas.append(repo.commit(f"commit {i}", when))
    return repo, shas


def test_recorded_time_is_running_max(inverted):
    repo, shas = inverted
    commits = first_parent_commits(repo.path, "main")
    assert [c.sha for c in commits] == shas
    assert [utc_day(c.recorded_ts) for c in commits] == [
        "2020-01-01",
        "2020-01-05",
        "2020-01-05",
        "2020-01-10",
    ]
    assert commits[2].committer_ts == ts("2020-01-03T12:00:00Z")
    assert [c.message for c in commits] == ["", "", "", ""]


def test_as_of_commit_uses_recorded_time(inverted):
    repo, shas = inverted
    commits = first_parent_commits(repo.path, "main")
    assert as_of_commit(commits, date(2020, 1, 4)).sha == shas[0]
    assert as_of_commit(commits, date(2020, 1, 5)).sha == shas[2]
    assert as_of_commit(commits, date(2019, 12, 31)) is None
    assert as_of_commit(commits, date(2030, 1, 1)).sha == shas[3]


def test_as_of_commit_includes_the_last_second_of_the_day(tmp_path):
    repo = TempRepo(tmp_path / "repo")
    repo.write("a.txt", "1\n")
    late = repo.commit("late", "2020-01-05T23:59:59Z")
    repo.write("a.txt", "2\n")
    next_day = repo.commit("next", "2020-01-06T00:00:00Z")
    commits = first_parent_commits(repo.path, "main")
    assert as_of_commit(commits, date(2020, 1, 4)) is None
    assert as_of_commit(commits, date(2020, 1, 5)).sha == late
    assert as_of_commit(commits, date(2020, 1, 6)).sha == next_day


def test_end_of_day_ts():
    assert end_of_day_ts(date(1970, 1, 1)) == 86399
    assert end_of_day_ts(date(1969, 12, 31)) == -1
    assert end_of_day_ts(date(2020, 1, 5)) == ts("2020-01-05T23:59:59Z")
    assert end_of_day_ts(date(2026, 10, 27)) == ts("2026-10-27T23:59:59Z")


def test_file_kinds(tmp_path):
    repo = TempRepo(tmp_path / "repo")
    repo.write("pep-0001.txt", pep_text("Draft"))
    repo.write("README", "not a proposal\n")
    repo.commit("add", "2020-01-01T12:00:00Z")
    repo.write("pep-0001.txt", pep_text("Final"))
    repo.commit("modify", "2020-01-02T12:00:00Z")
    repo.mv("pep-0001.txt", "peps/pep-0001.rst")
    repo.commit("move", "2020-01-03T12:00:00Z")
    repo.rm("peps/pep-0001.rst")
    repo.commit("delete", "2020-01-04T12:00:00Z")

    _, changes = changes_of(repo)
    assert [c.kind for c in changes] == ["A", "M", "R", "D"]
    added, modified, renamed, deleted = changes
    assert (added.path, added.old_path, added.old_blob) == ("pep-0001.txt", None, None)
    assert added.new_blob is not None
    assert modified.old_blob == added.new_blob
    assert modified.new_blob not in (None, added.new_blob)
    assert renamed.old_path == "pep-0001.txt"
    assert renamed.path == "peps/pep-0001.rst"
    assert renamed.old_blob == renamed.new_blob == modified.new_blob
    assert deleted.path == "peps/pep-0001.rst"
    assert deleted.new_blob is None
    assert deleted.old_blob == renamed.new_blob
    assert [c.commit.message for c in changes] == ["add", "modify", "move", "delete"]


def test_symlink_replaced_by_a_file_is_a_modification(tmp_path):
    # PEP 495 was first added as a symlink and 19 minutes later became a regular file (git: T).
    repo = TempRepo(tmp_path / "repo")
    (repo.path / "pep-0495.rst").symlink_to("elsewhere.rst")
    repo.commit("add as a symlink", "2020-01-01T12:00:00Z")
    repo.rm("pep-0495.rst")
    repo.write("pep-0495.rst", pep_text("Draft"))
    repo.commit("make it a file", "2020-01-01T12:19:00Z")

    _, changes = changes_of(repo)
    assert [c.kind for c in changes] == ["A", "M"]
    assert [(c.old_mode, c.new_mode) for c in changes] == [(None, "120000"), ("120000", "100644")]
    with BlobReader(repo.path) as reader:
        assert reader.read(changes[0].new_blob) == "elsewhere.rst"
        assert reader.read(changes[1].new_blob).startswith("PEP: 1\n")
    assert changes[1].old_blob == changes[0].new_blob


def test_modes_of_added_chmodded_renamed_and_deleted_files(tmp_path):
    repo = TempRepo(tmp_path / "repo")
    repo.write("pep-0001.txt", pep_text("Draft"))
    repo.commit("add", "2020-01-01T12:00:00Z")
    os.chmod(repo.path / "pep-0001.txt", 0o755)
    repo.commit("chmod", "2020-01-02T12:00:00Z")
    repo.mv("pep-0001.txt", "peps/pep-0001.rst")
    repo.commit("move", "2020-01-03T12:00:00Z")
    repo.rm("peps/pep-0001.rst")
    repo.commit("delete", "2020-01-04T12:00:00Z")

    _, changes = changes_of(repo)
    assert [(c.kind, c.old_mode, c.new_mode) for c in changes] == [
        ("A", None, "100644"),
        ("M", "100644", "100755"),
        ("R", "100755", "100755"),
        ("D", "100755", None),
    ]
    assert changes[1].old_blob == changes[1].new_blob  # a mode-only change keeps its blob


def test_rename_with_status_edit(tmp_path):
    repo = TempRepo(tmp_path / "repo")
    repo.write("pep-0001.txt", pep_text("Draft"))
    repo.commit("add", "2020-01-01T12:00:00Z")
    repo.rm("pep-0001.txt")
    repo.write("peps/pep-0001.rst", pep_text("Final"))
    moved = repo.commit("move into peps/ and finalise", "2020-01-02T12:00:00Z")

    _, changes = changes_of(repo)
    assert [c.kind for c in changes] == ["A", "R"]
    rename = changes[1]
    assert rename.commit.sha == moved
    assert (rename.old_path, rename.path) == ("pep-0001.txt", "peps/pep-0001.rst")
    assert rename.old_blob != rename.new_blob
    with BlobReader(repo.path) as reader:
        assert "Status: Final" in reader.read(rename.new_blob)
        assert "Status: Draft" in reader.read(rename.old_blob)


def test_rename_detection_does_not_depend_on_repository_config(tmp_path):
    repo = TempRepo(tmp_path / "repo")
    repo.config("diff.renames", "false")
    repo.write("pep-0001.txt", pep_text("Draft"))
    repo.commit("add", "2020-01-01T12:00:00Z")
    repo.mv("pep-0001.txt", "peps/pep-0001.rst")
    repo.commit("move", "2020-01-02T12:00:00Z")

    _, changes = changes_of(repo)
    assert [(c.kind, c.old_path, c.path) for c in changes] == [
        ("A", None, "pep-0001.txt"),
        ("R", "pep-0001.txt", "peps/pep-0001.rst"),
    ]


def numbered_pep(n: int, status: str) -> str:
    return f"PEP: {n}\nTitle: Example\nStatus: {status}\n\n{BODY}"


def test_rename_pairing_does_not_depend_on_the_repository_rename_limit(tmp_path):
    repo = TempRepo(tmp_path / "repo")
    repo.config("diff.renameLimit", "1")
    for n in (1, 2, 3, 4):
        repo.write(f"pep-000{n}.txt", numbered_pep(n, "Draft"))
    repo.commit("add", "2020-01-01T12:00:00Z")
    for n in (1, 2, 3, 4):  # each move also edits the file, so none of them is an exact rename
        repo.rm(f"pep-000{n}.txt")
        repo.write(f"peps/pep-000{n}.rst", numbered_pep(n, "Final"))
    move = repo.commit("move all and finalise", "2020-01-02T12:00:00Z")

    _, changes = changes_of(repo)
    assert [c.kind for c in changes if c.commit.sha == move] == ["R", "R", "R", "R"]


def test_renames_are_paired_when_half_of_the_file_is_kept(tmp_path):
    kept = [f"line {i:02d} " + "abcdefghij" * 4 + "\n" for i in range(10)]

    def kinds_after_moving_with(changed: int, name: str) -> list[str]:
        edited = [
            f"CHGD {i:02d} " + "klmnopqrst" * 4 + "\n" if i < changed else kept[i]
            for i in range(10)
        ]
        repo = TempRepo(tmp_path / name)
        repo.write("a.txt", "".join(kept))
        repo.commit("add", "2020-01-01T12:00:00Z")
        repo.rm("a.txt")
        repo.write("b.txt", "".join(edited))
        move = repo.commit("move and edit", "2020-01-02T12:00:00Z")
        _, changes = changes_of(repo, path_re=ANY_RE)
        return sorted(c.kind for c in changes if c.commit.sha == move)

    assert kinds_after_moving_with(5, "half") == ["R"]  # 50% of the file is kept
    assert kinds_after_moving_with(6, "less") == ["A", "D"]  # 40% is not enough


def test_rename_is_reported_when_either_side_matches(tmp_path):
    repo = TempRepo(tmp_path / "repo")
    eip_re = re.compile(r"^eip-(\d+)\.md$")
    repo.write("drafts/new.md", pep_text("Draft"))
    repo.write("eip-2.md", pep_text("Draft"))
    repo.write("unrelated.txt", "x\n")
    repo.commit("add", "2020-01-01T12:00:00Z")
    repo.mv("drafts/new.md", "eip-1.md")
    repo.mv("eip-2.md", "archive/eip-2.md")
    repo.commit("number one draft, archive another", "2020-01-02T12:00:00Z")

    _, changes = changes_of(repo, path_re=eip_re)
    assert [(c.kind, c.old_path, c.path) for c in changes[:1]] == [("A", None, "eip-2.md")]
    # Both renames are in one commit; only the order between them is up to git.
    assert sorted((c.kind, c.old_path, c.path) for c in changes[1:]) == [
        ("R", "drafts/new.md", "eip-1.md"),
        ("R", "eip-2.md", "archive/eip-2.md"),
    ]
    assert len(changes) == 3  # unrelated.txt is never reported


def test_merge_commit_dated_at_merge(tmp_path):
    repo = TempRepo(tmp_path / "repo")
    repo.write("pep-0001.txt", pep_text("Draft"))
    repo.write("other.txt", "one\n")
    repo.commit("add", "2020-01-01T12:00:00Z")
    repo.branch("side")
    repo.checkout("side")
    repo.write("pep-0001.txt", pep_text("Final"))
    side = repo.commit("finalise on the side branch", "2020-02-01T12:00:00Z")
    repo.checkout("main")
    repo.write("other.txt", "two\n")
    repo.commit("unrelated", "2020-03-01T12:00:00Z")
    merge = repo.merge("side", "merge side", "2020-04-01T12:00:00Z")

    commits, changes = changes_of(repo)
    assert side not in [c.sha for c in commits]
    assert commits[-1].sha == merge
    assert [c.kind for c in changes] == ["A", "M"]
    edit = changes[1]
    assert edit.commit.sha == merge
    assert utc_day(edit.commit.recorded_ts) == "2020-04-01"
    assert all(c.commit.sha != side for c in changes)
    with BlobReader(repo.path) as reader:
        assert "Status: Final" in reader.read(edit.new_blob)


def test_file_added_on_a_side_branch_appears_at_the_merge(tmp_path):
    repo = TempRepo(tmp_path / "repo")
    repo.write("other.txt", "one\n")
    repo.commit("start", "2020-01-01T12:00:00Z")
    repo.branch("side")
    repo.checkout("side")
    repo.write("pep-0002.txt", pep_text("Draft"))
    repo.commit("draft on the side branch", "2020-02-01T12:00:00Z")
    repo.checkout("main")
    repo.write("other.txt", "two\n")
    repo.commit("unrelated", "2020-03-01T12:00:00Z")
    merge = repo.merge("side", "merge side", "2020-04-01T12:00:00Z")

    _, changes = changes_of(repo)
    assert [(c.kind, c.path, c.commit.sha) for c in changes] == [("A", "pep-0002.txt", merge)]


def test_message_is_full_body(tmp_path):
    repo = TempRepo(tmp_path / "repo")
    repo.write("pep-0001.txt", pep_text("Draft"))
    repo.commit("Subject line\n\nBody line", "2020-01-01T12:00:00Z")
    _, changes = changes_of(repo)
    assert changes[0].commit.message == "Subject line\n\nBody line"


def test_author_names_of_the_first_parent_chain(tmp_path):
    accented = "Bj" + chr(0xF6) + "rn Example"  # names are UTF-8, not ASCII
    repo = TempRepo(tmp_path / "repo")
    repo.write("pep-0001.txt", pep_text("Draft"))
    first = repo.commit("add", "2020-01-01T12:00:00Z", author="A. Person <a@example.invalid>")
    repo.branch("side")
    repo.checkout("side")
    repo.write("pep-0001.txt", pep_text("Final"))
    side = repo.commit("on the side", "2020-02-01T12:00:00Z", author="Side <s@example.invalid>")
    repo.checkout("main")
    repo.write("other.txt", "x\n")
    second = repo.commit("unrelated", "2020-03-01T12:00:00Z", author=f"{accented} <b@x.invalid>")
    merge = repo.merge("side", "merge side", "2020-04-01T12:00:00Z")

    names = author_names(repo.path, "main")
    assert names == {first: "A. Person", second: accented, merge: "asofbench tests"}
    assert side not in names  # only the first-parent chain
    assert author_names(repo.path, first) == {first: "A. Person"}


def test_author_names_are_split_at_newlines_only(tmp_path):
    # str.splitlines would cut these names at the line separator, the next-line character and
    # the form feed; git ends each record with a newline and nothing else.
    odd = ["Line" + chr(0x2028) + "Separator", "Next" + chr(0x85) + "Line", "Form\x0cFeed"]
    repo = TempRepo(tmp_path / "repo")
    shas = []
    for i, name in enumerate(odd):
        repo.write("pep-0001.txt", pep_text(f"Draft {i}"))
        shas.append(repo.commit(f"edit {i}", f"2020-0{i + 1}-01T12:00:00Z",
                                author=f"{name} <n{i}@example.invalid>"))
    assert author_names(repo.path, "main") == dict(zip(shas, odd, strict=True))


def test_changes_carry_recorded_time_from_the_commit_list(inverted):
    repo, shas = inverted
    commits, changes = changes_of(repo, path_re=ANY_RE)
    assert [c.commit.sha for c in changes] == shas
    assert [c.commit.recorded_ts for c in changes] == [c.recorded_ts for c in commits]
    assert [c.commit.committer_ts for c in changes] == [c.committer_ts for c in commits]


def test_walk_stops_at_the_tip(tmp_path):
    repo = TempRepo(tmp_path / "repo")
    repo.write("pep-0001.txt", pep_text("Draft"))
    first = repo.commit("add", "2020-01-01T12:00:00Z")
    repo.write("pep-0001.txt", pep_text("Final"))
    repo.commit("modify", "2020-01-02T12:00:00Z")

    commits, changes = changes_of(repo, tip=first)
    assert [c.sha for c in commits] == [first]
    assert [c.kind for c in changes] == ["A"]


def test_file_changes_need_every_commit_of_the_walk(tmp_path):
    repo = TempRepo(tmp_path / "repo")
    repo.write("pep-0001.txt", pep_text("Draft"))
    repo.commit("add", "2020-01-01T12:00:00Z")
    repo.write("pep-0001.txt", pep_text("Final"))
    repo.commit("modify", "2020-01-02T12:00:00Z")

    commits = first_parent_commits(repo.path, "main")
    with pytest.raises(ValueError, match="not in the commit list"):
        file_changes(repo.path, "main", PEP_RE, commits[:1])


def test_unknown_revision_is_an_error(tmp_path):
    repo = TempRepo(tmp_path / "repo")
    repo.write("a.txt", "1\n")
    repo.commit("add", "2020-01-01T12:00:00Z")
    with pytest.raises(RuntimeError, match="no-such-ref"):
        first_parent_commits(repo.path, "no-such-ref")


def test_blob_reader_replaces_bad_bytes(tmp_path):
    repo = TempRepo(tmp_path / "repo")
    repo.write("pep-0001.txt", b"Status: Final\n\xff\xfe\n")
    repo.commit("bad bytes", "2020-01-01T12:00:00Z")
    _, changes = changes_of(repo)
    with BlobReader(repo.path) as reader:
        text = reader.read(changes[0].new_blob)
    assert text.startswith("Status: Final\n")
    assert "\ufffd" in text


def test_blob_reader_keeps_bom_and_line_endings(tmp_path):
    repo = TempRepo(tmp_path / "repo")
    repo.write("pep-0001.txt", b"\xef\xbb\xbfPEP: 1\r\nStatus: Final\r\n\r\nBody\r\n")
    repo.commit("bom and crlf", "2020-01-01T12:00:00Z")
    _, changes = changes_of(repo)
    with BlobReader(repo.path) as reader:
        assert reader.read(changes[0].new_blob) == "\ufeffPEP: 1\r\nStatus: Final\r\n\r\nBody\r\n"


def test_blob_reader_serves_many_blobs_from_one_process(tmp_path):
    repo = TempRepo(tmp_path / "repo")
    big = "x" * 300_000 + "\n"
    repo.write("empty.txt", "")
    repo.write("nul.txt", b"a\x00b\n")
    repo.write("big.txt", big)
    repo.write("multi.txt", "one\ntwo\n\nthree")
    repo.commit("blobs", "2020-01-01T12:00:00Z")
    _, changes = changes_of(repo, path_re=ANY_RE)
    blob = {c.path: c.new_blob for c in changes}
    with BlobReader(repo.path) as reader:
        assert reader.read(blob["big.txt"]) == big
        assert reader.read(blob["empty.txt"]) == ""
        assert reader.read(blob["multi.txt"]) == "one\ntwo\n\nthree"
        assert reader.read(blob["nul.txt"]) == "a\x00b\n"
        assert reader.read(blob["big.txt"]) == big


def test_blob_reader_rejects_what_is_not_a_stored_blob(tmp_path):
    repo = TempRepo(tmp_path / "repo")
    repo.write("pep-0001.txt", pep_text("Draft"))
    sha = repo.commit("add", "2020-01-01T12:00:00Z")
    with BlobReader(repo.path) as reader:
        with pytest.raises(LookupError):
            reader.read("0" * 40)
        with pytest.raises(ValueError, match="object id"):
            reader.read("HEAD:pep-0001.txt")
        with pytest.raises(ValueError, match="not a blob"):
            reader.read(sha)
        # A refused request must not desynchronise the stream for the next one.
        _, changes = changes_of(repo)
        assert reader.read(changes[0].new_blob).startswith("PEP: 1\n")


def test_blob_reader_needs_a_with_block(tmp_path):
    repo = TempRepo(tmp_path / "repo")
    repo.write("a.txt", "1\n")
    repo.commit("add", "2020-01-01T12:00:00Z")
    _, changes = changes_of(repo, path_re=ANY_RE)
    with pytest.raises(RuntimeError, match="with"):
        BlobReader(repo.path).read(changes[0].new_blob)


def test_check_clone_accepts_a_full_clone_and_refuses_the_rest(tmp_path):
    """Section 2: a full, non-shallow clone, read as itself and not as part of another one."""
    repo = TempRepo(tmp_path / "full")
    for i, status in enumerate(("Draft", "Final")):
        repo.write("pep-0001.txt", pep_text(status))
        repo.commit(f"revision {i}", f"2020-0{i + 1}-01T12:00:00Z")
    gitwalk.check_clone(repo.path)
    gitwalk.check_clone(clone(repo.path, tmp_path / "copy"))  # a full clone of a clone is fine
    linked = tmp_path / "linked"
    linked.symlink_to(repo.path, target_is_directory=True)
    gitwalk.check_clone(linked)  # the same clone under another name
    shallow = clone(repo.path, tmp_path / "shallow", "--depth", "1")
    with pytest.raises(gitwalk.CloneError, match="shallow"):
        gitwalk.check_clone(shallow)
    # A directory inside a repository is not a clone: git would read the enclosing repository.
    (repo.path / "peps").mkdir()
    with pytest.raises(gitwalk.CloneError, match="top level"):
        gitwalk.check_clone(repo.path / "peps")
    with pytest.raises(gitwalk.CloneError, match="not a directory"):
        gitwalk.check_clone(tmp_path / "nowhere")
    (tmp_path / "file").write_text("x\n")
    with pytest.raises(gitwalk.CloneError, match="not a directory"):
        gitwalk.check_clone(tmp_path / "file")


def test_check_clone_refuses_a_partial_clone_and_the_walk_never_fetches(tmp_path):
    """A blob-filtered clone is not shallow, but it lacks file contents and git would fetch them
    from the remote while the walk reads them: a network call and a write under a read-only walk."""
    repo = TempRepo(tmp_path / "full")
    repo.write("pep-0001.txt", pep_text("Draft"))
    repo.commit("revision 0", "2020-01-01T12:00:00Z")
    repo._git("config", "uploadpack.allowFilter", "true")  # let the clone below filter blobs
    partial = clone(repo.path, tmp_path / "partial", "--filter=blob:none", "--no-checkout")
    with pytest.raises(gitwalk.CloneError, match="partial clone"):
        gitwalk.check_clone(partial)
    assert gitwalk._env()["GIT_NO_LAZY_FETCH"] == "1"


def test_the_tree_reader_reads_a_directory_at_a_revision(tmp_path):
    """``files_at`` gives the regular files directly in a directory at HEAD or at a commit, by
    name, as their bytes: a symbolic link is not one of them. ``files_at_head`` reads HEAD, as its
    callers expect, and a revision that is neither HEAD nor a full object id is refused."""
    repo = TempRepo(tmp_path / "repo")
    repo.write("src/a.py", "x = 1\n")
    repo.write("src/notes.txt", "notes\n")
    repo.write("src/deeper/b.py", "y = 1\n")
    (repo.path / "src" / "link.py").symlink_to("a.py")
    first = repo.commit("first", "2020-01-01T12:00:00Z")
    repo.write("src/a.py", "x = 2\n")
    repo.commit("second", "2020-01-02T12:00:00Z")
    assert gitwalk.files_at(repo.path, "src", first) == {"a.py": b"x = 1\n",
                                                          "notes.txt": b"notes\n"}
    assert gitwalk.files_at(repo.path, "src") == gitwalk.files_at_head(repo.path, "src") == {
        "a.py": b"x = 2\n", "notes.txt": b"notes\n"}
    assert gitwalk.files_at(repo.path, "src", "1" * 40) == {}
    assert gitwalk.files_at(repo.path, "elsewhere", first) == {}
    for revision in ("--output=x", "main", first[:7]):
        with pytest.raises(ValueError, match="neither HEAD nor a full object id"):
            gitwalk.files_at(repo.path, "src", revision)


def test_only_read_only_git_commands_are_allowed(tmp_path):
    for command in ("commit", "checkout", "add", "gc", "config", "update-ref", "fetch", "clone"):
        with pytest.raises(ValueError, match="read-only"):
            gitwalk._command(tmp_path, command)
    for command in ("log", "rev-list", "cat-file", "show", "rev-parse"):
        assert gitwalk._command(tmp_path, command, "x")[:4] == ["git", "-C", str(tmp_path), command]


def test_inherited_git_variables_do_not_redirect_the_walk(tmp_path, monkeypatch):
    mine = TempRepo(tmp_path / "mine")
    mine.write("pep-0001.txt", pep_text("Draft"))
    sha = mine.commit("mine", "2020-01-01T12:00:00Z")
    other = TempRepo(tmp_path / "other")
    other.write("notes.txt", "other\n")
    other.commit("other", "2021-01-01T12:00:00Z")

    monkeypatch.setenv("GIT_DIR", str(other.path / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(other.path))
    monkeypatch.setenv("GIT_INDEX_FILE", str(other.path / ".git" / "index"))
    commits = first_parent_commits(mine.path, "main")
    assert [c.sha for c in commits] == [sha]
    changes = file_changes(mine.path, "main", PEP_RE, commits)
    assert [c.path for c in changes] == ["pep-0001.txt"]
    with BlobReader(mine.path) as reader:
        assert reader.read(changes[0].new_blob).startswith("PEP: 1\n")


def snapshot(root):
    return sorted(
        (str(p.relative_to(root)), p.stat().st_mtime_ns, p.stat().st_size)
        for p in root.rglob("*")
        if p.is_file()
    )


def test_the_walk_leaves_the_repository_untouched(tmp_path):
    repo = TempRepo(tmp_path / "repo")
    repo.write("pep-0001.txt", pep_text("Draft"))
    repo.commit("add", "2020-01-01T12:00:00Z")
    repo.branch("side")
    repo.checkout("side")
    repo.write("pep-0001.txt", pep_text("Final"))
    repo.commit("edit on the side", "2020-02-01T12:00:00Z")
    repo.checkout("main")
    repo.write("other.txt", "x\n")
    repo.commit("unrelated", "2020-03-01T12:00:00Z")
    repo.merge("side", "merge side", "2020-04-01T12:00:00Z")

    before = snapshot(repo.path)
    commits, changes = changes_of(repo, path_re=ANY_RE)
    as_of_commit(commits, date(2020, 3, 15))
    with BlobReader(repo.path) as reader:
        for change in changes:
            if change.new_blob:
                reader.read(change.new_blob)
    assert snapshot(repo.path) == before
