"""The episode template that every store system ingests (common/asofbench_systems/stream.py):
each kind of episode renders to the one exact text, the same text on every call, and a long text
renders whole. The streams themselves are section 8's; these episodes are synthetic."""

from __future__ import annotations

import json

import pytest
from asofbench_systems import stream

HEAD = ('PEP 9999 ("A Toy Proposal"): {kind} recorded at 2001-02-03T04:05:06Z in the peps '
        "repository, commit 0123456789abcdef0123456789abcdef01234567.\n"
        "\n"
        "Commit message:\n"
        "Toy commit\n"
        "\n")
EMPTY = "(empty: the change touched only lines this arm withholds, or the file's mode)"
DIFF = "@@ -1,3 +1,3 @@\n PEP: 9999\n-Status: Draft\n+Status: Final\n Title: A Toy Proposal\n"


def episode(kind: str, text: str, message: str = "Toy commit") -> dict:
    return {"episode_id": "0123456789ab:PEP-9999", "corpus": "pep", "repo": "peps",
            "proposal": "PEP-9999", "ref": 'PEP 9999 ("A Toy Proposal")',
            "recorded_at": "2001-02-03T04:05:06Z",
            "commit": "0123456789abcdef0123456789abcdef01234567", "message": message,
            "kind": kind, "text": text}


def test_a_create_renders_the_full_text_of_the_file():
    text = "PEP: 9999\nTitle: A Toy Proposal\nStatus: Draft\n\nBody.\n"
    assert stream.render_episode(episode("create", text)) == (
        HEAD.format(kind="create") + "Full text of the file:\n" + text)


def test_a_modify_renders_its_diff():
    assert stream.render_episode(episode("modify", DIFF)) == (
        HEAD.format(kind="modify") + "Unified diff of the file:\n" + DIFF)


def test_a_rename_renders_its_diff_and_says_where_the_file_came_from():
    assert stream.render_episode(episode("rename", DIFF)) == (
        HEAD.format(kind="rename")
        + "Unified diff of the file, which moved here from another repository or path:\n" + DIFF)


@pytest.mark.parametrize("kind,section", [
    ("modify", "Unified diff of the file"),
    ("rename", "Unified diff of the file, which moved here from another repository or path"),
])
def test_an_empty_diff_renders_the_section_line_and_the_empty_marker(kind, section):
    assert stream.render_episode(episode(kind, "")) == (
        HEAD.format(kind=kind) + f"{section}:\n" + EMPTY)


def test_a_delete_has_no_section_and_ends_with_the_deleted_line():
    rendered = stream.render_episode(episode("delete", ""))
    assert rendered == HEAD.format(kind="delete") + "The file was deleted."
    assert "Full text" not in rendered and "Unified diff" not in rendered


def test_a_message_of_several_lines_is_kept_as_it_is():
    message = "PEP 9999: mark as final\n\nResolution: a toy thread.\n  indented line"
    rendered = stream.render_episode(episode("modify", DIFF, message=message))
    assert "\nCommit message:\n" + message + "\n\nUnified diff of the file:\n" in rendered


def test_rendering_is_byte_identical_across_calls():
    for kind, text in [("create", "PEP: 9999\nÉtat: brouillon ✓\n"), ("modify", DIFF),
                       ("modify", ""), ("rename", ""), ("rename", DIFF), ("delete", "")]:
        e = episode(kind, text)
        first = stream.render_episode(e).encode("utf-8")
        second = stream.render_episode(json.loads(json.dumps(e))).encode("utf-8")
        assert first == second


def test_a_text_of_300000_characters_renders_whole():
    text = "".join(f"line {n:06d} of a long proposal\n" for n in range(10_000))
    text = (text * 2)[:300_000]
    assert len(text) == 300_000
    rendered = stream.render_episode(episode("create", text))
    assert rendered == HEAD.format(kind="create") + "Full text of the file:\n" + text
    assert rendered.endswith(text)


def test_an_unknown_kind_or_a_delete_with_text_is_refused():
    with pytest.raises(ValueError, match="kind"):
        stream.render_episode(episode("copy", DIFF))
    with pytest.raises(ValueError, match="delete"):
        stream.render_episode(episode("delete", "left over\n"))


def test_read_episodes_yields_the_json_lines_in_file_order(tmp_path):
    episodes = [episode("create", "PEP: 9999\n"), episode("modify", DIFF),
                {**episode("delete", ""), "episode_id": "fedcba987654:PEP-9999"}]
    path = tmp_path / "h-plus.jsonl"
    path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in episodes),
                    encoding="utf-8")
    assert list(stream.read_episodes(str(path))) == episodes


def test_read_episodes_keeps_line_separators_inside_strings_and_refuses_a_bad_line(tmp_path):
    # U+2028 and U+2029 may stand unescaped in JSON strings; a line ends at a newline only.
    e = episode("create", "first second third\r\n")
    path = tmp_path / "one.jsonl"
    path.write_bytes((json.dumps(e, ensure_ascii=False) + "\n").encode("utf-8"))
    assert list(stream.read_episodes(str(path))) == [e]
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps(e) + "\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        list(stream.read_episodes(str(bad)))
    bad.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        list(stream.read_episodes(str(bad)))
