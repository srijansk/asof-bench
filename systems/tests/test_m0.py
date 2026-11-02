"""M0, the Graphiti cost pilot of PREREGISTRATION.md section 13 (graphiti/m0.py): ``prepare``,
which selects the H+ episodes of the pilot's proposals, and ``project``, which applies the measured
rates to every ingestion and decides between ``graphiti`` and ``graphiti-reduced``. Synthetic files
only: no file of data/v0.1 and no stream is read."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest
from graphiti import m0

BASIS = ("claude-haiku-5-5 list prices on 2026-10-31 (Claude API, prompts up to 100,000 tokens): "
         "$0.10 per million input tokens, $0.50 per million output tokens")
MARKER = "TOY-PROPOSAL-TEXT"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def episode(n: int, proposal: str, text: str, corpus: str = "pep") -> dict:
    commit = f"{n:040x}"
    return {"episode_id": f"{commit[:12]}:{proposal}", "corpus": corpus,
            "repo": "peps" if corpus == "pep" else "EIPs", "proposal": proposal,
            "ref": f'{proposal} ("Toy")', "recorded_at": f"2001-01-{n:02d}T00:00:00Z",
            "commit": commit, "message": f"{MARKER} message {n}", "kind": "modify", "text": text}


def write_lines(path: Path, items: list[dict]) -> Path:
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items),
                    encoding="utf-8")
    return path


def stream_stats(path: Path) -> dict:
    # A line ends at a newline only: str.splitlines would also split at U+2028 inside a string.
    episodes = [json.loads(line) for line in path.read_bytes().split(b"\n") if line]
    return {"chars": sum(len(e["text"]) for e in episodes), "episodes": len(episodes),
            "sha256": sha256(path)}


def write_manifest(path: Path, streams: dict[str, Path], files: dict[str, Path],
                   basis: str = BASIS, budget: float = 500.0) -> Path:
    path.write_text(json.dumps({
        "graphiti_budget_basis": basis, "graphiti_budget_usd": budget,
        "files": {name: {"sha256": sha256(p)} for name, p in files.items()},
        "streams": {arm: stream_stats(p) for arm, p in streams.items()}}), encoding="utf-8")
    return path


class Prepared:
    """A synthetic prepare: an H+ stream, a pilot.json and a manifest that registers both."""

    def __init__(self, tmp_path: Path, episodes: list[dict], proposals: list[str]) -> None:
        self.stream = write_lines(tmp_path / "h-plus.jsonl", episodes)
        self.pilot = tmp_path / "pilot.json"
        self.pilot.write_text(json.dumps({"m0_graphiti_proposals": proposals}), encoding="utf-8")
        self.manifest = write_manifest(tmp_path / "MANIFEST.json", {"H+": self.stream},
                                       {"pilot.json": self.pilot})
        self.out = tmp_path / "m0"

    def run(self) -> int:
        return m0.main(["prepare", "--pilot", str(self.pilot), "--stream", str(self.stream),
                        "--manifest", str(self.manifest), "--out-dir", str(self.out)])


def test_prepare_selects_the_pilot_s_proposals_in_stream_order_and_counts_code_points(tmp_path):
    episodes = [episode(1, "PEP-1", "é😀\n"), episode(2, "PEP-2", "plain\n"),
                episode(3, "PEP-3", "line separator\n"), episode(4, "EIP-1", "x\n", "eth"),
                episode(5, "PEP-1", ""), episode(6, "PEP-3", "a" * 1000)]
    prepared = Prepared(tmp_path, episodes, ["PEP-3", "PEP-1"])
    assert prepared.run() == 0
    lines = prepared.stream.read_bytes().splitlines(keepends=True)
    written = (prepared.out / "m0-h-plus.jsonl").read_bytes()
    assert written == b"".join(lines[i] for i in (0, 2, 4, 5))
    result = json.loads((prepared.out / "prepare.json").read_text(encoding="utf-8"))
    assert result == {"proposals": 2, "episodes": 4,
                      "text_chars": 3 + len("line separator\n") + 0 + 1000,
                      "stream_sha256": sha256(prepared.stream),
                      "input_sha256": hashlib.sha256(written).hexdigest()}
    # é, 😀 and the newline are three code points (eight UTF-8 bytes); U+2028 is one.
    assert result["text_chars"] == 3 + 15 + 0 + 1000
    assert str(tmp_path) not in (prepared.out / "prepare.json").read_text(encoding="utf-8")


def test_prepare_refuses_a_stream_that_is_not_the_registered_one(tmp_path, capsys):
    prepared = Prepared(tmp_path, [episode(1, "PEP-1", "text\n")], ["PEP-1"])
    prepared.stream.write_text(prepared.stream.read_text(encoding="utf-8").replace("text", "edit"),
                               encoding="utf-8")
    assert prepared.run() == 2
    assert "SHA-256" in capsys.readouterr().err
    assert not (prepared.out / "m0-h-plus.jsonl").exists()
    assert not (prepared.out / "prepare.json").exists()


def test_prepare_refuses_a_pilot_json_that_is_not_the_registered_one(tmp_path, capsys):
    prepared = Prepared(tmp_path, [episode(1, "PEP-1", "text\n"), episode(2, "PEP-2", "t\n")],
                        ["PEP-1"])
    prepared.pilot.write_text(json.dumps({"m0_graphiti_proposals": ["PEP-2"]}), encoding="utf-8")
    assert prepared.run() == 2
    assert "pilot.json is not the registered pilot.json" in capsys.readouterr().err
    assert not prepared.out.exists()


def test_prepare_refuses_a_proposal_with_no_episode(tmp_path, capsys):
    prepared = Prepared(tmp_path, [episode(1, "PEP-1", "text\n")], ["PEP-1", "PEP-404"])
    assert prepared.run() == 2
    assert "PEP-404" in capsys.readouterr().err


def test_prepare_refuses_an_output_directory_inside_the_checkout(tmp_path, monkeypatch, capsys):
    prepared = Prepared(tmp_path, [episode(1, "PEP-1", "text\n")], ["PEP-1"])
    checkout = tmp_path / "checkout"
    monkeypatch.setattr(m0, "REPOSITORY", checkout.resolve())
    prepared.out = checkout / "runs" / "m0"
    assert prepared.run() == 2
    assert "inside the repository" in capsys.readouterr().err
    assert not (checkout / "runs").exists()


def order(candidates: list[str]) -> list[str]:
    """The candidates in the registered order, computed here independently of m0.py."""
    return sorted(candidates, key=lambda c: hashlib.sha256(
        f"asof-bench/v0.1|distractor|{c}".encode()).hexdigest())


class World:
    """A synthetic M0: three local streams over named proposals and candidates, a manifest that
    registers them and questions.jsonl, the questions, a summary and prepare.json. ``sizes``
    gives each proposal's characters in H+, H- and CF."""

    def __init__(self, tmp_path: Path, sizes: dict[str, tuple[int, int, int]],
                 named: list[str], basis: str = BASIS, budget: float = 500.0,
                 summary: dict | None = None) -> None:
        self.dir = tmp_path
        streams_dir = tmp_path / "streams"
        streams_dir.mkdir(parents=True)
        self.streams = {}
        for index, (arm, name) in enumerate((("H+", "h-plus.jsonl"), ("H-", "h-minus.jsonl"),
                                             ("CF", "cf.jsonl"))):
            items = [episode(n, proposal, MARKER + "x" * (chars[index] - len(MARKER)))
                     for n, (proposal, chars) in enumerate(sizes.items(), 1)]
            self.streams[arm] = write_lines(streams_dir / name, items)
        self.streams_dir = streams_dir
        questions = [{"id": f"F1-change-{p}", "family": "F1", "proposal": p} for p in named[:1]]
        questions += [{"id": f"F2-{p}", "family": "F2", "proposal": p} for p in named[1:2]]
        questions += [{"id": f"CF-{p}", "family": "CF", "proposal": p} for p in named[2:]]
        # A family whose proposals are not named, and an item that names none.
        others = [p for p in sizes if p not in named]
        questions += [{"id": "F3-toy", "family": "F3", "proposal": others[0]},
                      {"id": "F5-toy", "family": "F5", "proposal": None}]
        self.questions = write_lines(tmp_path / "questions.jsonl", questions)
        self.manifest = write_manifest(tmp_path / "MANIFEST.json", self.streams,
                                       {"questions.jsonl": self.questions}, basis, budget)
        input_sha256 = "1" * 64
        self.summary = tmp_path / "summary.json"
        self.summary.write_text(json.dumps(summary or {
            "arm": "H+", "stream": "m0-h-plus.jsonl", "stream_sha256": input_sha256,
            "episodes": 4, "ok": 4, "failed": 0, "text_chars": 1000,
            "input_tokens": 10_000_000, "output_tokens": 2_000_000, "calls": 40,
            "api_errors": 0, "calls_over_100k_input": 0,
            "removal": {"input_tokens": 0, "output_tokens": 0, "calls": 0, "api_errors": 0},
            "wall_s": 120.0, "elapsed_s": 70.0, "exit": 0, "graphiti_core": "0.31.2",
            "model": "claude-haiku-5-5"}), encoding="utf-8")
        self.prepare = tmp_path / "prepare.json"
        self.prepare.write_text(json.dumps({
            "proposals": 2, "episodes": 4, "text_chars": 1000,
            "stream_sha256": sha256(self.streams["H+"]), "input_sha256": input_sha256}),
            encoding="utf-8")
        self.out = tmp_path / "projection.json"

    def project(self) -> int:
        return m0.main(["project", "--summary", str(self.summary), "--prepare", str(self.prepare),
                        "--manifest", str(self.manifest), "--questions", str(self.questions),
                        "--streams-dir", str(self.streams_dir), "--out", str(self.out)])

    def projection(self) -> dict:
        return json.loads(self.out.read_text(encoding="utf-8"))

    def edit(self, path: Path, **changes) -> None:
        path.write_text(json.dumps({**json.loads(path.read_text(encoding="utf-8")), **changes}),
                        encoding="utf-8")


# With 10,000 input and 2,000 output tokens per character, each character of an ingestion costs
# 10,000 x $0.10/M + 2,000 x $0.50/M = $0.002.
SMALL = {"PEP-1": (10_000, 9_000, 10_000), "PEP-2": (5_000, 5_000, 6_000),
         "PEP-7": (8_000, 8_000, 8_000), "EIP-9": (7_000, 7_000, 7_000)}


def test_project_reproduces_the_hand_computed_rates_tokens_and_cost(tmp_path):
    world = World(tmp_path, SMALL, named=["PEP-1", "PEP-2"])
    assert world.project() == 0
    projection = world.projection()
    assert projection["rates"]["in_per_char"] == 10_000
    assert projection["rates"]["out_per_char"] == 2_000
    chars = {"H+": 30_000, "H-": 29_000, "CF": 31_000}
    rows = {row["ingestion"]: row for row in projection["ingestions"]}
    assert list(rows) == ["H+", "H-", "CF", "incremental"]
    for name, stream in (("H+", "H+"), ("H-", "H-"), ("CF", "CF"), ("incremental", "H+")):
        row = rows[name]
        assert (row["stream"], row["chars"]) == (stream, chars[stream])
        assert row["input_tokens"] == pytest.approx(10_000 * chars[stream])
        assert row["output_tokens"] == pytest.approx(2_000 * chars[stream])
        assert row["cost_usd"] == pytest.approx(0.002 * chars[stream])
    assert projection["total"]["chars"] == 30_000 * 2 + 29_000 + 31_000
    assert projection["total"]["cost_usd"] == pytest.approx(0.002 * 120_000)
    assert projection["prices_usd_per_million_tokens"] == {"input": 0.1, "output": 0.5}
    assert (projection["budget_usd"], projection["price_basis"]) == (500.0, BASIS)


def test_project_returns_graphiti_when_the_total_fits(tmp_path):
    world = World(tmp_path, SMALL, named=["PEP-1", "PEP-2"])
    assert world.project() == 0
    projection = world.projection()
    assert projection["total"]["cost_usd"] == pytest.approx(240.0)
    assert (projection["decision"], projection["k"], projection["reduced"]) == (
        "graphiti", None, None)
    assert "reason" not in projection


def test_project_returns_graphiti_reduced_with_the_largest_k_that_fits(tmp_path):
    candidates = order(["PEP-11", "PEP-12", "EIP-13", "ERC-14", "PEP-15"])
    sizes = {"PEP-1": (10_000, 9_000, 10_000), "PEP-2": (5_000, 5_000, 6_000)}
    # In the registered order the candidates cost $160, $40, $40, $240 and $8 (each character
    # $0.008 over the four ingestions), and the named proposals $120: $120 + $160 + $40 + $40 =
    # $360 fits $500 and the fourth candidate does not, so k is 3, though the fifth would fit.
    for candidate, chars in zip(candidates, (20_000, 5_000, 5_000, 30_000, 1_000), strict=True):
        sizes[candidate] = (chars, chars, chars)
    world = World(tmp_path, sizes, named=["PEP-1", "PEP-2"])
    assert world.project() == 0
    projection = world.projection()
    assert projection["total"]["cost_usd"] == pytest.approx(0.002 * (76_000 * 2 + 75_000 + 77_000))
    assert projection["decision"] == "graphiti-reduced"
    assert projection["k"] == 3
    reduced = projection["reduced"]
    assert (reduced["named_proposals"], reduced["candidates"], reduced["k"]) == (2, 5, 3)
    assert reduced["distractors"] == candidates[:3]
    rows = {row["ingestion"]: row for row in reduced["ingestions"]}
    assert rows["H+"]["chars"] == 15_000 + 30_000 and rows["H-"]["chars"] == 14_000 + 30_000
    assert rows["CF"]["chars"] == 16_000 + 30_000 and rows["incremental"]["chars"] == 45_000
    assert reduced["total"]["cost_usd"] == pytest.approx(360.0)
    assert reduced["next_candidate"] == {"id": candidates[3],
                                         "total_cost_usd": pytest.approx(600.0)}


def test_project_writes_the_projection_and_says_why_when_no_k_fits(tmp_path, capsys):
    candidates = order(["PEP-11", "PEP-12"])
    # The named proposal alone costs $800 (100,000 characters in each ingestion at $0.002).
    sizes = {"PEP-1": (100_000, 100_000, 100_000), candidates[0]: (1_000, 1_000, 1_000),
             candidates[1]: (1_000, 1_000, 1_000)}
    world = World(tmp_path, sizes, named=["PEP-1"])
    assert world.project() == 1
    projection = world.projection()
    assert (projection["decision"], projection["k"]) == ("graphiti-reduced", None)
    assert projection["reason"] == m0.NO_K
    assert "does not provide" in capsys.readouterr().err
    assert projection["rates"]["in_per_char"] == 10_000
    assert projection["total"]["cost_usd"] == pytest.approx(0.002 * 4 * 102_000)
    reduced = projection["reduced"]
    assert (reduced["k"], reduced["distractors"]) == (None, [])
    assert reduced["total"]["cost_usd"] == pytest.approx(800.0)


def test_project_refuses_a_price_basis_that_does_not_state_both_prices(tmp_path, capsys):
    for index, basis in enumerate(("claude-haiku-5-5 list prices: $0.10 per million input tokens",
                                   "claude-haiku-5-5 list prices: $0.50 per million output tokens",
                                   "claude-haiku-5-5: $0.10 / MTok input, $0.50 / MTok output",
                                   "$0.10 per million input tokens, $0.20 per million input "
                                   "tokens, $0.50 per million output tokens")):
        world = World(tmp_path / str(index), SMALL, named=["PEP-1"], basis=basis)
        assert world.project() == 2
        assert not world.out.exists()
        assert "price basis" in capsys.readouterr().err


def test_project_refuses_inputs_that_are_not_the_registered_ones(tmp_path, capsys):
    world = World(tmp_path / "streams", SMALL, named=["PEP-1"])
    world.streams["CF"].write_text(world.streams["CF"].read_text(encoding="utf-8") + "\n",
                                   encoding="utf-8")
    assert world.project() == 2
    assert "cf.jsonl is not the registered CF stream" in capsys.readouterr().err

    world = World(tmp_path / "questions", SMALL, named=["PEP-1"])
    world.questions.write_text(world.questions.read_text(encoding="utf-8") + "\n",
                               encoding="utf-8")
    assert world.project() == 2
    assert "questions.jsonl is not the registered questions.jsonl" in capsys.readouterr().err

    world = World(tmp_path / "prepare", SMALL, named=["PEP-1"])
    world.edit(world.prepare, stream_sha256="2" * 64)
    assert world.project() == 2
    assert "not prepared from the registered H+ stream" in capsys.readouterr().err
    assert not world.out.exists()


def test_project_refuses_a_summary_that_is_not_the_prepared_run_of_the_model(tmp_path, capsys):
    cases = [({"stream_sha256": "3" * 64}, "is not the run of the input"),
             ({"episodes": 5}, "counts 5 episodes"),
             ({"model": ["claude-haiku-5-5", "claude-sonnet-5-5"]}, "whose every call"),
             ({"model": "claude-sonnet-5-5"}, "whose every call"),
             ({"arm": "H-"}, "is not an H+ run")]
    for index, (changes, message) in enumerate(cases):
        world = World(tmp_path / str(index), SMALL, named=["PEP-1"])
        world.edit(world.summary, **changes)
        assert world.project() == 2
        assert message in capsys.readouterr().err
        assert not world.out.exists()


def test_project_writes_no_text_field_no_proposal_text_and_no_local_path(tmp_path):
    candidates = order(["PEP-11", "PEP-12"])
    # The named proposal costs $160 and each candidate $240: the total, $640, binds, and k is 1.
    sizes = {"PEP-1": (20_000, 20_000, 20_000), candidates[0]: (30_000, 30_000, 30_000),
             candidates[1]: (30_000, 30_000, 30_000)}
    world = World(tmp_path, sizes, named=["PEP-1"])
    assert world.project() == 0
    assert (world.projection()["decision"], world.projection()["k"]) == ("graphiti-reduced", 1)
    raw = world.out.read_text(encoding="utf-8")
    assert MARKER not in raw and "xxxxxxxxxx" not in raw
    assert str(tmp_path) not in raw

    def keys(value):
        if isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from keys(item)
        elif isinstance(value, list):
            for item in value:
                yield from keys(item)

    found = set(keys(json.loads(raw)))
    assert "text" not in found and "message" not in found
    projection = json.loads(raw)
    assert set(projection["inputs"]) == {"summary", "prepare", "manifest", "questions", "H+",
                                         "H-", "CF"}
    assert projection["inputs"]["H-"]["sha256"] == sha256(world.streams["H-"])
    assert projection["inputs"]["questions"]["sha256"] == sha256(world.questions)
    assert projection["command"] == ["m0.py", "project", "--summary", "summary.json",
                                     "--prepare", "prepare.json", "--manifest", "MANIFEST.json",
                                     "--questions", "questions.jsonl", "--streams-dir",
                                     "streams", "--out", "projection.json"]


def test_the_price_basis_parses_to_exact_prices():
    assert m0.parse_prices(BASIS) == (Fraction(1, 10), Fraction(1, 2))


def test_project_refuses_a_run_that_did_not_reach_the_stream_s_end(tmp_path, capsys):
    # The summary of a stopped run has the whole stream's characters but some episodes' tokens.
    cases = [({"exit": 2}, "its last invocation exited with status 2"),
             ({"exit": 3}, "its last invocation exited with status 3"),
             ({"exit": None}, "its last invocation exited with status None"),
             ({"ok": 2, "failed": 1}, "its log covers 3 of its 4 episodes")]
    for index, (changes, message) in enumerate(cases):
        world = World(tmp_path / str(index), SMALL, named=["PEP-1"])
        world.edit(world.summary, **changes)
        assert world.project() == 2
        assert message in capsys.readouterr().err
        assert not world.out.exists()
    # A finished run with an episode logged as failed (exit status 1) is projected.
    world = World(tmp_path / "failed", SMALL, named=["PEP-1"])
    world.edit(world.summary, exit=1, ok=3, failed=1)
    assert world.project() == 0


def test_project_refuses_a_run_in_which_a_call_failed(tmp_path, capsys):
    world = World(tmp_path, SMALL, named=["PEP-1"])
    world.edit(world.summary, api_errors=2)
    assert world.project() == 2
    assert "some calls failed inside Graphiti, so the rates would be understated" in (
        capsys.readouterr().err)
    assert not world.out.exists()


def test_the_rates_count_the_ingestion_attempts_and_not_the_removal_calls(tmp_path):
    world = World(tmp_path, SMALL, named=["PEP-1", "PEP-2"])
    world.edit(world.summary, removal={"input_tokens": 9_000_000, "output_tokens": 900_000,
                                       "calls": 30, "api_errors": 0})
    assert world.project() == 0
    projection = world.projection()
    assert (projection["rates"]["in_per_char"], projection["rates"]["out_per_char"]) == (
        10_000, 2_000)
    assert projection["pilot"]["removal"]["input_tokens"] == 9_000_000
    assert projection["total"]["cost_usd"] == pytest.approx(240.0)
