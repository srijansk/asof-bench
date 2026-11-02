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


def write_manifest(path: Path, streams: dict[str, Path], basis: str = BASIS,
                   budget: float = 500.0) -> Path:
    path.write_text(json.dumps({"graphiti_budget_basis": basis, "graphiti_budget_usd": budget,
                                "streams": {arm: stream_stats(p) for arm, p in streams.items()}}),
                    encoding="utf-8")
    return path


def test_prepare_selects_the_pilot_s_proposals_in_stream_order_and_counts_code_points(tmp_path):
    episodes = [episode(1, "PEP-1", "é😀\n"), episode(2, "PEP-2", "plain\n"),
                episode(3, "PEP-3", "line separator\n"), episode(4, "EIP-1", "x\n", "eth"),
                episode(5, "PEP-1", ""), episode(6, "PEP-3", "a" * 1000)]
    stream = write_lines(tmp_path / "h-plus.jsonl", episodes)
    manifest = write_manifest(tmp_path / "MANIFEST.json", {"H+": stream})
    pilot = tmp_path / "pilot.json"
    pilot.write_text(json.dumps({"m0_graphiti_proposals": ["PEP-3", "PEP-1"]}), encoding="utf-8")
    out = tmp_path / "m0"
    assert m0.main(["prepare", "--pilot", str(pilot), "--stream", str(stream), "--manifest",
                    str(manifest), "--out-dir", str(out)]) == 0
    lines = stream.read_bytes().splitlines(keepends=True)
    assert (out / "m0-h-plus.jsonl").read_bytes() == b"".join(lines[i] for i in (0, 2, 4, 5))
    prepared = json.loads((out / "prepare.json").read_text(encoding="utf-8"))
    assert prepared == {"proposals": 2, "episodes": 4,
                        "text_chars": 3 + len("line separator\n") + 0 + 1000,
                        "stream_sha256": sha256(stream)}
    # é, 😀 and the newline are three code points (eight UTF-8 bytes); U+2028 is one.
    assert prepared["text_chars"] == 3 + 15 + 0 + 1000


def test_prepare_refuses_a_stream_that_is_not_the_registered_one(tmp_path, capsys):
    stream = write_lines(tmp_path / "h-plus.jsonl", [episode(1, "PEP-1", "text\n")])
    manifest = write_manifest(tmp_path / "MANIFEST.json", {"H+": stream})
    stream.write_text(stream.read_text(encoding="utf-8").replace("text", "edit"), encoding="utf-8")
    pilot = tmp_path / "pilot.json"
    pilot.write_text(json.dumps({"m0_graphiti_proposals": ["PEP-1"]}), encoding="utf-8")
    out = tmp_path / "m0"
    assert m0.main(["prepare", "--pilot", str(pilot), "--stream", str(stream), "--manifest",
                    str(manifest), "--out-dir", str(out)]) == 2
    assert "SHA-256" in capsys.readouterr().err
    assert not (out / "m0-h-plus.jsonl").exists() and not (out / "prepare.json").exists()


def test_prepare_refuses_a_proposal_with_no_episode(tmp_path, capsys):
    stream = write_lines(tmp_path / "h-plus.jsonl", [episode(1, "PEP-1", "text\n")])
    manifest = write_manifest(tmp_path / "MANIFEST.json", {"H+": stream})
    pilot = tmp_path / "pilot.json"
    pilot.write_text(json.dumps({"m0_graphiti_proposals": ["PEP-1", "PEP-404"]}),
                     encoding="utf-8")
    assert m0.main(["prepare", "--pilot", str(pilot), "--stream", str(stream), "--manifest",
                    str(manifest), "--out-dir", str(tmp_path / "m0")]) == 2
    assert "PEP-404" in capsys.readouterr().err


def test_prepare_refuses_an_output_directory_inside_the_checkout(tmp_path, monkeypatch, capsys):
    stream = write_lines(tmp_path / "h-plus.jsonl", [episode(1, "PEP-1", "text\n")])
    manifest = write_manifest(tmp_path / "MANIFEST.json", {"H+": stream})
    pilot = tmp_path / "pilot.json"
    pilot.write_text(json.dumps({"m0_graphiti_proposals": ["PEP-1"]}), encoding="utf-8")
    checkout = tmp_path / "checkout"
    monkeypatch.setattr(m0, "REPOSITORY", checkout.resolve())
    assert m0.main(["prepare", "--pilot", str(pilot), "--stream", str(stream), "--manifest",
                    str(manifest), "--out-dir", str(checkout / "runs" / "m0")]) == 2
    assert "inside the repository" in capsys.readouterr().err
    assert not (checkout / "runs").exists()


def order(candidates: list[str]) -> list[str]:
    """The candidates in the registered order, computed here independently of m0.py."""
    return sorted(candidates, key=lambda c: hashlib.sha256(
        f"asof-bench/v0.1|distractor|{c}".encode()).hexdigest())


class World:
    """A synthetic M0: three local streams over named proposals and candidates, a manifest that
    registers them, the questions, a summary and prepare.json. ``sizes`` gives each proposal's
    characters in H+, H- and CF."""

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
        self.manifest = write_manifest(tmp_path / "MANIFEST.json", self.streams, basis, budget)
        questions = [{"id": f"F1-change-{p}", "family": "F1", "proposal": p} for p in named[:1]]
        questions += [{"id": f"F2-{p}", "family": "F2", "proposal": p} for p in named[1:2]]
        questions += [{"id": f"CF-{p}", "family": "CF", "proposal": p} for p in named[2:]]
        # A family whose proposals are not named, and an item that names none.
        others = [p for p in sizes if p not in named]
        questions += [{"id": "F3-toy", "family": "F3", "proposal": others[0]},
                      {"id": "F5-toy", "family": "F5", "proposal": None}]
        self.questions = write_lines(tmp_path / "questions.jsonl", questions)
        self.summary = tmp_path / "summary.json"
        self.summary.write_text(json.dumps(summary or {
            "arm": "H+", "episodes": 4, "ok": 4, "failed": 0, "text_chars": 1000,
            "input_tokens": 10_000_000, "output_tokens": 2_000_000, "calls": 40,
            "calls_over_100k_input": 0, "wall_s": 120.0, "graphiti_core": "0.31.2",
            "model": "claude-haiku-5-5"}), encoding="utf-8")
        self.prepare = tmp_path / "prepare.json"
        self.prepare.write_text(json.dumps({"proposals": 2, "episodes": 4, "text_chars": 1000,
                                            "stream_sha256": "0" * 64}), encoding="utf-8")
        self.out = tmp_path / "projection.json"

    def project(self) -> int:
        return m0.main(["project", "--summary", str(self.summary), "--prepare", str(self.prepare),
                        "--manifest", str(self.manifest), "--questions", str(self.questions),
                        "--streams-dir", str(self.streams_dir), "--out", str(self.out)])

    def projection(self) -> dict:
        return json.loads(self.out.read_text(encoding="utf-8"))


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


def test_project_refuses_a_price_basis_that_does_not_state_both_prices(tmp_path, capsys):
    for basis in ("claude-haiku-5-5 list prices: $0.10 per million input tokens",
                  "claude-haiku-5-5 list prices: $0.50 per million output tokens",
                  "claude-haiku-5-5: $0.10 / MTok input, $0.50 / MTok output",
                  "$0.10 per million input tokens, $0.20 per million input tokens, "
                  "$0.50 per million output tokens"):
        world = World(tmp_path / str(abs(hash(basis))), SMALL, named=["PEP-1"], basis=basis)
        assert world.project() == 2
        assert not world.out.exists()
        assert "price basis" in capsys.readouterr().err


def test_project_refuses_streams_or_a_summary_that_do_not_match(tmp_path, capsys):
    world = World(tmp_path, SMALL, named=["PEP-1"])
    world.streams["CF"].write_text(world.streams["CF"].read_text(encoding="utf-8") + "\n",
                                   encoding="utf-8")
    assert world.project() == 2
    assert "cf.jsonl" in capsys.readouterr().err
    other = World(tmp_path / "other", SMALL, named=["PEP-1"], summary={
        "arm": "H+", "episodes": 5, "ok": 5, "failed": 0, "text_chars": 1000,
        "input_tokens": 1, "output_tokens": 1, "calls": 1, "calls_over_100k_input": 0,
        "wall_s": 1.0, "graphiti_core": "0.31.2", "model": "claude-haiku-5-5"})
    assert other.project() == 2
    assert "prepare.json" in capsys.readouterr().err


def test_project_writes_no_text_field_and_no_proposal_text(tmp_path):
    candidates = order(["PEP-11", "PEP-12"])
    # The named proposal costs $160 and each candidate $240: the total, $640, binds, and k is 1.
    sizes = {"PEP-1": (20_000, 20_000, 20_000), candidates[0]: (30_000, 30_000, 30_000),
             candidates[1]: (30_000, 30_000, 30_000)}
    world = World(tmp_path, sizes, named=["PEP-1"])
    assert world.project() == 0
    assert (world.projection()["decision"], world.projection()["k"]) == ("graphiti-reduced", 1)
    raw = world.out.read_text(encoding="utf-8")
    assert MARKER not in raw and "xxxxxxxxxx" not in raw

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
    assert projection["command"][:2] == ["m0.py", "project"]


def test_the_price_basis_parses_to_exact_prices():
    assert m0.parse_prices(BASIS) == (Fraction(1, 10), Fraction(1, 2))
