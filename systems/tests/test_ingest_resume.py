"""Graphiti's ingestion loop (graphiti/ingest.py) with a fake Graphiti: which episodes a run and a
resumed run attempt, the previous-episode window and saga predecessor each episode is given, the
ingest log's lines, and the totals of summary.json. It also checks that config.json agrees with
the adapter and with the image's lock. Standard library only: neither graphiti-core nor anthropic
is imported."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest
from asofbench_systems import usage
from graphiti import adapter, ingest

KEY = "sk-ant-api00-made-up-for-tests-0123456789"
GRAPHITI = Path(__file__).resolve().parents[1] / "graphiti"


def episode(n: int, proposal: str = "PEP-1", corpus: str = "pep", kind: str = "modify",
            text: str | None = None) -> dict:
    repo = "peps" if corpus == "pep" else "EIPs"
    commit = f"{n:02d}" * 20
    return {"episode_id": f"{commit[:12]}:{proposal}", "corpus": corpus, "repo": repo,
            "proposal": proposal, "ref": f'{proposal} ("Toy {proposal}")',
            "recorded_at": f"20{n:02d}-01-01T12:00:00Z", "commit": commit,
            "message": f"Toy commit {n}", "kind": kind,
            "text": f"@@ -1 +1 @@\n-Status: Draft\n+Status: {n}\n" if text is None else text}


def write_stream(path: Path, episodes: list[dict]) -> Path:
    path.write_text("".join(json.dumps(e) + "\n" for e in episodes), encoding="utf-8")
    return path


def log_line(e: dict, ok: bool, uuid: str | None, wall_s: float = 2.0) -> dict:
    return {"episode_id": e["episode_id"], "proposal": e["proposal"], "group_id": e["corpus"],
            "episode_uuid": uuid, "ok": ok, "error": None if ok else "RuntimeError: earlier",
            "started_at": "2026-11-01T00:00:00Z", "wall_s": wall_s, "text_chars": len(e["text"])}


def usage_line(context: str | None, input_tokens: int, output_tokens: int, created: int = 0,
               read: int = 0) -> dict:
    return {"ts": "2026-11-01T00:00:00.000Z", "context": context, "model": "claude-haiku-5-5",
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "cache_creation_input_tokens": created, "cache_read_input_tokens": read,
            "stop_reason": "tool_use"}


def write_jsonl(path: Path, lines: list[dict]) -> None:
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class FakeEngine:
    """Stands in for Graphiti: records each add_episode's arguments, makes two model calls per
    episode through the run's usage log (as the instrumented client does), stores each episode
    it ingests, and fails the episodes named in ``fail``."""

    def __init__(self, usage_log: usage.UsageLog, stored: list[dict], fail: set[str]) -> None:
        self.usage_log, self.stored, self.fail = usage_log, stored, fail
        self.calls: list[dict] = []
        self.source, self.version, self.closed = "text", "0.31.2", False

    async def add_episode(self, **arguments) -> str:
        self.calls.append(arguments)
        for tokens in (1000, 2000):
            await asyncio.sleep(0)
            self.usage_log.record(usage.usage_context.get(), "claude-haiku-5-5",
                                  {"input_tokens": tokens, "output_tokens": tokens // 10},
                                  "tool_use")
        if arguments["name"] in self.fail:
            raise RuntimeError(f"toy failure quoting the key {KEY} and more: " + "x" * 400)
        uuid = f"uuid-{arguments['name']}"
        self.stored.append({"uuid": uuid, "name": arguments["name"],
                            "group_id": arguments["group_id"]})
        return uuid

    async def stored_episodes(self) -> list[dict]:
        return list(self.stored)

    async def close(self) -> None:
        self.closed = True


class Harness:
    def __init__(self, tmp_path: Path, episodes: list[dict]) -> None:
        self.stream = write_stream(tmp_path / "toy.jsonl", episodes)
        self.out = tmp_path / "out"
        self.out.mkdir()
        self.key_file = tmp_path / "key"
        self.key_file.write_text(KEY + "\n", encoding="ascii")
        self.stored: list[dict] = []
        self.engines: list[FakeEngine] = []

    def run(self, resume: bool, fail: set[str] = frozenset()) -> int:
        async def open_engine(usage_log, key_file):
            assert key_file == str(self.key_file)
            engine = FakeEngine(usage_log, self.stored, set(fail))
            self.engines.append(engine)
            return engine

        return asyncio.run(ingest.run(self.stream, "H+", self.out, resume=resume,
                                      open_engine=open_engine, preflight=None,
                                      key_file=str(self.key_file)))

    def log(self) -> list[dict]:
        return read_jsonl(self.out / "ingest-log.jsonl")

    def usage(self) -> list[dict]:
        return read_jsonl(self.out / "usage.jsonl")

    def summary(self) -> dict:
        return json.loads((self.out / "summary.json").read_text(encoding="utf-8"))


def test_resume_attempts_the_failed_and_unstarted_episodes_only(tmp_path):
    episodes = [episode(n, kind="create" if n == 1 else "modify") for n in range(1, 6)]
    harness = Harness(tmp_path, episodes)
    # The first run: episodes 1 to 3 ingested, 4 failed, and the run died before 5.
    earlier = [log_line(episodes[0], True, "u1", 3.0), log_line(episodes[1], True, "u2", 4.0),
               log_line(episodes[2], True, "u3", 5.0), log_line(episodes[3], False, None, 6.0)]
    write_jsonl(harness.out / "ingest-log.jsonl", earlier)
    earlier_usage = [usage_line(e["episode_id"], 500 * n, 40 * n, created=n, read=2 * n)
                     for n, e in enumerate(episodes[:4], 1)]
    write_jsonl(harness.out / "usage.jsonl", earlier_usage)
    harness.stored += [{"uuid": u, "name": e["episode_id"], "group_id": "pep"}
                       for u, e in zip(("u1", "u2", "u3"), episodes, strict=False)]

    assert harness.run(resume=True) == 0
    (engine,) = harness.engines
    assert [call["name"] for call in engine.calls] == [episodes[3]["episode_id"],
                                                       episodes[4]["episode_id"]]
    fourth, fifth = engine.calls
    # The window and the saga predecessor come from the logged uuids, oldest first.
    assert fourth["previous_episode_uuids"] == ["u1", "u2", "u3"]
    assert fourth["saga_previous_episode_uuid"] == "u3"
    new4 = f"uuid-{episodes[3]['episode_id']}"
    assert fifth["previous_episode_uuids"] == ["u2", "u3", new4]
    assert fifth["saga_previous_episode_uuid"] == new4
    assert {call["saga"] for call in engine.calls} == {"PEP-1"}
    assert engine.closed

    log = harness.log()
    assert log[:4] == earlier
    assert [(line["episode_id"], line["ok"]) for line in log[4:]] == [
        (episodes[3]["episode_id"], True), (episodes[4]["episode_id"], True)]

    lines = harness.usage()
    assert lines[:4] == earlier_usage and len(lines) == 8
    assert [line["context"] for line in lines[4:]] == [episodes[3]["episode_id"]] * 2 + [
        episodes[4]["episode_id"]] * 2
    summary = harness.summary()
    # Every usage line is summed once, the earlier run's and the resumed run's, cache tokens
    # included in the input.
    assert summary["input_tokens"] == sum(line["input_tokens"]
                                          + line["cache_creation_input_tokens"]
                                          + line["cache_read_input_tokens"] for line in lines)
    assert summary["input_tokens"] == (500 + 1000 + 1500 + 2000) + (1 + 2 + 3 + 4) + (
        2 + 4 + 6 + 8) + 2 * (1000 + 2000)
    assert summary["output_tokens"] == (40 + 80 + 120 + 160) + 2 * (100 + 200)
    assert (summary["calls"], summary["calls_over_100k_input"]) == (8, 0)
    assert (summary["episodes"], summary["ok"], summary["failed"]) == (5, 5, 0)
    assert summary["text_chars"] == sum(len(e["text"]) for e in episodes)
    assert summary["wall_s"] == pytest.approx(3.0 + 4.0 + 5.0 + 6.0 + sum(
        line["wall_s"] for line in log[4:]), abs=1e-3)
    assert (summary["graphiti_core"], summary["model"], summary["arm"]) == (
        "0.31.2", "claude-haiku-5-5", "H+")


def test_a_resume_with_nothing_to_do_attempts_nothing_and_counts_the_same(tmp_path):
    episodes = [episode(1, kind="create"), episode(2)]
    harness = Harness(tmp_path, episodes)
    assert harness.run(resume=False) == 0
    first, log = harness.summary(), harness.log()
    assert harness.run(resume=True) == 0
    assert harness.engines[1].calls == []
    assert harness.log() == log and harness.summary() == first


def test_a_failed_episode_is_logged_and_the_run_continues_without_it(tmp_path):
    episodes = [episode(1, kind="create"), episode(2), episode(3)]
    harness = Harness(tmp_path, episodes)
    assert harness.run(resume=False, fail={episodes[1]["episode_id"]}) == 1
    first, second, third = harness.log()
    assert (first["ok"], second["ok"], third["ok"]) == (True, False, True)
    assert second["episode_uuid"] is None
    assert second["error"].startswith("RuntimeError: toy failure quoting the key [redacted]")
    assert len(second["error"]) == len("RuntimeError: ") + 300
    assert KEY not in (harness.out / "ingest-log.jsonl").read_text(encoding="utf-8")
    assert set(second) == {"episode_id", "proposal", "group_id", "episode_uuid", "ok", "error",
                           "started_at", "wall_s", "text_chars"}
    # The third episode follows the last ingested one, not the failed one.
    calls = harness.engines[0].calls
    assert calls[2]["previous_episode_uuids"] == [first["episode_uuid"]]
    assert calls[2]["saga_previous_episode_uuid"] == first["episode_uuid"]
    summary = harness.summary()
    assert (summary["episodes"], summary["ok"], summary["failed"]) == (3, 2, 1)
    # Its retry comes after the episodes ingested since, which it now follows.
    assert harness.run(resume=True) == 0
    retry = harness.engines[1].calls
    assert [call["name"] for call in retry] == [episodes[1]["episode_id"]]
    assert retry[0]["previous_episode_uuids"] == [first["episode_uuid"], third["episode_uuid"]]
    assert (harness.summary()["ok"], harness.summary()["failed"]) == (3, 0)


def test_each_proposal_keeps_its_own_window_and_the_corpora_are_separate_groups(tmp_path):
    episodes = [episode(1, "PEP-1", kind="create"), episode(2, "EIP-7", "eth", kind="create"),
                episode(3, "PEP-2", kind="create"), episode(4, "EIP-7", "eth"),
                episode(5, "PEP-1"), episode(6, "PEP-1"), episode(7, "PEP-1"),
                episode(8, "PEP-1", kind="delete", text="")]
    harness = Harness(tmp_path, episodes)
    assert harness.run(resume=False) == 0
    calls = {call["name"]: call for call in harness.engines[0].calls}
    uuid = {e["episode_id"]: f"uuid-{e['episode_id']}" for e in episodes}
    ids = [e["episode_id"] for e in episodes]
    assert calls[ids[3]]["previous_episode_uuids"] == [uuid[ids[1]]]
    assert calls[ids[2]]["previous_episode_uuids"] == []
    assert calls[ids[2]]["saga_previous_episode_uuid"] is None
    assert calls[ids[7]]["previous_episode_uuids"] == [uuid[ids[4]], uuid[ids[5]], uuid[ids[6]]]
    assert {call["group_id"] for call in calls.values()} == {"pep", "eth"}
    # Within a group, episodes are added in stream order.
    order = [call["name"] for call in harness.engines[0].calls]
    pep = [e["episode_id"] for e in episodes if e["corpus"] == "pep"]
    assert [name for name in order if name in pep] == pep
    # Every model call is attributed to the episode it was made for.
    assert {line["context"] for line in harness.usage()} == set(ids)


def test_episode_arguments_follow_the_configuration():
    e = episode(3, "ERC-20", "eth", kind="rename", text="")
    arguments = adapter.episode_arguments(e, previous_episode_uuids=["a", "b"],
                                          saga_previous_episode_uuid="b", source="text")
    assert arguments["name"] == e["episode_id"]
    assert arguments["episode_body"].endswith(
        "Unified diff of the file, which moved here from another repository or path:\n"
        "(empty: the change touched only lines this arm withholds, or the file's mode)")
    assert arguments["source_description"] == f"EIPs commit {e['commit']}"
    assert arguments["reference_time"].isoformat() == "2003-01-01T12:00:00+00:00"
    assert (arguments["group_id"], arguments["saga"], arguments["update_communities"]) == (
        "eth", "ERC-20", False)
    assert (arguments["previous_episode_uuids"], arguments["saga_previous_episode_uuid"]) == (
        ["a", "b"], "b")
    assert set(arguments) == {"name", "episode_body", "source", "source_description",
                              "reference_time", "group_id", "update_communities",
                              "previous_episode_uuids", "saga", "saga_previous_episode_uuid"}


def test_a_new_run_refuses_an_output_or_a_store_that_holds_a_run(tmp_path):
    episodes = [episode(1, kind="create")]
    harness = Harness(tmp_path, episodes)
    assert harness.run(resume=False) == 0
    with pytest.raises(ingest.Refused, match="--resume"):
        harness.run(resume=False)
    (tmp_path / "other").mkdir()
    other = Harness(tmp_path / "other", episodes)
    other.stored.append({"uuid": "left-over", "name": "x", "group_id": "pep"})
    with pytest.raises(ingest.Refused, match="store"):
        other.run(resume=False)
    assert other.engines[0].calls == [] and other.engines[0].closed


def test_a_resume_refuses_a_store_or_a_log_that_does_not_match(tmp_path):
    episodes = [episode(1, kind="create"), episode(2)]
    harness = Harness(tmp_path, episodes)
    write_jsonl(harness.out / "ingest-log.jsonl", [log_line(episodes[0], True, "u1")])
    write_jsonl(harness.out / "usage.jsonl", [])
    with pytest.raises(ingest.Refused, match="lacks"):
        harness.run(resume=True)
    harness.stored += [{"uuid": "u1", "name": episodes[0]["episode_id"], "group_id": "pep"},
                       {"uuid": "orphan", "name": episodes[1]["episode_id"], "group_id": "pep"}]
    with pytest.raises(ingest.Refused, match="orphan"):
        harness.run(resume=True)
    write_jsonl(harness.out / "ingest-log.jsonl", [log_line(episode(9), True, "u9")])
    with pytest.raises(ingest.Refused, match="not in the stream"):
        harness.run(resume=True)


def test_a_stream_that_cannot_be_ingested_is_refused_before_anything_starts(tmp_path):
    episodes = [episode(1, kind="create"), episode(2)]
    harness = Harness(tmp_path, [episodes[0], episodes[0]])
    with pytest.raises(ingest.Refused, match="occurs twice"):
        harness.run(resume=False)
    write_stream(harness.stream, [{k: v for k, v in episodes[1].items() if k != "corpus"}])
    with pytest.raises(ingest.Refused, match="no corpus"):
        harness.run(resume=False)
    harness.stream.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ingest.Refused, match="cannot be read"):
        harness.run(resume=False)
    assert harness.engines == [] and not (harness.out / "ingest-log.jsonl").exists()


def test_resume_drops_an_unfinished_last_line(tmp_path):
    episodes = [episode(1, kind="create"), episode(2)]
    harness = Harness(tmp_path, episodes)
    write_jsonl(harness.out / "ingest-log.jsonl", [log_line(episodes[0], True, "u1")])
    with open(harness.out / "ingest-log.jsonl", "a", encoding="utf-8") as handle:
        handle.write('{"episode_id": "unfinished')
    write_jsonl(harness.out / "usage.jsonl", [usage_line(episodes[0]["episode_id"], 10, 1)])
    with open(harness.out / "usage.jsonl", "a", encoding="utf-8") as handle:
        handle.write('{"ts": "2026')
    harness.stored.append({"uuid": "u1", "name": episodes[0]["episode_id"], "group_id": "pep"})
    assert harness.run(resume=True) == 0
    assert [line["ok"] for line in harness.log()] == [True, True]
    assert harness.summary()["calls"] == 1 + 2


def test_summary_counts_calls_over_100k_input_tokens():
    episodes = [episode(1, kind="create")]
    log = [log_line(episodes[0], True, "u1")]
    lines = [usage_line("a", 100_000, 1), usage_line("a", 99_999, 1, created=1),
             usage_line("a", 90_000, 1, read=10_001), usage_line(None, 5, 1)]
    summary = ingest.summarize(episodes, log, lines, arm="CF", graphiti_core="0.31.2")
    assert summary["calls_over_100k_input"] == 1
    assert summary["calls"] == 4 and summary["arm"] == "CF"


def test_config_json_records_the_adapter_s_configuration_and_the_lock():
    config = json.loads((GRAPHITI / "config.json").read_text(encoding="utf-8"))
    assert config["llm"]["model"] == config["llm"]["small_model"] == adapter.MODEL
    assert config["llm"]["base_url"] == "https://api.anthropic.com"
    assert config["embedder"]["base_url"] == adapter.EMBEDDER_URL
    assert config["embedder"]["embedding_dim"] == adapter.EMBEDDING_DIM == 2048
    assert config["embedder"]["ingest_model"] == adapter.DOCUMENT_MODEL
    assert config["embedder"]["query_model"] == adapter.QUERY_MODEL
    assert config["episodes"]["previous_episodes"]["count"] == adapter.EPISODE_WINDOW_LEN
    assert config["concurrency"]["semaphore_limit"] == adapter.SEMAPHORE_LIMIT
    assert config["cross_encoder"]["raises"] == adapter.NO_RERANKER
    pins = {}
    lock = (GRAPHITI / "requirements.txt").read_text(encoding="utf-8")
    for match in re.finditer(r"^([A-Za-z0-9_.-]+)==([^ ;\\\n]+)(?: ;([^\\\n]*))?", lock, re.M):
        name, version, marker = match.groups()
        if marker is None or "linux" in marker:
            pins[name] = version
    assert adapter.recorded_packages(config) == pins
