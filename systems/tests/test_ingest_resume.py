"""Graphiti's ingestion loop (graphiti/ingest.py) with a fake Graphiti: strict stream order, the
in-place retries, the infrastructure stop and the stop after consecutive failures, what a resumed
run attempts, the previous-episode window and saga predecessor each episode is given, the ingest
log and run.json, and the totals of summary.json. It also checks that config.json agrees with the
adapter and with the image's lock. Standard library only: neither graphiti-core nor anthropic is
imported."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import re
import signal
import time
from pathlib import Path

import pytest
from asofbench_systems import llm, usage
from graphiti import adapter, ingest

KEY = "sk-ant-api00-made-up-for-tests-0123456789"
GRAPHITI = Path(__file__).resolve().parents[1] / "graphiti"
IMAGES = {name: "sha256:" + str(n) * 64 for n, name in enumerate(ingest.IMAGE_NAMES, 1)}
MODEL = "claude-haiku-5-5"
# A made-up dated snapshot id, which the fake's responses name for the model its calls ask for.
SNAPSHOT = "claude-haiku-5-5-20990101"


def failure(module: str, name: str) -> type[Exception]:
    """An exception class that stands in for one of a package's, by its module and name."""
    return type(name, (Exception,), {"__module__": module})


ServiceUnavailable = failure("neo4j.exceptions", "ServiceUnavailable")
SessionExpired = failure("neo4j.exceptions", "SessionExpired")
EmbedderDown = failure("openai._exceptions", "APIConnectionError")
Overloaded = failure("anthropic._exceptions", "OverloadedError")
BadRequest = failure("anthropic._exceptions", "BadRequestError")
GraphitiRateLimit = failure("graphiti_core.llm_client.errors", "RateLimitError")


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


def log_line(e: dict, ok: bool, uuid: str | None, wall_s: float = 2.0, attempts: int = 1,
             invocation: int = 1) -> dict:
    return {"episode_id": e["episode_id"], "proposal": e["proposal"], "group_id": e["corpus"],
            "episode_uuid": uuid, "ok": ok, "error": None if ok else "RuntimeError: earlier",
            "started_at": "2026-11-01T00:00:00Z", "wall_s": wall_s, "text_chars": len(e["text"]),
            "attempts": attempts, "invocation": invocation}


def usage_line(context: str | None, input_tokens: int, output_tokens: int, created: int = 0,
               read: int = 0, model: str = MODEL, invocation: int | None = None,
               attempt: int | None = None, response_model: str | None = MODEL) -> dict:
    return {"ts": "2026-11-01T00:00:00.000Z", "context": context, "invocation": invocation,
            "attempt": attempt, "model": model, "response_model": response_model,
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "cache_creation_input_tokens": created, "cache_read_input_tokens": read,
            "stop_reason": "tool_use"}


def write_jsonl(path: Path, lines: list[dict]) -> None:
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class FakeEngine:
    """Stands in for Graphiti: records each add_episode's arguments, makes two model calls per
    attempt through the run's usage log (as the instrumented client does), and stores each
    episode it ingests. ``outcomes`` gives, for an episode, what each of its attempts does: an
    exception to raise, "orphan" to store the episode and then raise, "swallowed" to record a
    call that raised and succeed all the same (as Graphiti does where it catches one),
    "overloaded" to record a call that raised OverloadedError and raise it (an infrastructure
    stop), "bad-request" to record a call that raised BadRequestError and raise it (a failed
    attempt), or None to succeed (the default once the list is used up). With ``broken``, reading
    the store raises an error that no rule expects."""

    def __init__(self, usage_log: usage.UsageLog, stored: list[dict],
                 outcomes: dict[str, list], broken: bool = False) -> None:
        self.usage_log, self.stored, self.broken = usage_log, stored, broken
        self.outcomes = {name: list(plan) for name, plan in outcomes.items()}
        self.calls: list[dict] = []
        self.removed: list[str] = []
        self.source, self.version, self.closed = "text", "0.31.2", False

    def call(self, tokens: int) -> None:
        self.usage_log.record(usage.attribution(), MODEL, SNAPSHOT,
                              {"input_tokens": tokens, "output_tokens": tokens // 10}, "tool_use")

    async def add_episode(self, **arguments) -> str:
        self.calls.append(arguments)
        for tokens in (1000, 2000):
            await asyncio.sleep(0)
            self.call(tokens)
        plan = self.outcomes.get(arguments["name"], [])
        outcome = plan.pop(0) if plan else None
        uuid = f"uuid-{arguments['name']}"
        if outcome == "orphan":
            self.stored.append({"uuid": uuid, "name": arguments["name"],
                                "group_id": arguments["group_id"]})
            raise RuntimeError("failed after writing")
        if outcome == "swallowed":
            self.usage_log.record_error(usage.attribution(), "claude-haiku-5-5",
                                        "OverloadedError")
        elif outcome == "overloaded":
            self.usage_log.record_error(usage.attribution(), "claude-haiku-5-5",
                                        "OverloadedError")
            raise Overloaded("still overloaded after the SDK's retries")
        elif outcome == "bad-request":
            self.usage_log.record_error(usage.attribution(), "claude-haiku-5-5",
                                        "BadRequestError")
            raise BadRequest("a request that the API refused")
        elif outcome is not None:
            raise outcome
        self.stored.append({"uuid": uuid, "name": arguments["name"],
                            "group_id": arguments["group_id"]})
        return uuid

    async def stored_episodes(self) -> list[dict]:
        if self.broken:
            raise KeyError("an error no rule expects")
        return list(self.stored)

    async def remove_episode(self, uuid: str) -> None:
        self.call(500)
        self.removed.append(uuid)
        self.stored[:] = [episode for episode in self.stored if episode["uuid"] != uuid]

    async def close(self) -> None:
        self.closed = True


class Harness:
    def __init__(self, tmp_path: Path, episodes: list[dict], arm: str = "H+") -> None:
        self.stream = write_stream(tmp_path / "toy.jsonl", episodes)
        self.arm = arm
        self.out = tmp_path / "out"
        self.out.mkdir()
        self.key_file = tmp_path / "key"
        self.key_file.write_text(KEY + "\n", encoding="ascii")
        self.stored: list[dict] = []
        self.engines: list[FakeEngine] = []
        self.messages: list[str] = []

    def open_engine(self, outcomes: dict[str, list] | None = None, broken: bool = False):
        async def open_engine(usage_log, key_file):
            assert key_file == str(self.key_file)
            engine = FakeEngine(usage_log, self.stored, outcomes or {}, broken)
            self.engines.append(engine)
            return engine

        return open_engine

    def run(self, resume: bool, outcomes: dict[str, list] | None = None,
            arm: str | None = None, skip: list[str] = (), broken: bool = False) -> int:
        return asyncio.run(ingest.run(self.stream, arm or self.arm, self.out, resume=resume,
                                      images=IMAGES, skip=skip,
                                      open_engine=self.open_engine(outcomes, broken),
                                      preflight=None, key_file=str(self.key_file),
                                      delays=(0.0, 0.0), progress=self.messages.append))

    def remove(self, uuids: list[str]) -> int:
        return asyncio.run(ingest.remove_episodes(uuids, self.out, images=IMAGES,
                                                  open_engine=self.open_engine(),
                                                  preflight=None, key_file=str(self.key_file),
                                                  progress=self.messages.append))

    def begun(self, log: list[dict], usage_lines: list[dict]) -> None:
        """The state an earlier invocation left: its run.json, log and usage lines."""
        sha = hashlib.sha256(self.stream.read_bytes()).hexdigest()
        (self.out / "run.json").write_text(json.dumps({
            "stream": self.stream.name, "stream_sha256": sha, "arm": self.arm,
            "invocations": [{"started_at": "2026-11-01T00:00:00Z", "resume": False,
                             "images": IMAGES, "elapsed_s": 40.0, "exit": 2}]}),
            encoding="utf-8")
        write_jsonl(self.out / "ingest-log.jsonl", log)
        write_jsonl(self.out / "usage.jsonl", usage_lines)

    def log(self) -> list[dict]:
        return read_jsonl(self.out / "ingest-log.jsonl")

    def usage(self) -> list[dict]:
        return read_jsonl(self.out / "usage.jsonl")

    def summary(self) -> dict:
        return json.loads((self.out / "summary.json").read_text(encoding="utf-8"))

    def record(self) -> dict:
        return json.loads((self.out / "run.json").read_text(encoding="utf-8"))


def test_resume_continues_after_the_logged_episodes_and_never_retries_a_failed_one(tmp_path):
    episodes = [episode(n, kind="create" if n == 1 else "modify") for n in range(1, 6)]
    harness = Harness(tmp_path, episodes)
    # The earlier invocation: episodes 1 to 3 ingested, 4 logged as failed after its attempts,
    # and it stopped before 5.
    earlier = [log_line(episodes[0], True, "u1", 3.0), log_line(episodes[1], True, "u2", 4.0),
               log_line(episodes[2], True, "u3", 5.0),
               log_line(episodes[3], False, None, 6.0, attempts=3)]
    earlier_usage = [usage_line(e["episode_id"], 500 * n, 40 * n, created=n, read=2 * n)
                     for n, e in enumerate(episodes[:4], 1)]
    harness.begun(earlier, earlier_usage)
    harness.stored += [{"uuid": u, "name": e["episode_id"], "group_id": "pep"}
                       for u, e in zip(("u1", "u2", "u3"), episodes, strict=False)]

    assert harness.run(resume=True) == ingest.EXIT_FAILED
    (engine,) = harness.engines
    # Only episode 5 is attempted: 4 is logged, so it is never attempted again.
    assert [call["name"] for call in engine.calls] == [episodes[4]["episode_id"]]
    (fifth,) = engine.calls
    # The window holds the proposal's last ingested episodes, all recorded before it, oldest
    # first; the failed episode 4 is not one of them.
    assert fifth["previous_episode_uuids"] == ["u1", "u2", "u3"]
    assert fifth["saga_previous_episode_uuid"] == "u3"
    assert fifth["saga"] == "PEP-1" and engine.closed

    log = harness.log()
    assert log[:4] == earlier
    assert [(line["episode_id"], line["ok"], line["attempts"]) for line in log[4:]] == [
        (episodes[4]["episode_id"], True, 1)]
    lines = harness.usage()
    assert lines[:4] == earlier_usage and len(lines) == 6
    summary = harness.summary()
    # Every usage line is summed once, the earlier invocation's and this one's, cache tokens
    # included in the input.
    assert summary["input_tokens"] == sum(line["input_tokens"]
                                          + line["cache_creation_input_tokens"]
                                          + line["cache_read_input_tokens"] for line in lines)
    assert summary["input_tokens"] == (500 + 1000 + 1500 + 2000) + (1 + 2 + 3 + 4) + (
        2 + 4 + 6 + 8) + (1000 + 2000)
    assert summary["output_tokens"] == (40 + 80 + 120 + 160) + (100 + 200)
    assert (summary["calls"], summary["calls_over_100k_input"]) == (6, 0)
    assert (summary["episodes"], summary["ok"], summary["failed"]) == (5, 4, 1)
    assert summary["text_chars"] == sum(len(e["text"]) for e in episodes)
    assert summary["wall_s"] == pytest.approx(3.0 + 4.0 + 5.0 + 6.0 + log[4]["wall_s"], abs=1e-3)
    record = harness.record()
    assert summary["elapsed_s"] == pytest.approx(40.0 + record["invocations"][1]["elapsed_s"])
    assert summary["invocations"] == 2 and record["invocations"][1]["resume"] is True
    assert record["invocations"][1]["exit"] == ingest.EXIT_FAILED
    assert (summary["graphiti_core"], summary["model"], summary["arm"]) == (
        "0.31.2", "claude-haiku-5-5", "H+")
    # The model every call asked for, and the ids the responses named, a snapshot's among them.
    assert summary["response_models"] == [MODEL, SNAPSHOT]
    assert (summary["exit"], summary["api_errors"]) == (ingest.EXIT_FAILED, 0)
    assert summary["removal"] == summary["unattributed"] == {
        "input_tokens": 0, "output_tokens": 0, "calls": 0, "api_errors": 0}
    assert summary["images"] == {name: [value] for name, value in IMAGES.items()}
    assert (summary["stream"], summary["stream_sha256"]) == (
        "toy.jsonl", hashlib.sha256(harness.stream.read_bytes()).hexdigest())


def test_a_resume_with_nothing_to_do_attempts_nothing_and_counts_the_same(tmp_path):
    episodes = [episode(1, kind="create"), episode(2)]
    harness = Harness(tmp_path, episodes)
    assert harness.run(resume=False) == ingest.EXIT_INGESTED
    first, log = harness.summary(), harness.log()
    assert harness.run(resume=True) == ingest.EXIT_INGESTED
    assert harness.engines[1].calls == []
    second = harness.summary()
    assert harness.log() == log and second["invocations"] == 2

    def counts(summary: dict) -> dict:
        return {key: value for key, value in summary.items()
                if key not in ("elapsed_s", "invocations")}

    assert counts(second) == counts(first)


def test_a_failed_attempt_is_retried_in_place_and_a_failed_episode_is_logged_once(tmp_path):
    episodes = [episode(1, kind="create"), episode(2), episode(3), episode(4)]
    ids = [e["episode_id"] for e in episodes]
    harness = Harness(tmp_path, episodes)
    message = RuntimeError(f"toy failure quoting the key {KEY} and more: " + "x" * 400)
    # Episode 2 fails once and then succeeds; episode 3 fails all three of its attempts.
    outcomes = {ids[1]: [ValueError("bad output")],
                ids[2]: [BadRequest("a"), BadRequest("b"), message]}
    assert harness.run(resume=False, outcomes=outcomes) == ingest.EXIT_FAILED
    calls = harness.engines[0].calls
    assert [call["name"] for call in calls] == [ids[0], ids[1], ids[1], ids[2], ids[2], ids[2],
                                                ids[3]]
    # Every attempt of an episode is given the same previous episodes and saga predecessor.
    assert calls[1]["previous_episode_uuids"] == calls[2]["previous_episode_uuids"] == [
        f"uuid-{ids[0]}"]
    assert all(call["saga_previous_episode_uuid"] == f"uuid-{ids[1]}" for call in calls[3:6])
    first, second, third, fourth = harness.log()
    assert (second["ok"], second["attempts"], second["error"]) == (
        True, 2, "ValueError: bad output")
    assert (third["ok"], third["attempts"], third["episode_uuid"]) == (False, 3, None)
    assert third["error"].startswith("RuntimeError: toy failure quoting the key [redacted]")
    assert len(third["error"]) == len("RuntimeError: ") + 300
    assert KEY not in (harness.out / "ingest-log.jsonl").read_text(encoding="utf-8")
    assert set(third) == set(ingest.LOG_FIELDS)
    # The fourth episode follows the last ingested one, not the failed one.
    assert calls[6]["previous_episode_uuids"] == [f"uuid-{ids[0]}", f"uuid-{ids[1]}"]
    assert (first["attempts"], fourth["ok"], fourth["error"]) == (1, True, None)
    # Every attempt's model calls are attributed to its episode.
    assert [line["context"] for line in harness.usage()].count(ids[2]) == 6
    assert any("attempt 1 failed" in text for text in harness.messages)
    # A resumed run never attempts the failed episode again.
    assert harness.run(resume=True) == ingest.EXIT_FAILED
    assert harness.engines[1].calls == []
    assert (harness.summary()["ok"], harness.summary()["failed"]) == (3, 1)


@pytest.mark.parametrize("error", [ServiceUnavailable("neo4j:7687"), SessionExpired("lost"),
                                   EmbedderDown("embedder:8080"), Overloaded("529"),
                                   GraphitiRateLimit("429")])
def test_an_infrastructure_error_stops_the_run_unlogged_and_a_resume_attempts_it(tmp_path,
                                                                                 error):
    episodes = [episode(n, kind="create" if n == 1 else "modify") for n in range(1, 5)]
    ids = [e["episode_id"] for e in episodes]
    harness = Harness(tmp_path, episodes)
    assert harness.run(resume=False, outcomes={ids[2]: [error]}) == ingest.EXIT_STOPPED
    # The run stops at once: no retry, and nothing after the episode.
    assert [call["name"] for call in harness.engines[0].calls] == ids[:3]
    assert [line["episode_id"] for line in harness.log()] == ids[:2]
    summary = harness.summary()
    assert (summary["ok"], summary["failed"], summary["calls"]) == (2, 0, 6)
    assert summary["exit"] == ingest.EXIT_STOPPED
    stopped = harness.record()["invocations"][0]
    assert stopped["exit"] == ingest.EXIT_STOPPED
    assert stopped["stop"] == {"episode_id": ids[2],
                               "error": ingest.infrastructure_error(error)}
    assert any("stopped (exit 2)" in text for text in harness.messages)
    # Once the infrastructure is back, a resume continues from that episode.
    assert harness.run(resume=True) == ingest.EXIT_INGESTED
    resumed = harness.engines[1].calls
    assert [call["name"] for call in resumed] == ids[2:]
    assert resumed[0]["previous_episode_uuids"] == [f"uuid-{ids[0]}", f"uuid-{ids[1]}"]
    assert [line["episode_id"] for line in harness.log()] == ids
    assert all(line["attempts"] == 1 for line in harness.log())
    # The stopped attempt's calls stay counted, under its episode.
    assert [line["context"] for line in harness.usage()].count(ids[2]) == 4
    assert [i["exit"] for i in harness.record()["invocations"]] == [ingest.EXIT_STOPPED,
                                                                     ingest.EXIT_INGESTED]


def test_a_stop_lets_the_other_group_finish_its_episode_and_start_no_other(tmp_path):
    episodes = [episode(1, "PEP-1", kind="create"), episode(2, "EIP-1", "eth", kind="create"),
                episode(3, "PEP-1"), episode(4, "EIP-1", "eth")]
    ids = [e["episode_id"] for e in episodes]
    harness = Harness(tmp_path, episodes)
    assert harness.run(resume=False, outcomes={ids[0]: [ServiceUnavailable()]}) == 2
    assert sorted(call["name"] for call in harness.engines[0].calls) == sorted(ids[:2])
    assert [(line["episode_id"], line["ok"]) for line in harness.log()] == [(ids[1], True)]
    assert harness.run(resume=True) == ingest.EXIT_INGESTED
    assert sorted(call["name"] for call in harness.engines[1].calls) == sorted(
        [ids[0], ids[2], ids[3]])


def test_five_consecutive_failures_in_a_group_stop_the_run(tmp_path):
    episodes = [episode(n, kind="create" if n == 1 else "modify") for n in range(1, 10)]
    ids = [e["episode_id"] for e in episodes]
    harness = Harness(tmp_path, episodes)
    outcomes = {ids[n]: [ValueError(str(n))] * 3 for n in range(1, 7)}
    assert harness.run(resume=False, outcomes=outcomes) == ingest.EXIT_FAILURES
    assert [(line["episode_id"], line["ok"]) for line in harness.log()] == [
        (ids[0], True)] + [(ids[n], False) for n in range(1, 6)]
    assert {call["name"] for call in harness.engines[0].calls} == set(ids[:6])
    assert harness.summary()["failed"] == 5
    assert any("5 consecutive pep episodes failed" in text for text in harness.messages)
    assert harness.record()["invocations"][0]["stop"] == {"group": "pep",
                                                          "error": "5 consecutive failures"}
    # A resume counts on from the log: the sixth failure in a row stops it again at once.
    assert harness.run(resume=True, outcomes={ids[6]: [ValueError("6")] * 3}) == 3
    assert [call["name"] for call in harness.engines[1].calls] == [ids[6]] * 3
    # An ingested episode ends the run of failures.
    assert harness.run(resume=True) == ingest.EXIT_FAILED
    assert [call["name"] for call in harness.engines[2].calls] == ids[7:]


def test_a_failed_attempt_that_left_the_episode_in_the_store_stops_the_run(tmp_path):
    episodes = [episode(1, kind="create"), episode(2)]
    ids = [e["episode_id"] for e in episodes]
    harness = Harness(tmp_path, episodes)
    assert harness.run(resume=False, outcomes={ids[1]: ["orphan"]}) == ingest.EXIT_STOPPED
    assert [line["episode_id"] for line in harness.log()] == [ids[0]]
    assert any(f"uuid-{ids[1]}" in text for text in harness.messages)
    # A resume refuses the store until the episode left in it is removed.
    with pytest.raises(ingest.Refused, match="remove-episodes"):
        harness.run(resume=True)
    harness.stored[:] = [s for s in harness.stored if s["name"] != ids[1]]
    assert harness.run(resume=True) == ingest.EXIT_INGESTED
    assert [line["episode_id"] for line in harness.log()] == ids


def test_each_proposal_keeps_its_own_window_and_the_corpora_are_separate_groups(tmp_path):
    episodes = [episode(1, "PEP-1", kind="create"), episode(2, "EIP-7", "eth", kind="create"),
                episode(3, "PEP-2", kind="create"), episode(4, "EIP-7", "eth"),
                episode(5, "PEP-1"), episode(6, "PEP-1"), episode(7, "PEP-1"),
                episode(8, "PEP-1", kind="delete", text="")]
    harness = Harness(tmp_path, episodes)
    assert harness.run(resume=False) == ingest.EXIT_INGESTED
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


def test_no_output_holds_a_local_path(tmp_path):
    episodes = [episode(1, kind="create"), episode(2)]
    harness = Harness(tmp_path, episodes)
    assert harness.run(resume=False, outcomes={episodes[1]["episode_id"]: [ValueError("x")]}) == 0
    for name in ("run.json", "summary.json", "usage.jsonl", "ingest-log.jsonl"):
        text = (harness.out / name).read_text(encoding="utf-8")
        assert str(tmp_path) not in text and "/" not in text


def test_a_resume_refuses_another_stream_or_arm_or_a_missing_run(tmp_path):
    episodes = [episode(1, kind="create"), episode(2)]
    harness = Harness(tmp_path, episodes)
    with pytest.raises(ingest.Refused, match="no run to resume"):
        harness.run(resume=True)
    assert harness.run(resume=False) == ingest.EXIT_INGESTED
    with pytest.raises(ingest.Refused, match="began with the H\\+ stream"):
        harness.run(resume=True, arm="H-")
    write_stream(harness.stream, [episode(1, kind="create"), episode(2, text="edited\n")])
    with pytest.raises(ingest.Refused, match="SHA-256"):
        harness.run(resume=True)
    assert len(harness.engines) == 1


def test_a_key_file_that_cannot_be_read_is_refused_naming_the_file_and_never_its_content(
        tmp_path):
    # On a Linux host the secret keeps the host file's owner and mode, which the images' user,
    # uid 10001, may not be able to read; Docker Desktop's file sharing hides this.
    episodes = [episode(1, kind="create")]
    harness = Harness(tmp_path, episodes)
    unreadable = tmp_path / "unreadable"
    unreadable.write_text(KEY + "\n", encoding="ascii")
    unreadable.chmod(0)
    no_key = tmp_path / "no-key"
    no_key.write_text(f"{KEY}\n{KEY}\n", encoding="ascii")
    cases = [(tmp_path / "missing", "cannot be read"), (tmp_path, "cannot be read"),
             (no_key, "does not hold an API key")]
    if os.geteuid() != 0:  # root reads a file whatever its mode
        cases.append((unreadable, "cannot be read (Permission denied)"))
    for path, why in cases:
        harness.key_file = path
        for attempt in (lambda: harness.run(resume=False), lambda: harness.remove(["any-uuid"])):
            with pytest.raises(ingest.Refused) as caught:
                attempt()
            message = str(caught.value)
            assert f"{path} {why}" in message and "uid 10001" in message, message
            assert KEY not in message
    assert harness.engines == [] and list(harness.out.iterdir()) == []


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
    assert harness.run(resume=False) == ingest.EXIT_INGESTED
    with pytest.raises(ingest.Refused, match="--resume"):
        harness.run(resume=False)
    (tmp_path / "other").mkdir()
    other = Harness(tmp_path / "other", episodes)
    other.stored.append({"uuid": "left-over", "name": "x", "group_id": "pep"})
    with pytest.raises(ingest.Refused, match="store"):
        other.run(resume=False)
    assert other.engines[0].calls == [] and other.engines[0].closed
    assert other.record()["invocations"][0]["exit"] == ingest.EXIT_STOPPED


def test_a_resume_refuses_a_store_or_a_log_that_does_not_match(tmp_path):
    episodes = [episode(1, kind="create"), episode(2), episode(3)]
    harness = Harness(tmp_path, episodes)
    harness.begun([log_line(episodes[0], True, "u1")], [])
    with pytest.raises(ingest.Refused, match="lacks"):
        harness.run(resume=True)
    harness.stored += [{"uuid": "u1", "name": episodes[0]["episode_id"], "group_id": "pep"},
                       {"uuid": "left", "name": episodes[1]["episode_id"], "group_id": "pep"}]
    with pytest.raises(ingest.Refused, match="remove-episodes"):
        harness.run(resume=True)
    write_jsonl(harness.out / "ingest-log.jsonl", [log_line(episode(9), True, "u9")])
    with pytest.raises(ingest.Refused, match="not in the stream"):
        harness.run(resume=True)
    # A log that skips an episode of its group is not the group's first episodes.
    write_jsonl(harness.out / "ingest-log.jsonl", [log_line(episodes[0], True, "u1"),
                                                   log_line(episodes[2], True, "u3")])
    with pytest.raises(ingest.Refused, match="first ones"):
        harness.run(resume=True)
    write_jsonl(harness.out / "ingest-log.jsonl", [log_line(episodes[0], True, "u1"),
                                                   log_line(episodes[0], False, None)])
    with pytest.raises(ingest.Refused, match="has a line already"):
        harness.run(resume=True)
    # A line must name the invocation and the attempts that ended it, which place its calls.
    for broken in ({"invocation": 0}, {"invocation": None}, {"attempts": 0}, {"attempts": "1"}):
        write_jsonl(harness.out / "ingest-log.jsonl",
                    [{**log_line(episodes[0], True, "u1"), **broken}])
        with pytest.raises(ingest.Refused, match="no invocation or no attempts"):
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
    harness.begun([log_line(episodes[0], True, "u1")],
                  [usage_line(episodes[0]["episode_id"], 10, 1)])
    with open(harness.out / "ingest-log.jsonl", "a", encoding="utf-8") as handle:
        handle.write('{"episode_id": "unfinished')
    with open(harness.out / "usage.jsonl", "a", encoding="utf-8") as handle:
        handle.write('{"ts": "2026')
    harness.stored.append({"uuid": "u1", "name": episodes[0]["episode_id"], "group_id": "pep"})
    assert harness.run(resume=True) == ingest.EXIT_INGESTED
    assert [line["ok"] for line in harness.log()] == [True, True]
    assert harness.summary()["calls"] == 1 + 2


def test_infrastructure_errors_are_told_from_the_episode_s_own():
    assert ingest.infrastructure_error(ServiceUnavailable()) == "neo4j ServiceUnavailable"
    assert ingest.infrastructure_error(ConnectionRefusedError()) == "builtins ConnectionError"
    assert ingest.infrastructure_error(Overloaded()) == "anthropic OverloadedError"
    assert ingest.infrastructure_error(ValueError("bad output")) is None
    assert ingest.infrastructure_error(BadRequest("prompt is too long")) is None
    # Through a cause, as Graphiti raises its RateLimitError from the SDK's, and inside a group.
    try:
        try:
            raise failure("anthropic._exceptions", "RateLimitError")("429")
        except Exception as cause:
            raise RuntimeError("wrapped") from cause
    except RuntimeError as wrapped:
        assert ingest.infrastructure_error(wrapped) == "anthropic RateLimitError"
    group = ExceptionGroup("several", [ValueError("x"), SessionExpired("lost")])
    assert ingest.infrastructure_error(group) == "neo4j SessionExpired"
    subclass = type("APITimeoutError", (EmbedderDown,), {"__module__": "openai._exceptions"})
    assert ingest.infrastructure_error(subclass()) == "openai APIConnectionError"


def test_the_image_ids_are_parsed_and_checked():
    text = ",".join(f"{name}={value}" for name, value in IMAGES.items())
    assert ingest.parse_image_ids(text) == IMAGES
    for bad in (None, "", text.replace("egress=", "proxy="), text + ",neo4j=" + IMAGES["neo4j"],
                text.replace(IMAGES["graphiti"], "asofbench/graphiti")):
        with pytest.raises(ingest.Refused, match="ASOFBENCH_IMAGE_IDS"):
            ingest.parse_image_ids(bad)


def record_of(elapsed: list[float | None]) -> dict:
    return {"arm": "CF", "stream": "cf.jsonl", "stream_sha256": "0" * 64,
            "invocations": [{"images": IMAGES, "elapsed_s": value, "exit": 0}
                            for value in elapsed]}


def test_summary_counts_calls_over_100k_input_tokens_and_names_every_model():
    episodes = [episode(1, kind="create")]
    a = episodes[0]["episode_id"]
    log = [log_line(episodes[0], True, "u1")]
    lines = [usage_line(a, 100_000, 1), usage_line(a, 99_999, 1, created=1),
             usage_line(a, 90_000, 1, read=10_001), usage_line(a, 5, 1)]
    # The invocation killed without an elapsed time counts as none.
    summary = ingest.summarize(episodes, log, lines, record_of([12.5, None, 3.0]),
                               graphiti_core="0.31.2")
    assert summary["calls_over_100k_input"] == 1
    assert (summary["calls"], summary["arm"], summary["model"]) == (4, "CF", "claude-haiku-5-5")
    assert (summary["elapsed_s"], summary["invocations"]) == (15.5, 3)
    # ``model`` is the model the calls asked for; the ids the responses named are listed apart,
    # and a call that raised names none.
    lines += [usage_line(a, 2, 1, response_model=SNAPSHOT),
              {**usage_line(a, 0, 0, response_model=None), "error": "OverloadedError"}]
    summary = ingest.summarize(episodes, log, lines, record_of([1.0]), graphiti_core="0.31.2")
    assert (summary["model"], summary["response_models"]) == (MODEL, [MODEL, SNAPSHOT])
    lines.append(usage_line(a, 1, 1, model="claude-sonnet-5-5", response_model="claude-sonnet-5-5"))
    summary = ingest.summarize(episodes, log, lines, record_of([1.0]), graphiti_core="0.31.2")
    assert summary["model"] == ["claude-haiku-5-5", "claude-sonnet-5-5"]
    assert summary["response_models"] == [MODEL, SNAPSHOT, "claude-sonnet-5-5"]
    empty = ingest.summarize(episodes, [], [], record_of([1.0]), graphiti_core="0.31.2")
    assert (empty["model"], empty["response_models"]) == (None, [])


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
    assert "plan" not in json.dumps(config)
    pins = {}
    lock = (GRAPHITI / "requirements.txt").read_text(encoding="utf-8")
    for match in re.finditer(r"^([A-Za-z0-9_.-]+)==([^ ;\\\n]+)(?: ;([^\\\n]*))?", lock, re.M):
        name, version, marker = match.groups()
        if marker is None or "linux" in marker:
            pins[name] = version
    assert adapter.recorded_packages(config) == pins


def test_the_environment_s_proxy_outlasts_the_client_s_request_timeout():
    """compose.yaml gives the egress proxy an idle timeout longer than the timeout that the
    client config.json records gives a request, so that the proxy never cuts a request the
    client still waits on, and config.json records both."""
    config = json.loads((GRAPHITI / "config.json").read_text(encoding="utf-8"))
    compose = (GRAPHITI / "compose.yaml").read_text(encoding="utf-8")
    (value,) = re.findall(r"^\s+EGRESS_IDLE_TIMEOUT:\s*\"?([0-9.]+)\"?\s*$", compose, re.M)
    client = inspect.signature(llm.make_async_client).parameters
    assert config["llm"]["timeout_s"] == client["timeout_s"].default
    assert config["llm"]["max_retries"] == client["max_retries"].default
    assert float(value) == config["network_policy"]["proxy"]["idle_timeout_s"]
    assert float(value) >= config["llm"]["timeout_s"] + 60


def test_a_second_invocation_stops_at_once_while_one_holds_the_lock(tmp_path):
    episodes = [episode(1, kind="create"), episode(2)]
    harness = Harness(tmp_path, episodes)
    with ingest.hold_lock(harness.out, "ingest"):
        holder = json.loads((harness.out / "run.lock").read_text(encoding="utf-8"))
        assert set(holder) == {"token", "command", "host", "pid", "since"}
        with pytest.raises(ingest.Refused, match="another invocation holds the lock"):
            harness.run(resume=False)
        with pytest.raises(ingest.Refused, match="another invocation holds the lock"):
            harness.remove(["any-uuid"])
        assert [path.name for path in harness.out.iterdir()] == ["run.lock"]
    assert harness.engines == [] and list(harness.out.iterdir()) == []
    # Once the lock is free, the run goes ahead, and leaves no lock behind.
    assert harness.run(resume=False) == ingest.EXIT_INGESTED
    assert not (harness.out / "run.lock").exists()


def test_a_lock_that_a_killed_invocation_left_says_how_old_its_heartbeat_is(tmp_path):
    episodes = [episode(1, kind="create")]
    harness = Harness(tmp_path, episodes)
    lock = harness.out / "run.lock"
    lock.write_text(json.dumps({"token": "gone", "command": "ingest", "host": "c0ffee",
                                "pid": 7, "since": "2026-11-01T00:00:00Z"}), encoding="utf-8")
    hour_ago = lock.stat().st_mtime - 3600
    os.utime(lock, (hour_ago, hour_ago))
    with pytest.raises(ingest.Refused, match=r"heartbeat is 36\d\d s old.*remove run.lock"):
        harness.run(resume=False)
    lock.unlink()
    assert harness.run(resume=False) == ingest.EXIT_INGESTED


def test_a_lock_is_touched_while_held_and_removed_only_by_its_holder(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "LOCK_HEARTBEAT_S", 0.05)
    lock = tmp_path / "run.lock"
    with ingest.hold_lock(tmp_path, "ingest"):
        os.utime(lock, (time.time() - 3600, time.time() - 3600))
        time.sleep(0.5)
        assert time.time() - lock.stat().st_mtime < 60
    assert not lock.exists()
    with ingest.hold_lock(tmp_path, "ingest"):
        lock.write_text(json.dumps({"token": "someone else's"}), encoding="utf-8")
    assert json.loads(lock.read_text(encoding="utf-8")) == {"token": "someone else's"}


def test_a_sigterm_unwinds_the_invocation_records_it_and_frees_the_lock(tmp_path):
    class Terminated(FakeEngine):
        async def add_episode(self, **arguments):
            os.kill(os.getpid(), signal.SIGTERM)
            await asyncio.sleep(5)

    episodes = [episode(1, kind="create")]
    harness = Harness(tmp_path, episodes)

    async def open_engine(usage_log, key_file):
        engine = Terminated(usage_log, harness.stored, {})
        harness.engines.append(engine)
        return engine

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(ingest.run(harness.stream, "H+", harness.out, resume=False, images=IMAGES,
                               open_engine=open_engine, preflight=None,
                               key_file=str(harness.key_file), delays=(0.0, 0.0),
                               progress=harness.messages.append))
    (invocation,) = harness.record()["invocations"]
    assert (invocation["exit"], invocation["crash"]) == (ingest.EXIT_CRASHED, "CancelledError")
    assert harness.engines[0].closed and not (harness.out / "run.lock").exists()


def test_a_crash_is_recorded_in_run_json_as_exit_4(tmp_path, monkeypatch, capsys):
    episodes = [episode(1, kind="create")]
    harness = Harness(tmp_path, episodes)
    with pytest.raises(KeyError):
        harness.run(resume=False, broken=True)
    (crash,) = harness.record()["invocations"]
    assert (crash["exit"], crash["crash"]) == (ingest.EXIT_CRASHED, "KeyError")
    assert crash["elapsed_s"] is not None and harness.engines[0].closed

    async def crashing(*args, **kwargs):
        raise RuntimeError("boom")

    # Whatever escapes main beyond a refusal ends the process with exit status 4.
    monkeypatch.setattr(ingest, "run", crashing)
    monkeypatch.setenv("ASOFBENCH_IMAGE_IDS",
                       ",".join(f"{name}={value}" for name, value in IMAGES.items()))
    assert ingest.main(["--stream", str(harness.stream), "--arm", "H+", "--out",
                        str(harness.out)]) == ingest.EXIT_CRASHED
    assert "crashed (RuntimeError)" in capsys.readouterr().err


def test_run_json_is_written_whole_or_not_at_all(tmp_path, monkeypatch):
    path = tmp_path / "run.json"
    ingest.write_json(path, {"version": 1})
    ingest.write_json(path, {"version": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 2}
    assert [entry.name for entry in tmp_path.iterdir()] == ["run.json"]

    def interrupted(source, target):
        raise OSError("interrupted before the rename")

    monkeypatch.setattr(ingest.os, "replace", interrupted)
    with pytest.raises(OSError):
        ingest.write_json(path, {"version": 3})
    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 2}


def test_the_operator_can_skip_an_episode_that_keeps_stopping_the_run(tmp_path):
    episodes = [episode(n, kind="create" if n == 1 else "modify") for n in range(1, 6)]
    ids = [e["episode_id"] for e in episodes]
    harness = Harness(tmp_path, episodes)
    stuck = {ids[1]: [ServiceUnavailable("each time")], ids[2]: [ServiceUnavailable("too")]}
    assert harness.run(resume=False, outcomes=stuck) == ingest.EXIT_STOPPED
    assert harness.run(resume=True, outcomes=stuck) == ingest.EXIT_STOPPED
    # Only on a resume, and only the first episode of its group with no log line, then the next.
    with pytest.raises(ingest.Refused, match="needs --resume"):
        harness.run(resume=False, skip=[ids[1]])
    with pytest.raises(ingest.Refused, match="not the first pep episode with no log line"):
        harness.run(resume=True, skip=[ids[2]])
    with pytest.raises(ingest.Refused, match="not the first pep episode with no log line"):
        harness.run(resume=True, skip=[ids[2], ids[1]])
    assert len(harness.record()["invocations"]) == 2
    assert harness.run(resume=True, skip=[ids[1], ids[2]]) == ingest.EXIT_FAILED
    skipped = harness.log()[1:3]
    assert [(line["episode_id"], line["ok"], line["error"], line["attempts"], line["episode_uuid"],
             line["invocation"]) for line in skipped] == [
        (ids[1], False, ingest.SKIPPED, 0, None, 3), (ids[2], False, ingest.SKIPPED, 0, None, 3)]
    assert harness.record()["invocations"][-1]["skipped"] == [ids[1], ids[2]]
    resumed = harness.engines[-1].calls
    assert [call["name"] for call in resumed] == ids[3:]
    # The window after the skipped episodes holds only ingested ones.
    assert resumed[0]["previous_episode_uuids"] == [f"uuid-{ids[0]}"]
    assert harness.summary()["failed"] == 2


def test_remove_episodes_refuses_what_it_must_and_records_its_invocation(tmp_path):
    episodes = [episode(1, kind="create"), episode(2)]
    ids = [e["episode_id"] for e in episodes]
    harness = Harness(tmp_path, episodes)
    with pytest.raises(ingest.Refused, match="holds no run"):
        harness.remove(["any-uuid"])
    assert harness.run(resume=False, outcomes={ids[1]: ["orphan"]}) == ingest.EXIT_STOPPED
    ingested = harness.log()[0]["episode_uuid"]
    with pytest.raises(ingest.Refused, match="records .* as ingested"):
        harness.remove([ingested])
    assert len(harness.record()["invocations"]) == 1
    assert harness.remove([f"uuid-{ids[1]}"]) == ingest.EXIT_INGESTED
    assert harness.engines[-1].removed == [f"uuid-{ids[1]}"] and harness.engines[-1].closed
    invocation = harness.record()["invocations"][-1]
    assert (invocation["command"], invocation["uuids"], invocation["removed"],
            invocation["exit"], invocation["images"]) == (
        "remove-episodes", [f"uuid-{ids[1]}"], [f"uuid-{ids[1]}"], 0, IMAGES)
    assert invocation["elapsed_s"] is not None
    removal = harness.usage()[-1]
    assert (removal["context"], removal["invocation"], removal["attempt"]) == (
        f"remove-episode:{ids[1]}", 2, None)
    assert harness.remove(["not-in-the-store"]) == ingest.EXIT_STOPPED
    assert harness.record()["invocations"][-1]["exit"] == ingest.EXIT_STOPPED
    # The resume attempts the episode again; the removal's calls are counted apart from the
    # ingestion attempts', which every attempt's calls make up.
    assert harness.run(resume=True) == ingest.EXIT_INGESTED
    summary = harness.summary()
    assert summary["removal"] == {"input_tokens": 500, "output_tokens": 50, "calls": 1,
                                  "api_errors": 0}
    assert (summary["input_tokens"], summary["calls"]) == (3 * 3000, 6)
    assert summary["elapsed_s"] == pytest.approx(sum(
        invocation["elapsed_s"] for invocation in harness.record()["invocations"]))


def by_outcome(ingested: int = 0, failed: int = 0, stopped: int = 0, unknown: int = 0) -> dict:
    return {"ingested": ingested, "failed": failed, "stopped": stopped, "unknown": unknown}


def errors_by_outcome(summary: dict) -> dict:
    """The calls that raised in the attempts of each outcome, from summary.json's split."""
    return {outcome: totals["api_errors"] for outcome, totals in summary["by_outcome"].items()}


def test_a_call_that_raised_inside_an_ingested_attempt_is_counted_as_ingested(tmp_path):
    episodes = [episode(1, kind="create"), episode(2)]
    harness = Harness(tmp_path, episodes)
    assert harness.run(resume=False, outcomes={episodes[1]["episode_id"]: ["swallowed"]}) == 0
    summary = harness.summary()
    assert (summary["ok"], summary["api_errors"], summary["calls"]) == (2, 1, 4)
    assert errors_by_outcome(summary) == by_outcome(ingested=1)
    (failed,) = [line for line in harness.usage() if line["error"]]
    assert (failed["context"], failed["invocation"], failed["attempt"]) == (
        episodes[1]["episode_id"], 1, 1)
    assert (failed["error"], failed["input_tokens"], failed["output_tokens"]) == (
        "OverloadedError", 0, 0)


def test_calls_that_raised_inside_failed_or_stopped_attempts_are_reported_apart(tmp_path):
    episodes = [episode(1, kind="create"), episode(2), episode(3)]
    ids = [e["episode_id"] for e in episodes]
    harness = Harness(tmp_path, episodes)
    # The second episode's first attempt fails on a call that the API refused, and its second
    # ingests it; the third episode's first attempt stops the run on a call that stayed
    # overloaded.
    assert harness.run(resume=False, outcomes={ids[1]: ["bad-request"],
                                               ids[2]: ["overloaded"]}) == ingest.EXIT_STOPPED
    assert errors_by_outcome(harness.summary()) == by_outcome(failed=1, stopped=1)
    # The resume ingests the third episode in its own first attempt. That attempt has the
    # stopped one's number, 1, in another invocation, so it does not take the stopped one's
    # failed call as its own.
    assert harness.run(resume=True) == ingest.EXIT_INGESTED
    summary = harness.summary()
    assert (summary["ok"], summary["api_errors"]) == (3, 2)
    assert errors_by_outcome(summary) == by_outcome(failed=1, stopped=1)
    assert [(line["episode_id"], line["invocation"], line["attempts"])
            for line in harness.log()] == [(ids[0], 1, 1), (ids[1], 1, 2), (ids[2], 2, 1)]
    assert [(line["context"], line["invocation"], line["attempt"], line["error"])
            for line in harness.usage() if line["error"]] == [
        (ids[1], 1, 1, "BadRequestError"), (ids[2], 1, 1, "OverloadedError")]
    # Every line names the attempt it belongs to.
    assert sorted({(line["context"], line["invocation"], line["attempt"])
                   for line in harness.usage()}) == sorted(
        [(ids[0], 1, 1), (ids[1], 1, 1), (ids[1], 1, 2), (ids[2], 1, 1), (ids[2], 2, 1)])


def test_summary_splits_the_attempts_tokens_and_calls_by_the_outcome_of_their_attempt(tmp_path):
    episodes = [episode(1, kind="create"), episode(2), episode(3)]
    ids = [e["episode_id"] for e in episodes]
    harness = Harness(tmp_path, episodes)
    # The second episode's first attempt fails and its second ingests it; the third's first
    # attempt stops the run on a call that stayed overloaded, and the resume ingests it.
    assert harness.run(resume=False, outcomes={ids[1]: [ValueError("bad output")],
                                               ids[2]: ["overloaded"]}) == ingest.EXIT_STOPPED
    assert harness.run(resume=True) == ingest.EXIT_INGESTED
    summary = harness.summary()
    # Each attempt of the fake makes two calls, of 1,000 and 2,000 input tokens and a tenth of
    # that out; the stopped one also makes the call that raised.
    attempt = {"input_tokens": 3000, "output_tokens": 300, "calls": 2, "api_errors": 0}
    assert summary["by_outcome"] == {
        "ingested": {name: 3 * value for name, value in attempt.items()},
        "failed": attempt, "stopped": {**attempt, "api_errors": 1},
        "unknown": dict.fromkeys(attempt, 0)}
    for name in attempt:
        assert summary[name] == sum(totals[name] for totals in summary["by_outcome"].values())
    assert "api_errors_by_outcome" not in summary


def test_a_call_is_placed_in_its_attempt_by_the_invocation_and_the_number_together():
    e = episode(1, kind="create")
    logged = {e["episode_id"]: log_line(e, True, "u1", attempts=2, invocation=2)}

    def outcome(invocation, attempt, context=e["episode_id"]):
        line = {**usage_line(context, 0, 0, invocation=invocation, attempt=attempt),
                "error": "BadRequestError"}
        return ingest.attempt_outcome(line, logged)

    assert [outcome(2, 2), outcome(2, 1), outcome(1, 1), outcome(1, 3)] == [
        "ingested", "failed", "stopped", "stopped"]
    assert outcome(1, 1, context=episode(2)["episode_id"]) == "stopped"
    # A line with no attempt, or one the log cannot have made, is placed nowhere.
    assert [outcome(None, None), outcome(2, None), outcome(2, 0), outcome(2, 3), outcome(3, 1),
            outcome(True, 1)] == ["unknown"] * 6


def test_summary_counts_ingestion_attempts_apart_from_removal_and_unattributed_calls():
    episodes = [episode(1, kind="create")]
    eid = episodes[0]["episode_id"]
    failed_call = {**usage_line(eid, 0, 0, invocation=1, attempt=1), "error": "BadRequestError"}
    lines = [usage_line(eid, 100, 10, invocation=1, attempt=1), failed_call,
             usage_line(eid, 50, 5, created=5, invocation=1, attempt=1),
             usage_line(f"remove-episode:{eid}", 1000, 100, invocation=2),
             {**usage_line(f"remove-episode:{eid}", 0, 0, invocation=2),
              "error": "RateLimitError"},
             usage_line(None, 7, 1)]
    summary = ingest.summarize(episodes, [log_line(episodes[0], True, "u1")], lines,
                               record_of([1.0]), graphiti_core="0.31.2")
    assert {key: summary[key] for key in ("input_tokens", "output_tokens", "calls",
                                          "api_errors")} == {
        "input_tokens": 155, "output_tokens": 15, "calls": 2, "api_errors": 1}
    assert errors_by_outcome(summary) == by_outcome(ingested=1)
    assert summary["by_outcome"]["ingested"] == {"input_tokens": 155, "output_tokens": 15,
                                                 "calls": 2, "api_errors": 1}
    assert summary["removal"] == {"input_tokens": 1000, "output_tokens": 100, "calls": 1,
                                  "api_errors": 1}
    assert summary["unattributed"] == {"input_tokens": 7, "output_tokens": 1, "calls": 1,
                                       "api_errors": 0}
    assert summary["exit"] == 0
