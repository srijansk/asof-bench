"""Graphiti as asof-bench configures it (PREREGISTRATION.md section 11; config.json records the
same configuration, and README.md describes it).

- graphiti-core 0.31.2 over Neo4j 5.26 community, reached at ``bolt://neo4j:7687`` with no
  authentication (``NEO4J_AUTH=none``, on the environment's internal network only).
- Construction model: Graphiti's own ``AnthropicClient`` with ``LLMConfig(model="claude-haiku-5-5",
  small_model="claude-haiku-5-5")`` and the client from ``asofbench_systems.llm``, which sends to
  https://api.anthropic.com with the key from the Docker secret; its ``temperature`` is the SDK's
  omit sentinel, so no request carries one. Everything else is Graphiti's own code path: its
  prompts, its forced tool call and its own ``max_tokens``. Every response is recorded in the
  run's usage log under the episode being ingested.
- Embedder: Graphiti's ``OpenAIEmbedder`` at ``http://embedder:8080/v1``, the environment's
  voyage-4-nano server, with ``embedding_dim=2048`` and the model name
  ``voyage-4-nano-document`` (the model's document prompt) for every text it embeds while
  ingesting; the answering path embeds questions with ``voyage-4-nano-query``.
- Cross-encoder: ``NoReranker``, whose ``rank`` raises, since no system reranks with a model.
- Episodes: ``group_id`` is the episode's corpus; one saga per proposal, named by its id; the
  previous episodes are the same proposal's last ``EPISODE_WINDOW_LEN`` ingested episodes, all
  recorded before it, oldest first, and the saga predecessor the last of them, a deliberate
  departure from ``add_episode``'s own default, which reads the group's 10 most recent episodes;
  the episode's text is ``asofbench_systems.stream.render_episode``; no entity or edge types, no
  custom extraction instructions, no community update.

The module's constants and ``episode_arguments`` use only the standard library and the shared
package; graphiti-core and anthropic are imported only inside the functions that build Graphiti,
so the module imports where neither is installed. ``python -m graphiti.adapter --self-test``,
which the image's build runs with no network, checks the installation against config.json, and
``python -m graphiti.adapter --remove-episodes <uuid>... --out <run's output>`` removes episodes
that an attempt left in the store, which a resumed run refuses (README.md).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from asofbench_systems import llm, usage
from asofbench_systems.stream import render_episode

MODEL = "claude-haiku-5-5"
NEO4J_URI = "bolt://neo4j:7687"
EMBEDDER_URL = "http://embedder:8080/v1"
EMBEDDING_DIM = 2048
DOCUMENT_MODEL = "voyage-4-nano-document"
QUERY_MODEL = "voyage-4-nano-query"
# The number of the same proposal's previous episodes that each episode is given: 3, the value of
# Graphiti's EPISODE_WINDOW_LEN (graphiti_core.utils.maintenance.graph_data_operations), which
# the self-test checks. Graphiti's own add_episode default, when no previous_episode_uuids is
# given, would read the group's 10 most recent episodes (RELEVANT_SCHEMA_LIMIT); the adapter's
# per-proposal, oldest-first window departs from that default on purpose (README.md).
EPISODE_WINDOW_LEN = 3
# Graphiti's DEFAULT_SEMAPHORE_LIMIT, left in force: SEMAPHORE_LIMIT is not set.
SEMAPHORE_LIMIT = 20
NO_RERANKER = "no reranking model is registered for asof-bench"
CONFIG = Path(__file__).resolve().with_name("config.json")
# How long open_engine waits for Neo4j to answer, and how often it asks.
NEO4J_WAIT_S, NEO4J_RETRY_S = 180.0, 2.0
EPISODIC_QUERY = "MATCH (e:Episodic) RETURN e.uuid AS uuid, e.name AS name, e.group_id AS group_id"


def parse_utc(recorded_at: str) -> datetime:
    """An episode's ``recorded_at``, written ``YYYY-MM-DDTHH:MM:SSZ``, as a datetime in UTC."""
    return datetime.strptime(recorded_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def episode_arguments(e: dict, *, previous_episode_uuids: list[str],
                      saga_previous_episode_uuid: str | None, source) -> dict:
    """The keyword arguments of ``Graphiti.add_episode`` for the episode ``e``: ``source`` is
    ``EpisodeType.text``, given by the caller so that this function needs no graphiti-core."""
    return {"name": e["episode_id"], "episode_body": render_episode(e), "source": source,
            "source_description": f"{e['repo']} commit {e['commit']}",
            "reference_time": parse_utc(e["recorded_at"]), "group_id": e["corpus"],
            "update_communities": False, "previous_episode_uuids": list(previous_episode_uuids),
            "saga": e["proposal"], "saga_previous_episode_uuid": saga_previous_episode_uuid}


def make_llm_client(client):
    """Graphiti's AnthropicClient over ``client`` (an ``anthropic.AsyncAnthropic``), with the
    construction model and no temperature."""
    from graphiti_core.llm_client.anthropic_client import AnthropicClient
    from graphiti_core.llm_client.config import LLMConfig

    config = LLMConfig(model=MODEL, small_model=MODEL, temperature=llm.omit_sentinel())
    return AnthropicClient(config=config, client=client)


def make_embedder(model: str = DOCUMENT_MODEL, client=None):
    """Graphiti's OpenAIEmbedder for the environment's voyage-4-nano server: ``model`` names the
    prompt (``voyage-4-nano-document`` for stored texts, ``voyage-4-nano-query`` for questions).
    ``client``, an ``openai.AsyncOpenAI``, is for tests only."""
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

    config = OpenAIEmbedderConfig(embedding_model=model, api_key="none", base_url=EMBEDDER_URL,
                                  embedding_dim=EMBEDDING_DIM)
    return OpenAIEmbedder(config=config, client=client)


def make_no_reranker():
    """A CrossEncoderClient whose ``rank`` raises: asof-bench registers no reranking model."""
    from graphiti_core.cross_encoder.client import CrossEncoderClient

    class NoReranker(CrossEncoderClient):
        async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
            raise RuntimeError(NO_RERANKER)

    return NoReranker()


class GraphitiEngine:
    """Graphiti as configured, with what ingest.py needs of it: ``add_episode``, which passes its
    arguments to ``Graphiti.add_episode`` unchanged and returns the new episode's uuid, the
    episodes in the store, ``remove_episode`` and ``close``."""

    def __init__(self, graphiti, source, version: str) -> None:
        self.graphiti, self.source, self.version = graphiti, source, version

    async def add_episode(self, **arguments) -> str:
        result = await self.graphiti.add_episode(**arguments)
        return result.episode.uuid

    async def stored_episodes(self) -> list[dict]:
        """Every Episodic node in the store, as ``{"uuid", "name", "group_id"}``."""
        result = await self.graphiti.driver.execute_query(EPISODIC_QUERY)
        return [record.data() for record in result.records]

    async def remove_episode(self, uuid: str) -> None:
        """Graphiti's own remove_episode."""
        await self.graphiti.remove_episode(uuid)

    async def close(self) -> None:
        await self.graphiti.close()


async def open_engine(usage_log: usage.UsageLog, key_file: str = llm.KEY_FILE) -> GraphitiEngine:
    """Graphiti on the environment's services, with every model response recorded in
    ``usage_log``; it waits for Neo4j to answer, then builds Graphiti's indices."""
    from graphiti_core import Graphiti
    from graphiti_core.driver.neo4j_driver import Neo4jDriver
    from graphiti_core.nodes import EpisodeType

    installed = check_installation()
    client = usage.instrument(llm.make_async_client(key_file), usage_log)
    driver = Neo4jDriver(NEO4J_URI, None, None)
    graphiti = Graphiti(graph_driver=driver, llm_client=make_llm_client(client),
                        embedder=make_embedder(), cross_encoder=make_no_reranker())
    deadline = time.monotonic() + NEO4J_WAIT_S
    while True:
        try:
            await driver.client.verify_connectivity()
            break
        except Exception:
            if time.monotonic() > deadline:
                await graphiti.close()
                raise
            await asyncio.sleep(NEO4J_RETRY_S)
    # Graphiti's Neo4j driver schedules its index build when it is made. Let that finish (it
    # fails if Neo4j was not answering yet), then build once more, so that the two never race.
    scheduled = getattr(driver, "_init_task", None)
    if scheduled is not None:
        with contextlib.suppress(Exception):
            await scheduled
    await graphiti.build_indices_and_constraints()
    return GraphitiEngine(graphiti, EpisodeType.text, installed["graphiti-core"])


def first_step(coroutine) -> BaseException | None:
    """Runs ``coroutine`` to its first suspension, in or out of a running event loop, and returns
    the exception it raises there, or None; the coroutine is closed either way."""
    try:
        coroutine.send(None)
    except StopIteration:
        return None
    except Exception as error:  # noqa: BLE001 - the exception is the result
        return error
    finally:
        coroutine.close()
    return None


def recorded_packages(config: dict) -> dict[str, str]:
    """The Python packages, and their versions, that config.json records as installed."""
    return dict(config["installed"]["python_packages"])


def check_installation(config_path: Path = CONFIG) -> dict[str, str]:
    """Checks, with no network, that the installed packages are those config.json records and
    that Graphiti builds as configured; returns the installed versions. RuntimeError naming each
    difference otherwise."""
    import importlib.metadata
    import inspect

    import anthropic
    from graphiti_core import helpers
    from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS
    from graphiti_core.telemetry import is_telemetry_enabled
    from graphiti_core.utils.maintenance import graph_data_operations

    config = json.loads(config_path.read_text(encoding="utf-8"))
    problems, installed = [], {}
    for name, version in recorded_packages(config).items():
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed[name] = None
        if installed[name] != version:
            problems.append(f"{name} is {installed[name]}, config.json records {version}")
    if (window := graph_data_operations.EPISODE_WINDOW_LEN) != EPISODE_WINDOW_LEN:
        problems.append(f"Graphiti's EPISODE_WINDOW_LEN is {window}")
    if helpers.SEMAPHORE_LIMIT != SEMAPHORE_LIMIT:
        problems.append(f"Graphiti's SEMAPHORE_LIMIT is {helpers.SEMAPHORE_LIMIT}")
    if DEFAULT_MAX_TOKENS != config["llm"]["max_tokens"]:
        problems.append(f"Graphiti's DEFAULT_MAX_TOKENS is {DEFAULT_MAX_TOKENS}")
    if is_telemetry_enabled():
        problems.append("Graphiti's telemetry is on (GRAPHITI_TELEMETRY_ENABLED is not false)")
    create = anthropic.AsyncAnthropic(api_key="self-test", base_url=llm.BASE_URL).messages.create
    if "temperature" not in inspect.signature(create).parameters:
        problems.append("the anthropic SDK's messages.create takes no temperature, which "
                        "Graphiti's AnthropicClient passes on every call")
    with tempfile.TemporaryDirectory() as scratch:
        key_file = Path(scratch) / "key"
        key_file.write_text("self-test-key\n", encoding="ascii")
        graphiti_llm = make_llm_client(llm.make_async_client(str(key_file)))
    if graphiti_llm.temperature is not llm.omit_sentinel():
        problems.append("the AnthropicClient's temperature is not the SDK's omit sentinel")
    if (graphiti_llm.model, graphiti_llm.small_model) != (MODEL, MODEL):
        problems.append(f"the AnthropicClient's models are {graphiti_llm.model}, "
                        f"{graphiti_llm.small_model}")
    embedder = make_embedder()
    if (embedder.config.embedding_dim, embedder.config.embedding_model) != (EMBEDDING_DIM,
                                                                            DOCUMENT_MODEL):
        problems.append(f"the embedder is {embedder.config.embedding_model} at "
                        f"{embedder.config.embedding_dim} dimensions")
    raised = first_step(make_no_reranker().rank("query", ["passage"]))
    if not isinstance(raised, RuntimeError) or str(raised) != NO_RERANKER:
        problems.append(f"NoReranker.rank gave {raised!r}, not RuntimeError({NO_RERANKER!r})")
    if problems:
        raise RuntimeError("; ".join(problems))
    return installed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Graphiti as asof-bench configures it.",
                                     allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true",
                      help="check the installation against config.json, with no network")
    mode.add_argument("--remove-episodes", nargs="+", metavar="UUID",
                      help="remove episodes that an attempt left in the store (README.md)")
    parser.add_argument("--out", type=Path,
                        help="with --remove-episodes: the run's output directory")
    args = parser.parse_args(argv)
    if args.remove_episodes:
        if args.out is None:
            parser.error("--remove-episodes needs --out, the run's output directory")
        # ingest.py drives the run's state: its lock, its run.json and its network check.
        from graphiti import ingest

        try:
            images = ingest.parse_image_ids(os.environ.get("ASOFBENCH_IMAGE_IDS"))
            return asyncio.run(ingest.remove_episodes(args.remove_episodes, args.out,
                                                      images=images))
        except ingest.Refused as refusal:
            print(f"remove-episodes: {refusal}", file=sys.stderr)
            return ingest.EXIT_STOPPED
        except SystemExit:
            raise
        except BaseException as error:  # noqa: BLE001 - a crash: recorded in run.json, exit 4
            return ingest.crashed(error, llm.KEY_FILE, "remove-episodes")
    try:
        installed = check_installation()
    except RuntimeError as error:
        print(f"graphiti adapter self-test: FAIL: {error}", file=sys.stderr)
        return 1
    print(f"graphiti adapter self-test: {len(installed)} packages as config.json records them, "
          f"graphiti-core {installed['graphiti-core']}, model {MODEL}, embedder {DOCUMENT_MODEL} "
          f"at {EMBEDDING_DIM} dimensions, no temperature, no reranker")
    return 0


if __name__ == "__main__":
    sys.exit(main())
