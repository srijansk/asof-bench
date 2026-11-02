# Graphiti

This directory holds the Graphiti adapter's ingestion path and the two host-side steps of M0, the Graphiti cost pilot. Graphiti ([arXiv:2501.13956](https://arxiv.org/abs/2501.13956)) is entered self-hosted with telemetry off, as `graphiti`, or as `graphiti-reduced` if M0's projection exceeds the budget ([section 11](../../PREREGISTRATION.md#11-systems-model-and-runs), [section 13](../../PREREGISTRATION.md#13-gates)). It runs in the environment that [systems/README.md](../README.md) describes, and `config.json` records its configuration, its network policy and what is installed. The answering path, which searches the store for a question, is added before any scored run.

## What the adapter does

`ingest` reads one arm's dated revision stream and adds each episode to Graphiti, which builds its temporal knowledge graph in Neo4j with the construction model, claude-haiku-5-5. Before it ingests anything it runs the network check (`env/netcheck.py`) and stops unless every check passes. Then it builds Graphiti as `adapter.py` configures it, checks that the Neo4j store holds exactly the episodes that its log records as ingested (none, for a new run), and adds the episodes, one group per corpus: within a group strictly in stream order, one at a time, each awaited before the next, as Graphiti requires, and the two groups concurrently. No episode is ever added after a later one of its group, so that no episode's extraction can read a revision recorded after it. Every model response is recorded, under the id of the episode being ingested, in a usage log.

## The configuration

- graphiti-core 0.31.2 (tag commit d72407dcedac911e1be9fc15b55e8c05a3efcc36) with its `anthropic` extra, and every package it installs at the version that graphiti-core's own `uv.lock` pins at that release. That lock pins the anthropic SDK at 0.111.0, of its 0.x line: Graphiti's `AnthropicClient` passes `temperature` on every call, and anthropic 1.0.0 removed that parameter from `messages.create`, so with anthropic 1.0 or later every call it makes would raise `TypeError`. The SDK at 0.111.0 sends the API version 2023-06-01.
- Neo4j 5.26 community (5.26.32, pinned by digest), with no authentication on the internal network, a heap of 256 MB growing to 512 MB and a page cache of 128 MB, its usage report off, restarted if it stops (`restart: unless-stopped`), and a named volume per run, so that each arm has its own store and arms never share one.
- The construction model is Graphiti's own `AnthropicClient` with `LLMConfig(model="claude-haiku-5-5", small_model="claude-haiku-5-5")` over the shared client of `common/asofbench_systems/llm.py`, which sends to `https://api.anthropic.com`, given in code so that `ANTHROPIC_BASE_URL` is never read, with the key read only from the Docker secret `/run/secrets/anthropic_api_key`, through the SDK's own httpx client, which honours `HTTPS_PROXY` and `NO_PROXY` (inside the environment, the egress proxy), with up to 8 retries and 600 seconds a request. Its `temperature` is the SDK's omit sentinel, `anthropic.omit`, so that no request carries `temperature`, `top_p` or `top_k`. Everything else is Graphiti's own code path: its prompts, its forced tool call and its own `max_tokens` (16,384, its default). The usage log wraps `client.messages.create` and records every response.
- The embedder is Graphiti's `OpenAIEmbedder` at `http://embedder:8080/v1`, the environment's voyage-4-nano server, with `api_key="none"` and `embedding_dim=2048`, and the model name `voyage-4-nano-document`, which selects the model's document prompt, for every text it embeds while ingesting: the texts it stores and the name searches it makes to deduplicate them. It sends no `dimensions` and keeps all 2048 floats. The answering path embeds questions with `voyage-4-nano-query`.
- The cross-encoder is `NoReranker`, whose `rank` raises `RuntimeError("no reranking model is registered for asof-bench")`. While ingesting, Graphiti looks for duplicate edges with `EDGE_HYBRID_SEARCH_RRF`, which fuses BM25 and cosine similarity by reciprocal rank, and for duplicate nodes by exact names and a direct cosine-similarity search, so nothing ranks with a model.
- An episode's `group_id` is its corpus (`pep` or `eth`). Each proposal has one saga, named by its id, and each episode is given, as `previous_episode_uuids`, the uuids of the same proposal's last 3 ingested episodes (Graphiti's `EPISODE_WINDOW_LEN`), all recorded before it, oldest first, none for its first episode, and as `saga_previous_episode_uuid` the last of them. Graphiti's `get_by_uuids` does not order the episodes it returns, so the window's order inside Graphiti's prompt is Graphiti's own. An episode's name is its `episode_id`, its body the one episode template every store system ingests (`asofbench_systems.stream.render_episode`), its source `EpisodeType.text`, its source description `<repo> commit <commit>` and its reference time its `recorded_at` in UTC. There are no entity or edge types, no custom extraction instructions and no community update.
- Failures. Graphiti's own retries and the SDK's apply first. An episode that still raises is attempted twice more in place, after 30 seconds and then 120, with the same previous episodes and saga predecessor; one that still raises is logged as failed and is never attempted again, by the run or a resumed one. Five logged failures in a row in a group stop the run (exit status 3). An infrastructure error stops the run at once (exit status 2) and leaves the episode unlogged, so that a resumed run attempts it again: Neo4j unavailable or a session lost (`ServiceUnavailable`, `SessionExpired`, `SessionError`, `DatabaseUnavailable`); the embedder unreachable or failing (openai's `APIConnectionError`, `InternalServerError`); and, after the SDK's own retries, the API unreachable, rate-limited, overloaded or failing, or refusing the key (anthropic's `APIConnectionError`, `RateLimitError`, `OverloadedError`, `InternalServerError`, `AuthenticationError`, `PermissionDeniedError`, and Graphiti's own `RateLimitError`, which wraps the SDK's). A failed attempt that left the episode in the store stops the run the same way. A stop lets each group finish the attempt it is making and starts nothing more.
- Telemetry is off (`GRAPHITI_TELEMETRY_ENABLED=false`). `SEMAPHORE_LIMIT` is not set, so Graphiti's default of 20 holds.

## Building the image

The image installs `requirements.txt` hash-checked and from wheels only, copies the adapter, the shared package and the network check one by one, and ends with a self-test that runs with no network: it checks every installed package's version against `config.json` and that Graphiti builds as configured.

```bash
docker build -t asofbench/graphiti -f systems/graphiti/Dockerfile systems
```

`requirements.txt` is compiled from `requirements.in` by the command its header records, run in this directory with uv 0.7.2. The egress proxy's and the embedding server's images are built as [systems/README.md](../README.md) says; `compose.yaml` can also build all three.

## Running an ingestion

`compose.yaml` runs the egress proxy, the embedding server, Neo4j and the Graphiti container, each with a memory limit. Give each run its own project name, so that its store is a volume of its own, and four variables. Three are absolute paths, since compose reads a relative one from the directory of `compose.yaml`, and every compose command on the file needs them, `down` included: a directory that holds the run's input stream and no other stream (mounted read-only at `/input`), the run's output directory (`/output`) and the file that holds the API key, which reaches the container only as a Docker secret. The fourth, `ASOFBENCH_IMAGE_IDS`, gives the IDs of the four images the run uses, which the run's outputs record and which the container cannot read for itself; `ingest` refuses to start without it.

```bash
image_id() { docker image inspect --format '{{.Id}}' "$1"; }
export ASOFBENCH_RUN_INPUT="$PWD/../asofbench-runs/m0/input"
export ASOFBENCH_RUN_OUTPUT="$PWD/../asofbench-runs/m0/output"
export ASOFBENCH_ANTHROPIC_KEY_FILE="$HOME/.config/asofbench/anthropic_api_key"
export ASOFBENCH_IMAGE_IDS="graphiti=$(image_id asofbench/graphiti),egress=$(image_id asofbench/egress),embedder=$(image_id asofbench/embedder),neo4j=$(image_id neo4j@sha256:249a1ec6e3c52d82101a5ed33d6ed2b5ed5b0017b21a28c0bfebc335e87fb177)"
docker compose -f systems/graphiti/compose.yaml -p asofbench-graphiti-m0 run --rm graphiti \
    ingest --stream /input/m0-h-plus.jsonl --arm H+ --out /output
```

It writes, in the output directory, and none of these files holds a local path:

- `run.json`, written when the run begins: the input stream's file name and SHA-256 and the arm, and for each invocation its start, whether it resumed, the image IDs it ran with, its elapsed time and its exit status;
- `netcheck.json`, the network check's report; a later invocation keeps the earlier one as `netcheck-<n>.json`;
- `usage.jsonl`, one line per model response: `ts`, `context` (the episode's id), `model`, `input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens` and `stop_reason`;
- `ingest-log.jsonl`, one line per episode, after its last attempt: `episode_id`, `proposal`, `group_id`, `episode_uuid` (null for a failed episode), `ok`, `error` (null when its first attempt succeeded, else the last failed attempt's exception class and the first 300 characters of its message, with the key taken out), `started_at` (its first attempt), `wall_s` (its attempts' time, without the waits between them), `text_chars` (its text in code points) and `attempts`;
- `summary.json`: `arm`, `stream` and `stream_sha256`, `episodes`, `ok`, `failed`, `text_chars`, `input_tokens` (with the cache-creation and cache-read tokens), `output_tokens`, `calls`, `calls_over_100k_input` (prompts beyond the 100,000 tokens up to which the price basis's prices hold), `wall_s`, `elapsed_s`, `invocations`, `graphiti_core`, `model` and `images`, over the whole log and every usage line, each counted once. `wall_s` is the sum of every episode's attempt time, so the two groups' time counts apart and it can exceed the run's duration; `elapsed_s` is the run's wall-clock time, the elapsed times of its invocations summed. `model` is the model that answered every usage line, or the list of them when there are several. `images` gives each image's IDs over the invocations.

The exit status is 0 when every episode is ingested, 1 when the run reached the stream's end with an episode logged as failed, 2 when the run is refused before it starts or the network check fails or an infrastructure error stops it, and 3 when five failures in a row in a group stop it. After a status of 2 or 3, run the same command with `--resume`. It checks that the stream and the arm are those the run began with, and continues each group from its first episode with no log line, after rebuilding each proposal's window and saga predecessor from the log; an episode logged as failed is not attempted again. Without `--resume`, `ingest` refuses an output directory that holds a run and a store that holds any episode.

Recovering from a stop:

- If Neo4j stops or restarts during a run (Docker restarts it if it dies), the run stops with exit status 2 and leaves the episode it was ingesting unlogged. Once `docker compose -f systems/graphiti/compose.yaml -p <run> ps` shows Neo4j healthy again, resume: the run continues from that episode. The same holds when the embedder or the API is unavailable, the API rate-limits or is overloaded, or the network check fails.
- Before it continues, a resumed run checks the store. If the stop came while Graphiti was writing the episode, the store holds an episode that the log does not record, and the resume refuses, naming its uuid. Remove it with Graphiti's own `remove_episode`, then resume. This command refuses an episode the log records as ingested, and records the model calls it makes in the run's usage log, under `remove-episode:<episode id>`, since `remove_episode` rebuilds the summaries of the entities the episode mentioned:

  ```bash
  docker compose -f systems/graphiti/compose.yaml -p <run> run --rm graphiti \
      python -m graphiti.adapter --remove-episodes <uuid> --out /output
  ```

- After exit status 3, read the five failures' `error` in the ingest log before you resume: a resume counts on from the log, so one more failure stops it again at once.

When the run is done, `docker compose -f systems/graphiti/compose.yaml -p <run> down -v` removes its containers, its networks and its store.

## M0

M0 ingests the H+ episodes of the 50 PEPs that pilot.json lists for it and projects the cost of every ingestion that Graphiti's entries need ([section 13](../../PREREGISTRATION.md#13-gates)). `m0.py` runs on the host and uses only the standard library.

1. Prepare the input, outside the checkout, since the episodes hold proposal text. pilot.json and the stream are checked against the SHA-256s the manifest registers first, and the step reports 50 proposals, 1,441 episodes and 3,651,707 characters of text. It writes `m0-h-plus.jsonl` and `prepare.json`, with `proposals`, `episodes`, `text_chars`, `stream_sha256` (the H+ stream's) and `input_sha256` (m0-h-plus.jsonl's):

   ```bash
   python systems/graphiti/m0.py prepare --pilot data/v0.1/pilot.json --stream streams/h-plus.jsonl \
       --manifest data/v0.1/MANIFEST.json --out-dir ../asofbench-runs/m0/input
   ```

2. Ingest `m0-h-plus.jsonl` with the command above.
3. Project. The step checks that the summary is a run of claude-haiku-5-5 alone over the prepared input (its stream's SHA-256 is prepare.json's `input_sha256`, and its episodes and characters are prepare.json's), and that prepare.json, questions.jsonl and the three local streams are the registered ones. The rates are the run's tokens over prepare.json's characters, applied to the H+, H- and CF streams and section 14's incremental path, whose characters are the H+ stream's again, each stream's characters taken from the manifest and priced at its price basis. When the total exceeds the budget, the decision is `graphiti-reduced`, and k is the largest number of other proposals, in the registered SHA-256 order, whose projected total, with the proposals that the F1, F2 and CF items name, fits the budget, each proposal's characters taken from the three local streams. When the named proposals alone exceed the budget, which the registration does not provide for, the projection is still written, with k null and a `reason`, and the step exits with status 1:

   ```bash
   python systems/graphiti/m0.py project --summary ../asofbench-runs/m0/output/summary.json \
       --prepare ../asofbench-runs/m0/input/prepare.json --manifest data/v0.1/MANIFEST.json \
       --questions data/v0.1/questions.jsonl --streams-dir streams --out ../asofbench-runs/m0/projection.json
   ```

`projection.json` records each step: the rates, each ingestion's characters, tokens and cost, the total, the budget, the decision, k with the candidates chosen, the price basis, each input file by its name and SHA-256, and the command line with each path reduced to its file name, so that it holds no local path.

## What is published

Beside the results report go M0's `projection.json`, with the pilot's two rates, the projected total and, when the budget binds, k, and each run's `run.json`, `summary.json`, `usage.jsonl` and `netcheck.json`, which hold ids, counts, sizes, hashes, image IDs and timings only, and no local path. The ingest log stays with the run's outputs, since an error's message can quote the model's output, and so does the Neo4j store, which holds proposal text, as do the input streams.

## Tests

`systems/tests/test_ingest_resume.py` checks the ingestion loop with a fake Graphiti (the order, the retries, the stops and the resume), and `config.json` against the adapter and the lock; `test_m0.py` checks both M0 steps on synthetic files; `test_stream_render.py` checks the episode template. They use only the standard library and run on the repository's own environment. `test_llm_client.py` needs the anthropic SDK and graphiti-core, so it runs inside the image, with no network at all, every request going to a mock transport or a stand-in proxy:

```bash
docker run --rm --network none -v "$PWD/systems:/systems:ro" asofbench/graphiti \
    python -m pytest -p no:cacheprovider /systems/tests
```
