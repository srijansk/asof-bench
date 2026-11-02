# Graphiti

This directory holds the Graphiti adapter's ingestion path and the two host-side steps of M0, the Graphiti cost pilot. Graphiti ([arXiv:2501.13956](https://arxiv.org/abs/2501.13956)) is entered self-hosted with telemetry off, as `graphiti`, or as `graphiti-reduced` if M0's projection exceeds the budget ([section 11](../../PREREGISTRATION.md#11-systems-model-and-runs), [section 13](../../PREREGISTRATION.md#13-gates)). It runs in the environment that [systems/README.md](../README.md) describes, and `config.json` records its configuration, its network policy and what is installed. The answering path, which searches the store for a question, is not here yet.

## What the adapter does

`ingest` reads one arm's dated revision stream and adds each episode to Graphiti, which builds its temporal knowledge graph in Neo4j with the construction model, claude-haiku-5-5. Before it ingests anything it runs the network check (`env/netcheck.py`) and stops unless every check passes. Then it builds Graphiti as `adapter.py` configures it, checks that the Neo4j store holds exactly the episodes that its log records as ingested (none, for a new run), and adds the episodes, one group per corpus: within a group one at a time in stream order, each awaited before the next, as Graphiti requires, and the two groups concurrently. Every model response is recorded, under the id of the episode being ingested, in a usage log.

## The configuration

- graphiti-core 0.31.2 (tag commit d72407dcedac911e1be9fc15b55e8c05a3efcc36) with its `anthropic` extra, and every package it installs at the version that graphiti-core's own `uv.lock` pins at that release. That lock pins the anthropic SDK at 0.111.0, of its 0.x line: Graphiti's `AnthropicClient` passes `temperature` on every call, and anthropic 1.0.0 removed that parameter from `messages.create`, so with anthropic 1.0 or later every call it makes would raise `TypeError`. The SDK at 0.111.0 sends the API version 2023-06-01.
- Neo4j 5.26 community (5.26.32, pinned by digest), with no authentication on the internal network, a heap of 256 MB growing to 512 MB and a page cache of 128 MB, its usage report off, and a named volume per run, so that each arm has its own store and arms never share one.
- The construction model is Graphiti's own `AnthropicClient` with `LLMConfig(model="claude-haiku-5-5", small_model="claude-haiku-5-5")` over the shared client of `common/asofbench_systems/llm.py`, which sends to `https://api.anthropic.com`, given in code so that `ANTHROPIC_BASE_URL` is never read, with the key read only from the Docker secret `/run/secrets/anthropic_api_key`, through the SDK's own httpx client, which honours `HTTPS_PROXY` and `NO_PROXY` (inside the environment, the egress proxy), with up to 8 retries and 600 seconds a request. Its `temperature` is the SDK's omit sentinel, `anthropic.omit`, so that no request carries `temperature`, `top_p` or `top_k`. Everything else is Graphiti's own code path: its prompts, its forced tool call and its own `max_tokens` (16,384, its default). The usage log wraps `client.messages.create` and records every response.
- The embedder is Graphiti's `OpenAIEmbedder` at `http://embedder:8080/v1`, the environment's voyage-4-nano server, with `api_key="none"` and `embedding_dim=2048`, and the model name `voyage-4-nano-document`, which selects the model's document prompt, for every text it embeds while ingesting: the texts it stores and the name searches it makes to deduplicate them. It sends no `dimensions` and keeps all 2048 floats. The answering path embeds questions with `voyage-4-nano-query`.
- The cross-encoder is `NoReranker`, whose `rank` raises `RuntimeError("no reranking model is registered for asof-bench")`. While ingesting, Graphiti looks for duplicate edges with `EDGE_HYBRID_SEARCH_RRF`, which fuses BM25 and cosine similarity by reciprocal rank, and for duplicate nodes by exact names and a direct cosine-similarity search, so nothing ranks with a model.
- An episode's `group_id` is its corpus (`pep` or `eth`). Each proposal has one saga, named by its id, and each episode is given the saga's previous episode (`saga_previous_episode_uuid`) and, as `previous_episode_uuids`, the uuids of the same proposal's last 3 ingested episodes (Graphiti's `EPISODE_WINDOW_LEN`), oldest first, none for its first episode. Its name is its `episode_id`, its body the one episode template every store system ingests (`asofbench_systems.stream.render_episode`), its source `EpisodeType.text`, its source description `<repo> commit <commit>` and its reference time its `recorded_at` in UTC. There are no entity or edge types, no custom extraction instructions and no community update.
- Graphiti's own retries and the SDK's apply; an episode that still raises is logged as failed and the run goes on. Telemetry is off (`GRAPHITI_TELEMETRY_ENABLED=false`). `SEMAPHORE_LIMIT` is not set, so Graphiti's default of 20 holds.

## Building the image

The image installs `requirements.txt` hash-checked and from wheels only, copies the adapter, the shared package and the network check one by one, and ends with a self-test that runs with no network: it checks every installed package's version against `config.json` and that Graphiti builds as configured.

```bash
docker build -t asofbench/graphiti -f systems/graphiti/Dockerfile systems
```

`requirements.txt` is compiled from `requirements.in` by the command its header records, run in this directory with uv 0.7.2. The egress proxy's and the embedding server's images are built as [systems/README.md](../README.md) says; `compose.yaml` can also build all three.

## Running an ingestion

`compose.yaml` runs the egress proxy, the embedding server, Neo4j and the Graphiti container, each with a memory limit. Give each run its own project name, so that its store is a volume of its own, and three variables, each an absolute path, since compose reads a relative one from the directory of `compose.yaml`: a directory that holds the run's input stream and no other stream (mounted read-only at `/input`), the run's output directory (`/output`) and the file that holds the API key, which reaches the container only as a Docker secret. Every compose command on the file needs the three, `down` included.

```bash
export ASOFBENCH_RUN_INPUT="$PWD/../asofbench-runs/m0/input"
export ASOFBENCH_RUN_OUTPUT="$PWD/../asofbench-runs/m0/output"
export ASOFBENCH_ANTHROPIC_KEY_FILE="$HOME/.config/asofbench/anthropic_api_key"
docker compose -f systems/graphiti/compose.yaml -p asofbench-graphiti-m0 run --rm graphiti \
    ingest --stream /input/m0-h-plus.jsonl --arm H+ --out /output
```

It writes, in the output directory:

- `netcheck.json`, the network check's report; a later run keeps the earlier one as `netcheck-<n>.json`;
- `usage.jsonl`, one line per model response: `ts`, `context` (the episode's id), `model`, `input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens` and `stop_reason`;
- `ingest-log.jsonl`, one line per episode attempt: `episode_id`, `proposal`, `group_id`, `episode_uuid` (null for a failed attempt), `ok`, `error` (null, or the exception's class and the first 300 characters of its message, with the key taken out), `started_at`, `wall_s` and `text_chars`, the episode's text in code points;
- `summary.json`: `arm`, `episodes`, `ok`, `failed`, `text_chars`, `input_tokens` (with the cache-creation and cache-read tokens), `output_tokens`, `calls`, `calls_over_100k_input` (prompts beyond the 100,000 tokens up to which the price basis's prices hold), `wall_s` (the sum of the attempts' wall times), `graphiti_core` and `model`, over the whole log and every usage line, each counted once.

If a run stops partway, or some episodes failed, run the same command with `--resume`: it attempts only the episodes the log does not record as ingested, in stream order, after rebuilding each proposal's saga predecessor and window from the log, and it appends to the logs. Without `--resume`, `ingest` refuses an output directory that holds a run and a store that holds any episode. A resumed run refuses a store that lacks an episode the log records, or holds one it does not, such as one an attempt wrote before it failed or was stopped. The exit status is 0 when every episode is ingested, 1 when one is not or the network check fails, and 2 when the run is refused before it starts. When the run is done, `docker compose -f systems/graphiti/compose.yaml -p asofbench-graphiti-m0 down -v` removes its containers, its networks and its store.

## M0

M0 ingests the H+ episodes of the 50 PEPs that pilot.json lists for it and projects the cost of every ingestion that Graphiti's entries need ([section 13](../../PREREGISTRATION.md#13-gates)). `m0.py` runs on the host and uses only the standard library.

1. Prepare the input, outside the checkout, since the episodes hold proposal text. The stream is checked against the manifest's SHA-256 first, and the step reports 50 proposals, 1,441 episodes and 3,651,707 characters of text:

   ```bash
   python systems/graphiti/m0.py prepare --pilot data/v0.1/pilot.json --stream streams/h-plus.jsonl \
       --manifest data/v0.1/MANIFEST.json --out-dir ../asofbench-runs/m0/input
   ```

2. Ingest `m0-h-plus.jsonl` with the command above.
3. Project. The rates are the run's tokens over prepare.json's characters, applied to the H+, H- and CF streams and section 14's incremental path, whose characters are the H+ stream's again, each stream's characters taken from the manifest and priced at its price basis. When the total exceeds the budget, the decision is `graphiti-reduced`, and k is the largest number of other proposals, in the registered SHA-256 order, whose projected total, with the proposals that the F1, F2 and CF items name, fits the budget, each proposal's characters taken from the three local streams after each is checked against the manifest:

   ```bash
   python systems/graphiti/m0.py project --summary ../asofbench-runs/m0/output/summary.json \
       --prepare ../asofbench-runs/m0/input/prepare.json --manifest data/v0.1/MANIFEST.json \
       --questions data/v0.1/questions.jsonl --streams-dir streams --out ../asofbench-runs/m0/projection.json
   ```

`projection.json` records each step: the rates, each ingestion's characters, tokens and cost, the total, the budget, the decision, k with the candidates chosen, the price basis, each input file's SHA-256 and the command line as typed.

## What is published

Beside the results report go M0's `projection.json`, with the pilot's two rates, the projected total and, when the budget binds, k, and each run's `summary.json`, `usage.jsonl` and `netcheck.json`, which hold ids, counts, sizes, hashes and timings only. The ingest log stays with the run's outputs, since an error's message can quote the model's output, and so does the Neo4j store, which holds proposal text, as do the input streams.

## Tests

`systems/tests/test_ingest_resume.py` checks the ingestion loop with a fake Graphiti, and `config.json` against the adapter and the lock; `test_m0.py` checks both M0 steps on synthetic files; `test_stream_render.py` checks the episode template. They use only the standard library and run on the repository's own environment. `test_llm_client.py` needs the anthropic SDK and graphiti-core, so it runs inside the image, with no network at all, every request going to a mock transport or a stand-in proxy:

```bash
docker run --rm --network none -v "$PWD/systems:/systems:ro" asofbench/graphiti \
    python -m pytest -p no:cacheprovider /systems/tests
```
