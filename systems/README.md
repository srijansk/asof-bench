# Systems

This directory holds what runs the systems that v0.1 enters ([section 11](../PREREGISTRATION.md#11-systems-model-and-runs)): each registered system's adapter and committed configuration, in a directory of its own; the code the adapters share, in `common/`; the run environment they all run in, in `env/`; the answering harness, `harness.py`, and the reader image in which the single-call entrants answer, in `reader/`; and the tests of all of them, in `tests/`. A system's committed configuration records its network policy (the routes out and the services inside, and that no provider-side tool is on) and what was installed.

## The environment rule

Every system runs in an environment from which no network route leads out except to the answering model's API and, for a system whose store a language model builds, the construction model's API, with the embedding model and every other service the system runs itself, such as a graph database, inside it ([section 8](../PREREGISTRATION.md#8-arms-and-the-dated-revision-stream)). From that environment neither data/v0.1, nor any file of this repository other than the system's own adapter and configuration, nor the corpora's live sources, their mirrors or a clone of them can be reached, and no provider-side tool is on. The environment is built first, its software, model weights and database included, and the rule holds from the first episode a store ingests or the first item a system answers, whichever comes first ([Entering a system](../docs/entering-a-system.md)).

## The run environment

A system runs under Docker on two networks: `inner`, an internal network with no route out, which holds the system's container and the services it runs itself, and `outer`, which only the egress proxy joins. The proxy is the one route out.

- `env/egress/` is the egress proxy, `proxy.py`, which uses only the standard library and listens on port 3128. It opens a tunnel only for `CONNECT host:port` to an entry of its allowlist, `EGRESS_ALLOW` (comma-separated `host:port` entries that name hosts by DNS name, by default `api.anthropic.com:443`), comparing the host in lower case and the port exactly. It refuses an IP address, even one that an allowed name resolves to, and any other host or port with 403, any other method, an absolute-form `GET http://...` among them, with 405, a malformed request with 400 and a request head over 16 KiB with 431, and it closes a tunnel that neither side has used for its idle timeout, `EGRESS_IDLE_TIMEOUT`, by default 660 seconds: a minute longer than the 600 seconds that the systems' shared Anthropic client (`common/asofbench_systems/llm.py`) gives a request, since a request made without streaming receives nothing until its response begins, and a tunnel closed sooner would cut a request that the provider may complete and bill, which the client would then send again. It logs one JSON line per request on stdout, with the keys `ts`, `client`, `method`, `target`, `allowed` and `reason`, and nothing of what passes through a tunnel. Its stated limits: it relays TLS uninspected, so it checks the host and port of the `CONNECT` request and never the server name (SNI) that the client's TLS handshake sends inside the tunnel, which is therefore not checked against the `CONNECT` host; and it has no cap on concurrent connections.
- `env/embedder/` is the embedding server, `server.py`, which serves voyage-4-nano (`voyageai/voyage-4-nano` at revision 5fb34e870f49f19e79c94d5e118f5ee730133279) on the CPU in bfloat16, the dtype its weights are published in, at its default 2048 dimensions, on port 8080, and returns each vector in float32, which holds the bfloat16 values exactly. `POST /v1/embeddings` is OpenAI-compatible and answers two model names: `voyage-4-nano-query` embeds with the model's `query` prompt, for a question, and `voyage-4-nano-document` with its `document` prompt, for every text a system stores, in `float` or `base64` (little-endian float32) encoding; `GET /v1/models` lists the two, and `GET /health` names the model, the revision and the dtypes served. The image's build installs `requirements.txt`, which pins every package with its hashes (torch's CPU-only wheel from the PyTorch CPU index, every other package from PyPI), downloads the model at its revision with `fetch_model.py`, which fails the build unless the weights have the registered size and SHA-256, and then, with no network at all, loads the model, checks that its weights are bfloat16, and checks the vectors it gives one text under each prompt. The server loads the model offline when it starts, and refuses to start unless its weights are bfloat16.
- `env/netcheck.py` is the network check, which a system's container runs before it ingests or answers anything, and which uses only the standard library. Through the proxy, each `--allow` target must open, complete a TLS handshake with its certificate verified and answer `GET /v1/models`, sent with no key, with 401, and each `--deny` target must be refused with 403; each `--direct` target, connected to without the proxy, must be out of reach; each `--services` target must accept a connection; and /proc/mounts may show mounts only at the `--mounts` paths and beneath them, besides the container's own root, the three files under /etc that Docker manages and the kernel's own filesystems at and beneath /proc, /sys and /dev. No `--allow` or `--deny` host may resolve from inside the container, since the proxy resolves the hosts it reaches and a name that resolves there is a DNS route out, while each `--services` name must resolve, through Docker's embedded DNS. The gateway check fails on any default route, IPv4 or IPv6, and, when there is none, tries every TCP port from 1 to 65535 on each gateway candidate (the next hop of each IPv4 route through a gateway, the first host address of each IPv4 subnet the container is on, which Docker gives a network's gateway, Docker Desktop's host addresses 192.168.65.1 and 192.168.65.254, the default bridge's gateway 172.17.0.1 and each `--gateway` address), sparing only the container's own addresses and those of the proxy and the services, and fails on any port that accepts a connection. It writes every check and its result to the `--report` file and exits 0 only when every check passes.

## Building and checking the environment

Build the two images, each with its own directory as the context:

```bash
docker build -t asofbench/egress systems/env/egress
docker build -t asofbench/embedder systems/env/embedder
```

The embedder's build downloads the packages and about 0.7 GB of model weights, and ends with the self-test. An image built with `systems/` as its context, such as a system's, has its Dockerfile copy only the files it needs, and `systems/.dockerignore` keeps the tests and caches out of that context. To check the environment by hand, start the two services on throwaway networks, wait until `docker logs asofbench-embedder` shows that the embedder is serving, and run the check from a container on `inner`, here one made from the proxy's image, which has Python. `inner` is created with no address of the host's on it (`com.docker.network.bridge.inhibit_ipv4`), since an internal network leads nowhere out but still gives the host its gateway address, where whatever the host serves on all its addresses answers: on Docker Desktop 29.6.1 the gateway check finds the RPC portmapper and `rpc.statd` of its virtual machine there.

```bash
docker network create --internal -o com.docker.network.bridge.inhibit_ipv4=true asofbench-inner
docker network create asofbench-outer
docker run -d --name asofbench-egress --network asofbench-outer asofbench/egress
docker network connect --alias egress asofbench-inner asofbench-egress
docker run -d --name asofbench-embedder --network asofbench-inner --network-alias embedder asofbench/embedder
docker create --name asofbench-netcheck --network asofbench-inner asofbench/egress python /tmp/netcheck.py \
    --proxy http://egress:3128 --allow api.anthropic.com:443 \
    --deny github.com:443,huggingface.co:443,pypi.org:443,peps.python.org:443,eips.ethereum.org:443,ercs.ethereum.org:443 \
    --deny host.docker.internal:443,gateway.docker.internal:443 \
    --direct 140.82.112.3:443,1.1.1.1:443 --services embedder:8080 --report /tmp/netcheck.json
docker cp systems/env/netcheck.py asofbench-netcheck:/tmp/netcheck.py
docker start -a asofbench-netcheck
```

It prints one line per check and its result. A system's own container runs the same check, with `--services` naming every service the system runs and `--mounts` its input, output and secret paths. Then remove the throwaway containers and networks:

```bash
docker rm -f asofbench-netcheck asofbench-egress asofbench-embedder
docker network rm asofbench-inner asofbench-outer
```

The tests need neither Docker nor the model: they run the proxy in-process against a local echo server, the embedding server's request handling with a fake model, and the network check against stand-in servers, on the repository's own environment, as CI does:

```bash
uv run --frozen --extra dev pytest systems/tests
```

`env/embedder/requirements.txt` is compiled from `requirements.in` by the command its header records, run in `env/embedder/` with uv 0.7.2.

## Answering: the harness and the reader

A system receives each item's question text and nothing else, and the harness that runs it keeps the rest ([section 8](../PREREGISTRATION.md#8-arms-and-the-dated-revision-stream)). `harness.py`, on the host, writes a run's requests, each under a random key, and keeps each key's item, arm, run and answer type to itself; a system's container answers each request; and `harness.py` turns the answers into the answer rows of [section 12](../PREREGISTRATION.md#12-answers-and-scoring), which `asofbench score` scores. The single-call entrants (closed book, and long context and the oracle, whose requests add a context) answer in the reader image, `reader/`, whose runner, `answer_requests`, makes one call per request with the registered settings of `common/asofbench_systems/answer.py`: `max_tokens` 32000, adaptive thinking, effort medium and the answer's JSON schema, with no temperature, top_p or top_k, no prefill and no tool. Each entrant's prompts, settings and network policy are in its directory's `config.json`, such as [closed-book/config.json](closed-book/config.json). Before a system's configuration is fixed, run it on the toy world ([examples/toy_world.py](../examples/toy_world.py)), never on data/v0.1 ([section 11](../PREREGISTRATION.md#11-systems-model-and-runs)).

A run directory, which `harness.py requests` makes outside the repository, holds `input/requests.jsonl`, the requests and nothing else; `output/`, where the container writes; and, beside them, `keys.jsonl` and `plan.json`, which hold each request's item and never leave the host. Mount `input/` and `output/`, never the run directory itself: the reader refuses a requests file that is not alone in its directory. Build the reader's image with `docker build`, not `docker compose build`, which would also rebuild the egress image, whose ID every run records. The key file's path and the images' IDs reach the run as in Graphiti's ([graphiti/README.md](graphiti/README.md#running-an-ingestion)); the reader refuses to start without the IDs. A pilot run of closed book, from the requests to the rows:

```bash
RUN="$PWD/../asofbench-runs/closed-book-pilot"
uv run python systems/harness.py requests --system closed-book --data data/v0.1 --out-dir "$RUN" \
    --items data/v0.1/pilot.json
docker build -t asofbench/reader -f systems/reader/Dockerfile systems
image_id() { docker image inspect --format '{{.Id}}' "$1"; }
export ASOFBENCH_RUN_INPUT="$RUN/input" ASOFBENCH_RUN_OUTPUT="$RUN/output"
export ASOFBENCH_ANTHROPIC_KEY_FILE="$HOME/.config/asofbench/anthropic_api_key"
export ASOFBENCH_IMAGE_IDS="reader=$(image_id asofbench/reader),egress=$(image_id asofbench/egress)"
docker compose -f systems/reader/compose.yaml -p asofbench-reader-pilot run --rm reader \
    answer_requests --requests /input/requests.jsonl --out /output/responses.jsonl --model claude-haiku-5-5
docker compose -f systems/reader/compose.yaml -p asofbench-reader-pilot logs --no-log-prefix egress > "$RUN/output/egress.log"
docker compose -f systems/reader/compose.yaml -p asofbench-reader-pilot down
uv run python systems/harness.py rows --system closed-book --run-dir "$RUN" --out "$RUN/answers.jsonl"
```

`answer_requests` first runs the network check, with the flags every system gives it, no services and the mounts `/input`, `/output` and `/run/secrets`, and answers nothing unless every check passes. It then answers the requests eight at a time, in the file's order, writing one response per request to `output/responses.jsonl` and every model call to `output/usage.jsonl` under the request's key, and records each invocation in `output/run.json`. Its exit status is 0 when every request has a response with an answer; 1 when every request has a response and some of them hold an error code instead of an answer (`refusal`, `max_tokens`, `unparseable`, or `api_error` for an API error that is the request's own, such as a prompt longer than the context window); 2 when it is refused before it starts, the network check fails, or an infrastructure error stops it (after the SDK's retries, the API unreachable, rate-limited, overloaded or failing, or refusing the key), and then the request under way has no response; 3 when five `api_error` responses in a row stop it; and 4 when it crashes, which run.json records with the exception's class. After a status of 2, run the same command with `--resume`: it checks that the requests file and the model are those the run began with, and answers every request that has no response, so that an outage never gives a request a null answer. A status of 3 is different: each of those `api_error` responses is its request's own and stays, but five in a row point at a cause that every request shares, such as a setting that the API rejects, and a resume answers only the requests that have no response. Such a run is discarded and run again in a new run directory once the cause is fixed, never resumed under changed settings. An invocation killed outright leaves `run.lock` in `output/`, and every later invocation exits at once with status 2, naming the lock's holder and the age of its heartbeat: check that no invocation runs (`docker compose -f systems/reader/compose.yaml -p <run> ps -a --status running` must list no `<run>-reader-run-<id>` container), then remove `run.lock` and resume.

`harness.py rows` writes one row per request with the nine fields of section 12, the answer shaped by the item's answer type and in no other way, and a null answer for a request that the system could not answer. A key with no response at all, which a run that stopped and was not resumed leaves, gets a row with a null answer and null costs, and `rows` exits 1: resume the run first. The rows carry the name that `--system` gives, the system's own or, for a toy run, an outside entry's that an entries file declares to `asofbench score` ([examples/README.md](../examples/README.md)).

The tests run without Docker and without the model. `test_harness.py` runs the harness on the toy world and scores its rows; `test_answer_requests.py` runs the reader's runner with a stand-in for the call; `test_reader_wiring.py` checks the image, the self-test's reading of a lock, the compose file, the lock and closed book's configuration against the code; and `test_answer.py` checks the answering call field by field, and the image's self-test, with the SDK that the reader's lock pins, which the repository's environment does not install, so CI runs it, with the shared client's tests, as `uv run --frozen --extra dev --with anthropic==0.78.0 --with httpx==0.28.1 python -m pytest systems/tests/test_answer.py systems/tests/test_llm_client.py`. Inside the image, with no network, all of them run as `docker run --rm --network none -v "$PWD/systems:/systems:ro" asofbench/reader python -m pytest -p no:cacheprovider /systems/tests`, where those that need git or the asofbench package skip. `reader/requirements.txt` is compiled from `requirements.in` by the command its header records, run in `reader/` with uv 0.7.2.
