# Systems

This directory holds what runs the systems that v0.1 enters ([section 11](../PREREGISTRATION.md#11-systems-model-and-runs)): each registered system's adapter and committed configuration, in a directory of its own; the code the adapters share, in `common/`; the run environment they all run in, in `env/`; and the tests of all three, in `tests/`. A system's committed configuration records its network policy (the routes out and the services inside, and that no provider-side tool is on) and what was installed.

## The environment rule

Every system runs in an environment from which no network route leads out except to the answering model's API and, for a system whose store a language model builds, the construction model's API, with the embedding model and every other service the system runs itself, such as a graph database, inside it ([section 8](../PREREGISTRATION.md#8-arms-and-the-dated-revision-stream)). From that environment neither data/v0.1, nor any file of this repository other than the system's own adapter and configuration, nor the corpora's live sources, their mirrors or a clone of them can be reached, and no provider-side tool is on. The environment is built first, its software, model weights and database included, and the rule holds from the first episode a store ingests or the first item a system answers, whichever comes first ([Entering a system](../docs/entering-a-system.md)).

## The run environment

A system runs under Docker on two networks: `inner`, an internal network with no route out, which holds the system's container and the services it runs itself, and `outer`, which only the egress proxy joins. The proxy is the one route out.

- `env/egress/` is the egress proxy, `proxy.py`, which uses only the standard library and listens on port 3128. It opens a tunnel only for `CONNECT host:port` to an entry of its allowlist, `EGRESS_ALLOW` (comma-separated `host:port` entries that name hosts by DNS name, by default `api.anthropic.com:443`), comparing the host in lower case and the port exactly. It refuses an IP address, even one that an allowed name resolves to, and any other host or port with 403, any other method, an absolute-form `GET http://...` among them, with 405, a malformed request with 400 and a request head over 16 KiB with 431, and it closes a tunnel that neither side has used for 300 seconds. It logs one JSON line per request on stdout, with the keys `ts`, `client`, `method`, `target`, `allowed` and `reason`, and nothing of what passes through a tunnel.
- `env/embedder/` is the embedding server, `server.py`, which serves voyage-4-nano (`voyageai/voyage-4-nano` at revision 5fb34e870f49f19e79c94d5e118f5ee730133279) on the CPU in float32, at its default 2048 dimensions, on port 8080. `POST /v1/embeddings` is OpenAI-compatible and answers two model names: `voyage-4-nano-query` embeds with the model's `query` prompt, for a question, and `voyage-4-nano-document` with its `document` prompt, for every text a system stores, in `float` or `base64` (little-endian float32) encoding; `GET /v1/models` lists the two, and `GET /health` names the model and revision served. The image's build installs `requirements.txt`, which pins every package with its hashes (torch's CPU-only wheel from the PyTorch CPU index, every other package from PyPI), downloads the model at its revision with `fetch_model.py`, which fails the build unless the weights have the registered size and SHA-256, and then, with no network at all, loads the model and checks the vectors it gives one text under each prompt. The server loads the model offline when it starts.
- `env/netcheck.py` is the network check, which a system's container runs before it ingests or answers anything, and which uses only the standard library. Through the proxy, each `--allow` target must open, complete a TLS handshake with its certificate verified and answer `GET /v1/models`, sent with no key, with 401, and each `--deny` target must be refused with 403; each `--direct` target, connected to without the proxy, must be out of reach; each `--services` target must accept a connection; and /proc/mounts may show mounts only at the `--mounts` paths and beneath them, besides the container's own root, the three files under /etc that Docker manages and the kernel's own filesystems at and beneath /proc, /sys and /dev. No `--allow` or `--deny` host may resolve from inside the container, since the proxy resolves the hosts it reaches and a name that resolves there is a DNS route out, while each `--services` name must resolve, through Docker's embedded DNS. The gateway check fails on any default route, IPv4 or IPv6, and, when there is none, tries every TCP port from 1 to 65535 on each gateway candidate (the next hop of each route through a gateway, the first host address of each subnet the container is on, which Docker gives a network's gateway, Docker Desktop's host addresses 192.168.65.1 and 192.168.65.254, the default bridge's gateway 172.17.0.1 and each `--gateway` address), sparing only the container's own addresses and those of the proxy and the services, and fails on any port that accepts a connection. It writes every check and its result to the `--report` file and exits 0 only when every check passes.

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
