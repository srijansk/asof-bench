# M0: the Graphiti cost pilot

M0 is the cost pilot of [section 13](../../PREREGISTRATION.md#13-gates): before any scored run, Graphiti ingests the H+ episodes of the 50 PEPs that pilot.json lists for it, the pilot measures Graphiti's LLM input and output tokens per character of episode text, and the projection applies the two rates to every ingestion that Graphiti's entries need. The projected total is $728.35 against the budget of $500, so Graphiti is entered as `graphiti-reduced`, with k = 728.

## Result

The rates are 14.5147 input tokens and 0.8239 output tokens per character, measured on the 1,441 episodes (3,651,707 characters of text) of the 50 PEPs. Applied to each ingestion, at the price basis that the manifest records (Claude Haiku 5.5's list prices on 2026-10-31, $0.10 per million input tokens and $0.50 per million output tokens for prompts up to 100,000 tokens; no call of the run went beyond 100,000 input tokens):

| Ingestion | Characters | Projected cost |
|---|---:|---:|
| H+ | 97,964,116 | $182.55 |
| H− | 96,972,538 | $180.70 |
| CF | 97,964,438 | $182.55 |
| Incremental path (section 14; the H+ characters again) | 97,964,116 | $182.55 |
| Total | 390,865,208 | $728.35 |

The total exceeds the budget, so under section 13 Graphiti is entered as `graphiti-reduced`, in F1, F2 and CF only: in every arm it ingests the episodes of the 529 proposals that the F1, F2 and CF items name, plus the first k = 728 of the 1,453 other proposals in ascending SHA-256 of `asof-bench/v0.1|distractor|<id>`, a projected $499.99. A 729th proposal (EIP-7915) would bring the total to $500.12. projection.json lists the 728 proposals, each ingestion's characters, tokens and cost for both sets, and every input by its SHA-256.

## What the rates count

The rates count the tokens of the ingestion attempts that ingested their episode or failed: here every episode was ingested and no attempt failed, so they count 12,048 calls, 53,003,273 input tokens and 3,008,803 output tokens. They leave out 51 calls (224,122 input and 25,654 output tokens) in attempts that the run's environment interrupted, when a service it runs was ended or an operator stopped the run, and that a resumed invocation then did again: counting them would charge Graphiti twice for one episode's ingestion. That rule was written into `m0.py` (commit 4ac6c16) before this run began. Counted, those calls would give rates of 14.5760 and 0.8310, a projected $732.12 and the same decision, `graphiti-reduced`, with k = 715.

No call raised an error, no episode was removed from the store, and no call fell outside an ingestion attempt. Every call asked for `claude-haiku-5-5`, and every response named that model. The ingestion attempts took 18,254 seconds in all, 12.7 seconds per episode on average.

## How it was run

The run used commit 1401bbc of this repository, with Graphiti 0.31.2 as [systems/graphiti](../../systems/graphiti/README.md) configures it, Neo4j 5.26.32 pinned by digest, and the embedding server serving voyage-4-nano in bfloat16. The image IDs are graphiti `sha256:339b3f23770d9b767b6ea73b52addd4fc31175af9848c3c87662d5c298d416fc`, egress `sha256:54cd0411c5ca1e769d85a912b73b3d0f58e652811bbc3d1f127df049fc030db6`, embedder `sha256:ef77a8be8a9c1cadc78a1fa299f3186c9425b6604a075e04fc1de5aefbc70c7b` and neo4j `sha256:249a1ec6e3c52d82101a5ed33d6ed2b5ed5b0017b21a28c0bfebc335e87fb177`; run.json records them for each invocation.

It ran on 2026-11-02 from 05:18:55 to 10:30:20 UTC, in six invocations. Before each one, the network check passed all 30 of its checks (netcheck.json, and netcheck-1.json to netcheck-5.json for the earlier invocations). The egress proxy's log, egress.log, records 5,675 tunnels, every one to `api.anthropic.com:443`, and 48 refused requests, which are the network check's probes of eight denied hosts at each invocation.

| Invocation | Started (UTC) | Exit | Why it ended |
|---|---|---:|---|
| 1 | 05:18:55 | 2 | The embedding server was ended and restarted (`APIConnectionError`), most likely by the Docker VM's out-of-memory killer: the VM's swap was full and the server had grown to about 1.5 GiB; resumed |
| 2 | 06:34:22 | 2 | The same; resumed |
| 3 | 07:07:57 | 4 | An operator's `docker stop`, right after an episode was logged, to restart the embedding server with the allocator settings below |
| 4 | 07:47:22 | 2 | Neo4j, which ran at its 1 GiB memory limit, restarted during a commit (`ServiceUnavailable`); the resume found no episode in the store that the log does not record |
| 5 | 08:43:41 | 4 | An operator's `docker stop`, right after an episode was logged, to raise Neo4j's memory limit |
| 6 | 08:49:56 | 0 | Every episode ingested |

From invocation 4 on, the run used operational overrides that change memory settings only: the embedding server's allocator settings (two glibc arenas and a fixed 128 KiB mmap threshold, overrides/invocations-4-5.yaml) and, from invocation 6, also Neo4j's container memory limit, raised from 1 GiB to 1.5 GiB with its heap and page cache unchanged (overrides/invocation-6.yaml). No model, prompt, weight, dtype or vector changed. The Docker VM had 7.75 GiB of memory, shared with other workloads.

## Commands

From the repository's root, with the local streams built as [docs/rebuilding.md](../../docs/rebuilding.md#building-the-streams) says. The input, prepared outside the checkout:

```bash
python systems/graphiti/m0.py prepare --pilot data/v0.1/pilot.json --stream streams/h-plus.jsonl \
    --manifest data/v0.1/MANIFEST.json --out-dir ../asofbench-runs/m0/input
```

The ingestion, as [systems/graphiti/README.md](../../systems/graphiti/README.md#running-an-ingestion) runs it (with its four variables set), then the same command with `--resume` after each stop, and from invocation 4 on with `-f` and the override file added to every compose command:

```bash
docker compose -f systems/graphiti/compose.yaml -p asofbench-graphiti-m0 run --rm graphiti \
    ingest --stream /input/m0-h-plus.jsonl --arm H+ --out /output
```

The projection:

```bash
python systems/graphiti/m0.py project --summary ../asofbench-runs/m0/output/summary.json \
    --prepare ../asofbench-runs/m0/input/prepare.json --manifest data/v0.1/MANIFEST.json \
    --questions data/v0.1/questions.jsonl --streams-dir streams --out ../asofbench-runs/m0/projection.json
```

## Files

This directory holds projection.json, the run's summary.json, run.json, usage.jsonl (one line per model call: ids, counts, models and timings), prepare.json, the network checks, egress.log and the two override files. It leaves out the input stream and the Neo4j store, which hold proposal text, and ingest-log.jsonl, whose `error` field can quote a model's output (it holds no error in this run). Before publication, every file here was checked to hold no string of 20 or more characters, and no run of 56 or more, from any episode's text or commit message, no local path and no API key.

## The abandoned first attempt

A first M0 run began on 2026-11-02 at 03:05 UTC, on commit 8c00483's images, with the embedding server serving voyage-4-nano upcast to float32. The embedding server was ended twice when the Docker VM ran out of memory (its swap full and about 450 MB free), and the run had ingested 35 episodes (353 calls, 1,400,907 input and 84,864 output tokens) when it was abandoned, before anything was projected. The embedding server was then changed to serve the model in bfloat16, the dtype its weights are published in (commit 4e41daf; on 84 test texts its vectors agree with float32's to a cosine of at least 0.99988), and because vectors of two dtypes must not mix in one store, M0 started again on a fresh store. That attempt's run.json, summary.json, usage.jsonl and network checks are in abandoned-float32/; its egress log was not kept.
