# Rebuilding the data

How to rebuild data/v0.1, its manifest and the three streams from the public histories, and check them byte for byte. [PREREGISTRATION.md](../PREREGISTRATION.md) registers every rule this page describes, and is the rule wherever the two seem to differ.

## Requirements

- Python 3.11 or later. The generator, the stream builder, the materialiser and the scorer use only the standard library, so every line that decides a label can be read and audited.
- [uv](https://docs.astral.sh/uv/), which every command below runs through; `uv run` installs the package from this checkout.
- git 2.32 or later (the generator keeps your git configuration out of its reads with `GIT_CONFIG_GLOBAL`, which git 2.32 introduced).
- About 450 MB of disk for the three clones, and about 110 MB for each arm's stream (see [Scale and cost](#scale-and-cost)).

## Getting the clones

The generator reads full, non-shallow clones of the three repositories. Put them outside this checkout:

```bash
git clone https://github.com/python/peps ../asofbench-clones/peps
git clone https://github.com/ethereum/EIPs ../asofbench-clones/EIPs
git clone https://github.com/ethereum/ERCs ../asofbench-clones/ERCs
```

Leave each clone's HEAD on its default branch (`main` for peps, `master` for EIPs and ERCs), where `git clone` leaves it. The generator walks the first-parent chain from HEAD and takes the last commit recorded by the end of S as the pin, so HEAD must be the default branch at or after the pin, or the pinned commit itself; a HEAD on another branch, or before the pin, gives another pin ([section 2](../PREREGISTRATION.md#2-corpora-and-pins)). The pins are in data/v0.1/MANIFEST.json (`pins`). The generator refuses a shallow or partial (`--filter`) clone, a directory that is not the top level of its own clone, and, given a registered manifest, a clone whose pin differs from it.

## Rebuilding the gold

`asofbench build` reads each clone up to its pin and writes the questions, gold, pilot list, mention-pair sample, timelines, relation edges, build report and `MANIFEST.json`, and with `--streams` the three arms' streams ([section 15](../PREREGISTRATION.md#15-manifest)). Build into a fresh directory outside the checkout, never over `data/v0.1`: `asofbench build` refuses, with exit status 2 and before it writes anything, an `--out` directory that already holds a MANIFEST.json, as data/v0.1 and every earlier build do, or that is the directory of `--expect-pins`, so remove `../asofbench-rebuild` before you build into it again. This command, with the registered Graphiti budget and its price basis (in single quotes, so that the shell does not expand `$0`), reproduces data/v0.1. The price basis's source is the [pricing page](https://platform.claude.com/docs/en/about-claude/pricing) of Anthropic, the provider of the answering models (claude-haiku-5-5 and claude-sonnet-5-5), which, fetched on 2026-10-31, gives Claude Haiku 5.5's input as "$0.10 / MTok" "for prompts up to 100,000 tokens" and its output as "$0.50 / MTok", the prices the basis records.

```bash
uv run asofbench build --clones ../asofbench-clones --out ../asofbench-rebuild \
    --streams ../asofbench-rebuild/streams --expect-pins data/v0.1/MANIFEST.json \
    --graphiti-budget-usd 500.0 \
    --price-basis 'claude-haiku-5-5 list prices on 2026-10-31 (Claude API, prompts up to 100,000 tokens): $0.10 per million input tokens, $0.50 per million output tokens'
```

`--expect-pins` refuses clones whose pins differ from the registered manifest's. A rebuild matches when its seven generated files equal those in data/v0.1 byte for byte and its `MANIFEST.json` equals the registered one apart from `generator.source_commit` and `generator.source_dirty`, which describe the checkout rather than the build (section 15). The manifest also records each stream's episode count, characters, estimated tokens and SHA-256, and the size and SHA-256 of each file beside it (sections 8 and 15), so a matching manifest means matching streams. To check:

```bash
for f in questions.jsonl gold.jsonl pilot.json timelines.jsonl edges.jsonl labelling/mention_pairs.jsonl REPORT.md; do
    cmp data/v0.1/$f ../asofbench-rebuild/$f && echo "$f is identical"
done
uv run python - data/v0.1/MANIFEST.json ../asofbench-rebuild/MANIFEST.json <<'EOF'
import json, sys
registered, rebuilt = (json.load(open(path)) for path in sys.argv[1:])
for manifest in (registered, rebuilt):
    del manifest["generator"]["source_commit"], manifest["generator"]["source_dirty"]
print("MANIFEST.json matches" if registered == rebuilt else "MANIFEST.json differs")
EOF
```

## Building the streams

A system reads its arm's stream, which is rebuilt from the clones because it holds the proposals' text. Without rebuilding the gold, build the three streams against the published data:

```bash
uv run asofbench stream --clones ../asofbench-clones --data data/v0.1 --arm H+ --out streams/h-plus.jsonl
uv run asofbench stream --clones ../asofbench-clones --data data/v0.1 --arm H- --out streams/h-minus.jsonl
uv run asofbench stream --clones ../asofbench-clones --data data/v0.1 --arm CF --out streams/cf.jsonl
```

Each command refuses clones whose pins differ from those of data/v0.1/MANIFEST.json, and writes the stream only if it, and every file it writes beside it, match what the manifest registers. Beside each stream it writes `<stream>.stats.json`; for H- also `<stream>.message_cues.json` and for CF `<stream>.cf_messages.json`, which the scorer reads. A system receives the stream file only, never the files beside it: the message-cues file lists, for each proposal, the statuses that kept messages name and when, and the CF counts list the CF proposals. `asofbench build --streams DIR` writes the same files into DIR. The `streams/` directory is git-ignored.

A stream is a JSONL file of episodes in recorded-time order, one per proposal per first-parent commit, with the fields `episode_id`, `corpus`, `repo`, `proposal`, `ref`, `recorded_at`, `commit`, `message`, `kind` (create, modify, rename or delete) and `text`: the full file for a create, a unified diff for a modify or a rename, and nothing for a delete (section 8).

## Scale and cost

Measured on the pinned clones and on a rebuild of v0.1:

- The three clones take about 450 MB of disk.
- Each arm's stream holds 26,486 episodes. H+ and CF hold about 98.0 million characters of episode text and H- 97.0 million, about 24.5 and 24.2 million estimated tokens (characters divided by four, as the manifest records them), and each file is about 110 MB.
- A rebuild with `--streams` took about 35 seconds, and each `asofbench stream` command about 12 seconds.
- A system entered in every arm and family writes 4,830 answer rows (875 H+, 705 H- and 30 CF items, three runs each); long context, entered in F1, F2 and CF only, writes 3,600.
- Grep at D needs 868 trees (475 dates in H+, 364 in H- and 29 in CF). Each takes a few seconds to materialise, and the H+ tree at S holds 1,974 files, about 32 MB, with a git-log.txt of 17,955 commits, about 4.4 MB.
- Long context reads, per run, about 10.3 million estimated tokens in H+, 10.2 million in H- and 1.1 million in CF (the named proposals' episode text and messages, divided by four). Its largest prompt is about 263,000 estimated tokens, and the prompts of 10 items (8 asked in both H+ and H-, and 2 CF items) exceed 100,000, past the prompt size the recorded price basis covers.

Not measured yet: Graphiti's ingestion cost. It is held to the M0 budget of 500 US dollars, and M0 has not run, so no projection exists. Before any scored run, M0 measures Graphiti's token rates on 50 PEPs and projects the cost of every ingestion Graphiti's entries need at claude-haiku-5-5's list prices of 2026-10-31, the day the budget was set, as the manifest records them; if the projection exceeds the budget, Graphiti is entered as `graphiti-reduced`.
