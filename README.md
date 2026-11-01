# asof-bench

**What did a project decide, as of a given date?** asof-bench is a [pre-registered](#status-and-roadmap) benchmark for the memory and retrieval systems that give language-model agents their context.

[![tests](https://github.com/srijansk/asof-bench/actions/workflows/tests.yml/badge.svg)](https://github.com/srijansk/asof-bench/actions/workflows/tests.yml) [![code: MIT](https://img.shields.io/badge/code-MIT-blue)](LICENSE) [![data: CC0 1.0](https://img.shields.io/badge/data-CC0%201.0-blue)](LICENSE-DATA) [![python: 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml) [![pre-registered: prereg-v0.1.2](https://img.shields.io/badge/pre--registered-prereg--v0.1.2-green)](PREREGISTRATION.md)

[Registration](PREREGISTRATION.md) · [Data](data/) · [Results: not yet](#status-and-roadmap) · [Documentation](#documentation) · [Citation](#citation)

Memory systems for agents promise to track how facts change. This benchmark tests that promise on real decision records: 905 questions about what the Python Enhancement Proposals (PEPs) and the Ethereum Improvement Proposals (EIPs and ERCs) had recorded as decided on a past date. Every answer key is computed from their git histories, not written by a model or annotator. Each system is compared with grep at D, the same model reading the repository as it stood on the question's date D.

**Status: frozen, no results yet.** The design was frozen on 2026-11-01, before any system answered. You can rebuild every answer key and score your own system ([quick start](#quick-start)).

## An example

An *as-of* query asks for the state of a record on a past date, not today; every status question asks one. PEP 695 is not one of the 905 items, so this page holds no answer key:

*What was the status of PEP 695 ("Type Parameter Syntax") as of the end of 2023-09-30 (UTC)? Answer with exactly one value from: Draft, Active, Accepted, Provisional, Deferred, Rejected, Withdrawn, Final, Superseded.*

The answer is the `Status:` header of the PEP's file as python/peps stood at the end of that day:

| Recorded (UTC) | Commit in python/peps | Status |
|---|---|---|
| 2022-07-03 | `3fe47842` PEP 695: Type Parameter Syntax (#2697) | Draft |
| 2023-04-13 | `39644631` PEP 695: Mark as accepted (#3103) | Accepted |
| 2024-02-17 | `16c8775b` PEP 695: Mark as Final (#3675) | Final |

So the answer is **Accepted**. A system that knows only today's status answers Final, and is wrong. Where the headline is read, every status header is withheld, so a system has to recover "Accepted" from the proposal's text, the commit messages (here, "Mark as accepted") and when each change was recorded.

## Quick start

You need Python 3.11 or later, [uv](https://docs.astral.sh/uv/) and git 2.32 or later. Look at an item:

```bash
git clone https://github.com/srijansk/asof-bench && cd asof-bench
head -n 1 data/v0.1/questions.jsonl
```

```text
{"answer_type":"status","arms":["H+","H-"],"as_of":"2021-09-03",...,"question":"What was the status of PEP 643 (\"Metadata for Package Source Distributions\") as of the end of 2021-09-03 (UTC)? Answer with exactly one value from: Draft, Active, ...","stratum":"change"}
```

Run every command, the scorer included, on a toy corpus of synthetic repositories in about two seconds. Develop a system there, never on a v0.1 item ([examples](examples/README.md)):

```bash
uv run python examples/toy_world.py ../asofbench-toy
```

Rebuild the questions and answer keys from the three public histories (about 450 MB of clones) and check them byte for byte:

```bash
git clone https://github.com/python/peps ../asofbench-clones/peps
git clone https://github.com/ethereum/EIPs ../asofbench-clones/EIPs
git clone https://github.com/ethereum/ERCs ../asofbench-clones/ERCs
uv run asofbench build --clones ../asofbench-clones --out ../asofbench-rebuild --expect-pins data/v0.1/MANIFEST.json
for f in questions.jsonl gold.jsonl pilot.json timelines.jsonl edges.jsonl labelling/mention_pairs.jsonl REPORT.md; do
    cmp data/v0.1/$f ../asofbench-rebuild/$f && echo "$f is identical"
done
```

Each file prints `is identical`. Then build the H- stream that most systems read, and the tree that grep at D reads on one date, where `peps/pep-0695` has no `Status:` line and `git-log.txt` holds the commit that accepted it:

```bash
uv run asofbench stream --clones ../asofbench-clones --data data/v0.1 --arm H- --out streams/h-minus.jsonl
uv run asofbench materialize --stream streams/h-minus.jsonl --as-of 2023-09-30 --arm H- --data data/v0.1 \
    --out ../asofbench-trees/h-minus-2023-09-30
```

[docs/rebuilding.md](docs/rebuilding.md) rebuilds the manifest and the three streams as registered.

## What it measures

| Family | What it asks | Items |
|---|---|---|
| F1 | a proposal's status as of the end of date D | 503 |
| F2 | which PEPs superseded a PEP, at the snapshot date | 82 |
| F3 | which proposals an Ethereum proposal requires, directly or through the proposals it requires, at the snapshot date | 120 |
| F4 | which of the proposals an Ethereum proposal required, directly or through the proposals it required, had status Final, as of D | 120 |
| F5 | how many proposals had a given status as of D | 50 |
| CF | F1's question over a stream whose statuses were permuted, so that an answer from what a model already knows about the real proposals is wrong | 30 |

The snapshot date S is 2026-10-27. Each item is asked in one or more arms, each with its own dated revision stream: every change to every proposal file in the order the repository recorded it, with its commit message. H+ shows the headers, H- withholds the status, decision and relation headers, and CF permutes the statuses of 30 proposals.

The headline compares seven systems with **grep at D**: BM25, dense retrieval, hybrid retrieval, a Personalized PageRank retriever, HippoRAG 2, Graphiti and long context. Grep at D is the same model, given read-only shell tools over the proposal files and the commit log as they stood at the end of the date. It is read on the 300 H- F1-change items, 150 per corpus, whose answer on the date differs from the proposal's status at S. (When no question offers that status, as for Stagnant, the comparison is with its last status before S that a question offers.) Each choice has a reason:

- **Grep at D**, because checking the repository out at the date and reading it is the answer version control already gives; a system that keeps context is worth its machinery where it beats that reader.
- **F1-change**, because on these items today's status is never the answer, so knowing the present, from a store or from training, does not help.
- **H-**, because with the headers visible grep at D answers by reading the status line; with them withheld, a system has to recover the decision from the prose, the commit messages and when the changes were recorded.

A system is above or below grep at D on a corpus only when an exact McNemar test on per-item majority results over three runs gives p below 0.05, Holm-adjusted over the 14 comparisons (7 systems on 2 corpora). Only the registered full run makes that claim. One model, claude-haiku-5-5, answers for every system and builds every store a language model builds, so the comparison holds the reader fixed. Draft is the answer to 61% of the PEP and 65% of the Ethereum headline items, the floor every accuracy is read against. [docs/design.md](docs/design.md) and [docs/scoring.md](docs/scoring.md) have the full rules, and sections [7](PREREGISTRATION.md#7-question-families) and [12](PREREGISTRATION.md#12-answers-and-scoring) of the registration register them.

## Status and roadmap

A benchmark's result depends on choices: which items count, which baseline, which test, when to stop. Made after the results are in, those choices can bend toward a result. So every one of them was fixed by the annotated tag `prereg-v0.1.2` on 2026-11-01 (UTC), before any system had answered a question; the author's own retriever, `ppr`, is an entrant, so the freeze binds the author too. Before this tag, v0.1 was frozen twice in private and revised after review. One revision drew the Ethereum headline items again (74 of the 150 are new): under the first rule a move to Stagnant counted as a change, so on 80 items the proposal's last status before Stagnant was right whatever the date. [DEVIATIONS.md](DEVIATIONS.md) lists every change with its reason, and records every change after the tag.

What happens next, in order ([section 13](PREREGISTRATION.md#13-gates)):

1. The adapters and configurations of the registered systems are committed here before any of them answers a question, and their maintainers are invited to review them.
2. M0, the Graphiti cost pilot: Graphiti ingests 50 PEPs, and its measured cost projects whether every ingestion it needs fits the budget of 500 US dollars; if not, it is entered as `graphiti-reduced`, which ingests only part of each stream.
3. M1, the pilot: 180 items, six systems, three runs each. Its gates stop the comparison if every system scores 85% or more on the pilot's headline items, since the questions are then too easy. They stop it too if both graph-memory systems fail present-day questions about as often as past ones, or if grep at D and long context already answer them and Graphiti does not differ significantly from the better of the two. If no two systems differ, the null is reported.
4. The full runs, and the results report.

This is asof-bench v0.1: the data in data/v0.1, frozen with the registration; the package installs as `asofbench` 0.1.0. A result is comparable with the registered ones when it is scored on data/v0.1 with the registered models and settings ([section 11](PREREGISTRATION.md#11-systems-model-and-runs)).

## Enter a system

A system receives each item's question text and nothing else from data/v0.1. From the corpora it reads only what the registration gives it: most systems ingest their arm's whole stream, and grep at D reads the files and the log as of the date. It runs where the only routes out lead to the answering model's API and, for a store a language model builds, the construction model's. Each item is asked three times, and each answer is one JSON row, such as `{"id": ..., "system": "my-system", "arm": "H-", "run": 1, "answer": "Draft", ...}`, that `asofbench score` scores. A system that is not a v0.1 entrant can be scored today as an outside entry, declared in an entries file such as `{"my-system": {}}`. The results show it as an outside entry and never compare it in the headline. [docs/entering-a-system.md](docs/entering-a-system.md) has the full rules, and [docs/data-format.md](docs/data-format.md) the answer rows. If you maintain one of the entered systems, you are invited to review its adapter before any scored run: open an issue.

## Documentation

| To | Read |
|---|---|
| learn the design: the families, the arms, the extraction track | [docs/design.md](docs/design.md) |
| rebuild the data, the manifest and the streams | [docs/rebuilding.md](docs/rebuilding.md) |
| enter a system: what it receives, the network rule, the answer rows | [docs/entering-a-system.md](docs/entering-a-system.md) |
| score answers, and read the pilot, the gates and the headline | [docs/scoring.md](docs/scoring.md) |
| read the data files field by field | [docs/data-format.md](docs/data-format.md) |
| find your way around the code | [docs/code-map.md](docs/code-map.md) |
| look up a term | [docs/glossary.md](docs/glossary.md) |
| check the licences of the data and its sources | [docs/data-and-licences.md](docs/data-and-licences.md) |
| read the rules themselves | [PREREGISTRATION.md](PREREGISTRATION.md), the rule wherever another document seems to differ; start with sections 1, 7 and 12 |
| see what changed, before and after the freeze | [DEVIATIONS.md](DEVIATIONS.md) |

The registration is [PREREGISTRATION.md](PREREGISTRATION.md), [labelling/GUIDELINES.md](labelling/GUIDELINES.md), every module in `src/asofbench` and the generated files in data/v0.1, frozen by the tag `prereg-v0.1.2`. `src/asofbench` holds the generator, the stream builder, the materialiser and the scorer, which use only the Python standard library. Beside them are [docs/](docs/), [examples/](examples/), the tests in `tests/`, and [data/](data/), which describes the data and how to keep it out of training corpora.

## Known limitations

- Dates are recorded time, when a change entered the repository's first-parent chain, not when it was decided. In python/peps, 14 pull merges from 2016 to 2018 date the changes they bring in at the merge; no gold value depends on them.
- H- keeps commit messages, and for 90 of the 300 headline items a kept message names the answer by the item's date; results report them apart.
- H- keeps, with an empty diff and its message, the episode of each commit that changed only withheld header lines. One recorded after the item's date marks 198 of the 300 headline items: a system reading the stream can see it, and grep at D cannot. The headline's first reading leaves those items out, so on Ethereum it can show only a large difference.

[docs/limitations.md](docs/limitations.md) gives each in full, with the counts behind it.

## Related work

LongMemEval ([arXiv:2410.10813](https://arxiv.org/abs/2410.10813)) asks 500 curated questions over user-assistant chat histories; LoCoMo ([arXiv:2402.17753](https://arxiv.org/abs/2402.17753)) builds long dialogues with language-model agents, verified by annotators. TimeQA ([arXiv:2108.06314](https://arxiv.org/abs/2108.06314)) asks what a fact was at a time, over facts mined from Wikidata and verified by crowd workers. TemporalWiki ([arXiv:2204.14211](https://arxiv.org/abs/2204.14211)) uses the differences between consecutive Wikipedia and Wikidata snapshots to train and evaluate language models as knowledge changes. CronQuestions ([arXiv:2106.01515](https://arxiv.org/abs/2106.01515)) answers questions over temporal knowledge graphs. The two closest, StateMemBench ([arXiv:2608.19652](https://arxiv.org/abs/2608.19652)) and FEB-30 ([arXiv:2610.22209](https://arxiv.org/abs/2610.22209)), construct their histories. This benchmark takes real, dated, public decision records and computes its answer keys from version control. It asks about past dates as well as the present, and reads its headline where today's answer is wrong, with the headers that define the answer withheld. [docs/design.md](docs/design.md#how-it-differs-from-related-work) lists the rest.

## Contributing

Open an issue at https://github.com/srijansk/asof-bench/issues. For a suspected wrong answer key, the [gold-error template](.github/ISSUE_TEMPLATE/gold-error.md) asks for the item id, the answer you expect, and the repository, path and commit that show it. [CONTRIBUTING.md](CONTRIBUTING.md) says how to find an item's commit, which files are frozen, and how to run the tests.

## Disclosures

### Competing interests

The benchmark's author develops and promotes [crewlore](https://github.com/srijansk/crewlore) outside this benchmark, so crewlore is not entered: a result for it would be a result for the author's own project. Nor is any system built on another, unpublished benchmark the author co-wrote. The Personalized PageRank retriever (`ppr`) is the author's too, but it is written for this benchmark as a reference implementation of a published pattern, is in this repository before any scored run, and every table of the results report labels it as the author's own ([section 11](PREREGISTRATION.md#11-systems-model-and-runs)).

The author works at a company that builds software that gives AI agents an organisation's knowledge. The company has no role in this benchmark: none of its systems is entered, it funds no part of the work, and it has no say in the design or the results. The author holds the rights in this work; the company claims none.

### Acknowledgements

Most of the code, the data pipeline and these documents were written by Claude (Anthropic's model) at the author's direction, and separate Claude instances reviewed each change. The author set the question and the design, approved the decisions that shape the results, and is responsible for the whole.

## Citation

[CITATION.cff](CITATION.cff) holds the citation, which GitHub's "Cite this repository" reads. In BibTeX:

```bibtex
@misc{saket2026asofbench,
  author       = {Saket, Srijan},
  title        = {asof-bench: What Did a Project Decide, as of a Given Date?},
  year         = {2026},
  version      = {prereg-v0.1.2},
  howpublished = {\url{https://github.com/srijansk/asof-bench}}
}
```

## Licence

Code, tests and documents: MIT ([LICENSE](LICENSE)). The data in data/v0.1, the published results and every file published beside them: CC0 1.0 ([LICENSE-DATA](LICENSE-DATA)); the facts come from the public git histories at the pinned commits, no proposal text is published, and [docs/data-and-licences.md](docs/data-and-licences.md) gives every source's terms. This project is not affiliated with or endorsed by the Python Software Foundation or the Ethereum Foundation.
