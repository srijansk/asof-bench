# The design of asof-bench v0.1

What the benchmark asks, how its items and arms are built, the extraction track, and how it differs from related work. [PREREGISTRATION.md](../PREREGISTRATION.md) registers every rule this page describes, and is the rule wherever the two seem to differ.

## What it asks

The 905 items of [data/v0.1/questions.jsonl](../data/v0.1/questions.jsonl) fall into six families ([section 7](../PREREGISTRATION.md#7-question-families)). "Ethereum" means the EIPs and ERCs, which share one number space and are read as one series ([section 6](../PREREGISTRATION.md#6-eips-and-ercs-are-one-series)).

| Family | What it asks | Corpora | Answer | Items |
|---|---|---|---|---|
| F1 | a proposal's status as of the end of date D | PEPs, Ethereum | one status from the listed choices | 503 |
| F2 | which PEPs superseded a PEP, at S | PEPs | a list of PEP numbers | 82 |
| F3 | which proposals an Ethereum proposal requires, directly or through the proposals it requires, at S | Ethereum | a list of proposal numbers | 120 |
| F4 | which of the proposals an Ethereum proposal required, directly or through the proposals it required, had status Final, as of D | Ethereum | a list of proposal numbers | 120 |
| F5 | how many proposals had a given status as of D | PEPs, Ethereum | one integer | 50 |
| CF | F1's question, asked over a stream whose statuses were permuted | PEPs, Ethereum | one status | 30 |

F1 has three strata: change (D is a date at which the status differs from the proposal's last offered status at S: its status at S, or, when no question offers that status, as for Stagnant, the last status before S that a question offers; 150 PEP and 150 Ethereum items), current (D = S; 50 and 50) and recent (dates inside status intervals that began with a transition recorded on or after 2026-01-01; 43 and 60). The counts are those of [data/v0.1/MANIFEST.json](../data/v0.1/MANIFEST.json) (`item_counts`).

Each item is asked in one or more arms, each with its own dated revision stream: a file of episodes, an episode being one proposal's change at one commit, with its commit message ([section 8](../PREREGISTRATION.md#8-arms-and-the-dated-revision-stream)):

| Arm | What the stream holds | Families | Items |
|---|---|---|---|
| H+ | every change to every proposal file, headers visible | F1, F2, F3, F4, F5 | 875 |
| H- | the same changes with the status, decision and relation header lines withheld; identity fields (number, title, author, created, type and category) and commit messages kept, as is the episode of a change that touched only withheld lines, whose diff is then empty | F1, F2, F3 | 705 |
| CF | H+ with every status header of the 30 CF proposals permuted throughout their history, and the messages of the commits that change their recorded status values replaced | CF | 30 |

The registration writes the arm H− with a minus sign; these documents and the commands write `H-`, and the scorer reads both.

## The extraction track

Each engine that builds a store from the H- stream (a graph or a set of statements) exports it, and a deterministic mapping from that export to (proposal, status, valid-from) statements and (proposal, relation, proposal) edges is committed to this repository as an addendum, `addenda/<engine>.md`, by pull request, before the engine's first scored run; the addendum cannot change questions or gold ([section 9](../PREREGISTRATION.md#9-extraction-track-headers-withheld)). The store-level scores are status recall at S and at the dates of the 503 F1 items, and relation recall against [data/v0.1/edges.jsonl](../data/v0.1/edges.jsonl). `asofbench score` does not compute them: they are published beside the results report, with the command and the outputs that produced them, and with the inputs that hold no proposal text: the engine's committed mapping, the statements and edges it produces (facts, like edges.jsonl), the gold they are scored against, and the export's size and SHA-256. The exports themselves contain proposal text and stay local, like the streams, and published label files carry the labels and notes without quoted passages ([section 12](../PREREGISTRATION.md#12-answers-and-scoring)).

edges.jsonl holds the header edges at S that H- withholds, one JSON record per edge with the fields `canary`, `corpus`, `relation`, `source` and `target`. PEP m `supersedes` PEP n when, at S, n lists m in Superseded-By (or Replaced-By, its pre-2011 name) or m lists n in Replaces, which is F2's relation exactly; proposal N `requires` M when N's requires header at S lists M, in either corpus. Self-references and numbers that are not proposals at S are dropped, and the records are sorted by relation, then source id, then target id, each compared as a string. At v0.1's pins there are 1,885 edges: 1,798 Ethereum and 41 PEP requires edges, and 46 PEP supersedes edges, which are the gold members of F2's 41 positives.

The precision of typed relations is estimated on 200 labelled mention pairs, 100 per corpus, in [data/v0.1/labelling/mention_pairs.jsonl](../data/v0.1/labelling/mention_pairs.jsonl): pairs of proposals present at S in which one body at S mentions the other and no header relation at S joins them, which two annotators label blind to every system output, following [labelling/GUIDELINES.md](../labelling/GUIDELINES.md). Each mention is recorded by its `repo`, `path`, `commit`, `start` and `end`, not by its text. The offsets count Unicode code points, not bytes, into the file's text at the pinned commit as the generator decodes it (UTF-8, with undecodable bytes replaced by U+FFFD, and a byte-order mark and carriage returns kept), with `start` inclusive and `end` exclusive, so that Python's `text[start:end]` is the mention; a language whose strings count UTF-16 units, as JavaScript's do, or bytes, as Rust's do, must convert. Annotators build a local sheet of the passages around the mentions from the clones; it holds proposal text, so it is git-ignored and never published:

```bash
uv run asofbench labelling-sheet --clones ../asofbench-clones --data data/v0.1 --out labelling-sheet-annotator-1.md
```

## How it differs from related work

asof-bench should be read next to the work listed in [section 16](../PREREGISTRATION.md#16-prior-work). The two closest construct their histories. StateMemBench ("Can Agent Memory Systems Track Evolving State?", [arXiv:2608.19652](https://arxiv.org/abs/2608.19652)) poses multi-session scenarios whose answers must reflect the current state rather than a superseded one. FEB-30 ("Beyond Downstream QA: Rethinking How We Evaluate Agent Memory", [arXiv:2610.22209](https://arxiv.org/abs/2610.22209)) generates its documents from known fact trajectories, which gives it deterministic gold without human annotation and lets it inspect the memory store directly. asof-bench instead takes real, dated, public decision records, asks about past dates as well as the current one, takes its gold from version control, and withholds the headers that define the gold (H-), so that an answer has to come from the prose, the commit messages, and when episodes were recorded.

Section 16 also lists ForgetEval ([arXiv:2606.15903](https://arxiv.org/abs/2606.15903)), Bemula ([arXiv:2610.18052](https://arxiv.org/abs/2610.18052)), "Revoked but Still Authoritative" ([arXiv:2609.08258](https://arxiv.org/abs/2609.08258)), MemLeak ([arXiv:2606.29788](https://arxiv.org/abs/2606.29788)) and CAR ([arXiv:2604.14488](https://arxiv.org/abs/2604.14488)), and two proposal-graph explorers: PEP Map ([komo-fr/pep-map](https://github.com/komo-fr/pep-map)), which draws the citations between PEPs, and epge ([a16k-lab/epge](https://github.com/a16k-lab/epge)), an interactive graph explorer for EIPs and ERCs.
