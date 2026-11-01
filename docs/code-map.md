# The code

How src/asofbench is laid out, which command runs which module, and where the registered numbers live. Every "section N" in the code, its docstrings and `asofbench --help`, and every "spec section N", is section N of [PREREGISTRATION.md](../PREREGISTRATION.md), which registers every rule the code applies and is the rule wherever this page seems to differ. src/asofbench is part of the registration, frozen by the tag `prereg-v0.1.2`: read it freely, and change it only as a recorded deviation in [DEVIATIONS.md](../DEVIATIONS.md). It uses only the Python standard library, so every line that decides a label can be read and audited.

## The pipeline

| Step | Command | Reads | Writes | Module |
|---|---|---|---|---|
| 1. The questions and their answers | `asofbench build --clones DIR --out DIR` | the clones of python/peps, ethereum/EIPs and ethereum/ERCs, each up to its pin | questions.jsonl, gold.jsonl, pilot.json, labelling/mention_pairs.jsonl, timelines.jsonl, edges.jsonl, REPORT.md and MANIFEST.json | timeline.py, questions.py, manifest.py |
| 2. An arm's stream | `asofbench stream --clones DIR --data DIR --arm ARM --out FILE` | the clones and the build's manifest | the stream and the files beside it | stream.py |
| 3. What grep at D reads | `asofbench materialize --stream FILE --as-of DATE --out DIR` | the stream alone (with `--arm` and `--data`, refusing one that is not the registered stream) | one file per proposal present at the end of the date, and git-log.txt | materialize.py |
| 4. Answers | a system's adapter and the harness that runs it; the registered systems' adapters are committed before any scored run ([section 11](../PREREGISTRATION.md#11-systems-model-and-runs)) | each item's question text, and what section 11 gives the system | answer rows | not in this repository yet |
| 5. Scores | `asofbench score --data DIR --answers FILE --out DIR` and the four model flags | questions.jsonl, gold.jsonl, pilot.json and the answer rows | results.md, results.json and gates.json | scoring.py |
| Annotators | `asofbench labelling-sheet` | the clones and labelling/mention_pairs.jsonl | a local sheet of passages, never published | labelling.py |

[docs/data-format.md](data-format.md) describes every file these steps read and write, and [examples/toy_world.py](../examples/toy_world.py) runs steps 1, 2, 3 and 5 on a toy corpus.

## The modules

| Module | Lines | What it does | Start with |
|---|---|---|---|
| cli.py | 329 | the `asofbench` command: parses the arguments, runs `build` and hands each other command to its module | `main` |
| corpora.py | 80 | what every part shares: S, the selection seed, the canary, the three repositories and their aliases and exclusions | `REPOS`, `S` |
| gitwalk.py | 428 | a read-only walk of a clone's first-parent chain, recorded time and the as-of commit (section 3) | `first_parent_commits`, `as_of_commit` |
| headers.py | 289 | reading a proposal's header (section 4) and withholding fields from it for H- (section 8) | `parse_header`, `strip_withheld`, `WITHHELD` |
| statuses.py | 181 | status normalisation, outcome classes and the counterfactual permutation (section 5) | `normalize_status`, `normalize_answer_status`, `permute` |
| mentions.py | 53 | which proposals a text mentions (section 7) | `mentions` |
| timeline.py | 614 | each proposal's versions and status intervals, ERC stitching and the data-quality report (sections 2 to 6) | `load_corpus`, `Proposal` |
| select.py | 62 | the deterministic hash order and the choice of a day (section 7) | `order`, `pick_day` |
| questions.py | 1,519 | the six families' items and answers, the mention-pair sample, the output files and REPORT.md (sections 7 to 10) | `build_items`, `write_outputs`, `TEMPLATES`, `SIZES` |
| stream.py | 570 | the three arms' streams and the files beside them (section 8) | `Episode`, `episodes`, `write_arm` |
| materialize.py | 305 | a stream replayed as the proposal files and the log as of a date (section 11) | `replay`, `git_log` |
| manifest.py | 216 | MANIFEST.json and the generator's provenance (section 15) | `build_manifest` |
| scoring.py | 2,975 | the scorer: answer rows, scores, intervals, aggregation, the gates, the headline comparison and results.md (sections 10 to 13) | `main`, `score_answer` |
| labelling.py | 156 | the annotators' local sheet (section 9) | `main` |

The registered numbers live in a few places: corpora.py (S, the seed, the canary, the repositories, `ALIASES`, `EXCLUDED`, `ERC_SPLIT`), select.py (`MARGIN`), statuses.py (`CANONICAL`, `CHOICES`, the outcome classes and the counterfactual swaps), headers.py (`WITHHELD`), questions.py from line 46 to 129 (`SIZES`, `TEMPLATES` and the other sampling constants), and scoring.py from line 190 to 254 (the entrants, the runs, the registered models and build, the gate thresholds and the row fields) and from line 1649 (`HEADLINE_BASELINE`, `HEADLINE_SYSTEMS` and the headline's family).

scoring.py, by its section markers:

| Lines | Part | Start with |
|---|---|---|
| 1 to 260 | the rules in prose, the imports and the registered constants | `REGISTERED_SYSTEMS`, `REGISTERED_ANSWERING_MODEL`, `REGISTERED_DATA` |
| 261 to 322 | statistics | `wilson`, `mcnemar_exact`, `holm` |
| 323 to 413 | one answer | `score_answer` |
| 414 to 594 | aggregation | `aggregate` |
| 595 to 826 | the pilot gates (section 13) | `pilot_gates` |
| 827 to 1643 | the inputs of `asofbench score`, their checks and the reported splits | `Scope` |
| 1644 to 2149 | the headline comparison (section 12) | `HEADLINE_SYSTEMS` |
| 2150 to 2975 | results.md and the command | `main` |

## The tests

test_gitwalk.py, test_headers.py, test_statuses.py, test_mentions.py and test_select.py each cover one module. test_timeline.py, test_stream.py, test_materialize.py and test_labelling.py run their module on small git repositories they build, laid out like the clones. test_questions.py and test_sampling.py cover the families and the sampling on hand-built corpora; test_scoring.py covers the scorer, alone and through the command; test_build.py covers `asofbench build` and the manifest end to end, and the document tests that check that the README, these pages and the registration state each rule as the code applies it; test_docs.py checks that the links between the documents resolve, and the field lists of docs/data-format.md against the data; test_streamcheck.py checks the streams against an independent reading of the clones. The test_corpus_*.py files run on the real clones and are skipped unless `ASOFBENCH_CLONES` names them. The test names state the rules: `uv run --frozen --extra dev pytest --collect-only -q tests/test_scoring.py` lists the scorer's.

## Versions

v0.1 is the benchmark's version, and data/v0.1 its registered build. 0.1.0 is the Python package's version (pyproject.toml, `asofbench.__version__` and the manifest's `generator.version`). `prereg-v0.1.2` is the tag that froze v0.1's registration, after two earlier, private freezes that [DEVIATIONS.md](../DEVIATIONS.md) lists. The manifest's `generator.source_commit` names c9a3ac7, the commit data/v0.1 was built from; the tag is on its child, which copies that manifest into section 15 and changes no module of src/asofbench.
