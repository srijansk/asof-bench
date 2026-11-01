# The data files

What each file of data/v0.1 holds, field by field, and the files the commands write. A system under test receives each item's `question` text and nothing else from data/v0.1 ([section 8](../PREREGISTRATION.md#8-arms-and-the-dated-revision-stream)): gold.jsonl, timelines.jsonl, edges.jsonl and REPORT.md hold or reveal answers, and so do an item's other fields (an id that begins `F1-change-` says that the status at D differs from the proposal's last offered status at S). Every record of every data file carries `canary`, the canary string, and its keys are written in sorted order. [PREREGISTRATION.md](../PREREGISTRATION.md) registers every rule this page describes, and is the rule wherever the two seem to differ.

## questions.jsonl

One item per line, 905 lines ([section 7](../PREREGISTRATION.md#7-question-families)).

| Field | Type | Meaning |
|---|---|---|
| `id` | string | `<family>-<stratum>-<proposal>-<date>`, such as `F1-change-PEP-643-2021-09-03`; for F5, `F5-count-<corpus>-<status>-<date>` |
| `family` | string | `F1`, `F2`, `F3`, `F4`, `F5` or `CF` |
| `stratum` | string | F1: `change`, `current` or `recent`; F2: `positive` or `negative`; F3: `size-2-4`, `size-5-14` or `size-15-plus`; F4: `closure-final`; F5: `count`; CF: `counterfactual` |
| `corpus` | string | `pep` or `eth` (the EIPs and ERCs, one series) |
| `proposal` | string or null | the proposal the question names, by its id at S, such as `PEP-643`, `EIP-2256` or `ERC-6384`; null for F5 |
| `as_of` | string | the item's date D, as YYYY-MM-DD; S for F2 and F3 |
| `arms` | list | the arms the item is asked in: `H+` and `H-` for F1, F2 and F3; `H+` for F4 and F5; `CF` for CF |
| `question` | string | the question text, the only field a system receives |
| `answer_type` | string | `status` (F1 and CF), `set` (F2, F3 and F4) or `count` (F5) |
| `choices` | list or null | the statuses a status question offers, in the order it lists them; null for sets and counts |
| `canary` | string | the canary string |

## gold.jsonl

One answer key per line, in the order of questions.jsonl ([sections 3 to 7](../PREREGISTRATION.md#3-recorded-time-and-as-of)).

| Field | Type | Meaning |
|---|---|---|
| `id` | string | the item's id |
| `gold` | string, list or integer | a status (for CF, the permuted status the counterfactual stream records), a sorted list of proposal numbers (F2, F3 and F4), or a count (F5) |
| `gold_outcome` | string or null | the outcome class of a status answer: `adopted`, `declined`, `superseded` or `open` ([section 5](../PREREGISTRATION.md#5-status-normalisation)); null for sets and counts |
| `source` | object or null | for F1 and CF, the `repo`, `path` and `commit` of the file whose header sets the answer, `commit` being that repository's as-of commit; null for the other families |
| `as_of_commits` | object | for each repository, `peps`, `EIPs` and `ERCs`, the last first-parent commit recorded by the end of the item's date, or null when the repository has none by then ([section 3](../PREREGISTRATION.md#3-recorded-time-and-as-of)) |
| `meta` | object | facts about the item that the reports split by; the next table |
| `canary` | string | the canary string |

`meta` holds `rank`, the item's place in its family, stratum and corpus in frozen order, and, by family:

| Family | Key | Meaning |
|---|---|---|
| F1, F2, F3, F4, CF | `ref` | the proposal as the question writes it, such as `PEP 643 ("Metadata for Package Source Distributions")` |
| F1, CF | `interval_start` | when the status the item asks about began: the recorded time of the commit that set it |
| F1 | `status_at_s` | the proposal's status at S |
| F1 | `gold_equals_status_at_s` | whether the answer is the status at S |
| F1 | `h_minus_header_only_after_d` | whether the H- stream holds an episode of the proposal, recorded after D, that changed only withheld header lines ([section 12](../PREREGISTRATION.md#12-answers-and-scoring) reports H- F1 split by it) |
| F2 | `recoverable` | for each arm, whether the asked PEP's own episodes can show the answer |
| F2 | `mentions` | negatives only: the PEPs that the asked PEP's text mentions at S |
| F3 | `size` | the number of proposals in the answer |
| F3 | `max_hops` | the most requires-steps from the asked proposal to one of them |
| F4 | `closure` | the proposals the asked proposal required, directly or through others, as of D |
| F4 | `gold_at_s` | the answer the same question would have at S |
| F4 | `status_at_d` | the asked proposal's own status as of D |
| F5 | `series` | the proposals counted, such as `PEPs (PEP 0 is excluded)` |
| F5 | `status` | the status counted |
| F5 | `registered_date` | the date the registration names for the item |
| F5 | `moved_days` | how many days later the item's date is than that date, when it had to move ([section 7](../PREREGISTRATION.md#7-question-families)) |
| CF | `true_status` | the status the real history records as of D |
| CF | `decision_word` | whether the proposal's text as of D holds one of the decision words |
| CF | `decision_words` | which of them it holds |
| CF | `header_cue` | whether a header field other than the status records the real status, which the counterfactual edit leaves alone |
| CF | `header_cue_fields` | which fields do |

## pilot.json

The pilot's items, one JSON object ([section 13](../PREREGISTRATION.md#13-gates)).

| Field | Type | Meaning |
|---|---|---|
| `groups` | list | each group's `family`, `stratum`, `corpus`, `arms` and `ids`: the first 50 F1-change and the first 25 F1-current items of each corpus in frozen order, and every CF item |
| `ids` | list | the 180 items of the groups together |
| `m0_graphiti_proposals` | list | the 50 PEPs that Graphiti ingests in M0, the cost pilot |
| `canary` | string | the canary string |

## timelines.jsonl

One proposal per line, 1,982 lines: every status it held, as the first-parent chain records it ([sections 2 to 6](../PREREGISTRATION.md#2-corpora-and-pins)).

| Field | Type | Meaning |
|---|---|---|
| `pid` | string | the proposal's id at S, such as `PEP-1` |
| `family` | string | `pep` or `eth` |
| `series` | string | `PEP`, `EIP` or `ERC` |
| `number` | integer | the proposal's number |
| `title` | string | its title |
| `present_at_s` | boolean | whether its file is present at S |
| `intervals` | list | its intervals in order, each with `status` (a status as the proposal's header records it, normalised, or `absent` while its file is deleted and `missing` while its header has no status), `start` and `end` (recorded times; `end` is null while the interval still holds at the pin), `commit` (the commit that began it), `repo` and `is_transition` (true when that commit changed the status from an earlier one; false for the file's first status, its first status after an absence, and an interval without a status) |
| `canary` | string | the canary string |

## edges.jsonl

The relation headers at S, one edge per line, 1,885 lines ([section 7](../PREREGISTRATION.md#7-question-families)).

| Field | Type | Meaning |
|---|---|---|
| `source` | string | the proposal whose header states the relation |
| `target` | string | the proposal it names |
| `relation` | string | `requires` (in both corpora) or `supersedes` (PEPs only) |
| `corpus` | string | `pep` or `eth` |
| `canary` | string | the canary string |

## labelling/mention_pairs.jsonl

The 200 mention pairs of the extraction track ([section 9](../PREREGISTRATION.md#9-extraction-track-headers-withheld)). No proposal text is published: a mention is recorded by its position.

| Field | Type | Meaning |
|---|---|---|
| `pair_id` | string | the two proposals' ids joined by `~`, such as `PEP-236~PEP-238` |
| `corpus` | string | `pep` or `eth` |
| `rank` | integer | the pair's place in the sample's frozen order |
| `a`, `b` | string | the two proposals' ids |
| `a_title`, `b_title` | string | their titles |
| `a_mentions_b`, `b_mentions_a` | boolean | whether each one's text mentions the other at S |
| `a_source`, `b_source` | object | the `repo`, `path` and `commit` of each one's file at S |
| `a_mention_spans`, `b_mention_spans` | list | up to two mentions of the other in each one's text, each with `repo`, `path`, `commit`, `start` and `end`, offsets in Unicode code points into the file's text at that commit |
| `canary` | string | the canary string |

## REPORT.md and MANIFEST.json

REPORT.md is the build report: the pins, the items by family, stratum and corpus, the eligible pools, the trivial-answer floors and the data-quality notes. MANIFEST.json records the build ([section 15](../PREREGISTRATION.md#15-manifest)):

| Field | Meaning |
|---|---|
| `pins` | the pinned commit of each repository |
| `snapshot` | S |
| `seed` | the selection seed |
| `files` | the size in bytes, line count and SHA-256 of every generated file |
| `items` | the total of items |
| `item_counts` | the items of each family, stratum and corpus |
| `eligible_pools` | the eligible pool of each family, stratum and corpus |
| `labelling_pairs` | the mention pairs per corpus |
| `streams` | for each arm's stream, its episode count, characters, estimated tokens and SHA-256, and under `sidecars` the size and SHA-256 of each file beside it |
| `generator` | the generator's version, the SHA-256 over its source files (`source_sha256`), `source_commit` and `source_dirty` |
| `graphiti_budget_usd` | the Graphiti budget of M0, in US dollars |
| `graphiti_budget_basis` | the price basis the budget was set at |
| `canary` | the canary string |

## Streams

`asofbench stream` writes an arm's stream from the clones; the stream is never committed, since it holds the proposals' text ([section 8](../PREREGISTRATION.md#8-arms-and-the-dated-revision-stream)). It is a JSONL file of episodes in recorded-time order, one per proposal per first-parent commit:

| Field | Meaning |
|---|---|
| `episode_id` | the commit's first 12 hex digits and the proposal's id, joined by `:` |
| `corpus` | `pep` or `eth` |
| `repo` | the repository the version came from: `peps`, `EIPs` or `ERCs` |
| `proposal` | the proposal's id at S |
| `ref` | the proposal as the questions write it |
| `recorded_at` | the commit's recorded time, in UTC |
| `commit` | the commit |
| `message` | its message |
| `kind` | `create`, `modify`, `rename` or `delete` |
| `text` | for a create, the whole file; for a modify or a rename, a unified diff; for a delete, nothing |

Beside each stream the command writes `<stream>.stats.json` (its episodes, characters, estimated tokens and SHA-256), and for H- `<stream>.message_cues.json` (for each proposal, the statuses its kept commit messages name, with when) and for CF `<stream>.cf_messages.json`. Only the scorer reads these; a system receives the stream file alone. `asofbench materialize` replays a stream as of a date into a directory: one file per proposal present at the end of the date, named `<repo>/<series>-<number>`, such as `peps/pep-0008` or `EIPs/eip-1559`, and `git-log.txt`, the log of the commits up to the date, which is what grep at D reads ([Trees for grep at D](entering-a-system.md#trees-for-grep-at-d)).

## Answer rows, entries and items

A system's answers are JSONL, one row per item, arm and run ([Answer rows](entering-a-system.md#answer-rows)):

| Field | Meaning |
|---|---|
| `id` | the item's id |
| `system` | the system's identifier, or a name an entries file declares |
| `arm` | `H+`, `H-` or `CF` |
| `run` | 1, 2 or 3 |
| `answer` | a string from `choices` for F1 and CF, a list of proposal numbers for F2, F3 and F4, an integer for F5; any other shape scores zero |
| `tokens_in` | the answer's input tokens |
| `tokens_out` | its output tokens |
| `wall_s` | its wall time in seconds |
| `tool_calls` | its number of tool calls, or whether any was made |

An entries file (`--entries`) is a JSON object that maps each system to the `arms` and `families` it is entered in, and an items file (`--items`) is a JSON list of item ids, or pilot.json itself ([Scoring](scoring.md#scoring)). `asofbench score` writes results.md, results.json and gates.json.
