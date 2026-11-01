# Labelling guidelines: mention pairs

These guidelines fix how two annotators label the mention-pair sample in `mention_pairs.jsonl` (pre-registration §9). The labels estimate how often a system that types a mention as a relation is wrong, so they must be made before anyone looks at a system's output, and from the proposal texts alone.

## What you label

Each record is one pair of proposals, A and B, with A the lower number. In the proposal texts at the pinned commit, at least one of them mentions the other, and no header relates them: for PEPs no Superseded-By (or Replaced-By), Replaces or Requires header, and for Ethereum proposals no Requires header, the only header relation pre-registration §9 checks there. The record gives both titles, which side mentions which, and, for each side that mentions the other, where up to two of its mentions are (commit, path and character offsets), taken in text order so that their passages do not overlap, but no proposal text. The offsets count Unicode code points into the file's text at the pinned commit as the generator decodes it: UTF-8, with undecodable bytes replaced and a byte-order mark and carriage returns kept (pre-registration §4). Build your sheet of passages from the pinned clones with `asofbench labelling-sheet --clones DIR --data DIR --out labelling-sheet-<annotator>.md`: it shows the 200 characters of the body centred on each mention, and it stays on your machine (it is git-ignored and never published). Read the passages first; open the full texts at the pinned commit whenever the passages do not settle the label.

Give each pair exactly one of six labels:

| Label | Use it when the texts state that |
|---|---|
| `supersedes A→B` | A takes B's place: A replaces, supersedes or obsoletes B, or B says it was replaced by A |
| `supersedes B→A` | the same, with B taking A's place |
| `requires A→B` | A depends on B, by EIP-1's definition (`EIPS/eip-1.md` at the pinned commit): "A `requires` dependency is created when the current EIP cannot be understood or implemented without a concept or technical element from another EIP." So A cannot be understood or implemented without a concept or technical element from B. PEP pairs use the same test |
| `requires B→A` | the same, with B depending on A |
| `other` | a different specific relationship: one extends or amends the other, is an alternative to it, reverts it, generalises it, or is its follow-up; name it in `note` |
| `none` | the mention is incidental: background, an example, a citation, a list of related proposals, a "see also" |

## Rules

1. Judge what the texts say, not what you know about the proposals' history.
2. Using, citing or building an example on another proposal is not a requirement. EIP-1 says a mention does not by itself create a dependency; apply the same reading to PEPs.
3. A proposal that discusses another and rejects its approach, without taking its place, is `other` (note: "rejected alternative"), not `supersedes`.
4. If you hesitate between a typed label and `none`, choose `none` and say why in `note`.
5. If the two texts disagree (one claims a relation the other does not mention), label what the stronger statement says and note the disagreement.

## Output

Write one JSON object per line to `labels-<annotator>.jsonl`: `{"pair_id": "...", "label": "...", "note": "..."}`, with `note` empty when there is nothing to say. Do not quote the proposal texts in `note`, not even a few words: refer to a mention by its side and its place in the record, or say in your own words what the text states, since the label files are published and the proposal texts are not (pre-registration §12). Annotators label on the understanding that the label files, notes included, are dedicated to the public domain under CC0 1.0 (the legal code in LICENSE-DATA) when they are published; an annotator who does not agree does not label. Label independently; do not compare files until both are complete.

## Agreement and adjudication

Cohen's κ is computed over the six labels on the two independent labellings, before any discussion. The sample is used only if κ is at least 0.6. The annotators then discuss every disagreement and record one adjudicated label per pair in `labels-adjudicated.jsonl`, with a note on how each disagreement was settled; the adjudicated labels are the ones used (pre-registration §9).
