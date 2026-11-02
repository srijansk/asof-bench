"""M0, the Graphiti cost pilot (PREREGISTRATION.md section 13): its two host-side steps, before and
after the ingestion that runs inside the run environment. It uses only the standard library.

    python systems/graphiti/m0.py prepare --pilot data/v0.1/pilot.json \\
        --stream streams/h-plus.jsonl --manifest data/v0.1/MANIFEST.json --out-dir <dir>

checks pilot.json and the H+ stream against the SHA-256s that the manifest registers, and writes
``<dir>/m0-h-plus.jsonl``, the H+ episodes of the 50 proposals that pilot.json lists for M0
(``m0_graphiti_proposals``), as the stream holds them and in its order, and ``<dir>/prepare.json``
(``proposals``, ``episodes``, ``text_chars``, the code points of their ``text`` as the manifest
counts them, ``stream_sha256``, the H+ stream's, and ``input_sha256``, m0-h-plus.jsonl's). The
episodes hold proposal text, so ``<dir>`` must be outside the checkout.

    python systems/graphiti/m0.py project --summary <run>/summary.json \\
        --prepare <dir>/prepare.json --manifest data/v0.1/MANIFEST.json \\
        --questions data/v0.1/questions.jsonl --streams-dir streams --out projection.json

checks that the summary is the finished run of claude-haiku-5-5 over the prepared input (its
stream's SHA-256 is prepare.json's ``input_sha256``; its last invocation exited with status 0 or
1 and its log covers every episode; no call raised) and that prepare.json, questions.jsonl and
the three local streams are the registered ones, then applies section 13's projection and writes
it, step by step, to ``projection.json``:

1. the rates, ``in_per_char = input_tokens / text_chars`` and ``out_per_char = output_tokens /
   text_chars``, with the tokens of the run's ingestion attempts from its summary.json (every
   attempt counted, the calls that removed episodes not; input tokens include the
   cache-creation and cache-read tokens) and the characters from prepare.json;
2. the ingestions that Graphiti's entries need, the H+, H- and CF streams and section 14's
   incremental path, whose characters are the H+ stream's again, each with its stream's
   characters from the manifest; its projected tokens, the rates times those characters; and its
   cost at the manifest's price basis (``graphiti_budget_basis``, which must state the price of a
   million input tokens and of a million output tokens);
3. the decision: ``graphiti`` when the total fits the manifest's budget
   (``graphiti_budget_usd``), and ``graphiti-reduced`` otherwise, with k, the largest number of
   other proposals whose projected total, with the proposals that the F1, F2 and CF items name,
   fits the budget: the candidates are the other proposals of the H+ stream, in ascending SHA-256
   of ``asof-bench/v0.1|distractor|<candidate id>``, and each proposal's characters per ingestion
   come from the three local streams (``h-plus.jsonl``, ``h-minus.jsonl`` and ``cf.jsonl`` in
   ``--streams-dir``). When the named proposals alone exceed the budget, which the registration
   does not provide for, k is null and ``reason`` says so.

The projection records the rates, the characters, tokens and cost of each ingestion, the total,
the budget, the decision, k, the price basis, each input file by name and SHA-256, and the
command line with each path reduced to its file name, so that it holds no local path; it holds
proposal ids and counts, and no proposal text. Arithmetic is exact (fractions); the file gives
floats. Exit status 0; 1 when the projection is written but no k fits; 2, with the reason, when
an input is refused and nothing is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

MODEL = "claude-haiku-5-5"
SEED = "asof-bench/v0.1"
NAMED_FAMILIES = ("F1", "F2", "CF")
STREAM_FILES = {"H+": "h-plus.jsonl", "H-": "h-minus.jsonl", "CF": "cf.jsonl"}
# Graphiti's ingestions (section 13): each arm's stream, and section 14's incremental path, which
# ingests the H+ episodes again.
INGESTIONS = (("H+", "H+"), ("H-", "H-"), ("CF", "CF"), ("incremental", "H+"))
PRICE = r"\$([0-9]+(?:\.[0-9]+)?) per million {} tokens"
MILLION = 1_000_000
CHUNK = 1 << 20
NO_K = ("the named proposals alone are projected beyond the budget, so no k fits; the "
        "registration does not provide for this case")
# The exit statuses of an ingestion that reached the stream's end (graphiti/ingest.py).
FINISHED = (0, 1)


class Refused(Exception):
    """An input that the step cannot use; nothing is written."""


def repository_root() -> Path | None:
    """The checkout this file is in: the nearest directory above it that holds the registration
    and systems/; None when there is none, as where systems/ alone is mounted."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "PREREGISTRATION.md").is_file() and (parent / "systems").is_dir():
            return parent
    return None


REPOSITORY = repository_root()


def inside_repository(path: Path) -> bool:
    if REPOSITORY is None:
        return False
    path = path.resolve()
    return path == REPOSITORY or REPOSITORY in path.parents


def name_of(path: Path) -> str:
    """A path as the outputs record it: its file name, with no directory."""
    return path.name or str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise Refused(f"{name_of(path)} cannot be read as JSON: {error}") from None
    if not isinstance(value, dict):
        raise Refused(f"{name_of(path)} is not a JSON object")
    return value


def episodes_of(path: Path):
    """The stream's episodes, each with its line as the file holds it."""
    with open(path, "rb") as handle:
        for number, raw in enumerate(handle, 1):
            try:
                episode = json.loads(raw)
            except ValueError:
                raise Refused(f"{name_of(path)}: line {number} is not JSON") from None
            yield raw, episode


def registered(manifest: dict, *keys: str) -> str:
    """The SHA-256 that the manifest registers under ``keys``; Refused when it registers none."""
    value = manifest
    for key in keys:
        value = value.get(key) if isinstance(value, dict) else None
    if not isinstance(value, dict) or not isinstance(value.get("sha256"), str):
        raise Refused(f"the manifest registers no SHA-256 for {'/'.join(keys)}")
    return value["sha256"]


def check_file(path: Path, expected: str, what: str) -> str:
    """The file's SHA-256, after checking it is ``expected``; Refused otherwise."""
    actual = sha256_file(path)
    if actual != expected:
        raise Refused(f"{name_of(path)} is not the registered {what}: its SHA-256 is {actual}, "
                      f"the manifest's {expected}")
    return actual


def check_stream(path: Path, manifest: dict, arm: str) -> str:
    """The stream's SHA-256, after checking it against the manifest's for ``arm``."""
    return check_file(path, registered(manifest, "streams", arm), f"{arm} stream")


def parse_prices(basis: str) -> tuple[Fraction, Fraction]:
    """The price of a million input tokens and of a million output tokens, in US dollars, that
    the price basis states; Refused unless it states each exactly once."""
    prices = []
    for kind in ("input", "output"):
        found = {Fraction(value) for value in re.findall(PRICE.format(kind), basis)}
        if len(found) != 1:
            raise Refused(f"the price basis {basis!r} does not state one price per million "
                          f"{kind} tokens")
        prices.append(found.pop())
    return prices[0], prices[1]


def prepare(pilot_path: Path, stream: Path, manifest_path: Path, out_dir: Path) -> dict:
    if inside_repository(out_dir):
        raise Refused(f"{out_dir} is inside the repository, and the episodes hold proposal text: "
                      "give a directory outside it")
    manifest = load_json(manifest_path)
    check_file(pilot_path, registered(manifest, "files", "pilot.json"), "pilot.json")
    proposals = load_json(pilot_path).get("m0_graphiti_proposals")
    if (not isinstance(proposals, list) or not proposals
            or not all(isinstance(p, str) for p in proposals)
            or len(set(proposals)) != len(proposals)):
        raise Refused(f"{name_of(pilot_path)} lists no m0_graphiti_proposals, or lists one twice")
    sha = check_stream(stream, manifest, "H+")
    out = out_dir / "m0-h-plus.jsonl"
    if out.exists() or (out_dir / "prepare.json").exists():
        raise Refused(f"{out_dir} already holds a prepared input")
    wanted, seen, kept, chars = set(proposals), set(), [], 0
    for raw, episode in episodes_of(stream):
        if episode.get("proposal") in wanted:
            kept.append(raw)
            seen.add(episode["proposal"])
            chars += len(episode["text"])
    if missing := [p for p in proposals if p not in seen]:
        raise Refused(f"the stream has no episode of {', '.join(missing)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    data = b"".join(kept)
    out.write_bytes(data)
    prepared = {"proposals": len(proposals), "episodes": len(kept), "text_chars": chars,
                "stream_sha256": sha, "input_sha256": hashlib.sha256(data).hexdigest()}
    (out_dir / "prepare.json").write_text(json.dumps(prepared, indent=2) + "\n", encoding="utf-8")
    return prepared


def candidate_key(candidate: str) -> str:
    return hashlib.sha256(f"{SEED}|distractor|{candidate}".encode()).hexdigest()


def priced(chars: dict[str, int], rates: tuple[Fraction, Fraction],
           prices: tuple[Fraction, Fraction]) -> tuple[list[dict], dict, Fraction]:
    """Each ingestion's characters, projected tokens and cost, given each stream's characters;
    the total; and the exact total cost."""
    rows, total = [], {"chars": 0, "input_tokens": Fraction(0), "output_tokens": Fraction(0),
                       "cost": Fraction(0)}
    for ingestion, stream in INGESTIONS:
        tokens_in, tokens_out = rates[0] * chars[stream], rates[1] * chars[stream]
        cost = (tokens_in * prices[0] + tokens_out * prices[1]) / MILLION
        rows.append({"ingestion": ingestion, "stream": stream, "chars": chars[stream],
                     "input_tokens": float(tokens_in), "output_tokens": float(tokens_out),
                     "cost_usd": float(cost)})
        total["chars"] += chars[stream]
        total["input_tokens"] += tokens_in
        total["output_tokens"] += tokens_out
        total["cost"] += cost
    return rows, {"chars": total["chars"], "input_tokens": float(total["input_tokens"]),
                  "output_tokens": float(total["output_tokens"]),
                  "cost_usd": float(total["cost"])}, total["cost"]


def project(summary_path: Path, prepare_path: Path, manifest_path: Path, questions_path: Path,
            streams_dir: Path, command: list[str]) -> dict:
    summary, prepared = load_json(summary_path), load_json(prepare_path)
    manifest = load_json(manifest_path)
    if summary.get("model") != MODEL or summary.get("arm") != "H+":
        raise Refused(f"{name_of(summary_path)} is not an H+ run whose every call {MODEL} "
                      f"answered: its arm is {summary.get('arm')!r} and its model "
                      f"{summary.get('model')!r}")
    if prepared.get("stream_sha256") != registered(manifest, "streams", "H+"):
        raise Refused(f"{name_of(prepare_path)} was not prepared from the registered H+ stream: "
                      f"its stream_sha256 is {prepared.get('stream_sha256')}, the manifest's "
                      f"{registered(manifest, 'streams', 'H+')}")
    if summary.get("stream_sha256") != prepared.get("input_sha256"):
        raise Refused(f"{name_of(summary_path)} is not the run of the input that "
                      f"{name_of(prepare_path)} describes: the run's stream has SHA-256 "
                      f"{summary.get('stream_sha256')}, the prepared input "
                      f"{prepared.get('input_sha256')}")
    if (summary.get("episodes"), summary.get("text_chars")) != (prepared.get("episodes"),
                                                                prepared.get("text_chars")):
        raise Refused(f"{name_of(summary_path)} counts {summary.get('episodes')} episodes and "
                      f"{summary.get('text_chars')} characters, {name_of(prepare_path)} "
                      f"{prepared.get('episodes')} and {prepared.get('text_chars')}")
    # A run that stopped before the stream's end has the whole stream's characters but only
    # some episodes' tokens, which would understate the rates.
    if summary.get("exit") not in FINISHED:
        raise Refused(f"{name_of(summary_path)} is not a finished run: its last invocation "
                      f"exited with status {summary.get('exit')!r}, not 0 or 1; resume it first")
    logged = (summary.get("ok") or 0) + (summary.get("failed") or 0)
    if logged != summary.get("episodes"):
        raise Refused(f"{name_of(summary_path)} is not a finished run: its log covers {logged} "
                      f"of its {summary.get('episodes')} episodes")
    if summary.get("api_errors") != 0:
        raise Refused(f"{name_of(summary_path)} records {summary.get('api_errors')!r} calls that "
                      "raised: some calls failed inside Graphiti, so the rates would be "
                      "understated")
    text_chars = prepared["text_chars"]
    if not isinstance(text_chars, int) or text_chars <= 0:
        raise Refused(f"{name_of(prepare_path)} counts no characters")
    basis = manifest.get("graphiti_budget_basis")
    if not isinstance(basis, str):
        raise Refused("the manifest has no price basis (graphiti_budget_basis)")
    prices = parse_prices(basis)
    budget = manifest.get("graphiti_budget_usd")
    if not isinstance(budget, int | float) or isinstance(budget, bool) or budget <= 0:
        raise Refused("the manifest has no Graphiti budget (graphiti_budget_usd)")
    hashes = {name: {"file": name_of(path), "sha256": sha256_file(path)}
              for name, path in (("summary", summary_path), ("prepare", prepare_path),
                                 ("manifest", manifest_path))}
    hashes["questions"] = {"file": name_of(questions_path), "sha256": check_file(
        questions_path, registered(manifest, "files", "questions.jsonl"), "questions.jsonl")}
    for arm, name in STREAM_FILES.items():
        hashes[arm] = {"file": name, "sha256": check_stream(streams_dir / name, manifest, arm)}

    rates = (Fraction(summary["input_tokens"], text_chars),
             Fraction(summary["output_tokens"], text_chars))
    full = {stream: manifest["streams"][stream]["chars"] for stream in STREAM_FILES}
    rows, total, cost = priced(full, rates, prices)
    projection = {
        "rule": "PREREGISTRATION.md section 13, M0 (systems/graphiti/m0.py project)",
        "command": command,
        "inputs": hashes,
        "pilot": {key: summary.get(key) for key in ("episodes", "ok", "failed", "text_chars",
                                                    "input_tokens", "output_tokens", "calls",
                                                    "api_errors", "calls_over_100k_input",
                                                    "removal", "wall_s", "elapsed_s", "exit",
                                                    "graphiti_core", "model", "images")},
        "rates": {"in_per_char": float(rates[0]), "out_per_char": float(rates[1]),
                  "rule": "input_tokens / text_chars and output_tokens / text_chars: the tokens "
                          "of the run's ingestion attempts, every attempt counted and the calls "
                          "that removed episodes not (summary.json; input tokens with the "
                          "cache-creation and cache-read tokens), over prepare.json's "
                          "characters"},
        "price_basis": basis,
        "prices_usd_per_million_tokens": {"input": float(prices[0]), "output": float(prices[1])},
        "ingestions": rows,
        "total": total,
        "budget_usd": float(budget),
        "fits_budget": cost <= Fraction(budget),
        "decision": "graphiti",
        "k": None,
        "reduced": None,
    }
    if projection["pilot"]["failed"]:
        print(f"m0 project: {projection['pilot']['failed']} of the pilot's episodes failed; the "
              "rates are measured on the attempts as they ran", file=sys.stderr)
    if cost > Fraction(budget):
        reduced = reduce(questions_path, streams_dir, rates, prices, Fraction(budget))
        projection["decision"], projection["k"] = "graphiti-reduced", reduced["k"]
        projection["reduced"] = reduced
        if reduced["k"] is None:
            projection["reason"] = NO_K
    return projection


def reduce(questions_path: Path, streams_dir: Path, rates: tuple[Fraction, Fraction],
           prices: tuple[Fraction, Fraction], budget: Fraction) -> dict:
    """graphiti-reduced's k: the named proposals plus the first k candidates whose projected
    total fits the budget, with each proposal's characters per stream from the local streams;
    k is None when the named proposals alone exceed the budget."""
    named = set()
    with open(questions_path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                item = json.loads(line)
            except ValueError:
                raise Refused(f"{name_of(questions_path)}: line {number} is not JSON") from None
            if item.get("family") in NAMED_FAMILIES:
                named.add(item["proposal"])
    per_stream: dict[str, dict[str, int]] = {}
    for stream, name in STREAM_FILES.items():
        counts: dict[str, int] = {}
        for _, episode in episodes_of(streams_dir / name):
            counts[episode["proposal"]] = counts.get(episode["proposal"], 0) + len(episode["text"])
        per_stream[stream] = counts
    if missing := sorted(named - set(per_stream["H+"])):
        raise Refused(f"the H+ stream has no episode of the named {', '.join(missing[:10])}")
    candidates = sorted(set(per_stream["H+"]) - named, key=candidate_key)

    def chars_of(proposals) -> dict[str, int]:
        return {stream: sum(per_stream[stream].get(p, 0) for p in proposals)
                for stream in STREAM_FILES}

    chosen = list(named)
    rows, total, cost = priced(chars_of(chosen), rates, prices)
    reduced = {"named_proposals": len(named), "candidates": len(candidates),
               "candidate_order": f"ascending SHA-256 of {SEED}|distractor|<candidate id>",
               "k": None, "distractors": [], "ingestions": rows, "total": total,
               "next_candidate": None}
    if cost > budget:
        return reduced
    k = 0
    for candidate in candidates:
        trial_rows, trial_total, trial_cost = priced(chars_of([*chosen, candidate]), rates,
                                                     prices)
        if trial_cost > budget:
            reduced["next_candidate"] = {"id": candidate, "total_cost_usd": float(trial_cost)}
            break
        chosen.append(candidate)
        k, rows, total = k + 1, trial_rows, trial_total
    reduced.update(k=k, distractors=candidates[:k], ingestions=rows, total=total)
    return reduced


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="m0.py", description="M0, the Graphiti cost pilot "
                                     "(PREREGISTRATION.md section 13).", allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="write the pilot's H+ episodes and prepare.json",
                               allow_abbrev=False)
    prep.add_argument("--pilot", type=Path, required=True)
    prep.add_argument("--stream", type=Path, required=True, help="the registered H+ stream")
    prep.add_argument("--manifest", type=Path, required=True)
    prep.add_argument("--out-dir", type=Path, required=True)
    proj = commands.add_parser("project", help="project the cost and write projection.json",
                               allow_abbrev=False)
    proj.add_argument("--summary", type=Path, required=True, help="the run's summary.json")
    proj.add_argument("--prepare", type=Path, required=True)
    proj.add_argument("--manifest", type=Path, required=True)
    proj.add_argument("--questions", type=Path, required=True)
    proj.add_argument("--streams-dir", type=Path, required=True,
                      help="holding h-plus.jsonl, h-minus.jsonl and cf.jsonl")
    proj.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            prepared = prepare(args.pilot, args.stream, args.manifest, args.out_dir)
            print(f"m0 prepare: {prepared['proposals']} proposals, {prepared['episodes']} "
                  f"episodes, {prepared['text_chars']} characters of text, written to "
                  f"{args.out_dir / 'm0-h-plus.jsonl'}")
            return 0
        command = ["m0.py", "project", "--summary", name_of(args.summary), "--prepare",
                   name_of(args.prepare), "--manifest", name_of(args.manifest), "--questions",
                   name_of(args.questions), "--streams-dir", name_of(args.streams_dir), "--out",
                   name_of(args.out)]
        projection = project(args.summary, args.prepare, args.manifest, args.questions,
                             args.streams_dir, command)
    except (Refused, OSError) as error:
        print(f"m0 {args.command}: {error}", file=sys.stderr)
        return 2
    args.out.write_text(json.dumps(projection, indent=2) + "\n", encoding="utf-8")
    k = "" if projection["k"] is None else f", k {projection['k']}"
    print(f"m0 project: ${projection['total']['cost_usd']:.2f} projected against a budget of "
          f"${projection['budget_usd']:.2f}: {projection['decision']}{k}; written to {args.out}")
    if "reason" in projection:
        print(f"m0 project: {projection['reason']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
