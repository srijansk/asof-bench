"""The ``asofbench`` command: ``build``, ``stream``, ``materialize``, ``score`` and
``labelling-sheet``.

``build`` reads the three clones up to their as-of-S pins and writes the questions, gold, pilot
list, mention-pair sample, timelines, report and manifest, and with ``--streams`` the three arms'
streams. It refuses, before it writes anything, an output directory that already holds a
``MANIFEST.json`` or that holds ``--expect-pins``. The other commands parse their arguments here
and hand them to the ``main`` of
``asofbench.stream``, ``asofbench.materialize``, ``asofbench.scoring`` and
``asofbench.labelling``, which are imported only when those commands run. Every command reports an
input it cannot use with exit status 2 and a message, not a traceback. An arm may be written
``H-`` or with the registration's minus sign.
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
import re
import sys
import time
import unicodedata
from datetime import date
from pathlib import Path

ARMS = ("H+", "H-", "CF")
MINUS = chr(0x2212)  # the registration writes the arm H-, with a minus sign


def arm(value: str) -> str:
    """An arm name as given, with the minus sign read as the hyphen: ``H-`` either way."""
    return value.strip().replace(MINUS, "-")


def _day(value: str) -> date:
    if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value) is None:
        raise argparse.ArgumentTypeError(f"{value!r} is not a date written YYYY-MM-DD")
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a date") from None


def _usd(value: str) -> float:
    try:
        amount = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an amount of US dollars") from None
    if not math.isfinite(amount) or amount <= 0:
        raise argparse.ArgumentTypeError(f"{value!r}: a budget is a positive amount of US dollars")
    return amount


def _text(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("an empty price basis says nothing")
    return value.strip()


REFUSED_CATEGORIES = ("Cc", "Cf", "Zl", "Zp")  # control, format, line and paragraph separators


def _heading_text(value: str, blank: str) -> str:
    """``value`` stripped, for a model or the answering settings, which every heading of
    results.md names: refused when blank (``blank`` says what to give) or when it holds a
    character of Unicode category Cc, Cf, Zl or Zp, such as a newline, a tab, a bidirectional
    control or a line separator, which would break or disguise a heading."""
    text = value.strip()
    if not text:
        raise argparse.ArgumentTypeError(blank)
    if any(unicodedata.category(ch) in REFUSED_CATEGORIES for ch in text):
        raise argparse.ArgumentTypeError("holds a control, format or separator character (Unicode "
                                         "category Cc, Cf, Zl or Zp, such as a newline, a tab, a "
                                         "bidirectional control or a line separator), which the "
                                         "headings of results.md cannot carry")
    return text


def _model_or_none(value: str, blank: str) -> str:
    """A model that may be none: its id, or ``none`` when its case-folded form is none, so that
    every record and heading spells it the same way."""
    text = _heading_text(value, blank)
    return "none" if text.casefold() == "none" else text


def _construction_model(value: str) -> str:
    """A construction model: an identifier, or none when no store in the run is LLM-built
    (section 12)."""
    return _model_or_none(value, "give the construction model's id, or none when no store in the "
                                 "run is LLM-built")


def _answering_model(value: str) -> str:
    """An answering model: an identifier, never none (section 12)."""
    model = _heading_text(value, "give the answering model's id")
    if model.casefold() == "none":
        raise argparse.ArgumentTypeError("every scored system answers through a model, so give "
                                         "its id")
    return model


def _answering_settings(value: str) -> str:
    """The settings of every answering call, as text (section 11)."""
    return _heading_text(value, "give the settings of every answering call (such as the effort, "
                                "thinking and temperature)")


def _embedding_model(value: str) -> str:
    """An embedding model: an identifier, or none when no system of the run embeds text
    (section 11)."""
    return _model_or_none(value, "give the embedding model's id, or none when no system of the "
                                 "run embeds text")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asofbench", description="What did a project decide, as of date D?"
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    build = commands.add_parser("build", help="generate the questions, gold and manifest")
    build.add_argument("--clones", type=Path, required=True,
                       help="directory holding the peps, EIPs and ERCs clones")
    build.add_argument("--out", type=Path, required=True,
                       help="a fresh output directory: one that already holds a MANIFEST.json, "
                            "such as data/v0.1, or that holds --expect-pins is refused")
    build.add_argument("--expect-pins", type=Path, default=None, dest="expect_pins",
                       metavar="MANIFEST.json",
                       help="a registered manifest: refuse clones whose pins differ from its own")
    build.add_argument("--streams", type=Path, default=None,
                       help="also write the three arms' streams into this directory and record "
                            "their stats and SHA-256, and those of the files beside them, in the "
                            "manifest (sections 8 and 15)")
    build.add_argument("--graphiti-budget-usd", type=_usd, default=None,
                       dest="graphiti_budget_usd", metavar="FLOAT",
                       help="the M0 gate's Graphiti budget in US dollars (section 13); needs "
                            "--price-basis")
    build.add_argument("--price-basis", type=_text, default=None, dest="price_basis",
                       metavar="TEXT",
                       help="the prices the budget is at: the answering model's list prices and "
                            "the date they were read")

    stream = commands.add_parser("stream", help="build a dated revision stream (section 8)")
    stream.add_argument("--clones", type=Path, required=True)
    stream.add_argument("--data", type=Path, required=True, help="the build's output directory")
    stream.add_argument("--arm", type=arm, choices=ARMS, required=True)
    stream.add_argument("--out", type=Path, required=True)

    materialize = commands.add_parser(
        "materialize", help="an arm's stream as the proposal files as of a date (section 11)")
    materialize.add_argument("--stream", type=Path, required=True)
    materialize.add_argument("--as-of", type=_day, required=True, dest="as_of",
                             metavar="YYYY-MM-DD", help="the files as of the end of this day (UTC)")
    materialize.add_argument("--out", type=Path, required=True,
                             help="a new or empty directory for the files")
    materialize.add_argument("--arm", type=arm, choices=ARMS, default=None,
                             help="with --data: refuse a stream that is not this arm's registered "
                                  "one")
    materialize.add_argument("--data", type=Path, default=None,
                             help="with --arm: the build whose MANIFEST.json registers the stream")

    score = commands.add_parser("score", help="score system answers (section 12)")
    score.add_argument("--data", type=Path, required=True, help="the build's output directory")
    score.add_argument("--answers", type=Path, nargs="+", required=True)
    score.add_argument("--out", type=Path, required=True)
    score.add_argument("--answering-model", type=_answering_model, required=True,
                       dest="answering_model", metavar="ID",
                       help="the model that answered every item of these answers (section 11), "
                            "never none; score each answering model in its own run")
    score.add_argument("--answering-settings", type=_answering_settings, required=True,
                       dest="answering_settings", metavar="TEXT",
                       help="the settings of every answering call, such as \"effort medium, "
                            "adaptive thinking, default temperature\" (section 11)")
    score.add_argument("--construction-model", type=_construction_model, required=True,
                       dest="construction_model", metavar="ID",
                       help="the model that built every LLM-built store of the run, or none when "
                            "no store in the run is LLM-built (section 11)")
    score.add_argument("--embedding-model", type=_embedding_model, required=True,
                       dest="embedding_model", metavar="ID",
                       help="the one embedding model of every system of the run that embeds "
                            "text, or none when no system of the run embeds (section 11)")
    score.add_argument("--training-cutoff", type=_day, default=None, dest="training_cutoff",
                       metavar="YYYY-MM-DD",
                       help="the first day after the answering model's training data: F1-recent "
                            "is then also split by whether each item's transition was recorded "
                            "before it or on or after it (section 10)")
    score.add_argument("--entries", type=Path, default=None,
                       help="a JSON file mapping each system to {\"arms\": [...], \"families\": "
                            "[...]}: what it is entered in. Without it every system is entered "
                            "in every arm and family, except long-context and graphiti-reduced "
                            "(F1, F2 and CF only)")
    score.add_argument("--items", type=Path, default=None,
                       help="a JSON file with a list of item ids, or pilot.json's structure: "
                            "only these items count (a pilot run)")
    score.add_argument("--cf-messages", type=Path, default=None, dest="cf_messages",
                       help="the <stream>.cf_messages.json that `asofbench stream --arm CF` "
                            "wrote: the report then gives each CF item's message count. Refused "
                            "if its SHA-256 is not the one the build's manifest registers")
    score.add_argument("--message-cues", type=Path, default=None, dest="message_cues",
                       help="the <stream>.message_cues.json that `asofbench stream --arm H-` "
                            "wrote: the report then says, for each H- F1 item, whether a kept "
                            "message names its gold status by its date. Refused if its SHA-256 "
                            "is not the one the build's manifest registers")

    sheet = commands.add_parser(
        "labelling-sheet", help="the annotators' local sheet of passages (section 9)")
    sheet.add_argument("--clones", type=Path, required=True,
                       help="directory holding the pinned peps, EIPs and ERCs clones")
    sheet.add_argument("--data", type=Path, required=True, help="the build's output directory")
    sheet.add_argument("--out", type=Path, required=True,
                       help="the sheet, a Markdown file; keep it local (labelling-sheet*.md is "
                            "git-ignored)")
    return parser


def _same_directory(a: Path, b: Path) -> bool:
    """Whether the paths ``a`` and ``b`` name one directory, however each is written: the same
    file when both exist (so letter case is read as the volume reads it), else the same resolved
    path."""
    try:
        if os.path.exists(a) and os.path.exists(b):
            return os.path.samefile(a, b)
        return a.resolve() == b.resolve()
    except (OSError, RuntimeError):  # a path that cannot be resolved, such as a symlink loop
        return os.path.abspath(a) == os.path.abspath(b)


def _refused_out(out: Path, pins: Path | None) -> str | None:
    """Why ``build`` will not write into ``out``, or None: the directory of ``--expect-pins``, the
    registered manifest, or a directory that already holds a MANIFEST.json, a build such as
    data/v0.1, whose files and manifest a build there would write over."""
    if pins is not None and _same_directory(out, pins.parent):
        return (f"{out} is the directory of --expect-pins {pins}, the registered manifest: build "
                "into a fresh directory, never over data/v0.1")
    if os.path.lexists(out / "MANIFEST.json"):
        return (f"{out} already holds a MANIFEST.json, so a build there would write over that "
                "build's files: build into a fresh directory, never over data/v0.1")
    return None


def _build(args: argparse.Namespace) -> int:
    from .gitwalk import CloneError
    from .manifest import (
        ManifestError,
        build_manifest,
        provenance,
        read_manifest,
        registered_pins,
    )
    from .questions import build_items, write_outputs
    from .timeline import load_corpus

    refused = _refused_out(args.out, args.expect_pins)  # before anything is read or written
    if refused is not None:
        print(f"asofbench build: {refused}", file=sys.stderr)
        return 2
    start = time.monotonic()
    try:
        provenance()  # the generator's own source files, read before anything is written
        expected = None
        if args.expect_pins is not None:
            expected = registered_pins(read_manifest(args.expect_pins), args.expect_pins)
        corpus = load_corpus(args.clones, expect_pins=expected)
    except (CloneError, ManifestError) as e:
        print(f"asofbench build: {e}", file=sys.stderr)
        return 2
    with corpus.readers() as readers:
        items = build_items(corpus, readers)
    write_outputs(items, corpus, args.out)
    streams = None
    if args.streams is not None:
        from .stream import StreamError, read_cf, write_streams

        try:
            cf = read_cf(args.out)
        except StreamError as e:
            print(f"asofbench build: {e}", file=sys.stderr)
            return 2
        streams = write_streams(corpus, cf, args.streams)
    budget = None
    if args.graphiti_budget_usd is not None:
        budget = (args.graphiti_budget_usd, args.price_basis)
    manifest = build_manifest(args.out, corpus, items.pools, streams=streams,
                              graphiti_budget=budget)
    seconds = time.monotonic() - start
    print(f"asofbench build: {manifest['items']} items, "
          f"{sum(manifest['labelling_pairs'].values())} mention pairs, written to {args.out} "
          f"in {seconds:.1f} s")
    for key, n in manifest["item_counts"].items():
        print(f"  {key}: {n}")
    for arm_name, stats in (manifest["streams"] or {}).items():
        print(f"  stream {arm_name}: {stats['episodes']} episodes, {stats['chars']} characters, "
              f"sha256 {stats['sha256']}")
    return 0


def _delegate(module: str, args: argparse.Namespace) -> int:
    try:
        target = importlib.import_module(f"asofbench.{module}")
    except ModuleNotFoundError as e:
        if e.name != f"asofbench.{module}":
            raise
        print(f"asofbench {args.command}: asofbench.{module} is not installed", file=sys.stderr)
        return 2
    return target.main(args)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "build":
        if (args.graphiti_budget_usd is None) != (args.price_basis is None):
            parser.error("build: --graphiti-budget-usd and --price-basis go together (section 13)")
        return _build(args)
    if args.command == "stream":
        return _delegate("stream", args)
    if args.command == "materialize":
        if (args.arm is None) != (args.data is None):
            parser.error("materialize: --arm and --data go together")
        return _delegate("materialize", args)
    if args.command == "labelling-sheet":
        return _delegate("labelling", args)
    return _delegate("scoring", args)


if __name__ == "__main__":
    sys.exit(main())
