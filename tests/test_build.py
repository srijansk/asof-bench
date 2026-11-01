"""``asofbench build`` and the manifest, end to end on small git repositories (spec section 15)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
import types
from collections import Counter
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

import pytest
from githistory import git_env, pull_merges
from gitrepo import TempRepo, clone

from asofbench import __version__, cli
from asofbench.corpora import CANARY
from asofbench.gitwalk import files_at, is_top_level
from asofbench.headers import CUE_NAMES, parse_header
from asofbench.manifest import (
    OUTPUT_FILES,
    SOURCE_DIR,
    ManifestError,
    _is_source,
    build_manifest,
    provenance,
    source_files,
    source_sha256,
)
from asofbench.manifest import SIDECARS as MANIFEST_SIDECARS
from asofbench.materialize import LOG_FILE
from asofbench.questions import DECISION_WORDS
from asofbench.scoring import (
    _META_FLAGS,
    ALPHA,
    CONSTRUCTED_SYSTEMS,
    CORPORA,
    EMBEDDING_SYSTEMS,
    GRAPHITI,
    GRAPHITI_REDUCED,
    HEADLINE_BASELINE,
    HEADLINE_FAMILY,
    HEADLINE_SYSTEMS,
    REGISTERED_ANSWERING_MODEL,
    REGISTERED_ANSWERING_SETTINGS,
    REGISTERED_CONSTRUCTION_MODEL,
    REGISTERED_DATA,
    REGISTERED_EMBEDDING_MODEL,
    REGISTERED_FAMILIES,
    REGISTERED_SYSTEMS,
    binomial_tail,
    holm,
    mcnemar_exact,
)
from asofbench.statuses import normalize_status, permute
from asofbench.timeline import load_corpus

FILES = ("questions.jsonl", "gold.jsonl", "pilot.json", "labelling/mention_pairs.jsonl",
         "timelines.jsonl", "edges.jsonl", "REPORT.md")
ROOT = Path(__file__).resolve().parents[1]


def pep(number: int, status: str, body: str = "Plain text.\n") -> str:
    return f"PEP: {number}\nTitle: Proposal {number}\nStatus: {status}\n\n{body}"


def eip(number: int, status: str, requires: str = "", body: str = "Plain text.\n") -> str:
    extra = f"requires: {requires}\n" if requires else ""
    return f"---\neip: {number}\ntitle: Proposal {number}\nstatus: {status}\n{extra}---\n{body}"


def noon(day: str) -> str:
    return f"{day}T12:00:00Z"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("clones")
    repos = {name: TempRepo(root / name) for name in ("peps", "EIPs", "ERCs")}
    for repo in repos.values():
        repo.write("README.md", "Proposals.\n")
        repo.commit("start", "2000-01-01T00:00:00Z")
    peps, eips, ercs = repos["peps"], repos["EIPs"], repos["ERCs"]
    for day, files in [
        ("2001-01-01", {"pep-0001.txt": pep(1, "Draft", "Compare PEP 2.\n")}),
        ("2002-01-01", {"pep-0002.txt": pep(2, "Draft")}),
        ("2005-01-01", {"pep-0001.txt": pep(1, "Final", "Compare PEP 2.\n")}),
        ("2006-01-01", {"pep-0002.txt": pep(2, "Rejected")}),
    ]:
        for path, text in files.items():
            peps.write(path, text)
        peps.commit(f"Update on {day}", noon(day))
    for day, files in [
        ("2015-11-19", {"EIPS/eip-20.md": eip(20, "Draft")}),
        ("2016-01-01", {"EIPS/eip-1.md": eip(1, "Draft", body="See EIP-2.\n")}),
        ("2016-02-01", {"EIPS/eip-2.md": eip(2, "Draft", requires="1")}),
        ("2016-03-01", {"EIPS/eip-3.md": eip(3, "Draft", requires="2")}),
        ("2017-01-01", {"EIPS/eip-2.md": eip(2, "Final", requires="1")}),
        ("2017-09-11", {"EIPS/eip-20.md": eip(20, "Final")}),
        ("2018-01-01", {"EIPS/eip-1.md": eip(1, "Living", body="See EIP-2.\n")}),
        ("2023-10-26", {"EIPS/eip-20.md": eip(20, "Moved")}),
    ]:
        for path, text in files.items():
            eips.write(path, text)
        eips.commit(f"Update on {day}", noon(day))
    ercs.write("ERCS/erc-20.md", eip(20, "Final"))
    ercs.commit("complete erc move", "2023-10-25T15:11:00Z")
    return root


def build(world: Path, out: Path) -> int:
    return cli.main(["build", "--clones", str(world), "--out", str(out)])


@pytest.fixture(scope="module")
def built(world, tmp_path_factory):
    out = tmp_path_factory.mktemp("data") / "v0.1"
    assert build(world, out) == 0
    return out


def git_show(repo: Path, spec: str) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL="/dev/null")
    done = subprocess.run(["git", "-C", str(repo), "show", "--no-textconv", spec], env=env,
                          capture_output=True, check=True)
    return done.stdout.decode("utf-8", errors="replace")


def test_build_writes_every_output(built):
    for name in (*FILES, "MANIFEST.json"):
        assert (built / name).is_file(), name


def test_manifest(built, world):
    manifest = json.loads((built / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["canary"] == CANARY
    assert manifest["pins"] == load_corpus(world).pins
    # Section 13: the Graphiti budget and its price basis are null until they are set, and a
    # build without --streams records no stream.
    assert manifest["graphiti_budget_usd"] is None and manifest["graphiti_budget_basis"] is None
    assert manifest["streams"] is None
    assert manifest["generator"] == {"name": "asofbench", "version": __version__, **provenance()}
    assert set(manifest["files"]) == set(FILES)
    for name, entry in manifest["files"].items():
        data = (built / name).read_bytes()
        assert entry["sha256"] == hashlib.sha256(data).hexdigest(), name
        assert entry["bytes"] == len(data)
        assert entry["lines"] == data.count(b"\n")
    questions = [json.loads(line) for line in (built / "questions.jsonl").read_text().splitlines()]
    counts: dict[str, int] = {}
    for row in questions:
        key = f"{row['family']}/{row['stratum']}/{row['corpus']}"
        counts[key] = counts.get(key, 0) + 1
    assert manifest["item_counts"] == dict(sorted(counts.items()))
    assert manifest["items"] == len(questions)
    assert manifest["files"]["questions.jsonl"]["records"] == len(questions)


def test_the_build_publishes_the_header_edges_at_s(built):
    """Section 9: data/v0.1/edges.jsonl, which the manifest lists (``test_manifest``), holds
    every header edge at S; this world has two requires edges and no supersession."""
    lines = (built / "edges.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [
        {"canary": CANARY, "corpus": "eth", "relation": "requires", "source": "EIP-2",
         "target": "EIP-1"},
        {"canary": CANARY, "corpus": "eth", "relation": "requires", "source": "EIP-3",
         "target": "EIP-2"},
    ]


def test_the_version_is_0_1_0_in_the_package_and_the_project():
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'^version = "([^"]+)"$', project, re.MULTILINE).group(1) == "0.1.0"
    assert __version__ == "0.1.0"


def frozen_manifest_problems(registration: str, stored: dict) -> list[str]:
    """What keeps ``stored`` (data/v0.1/MANIFEST.json) from being the frozen manifest of section
    15: its copy in the registration, a clean commit and the Graphiti budget."""
    problems = []
    section = registration.split("\n## 15. Manifest\n", 1)[1].split("\n## 16. ", 1)[0]
    blocks = re.findall(r"```json\n(.*?)\n```", section, re.DOTALL)
    if len(blocks) != 1 or json.loads(blocks[0]) != stored:
        problems.append("section 15's JSON block is not data/v0.1/MANIFEST.json")
    generator = stored["generator"]
    if not re.fullmatch(r"[0-9a-f]{40}", str(generator["source_commit"])):
        problems.append(f"source_commit is {generator['source_commit']!r}, not a commit id")
    if generator["source_dirty"] is not False:
        problems.append("source_dirty is not false")
    budget = stored["graphiti_budget_usd"]
    if not (isinstance(budget, (int, float)) and budget > 0 and stored["graphiti_budget_basis"]):
        problems.append("no Graphiti budget with its price basis")
    return problems


def source_commit_problems(root: Path, generator: dict) -> list[str]:
    """With read-only git in the checkout at ``root``: whether a manifest's ``source_commit``
    (in ``generator``) is a commit of the repository, an ancestor of HEAD, and holds src/asofbench
    files whose digest, by the manifest's own rule (``manifest.source_sha256`` over the regular
    files ``gitwalk.files_at`` reads and ``manifest._is_source`` takes), is the manifest's
    ``source_sha256``."""
    commit = str(generator["source_commit"])

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(root), *args], env=git_env(), capture_output=True,
                              check=False)

    if (not re.fullmatch(r"[0-9a-f]{40}", commit)
            or git("cat-file", "-e", f"{commit}^{{commit}}").returncode != 0):
        return [f"source_commit {commit!r} is not a commit of this repository"]
    problems = []
    if git("merge-base", "--is-ancestor", commit, "HEAD").returncode != 0:
        problems.append(f"source_commit {commit} is not an ancestor of HEAD")
    files = {name: data for name, data in files_at(root, SOURCE_DIR, commit).items()
             if _is_source(name)}
    digest = source_sha256(files)
    if digest != generator["source_sha256"]:
        problems.append(f"{SOURCE_DIR} at source_commit {commit} has the digest {digest}, not "
                        f"source_sha256 {generator['source_sha256']}")
    return problems


def is_shallow(root: Path) -> bool:
    """Whether the repository at ``root`` is a shallow clone (read-only git)."""
    done = subprocess.run(["git", "-C", str(root), "rev-parse", "--is-shallow-repository"],
                          env=git_env(), capture_output=True, text=True, check=False)
    return done.stdout.strip() == "true"


def frozen_history_problems(root: Path, generator: dict) -> list[str] | None:
    """``source_commit_problems`` in the checkout at ``root``, or None where the history cannot
    answer them: outside a git checkout of its own, and in a shallow clone, which holds the commit
    that carries the frozen manifest but not, necessarily, the earlier commit the manifest names,
    so that the commit would read as unknown although nothing is wrong."""
    if not is_top_level(root) or is_shallow(root):
        return None
    return source_commit_problems(root, generator)


def stored_manifest() -> dict:
    """data/v0.1/MANIFEST.json."""
    return json.loads((ROOT / "data" / "v0.1" / "MANIFEST.json").read_text(encoding="utf-8"))


def test_the_registration_carries_the_frozen_manifest():
    # Section 15: the manifest is copied into the registration when it is frozen, so the two
    # cannot drift apart, and a frozen manifest names a clean commit and records the Graphiti
    # budget. This reads no git history, so it runs everywhere, in the source distribution and a
    # shallow clone too; the commit itself is looked up in the history by the next test.
    registration = (ROOT / "PREREGISTRATION.md").read_text(encoding="utf-8")
    assert frozen_manifest_problems(registration, stored_manifest()) == []


def test_the_frozen_manifests_source_commit_is_in_the_history():
    # Section 15: in a full git checkout, the frozen manifest's source_commit is a commit of this
    # repository, an ancestor of HEAD, whose src/asofbench gives the manifest's source_sha256.
    # Skipped outside a checkout of its own (the source distribution) and in a shallow clone,
    # which need not hold that commit (frozen_history_problems); the test above still runs there.
    history = frozen_history_problems(ROOT, stored_manifest()["generator"])
    if history is None:
        pytest.skip("not a full git checkout (outside one, or a shallow clone), so the manifest's "
                    "source commit cannot be read")
    assert history == []


def registration_section(text: str, number: int) -> str:
    """Section ``number`` of the registration, from its heading to the next one."""
    return text.split(f"\n## {number}. ", 1)[1].split("\n## ", 1)[0]


# The options of ``asofbench score`` that the registration does not name: the command's plumbing
# (its help, the build it reads, the answer files and the output directory), which carries no
# registered value. Every other option of the command must be named.
UNNAMED_SCORE_OPTIONS = ("-h", "--help", "--data", "--answers", "--out")


def score_options() -> set[str]:
    """The option strings of ``asofbench score``, from the command's own parser."""
    (commands,) = [action for action in cli._parser()._actions
                   if isinstance(action, argparse._SubParsersAction)]
    return set(commands.choices["score"]._option_string_actions)


def backticked(text: str) -> list[str]:
    return re.findall(r"`([^`]+)`", text)


def unregistered_names(text: str) -> list[str]:
    """The names the code writes or reads that the registration does not give: the generated
    files (section 15), the files beside each arm's stream and the log beside a materialised tree
    (sections 8 and 11), the header fields and decision words of the CF cues (section 7), the
    entrants' identifiers and the entrants that embed text or build their store with a language
    model (section 11), and the score flags, the gold meta flags the scorer reads and the
    headline's systems (section 12)."""
    missing = []
    # Sections 8 and 11: the log that `asofbench materialize` writes beside the tree.
    missing += [f"section {number}: {LOG_FILE}" for number in (8, 11)
                if f"`{LOG_FILE}`" not in registration_section(text, number)]
    # Every option of the score command but its plumbing is named, in backticks, in the text, so
    # a new option cannot go unregistered; the plumbing list must hold real options only.
    options = score_options()
    missing += [f"asofbench score has no option {option}" for option in UNNAMED_SCORE_OPTIONS
                if option not in options]
    for flag in sorted(options - set(UNNAMED_SCORE_OPTIONS)):
        if not re.search(rf"`{re.escape(flag)}[` ]", text):
            missing.append(f"the score flag {flag}")
    # Section 11: the identifiers, in the scorer's order, and the two sets the scorer checks the
    # embedding and construction models against.
    entrants = registration_section(text, 11)
    found = re.search(r"The scorer knows the entrants by these identifiers: (.*?), and rejects",
                      entrants, re.DOTALL)
    if found is None or backticked(found.group(1)) != list(REGISTERED_SYSTEMS):
        missing.append(f"section 11's identifiers: {', '.join(REGISTERED_SYSTEMS)}")
    for lead, systems in (("The registered systems that embed text", EMBEDDING_SYSTEMS),
                          ("those whose store a language model builds", CONSTRUCTED_SYSTEMS)):
        found = re.search(rf"{lead}[^.;]*? are ((?:`[^`]+`(?:, | and )?)+)", entrants)
        if found is None or backticked(found.group(1)) != list(systems):
            missing.append(f"section 11, {lead}: {', '.join(systems)}")
    # Section 12: the gold meta flags the scorer reads; the headline's systems, with
    # graphiti-reduced beside graphiti for M0's fallback, and its baseline; and the registered
    # family that the headline's correction runs over, every compared system on each corpus.
    scoring_text = registration_section(text, 12)
    found = re.search(r"The scorer reads the flags that the build records in the gold `meta` "
                      r"\((.*?)\);", scoring_text, re.DOTALL)
    listed = set() if found is None else set(backticked(found.group(1)))
    missing += [f"section 12's list of the meta flags the scorer reads: {key}"
                for key in sorted({key for flags in _META_FLAGS.values() for key in flags})
                if key not in listed]
    found = re.search(r"Systems: each registered system that keeps or retrieves context from the "
                      r"stream or its proposals, (.*?), each against grep at D, `([^`]+)`",
                      scoring_text, re.DOTALL)
    compared = [name for system in HEADLINE_SYSTEMS
                for name in ((system, GRAPHITI_REDUCED) if system == GRAPHITI else (system,))]
    if found is None or (backticked(found.group(1)), found.group(2)) != (compared,
                                                                        HEADLINE_BASELINE):
        missing.append(f"section 12's headline systems: {', '.join(compared)} against "
                       f"{HEADLINE_BASELINE}")
    family = (f"Holm, over the {HEADLINE_FAMILY} registered comparisons, the "
              f"{len(HEADLINE_SYSTEMS)} compared systems times the {len(CORPORA)} corpora")
    if family not in scoring_text or HEADLINE_FAMILY != len(HEADLINE_SYSTEMS) * len(CORPORA):
        missing.append(f"section 12's headline correction: {family}")
    files = registration_section(text, 15).split("The manifest as frozen", 1)[0]
    missing += [f"section 15: {name}" for name in OUTPUT_FILES if name not in files]
    streams = registration_section(text, 8)
    for arm, suffixes in MANIFEST_SIDECARS.items():
        for suffix in suffixes:
            shared = all(suffix in others for others in MANIFEST_SIDECARS.values())
            written = f"`<stream>.{suffix}`"
            if not shared:  # named with its arm, as "for H- `<stream>.message_cues.json`"
                written = f"for {arm.replace('-', chr(0x2212))} {written}"
            if written not in streams and f"section 8: {written}" not in missing:
                missing.append(f"section 8: {written}")
    # The header-cue rule is the sentence of section 7 that begins "A cue is". It lists the
    # decision words as one phrase, in the code's order, and names every field the cue reads
    # (headers.CUE_NAMES: the last-call names, Superseded-By's two, withdrawal-reason, Accepted and
    # Resolution); a field matches only whole, so review-period-end is not found inside
    # review-period-ends.
    cf = registration_section(text, 7).split("**CF, counterfactual edits", 1)[1]
    found = re.search(r"A cue is .*?\.(?=\s|$)", cf, re.DOTALL)
    rule = "" if found is None else found.group(0)
    words = ", ".join(DECISION_WORDS[:-1]) + " or " + DECISION_WORDS[-1]
    if words not in rule:
        missing.append(f"section 7's header-cue rule: {words}")
    for name in CUE_NAMES:
        if not re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", rule, re.IGNORECASE):
            missing.append(f"section 7's header-cue rule: {name}")
    return missing


def test_the_registration_names_the_outputs_the_files_beside_each_stream_and_the_cf_cues():
    # Sections 7, 8, 11, 12 and 15 name what the code writes and reads, so a new output, a
    # renamed file beside a stream or a tree, a header field the CF cue reads, a score flag, an
    # entrant's identifier, a meta flag the scorer reads or a change to the headline's systems
    # cannot go unregistered.
    assert unregistered_names((ROOT / "PREREGISTRATION.md").read_text(encoding="utf-8")) == []


# The headline's claim rule (section 12, scoring._claimed) in the words that section 1, section 12
# and the README use for it: the direction comes from the majority results the McNemar test
# compares, not from the accuracy averaged over the runs.
CLAIM_RULE = "it is right where grep at D is wrong on more items than the reverse"
FAMILY_ORDER = ("F1", "F2", "F3", "F4", "F5", "CF")  # section 7's order


# Words that the headline's documents must not use: the family of the comparisons a run makes,
# which the fixed family replaced, and any claim that an adjusted p is the same whatever systems a
# run holds, which Holm's step-down does not give (only that leaving systems out never lowers it).
STALE_HEADLINE_WORDS = ("over every comparison the run makes",
                        "Holm-adjusted over every comparison", "whichever systems",
                        "the same adjusted p", "invariant")


def headline_rule_gaps(registration: str, readme: str) -> list[str]:
    """Where the registration's sections 1 and 12 or the README no longer state the headline's
    rule: the claim's direction from the majority results; Holm over the registered family of
    ``HEADLINE_FAMILY`` comparisons, a comparison not made counting as p = 1, with what that
    guarantees (leaving systems out never lowers a remaining adjusted p); each reading's own Holm
    family; a comparison made only where both sides have a scored answer; and the registered full
    run, whose rows alone carry a claim, which uses the registered entries (an entries file that
    only declares outside entries, or none) and needs grep at D and each compared system to answer
    every compared item in runs 1, 2 and 3, each unmet condition named, as the scorer words it, in
    section 12 and the README's examples. Also where any of them still uses the old family or
    claims invariance (``STALE_HEADLINE_WORDS``), and where section 12's default entries leave out
    a system that is entered in fewer families by default (scoring.REGISTERED_FAMILIES)."""
    from asofbench import scoring

    places = {"section 1": registration_section(registration, 1),
              "section 12": registration_section(registration, 12), "README.md": readme}
    # The scorer's own reasons for a run whose entries file narrows a registered system and for a
    # side that leaves runs out (scoring._narrowed_words, scoring._answered_words), as the two
    # documents quote them.
    examples = (f'"{scoring._narrowed_words("closed-book", ["H-"], None)}"',
                f'"{scoring._answered_words(HEADLINE_BASELINE, 1, 150, "pep")}"')
    family = f"{HEADLINE_FAMILY} registered comparisons"
    gaps = [f"{where}: {phrase}" for where, text in places.items()
            for phrase in (CLAIM_RULE, "Holm-adjusted", "no detectable difference", family,
                           "registered full run")
            if phrase not in text]
    never_lowers = "leaving systems out of a run never lowers the adjusted p of a comparison that"
    complete = "grep at D and each compared system answering every compared item in all three runs"
    for where, phrases in (
        ("section 1", ("in which a comparison that a run does not make counts as p = 1",
                       "the scoring run of every item with the registered entries", complete)),
        ("section 12", (f"Correction: Holm, over the {family}",
                        "a comparison that the run does not make counts as p = 1", never_lowers,
                        f"Each reading is Holm-adjusted on its own over the {family}, a "
                        "comparison it does not make counting as p = 1, not together with the "
                        "headline's",
                        "both have at least one scored answer", "Only its rows carry a `claim`",
                        "when it uses the registered entries, with no entries file or with one "
                        "that only declares outside entries",
                        f"have a scored answer, as above, to every H{chr(0x2212)} F1-change "
                        "item of both corpora in runs 1, 2 and 3", "before the headline's table",
                        *examples)),
        ("README.md", (f"**Correction**: Holm, over the {family}",
                       "a comparison the run does not make counts as p = 1", never_lowers,
                       f"each Holm-adjusted on its own over the {family}",
                       "both have at least one scored answer", "Only its rows carry `claim`",
                       "with the registered entries, that makes all 14 comparisons", complete,
                       "its entries are the registered ones, with no entries file or one that "
                       "only declares outside entries",
                       "have a scored answer to every H- F1-change item of both corpora in runs "
                       "1, 2 and 3", "its reasons in its opening paragraph, before the table",
                       *examples)),
    ):
        gaps += [f"{where}: {phrase}" for phrase in phrases if phrase not in places[where]]
    gaps += [f"{where}: still says {word!r}" for where, text in places.items()
             for word in STALE_HEADLINE_WORDS if word in text]
    found = re.search(r"Entries are declared per system: (.*?)\.(?: |$)", places["section 12"])
    sentence = "" if found is None else found.group(1)
    for system, families in REGISTERED_FAMILIES.items():
        order = [family for family in FAMILY_ORDER if family in families]
        words = f"{', '.join(order[:-1])} and {order[-1]} only"
        if (f"`{system}`" not in sentence and system.replace("-", " ") not in sentence) or (
                words not in sentence):
            gaps.append(f"section 12's default entries: {system} in {words}")
    return gaps


def test_the_registration_and_the_readme_state_the_headline_rule():
    # The rule the headline exists to apply is stated the same way in section 1, section 12 and
    # the README, and the default entries name every system entered in fewer families.
    registration = (ROOT / "PREREGISTRATION.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert headline_rule_gaps(registration, readme) == []


# The README's statement of how large a difference the headline's reading without header-only
# items can show on Ethereum (Known limitations), each part as a pattern of its own words: the
# necessary condition with the pairs it names, Holm's multiplier, the fewest items one way by
# Holm place, and the closing sentence.
POWER = (
    r"On (\d+) items the exact McNemar test gives p below 0\.05 only when at least (\d+) more of "
    r"them are right for one system and wrong for the other than the reverse, and not always "
    r"then: ((?:\d+ to \d+, )*\d+ to \d+ and \d+ to \d+) pass, and (\d+) to (\d+) does not",
    r"multiplies the k-th smallest of the reading's (\d+) p values in a full run by (\d+) - k",
    r"so a result on these (\d+) items needs, with none the other way, (\d+) such items as the "
    r"smallest p, (\d+) from the (\d+)(?:st|nd|rd|th) smallest on, (\d+) from the "
    r"(\d+)(?:st|nd|rd|th), (\d+) from the (\d+)(?:st|nd|rd|th) and (\d+) only as the "
    r"(\d+)(?:st|nd|rd|th), with every comparison before it rejected too",
    r"at least (\d+) more of its (\d+) items one way than the other, (\d+) to (\d+) counting only "
    r"as the last comparison Holm rejects, and (\d+) to (\d+) needed as the smallest p",
)


def power_numbers(data: Path) -> dict:
    """The same facts from data/v0.1 and the scorer's own test and correction: the Ethereum H-
    F1-change items without a header-only episode after their date (n); the fewest more items
    right for one system and wrong for the other than the reverse with which the exact McNemar
    test passes at 0.05 on n items; a full run's comparisons (m); and, for each place k of
    Holm's order (every comparison before it rejected), the fewest items one way, none the other,
    that a result there needs."""
    questions = [json.loads(line) for line in (data / "questions.jsonl").read_text(
        encoding="utf-8").splitlines()]
    gold = {json.loads(line)["id"]: json.loads(line)
            for line in (data / "gold.jsonl").read_text(encoding="utf-8").splitlines()}
    n = sum(1 for q in questions if (q["family"], q["stratum"], q["corpus"]) == (
        "F1", "change", "eth") and not gold[q["id"]]["meta"]["h_minus_header_only_after_d"])
    m = len(HEADLINE_SYSTEMS) * len(CORPORA)
    lead = min(b - c for b in range(n + 1) for c in range(n + 1 - b)
               if b >= c and mcnemar_exact(b, c) < ALPHA)

    def passes(b: int, k: int) -> bool:  # b items one way, none the other, at Holm's place k
        p = [0.0] * (k - 1) + [mcnemar_exact(b, 0)] + [1.0] * (m - k)
        return holm(p)[k - 1] < ALPHA

    need = [min(b for b in range(n + 1) if passes(b, k)) for k in range(1, m + 1)]
    return {"n": n, "m": m, "lead": lead, "need": need}


def power_gaps(readme: str, data: Path) -> list[tuple]:
    """Each part of ``POWER`` that the README does not state as the data and the scorer's test
    and correction give it: (part, what the README says, what holds)."""
    held = power_numbers(data)
    n, m, lead, need = held["n"], held["m"], held["lead"], held["need"]
    firsts: dict[int, int] = {}
    for k, b in enumerate(need, 1):
        firsts.setdefault(b, k)
    steps = [x for b, k in list(firsts.items())[1:] for x in (b, k)]
    only_last = need[-1] if len(need) > 1 and need[-2] > need[-1] else None
    found = [re.search(pattern, readme) for pattern in POWER]
    gaps = []
    if found[0] is None:
        gaps.append(("the necessary condition", None, (n, lead)))
    else:
        said_n, said_lead, passing, *failing = found[0].groups()
        pairs = [(int(b), int(c)) for b, c in re.findall(r"(\d+) to (\d+)", passing)]
        b, c = map(int, failing)
        if (int(said_n), int(said_lead)) != (n, lead):
            gaps.append(("the necessary condition", (int(said_n), int(said_lead)), (n, lead)))
        gaps += [("a pair said to pass", pair, mcnemar_exact(*pair)) for pair in pairs
                 if sum(pair) > n or mcnemar_exact(*pair) >= ALPHA]
        if b + c > n or b - c < lead or mcnemar_exact(b, c) < ALPHA:  # leads enough, still fails
            gaps.append(("the pair said not to pass", (b, c), mcnemar_exact(b, c)))
    for part, pattern_found, numbers in (
        ("Holm's multiplier", found[1], (m, m + 1)),
        ("the fewest by Holm place", found[2], (n, need[0], *steps)),
        ("the closing sentence", found[3], (lead, n, only_last, 0, need[0], 0)),
    ):
        said = None if pattern_found is None else tuple(int(g) for g in pattern_found.groups())
        if said != numbers:
            gaps.append((part, said, numbers))
    return gaps


# The network rule (sections 8 and 11, and the README's step 2) in its egress form, phrase by
# phrase: the routes out it allows (the answering model's API, and the construction model's for a
# system whose store a language model builds), the embedding model inside the environment and no
# reranking model, the services a system runs itself, the files of this repository and the
# sources out of reach, the provider-side tools and the committed record of the policy.
NETWORK_RULE = (
    "in an environment from which no network route leads out except to the answering model's "
    "API",
    "for a system whose store a language model builds, the construction model's API",
    "the embedding model runs inside that environment",
    "no system calls a reranking model",
    "run inside that environment and hold only the system's own software and models and what the "
    "system wrote there",
    "nor any file of this repository other than the system's own adapter and configuration",
    "nor the corpora's live sources",
    "their mirrors or a clone of them can be reached",
    "provider-side tool",
    "network policy",
)
# Earlier forms of the rule: network access, which a system's own services would break; a route
# out to an embedding or a reranking API, which the registered embedding model, run inside the
# environment, and the absence of a reranker leave no use for; and this repository out of reach
# as a whole, which would put a system's own adapter out of its own reach.
STALE_NETWORK_WORDS = ("no network access except", "the embedding model's API",
                       "the reranker's API", "nor this repository, nor")


def network_rule_gaps(registration: str, readme: str) -> list[str]:
    """Where sections 8 and 11 or the README do not state the network rule in its egress form, or
    still state it in an earlier form (``STALE_NETWORK_WORDS``)."""
    places = {"section 8": registration_section(registration, 8),
              "section 11": registration_section(registration, 11), "README.md": readme}
    gaps = [f"{where}: {phrase}" for where, text in places.items() for phrase in NETWORK_RULE
            if phrase not in text]
    return gaps + [f"{where}: still says {word!r}" for where, text in places.items()
                   for word in STALE_NETWORK_WORDS if word in text]


def test_the_registration_and_the_readme_state_the_network_rule_as_egress():
    registration = (ROOT / "PREREGISTRATION.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert network_rule_gaps(registration, readme) == []


def test_the_readme_rebuilds_with_the_registered_budget_and_its_price_basis():
    """The README's rebuild command gives the Graphiti budget and the price basis that
    data/v0.1/MANIFEST.json records (section 13), so that it reproduces the registered manifest;
    every Graphiti budget the README and the registration state in US dollars is that one; and
    every date they give the list prices on is the basis's, the day the budget was set (section
    15's copy of the manifest, which the freeze replaces, aside)."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    registration = (ROOT / "PREREGISTRATION.md").read_text(encoding="utf-8")
    manifest = json.loads((ROOT / "data" / "v0.1" / "MANIFEST.json").read_text(encoding="utf-8"))
    command = re.search(r"uv run asofbench build .*?(?=\n```)", readme, re.DOTALL).group(0)
    budget = re.search(r"--graphiti-budget-usd (\S+)", command).group(1)
    basis = re.search(r"--price-basis '([^']*)'", command).group(1)
    assert (float(budget), basis) == (manifest["graphiti_budget_usd"],
                                      manifest["graphiti_budget_basis"])
    stated = re.findall(r"budget of ([\d,.]+) US dollars", readme + registration)
    assert stated and {float(s.replace(",", "")) for s in stated} == {
        manifest["graphiti_budget_usd"]}, stated
    day = re.search(r"list prices on (\d{4}-\d{2}-\d{2})", manifest["graphiti_budget_basis"])
    outside = (registration.split("\n## 15. Manifest\n", 1)[0]
               + registration.split("\n## 16. ", 1)[1])
    dated = re.findall(r"list prices (?:on|of) (\d{4}-\d{2}-\d{2})", readme + outside)
    assert day is not None and len(dated) >= 2 and set(dated) == {day.group(1)}, dated


# The four model flags of ``asofbench score`` and the values section 11 registers for them, as the
# scorer holds them (``REGISTERED_*``): only a run that gives all four can be a registered run.
MODEL_FLAGS = {"--answering-model": REGISTERED_ANSWERING_MODEL,
               "--answering-settings": REGISTERED_ANSWERING_SETTINGS,
               "--construction-model": REGISTERED_CONSTRUCTION_MODEL,
               "--embedding-model": REGISTERED_EMBEDDING_MODEL}
SENSITIVITY_MODEL = "claude-sonnet-5-5"  # section 11: the answering model of the sensitivity check


def readme_score_commands(readme: str) -> list[dict[str, str]]:
    """Each ``asofbench score`` command the README gives, as its options and their values: those in
    its bash blocks, continuation lines joined, then those it gives inline, as the sensitivity
    check's options are (an inline span counts when it gives every model flag)."""
    commands = []
    lines = [line for block in re.findall(r"```bash\n(.*?)\n```", readme, re.DOTALL)
             for line in block.replace("\\\n", " ").splitlines() if "asofbench score " in line]
    spans = [span for span in re.findall(r"`([^`\n]+)`", readme)
             if all(f"{flag} " in span for flag in MODEL_FLAGS)]
    for words in [shlex.split(text) for text in lines + spans]:
        commands.append({word: words[i + 1] for i, word in enumerate(words[:-1])
                         if word.startswith("--")})
    return commands


# The values either document may give a model flag: the registered one, the sensitivity check's
# answering model, none where a run needs no such model, and the placeholder that names the option.
ALLOWED_MODEL_VALUES = {
    "--answering-model": {REGISTERED_ANSWERING_MODEL, SENSITIVITY_MODEL, "ID"},
    "--answering-settings": {REGISTERED_ANSWERING_SETTINGS, "TEXT"},
    "--construction-model": {REGISTERED_CONSTRUCTION_MODEL, "none", "ID"},
    "--embedding-model": {REGISTERED_EMBEDDING_MODEL, "none", "ID"},
}


def registered_model_gaps(registration: str, readme: str) -> list[str]:
    """Where section 11 or the README's score commands do not give the registered values of the
    four model flags: section 11 names each flag with its value, as the scored runs give it; every
    value that the registration or the README gives a model flag, and every answering settings the
    README quotes, is a registered one (or none, the sensitivity check's model or a placeholder);
    the README's registered full run, its first command, gives all four; every README command
    gives the registered answering settings, and a construction or embedding model that is the
    registered one or none; the sensitivity check gives its own answering model and the
    registered three others; and the registered full run and the sensitivity check both give the
    training cutoff and the two files beside the H- and CF streams as ``REGISTERED_RUN_OPTIONS``
    gives them."""
    section = registration_section(registration, 11)
    gaps = []
    for flag, value in [*MODEL_FLAGS.items(), ("--answering-model", SENSITIVITY_MODEL)]:
        written = f'`{flag} "{value}"`' if " " in value else f"`{flag} {value}`"
        if written not in section:
            gaps.append(f"section 11 does not give {written}")
    for where, text in (("PREREGISTRATION.md", registration), ("README.md", readme)):
        for flag, allowed in ALLOWED_MODEL_VALUES.items():
            for quoted, bare in re.findall(rf'{flag} (?:"([^"]*)"|([^\s`"]+))', text):
                if (quoted or bare) not in allowed:
                    gaps.append(f"{where} gives {flag} {quoted or bare!r}")
        gaps += [f"{where} quotes the answering settings {said!r}"
                 for said in re.findall(r'`"(effort [^"`]*)"`', text)
                 if said != REGISTERED_ANSWERING_SETTINGS]
    commands = readme_score_commands(readme)
    if len(commands) < 3:
        return [*gaps, f"the README gives {len(commands)} score commands, not 3 or more"]
    gaps += [f"the registered full run gives {flag} {commands[0].get(flag)!r}, not {value!r}"
             for flag, value in MODEL_FLAGS.items() if commands[0].get(flag) != value]
    for n, command in enumerate(commands, 1):
        if command.get("--answering-settings") != REGISTERED_ANSWERING_SETTINGS:
            gaps.append(f"score command {n} gives the answering settings "
                        f"{command.get('--answering-settings')!r}")
        gaps += [f"score command {n} gives {flag} {command.get(flag)!r}"
                 for flag in ("--construction-model", "--embedding-model")
                 if command.get(flag) not in (MODEL_FLAGS[flag], "none")]
    checks = [c for c in commands if c.get("--answering-model") == SENSITIVITY_MODEL]
    if len(checks) != 1 or any(checks[0].get(flag) != MODEL_FLAGS[flag]
                               for flag in MODEL_FLAGS if flag != "--answering-model"):
        gaps.append(f"the README gives {len(checks)} sensitivity checks with the registered "
                    "settings, construction model and embedding model, not one")
    # The registered runs, the full run and the sensitivity check, also pass the training cutoff
    # and the two files beside the H- and CF streams (sections 10 and 12).
    for name, command in [("the registered full run", commands[0]),
                          *[("the sensitivity check", check) for check in checks]]:
        gaps += [f"{name} gives {flag} {command.get(flag)!r}, not {value!r}"
                 for flag, value in REGISTERED_RUN_OPTIONS.items() if command.get(flag) != value]
    return gaps


# The options that the registered runs pass besides the models (sections 10 and 12): the training
# cutoff and the files that the stream builder writes beside the H- and CF streams.
REGISTERED_RUN_OPTIONS = {"--training-cutoff": "2026-07-01",
                          "--message-cues": "streams/h-minus.jsonl.message_cues.json",
                          "--cf-messages": "streams/cf.jsonl.cf_messages.json"}


def test_the_registration_and_the_readme_give_the_registered_models():
    """The four values the scorer holds as the registered ones (``REGISTERED_ANSWERING_MODEL``,
    ``REGISTERED_ANSWERING_SETTINGS``, ``REGISTERED_CONSTRUCTION_MODEL`` and
    ``REGISTERED_EMBEDDING_MODEL``) are those section 11 registers and those the README's score
    commands give, letter for letter, since the scorer compares them so."""
    registration = (ROOT / "PREREGISTRATION.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert registered_model_gaps(registration, readme) == []


def readme_quotes(readme: str) -> list[str]:
    """The lines the README quotes from results.md: its block quotes, without the marker."""
    return [line[2:] for line in readme.splitlines() if line.startswith("> ")]


def example_answers(data: Path) -> dict[str, list[dict]]:
    """Answer rows for the README's three score examples on ``data``: every compared system and
    grep at D on every H- F1-change item, for the registered full run (``asofbench-answers``);
    bm25 alone on every item (``asofbench-answers-bm25``); and the six pilot systems on the pilot's
    items, for the sensitivity check. Half the answers are right; three runs each."""
    questions = [json.loads(line) for line in (data / "questions.jsonl").read_text(
        encoding="utf-8").splitlines()]
    gold = {g["id"]: g["gold"] for g in map(json.loads, (data / "gold.jsonl").read_text(
        encoding="utf-8").splitlines())}
    pilot = set(json.loads((data / "pilot.json").read_text(encoding="utf-8"))["ids"])

    def rows(system, found, arms=None):
        for k, q in enumerate(found):
            wrong = (next(c for c in q["choices"] if c != gold[q["id"]])
                     if q["answer_type"] == "status" else None)
            for arm in [a for a in q["arms"] if arms is None or a in arms]:
                for run in (1, 2, 3):
                    yield {"id": q["id"], "system": system, "arm": arm, "run": run,
                           "answer": gold[q["id"]] if k % 2 else wrong}

    change = [q for q in questions if (q["family"], q["stratum"]) == ("F1", "change")]
    piloted = [q for q in questions if q["id"] in pilot]
    return {
        "asofbench-answers": [r for s in (HEADLINE_BASELINE, *HEADLINE_SYSTEMS)
                              for r in rows(s, change, {"H-"})],
        "asofbench-answers-bm25": list(rows("bm25", questions)),
        "sensitivity": [r for s in ("closed-book", "bm25", "grep-at-d", "long-context",
                                    "hipporag2", "graphiti") for r in rows(s, piloted)],
    }


def quote_gaps(readme: str, written: list[str]) -> list[str]:
    """Each line the README quotes from results.md that no results.md in ``written`` holds."""
    lines = {line for text in written for line in text.splitlines()}
    quotes = readme_quotes(readme)
    return [quote for quote in quotes if not any(quote in line for line in lines)] or (
        [] if quotes else ["the README quotes no results.md line"])


def test_the_readme_quotes_what_the_scorer_writes_for_its_own_examples(tmp_path):
    """Every line the README quotes from results.md is a line, or a part of one, that the scorer
    writes when the README's own score commands (the registered full run, the run of bm25 alone and
    the sensitivity check) are run on data/v0.1 with answers in the README's places; the two files
    beside the streams, which a unit run does not build, are left out, and no quoted line depends
    on them."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    data = ROOT / "data" / "v0.1"
    answers = example_answers(data)
    written = []
    for n, command in enumerate(readme_score_commands(readme)):
        given = command.get("--answers", "")
        name = next((key for key in ("asofbench-answers-bm25", "asofbench-answers")
                     if f"/{key}/" in given), "sensitivity")
        path = tmp_path / f"{n}-{name}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in answers[name]), encoding="utf-8")
        options = {k: v for k, v in command.items()
                   if k not in ("--data", "--answers", "--out", "--message-cues", "--cf-messages")}
        if "--items" in options:
            options["--items"] = str(ROOT / options["--items"])
        out = tmp_path / f"out-{n}"
        assert cli.main(["score", "--data", str(data), "--answers", str(path), "--out", str(out),
                         *(word for pair in options.items() for word in pair)]) == 0, command
        written.append((out / "results.md").read_text(encoding="utf-8"))
    assert len(written) >= 3
    assert quote_gaps(readme, written) == []


# The registered pilot run's earlier definition (section 13 and the README at prereg-v0.1.1),
# under which a run of every item, or of any list that held every F1 item of pilot.json, was it.
STALE_PILOT_WORDS = ("with no `--items`, or with any `--items` list that holds them all",
                     "whose scope holds every F1 item of pilot.json", "or of every item")
# The registered pilot run's scope (section 13 and the README), in the words both use.
PILOT_SCOPE = ("is exactly the items of pilot.json, given with `--items` (`--items "
               "data/v0.1/pilot.json`, or a list of exactly those item ids in any order)")


def pilot_scope_reasons(data: Path) -> tuple[str, str]:
    """The reasons the scorer gives on ``data`` (``scoring._pilot_reasons``) why a run of every
    item, and a run of pilot.json's PEP items alone, are not the registered pilot run, on the
    registered models and build."""
    from asofbench import scoring

    questions = {q["id"]: q for q in map(json.loads, (data / "questions.jsonl").read_text(
        encoding="utf-8").splitlines())}
    listed, _ = scoring._items(data / "pilot.json", questions)
    families = tuple(sorted({q["family"] for q in questions.values()}))
    models = dict(zip(("answering_model", "answering_settings", "construction_model",
                       "embedding_model"), MODEL_FLAGS.values(), strict=True))
    pep = frozenset(i for i in listed if questions[i]["corpus"] == "pep")
    (every,), (alone,) = (scoring._pilot_reasons(listed, scoring.Scope(families, items=items),
                                                 questions, scoring.pilot_systems(), models, [])
                          for items in (None, pep))
    return every, alone


def decision_gaps(registration: str, readme: str, data: Path = ROOT / "data" / "v0.1") -> list[str]:
    """Where section 13 or the README does not give gates.json's ``decision`` for a pilot that is
    not evaluable as the scorer's one constructor of it (``scoring._not_evaluable``) writes it:
    "not evaluable: " and the reasons, then the scorer's sentence on the registered pilot run
    (``REGISTERED_PILOT``) and its sentence on the six pilot systems, each quoted as the code writes
    it, all joined by "; ". Also where either does not say that the decision is a top-level field
    in every shape, or define the registered pilot run's scope as exactly the items of pilot.json
    given with --items (``PILOT_SCOPE``), so that a run of every item is never it; where section 13
    does not quote the reasons the scorer gives on data/v0.1 for a run of every item and for a run
    of pilot.json's PEP items (``pilot_scope_reasons``); and where either still gives the earlier
    definition (``STALE_PILOT_WORDS``)."""
    from asofbench import scoring

    built = scoring._not_evaluable("accuracy_mean_over_runs", {}, {}, ["bm25"], [], {},
                                   ["a reason"])["decision"]
    head, *parts = built.split("; ")
    assert head == "not evaluable: a reason" and parts[0] == "no H- F1 answers from bm25", built
    assert parts[1] == scoring.REGISTERED_PILOT, built
    places = {"section 13": registration_section(registration, 13), "README.md": readme}
    top = "in both shapes the decision is gates.json's top-level field `decision`"
    gaps = [f"{where} does not quote {phrase!r}" for where, text in places.items()
            for phrase in ('"not evaluable: "', f'"{scoring.REGISTERED_PILOT}"', f'"{parts[2]}"',
                           'all joined by "; "', top, PILOT_SCOPE)
            if phrase not in text]
    every, alone = pilot_scope_reasons(data)
    for where, phrases in (
        ("section 13", ("A run with no `--items`, which scores every item, is therefore never the "
                        "registered pilot run", f'"{every}" in a run of every item',
                        f'"{alone}" in a run of pilot.json\'s PEP items alone')),
        ("README.md", ("A run of every item, the registered full run included, is never the "
                       "registered pilot run",)),
    ):
        gaps += [f"{where} does not say {phrase!r}" for phrase in phrases
                 if phrase not in places[where]]
    return gaps + [f"{where} still says {word!r}" for where, text in places.items()
                   for word in STALE_PILOT_WORDS if word in text]


def test_the_registration_and_the_readme_give_the_decision_of_a_pilot_that_is_not_evaluable():
    """Section 13 and the README say what gates.json's ``decision`` holds for a pilot that is not
    evaluable, in the scorer's own words."""
    registration = (ROOT / "PREREGISTRATION.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert decision_gaps(registration, readme) == []


# The CC0 1.0 legal code as Creative Commons serves it in plain text
# (https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt, fetched on 2026-10-31, the text
# its legal code page links as "Plain Text"), byte for byte the LICENSE.md of the EIP and ERC
# repositories at v0.1's pins.
CC0_LEGAL_CODE_SHA256 = "a2010f343487d3f7618affe54f789f5487602331c0a8d03f49e9a7c547cf0499"


def test_the_licences_are_the_mit_licence_and_the_cc0_legal_code():
    """LICENSE-DATA is the CC0 1.0 legal code and nothing else, so that licence detectors recognise
    it, and git converts the line endings of neither licence file (.gitattributes), so that a
    checkout on any platform holds those bytes; the README says which files it covers, the
    published results and every file published beside them, the label files included, among them;
    the labelling guidelines say that annotators label on the understanding that their label files
    are dedicated so; the package's metadata names both licences and both files; and CITATION.cff,
    whose licence list the CFF format reads as alternatives, names the dataset's licence alone and
    both in its abstract, with the same files."""
    data = (ROOT / "LICENSE-DATA").read_bytes()
    assert hashlib.sha256(data).hexdigest() == CC0_LEGAL_CODE_SHA256
    assert data.startswith(b"Creative Commons Legal Code\n\nCC0 1.0 Universal\n")
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
    assert "LICENSE -text" in attributes and "LICENSE-DATA -text" in attributes
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    licences = readme.split("\n## Data and licences\n", 1)[1].split("\n## ", 1)[0]
    assert "[LICENSE-DATA](LICENSE-DATA) holds the CC0 1.0 legal code" in licences
    assert "the published results of v0.1 (results.md, results.json and gates.json" in licences
    assert "every file published beside the results report" in licences
    assert "the label files included, other than the code and documents above" in licences
    assert "covers those files and no other file of this repository" in licences
    assert "TODO(author): confirm the data licence" not in readme
    guidelines = (ROOT / "labelling" / "GUIDELINES.md").read_text(encoding="utf-8")
    assert ("Annotators label on the understanding that the label files, notes included, are "
            "dedicated to the public domain under CC0 1.0") in guidelines
    assert "an annotator who does not agree does not label" in guidelines
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert (project["license"], project["license-files"]) == ("MIT AND CC0-1.0",
                                                              ["LICENSE", "LICENSE-DATA"])
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert re.search(r"^license: CC0-1\.0$", citation, re.MULTILINE)
    abstract = re.search(r"^abstract: (.*)$", citation, re.MULTILINE).group(1)
    assert "CC0 1.0" in abstract and "MIT licence" in abstract and "published results" in abstract
    assert "every file published beside the results report, the label files included" in abstract


# The README's F1-recent counts by the training cutoff ("What the scores mean"), each as a pattern
# of its own words: the two groups, the June 2026 items of the first, and the gold that is the
# status at S in the second.
README_RECENT = (
    r"recorded before 2026-07-01 \((\d+) of the (\d+) PEP and (\d+) of the (\d+) Ethereum items\) "
    r"or on or after it \((\d+) and (\d+)\)",
    r"the first holds (\d+) PEP and (\d+) Ethereum items whose transitions were recorded in June "
    r"2026",
    r"For all (\d+) PEP items of the second group, and for (\d+) of its (\d+) Ethereum items, the "
    r"gold is the status at S",
)


def readme_recent_counts(readme: str, data: Path) -> tuple[list, list]:
    """The README's F1-recent counts (``README_RECENT``) and the same counts from data/v0.1's gold
    (``meta.interval_start`` and ``meta.gold_equals_status_at_s``)."""
    questions = [json.loads(line) for line in (data / "questions.jsonl").read_text(
        encoding="utf-8").splitlines()]
    meta = {g["id"]: g["meta"] for g in map(json.loads, (data / "gold.jsonl").read_text(
        encoding="utf-8").splitlines())}
    recent = {c: [meta[q["id"]] for q in questions
                  if (q["family"], q["stratum"], q["corpus"]) == ("F1", "recent", c)]
              for c in ("pep", "eth")}
    before = {c: [m for m in found if m["interval_start"][:10] < TRAINING_CUTOFF]
              for c, found in recent.items()}
    after = {c: [m for m in found if m["interval_start"][:10] >= TRAINING_CUTOFF]
             for c, found in recent.items()}
    june = {c: sum(m["interval_start"][:7] == "2026-06" for m in found)
            for c, found in before.items()}
    at_s = {c: sum(m["gold_equals_status_at_s"] for m in found) for c, found in after.items()}
    held = [len(before["pep"]), len(recent["pep"]), len(before["eth"]), len(recent["eth"]),
            len(after["pep"]), len(after["eth"]), june["pep"], june["eth"], at_s["pep"],
            at_s["eth"], len(after["eth"])]
    stated = []
    for pattern in README_RECENT:
        found = re.search(pattern, readme)
        stated += [None] if found is None else [int(g) for g in found.groups()]
    return stated, held


def test_the_readme_states_the_f1_recent_counts_that_data_v0_1_holds():
    # What the scores mean: the F1-recent items before the training cutoff and on or after it, the
    # June 2026 items, and those whose gold is the status at S, as gold.jsonl's meta gives them.
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    stated, held = readme_recent_counts(readme, ROOT / "data" / "v0.1")
    assert stated == held


def test_the_readme_states_how_large_a_difference_the_ethereum_reading_can_show():
    # Known limitations: on the 23 Ethereum items left without header-only episodes, the bound
    # follows Holm's step-down order, so it depends on where the comparison's p falls.
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert power_gaps(readme, ROOT / "data" / "v0.1") == []


# The counts the registration gives at v0.1's pins, each as a pattern of its own words; a share
# in parentheses is compared at the precision the text prints it.
STATED = {
    "F1-change floors": r"the floor is (\w+), (\d+) of (\d+) PEP items \((\d\.\d+)\) and (\d+) of "
                        r"(\d+) Ethereum items \((\d\.\d+)\)",
    "pilot floors": r"for the pilot, (\w+), (\d+) of (\d+) PEP items \((\d\.\d+)\) and (\d+) of "
                    r"(\d+) Ethereum items \((\d\.\d+)\)",
    "marks": r"that takes (\d+) of (\d+) PEP items or (\d+) of (\d+) Ethereum items in a full "
             r"run, and (\d+) of (\d+) or (\d+) of (\d+) in the pilot",
    "F1-recent at S": r"it does for (\d+) of the (\d+) PEP and (\d+) of the (\d+) Ethereum "
                      r"F1-recent items",
    "F1 items": r"the share of the (\d+) F1 items",
    "F2 recoverable": r"(\d+) of the (\d+) F2 items are recoverable in H\+ \((\d+) of the (\d+) "
                      r"positives\) and (\d+) in H. \((\d+) of the (\d+) positives\)",
    "CF decision words": r"or supersede \((\d+) of the (\d+) at v0\.1's pins\)",
    "CF header cues": r"(\d+) of the (\d+) CF items have a header cue, (\d+) Ethereum and "
                      r"(\d+) PEP",
    "edges": r"the file has ([\d,]+) edges: requires ([\d,]+) Ethereum and (\d+) PEP, and "
             r"supersedes (\d+) PEP",
    "header-only flags": r"the flag is true for (\d+) of the (\d+) F1 items, among them (\d+) of "
                         r"the (\d+) H. F1-change items \((\d+) PEP and (\d+) Ethereum\), so the "
                         r"first reading rests on (\d+) PEP and (\d+) Ethereum items",
    "training cutoff": r"The registered runs pass `--training-cutoff (\d{4}-\d{2}-\d{2})`, the "
                       r"first day after that month, for both models; at v0\.1's pins the "
                       r"transitions of (\d+) of the (\d+) PEP and (\d+) of the (\d+) Ethereum "
                       r"F1-recent items were recorded on or after it",
    "pilot floor": r"the pilot's trivial-answer floor, (\w+) on (\d+) of its (\d+) F1-change "
                   r"items",
    "F2 positive ranks": r"F2's positives, which are not sampled, are ranked (\d+) to (\d+) in "
                         r"number order",
}
# Section 10: the day the registered runs pass as --training-cutoff, the first day after the
# answering models' stated training-data cutoff month.
TRAINING_CUTOFF = "2026-07-01"


def stated_counts(text: str) -> dict[str, tuple | None]:
    """Each count of ``STATED`` as the registration writes it (a word stays a word), or None
    when the registration no longer says it."""
    def value(group: str):
        return int(group.replace(",", "")) if re.fullmatch(r"[\d,]+", group) else group

    found = {}
    for what, pattern in STATED.items():
        match = re.search(pattern, text)
        found[what] = None if match is None else tuple(value(g) for g in match.groups())
    return found


def data_counts(data: Path) -> dict[str, tuple]:
    """The same counts, read from the registered data files."""
    def records(name):
        return [json.loads(line) for line in (data / name).read_text(encoding="utf-8").splitlines()]

    questions = {q["id"]: q for q in records("questions.jsonl")}
    gold = {g["id"]: g for g in records("gold.jsonl")}
    pilot = set(json.loads((data / "pilot.json").read_text(encoding="utf-8"))["ids"])

    def ids(family, stratum=None, corpus=None, within=None):
        return [i for i, q in questions.items() if q["family"] == family
                and stratum in (None, q["stratum"]) and corpus in (None, q["corpus"])
                and (within is None or i in within)]

    def floor(found):  # the most common gold, how many items have it, of how many, and the share
        (value, n), = Counter(json.dumps(gold[i]["gold"]) for i in found).most_common(1)
        return json.loads(value), n, len(found), Fraction(n, len(found))

    def floors(pep, eth):  # as the registration writes them: one gold value for both corpora
        return (pep[0] if pep[0] == eth[0] else f"{pep[0]} and {eth[0]}", *pep[1:], *eth[1:])

    def marked_from(right, n):  # the fewest right items that the one-sided test marks
        return next(k for k in range(n + 1) if binomial_tail(k, n, Fraction(right, n)) < ALPHA), n

    full = [floor(ids("F1", "change", c)) for c in ("pep", "eth")]
    scoped = [floor(ids("F1", "change", c, pilot)) for c in ("pep", "eth")]
    recent = [ids("F1", "recent", c) for c in ("pep", "eth")]
    f2, positives = ids("F2"), ids("F2", "positive")
    cf = ids("CF")
    cued = [i for i in cf if gold[i]["meta"]["header_cue"]]
    edges = Counter((e["relation"], e["corpus"]) for e in records("edges.jsonl"))

    def recoverable(found, arm):
        return sum(gold[i]["meta"]["recoverable"][arm] for i in found)

    def flagged(found):  # a header-only H- episode after the item's date (sections 8 and 12)
        return sum(gold[i]["meta"]["h_minus_header_only_after_d"] for i in found)

    change = [ids("F1", "change", c) for c in ("pep", "eth")]

    def after_cutoff(found):  # the scorer's rule: the transition's day on or after the cutoff
        return sum(gold[i]["meta"]["interval_start"][:10] >= TRAINING_CUTOFF for i in found)

    def ranked_by_number(found):  # (first, last) rank when the ranks count 0, 1, ... by number
        ranks = [gold[i]["meta"]["rank"] for i in found]
        numbers = [int(questions[i]["proposal"].split("-")[1]) for i in found]
        if ranks != list(range(len(found))) or numbers != sorted(numbers):
            return ("not ranked 0, 1, ... in number order",)
        return ranks[0], ranks[-1]

    return {
        "F1-change floors": floors(*full),
        "pilot floors": floors(*scoped),
        "marks": tuple(x for _, right, n, _ in (*full, *scoped) for x in marked_from(right, n)),
        "F1-recent at S": tuple(x for found in recent for x in (
            sum(gold[i]["meta"]["gold_equals_status_at_s"] for i in found), len(found))),
        "F1 items": (len(ids("F1")),),
        "F2 recoverable": (recoverable(f2, "H+"), len(f2), recoverable(positives, "H+"),
                           len(positives), recoverable(f2, "H-"), recoverable(positives, "H-"),
                           len(positives)),
        "CF decision words": (sum(gold[i]["meta"]["decision_word"] for i in cf), len(cf)),
        "CF header cues": (len(cued), len(cf), sum(questions[i]["corpus"] == "eth" for i in cued),
                           sum(questions[i]["corpus"] == "pep" for i in cued)),
        "edges": (sum(edges.values()), edges[("requires", "eth")], edges[("requires", "pep")],
                  edges[("supersedes", "pep")]),
        "header-only flags": (flagged(ids("F1")), len(ids("F1")), flagged(change[0] + change[1]),
                              len(change[0]) + len(change[1]), flagged(change[0]),
                              flagged(change[1]), len(change[0]) - flagged(change[0]),
                              len(change[1]) - flagged(change[1])),
        "training cutoff": (TRAINING_CUTOFF, *(x for found in recent
                                               for x in (after_cutoff(found), len(found)))),
        "pilot floor": floor(ids("F1", "change", within=pilot))[:3],
        "F2 positive ranks": ranked_by_number(positives),
    }


def agrees(stated: tuple | None, data: tuple) -> bool:
    """Whether the registration's words give the data's values: a share (a Fraction in the data)
    rounded to as many decimals as the text prints, everything else exactly."""
    if stated is None or len(stated) != len(data):
        return False
    for written, value in zip(stated, data, strict=True):
        if isinstance(value, Fraction):
            places = len(written.split(".")[1]) if isinstance(written, str) else -1
            if places < 1 or f"{float(value):.{places}f}" != written:
                return False
        elif written != value:
            return False
    return True


def counts_that_differ(text: str, data: Path) -> dict[str, tuple]:
    """Each count the registration gives that the data does not hold: (text, data)."""
    stated, held = stated_counts(text), data_counts(data)
    return {what: (stated[what], held[what]) for what in held if not agrees(stated[what],
                                                                             held[what])}


def test_the_registration_states_the_counts_that_data_v0_1_holds():
    # The counts that sections 7, 9, 10, 12 and 13 give at v0.1's pins are those of the
    # registered data, so a change to a rule or a rebuild cannot leave the registration's numbers
    # behind.
    text = (ROOT / "PREREGISTRATION.md").read_text(encoding="utf-8")
    assert counts_that_differ(text, ROOT / "data" / "v0.1") == {}


# The numbers the registration gives about the pinned clones themselves (sections 2, 3 and 5),
# each as a pattern of its own words.
STATED_AT_PINS = {
    "pull merges": r"In python/peps, (\d+) first-parent merges named \"Merge branch 'master' of "
                   r"github\.com:python/peps\" are pull merges: [^\n]*? Together they hide changes "
                   r"to (\d+) PEPs, over spans of up to (\d+) days between the first hidden change "
                   r"and its merge; the largest, ([0-9a-f]{7}), dates changes made from "
                   r"(\d{4}-\d{2}-\d{2}) at (\d{4}-\d{2}-\d{2})",
    "Superseded in PEP 1": r"the commit that added Superseded to PEP 1 on (\d{4}-\d{2}-\d{2}) also "
                           r"changed PEPs (\d+), (\d+) and (\d+) from Replaced to Superseded \(and "
                           r"PEP (\d+) from Rejected to Superseded\)",
    "PEP licences": r"of its ([\d,]+) PEP files, (\d+) are in the public domain or under CC0, "
                    r"(\d+) use the Open Publication License, (\d+) state no licence, and PEP "
                    r"(\d+), the April Fool joke",
    "PEP licence files": r"no top-level licence file \(two tooling directories, `([^`]+)` and "
                         r"`([^`]+)`, carry their own\)",
    "Ethereum licences": r"(\d+) EIP files, EIP-(\d+) and EIP-(\d+), state CC BY-NC-SA 4\.0 "
                         r"instead, (\d+) proposal files \((\d+) EIP and (\d+) ERC\) state no "
                         r"licence of their own, and the (\d+) EIP files that only mark a proposal "
                         r"as moved",
    "inversions": r"(\d+) of the ([\d,]+) first-parent commits up to the PEP repository's pin "
                  r"carry a committer date earlier than their parent's, one of them by ([\d,]+) "
                  r"days, about (\w+) years",
    "asset licences": r"\((\w+) of their asset directories carry licence files of their own: CC "
                      r"BY 4\.0 for (\w+), CC BY-NC-ND 4\.0 for (\w+), MIT for (\w+), CC0 1\.0 "
                      r"for (\w+) and Apache 2\.0 for (\w+); the benchmark reads no asset\)",
}
SMALL_NUMBERS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
                 8: "eight", 9: "nine", 10: "ten"}
# The licence an asset directory's licence file states, by the words that open its text.
ASSET_LICENCES = (("Attribution-NonCommercial-NoDerivatives 4.0 International", "CC BY-NC-ND 4.0"),
                  ("Attribution 4.0 International", "CC BY 4.0"),
                  ("CC0 1.0 Universal", "CC0 1.0"),
                  ("Apache License", "Apache 2.0"),
                  ("Permission is hereby granted, free of charge", "MIT"))
PROPOSAL_FILES = {"peps": re.compile(r"(?:peps/)?pep-(\d{4})\.(?:txt|rst)"),
                  "EIPs": re.compile(r"(?:EIPS/)?eip-(\d+)\.(?:md|mediawiki)"),
                  "ERCs": re.compile(r"ERCS/erc-(\d+)\.md")}
NUMBER_WORDS = {9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}


def read_git(clones: Path, repo: str, *args: str, given: bytes | None = None) -> bytes:
    """A read-only git command in one of the clones, with no configuration and no lock taken
    (githistory.git_env), its output as bytes."""
    return subprocess.run(["git", "-C", str(clones / repo), *args], env=git_env(), input=given,
                          capture_output=True, check=True).stdout


def proposal_texts(clones: Path, repo: str, pin: str) -> dict[int, str]:
    """Every proposal file of ``repo`` at ``pin`` (section 2's patterns), by number, as text."""
    names = [name for name in read_git(clones, repo, "ls-tree", "-r", "--name-only", pin)
             .decode("utf-8").splitlines() if PROPOSAL_FILES[repo].fullmatch(name)]
    raw = read_git(clones, repo, "cat-file", "--batch",
                   given="".join(f"{pin}:{name}\n" for name in names).encode("utf-8"))
    texts, pos = {}, 0
    for name in names:
        end = raw.index(b"\n", pos)
        size = int(raw[pos:end].split()[2])
        number = int(PROPOSAL_FILES[repo].fullmatch(name).group(1))
        texts[number] = raw[end + 1:end + 1 + size].decode("utf-8", errors="replace")
        pos = end + 1 + size + 1
    assert len(texts) == len(names), repo  # one file per number at the pins
    return texts


def status_line(text: str) -> str:
    found = re.search(r"(?im)^status:[ \t]*(.*)$", text)
    return "" if found is None else found.group(1).strip().casefold()


def pep_licence(text: str) -> str:
    """A PEP's licence, as its own Copyright section states it: the text after its last line
    that reads ``Copyright`` (the heading, in the .txt and the .rst form). PEP 401's section is
    an April Fool joke, as the PEP is; a PEP without the section, or whose section names neither
    the public domain nor CC0 nor the Open Publication License, states no licence."""
    if "april fool" in status_line(text):
        return "April Fool"
    lines = text.split("\n")
    headings = [i for i, line in enumerate(lines) if line.strip() == "Copyright"]
    section = "\n".join(lines[headings[-1] + 1:]).casefold() if headings else ""
    if "open publication license" in section:
        return "Open Publication License"
    return "public domain or CC0" if "public domain" in section or "cc0" in section else "none"


def eth_licence(text: str) -> str:
    """An EIP or ERC file's licence: a file whose status is Moved only marks a proposal moved to
    the ERC repository; otherwise CC BY-NC-SA 4.0 when the file names it anywhere, CC0 or the
    public domain when it names either, and none when it names none of them."""
    folded = text.casefold()
    if status_line(text) == "moved":
        return "moved"
    if "by-nc-sa" in folded or "attribution-noncommercial-sharealike" in folded:
        return "CC BY-NC-SA 4.0"
    if "cc0" in folded or "public domain" in folded:
        return "CC0 or public domain"
    return "none"


def stated_value(group: str) -> int | str:
    """A number as the registration writes it (``10,959``) as an int; any other word as written."""
    return int(group.replace(",", "")) if re.fullmatch(r"[\d,]+", group) else group


def superseded_commit(clones: Path, pin: str) -> tuple:
    """Section 5: the first first-parent commit that adds Superseded to PEP 1's file, its date in
    UTC, and the PEPs whose status it changes to Superseded, first those it changes from
    Replaced, then those from Rejected; any other status change it makes follows, so it shows."""
    first = read_git(clones, "peps", "log", "--first-parent", "--reverse", "-S", "Superseded",
                     "--format=%H %ct", pin, "--", "pep-0001.txt").decode().splitlines()[0]
    sha, committed = first.split()
    diff = read_git(clones, "peps", "show", "--first-parent", "--no-ext-diff", "--format=",
                    sha).decode("utf-8", errors="replace")
    changes, number, before = [], None, None
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            found = re.search(r" b/(?:peps/)?pep-(\d{4})\.(?:txt|rst)$", line)
            number, before = None if found is None else int(found.group(1)), None
        elif number is not None and line.startswith("-Status:"):
            before = line.split(":", 1)[1].strip()
        elif number is not None and line.startswith("+Status:"):
            changes.append((number, before, line.split(":", 1)[1].strip()))

    def into_superseded(old: str) -> list[int]:
        return sorted(n for n, was, now in changes if (was, now) == (old, "Superseded"))

    others = [change for change in changes if change[1:] not in (("Replaced", "Superseded"),
                                                                 ("Rejected", "Superseded"))]
    day = datetime.fromtimestamp(int(committed), UTC).date().isoformat()
    return (day, *into_superseded("Replaced"), *into_superseded("Rejected"), *others)


def asset_licences(clones: Path, pins: dict[str, str]) -> Counter:
    """The licence of each asset directory of the EIP and ERC repositories that holds a licence
    file (LICENSE, LICENCE or COPYING, in any letter case and with any extension) at its pin,
    counted by licence; a directory with a licence file that names none of ``ASSET_LICENCES`` is
    counted as unknown."""
    found = Counter()
    for repo in ("EIPs", "ERCs"):
        names = read_git(clones, repo, "ls-tree", "-r", "--name-only", pins[repo]).decode()
        for name in names.splitlines():
            directory, _, base = name.rpartition("/")
            if not (directory.startswith("assets/")
                    and re.fullmatch(r"(?i)(licen[cs]e|copying)(\.(md|txt|rst))?", base)):
                continue
            text = read_git(clones, repo, "show", f"{pins[repo]}:{name}").decode(
                "utf-8", errors="replace")
            found[next((kind for words, kind in ASSET_LICENCES if words in text), "unknown")] += 1
    return found


def clone_counts(clones: Path, data: Path) -> dict[str, tuple]:
    """The numbers of ``STATED_AT_PINS``, read from the clones at the pins of ``data``'s
    manifest; the pull merges as tests/githistory.py reads them, as for test_corpus_history."""
    pins = json.loads((data / "MANIFEST.json").read_text(encoding="utf-8"))["pins"]
    merges = pull_merges(clones, "peps", pins["peps"])
    largest = max(merges, key=lambda m: m["recorded"] - m["earliest"])
    peps = {n: pep_licence(t) for n, t in proposal_texts(clones, "peps", pins["peps"]).items()}
    by_kind = Counter(peps.values())
    (april_fool,) = [n for n, kind in peps.items() if kind == "April Fool"]
    files = read_git(clones, "peps", "ls-tree", "-r", "--name-only", pins["peps"]).decode()
    licence_dirs = sorted(name.rsplit("/", 1)[0] if "/" in name else "" for name in
                          files.splitlines() if name.rsplit("/", 1)[-1].startswith(
                              ("LICENCE", "LICENSE", "COPYING")))
    eips = {n: eth_licence(t) for n, t in proposal_texts(clones, "EIPs", pins["EIPs"]).items()}
    ercs = {n: eth_licence(t) for n, t in proposal_texts(clones, "ERCs", pins["ERCs"]).items()}
    non_commercial = sorted(n for n, kind in eips.items() if kind == "CC BY-NC-SA 4.0")
    no_eip, no_erc = (sum(kind == "none" for kind in found.values()) for found in (eips, ercs))
    committed = [int(t) for t in read_git(clones, "peps", "log", "--first-parent", "--reverse",
                                          "--format=%ct", pins["peps"]).split()]
    gaps = [parent - child for parent, child in zip(committed, committed[1:], strict=False)
            if child < parent]
    assets = asset_licences(clones, pins)
    return {
        "pull merges": (len(merges), len(set().union(*(m["numbers"] for m in merges))),
                        int((largest["recorded"] - largest["earliest"]) / 86400),
                        stated_value(largest["sha"][:7]), largest["start"].isoformat(),
                        largest["end"].isoformat()),
        "Superseded in PEP 1": superseded_commit(clones, pins["peps"]),
        "PEP licences": (len(peps), by_kind["public domain or CC0"],
                         by_kind["Open Publication License"], by_kind["none"], april_fool),
        "PEP licence files": tuple(licence_dirs),
        "Ethereum licences": (len(non_commercial), *non_commercial, no_eip + no_erc, no_eip, no_erc,
                              sum(kind == "moved" for kind in eips.values())),
        "inversions": (len(gaps), len(committed), int(max(gaps) / 86400),
                       NUMBER_WORDS[round(max(gaps) / 86400 / 365.25)]),
        "asset licences": (("unknown", dict(assets)) if assets["unknown"] else tuple(
            SMALL_NUMBERS[n] for n in (sum(assets.values()), *(assets[kind] for kind in (
                "CC BY 4.0", "CC BY-NC-ND 4.0", "MIT", "CC0 1.0", "Apache 2.0"))))),
    }


def clone_counts_that_differ(text: str, held: dict[str, tuple]) -> dict[str, tuple]:
    """Each number of ``STATED_AT_PINS`` that the registration ``text`` does not give as the
    clones hold it (``clone_counts``): (text, clones)."""
    stated = {what: re.search(pattern, text) for what, pattern in STATED_AT_PINS.items()}
    stated = {what: None if found is None else tuple(map(stated_value, found.groups()))
              for what, found in stated.items()}
    return {what: (stated[what], held[what]) for what in held if stated[what] != held[what]}


@pytest.mark.corpus
def test_the_registration_states_the_counts_that_the_clones_hold(clones):
    # Sections 2, 3 and 5's numbers about the repositories at v0.1's pins, from the clones with
    # read-only git: the licences each proposal file states, the PEP chain's dates and pull
    # merges, and the commit that brought Superseded into PEP 1.
    text = (ROOT / "PREREGISTRATION.md").read_text(encoding="utf-8")
    assert clone_counts_that_differ(text, clone_counts(clones, ROOT / "data" / "v0.1")) == {}


def f1_flags_that_differ(data: Path) -> tuple[int, list[str]]:
    """The number of F1 items in ``data`` and those whose ``meta.gold_equals_status_at_s`` is not
    the comparison of their gold with ``meta.status_at_s``, the fact the flag repeats."""
    def records(name):
        return [json.loads(line) for line in (data / name).read_text(encoding="utf-8").splitlines()]

    f1 = {q["id"] for q in records("questions.jsonl") if q["family"] == "F1"}
    rows = [g for g in records("gold.jsonl") if g["id"] in f1]
    meta = {g["id"]: g["meta"] for g in rows}
    differ = [g["id"] for g in rows
              if meta[g["id"]]["gold_equals_status_at_s"] is not (g["gold"] == meta[g["id"]]
                                                                   ["status_at_s"])]
    return len(rows), differ


def test_the_f1_flag_repeats_the_gold_against_the_status_at_s_in_data_v0_1():
    # Section 10 splits F1-recent by this flag, and every F1 record also holds the status at S it
    # summarises; the two must agree on every registered item, so a build change cannot let them
    # drift apart unseen.
    assert f1_flags_that_differ(ROOT / "data" / "v0.1") == (503, [])


def test_the_scorers_registered_build_is_data_v0_1():
    """Sections 12 and 13: the scorer's registered build (``REGISTERED_DATA``) is data/v0.1, by
    the SHA-256s of its questions, gold and pilot list, as its manifest registers them; a rebuild
    that changes one of them fails here until the constant follows."""
    data = ROOT / "data" / "v0.1"
    manifest = json.loads((data / "MANIFEST.json").read_text(encoding="utf-8"))
    held = {name: hashlib.sha256((data / name).read_bytes()).hexdigest()
            for name in ("questions.jsonl", "gold.jsonl", "pilot.json")}
    assert REGISTERED_DATA == held == {name: manifest["files"][name]["sha256"] for name in held}


def listing_digest(files: dict[str, bytes]) -> str:
    """Section 15's SHA-256 over the source files, as documented: the SHA-256 of one line
    ``<SHA-256 of the file>  <name>`` per file, in the byte order of the names (what
    ``LC_ALL=C sha256sum *.py`` prints)."""
    lines = "".join(f"{hashlib.sha256(data).hexdigest()}  {name}\n"
                    for name, data in sorted(files.items()))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def test_the_frozen_manifest_must_name_a_commit_of_the_history_that_holds_its_source(tmp_path):
    """The guard's checks on a frozen manifest, with read-only git: its ``source_commit`` is a
    commit of the repository, an ancestor of HEAD, and holds src/asofbench files whose digest
    (section 15's rule) is its ``source_sha256``."""
    repo = TempRepo(tmp_path / "checkout")
    source = {"__init__.py": b"x = 1\n", "cli.py": b"y = 2\n"}
    for name, data in source.items():
        repo.write(f"src/asofbench/{name}", data)
    repo.write("src/asofbench/notes.txt", "not source\n")
    first = repo.commit("the source", "2020-01-01T00:00:00Z")
    repo.branch("side")
    repo.write("README.md", "Read me.\n")
    head = repo.commit("the manifest", "2020-01-02T00:00:00Z")
    repo.checkout("side")
    repo.write("NOTES.md", "Elsewhere.\n")
    side = repo.commit("another line", "2020-01-03T00:00:00Z")
    repo.checkout("main")
    digest = listing_digest(source)
    good = {"source_commit": first, "source_sha256": digest}
    assert source_commit_problems(repo.path, good) == []
    assert source_commit_problems(repo.path, {**good, "source_commit": head}) == []
    assert source_commit_problems(repo.path, {**good, "source_sha256": "0" * 64}) == [
        f"src/asofbench at source_commit {first} has the digest {digest}, not source_sha256 "
        f"{'0' * 64}"]
    assert source_commit_problems(repo.path, {**good, "source_commit": side}) == [
        f"source_commit {side} is not an ancestor of HEAD"]
    for unseen in ("1" * 40, "uncommitted"):
        assert source_commit_problems(repo.path, {**good, "source_commit": unseen}) == [
            f"source_commit {unseen!r} is not a commit of this repository"]


def test_the_guard_reads_no_history_in_a_shallow_clone(tmp_path):
    """A shallow clone (``git clone --depth 1``) holds the commit that carries the manifest but
    not, necessarily, the earlier commit the manifest names, so the guard checks no history there,
    as outside a checkout, rather than report that commit as unknown."""
    repo = TempRepo(tmp_path / "origin")
    source = {"__init__.py": b"x = 1\n"}
    repo.write("src/asofbench/__init__.py", source["__init__.py"])
    first = repo.commit("the source", "2020-01-01T00:00:00Z")
    repo.write("README.md", "Read me.\n")
    repo.commit("the manifest", "2020-01-02T00:00:00Z")
    generator = {"source_commit": first, "source_sha256": listing_digest(source)}
    assert frozen_history_problems(repo.path, generator) == []
    shallow = clone(repo.path, tmp_path / "shallow", "--depth", "1")
    assert source_commit_problems(shallow, generator) == [
        f"source_commit {first!r} is not a commit of this repository"]
    assert frozen_history_problems(shallow, generator) is None
    assert frozen_history_problems(tmp_path / "not-a-checkout", generator) is None


def test_the_guard_and_the_manifest_take_the_same_source_files(tmp_path):
    """Section 15's rule, in git as on disk: the source files are the regular ``*.py`` files of
    src/asofbench. At a commit the guard reads them with ``gitwalk.files_at``, which keeps no
    symbolic link; on disk ``source_files`` refuses a ``*.py`` entry that is a symbolic link, or
    anything else that is not a regular file, with a clear error, rather than read through it.
    The package's own source holds regular files only, so the rule leaves its files as they were."""
    repo = TempRepo(tmp_path / "checkout")
    repo.write("src/asofbench/__init__.py", "x = 1\n")
    (repo.path / "src" / "asofbench" / "link.py").symlink_to("__init__.py")
    sha = repo.commit("a module that is a symbolic link", "2020-01-01T00:00:00Z")
    assert source_commit_problems(repo.path, {"source_commit": sha, "source_sha256": listing_digest(
        {"__init__.py": b"x = 1\n"})}) == []
    with pytest.raises(ManifestError, match=r"link\.py is a symbolic link"):
        source_files(repo.path / "src" / "asofbench")
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "a.py").write_bytes(b"x = 1\n")
    (tree / "notes.txt").write_text("not source\n", encoding="utf-8")
    (tree / "__pycache__").mkdir()
    assert source_files(tree) == {"a.py": b"x = 1\n"}
    (tree / "c.py").mkdir()
    with pytest.raises(ManifestError, match=r"c\.py is not a regular file"):
        source_files(tree)
    package = ROOT / "src" / "asofbench"
    assert set(source_files(package)) == {p.name for p in package.iterdir()
                                          if p.name.endswith(".py")}


def test_build_stops_before_writing_on_a_source_file_that_is_not_regular(world, tmp_path,
                                                                        monkeypatch, capsys):
    """``asofbench build`` reports a generator module that is a symbolic link with exit 2 and a
    message, before it reads the clones or writes anything."""
    from asofbench import manifest

    installed = tmp_path / "site-packages" / "asofbench"
    installed.mkdir(parents=True)
    for path in (ROOT / "src" / "asofbench").glob("*.py"):
        (installed / path.name).write_bytes(path.read_bytes())
    (tmp_path / "elsewhere.py").write_bytes(b"y = 2\n")
    (installed / "extra.py").symlink_to(tmp_path / "elsewhere.py")
    monkeypatch.setattr(manifest, "_PACKAGE", installed)
    monkeypatch.setattr(manifest, "_REPO", installed.parents[1])
    out = tmp_path / "out"
    assert cli.main(["build", "--clones", str(world), "--out", str(out)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("asofbench build: ") and "extra.py is a symbolic link" in err
    assert not out.exists()


def test_provenance_names_the_commit_only_for_a_clean_checkout(tmp_path):
    source = {"__init__.py": b"x = 1\n", "cli.py": b"y = 2\n"}
    repo = TempRepo(tmp_path / "checkout")
    for name, data in source.items():
        repo.write(f"src/asofbench/{name}", data)
    repo.write("README.md", "Read me.\n")
    sha = repo.commit("the source", "2020-01-01T00:00:00Z")
    clean = {"source_commit": sha, "source_dirty": False, "source_sha256": listing_digest(source)}
    assert provenance(repo.path) == clean
    # Files that are not the generator's source leave it clean: a build's output, notes, caches.
    repo.write("data/v0.1/gold.jsonl", "{}\n")
    repo.write("src/asofbench/notes.txt", "notes\n")
    repo.write("src/asofbench/__pycache__/cli.cpython-312.pyc", b"\0")
    assert provenance(repo.path) == clean
    # An edited, added or removed source file makes it dirty, and names no commit.
    repo.write("src/asofbench/cli.py", "y = 3\n")
    assert provenance(repo.path) == {"source_commit": "uncommitted", "source_dirty": True,
                                     "source_sha256": listing_digest({**source,
                                                                      "cli.py": b"y = 3\n"})}
    repo.write("src/asofbench/cli.py", source["cli.py"])
    assert provenance(repo.path) == clean
    repo.write("src/asofbench/extra.py", "z = 1\n")
    assert provenance(repo.path)["source_dirty"] is True
    repo.rm("src/asofbench/extra.py")
    repo.rm("src/asofbench/cli.py")
    assert provenance(repo.path)["source_commit"] == "uncommitted"


def test_provenance_of_a_tree_that_is_not_a_checkout_of_its_own(tmp_path):
    outer = TempRepo(tmp_path / "outer")
    outer.write("vendored/src/asofbench/__init__.py", "x = 1\n")
    outer.commit("a copy inside another repository", "2020-01-01T00:00:00Z")
    inside = provenance(outer.path / "vendored")
    assert inside == {"source_commit": "uncommitted", "source_dirty": True,
                      "source_sha256": listing_digest({"__init__.py": b"x = 1\n"})}
    unborn = TempRepo(tmp_path / "unborn")
    unborn.write("src/asofbench/__init__.py", "x = 1\n")
    assert provenance(unborn.path)["source_commit"] == "uncommitted"
    nowhere = provenance(tmp_path / "not-a-repo")
    assert (nowhere["source_commit"], nowhere["source_dirty"]) == ("uncommitted", True)
    # The generator's own tree: its digest is over its own source files.
    files = {p.name: p.read_bytes() for p in (ROOT / "src" / "asofbench").glob("*.py")}
    assert provenance()["source_sha256"] == listing_digest(files)


def test_provenance_hashes_the_package_it_runs_from_wherever_it_is_installed(tmp_path, monkeypatch):
    """An installed copy (a wheel, ``pip install .``) sits in no checkout: its digest is still over
    its own files, not over an empty listing, and it names no commit."""
    from asofbench import manifest

    installed = tmp_path / "site-packages" / "asofbench"
    installed.mkdir(parents=True)
    files = {p.name: p.read_bytes() for p in (ROOT / "src" / "asofbench").glob("*.py")}
    for name, data in files.items():
        (installed / name).write_bytes(data)
    monkeypatch.setattr(manifest, "_PACKAGE", installed, raising=False)
    monkeypatch.setattr(manifest, "_REPO", installed.parents[1])
    assert provenance() == {"source_commit": "uncommitted", "source_dirty": True,
                            "source_sha256": listing_digest(files)}


@pytest.fixture(scope="module")
def built_with_streams(world, tmp_path_factory):
    root = tmp_path_factory.mktemp("with-streams")
    argv = ["build", "--clones", str(world), "--out", str(root / "v0.1"),
            "--streams", str(root / "streams"), "--graphiti-budget-usd", "250.5",
            "--price-basis", "list prices of the answering model on 2026-10-29"]
    assert cli.main(argv) == 0
    return root / "v0.1", root / "streams"


STREAM_FILES = {"H+": "h-plus.jsonl", "H-": "h-minus.jsonl", "CF": "cf.jsonl"}
# The files the stream builder writes beside each arm's stream, as <stream>.<suffix>.
SIDECARS = {"H+": ("stats.json",), "H-": ("stats.json", "message_cues.json"),
            "CF": ("stats.json", "cf_messages.json")}


def sidecars_on_disk(streams: Path, arm: str) -> dict:
    found = {}
    for suffix in SIDECARS[arm]:
        data = Path(f"{streams / STREAM_FILES[arm]}.{suffix}").read_bytes()
        found[suffix] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    return found


def test_a_build_with_streams_registers_each_arm(built_with_streams, built):
    data, streams = built_with_streams
    manifest = json.loads((data / "MANIFEST.json").read_text(encoding="utf-8"))
    assert set(manifest["streams"]) == set(STREAM_FILES)
    for arm, name in STREAM_FILES.items():
        raw = (streams / name).read_bytes()
        rows = [json.loads(line) for line in raw.decode("utf-8").split("\n") if line]
        chars = sum(len(row["text"]) for row in rows)
        stats = {"episodes": len(rows), "chars": chars, "tokens_est": chars // 4,
                 "sha256": hashlib.sha256(raw).hexdigest()}
        # Section 15: the stream's stats, and each file beside it with its SHA-256 and size.
        assert len(rows) > 5, arm
        assert manifest["streams"][arm] == {**stats, "sidecars": sidecars_on_disk(streams, arm)}
        assert json.loads(Path(f"{streams / name}.stats.json").read_text()) == stats, arm
    # Nothing else is written beside the streams.
    written = {f"{name}.{suffix}" for arm, name in STREAM_FILES.items() for suffix in SIDECARS[arm]}
    assert {p.name for p in streams.iterdir()} == written | set(STREAM_FILES.values())
    # Section 13: the budget and its price basis, as given.
    assert manifest["graphiti_budget_usd"] == 250.5
    assert manifest["graphiti_budget_basis"] == "list prices of the answering model on 2026-10-29"
    # The data files are those of a build without streams; only the manifest records more.
    for name in FILES:
        assert (data / name).read_bytes() == (built / name).read_bytes(), name


def test_the_stream_command_checks_its_output_against_the_manifest(world, built_with_streams,
                                                                     tmp_path, capsys):
    data, streams = built_with_streams
    for arm, name in STREAM_FILES.items():
        out = tmp_path / name
        argv = ["stream", "--clones", str(world), "--data", str(data), "--arm", arm,
                "--out", str(out)]
        assert cli.main(argv) == 0, arm
        assert out.read_bytes() == (streams / name).read_bytes(), arm
    changed = tmp_path / "changed"
    shutil.copytree(data, changed)
    manifest = json.loads((changed / "MANIFEST.json").read_text(encoding="utf-8"))
    manifest["streams"]["H-"]["sha256"] = "0" * 64
    (changed / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    out = tmp_path / "again.jsonl"
    argv = ["stream", "--clones", str(world), "--data", str(changed), "--arm", "H-",
            "--out", str(out)]
    assert cli.main(argv) == 2
    err = capsys.readouterr().err
    assert "registered" in err and "0" * 64 in err
    assert not out.exists() and not Path(f"{out}.stats.json").exists()
    assert not Path(f"{out}.message_cues.json").exists()
    # Section 15: the files beside the stream are checked as the stream is. A registered file
    # that the command would not write the same way stops it, and nothing is written.
    for arm, suffix in (("H-", "message_cues.json"), ("CF", "cf_messages.json"),
                        ("H+", "stats.json")):
        changed = tmp_path / f"changed-{suffix}"
        shutil.copytree(data, changed)
        manifest = json.loads((changed / "MANIFEST.json").read_text(encoding="utf-8"))
        manifest["streams"][arm]["sidecars"][suffix]["sha256"] = "1" * 64
        (changed / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
        out = tmp_path / f"again-{arm}-{suffix}"
        argv = ["stream", "--clones", str(world), "--data", str(changed), "--arm", arm,
                "--out", str(out)]
        assert cli.main(argv) == 2, suffix
        err = capsys.readouterr().err
        assert suffix in err and "1" * 64 in err and "registered" in err, suffix
        assert not list(tmp_path.glob(f"again-{arm}-{suffix}*")), suffix


def test_score_checks_the_stream_companion_files_against_the_manifest(built_with_streams, built,
                                                                      tmp_path, capsys):
    """Section 15: ``asofbench score`` refuses a --message-cues or --cf-messages file whose
    SHA-256 differs from the one the build's manifest registers, and says in a visible line when
    either file is not given, or is not registered and so not checked."""
    data, streams = built_with_streams
    first = json.loads((data / "questions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    answers = tmp_path / "a.jsonl"
    answers.write_text(json.dumps({"id": first["id"], "system": "dense", "arm": first["arms"][0],
                                   "run": 1, "answer": "Draft"}) + "\n", encoding="utf-8")
    cues = streams / "h-minus.jsonl.message_cues.json"
    counts = streams / "cf.jsonl.cf_messages.json"

    def score(out, build, *extra):
        return cli.main(["score", "--data", str(build), "--answers", str(answers), "--out",
                         str(out), "--answering-model", "claude-haiku-5-5",
                         "--answering-settings", "effort medium, adaptive thinking, default "
                         "temperature", "--construction-model", "none", "--embedding-model",
                         "test-embedding-model", *extra])  # dense embeds text

    checked = tmp_path / "checked"
    assert score(checked, data, "--message-cues", str(cues), "--cf-messages", str(counts)) == 0
    results = json.loads((checked / "results.json").read_text(encoding="utf-8"))
    assert results["message_cues"]["checked_against_manifest"] is True
    assert results["cf_messages"]["checked_against_manifest"] is True
    text = (checked / "results.md").read_text(encoding="utf-8")
    assert f"H- message cues: {cues.name}, the file the build's manifest registers" in text
    assert f"CF message counts: {counts.name}, the file the build's manifest registers" in text
    # The same content written another way is another file: refused, and nothing is written.
    for option, given in (("--message-cues", cues), ("--cf-messages", counts)):
        edited = tmp_path / given.name
        edited.write_text(json.dumps(json.loads(given.read_text(encoding="utf-8")), indent=1),
                          encoding="utf-8")
        refused = tmp_path / f"refused{option}"
        assert score(refused, data, option, str(edited)) == 2
        err = capsys.readouterr().err
        assert given.name in err and "SHA-256" in err and "MANIFEST.json" in err, option
        assert not refused.exists()
    # Without either file, the report says so.
    plain = tmp_path / "plain"
    assert score(plain, data) == 0
    text = (plain / "results.md").read_text(encoding="utf-8")
    assert "CF message counts: no file given (--cf-messages)" in text
    assert "H- message cues: no file given (--message-cues)" in text
    # A build without streams registers no such file: it is read, and the report says that it
    # was not checked.
    unchecked = tmp_path / "unchecked"
    assert score(unchecked, built, "--message-cues", str(cues)) == 0
    results = json.loads((unchecked / "results.json").read_text(encoding="utf-8"))
    assert results["message_cues"]["checked_against_manifest"] is False
    assert "does not register" in (unchecked / "results.md").read_text(encoding="utf-8")


def test_a_registered_stream_may_carry_its_sidecars_and_only_its_own():
    from asofbench.manifest import ManifestError, registered_stream

    stats = {"episodes": 3, "chars": 9, "tokens_est": 2, "sha256": "a" * 64}
    entry = {"bytes": 10, "sha256": "b" * 64}
    plain = {"streams": {"H-": stats}}  # a manifest written before sidecars were registered
    assert registered_stream(plain, "H-", Path("m.json")) == stats
    full = {"streams": {"H-": {**stats, "sidecars": {"stats.json": entry,
                                                     "message_cues.json": entry}}}}
    assert registered_stream(full, "H-", Path("m.json"))["sidecars"]["stats.json"] == entry
    for sidecars in ({"stats.json": entry},  # the H- arm's cues are missing
                     {"stats.json": entry, "message_cues.json": entry, "cf_messages.json": entry},
                     {"stats.json": entry, "message_cues.json": {"sha256": "b" * 64}},
                     "stats.json"):
        with pytest.raises(ManifestError):
            registered_stream({"streams": {"H-": {**stats, "sidecars": sidecars}}}, "H-",
                              Path("m.json"))


@pytest.mark.parametrize("extra", [
    ["--graphiti-budget-usd", "100"],  # a budget without its price basis
    ["--price-basis", "list prices"],  # a basis without a budget
    ["--graphiti-budget-usd", "-5", "--price-basis", "list prices"],
    ["--graphiti-budget-usd", "0", "--price-basis", "list prices"],
    ["--graphiti-budget-usd", "nan", "--price-basis", "list prices"],
    ["--graphiti-budget-usd", "inf", "--price-basis", "list prices"],
    ["--graphiti-budget-usd", "100", "--price-basis", "  "],
])
def test_the_graphiti_budget_needs_a_positive_amount_and_its_price_basis(world, tmp_path, extra):
    with pytest.raises(SystemExit):
        cli.main(["build", "--clones", str(world), "--out", str(tmp_path / "out"), *extra])
    assert not (tmp_path / "out").exists()


def test_build_twice_is_byte_identical(world, built, tmp_path):
    again = tmp_path / "again"
    assert build(world, again) == 0
    for name in (*FILES, "MANIFEST.json"):
        assert (again / name).read_bytes() == (built / name).read_bytes(), name


def test_f1_and_cf_gold_match_git_show(built, world):
    gold = [json.loads(line) for line in (built / "gold.jsonl").read_text().splitlines()]
    checked = 0
    for row in gold:
        if not row["id"].startswith(("F1-", "CF-")):
            continue
        source = row["source"]
        assert source["commit"] == row["as_of_commits"][source["repo"]]
        text = git_show(world / source["repo"], f"{source['commit']}:{source['path']}")
        family = "pep" if source["repo"] == "peps" else "eth"
        status = normalize_status(family, parse_header(family, text).get("status"))
        expected = permute(family, status) if row["id"].startswith("CF-") else status
        assert row["gold"] == expected, row["id"]
        checked += 1
    assert checked > 5


def linked_clones(world: Path, root: Path, **own) -> Path:
    """A clones directory whose repositories are links to the world's, except those in ``own``,
    which are made by the callable given for them."""
    root.mkdir()
    for name in ("peps", "EIPs", "ERCs"):
        if name in own:
            own[name](root / name)
        else:
            (root / name).symlink_to(world / name, target_is_directory=True)
    return root


def test_build_refuses_a_shallow_clone(world, tmp_path, capsys):
    clones = linked_clones(world, tmp_path / "clones",
                           peps=lambda target: clone(world / "peps", target, "--depth", "1"))
    out = tmp_path / "out"
    assert build(clones, out) == 2
    err = capsys.readouterr().err
    assert err.startswith("asofbench build: ") and "shallow" in err and "peps" in err
    assert not out.exists()
    # The same history in full is read.
    full = linked_clones(world, tmp_path / "full", peps=lambda t: clone(world / "peps", t))
    assert build(full, tmp_path / "out-full") == 0


def test_build_refuses_a_directory_inside_another_repository(tmp_path, capsys):
    outer = TempRepo(tmp_path / "outer")
    for name in ("peps", "EIPs", "ERCs"):
        outer.write(f"{name}/pep-0001.txt", pep(1, "Draft"))
    outer.commit("three plain directories in one repository", "2020-01-01T00:00:00Z")
    assert build(outer.path, tmp_path / "out") == 2
    assert "top level" in capsys.readouterr().err


def test_build_refuses_a_clones_directory_that_is_not_there(tmp_path, capsys):
    assert build(tmp_path / "nowhere", tmp_path / "out") == 2
    err = capsys.readouterr().err
    assert "not a directory" in err and "Traceback" not in err


def test_build_checks_the_expected_pins(world, built, tmp_path, capsys):
    def build_against(manifest: Path, name: str) -> int:
        return cli.main(["build", "--clones", str(world), "--out", str(tmp_path / name),
                         "--expect-pins", str(manifest)])

    registered = built / "MANIFEST.json"
    assert build_against(registered, "same") == 0
    content = json.loads(registered.read_text(encoding="utf-8"))
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({**content, "pins": {**content["pins"], "EIPs": "0" * 40}}))
    assert build_against(wrong, "wrong") == 2
    err = capsys.readouterr().err
    assert "EIPs" in err and "0" * 40 in err and content["pins"]["EIPs"] in err
    assert not (tmp_path / "wrong").exists()
    no_erc_pin = tmp_path / "short.json"
    no_erc_pin.write_text(json.dumps({"pins": {k: v for k, v in content["pins"].items()
                                               if k != "ERCs"}}))
    assert build_against(no_erc_pin, "short") == 2
    assert "ERCs" in capsys.readouterr().err
    for name, text in (("bad.json", "{not json"), ("list.json", "[1, 2]"),
                       ("nopins.json", '{"pins": "x"}')):
        (tmp_path / name).write_text(text)
        assert build_against(tmp_path / name, name) == 2
        assert name in capsys.readouterr().err
    assert build_against(tmp_path / "missing.json", "missing") == 2
    assert "missing.json" in capsys.readouterr().err


def test_build_refuses_to_write_over_a_build_or_the_registered_manifest(world, built, tmp_path,
                                                                        capsys):
    """``asofbench build`` refuses an --out directory that already holds a MANIFEST.json, such as
    data/v0.1, or that is the directory of --expect-pins, with exit 2 and a message, before it
    writes anything, the streams included."""
    target = tmp_path / "v0.1"
    shutil.copytree(built, target)

    def files() -> dict:
        return {str(p.relative_to(tmp_path)): p.read_bytes() for p in sorted(tmp_path.rglob("*"))
                if p.is_file()}

    before = files()
    for extra, says in (
        ([], f"asofbench build: {target} already holds a MANIFEST.json"),
        (["--expect-pins", str(target / "MANIFEST.json")],
         f"asofbench build: {target} is the directory of --expect-pins "
         f"{target / 'MANIFEST.json'}"),
    ):
        argv = ["build", "--clones", str(world), "--out", str(target), "--streams",
                str(target / "streams"), *extra]
        assert cli.main(argv) == 2, extra
        err = capsys.readouterr().err
        assert err.startswith(says) and "build into a fresh directory" in err, err
        assert files() == before and not (target / "streams").exists()
    # The directory of --expect-pins, whatever the manifest there is called and however the
    # directory is named.
    pins = tmp_path / "pins"
    pins.mkdir()
    shutil.copy(built / "MANIFEST.json", pins / "registered.json")
    link = tmp_path / "link"
    link.symlink_to(pins, target_is_directory=True)
    for out in (pins, link):
        argv = ["build", "--clones", str(world), "--out", str(out), "--expect-pins",
                str(pins / "registered.json")]
        assert cli.main(argv) == 2, out
        assert "is the directory of --expect-pins" in capsys.readouterr().err
        assert sorted(p.name for p in pins.iterdir()) == ["registered.json"]
    # A fresh directory beside it is built as before.
    assert cli.main(["build", "--clones", str(world), "--out", str(tmp_path / "fresh"),
                     "--expect-pins", str(pins / "registered.json")]) == 0


def test_build_refuses_the_directory_of_expect_pins_in_another_letter_case(world, built, tmp_path,
                                                                          capsys):
    """On a volume that ignores letter case, as macOS's does by default, the directory of
    --expect-pins written in other letters is the same directory, and is refused."""
    pins = tmp_path / "pins"
    pins.mkdir()
    shutil.copy(built / "MANIFEST.json", pins / "registered.json")
    other = tmp_path / "PINS"
    if not other.exists():
        pytest.skip("the temporary directory's volume tells letter case apart, so PINS is "
                    "another directory")
    argv = ["build", "--clones", str(world), "--out", str(other), "--expect-pins",
            str(pins / "registered.json")]
    assert cli.main(argv) == 2
    assert "is the directory of --expect-pins" in capsys.readouterr().err
    assert sorted(p.name for p in pins.iterdir()) == ["registered.json"]


def test_build_manifest_needs_every_output(world, tmp_path):
    with pytest.raises(FileNotFoundError):
        build_manifest(tmp_path, load_corpus(world))


def test_stream_and_score_are_dispatched_lazily(monkeypatch):
    calls = []
    for name in ("stream", "scoring"):
        module = types.ModuleType(f"asofbench.{name}")
        module.main = lambda args, name=name: calls.append((name, args)) or 7
        monkeypatch.setitem(sys.modules, f"asofbench.{name}", module)
    stream_argv = ["stream", "--clones", "c", "--data", "d", "--arm", "H-", "--out", "s.jsonl"]
    assert cli.main(stream_argv) == 7
    assert cli.main(["score", "--data", "d", "--answers", "a.jsonl", "b.jsonl", "--out", "r",
                     "--answering-model", "claude-haiku-5-5", "--answering-settings",
                     "effort medium, adaptive thinking, default temperature",
                     "--construction-model", "none", "--embedding-model", "none"]) == 7
    (stream, stream_args), (score, score_args) = calls
    assert stream == "stream" and (stream_args.clones, stream_args.data) == (Path("c"), Path("d"))
    assert (stream_args.arm, stream_args.out) == ("H-", Path("s.jsonl"))
    assert score == "scoring" and score_args.answers == [Path("a.jsonl"), Path("b.jsonl")]
    assert (score_args.data, score_args.out) == (Path("d"), Path("r"))


def test_stream_arm_is_checked(monkeypatch):
    module = types.ModuleType("asofbench.stream")
    module.main = lambda args: 0
    monkeypatch.setitem(sys.modules, "asofbench.stream", module)
    with pytest.raises(SystemExit):
        cli.main(["stream", "--clones", "c", "--data", "d", "--arm", "H", "--out", "s"])


def test_the_minus_sign_names_the_h_minus_arm(monkeypatch):
    """The registration writes H- with a minus sign (U+2212); the commands read it as H-."""
    calls = []
    for name in ("stream", "materialize"):
        module = types.ModuleType(f"asofbench.{name}")
        module.main = lambda args, name=name: calls.append((name, args.arm)) or 0
        monkeypatch.setitem(sys.modules, f"asofbench.{name}", module)
    minus = "H" + chr(0x2212)
    assert cli.main(["stream", "--clones", "c", "--data", "d", "--arm", minus, "--out", "s"]) == 0
    assert cli.main(["materialize", "--stream", "s", "--as-of", "2020-01-01", "--out", "o",
                     "--arm", minus, "--data", "d"]) == 0
    assert calls == [("stream", "H-"), ("materialize", "H-")]


def test_a_command_is_required():
    with pytest.raises(SystemExit):
        cli.main([])
