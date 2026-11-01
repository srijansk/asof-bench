"""The build manifest, ``MANIFEST.json`` (spec section 15).

It records the pinned commit of each repository; the name, size, line count, record count and
SHA-256 of every generated file; item counts per family, stratum and corpus; the episode count,
character count, estimated tokens and SHA-256 of each arm's stream when the build writes the
streams (``--streams``), with the SHA-256 and size of each file the stream builder writes beside
the stream (``SIDECARS``), null otherwise; the generator's version, a SHA-256 over its source
files and the commit it was built from; and the Graphiti budget for the M0 gate in US dollars
with its price basis, null until they are set (section 13).

**Provenance.** The generator's source is the ``*.py`` files of its package directory
(``src/asofbench`` in a checkout), regular files only: a ``*.py`` entry that is a symbolic link
or not a regular file stops the build with a message. Their digest,
``source_sha256``, is the SHA-256 of one line ``<SHA-256 of the file>  <name>`` per file in the
byte order of the names, which is what ``LC_ALL=C sha256sum *.py`` prints in that directory (the
order of a shell's glob follows its collation, so another locale can give another order).
``source_commit`` is the commit HEAD names when the checkout is the top level of its own
repository and those files are, name for name and byte for byte, the ones HEAD holds; otherwise
it is ``"uncommitted"`` and ``source_dirty`` is true. Other files, such as a build's output, do
not count.
"""

from __future__ import annotations

import json
from collections import Counter
from hashlib import sha256
from pathlib import Path

from . import __version__
from .corpora import CANARY, SEED, S
from .gitwalk import files_at_head, head_commit, is_top_level
from .questions import dumps
from .timeline import Corpus

OUTPUT_FILES = (
    "questions.jsonl",
    "gold.jsonl",
    "pilot.json",
    "labelling/mention_pairs.jsonl",
    "timelines.jsonl",
    "edges.jsonl",
    "REPORT.md",
)
STREAM_STATS = ("episodes", "chars", "tokens_est", "sha256")  # per arm, under "streams"
# The files the stream builder writes beside each arm's stream, as ``<stream>.<suffix>``: its
# stats, the H- arm's message cues (section 12) and the CF arm's message counts (section 7). The
# manifest records each one's SHA-256 and size under ``streams.<arm>.sidecars``.
STATS_FILE, CUES_FILE, CF_MESSAGES_FILE = "stats.json", "message_cues.json", "cf_messages.json"
SIDECARS = {"H+": (STATS_FILE,), "H-": (STATS_FILE, CUES_FILE),
            "CF": (STATS_FILE, CF_MESSAGES_FILE)}
SOURCE_DIR = "src/asofbench"  # the generator's source, relative to the checkout
_PACKAGE = Path(__file__).resolve().parent  # the source files this module runs from
_REPO = _PACKAGE.parents[1]  # the asof-bench checkout, when the package runs from one


class ManifestError(Exception):
    """A manifest that cannot be read; the command reports it and exits 2."""


def read_manifest(path: Path) -> dict:
    """``path`` read as a JSON object, or ``ManifestError``."""
    path = Path(path)
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ManifestError(f"{path} not found") from None
    except (OSError, ValueError):  # a decoding error is a ValueError too
        raise ManifestError(f"{path} is not a JSON manifest") from None
    if not isinstance(content, dict):
        raise ManifestError(f"{path} is not a JSON object")
    return content


def registered_pins(manifest: dict, path: Path) -> dict[str, str]:
    """The ``pins`` of a manifest read from ``path``: repository -> commit."""
    pins = manifest.get("pins")
    if not isinstance(pins, dict) or not all(
        isinstance(repo, str) and isinstance(sha, str) for repo, sha in pins.items()
    ):
        raise ManifestError(f"{path} has no pins (an object of repository -> commit)")
    return pins


def _is_file_entry(entry: object) -> bool:
    return (isinstance(entry, dict) and set(entry) == {"bytes", "sha256"}
            and isinstance(entry["sha256"], str) and isinstance(entry["bytes"], int))


def registered_stream(manifest: dict | None, arm: str, path: Path) -> dict | None:
    """What a manifest read from ``path`` registers for ``arm``'s stream: its stats and, under
    ``sidecars``, the SHA-256 and size of each file beside it; None if it registers none (no
    manifest, a build without streams, or no entry for the arm). A manifest written before the
    sidecars were registered has the stats alone."""
    streams = None if manifest is None else manifest.get("streams")
    if streams is None:
        return None
    if not isinstance(streams, dict):
        raise ManifestError(f"{path}: streams is not an object of arm -> stats")
    entry = streams.get(arm)
    if entry is None:
        return None
    if not isinstance(entry, dict) or set(entry) - {"sidecars"} != set(STREAM_STATS):
        raise ManifestError(f"{path} registers the {arm} stream without exactly its "
                            f"{', '.join(STREAM_STATS)}")
    if "sidecars" in entry:
        sidecars, wanted = entry["sidecars"], set(SIDECARS.get(arm, ()))
        if (not isinstance(sidecars, dict) or set(sidecars) != wanted
                or not all(_is_file_entry(e) for e in sidecars.values())):
            raise ManifestError(f"{path} registers the files beside the {arm} stream without "
                                f"exactly {', '.join(sorted(wanted))}, each with its bytes and "
                                "sha256")
    return entry


def _is_source(name: str) -> bool:
    return name.endswith(".py") and not name.startswith(".")


def source_files(directory: Path) -> dict[str, bytes]:
    """The generator's source files in ``directory`` (its ``*.py`` files), by name: regular files
    only, as git holds them (``gitwalk.files_at``), so a ``*.py`` entry that is a symbolic link or
    anything else that is not a regular file is an error, never read through or left out."""
    directory = Path(directory)
    if not directory.is_dir():
        return {}
    found = {}
    for path in sorted(directory.iterdir()):
        if not _is_source(path.name):
            continue
        if path.is_symlink():
            raise ManifestError(f"{path}: {path.name} is a symbolic link, not a regular file; "
                                "the generator's source files are regular files")
        if not path.is_file():
            raise ManifestError(f"{path}: {path.name} is not a regular file; the generator's "
                                "source files are regular files")
        found[path.name] = path.read_bytes()
    return found


def source_sha256(files: dict[str, bytes]) -> str:
    """The SHA-256 of the lines ``<SHA-256 of the file>  <name>``, one per file in name order."""
    listing = "".join(f"{sha256(data).hexdigest()}  {name}\n"
                      for name, data in sorted(files.items()))
    return sha256(listing.encode("utf-8")).hexdigest()


def provenance(root: Path | None = None) -> dict:
    """``source_commit``, ``source_dirty`` and ``source_sha256`` of the checkout at ``root``; by
    default, of the files this module runs from, which an installed copy (a wheel) also hashes
    though it sits in no checkout. See the module's description."""
    if root is None:
        root, working = _REPO, source_files(_PACKAGE)
    else:
        root = Path(root)
        working = source_files(root / SOURCE_DIR)
    found = {"source_commit": "uncommitted", "source_dirty": True,
             "source_sha256": source_sha256(working)}
    head = head_commit(root) if is_top_level(root) else None
    if head is not None:
        committed = {name: data for name, data in files_at_head(root, SOURCE_DIR).items()
                     if _is_source(name)}
        if committed == working:
            found.update(source_commit=head, source_dirty=False)
    return found


def build_manifest(out: Path, corpus: Corpus, pools: dict | None = None, *,
                   streams: dict[str, dict] | None = None,
                   graphiti_budget: tuple[float, str] | None = None) -> dict:
    """Write ``out/MANIFEST.json`` for the files ``write_outputs`` wrote, and return it.

    ``pools`` (``Build.pools``) adds the eligible pool of each family, stratum and corpus.
    ``streams`` maps each arm to its stream's stats and the files beside it
    (``stream.write_streams``), and ``graphiti_budget`` is the M0 gate's budget in US dollars
    with its price basis; each stays null without them.
    """
    out = Path(out)
    files = {}
    for name in OUTPUT_FILES:
        data = (out / name).read_bytes()
        entry = {"sha256": sha256(data).hexdigest(), "bytes": len(data), "lines": data.count(b"\n")}
        if name.endswith(".jsonl"):
            entry["records"] = len(data.splitlines())
        elif name == "pilot.json":
            entry["items"] = len(json.loads(data)["ids"])
        files[name] = entry
    questions = (out / "questions.jsonl").read_text(encoding="utf-8").splitlines()
    counts = Counter()
    for line in questions:
        row = json.loads(line)
        counts[f"{row['family']}/{row['stratum']}/{row['corpus']}"] += 1
    pairs = (out / "labelling" / "mention_pairs.jsonl").read_text(encoding="utf-8").splitlines()
    budget, basis = (None, None) if graphiti_budget is None else graphiti_budget
    manifest = {
        "canary": CANARY,
        "files": files,
        "generator": {"name": "asofbench", "version": __version__, **provenance()},
        "graphiti_budget_basis": basis,
        "graphiti_budget_usd": budget,
        "item_counts": dict(sorted(counts.items())),
        "items": len(questions),
        "labelling_pairs": dict(sorted(Counter(json.loads(p)["corpus"] for p in pairs).items())),
        "pins": dict(sorted(corpus.pins.items())),
        "seed": SEED,
        "snapshot": S.isoformat(),
        "streams": None if streams is None else {
            arm: {key: stats[key] for key in (*STREAM_STATS, "sidecars")}
            for arm, stats in sorted(streams.items())
        },
    }
    if pools is not None:
        manifest["eligible_pools"] = {"/".join(key): n for key, n in sorted(pools.items())}
    with (out / "MANIFEST.json").open("w", encoding="utf-8", newline="\n") as f:
        f.write(dumps(manifest) + "\n")
    return manifest
