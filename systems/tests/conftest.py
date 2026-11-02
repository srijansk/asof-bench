"""Puts systems/ and systems/common/ on the import path, so that the tests import the run
environment's modules as ``env.egress.proxy``, ``env.embedder.server`` and ``env.netcheck``, and
the systems' shared package from systems/common/. Run with ``pytest systems/tests``. It also holds
the toy world with its three streams registered, which the tests of the harness share."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SYSTEMS = Path(__file__).resolve().parent.parent

for entry in (SYSTEMS, SYSTEMS / "common"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))


@pytest.fixture(scope="session")
def world(tmp_path_factory) -> Path:
    """The toy world's repositories (examples/toy_world.py), built into questions and the three
    arms' streams by `asofbench build --streams`, so that the manifest registers the streams:
    toy-data/ and streams/, and the clones in clones/. The tests read it and change nothing. It
    needs git and the asofbench package, which the modules that ask for it skip without."""
    import importlib.util

    from asofbench import cli

    spec = importlib.util.spec_from_file_location(
        "toy_world", SYSTEMS.parent / "examples" / "toy_world.py")
    toy_world = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(toy_world)
    root = tmp_path_factory.mktemp("toy-streams") / "world"
    clones = toy_world.toy_clones(root / "clones")
    assert cli.main(["build", "--clones", str(clones), "--out", str(root / "toy-data"),
                     "--streams", str(root / "streams")]) == 0
    return root
