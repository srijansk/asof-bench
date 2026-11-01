"""Puts systems/ and systems/common/ on the import path, so that the tests import the run
environment's modules as ``env.egress.proxy``, ``env.embedder.server`` and ``env.netcheck``, and
the systems' shared package from systems/common/. Run with ``pytest systems/tests``."""

from __future__ import annotations

import sys
from pathlib import Path

SYSTEMS = Path(__file__).resolve().parent.parent

for entry in (SYSTEMS, SYSTEMS / "common"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))
