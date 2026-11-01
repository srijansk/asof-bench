"""Fixtures shared by the test modules."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def clones() -> Path:
    """The directory that holds the ``peps``, ``EIPs`` and ``ERCs`` clones.

    It comes from ``ASOFBENCH_CLONES``; every test that needs it is skipped when that is unset.
    """
    value = os.environ.get("ASOFBENCH_CLONES")
    if not value:
        pytest.skip("ASOFBENCH_CLONES is not set to the directory of the corpus clones")
    return Path(value)
