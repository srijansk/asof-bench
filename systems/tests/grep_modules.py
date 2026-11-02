"""Grep at D's modules, for the tests that use them. systems/grep-at-d is no Python package (its
name is the registration's identifier, with hyphens), so each module is loaded from its file, once,
under a name of its own: the agent (agent.py, which the reader image holds as grep_at_d.agent),
the sandbox's command server (sandbox/command_server.py) and the host's runner of the dates
(run_dates.py). Standard library only; pytest collects no test here."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

GREP_AT_D = Path(__file__).resolve().parents[1] / "grep-at-d"


def load(name: str, path: Path) -> ModuleType:
    """The module at ``path``, loaded once under ``name``."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def agent() -> ModuleType:
    return load("grep_at_d_agent", GREP_AT_D / "agent.py")


def command_server() -> ModuleType:
    return load("grep_at_d_command_server", GREP_AT_D / "sandbox" / "command_server.py")


def run_dates() -> ModuleType:
    return load("grep_at_d_run_dates", GREP_AT_D / "run_dates.py")
