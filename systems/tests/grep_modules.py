"""Grep at D's modules, for the tests that use them. systems/grep-at-d is no Python package by its
name (the registration's identifier, with hyphens), and grep at D's image holds its agent as the
package ``grep_at_d`` (/app/grep_at_d). So the tests import the directory under that same name:
``package`` puts a package ``grep_at_d`` whose path is systems/grep-at-d in ``sys.modules``, once,
and the modules are then ``grep_at_d.agent`` and ``grep_at_d.selftest``, as in the image, and
``grep_at_d.run_dates`` and ``grep_at_d.sandbox.command_server``, which the image does not hold.
Inside an image whose own ``grep_at_d`` is on the path, the tests still read the files they are
given. Standard library only; pytest collects no test here."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

GREP_AT_D = Path(__file__).resolve().parents[1] / "grep-at-d"


def package() -> ModuleType:
    """The package ``grep_at_d``, which is systems/grep-at-d."""
    found = sys.modules.get("grep_at_d")
    if found is None or list(getattr(found, "__path__", [])) != [str(GREP_AT_D)]:
        found = ModuleType("grep_at_d")
        found.__path__ = [str(GREP_AT_D)]
        sys.modules["grep_at_d"] = found
    return found


def module(name: str) -> ModuleType:
    package()
    return importlib.import_module(f"grep_at_d.{name}")


def agent() -> ModuleType:
    return module("agent")


def selftest() -> ModuleType:
    return module("selftest")


def command_server() -> ModuleType:
    return module("sandbox.command_server")


def run_dates() -> ModuleType:
    return module("run_dates")
