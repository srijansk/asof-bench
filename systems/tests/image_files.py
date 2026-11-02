"""A system's Dockerfile and its modules' imports, read as the build and the interpreter read them,
for the tests that guard each image's list of the shared package's files (test_graphiti_wiring.py
and test_reader_wiring.py). An image copies the shared package's files one by one, so a file that
its entry point imports and its Dockerfile lacks fails only when the container first runs that
entry point, and the tests that run inside the image read the files they are given, not the
image's own. Standard library only; pytest collects no test here."""

from __future__ import annotations

import ast
import re
from pathlib import Path


def copied_shared_files(dockerfile: str) -> set[str]:
    """The files of the shared package that the COPY instructions of ``dockerfile`` copy."""
    return copied_python_files(dockerfile, "common/asofbench_systems")


def copied_python_files(dockerfile: str, directory: str) -> set[str]:
    """The Python files of ``directory``, a path in the build's context, that the COPY
    instructions of ``dockerfile`` copy. Docker reads a line that starts with # as a comment
    wherever it falls, and a backslash at the end of a line as continuing the instruction on the
    next, so a path named in a comment or in another instruction copies nothing."""
    instructions, current = [], ""
    for line in dockerfile.splitlines():
        if line.lstrip().startswith("#"):
            continue
        if line.rstrip().endswith("\\"):
            current += line.rstrip()[:-1] + " "
        else:
            instructions.append(current + line)
            current = ""
    if current:
        instructions.append(current)
    copied = set()
    for instruction in instructions:
        words = instruction.split()
        if words and words[0].upper() == "COPY":
            sources = [word for word in words[1:] if not word.startswith("--")][:-1]
            copied |= {match[1] for source in sources
                       if (match := re.fullmatch(rf"{re.escape(directory)}/(\w+\.py)", source))}
    return copied


def shared_files_imported_by(path: Path, package: Path) -> set[str]:
    """The files of the shared package, at ``package``, that the module at ``path`` imports. A
    relative import names a shared file only in a module of the package itself, where a single
    dot is the package; in any other module it names that module's own package."""
    inside = path.parent == package
    files = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level == 0 and module == "asofbench_systems":
                files |= {f"{alias.name}.py" for alias in node.names}
            elif node.level == 0 and module.startswith("asofbench_systems."):
                files.add(f"{module.split('.')[1]}.py")
            elif node.level == 1 and inside and module:
                files.add(f"{module.split('.')[0]}.py")
            elif node.level == 1 and inside:
                files |= {f"{alias.name}.py" for alias in node.names}
        elif isinstance(node, ast.Import):
            files |= {f"{alias.name.split('.')[1]}.py" for alias in node.names
                      if alias.name.startswith("asofbench_systems.")}
    return files


def shared_files_needed(entries: list[Path], package: Path) -> set[str]:
    """The files of the shared package that the modules at ``entries`` import, and those that
    those import in turn, with the package's ``__init__.py``."""
    needed, pending = {"__init__.py"}, list(entries)
    while pending:
        for name in shared_files_imported_by(pending.pop(), package) - needed:
            needed.add(name)
            pending.append(package / name)
    return needed


def shared_file_differences(dockerfile: str, needed: set[str]) -> tuple[set[str], set[str]]:
    """The needed shared files that the Dockerfile does not copy, and those it copies that are
    not needed."""
    copied = copied_shared_files(dockerfile)
    return needed - copied, copied - needed
