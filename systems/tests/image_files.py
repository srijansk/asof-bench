"""A system's Dockerfile and its modules' imports, read as the build and the interpreter read them,
for the tests that guard each image's list of the shared package's files (test_graphiti_wiring.py
and test_reader_wiring.py). An image copies the shared package's files one by one, so a file that
its entry point imports and its Dockerfile lacks fails only when the container first runs that
entry point, and the tests that run inside the image read the files they are given, not the
image's own. Also the blocks of a compose file, as the wiring tests read them (``block`` and
``items``). Standard library only; pytest collects no test here."""

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
    those import in turn. The package's ``__init__.py`` is always one of them, since importing
    any of its modules runs it first, and it is read for imports like any other shared file."""
    needed, pending = {"__init__.py"}, [package / "__init__.py", *entries]
    while pending:
        for name in shared_files_imported_by(pending.pop(), package) - needed:
            needed.add(name)
            pending.append(package / name)
    return needed


def package_files_needed(entry: Path, package: str, directory: Path) -> set[str]:
    """The modules of the package ``package``, whose files are in ``directory``, that the module
    at ``entry`` imports, at any depth of its code, and those that those import in turn, with
    ``entry``'s own file when it is in ``directory``: the files of ``directory`` that an image
    that runs ``entry`` needs."""
    needed = {entry.name} if entry.parent == directory else set()
    pending = [entry]
    while pending:
        for node in ast.walk(ast.parse(pending.pop().read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == package:
                names = [alias.name for alias in node.names]
            elif (isinstance(node, ast.ImportFrom) and node.level == 0
                  and (node.module or "").startswith(f"{package}.")):
                names = [node.module.split(".")[1]]
            elif isinstance(node, ast.Import):
                names = [alias.name.split(".")[1] for alias in node.names
                         if alias.name.startswith(f"{package}.")]
            for name in names:
                if f"{name}.py" not in needed:
                    needed.add(f"{name}.py")
                    pending.append(directory / f"{name}.py")
    return needed


def shared_file_differences(dockerfile: str, needed: set[str]) -> tuple[set[str], set[str]]:
    """The needed shared files that the Dockerfile does not copy, and those it copies that are
    not needed."""
    copied = copied_shared_files(dockerfile)
    return needed - copied, copied - needed


def block(text: str, name: str, indent: int) -> list[str]:
    """The lines of the YAML key ``name`` at ``indent`` spaces, from its line to the next line that
    is indented as little, blank lines at its end left out."""
    lines = text.splitlines()
    start = lines.index(" " * indent + f"{name}:")
    end = next((n for n in range(start + 1, len(lines))
                if lines[n].strip() and len(lines[n]) - len(lines[n].lstrip()) <= indent),
               len(lines))
    found = lines[start:end]
    while found and not found[-1].strip():
        found.pop()
    return found


def items(lines: list[str], key: str) -> list[str]:
    """The list items under ``key`` in a service's lines."""
    found = block("\n".join(lines), key, 4)[1:]
    return [line.strip()[2:] for line in found if line.strip().startswith("- ")]
