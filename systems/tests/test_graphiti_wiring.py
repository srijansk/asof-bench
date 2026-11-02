"""How Graphiti uses what the systems share: the values it gives the shared run plumbing
(common/asofbench_systems/run.py, tested in test_run.py), which are its infrastructure errors, its
image names and README, and its network check's services, mounts and waits; the names it takes
from that module and defines none of, the network check's shared flags among them; the
Dockerfile's list of the shared package's files against the closure of what adapter.py and
ingest.py import (the files they import, those files' imports in turn, and the package's
``__init__.py``), and the build's import of ingest; and the memory settings that compose.yaml
gives the embedder and Neo4j against the ones that config.json records. Standard library only."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest
from asofbench_systems import run as plumbing
from graphiti import ingest
from image_files import (
    block,
    copied_shared_files,
    shared_file_differences,
    shared_files_needed,
)

GRAPHITI = Path(__file__).resolve().parents[1] / "graphiti"
PACKAGE = GRAPHITI.parent / "common" / "asofbench_systems"
IMAGES = {name: "sha256:" + str(n) * 64 for n, name in enumerate(ingest.IMAGE_NAMES, 1)}
# The names that ingest.py takes from the shared module, and so must not define.
SHARED_NAMES = ("Refused", "Stop", "hold_lock", "cancelled_on_sigterm", "recorded", "write_json",
                "read_jsonl", "drop_unfinished_line", "utc_now", "new_invocation", "append_line",
                "error_text", "sha256_file", "crashed", "NETCHECK", "NETCHECK_ARGS", "RUN",
                "EXIT_DONE", "EXIT_FAILED", "EXIT_STOPPED", "EXIT_FAILURES", "EXIT_CRASHED")


def failure(module: str, name: str) -> type[Exception]:
    """An exception class that stands in for one of a package's, by its module and name."""
    return type(name, (Exception,), {"__module__": module})


def module_level_bindings(tree: ast.Module) -> dict[str, ast.stmt]:
    """The names that a module's own statements bind by assignment or by definition, each with
    the statement that binds it (imports are not bindings in this sense)."""
    bound = {}
    for statement in tree.body:
        if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            bound[statement.name] = statement
        elif isinstance(statement, ast.Assign | ast.AnnAssign | ast.AugAssign):
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            for target in targets:
                for node in ast.walk(target):
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                        bound[node.id] = statement
    return bound


def test_ingest_takes_the_shared_names_from_the_shared_module_and_defines_none_of_them():
    tree = ast.parse((GRAPHITI / "ingest.py").read_text(encoding="utf-8"))
    imported = {alias.name for statement in tree.body
                if isinstance(statement, ast.ImportFrom)
                and statement.module == "asofbench_systems.run" for alias in statement.names}
    bound = module_level_bindings(tree)
    # Each is imported from the shared module, and none is bound again in ingest.py: a copy of
    # an exit status or a file name would be an assignment, which `is` cannot tell from the
    # shared name itself, since equal small numbers and strings are the same object.
    assert set(SHARED_NAMES) <= imported
    assert not set(SHARED_NAMES) & set(bound)
    for name in SHARED_NAMES:
        assert getattr(ingest, name) is getattr(plumbing, name), name
    # The one alias is Graphiti's name for the exit status 0, and no other exit status is its own.
    aliases = {name: statement.value.id for name, statement in bound.items()
               if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Name)
               and statement.value.id in imported}
    assert aliases == {"EXIT_INGESTED": "EXIT_DONE"}
    assert {name for name in bound if name.startswith("EXIT_")} == {"EXIT_INGESTED"}
    assert ingest.EXIT_INGESTED == plumbing.EXIT_DONE == 0


def test_graphiti_stops_on_the_infrastructure_errors_that_it_stopped_on_before_they_moved():
    # The table as ingest.py held it before its Anthropic and connection entries moved to the
    # shared module.
    before = {
        "neo4j": {"ServiceUnavailable", "SessionExpired", "SessionError", "DatabaseUnavailable"},
        "openai": {"APIConnectionError", "InternalServerError"},
        "anthropic": {"APIConnectionError", "RateLimitError", "OverloadedError",
                      "InternalServerError", "AuthenticationError", "PermissionDeniedError"},
        "graphiti_core": {"RateLimitError"},
        "builtins": {"ConnectionError"},
    }
    assert ingest.INFRASTRUCTURE == before
    # It is the shared table and Graphiti's own entries, for the services that it runs.
    assert {package: ingest.INFRASTRUCTURE[package] for package in plumbing.INFRASTRUCTURE} == (
        plumbing.INFRASTRUCTURE)
    assert set(ingest.INFRASTRUCTURE) - set(plumbing.INFRASTRUCTURE) == {"neo4j", "openai",
                                                                         "graphiti_core"}
    # Its classes are frozensets, so that nothing done to this table changes the shared one's.
    assert all(isinstance(classes, frozenset) for classes in ingest.INFRASTRUCTURE.values())
    with pytest.raises(AttributeError):
        ingest.INFRASTRUCTURE["anthropic"].add("Boom")
    for package, classes in before.items():
        for name in classes:
            error = failure(f"{package}._exceptions", name)()
            assert ingest.infrastructure_error(error) == f"{package} {name}"
    assert ingest.infrastructure_error(ValueError("bad output")) is None
    assert ingest.infrastructure_error(failure("anthropic._exceptions", "BadRequestError")(
        "prompt is too long")) is None


def test_graphiti_s_key_check_refuses_with_the_text_it_gave_before_the_check_moved(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(ingest.Refused) as caught:
        ingest.check_key_file(str(missing))
    assert str(caught.value) == (
        f"the API key file {missing} cannot be read (No such file or directory): it reaches the "
        "container as the Docker secret, which the container's user, uid 10001, must be able to "
        "read (systems/graphiti/README.md)")


def test_the_image_ids_are_graphiti_s_four_and_a_refusal_points_at_its_readme():
    assert list(IMAGES) == ["graphiti", "egress", "embedder", "neo4j"]
    text = ",".join(f"{name}={value}" for name, value in IMAGES.items())
    assert ingest.parse_image_ids(text) == IMAGES
    for bad in (None, text.replace("egress=", "proxy=")):
        with pytest.raises(ingest.Refused) as caught:
            ingest.parse_image_ids(bad)
        assert str(caught.value) == (
            "ASOFBENCH_IMAGE_IDS must give the ID (sha256:...) of each image the run uses, as "
            "name=ID entries for graphiti, egress, embedder, neo4j (systems/graphiti/README.md)")


def test_graphiti_s_network_check_gets_its_services_mounts_and_waits(tmp_path, monkeypatch):
    waits, commands = [], []
    monkeypatch.setattr(plumbing, "wait_for_services", lambda *arguments: waits.append(arguments))

    def fake_run(command, check):
        commands.append(command)
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr(plumbing.subprocess, "run", fake_run)
    assert ingest.netcheck_preflight(tmp_path) == 7
    # The services are waited for in the order that SERVICES gives, within 600 s, every 2 s.
    assert waits == [((("neo4j", 7687), ("embedder", 8080)), 600.0, 2.0)]
    # The command as it was when the services and the mounts were part of NETCHECK_ARGS, so that
    # netcheck.json lists its checks as it always did.
    script = GRAPHITI.parent / "env" / "netcheck.py"
    assert script.is_file()
    assert commands == [[
        sys.executable, str(script), "--proxy", "http://egress:3128",
        "--allow", "api.anthropic.com:443",
        "--deny", "github.com:443,huggingface.co:443,pypi.org:443,peps.python.org:443,"
                  "eips.ethereum.org:443,ercs.ethereum.org:443",
        "--deny", "host.docker.internal:443,gateway.docker.internal:443",
        "--direct", "140.82.112.3:443,1.1.1.1:443",
        "--services", "embedder:8080,neo4j:7687",
        "--mounts", "/input,/output,/run/secrets", "--report", str(tmp_path / "netcheck.json")]]


# The Dockerfile copies the shared package's files one by one. The build's self-test imports the
# adapter only, which imports neither ingest.py nor the shared run.py, so the build also imports
# ingest (the last test below pins that line): without it, a file that ingest.py needs and the
# Dockerfile lacks would fail only when the container first ran ingest, since the tests that run
# inside the image read the files they are given, not the image's own. The guard here finds the
# same omission on the host, before a build. image_files.py reads the Dockerfile and the imports
# as the build and the interpreter do.


def test_the_image_copies_the_shared_files_that_graphiti_imports_and_no_others():
    needed = shared_files_needed([GRAPHITI / "adapter.py", GRAPHITI / "ingest.py"], PACKAGE)
    # The files that Graphiti's modules cannot do without, so that an empty result is no pass.
    assert {"__init__.py", "llm.py", "run.py", "stream.py", "usage.py"} <= needed
    dockerfile = (GRAPHITI / "Dockerfile").read_text(encoding="utf-8")
    assert shared_file_differences(dockerfile, needed) == (set(), set())


def test_a_file_named_only_in_a_comment_or_in_another_instruction_is_not_copied():
    needed = shared_files_needed([GRAPHITI / "adapter.py", GRAPHITI / "ingest.py"], PACKAGE)
    # Built from the closure as it is, not from a list of it: every needed file but run.py is
    # copied, by one continued COPY with a comment line inside it, and run.py is named only in a
    # comment and in a RUN instruction, which copy nothing. So when Graphiti first imports
    # another shared file, this test still says what the guard does with a comment.
    sources = [f"common/asofbench_systems/{name}" for name in sorted(needed - {"run.py"})]
    assert len(sources) >= 2 and "other.py" not in needed
    dockerfile = (
        "FROM python:3.12-slim\n"
        "# COPY common/asofbench_systems/run.py /app/asofbench_systems/\n"
        f"COPY {sources[0]} \\\n"
        "     # common/asofbench_systems/run.py\n"
        f"     {' '.join(sources[1:])} \\\n"
        "     /app/asofbench_systems/\n"
        "RUN ls common/asofbench_systems/run.py /tmp/\n")
    # The continued COPY is read whole, with its comment line dropped, and copies all the others.
    assert copied_shared_files(dockerfile) == needed - {"run.py"}
    assert shared_file_differences(dockerfile, needed) == ({"run.py"}, set())
    # Naming it in a COPY copies it, and a COPY of a file that is not needed is an extra.
    assert shared_file_differences(
        dockerfile + "COPY common/asofbench_systems/run.py /app/asofbench_systems/\n"
        "copy common/asofbench_systems/other.py /app/asofbench_systems/\n",
        needed) == (set(), {"other.py"})


def test_a_relative_import_between_shared_modules_is_followed(tmp_path):
    package = tmp_path / "asofbench_systems"
    package.mkdir()
    sources = {"__init__": "", "a": "from .b import thing\nfrom . import c, d\n",
               "b": "import json\n", "c": "", "d": "", "e": "", "f": "", "unused": ""}
    for name, source in sources.items():
        (package / f"{name}.py").write_text(source, encoding="utf-8")
    entry = tmp_path / "entry.py"
    # The absolute imports in each form; the relative one names the entry's own package.
    entry.write_text("from asofbench_systems import a\nfrom asofbench_systems.e import thing\n"
                     "import asofbench_systems.f\nfrom . import not_shared\n", encoding="utf-8")
    assert shared_files_needed([entry], package) == {
        "__init__.py", "a.py", "b.py", "c.py", "d.py", "e.py", "f.py"}


def test_the_package_s_init_is_read_for_imports_like_any_other_shared_file(tmp_path):
    package = tmp_path / "asofbench_systems"
    package.mkdir()
    # Importing any module of the package runs __init__.py first, so what it imports is needed
    # by every entry, in whatever form it imports it, and the files those import in turn.
    init = "from .g import thing\nfrom . import h\nfrom asofbench_systems.i import x\n"
    sources = {"__init__": init, "a": "", "g": "from . import j\n", "h": "", "i": "", "j": "",
               "unused": ""}
    for name, source in sources.items():
        (package / f"{name}.py").write_text(source, encoding="utf-8")
    entry = tmp_path / "entry.py"
    entry.write_text("from asofbench_systems import a\n", encoding="utf-8")
    assert shared_files_needed([entry], package) == {
        "__init__.py", "a.py", "g.py", "h.py", "i.py", "j.py"}
    assert shared_files_needed([], package) == {"__init__.py", "g.py", "h.py", "i.py", "j.py"}


def test_the_build_imports_ingest_after_the_self_test_with_the_path_and_the_user_set():
    dockerfile = (GRAPHITI / "Dockerfile").read_text(encoding="utf-8")
    self_test = "\nRUN --network=none python -m graphiti.adapter --self-test\n"
    imports = '\nRUN --network=none python -c "import graphiti.ingest"\n'
    assert dockerfile.count(self_test) == dockerfile.count(imports) == 1
    # Once the files are copied and the image's path and user are set, as the container's
    # `ingest` command runs: an image that lacks a file that ingest imports fails here, at the
    # build, and not when the container first ingests.
    for earlier in ("\nCOPY graphiti/adapter.py graphiti/ingest.py graphiti/config.json "
                    "/app/graphiti/\n", "\nENV PYTHONPATH=/app\n", "\nUSER 10001:10001\n",
                    self_test):
        assert dockerfile.index(earlier) < dockerfile.index(imports), earlier
    assert 'exec python -m graphiti.ingest "$@"' in dockerfile


def settings(lines: list[str]) -> dict[str, str]:
    """The ``key: value`` lines of a compose service, as a dict, a quoted value without its
    quotes."""
    pairs = (line.strip().split(": ", 1) for line in lines if ": " in line)
    return {key: value.strip('"') for key, value in pairs}


def test_compose_gives_the_embedder_its_allocator_settings_and_neo4j_1_5_gib_as_config_records():
    compose = (GRAPHITI / "compose.yaml").read_text(encoding="utf-8")
    config = json.loads((GRAPHITI / "config.json").read_text(encoding="utf-8"))
    embedder, neo4j = block(compose, "embedder", 2), block(compose, "neo4j", 2)
    # The embedding server's glibc allocator, for its memory only: two arenas and a fixed 128 KiB
    # mmap threshold, which M0 needed to keep the server near 0.5 GiB.
    allocator = settings(block("\n".join(embedder), "environment", 4)[1:])
    assert allocator == {"MALLOC_ARENA_MAX": "2", "MALLOC_MMAP_THRESHOLD_": "131072",
                         "MALLOC_TRIM_THRESHOLD_": "131072"}
    assert config["embedder"]["memory"] == {"mem_limit": settings(embedder)["mem_limit"],
                                            "environment": allocator}
    # Neo4j's container limit is 1.5 GiB, with its heap and page cache as they were.
    environment = settings(block("\n".join(neo4j), "environment", 4)[1:])
    assert settings(neo4j)["mem_limit"] == "1536m"
    assert config["graph_database"]["memory"] == {
        "heap_initial": environment["NEO4J_server_memory_heap_initial__size"],
        "heap_max": environment["NEO4J_server_memory_heap_max__size"],
        "pagecache": environment["NEO4J_server_memory_pagecache_size"],
        "mem_limit": "1536m"}
    assert (environment["NEO4J_server_memory_heap_initial__size"],
            environment["NEO4J_server_memory_heap_max__size"],
            environment["NEO4J_server_memory_pagecache_size"]) == ("256m", "512m", "128m")
