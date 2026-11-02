"""How the reader is put together: its Dockerfile's list of the shared package's files against the
ones that its runner imports, and the rest of the image as the first plan builds Graphiti's; its
compose file (the egress service as Graphiti's, the reader on the internal network only, its two
mounts and the key as a Docker secret); its network check (the shared flags, no services, its
mounts, after the proxy accepts connections); its lock; closed book's config.json against the
code it records; and the CI step that runs the answering tests with the SDK that the lock pins.
Standard library only."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from asofbench_systems import answer, llm
from asofbench_systems import run as plumbing
from image_files import copied_shared_files, shared_file_differences, shared_files_needed
from reader import answer_requests as reader

SYSTEMS = Path(__file__).resolve().parents[1]
READER = SYSTEMS / "reader"
PACKAGE = SYSTEMS / "common" / "asofbench_systems"
CONFIG = SYSTEMS / "closed-book" / "config.json"
PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)[ \t]*(;[^\\\n]*)?", re.M)


def lock_pins(marked: bool = False) -> dict[str, str]:
    """The lock's pins, those with an environment marker left out unless ``marked``."""
    text = (READER / "requirements.txt").read_text(encoding="utf-8")
    return {name: version for name, version, marker in PIN.findall(text)
            if marked or not marker.strip()}


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


def test_the_image_copies_the_shared_files_that_the_runner_imports_and_no_others():
    needed = shared_files_needed([READER / "answer_requests.py"], PACKAGE)
    # The files that the runner cannot do without, so that an empty result is no pass.
    assert {"__init__.py", "answer.py", "llm.py", "run.py", "usage.py"} <= needed
    assert "stream.py" not in needed
    dockerfile = (READER / "Dockerfile").read_text(encoding="utf-8")
    assert shared_file_differences(dockerfile, needed) == (set(), set())
    assert copied_shared_files(dockerfile) == needed


def test_the_image_is_built_as_graphiti_s_is():
    dockerfile = (READER / "Dockerfile").read_text(encoding="utf-8")
    graphiti = (SYSTEMS / "graphiti" / "Dockerfile").read_text(encoding="utf-8")
    base = re.compile(r"^FROM (python:3\.12-slim@sha256:[0-9a-f]{64}) AS deps$", re.M)
    assert base.findall(dockerfile) == base.findall(graphiti) != []
    assert ("RUN pip install --require-hashes --only-binary :all: --index-url "
            "https://pypi.org/simple \\\n        -r /app/requirements.txt \\\n    && pip check"
            in dockerfile)
    assert "COPY reader/requirements.txt /app/requirements.txt\n" in dockerfile
    assert "COPY env/netcheck.py /app/env/netcheck.py\n" in dockerfile
    assert "COPY reader/answer_requests.py /app/reader/\n" in dockerfile
    # The runner's own user, and a self-test with no network through the command every run uses.
    assert "useradd --system --uid 10001 --gid 10001" in dockerfile
    assert "\nUSER 10001:10001\n" in dockerfile
    assert dockerfile.index("\nUSER 10001:10001\n") < dockerfile.index(
        "\nRUN --network=none answer_requests --self-test\n")
    assert 'exec python -m reader.answer_requests "$@"' in dockerfile
    # The paths the runner finds its lock and the network check at, in the image's layout.
    assert reader.REQUIREMENTS == READER.parent / "requirements.txt"
    assert reader.NETCHECK_SCRIPT == SYSTEMS / "env" / "netcheck.py"
    assert reader.NETCHECK_SCRIPT.is_file()


def test_the_compose_file_runs_the_reader_on_the_internal_network_only():
    compose = (READER / "compose.yaml").read_text(encoding="utf-8")
    graphiti = (SYSTEMS / "graphiti" / "compose.yaml").read_text(encoding="utf-8")
    # The egress service as Graphiti's, to the letter: its allowlist, its idle timeout, its
    # networks, its memory limit, its kill order and its restarts.
    assert block(compose, "egress", 2) == block(graphiti, "egress", 2)
    service = block(compose, "reader", 2)
    assert "    image: asofbench/reader" in service
    assert items(service, "networks") == ["inner"]
    assert items(service, "volumes") == ["${ASOFBENCH_RUN_INPUT:?}:/input:ro",
                                         "${ASOFBENCH_RUN_OUTPUT:?}:/output"]
    assert items(service, "secrets") == ["anthropic_api_key"]
    assert block("\n".join(service), "environment", 4)[1:] == [
        "      HTTPS_PROXY: http://egress:3128",
        "      ASOFBENCH_IMAGE_IDS: ${ASOFBENCH_IMAGE_IDS:-}"]
    assert "    mem_limit: 1g" in service and "    oom_score_adj: 1000" in service
    assert block(compose, "networks", 0) == [
        "networks:", "  inner:", "    internal: true", "    driver_opts:",
        '      com.docker.network.bridge.inhibit_ipv4: "true"', "  outer: {}"]
    assert block(compose, "secrets", 0) == [
        "secrets:", "  anthropic_api_key:", "    file: ${ASOFBENCH_ANTHROPIC_KEY_FILE:?}"]
    # The two services and nothing else, and no port published on the host.
    assert re.findall(r"^  (\w+):$", compose.split("\nnetworks:")[0], re.M) == ["egress",
                                                                                 "reader"]
    assert "ports:" not in compose and "network_mode" not in compose
    assert "ANTHROPIC_API_KEY" not in compose and "ANTHROPIC_BASE_URL" not in compose


def test_the_reader_s_network_check_gets_the_shared_flags_no_services_and_its_mounts(tmp_path,
                                                                                    monkeypatch):
    waits, commands = [], []
    monkeypatch.setattr(plumbing, "wait_for_services", lambda *arguments: waits.append(arguments))

    def fake_run(command, check):
        commands.append(command)
        return subprocess.CompletedProcess(command, 5)

    monkeypatch.setattr(plumbing.subprocess, "run", fake_run)
    assert reader.netcheck_preflight(tmp_path) == 5
    # The proxy first, which starts with the reader; then no service, since the reader runs none.
    assert waits == [((("egress", 3128),), 60.0, 1.0), ((), 0.0, 1.0)]
    assert commands == [[sys.executable, str(SYSTEMS / "env" / "netcheck.py"),
                         *plumbing.NETCHECK_ARGS, "--services", "",
                         "--mounts", "/input,/output,/run/secrets",
                         "--report", str(tmp_path / "netcheck.json")]]


def test_the_reader_s_images_are_its_own_and_the_proxy_s():
    images = {"reader": "sha256:" + "1" * 64, "egress": "sha256:" + "2" * 64}
    assert reader.parse_image_ids(",".join(f"{k}={v}" for k, v in images.items())) == images
    with pytest.raises(plumbing.Refused) as caught:
        reader.parse_image_ids(f"reader={images['reader']}")
    assert str(caught.value).endswith("name=ID entries for reader, egress (systems/README.md)")


def test_the_lock_pins_the_registered_sdk_with_hashes_for_every_package():
    pins = lock_pins(marked=True)
    assert pins["anthropic"] == "0.78.0" and pins["httpx"] == "0.28.1"
    assert re.findall(r"^anthropic==(\S+)$", (READER / "requirements.in").read_text(
        encoding="utf-8"), re.M) == ["0.78.0"]
    text = (READER / "requirements.txt").read_text(encoding="utf-8")
    entries = re.split(r"\n(?=[A-Za-z0-9])", text.split("\n", 2)[2])
    assert len(entries) == len(pins)
    for entry in entries:
        assert re.search(r"--hash=sha256:[0-9a-f]{64}", entry), entry.splitlines()[0]
    assert ("uv pip compile requirements.in --universal --python-version 3.12 --generate-hashes "
            "--exclude-newer 2026-11-01T00:00:00Z -o requirements.txt") in text.splitlines()[1]


def test_closed_book_s_config_records_the_settings_that_the_code_sends():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    settings = answer.answering_settings(answer.MODELS[0])
    assert (config["system"], config["inputs"]) == ("closed-book",
                                                    "the question text and nothing else")
    assert config["prompts"]["system"] == answer.SYSTEM_PROMPT == settings["system"]
    assert config["prompts"]["user"] == answer.user_message("{question}", None) == (
        "Question: {question}")
    recorded = config["answering"]
    assert (recorded["model"], recorded["max_tokens"], recorded["thinking"],
            recorded["output_config"]) == (settings["model"], settings["max_tokens"],
                                           settings["thinking"], settings["output_config"])
    assert recorded["output_config"]["format"]["schema"] == answer.ANSWER_SCHEMA
    assert recorded["sensitivity_check_model"].startswith(answer.MODELS[1])
    assert recorded["base_url"] == llm.BASE_URL
    assert recorded["scorer_settings"] == "effort medium, adaptive thinking, default temperature"
    assert config["requests"]["fields"] == ["key", "question"]
    assert set(config["reply"]["not_answered"]) == set(answer.ERRORS)
    assert (config["run"]["concurrency"], config["run"]["max_retries"],
            config["run"]["timeout_s"]) == (reader.CONCURRENCY, reader.MAX_RETRIES,
                                            reader.TIMEOUT_S)
    assert config["network_policy"]["services_inside"] == []
    (route,) = config["network_policy"]["routes_out"]
    assert (route["host"], route["port"]) == ("api.anthropic.com", 443)
    assert config["network_policy"]["provider_side_tools"].startswith("none")
    compose = (READER / "compose.yaml").read_text(encoding="utf-8")
    (idle,) = re.findall(r'^\s+EGRESS_IDLE_TIMEOUT:\s*"([0-9]+)"$', compose, re.M)
    assert config["network_policy"]["proxy"]["idle_timeout_s"] == int(idle) > reader.TIMEOUT_S
    installed = config["installed"]
    assert installed["python_packages"] == lock_pins()
    (base,) = re.findall(r"^FROM (\S+) AS deps$", (READER / "Dockerfile").read_text(
        encoding="utf-8"), re.M)
    assert installed["base_image"] == base
    # No local path, and the registered scorer settings where the scorer is installed.
    text = CONFIG.read_text(encoding="utf-8")
    assert str(SYSTEMS.parent.resolve()) not in text and str(Path.home()) not in text
    try:
        from asofbench.scoring import REGISTERED_ANSWERING_SETTINGS
    except ImportError:
        return
    assert recorded["scorer_settings"] == REGISTERED_ANSWERING_SETTINGS


def test_ci_runs_the_answering_tests_with_the_sdk_and_httpx_that_the_lock_pins():
    workflow = SYSTEMS.parent / ".github" / "workflows" / "tests.yml"
    if not workflow.is_file():
        pytest.skip("the repository's workflows are not beside systems/ here")
    pins = lock_pins()
    step = (f"--extra dev --with anthropic=={pins['anthropic']} --with httpx=={pins['httpx']} "
            "python -m pytest systems/tests/test_answer.py systems/tests/test_llm_client.py")
    assert step in workflow.read_text(encoding="utf-8")
