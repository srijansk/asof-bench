"""How grep at D is put together (systems/grep-at-d): its compose file (the egress service as the
reader's, the sandbox on sandbox_net alone with the tree read-only at /repo, a read-only root, a
16 MiB tmpfs at /tmp, no capability and no secret, and the agent, in grep at D's image, on inner
and sandbox_net); grep at D's image, the reader image with the agent's files added and no others,
every module that they import being the reader image's; the sandbox's Dockerfile
(debian:stable-slim pinned by digest, gawk and python3 from a fixed snapshot of the Debian
archive, a user that is not root, the command server and its self-test with no network); and its
config.json against the code it records, the sandbox's package versions against the packages that
its Dockerfile lists, and against closed book's where the two share the reader image. Standard
library only, with packaging, which pytest installs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import grep_modules
from asofbench_systems import answer, llm
from image_files import (
    block,
    copied_python_files,
    copied_shared_files,
    items,
    package_files_needed,
    shared_files_needed,
)
from reader import answer_requests as reader

agent = grep_modules.agent()
server = grep_modules.command_server()

SYSTEMS = Path(__file__).resolve().parents[1]
GREP_AT_D = SYSTEMS / "grep-at-d"
READER = SYSTEMS / "reader"
PACKAGE = SYSTEMS / "common" / "asofbench_systems"
DOCKERFILE = (GREP_AT_D / "Dockerfile").read_text(encoding="utf-8")
READER_DOCKERFILE = (READER / "Dockerfile").read_text(encoding="utf-8")
COMPOSE = (GREP_AT_D / "compose.yaml").read_text(encoding="utf-8")
SANDBOX_DOCKERFILE = (GREP_AT_D / "sandbox" / "Dockerfile").read_text(encoding="utf-8")
CONFIG = json.loads((GREP_AT_D / "config.json").read_text(encoding="utf-8"))
PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)[ \t]*(;[^\\\n]*)?", re.M)
# The sandbox's base image and the snapshot of the Debian archive that its packages come from, as
# its Dockerfile names them; and its scratch space and process limit, as the compose file sets them.
(BASE_IMAGE,) = re.findall(r"^FROM (\S+)$", SANDBOX_DOCKERFILE, re.M)
(SNAPSHOT,) = set(re.findall(r"http://snapshot\.debian\.org/archive/debian(?:-security)?/"
                             r"([0-9]{8}T[0-9]{6}Z)", SANDBOX_DOCKERFILE))
(TMPFS_BYTES,) = map(int, re.findall(r"^          size: ([0-9]+)$", COMPOSE, re.M))
(PIDS_LIMIT,) = map(int, re.findall(r"^    pids_limit: ([0-9]+)$", COMPOSE, re.M))


def test_the_sandbox_is_on_sandbox_net_alone_with_the_tree_read_only_and_nothing_else():
    service = block(COMPOSE, "sandbox", 2)
    assert "    image: asofbench/sandbox" in service
    assert items(service, "networks") == ["sandbox_net"]
    assert block("\n".join(service), "volumes", 4) == [
        "    volumes:",
        "      - ${ASOFBENCH_RUN_TREE:?}:/repo:ro",
        "      - type: tmpfs",
        "        target: /tmp",
        "        tmpfs:",
        "          size: 16777216"]
    assert TMPFS_BYTES == 16 * 1024 * 1024
    for line in ("    read_only: true", "    init: true", "    ipc: none", "    pids_limit: 256",
                 "    mem_limit: 512m", "    oom_score_adj: 1000"):
        assert line in service, line
    assert items(service, "cap_drop") == ["ALL"]
    assert items(service, "security_opt") == ["no-new-privileges:true"]
    # No secret, no environment, no route out, no restart that could change it under a run.
    for key in ("secrets:", "environment:", "ports:", "network_mode:", "restart:", "depends_on:"):
        assert not any(line.strip().startswith(key) for line in service), key


def test_the_agent_runs_in_grep_at_d_s_image_on_inner_and_sandbox_net():
    reader_compose = (SYSTEMS / "reader" / "compose.yaml").read_text(encoding="utf-8")
    # The egress service as the reader's (and so Graphiti's), to the letter.
    assert block(COMPOSE, "egress", 2) == block(reader_compose, "egress", 2)
    service = block(COMPOSE, "agent", 2)
    assert "    image: asofbench/grep-at-d" in service
    assert block("\n".join(service), "build", 4)[1:] == ["      context: ..",
                                                         "      dockerfile: grep-at-d/Dockerfile"]
    assert items(service, "networks") == ["inner", "sandbox_net"]
    assert items(service, "volumes") == ["${ASOFBENCH_RUN_INPUT:?}:/input:ro",
                                         "${ASOFBENCH_RUN_OUTPUT:?}:/output"]
    assert items(service, "secrets") == ["anthropic_api_key"]
    assert block("\n".join(service), "environment", 4)[1:] == [
        "      HTTPS_PROXY: http://egress:3128",
        "      ASOFBENCH_IMAGE_IDS: ${ASOFBENCH_IMAGE_IDS:-}"]
    assert block("\n".join(service), "depends_on", 4)[1:] == [
        "      egress:", "        condition: service_started",
        "      sandbox:", "        condition: service_started"]
    assert "    mem_limit: 1g" in service and "    oom_score_adj: 1000" in service
    assert block(COMPOSE, "networks", 0) == [
        "networks:",
        "  inner:", "    internal: true", "    driver_opts:",
        '      com.docker.network.bridge.inhibit_ipv4: "true"',
        "  sandbox_net:", "    internal: true", "    driver_opts:",
        '      com.docker.network.bridge.inhibit_ipv4: "true"',
        "  outer: {}"]
    assert block(COMPOSE, "secrets", 0) == [
        "secrets:", "  anthropic_api_key:", "    file: ${ASOFBENCH_ANTHROPIC_KEY_FILE:?}"]
    # The three services and nothing else, and no port published on the host.
    assert re.findall(r"^  (\w+):$", COMPOSE.split("\nnetworks:")[0], re.M) == [
        "egress", "sandbox", "agent"]
    assert "ports:" not in COMPOSE and "network_mode" not in COMPOSE
    assert "ANTHROPIC_API_KEY" not in COMPOSE and "ANTHROPIC_BASE_URL" not in COMPOSE
    # The proxy alone is on outer: the sandbox reaches neither it nor the API.
    assert [name for name in ("egress", "sandbox", "agent")
            if "outer" in items(block(COMPOSE, name, 2), "networks")] == ["egress"]


def test_grep_at_d_s_image_is_the_reader_s_with_the_agent_s_files_and_no_others():
    # Built FROM the reader image by its local tag, whose ID each run records with its own.
    assert re.findall(r"^FROM (\S+)$", DOCKERFILE, re.M) == ["asofbench/reader"]
    own = package_files_needed(GREP_AT_D / "agent.py", "grep_at_d", GREP_AT_D)
    # The agent imports its self-test (for --self-test), which imports the agent back.
    assert own == {"agent.py", "selftest.py"}
    assert copied_python_files(DOCKERFILE, "grep-at-d") == own
    assert "COPY grep-at-d/agent.py grep-at-d/selftest.py /app/grep_at_d/\n" in DOCKERFILE
    # Nothing that the reader image holds is copied again, and nothing of the sandbox's.
    assert copied_shared_files(DOCKERFILE) == set()
    assert copied_python_files(DOCKERFILE, "reader") == set()
    assert copied_python_files(DOCKERFILE, "env") == set()
    assert "sandbox" not in "".join(line for line in DOCKERFILE.splitlines()
                                    if line.startswith("COPY"))
    # Every module that the agent and its self-test import, and those import in turn, is one
    # that the reader image copies.
    entries = [GREP_AT_D / name for name in sorted(own)]
    reader_own = set().union(*(package_files_needed(entry, "reader", READER)
                               for entry in entries))
    assert reader_own == {"answer_requests.py", "selftest.py"}
    assert reader_own <= copied_python_files(READER_DOCKERFILE, "reader")
    shared = shared_files_needed([*entries, *(READER / name for name in sorted(reader_own))],
                                 PACKAGE)
    assert {"__init__.py", "answer.py", "llm.py", "run.py", "usage.py"} <= shared
    assert shared <= copied_shared_files(READER_DOCKERFILE)
    # The command that every run uses, the user it runs as, and the self-test with no network
    # after the user is set.
    assert 'exec python -m grep_at_d.agent "$@"' in DOCKERFILE
    user = DOCKERFILE.index("\nUSER 10001:10001\n")
    assert DOCKERFILE.index("\nUSER root\n") < DOCKERFILE.index("\nCOPY grep-at-d/") < user
    assert user < DOCKERFILE.index("\nRUN --network=none grep_at_d --self-test\n")
    assert DOCKERFILE.rstrip().endswith('CMD ["grep_at_d", "--help"]')


def test_the_sandbox_image_is_debian_stable_slim_by_digest_with_the_tools_and_no_root():
    assert re.fullmatch(r"debian:stable-slim@sha256:[0-9a-f]{64}", BASE_IMAGE)
    # Every package from the snapshot, the archive as it stood at that time, so that a rebuild
    # installs the same versions: no other source is left in the image.
    assert "apt-get install -y --no-install-recommends gawk python3" in SANDBOX_DOCKERFILE
    assert "deb.debian.org" not in SANDBOX_DOCKERFILE
    # Each source is signed by trixie's archive keyring, the .pgp file that the base image's own
    # sources name (the .gpg name beside it is a compatibility link to it).
    signed_by = re.findall(r"'Signed-By: (\S+)'", SANDBOX_DOCKERFILE)
    assert signed_by == ["/usr/share/keyrings/debian-archive-keyring.pgp"] * 2
    assert "rm -f /etc/apt/sources.list /etc/apt/sources.list.d/*" in SANDBOX_DOCKERFILE
    user = SANDBOX_DOCKERFILE.index("\nUSER 10001:10001\n")
    assert SANDBOX_DOCKERFILE.index("apt-get install") < user < SANDBOX_DOCKERFILE.index(
        "\nRUN --network=none python3 /opt/sandbox/command_server.py --self-test\n")
    assert "COPY command_server.py /opt/sandbox/command_server.py\n" in SANDBOX_DOCKERFILE
    assert ('CMD ["python3", "/opt/sandbox/command_server.py", "--root", "/repo", "--scratch", '
            f'"/tmp", "--port", "{server.PORT}"]') in SANDBOX_DOCKERFILE
    # The self-test checks the tools that the brief gives the sandbox: bash, coreutils, grep,
    # findutils, sed and gawk.
    assert {"bash", "cat", "head", "ls", "sort", "wc", "grep", "find", "sed", "gawk"} <= set(
        server.TOOLS)


def test_grep_at_d_s_config_records_what_the_code_sends_and_runs():
    assert CONFIG["system"] == "grep-at-d"
    prompts = CONFIG["prompts"]
    assert prompts["system"] == agent.SYSTEM_PROMPT
    assert prompts["user"] == answer.user_message("{question}", None)
    assert prompts["tool"] == agent.TOOL and prompts["tool_choice"] == agent.TOOL_CHOICE
    assert prompts["cap_turn"] == agent.CAP_TURN
    assert prompts["tool_result"]["stderr_label"] == agent.STDERR_LABEL
    assert prompts["tool_result"]["exit_line"] == agent.EXIT_LINE.format("{exit_code}")
    assert prompts["tool_result"]["timed_out_line"] == agent.TIMED_OUT_LINE
    assert prompts["tool_result"]["not_run"] == agent.NOT_RUN
    settings = agent.settings(answer.MODELS[0])
    recorded = CONFIG["answering"]
    assert (recorded["model"], recorded["max_tokens"], recorded["thinking"],
            recorded["output_config"]) == (settings["model"], settings["max_tokens"],
                                           settings["thinking"], settings["output_config"])
    assert recorded["sensitivity_check_model"].startswith(answer.MODELS[1])
    assert recorded["base_url"] == llm.BASE_URL
    assert recorded["prompt_caching"]["cache_control"] == agent.CACHE
    limits = CONFIG["limits"]
    assert (limits["max_tool_calls"], limits["command_timeout_s"], limits["output_characters"],
            limits["command_bytes"]) == (agent.MAX_TOOL_CALLS, server.TIMEOUT_S, server.LIMIT,
                                         server.MAX_COMMAND_BYTES)
    assert set(CONFIG["reply"]["not_answered"]) == set(agent.ERRORS)
    sandbox = CONFIG["sandbox"]
    assert sandbox["base_image"] == BASE_IMAGE and sandbox["debian_snapshot"] == SNAPSHOT
    assert (sandbox["tmpfs_bytes"], sandbox["pids_limit"], sandbox["port"]) == (
        TMPFS_BYTES, PIDS_LIMIT, server.PORT)
    assert sandbox["client_timeout_s"] == agent.SANDBOX_TIMEOUT_S
    # What the commands may use, and how their output is read, in the command server's words.
    assert all(tool in sandbox["tools"] for tool in ("python3", "gawk", "bash", "grep", "sed"))
    read_as = "read as UTF-8 with each byte that is not UTF-8 replaced by U+FFFD"
    assert read_as in sandbox["commands"] and read_as in " ".join(server.__doc__.split())
    assert "turns thinking off" not in CONFIG["answering"]["tool_choice"]
    # The stop after api_error responses in a row spans the dates.
    assert "--api-errors-before" in CONFIG["run"]["api_errors_across_dates"]
    assert "api_errors_before" in CONFIG["run"]["api_errors_across_dates"]
    run = CONFIG["run"]
    assert (run["concurrency"], run["max_retries"], run["timeout_s"],
            run["max_consecutive_api_errors"]) == (reader.CONCURRENCY, reader.MAX_RETRIES,
                                                   reader.TIMEOUT_S,
                                                   reader.MAX_CONSECUTIVE_API_ERRORS)
    policy = CONFIG["network_policy"]
    (route,) = policy["routes_out"]
    assert (route["host"], route["port"]) == ("api.anthropic.com", 443)
    (inside,) = policy["services_inside"]
    assert (inside["name"], inside["host"], inside["port"]) == ("sandbox", *agent.SANDBOX)
    assert inside["routes_out"] == "none"
    assert policy["provider_side_tools"].startswith("none")
    (idle,) = re.findall(r'^\s+EGRESS_IDLE_TIMEOUT:\s*"([0-9]+)"$', COMPOSE, re.M)
    assert policy["proxy"]["idle_timeout_s"] == int(idle) > reader.TIMEOUT_S
    # What is installed: grep at D's own images, and, as closed book records them, what the two
    # genuinely share, the reader image's Python, base image and packages and the proxy's image.
    installed = CONFIG["installed"]
    closed = json.loads((SYSTEMS / "closed-book" / "config.json").read_text(encoding="utf-8"))
    for shared_entry in ("python", "base_image", "python_packages"):
        assert installed[shared_entry] == closed["installed"][shared_entry], shared_entry
    (reader_base,) = re.findall(r"^FROM (\S+) AS deps$", READER_DOCKERFILE, re.M)
    assert installed["base_image"] == reader_base
    lock = (READER / "requirements.txt").read_text(encoding="utf-8")
    assert installed["python_packages"] == {
        name: version for name, version, marker in PIN.findall(lock) if not marker.strip()}
    assert set(installed["images"]) == {"asofbench/grep-at-d", "asofbench/reader",
                                        "asofbench/egress", "asofbench/sandbox"}
    assert installed["images"]["asofbench/egress"] == closed["installed"]["images"][
        "asofbench/egress"]
    assert "FROM asofbench/reader" in installed["images"]["asofbench/grep-at-d"]
    assert "grep_at_d --self-test" in installed["images"]["asofbench/grep-at-d"]
    assert installed["sandbox"]["base_image"] == BASE_IMAGE
    assert installed["sandbox"]["debian_snapshot"] == SNAPSHOT
    # The version of each package that the sandbox's build lists with dpkg-query, and of no
    # other, as the build installed it from the snapshot that fixes it.
    (listed,) = re.findall(r"dpkg-query -W -f '\$\{Package\}=\$\{Version\}\\n' ([a-z0-9 ]+)$",
                           SANDBOX_DOCKERFILE, re.M)
    versions = installed["sandbox"]["package_versions"]
    assert sorted(versions) == sorted(listed.split())
    assert all(re.fullmatch(r"([0-9]+:)?[0-9][A-Za-z0-9.+~-]*", version)
               for version in versions.values())
    assert SNAPSHOT in sandbox["packages"] and "package_versions" in sandbox["packages"]
    # Where each date's tree came from is recorded on the host, in a file that no container is
    # given.
    assert "trees.json" in CONFIG["run"]["trees"] and "given to no container" in (
        CONFIG["run"]["trees"])
    assert "trees.json" in CONFIG["network_policy"]["unreachable"]
    assert all(name in installed["per_run"] for name in ("grep at D's", "the reader's"))
    assert CONFIG["requests"]["fields"] == ["key", "question"]
    text = (GREP_AT_D / "config.json").read_text(encoding="utf-8")
    assert str(SYSTEMS.parent.resolve()) not in text and str(Path.home()) not in text
    try:
        from asofbench.scoring import REGISTERED_ANSWERING_SETTINGS
    except ImportError:
        return
    assert recorded["scorer_settings"] == REGISTERED_ANSWERING_SETTINGS
