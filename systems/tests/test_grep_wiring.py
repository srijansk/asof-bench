"""How grep at D is put together (systems/grep-at-d): its compose file (the egress service as the
reader's, the sandbox on sandbox_net alone with the tree read-only at /repo, a read-only root, a
16 MiB tmpfs at /tmp, no capability and no secret, and the agent, in the reader image, on inner
and sandbox_net); the sandbox's Dockerfile (debian:stable-slim pinned by digest, gawk and python3
from a fixed snapshot of the Debian archive, a user that is not root, the command server and its
self-test with no network); and its config.json against the code it records. Standard library
only, with packaging, which pytest installs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import grep_modules
from asofbench_systems import answer, llm
from image_files import block, items
from reader import answer_requests as reader

agent = grep_modules.agent()
server = grep_modules.command_server()

SYSTEMS = Path(__file__).resolve().parents[1]
GREP_AT_D = SYSTEMS / "grep-at-d"
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


def test_the_agent_runs_in_the_reader_image_on_inner_and_sandbox_net():
    reader_compose = (SYSTEMS / "reader" / "compose.yaml").read_text(encoding="utf-8")
    # The egress service as the reader's (and so Graphiti's), to the letter.
    assert block(COMPOSE, "egress", 2) == block(reader_compose, "egress", 2)
    service = block(COMPOSE, "agent", 2)
    assert "    image: asofbench/reader" in service
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


def test_the_sandbox_image_is_debian_stable_slim_by_digest_with_the_tools_and_no_root():
    assert re.fullmatch(r"debian:stable-slim@sha256:[0-9a-f]{64}", BASE_IMAGE)
    # Every package from the snapshot, the archive as it stood at that time, so that a rebuild
    # installs the same versions: no other source is left in the image.
    assert "apt-get install -y --no-install-recommends gawk python3" in SANDBOX_DOCKERFILE
    assert "deb.debian.org" not in SANDBOX_DOCKERFILE
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
    # The reader image, as the single-call entrants record it, and the sandbox's image.
    closed = json.loads((SYSTEMS / "closed-book" / "config.json").read_text(encoding="utf-8"))
    assert CONFIG["installed"]["reader"] == closed["installed"]
    lock = (SYSTEMS / "reader" / "requirements.txt").read_text(encoding="utf-8")
    assert CONFIG["installed"]["reader"]["python_packages"] == {
        name: version for name, version, marker in PIN.findall(lock) if not marker.strip()}
    assert CONFIG["installed"]["sandbox"]["base_image"] == BASE_IMAGE
    assert CONFIG["requests"]["fields"] == ["key", "question"]
    text = (GREP_AT_D / "config.json").read_text(encoding="utf-8")
    assert str(SYSTEMS.parent.resolve()) not in text and str(Path.home()) not in text
    try:
        from asofbench.scoring import REGISTERED_ANSWERING_SETTINGS
    except ImportError:
        return
    assert recorded["scorer_settings"] == REGISTERED_ANSWERING_SETTINGS
