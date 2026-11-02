# The reader: the answering run of the single-call entrants, closed book first
# (answer_requests.py), and grep at D's agent, which answers through the same runner
# (grep-at-d/agent.py; its sandbox is an image of its own, grep-at-d/sandbox). Build it with
# systems/ as the context, so that it can take the shared package, the network check and the
# agent from beside it:
#
#     docker build -t asofbench/reader -f systems/reader/Dockerfile systems
#
# Only the build reaches the network: it installs the locked packages, hash-checked, and the
# container then runs on the environment's internal network, whose one route out is the proxy.
#
# python:3.12-slim (Python 3.12.16 on Debian trixie), its multi-platform index pinned by digest.
FROM python:3.12-slim@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159 AS deps

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Every package exactly as requirements.txt pins it, hash-checked and from wheels only, from PyPI.
COPY reader/requirements.txt /app/requirements.txt
RUN pip install --require-hashes --only-binary :all: --index-url https://pypi.org/simple \
        -r /app/requirements.txt \
    && pip check

FROM deps

RUN groupadd --system --gid 10001 reader \
    && useradd --system --uid 10001 --gid 10001 --create-home --home-dir /home/reader \
       --shell /usr/sbin/nologin reader

# The reader's own files, one by one: the shared package's files that the runner, its self-test
# and grep at D's agent import, the network check, the runner, the self-test and the agent. No
# entrant's configuration and no other file of the repository.
COPY common/asofbench_systems/__init__.py common/asofbench_systems/answer.py \
     common/asofbench_systems/llm.py common/asofbench_systems/run.py \
     common/asofbench_systems/usage.py /app/asofbench_systems/
COPY env/netcheck.py /app/env/netcheck.py
COPY reader/answer_requests.py reader/selftest.py /app/reader/
COPY grep-at-d/agent.py /app/grep_at_d/agent.py
RUN printf '#!/bin/sh\nexec python -m reader.answer_requests "$@"\n' > /usr/local/bin/answer_requests \
    && printf '#!/bin/sh\nexec python -m grep_at_d.agent "$@"\n' > /usr/local/bin/grep_at_d \
    && chmod 755 /usr/local/bin/answer_requests /usr/local/bin/grep_at_d

ENV PYTHONPATH=/app
WORKDIR /app
USER 10001:10001

# The self-test (selftest.py) runs with no network at all, through the command that every run
# uses: it imports every module the runner imports, the SDK and httpx among them, checks that the
# installed packages are the ones requirements.txt pins and that the network check is in place,
# and makes one answering call through the installed SDK to an httpx MockTransport, which must
# send the registered settings and nothing else, so that an image that builds is one that
# answers as registered.
RUN --network=none answer_requests --self-test
# Grep at D's self-test (grep-at-d/agent.py), through its own command, with no network either:
# one answer of two turns through the installed SDK to a MockTransport, with a stand-in for the
# sandbox, must send the registered settings and the tool on each turn, send the first reply back
# as it came and mark each request's last block alone for the prompt cache.
RUN --network=none grep_at_d --self-test

CMD ["answer_requests", "--help"]
