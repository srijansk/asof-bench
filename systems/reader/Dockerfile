# The reader: the answering run of the single-call entrants, closed book first
# (answer_requests.py). Build it with systems/ as the context, so that it can take the shared
# package and the network check from beside it:
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

# The reader's own files, one by one: the shared package's files that the runner and its
# self-test import, the network check, the runner and the self-test. No entrant's configuration
# and no other file of the repository.
COPY common/asofbench_systems/__init__.py common/asofbench_systems/answer.py \
     common/asofbench_systems/llm.py common/asofbench_systems/run.py \
     common/asofbench_systems/usage.py /app/asofbench_systems/
COPY env/netcheck.py /app/env/netcheck.py
COPY reader/answer_requests.py reader/selftest.py /app/reader/
RUN printf '#!/bin/sh\nexec python -m reader.answer_requests "$@"\n' > /usr/local/bin/answer_requests \
    && chmod 755 /usr/local/bin/answer_requests

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

CMD ["answer_requests", "--help"]
