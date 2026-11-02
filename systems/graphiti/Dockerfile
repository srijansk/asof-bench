# Graphiti's ingestion on the run environment (ingest.py, adapter.py). Build it with systems/ as
# the context, so that it can take the shared package and the network check from beside it:
#
#     docker build -t asofbench/graphiti -f systems/graphiti/Dockerfile systems
#
# Only the build reaches the network: it installs the locked packages, hash-checked, and the
# container then runs on the environment's internal network, whose one route out is the proxy.
#
# python:3.12-slim (Python 3.12.16 on Debian trixie), its multi-platform index pinned by digest.
FROM python:3.12-slim@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159 AS deps

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GRAPHITI_TELEMETRY_ENABLED=false

# Every package exactly as requirements.txt pins it, hash-checked and from wheels only, from PyPI.
COPY graphiti/requirements.txt /app/requirements.txt
RUN pip install --require-hashes --only-binary :all: --index-url https://pypi.org/simple \
        -r /app/requirements.txt \
    && pip check

FROM deps

RUN groupadd --system --gid 10001 graphiti \
    && useradd --system --uid 10001 --gid 10001 --create-home --home-dir /home/graphiti \
       --shell /usr/sbin/nologin graphiti

# The adapter's own files, one by one: the shared package, the network check and the adapter.
COPY common/asofbench_systems/__init__.py common/asofbench_systems/llm.py \
     common/asofbench_systems/run.py common/asofbench_systems/stream.py \
     common/asofbench_systems/usage.py /app/asofbench_systems/
COPY env/netcheck.py /app/env/netcheck.py
COPY graphiti/adapter.py graphiti/ingest.py graphiti/config.json /app/graphiti/
RUN printf '#!/bin/sh\nexec python -m graphiti.ingest "$@"\n' > /usr/local/bin/ingest \
    && chmod 755 /usr/local/bin/ingest

ENV PYTHONPATH=/app
WORKDIR /app
USER 10001:10001

# The self-test runs with no network at all: it checks that the installed packages are the ones
# config.json records and that Graphiti builds as the adapter configures it, so that an image
# that builds is one that ingests as configured.
RUN --network=none python -m graphiti.adapter --self-test
# The self-test imports the adapter only. The `ingest` command imports more, the shared run.py
# among it. An image that lacked a file that ingest imports would build and fail only when its
# container first ingested, and the tests that run in the image read the files they are given,
# not the image's own. So the build imports ingest as the command does, with no network, and
# fails here instead.
RUN --network=none python -c "import graphiti.ingest"

CMD ["ingest", "--help"]
