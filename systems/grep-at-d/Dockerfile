# Grep at D's agent (agent.py), in an image of its own: the reader image, with the agent, its
# self-test and its command added and nothing else, so that the reader image, which the
# single-call entrants answer in, holds nothing of grep at D. Build the reader first, then this,
# both with systems/ as the context:
#
#     docker build -t asofbench/reader -f systems/reader/Dockerfile systems
#     docker build -t asofbench/grep-at-d -f systems/grep-at-d/Dockerfile systems
#
# It is built FROM the reader image by its local tag: the reader's packages, its runner, the
# shared package and the network check are the reader image's, and this build adds no package.
# Each run records the IDs of both images, and run_dates.py checks that this one's layers begin
# with the reader's, that is, that it was built from the reader image whose ID the run records.
FROM asofbench/reader

USER root
# Grep at D's own files: the agent, which the reader's runner answers through, and its
# self-test. The modules they import from the shared package and the reader are the reader
# image's.
COPY grep-at-d/agent.py grep-at-d/selftest.py /app/grep_at_d/
RUN printf '#!/bin/sh\nexec python -m grep_at_d.agent "$@"\n' > /usr/local/bin/grep_at_d \
    && chmod 755 /usr/local/bin/grep_at_d
USER 10001:10001

# The self-test (selftest.py) runs with no network at all, through the command that every run
# uses: one answer of two turns through the installed SDK to an httpx MockTransport, with a
# stand-in for the sandbox, must send the registered settings and the tool on each turn, send
# the first reply back as it came and mark each request's last block alone for the prompt cache,
# so that an image that builds is one whose agent answers as registered.
RUN --network=none grep_at_d --self-test

CMD ["grep_at_d", "--help"]
