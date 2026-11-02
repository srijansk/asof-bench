# Grep at D's sandbox: the command server (command_server.py), which runs the agent's read-only
# shell commands, one at a time, in the tree that run_dates.py builds for one arm and date and the
# compose file mounts read-only at /repo. Build it with this directory as the context:
#
#     docker build -t asofbench/sandbox systems/grep-at-d/sandbox
#
# Only the build reaches the network, for Debian's packages; the container then runs on a network
# that only the agent's container shares, which leads nowhere else.
#
# debian:stable-slim (Debian 13, trixie, built 2026-10-27), its multi-platform index pinned by
# digest. bash, coreutils, grep, findutils and sed come with it, as Debian's essential packages;
# gawk, and python3 for the command server, are installed from the snapshot of the Debian archive
# at the time the base image was built, so that a rebuild installs the same versions: the image's
# own package sources are removed, and the snapshot's are the only ones apt reads. The build log
# lists the version of each package the sandbox's commands use.
FROM debian:stable-slim@sha256:13e0a2e78aa0554ad5ff66e4e260c6dbcdff807eb1f39d43b9fb0288aae88b04

RUN rm -f /etc/apt/sources.list /etc/apt/sources.list.d/* \
    && printf '%s\n' \
       'Types: deb' \
       'URIs: http://snapshot.debian.org/archive/debian/20261027T000000Z' \
       'Suites: trixie trixie-updates' \
       'Components: main' \
       'Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg' \
       'Check-Valid-Until: no' \
       '' \
       'Types: deb' \
       'URIs: http://snapshot.debian.org/archive/debian-security/20261027T000000Z' \
       'Suites: trixie-security' \
       'Components: main' \
       'Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg' \
       'Check-Valid-Until: no' \
       > /etc/apt/sources.list.d/snapshot.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends gawk python3 \
    && rm -rf /var/lib/apt/lists/* \
    && dpkg-query -W -f '${Package}=${Version}\n' bash coreutils grep findutils sed gawk python3

RUN groupadd --system --gid 10001 sandbox \
    && useradd --system --uid 10001 --gid 10001 --no-create-home --home-dir /nonexistent \
       --shell /usr/sbin/nologin sandbox \
    && mkdir /repo

COPY command_server.py /opt/sandbox/command_server.py

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /repo
USER 10001:10001

# The self-test runs with no network at all: it checks that the sandbox's tools are installed and
# that a command runs, its output cut at a limit with the final line, so that an image that builds
# is one that runs the agent's commands.
RUN --network=none python3 /opt/sandbox/command_server.py --self-test

EXPOSE 7000
CMD ["python3", "/opt/sandbox/command_server.py", "--root", "/repo", "--scratch", "/tmp", "--port", "7000"]
