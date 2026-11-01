# The egress proxy, the one route out of a system's environment (proxy.py). Build it with this
# directory as the context: docker build -t asofbench/egress systems/env/egress
#
# python:3.12-slim (Python 3.12.16 on Debian trixie), its multi-platform index pinned by digest.
FROM python:3.12-slim@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --system --gid 10001 egress \
    && useradd --system --uid 10001 --gid 10001 --no-create-home --home-dir /nonexistent \
       --shell /usr/sbin/nologin egress

COPY proxy.py /app/proxy.py

USER 10001:10001
EXPOSE 3128
CMD ["python", "/app/proxy.py"]
