# The embedding server: voyage-4-nano behind an OpenAI-compatible endpoint (server.py). Build it
# with this directory as the context: docker build -t asofbench/embedder systems/env/embedder
#
# Only the build reaches the network: it installs the locked packages and downloads the model at
# its registered revision, checking both, and the container then runs with no route out.
#
# python:3.12-slim (Python 3.12.16 on Debian trixie), its multi-platform index pinned by digest.
FROM python:3.12-slim@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# Every package exactly as requirements.txt pins it, hash-checked and from wheels only: torch's
# CPU-only wheel from the PyTorch CPU index's torch page, every other package from PyPI.
COPY requirements.txt /app/requirements.txt
RUN pip install --require-hashes --only-binary :all: --index-url https://pypi.org/simple \
        --find-links https://download.pytorch.org/whl/cpu/torch/ -r /app/requirements.txt \
    && pip check

# The model at its registered revision: fetch_model.py fails the build unless the weights have the
# registered size and SHA-256. The download's cache goes in the same layer as it came.
COPY fetch_model.py /app/fetch_model.py
RUN HF_HOME=/tmp/hf-build python /app/fetch_model.py && rm -rf /tmp/hf-build

RUN groupadd --system --gid 10001 embedder \
    && useradd --system --uid 10001 --gid 10001 --create-home --home-dir /home/embedder \
       --shell /usr/sbin/nologin embedder

COPY server.py /app/server.py

# From here on the model is read offline. The self-test runs with no network at all: it loads the
# model as the server does, in bfloat16, which leaves the model's code file in the user's module
# cache, checks that its weights are bfloat16 and checks the float32 vectors it gives one text,
# so that an image that builds is one that serves.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/home/embedder/.cache/huggingface
USER 10001:10001
RUN --network=none python /app/server.py --self-test

EXPOSE 8080
CMD ["python", "/app/server.py"]
