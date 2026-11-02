"""The embedding server: voyage-4-nano behind an OpenAI-compatible endpoint, inside a system's
environment (spec section 11).

Every system that embeds text uses one model, voyageai/voyage-4-nano at the registered revision,
run inside its own environment at its default 2048 dimensions, with the model's ``query`` prompt
for a question and its ``document`` prompt for every text a system stores. This server is that
model's endpoint. ``POST /v1/embeddings`` answers two model names, one per prompt,
``voyage-4-nano-query`` and ``voyage-4-nano-document``, so a system chooses the prompt by the
model name its OpenAI client sends; ``GET /v1/models`` lists the two, and ``GET /health`` says
which model and revision are served. It answers ``encoding_format`` ``float`` and ``base64``
(little-endian float32, which the OpenAI SDK asks for by default), for a string or a list of
strings, and refuses token arrays, empty texts and any ``dimensions`` but 2048.

The model is loaded once, at start, from the snapshot that the image's build downloaded and
checked (fetch_model.py), with the network off (``HF_HUB_OFFLINE`` and ``TRANSFORMERS_OFFLINE``),
on the CPU in bfloat16, the dtype its weights are published in (its config.json's
``torch_dtype``); the vectors are the model's own, normalised by its Normalize module, and each is
converted to float32, which holds every bfloat16 value exactly, before it is checked or
serialised. One lock serialises the requests, since one model runs on the CPU. ``usage`` counts
the tokens the model reads: each text with its prompt, as the model's tokenizer splits them.

The request handling imports only the standard library: torch, transformers and
sentence-transformers are imported only where the model is loaded and run.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import signal
import struct
import sys
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

MODEL_ID = "voyageai/voyage-4-nano"
REVISION = "5fb34e870f49f19e79c94d5e118f5ee730133279"
MODEL_DIR = "/models/voyage-4-nano"
DIMENSIONS = 2048
# The prompts that the model's config_sentence_transformers.json defines at the registered
# revision, by name; the server refuses to start if the loaded model's differ.
PROMPTS = {"query": "Represent the query for retrieving supporting documents: ",
           "document": "Represent the document for retrieval: "}
# The model names a client sends, and the prompt each one selects.
PROMPT_NAMES = {"voyage-4-nano-query": "query", "voyage-4-nano-document": "document"}
ENCODINGS = ("float", "base64")
ROUTES = {"/v1/embeddings": "POST", "/v1/models": "GET", "/health": "GET"}
LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 8080
# The dtype the model runs in, that of its published weights, and the dtype of the vectors served.
WEIGHTS_DTYPE, VECTOR_DTYPE = "bfloat16", "float32"
# How far from 1 the norm of a vector may be in the self-test. The Normalize module divides in
# bfloat16, which rounds the norm it divides by and then each quotient to 8 significant bits, a
# relative error of at most 2^-8 each, so the norm of a vector it returns is within 2^-7 of 1.
NORM_TOLERANCE = 2 ** -7
SELF_TEST_TEXT = "toy text"

# An encoder takes the texts and a prompt name, and returns one vector per text and the number of
# tokens the model read.
Encoder = Callable[[list[str], str], tuple[Sequence[Sequence[float]], int]]


def prompt_for(model: str) -> str:
    """The prompt that the model name ``model`` selects."""
    return PROMPTS[PROMPT_NAMES[model]]


def error(status: int, message: str, param: str | None = None) -> tuple[int, dict]:
    """An error in the OpenAI API's shape."""
    kind = "invalid_request_error" if status < 500 else "server_error"
    return status, {"error": {"message": message, "type": kind, "param": param, "code": None}}


def handle_embeddings(body: object, encode: Encoder) -> tuple[int, dict]:
    """The status and JSON body that answer a ``POST /v1/embeddings`` whose JSON body is
    ``body``; ``encode`` is called only for a valid request."""
    if not isinstance(body, dict):
        return error(400, "The request body must be a JSON object.")
    model = body.get("model")
    if not isinstance(model, str) or model not in PROMPT_NAMES:
        return error(400, f"model must be one of {', '.join(sorted(PROMPT_NAMES))}.", "model")
    texts = body.get("input")
    if isinstance(texts, str):
        texts = [texts]
    if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
        return error(400, "input must be a string or a list of strings; token arrays are not "
                     "accepted.", "input")
    if not texts or not all(texts):
        return error(400, "input must hold at least one text, and no text may be empty.",
                     "input")
    encoding = body.get("encoding_format")
    encoding = "float" if encoding is None else encoding
    if encoding not in ENCODINGS:
        return error(400, "encoding_format must be float or base64.", "encoding_format")
    dimensions = body.get("dimensions")
    if dimensions is not None and (type(dimensions) is not int or dimensions != DIMENSIONS):
        return error(400, f"dimensions must be {DIMENSIONS}, the model's default, or be left out.",
                     "dimensions")
    vectors, tokens = encode(texts, PROMPT_NAMES[model])
    if len(vectors) != len(texts) or any(len(values) != DIMENSIONS for values in vectors):
        return error(500, "The model returned the wrong number or size of vectors.")
    data = []
    for index, values in enumerate(vectors):
        packed = struct.pack(f"<{DIMENSIONS}f", *values)
        embedding = (base64.b64encode(packed).decode("ascii") if encoding == "base64"
                     else list(struct.unpack(f"<{DIMENSIONS}f", packed)))
        data.append({"object": "embedding", "index": index, "embedding": embedding})
    tokens = int(tokens)
    return 200, {"object": "list", "data": data, "model": model,
                 "usage": {"prompt_tokens": tokens, "total_tokens": tokens}}


def handle_models() -> tuple[int, dict]:
    """The answer to ``GET /v1/models``: the two model names."""
    return 200, {"object": "list",
                 "data": [{"id": name, "object": "model", "created": 0, "owned_by": MODEL_ID}
                          for name in sorted(PROMPT_NAMES)]}


def respond(method: str, path: str, raw: bytes, encode: Encoder,
            health: dict) -> tuple[int, dict]:
    """The status and JSON body that answer ``method`` on ``path`` with the request body
    ``raw``."""
    path = urlsplit(path).path
    if path not in ROUTES:
        return error(404, f"There is no {path} here.")
    if method != ROUTES[path]:
        return error(405, f"{path} answers {ROUTES[path]} only.")
    if path == "/health":
        return 200, health
    if path == "/v1/models":
        return handle_models()
    try:
        body = json.loads(raw)
    except ValueError:
        return error(400, "The request body is not JSON.")
    return handle_embeddings(body, encode)


def check_model(encode: Encoder) -> list[str]:
    """What is wrong with the vectors that ``encode`` gives one text under each prompt: each must
    have DIMENSIONS components and unit norm within NORM_TOLERANCE, and the two must differ."""
    problems, vectors = [], {}
    for model in sorted(PROMPT_NAMES):
        status, answer = handle_embeddings({"model": model, "input": SELF_TEST_TEXT}, encode)
        if status != 200:
            problems.append(f"{model}: status {status}, {answer['error']['message']}")
            continue
        values = answer["data"][0]["embedding"]
        norm = math.sqrt(math.fsum(value * value for value in values))
        if abs(norm - 1) > NORM_TOLERANCE:
            problems.append(f"{model}: the vector's norm is {norm}, not 1")
        vectors[model] = values
    if len(vectors) == 2 and len({tuple(values) for values in vectors.values()}) == 1:
        problems.append("the query and document prompts give the same vector")
    return problems


def load_encoder(model_dir: str = MODEL_DIR) -> Encoder:
    """voyage-4-nano from its local snapshot, loaded offline on the CPU in bfloat16 and checked to
    be the registered pipeline, as an encoder that returns float32 vectors and serialises its
    calls through one lock."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import Normalize

    weights, vector = getattr(torch, WEIGHTS_DTYPE), getattr(torch, VECTOR_DTYPE)
    model = SentenceTransformer(model_dir, trust_remote_code=True, device="cpu",
                                model_kwargs={"dtype": weights})
    problems = [f"the model's {name} prompt is {model.prompts.get(name)!r}, not {prompt!r}"
                for name, prompt in PROMPTS.items() if model.prompts.get(name) != prompt]
    if model.get_embedding_dimension() != DIMENSIONS:
        problems.append(f"the model has {model.get_embedding_dimension()} dimensions")
    if not isinstance(model[-1], Normalize):
        problems.append(f"the model's last module is {type(model[-1]).__name__}, not Normalize")
    dtypes = {parameter.dtype for parameter in model.parameters()}
    if dtypes != {weights}:
        problems.append(f"the model's weights are {sorted(map(str, dtypes))}, not "
                        f"{WEIGHTS_DTYPE}")
    if problems:
        raise RuntimeError("; ".join(problems))
    lock = threading.Lock()

    def encode(texts: list[str], prompt_name: str) -> tuple[list[list[float]], int]:
        with lock:
            # The model's own preprocessing, with the prompt as encode applies it, so that the
            # count is of the tokens the model reads.
            features = model.preprocess(texts, prompt=model.prompts[prompt_name])
            vectors = model.encode(texts, prompt_name=prompt_name, convert_to_tensor=True,
                                   show_progress_bar=False)
        # Each vector in float32, which holds the bfloat16 values the model gave exactly.
        return vectors.to(vector).tolist(), int(features["attention_mask"].sum())

    return encode


def timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class Handler(BaseHTTPRequestHandler):
    """The HTTP layer over ``respond``; it logs one JSON line per request on stdout, with no
    text of any request in it."""

    protocol_version = "HTTP/1.1"
    server_version = "asofbench-embedder"
    sys_version = ""

    def do_GET(self) -> None:
        self._answer(b"")

    def do_POST(self) -> None:
        length = self.headers.get("Content-Length", "0")
        if "Transfer-Encoding" in self.headers or not re.fullmatch("[0-9]+", length):
            # Where such a body ends is not known, so the connection cannot carry another request.
            self.close_connection = True
            self._answer(b"", error(400, "The body must come with a Content-Length."))
            return
        self._answer(self.rfile.read(int(length)))

    def _answer(self, raw: bytes, answer: tuple[int, dict] | None = None) -> None:
        started = time.monotonic()
        if answer is None:
            try:
                answer = respond(self.command, self.path, raw, self.server.encode,
                                 self.server.health)
            except Exception:
                traceback.print_exc(file=sys.stderr)
                answer = error(500, "The model failed on this request.")
        status, body = answer
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        usage = body.get("usage") or {}
        record = {"ts": timestamp(), "method": self.command, "path": urlsplit(self.path).path,
                  "status": status, "inputs": len(body["data"]) if usage else None,
                  "tokens": usage.get("prompt_tokens"),
                  "seconds": round(time.monotonic() - started, 3)}
        sys.stdout.write(json.dumps(record) + "\n")
        sys.stdout.flush()

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        """Replaced by the JSON line that ``_answer`` writes; errors are still logged on
        stderr."""


class EmbeddingServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], encode: Encoder, health: dict) -> None:
        super().__init__(address, Handler)
        self.encode = encode
        self.health = health


def _interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serves voyage-4-nano through an "
                                                 "OpenAI-compatible endpoint on port 8080.")
    parser.add_argument("--self-test", action="store_true",
                        help="load the model, check its weights' dtype and the vectors it gives "
                             "one text under each prompt, and exit")
    args = parser.parse_args(argv)
    encode = load_encoder()
    if args.self_test:
        problems = check_model(encode)
        for problem in problems:
            print(f"embedder: {problem}", file=sys.stderr)
        if not problems:
            print(f"embedder: self-test passed: {WEIGHTS_DTYPE} weights, {DIMENSIONS} dimensions, "
                  f"{VECTOR_DTYPE} vectors of unit norm within 2^-7, and a different vector for "
                  "each prompt")
        return 1 if problems else 0
    health = {"status": "ok", "model": MODEL_ID, "revision": REVISION, "dimensions": DIMENSIONS,
              "weights_dtype": WEIGHTS_DTYPE, "vector_dtype": VECTOR_DTYPE,
              "models": sorted(PROMPT_NAMES)}
    server = EmbeddingServer((LISTEN_HOST, LISTEN_PORT), encode, health)
    signal.signal(signal.SIGTERM, _interrupt)
    print(f"embedder: serving {MODEL_ID} at {REVISION} in {WEIGHTS_DTYPE} on "
          f"{LISTEN_HOST}:{LISTEN_PORT}", file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
