"""The embedding server's request handling (env/embedder/server.py), which serves voyage-4-nano
behind an OpenAI-compatible endpoint inside a system's environment (spec section 11). The tests
inject a fake encoder, so no model is loaded and none of torch, transformers or
sentence-transformers is imported; and they check the build-time check of the model's weights
(env/embedder/fetch_model.py) on small files."""

from __future__ import annotations

import base64
import hashlib
import json
import struct
import sys

from env.embedder import fetch_model, server

QUERY_MODEL, DOCUMENT_MODEL = "voyage-4-nano-query", "voyage-4-nano-document"
# The prompts as the model's config_sentence_transformers.json defines them at the registered
# revision, and as section 11 of the registration quotes them.
QUERY_PROMPT = "Represent the query for retrieving supporting documents: "
DOCUMENT_PROMPT = "Represent the document for retrieval: "


def vector(i: int) -> list[float]:
    """A 2048-dimensional vector, different for each ``i``, whose components float32 holds
    exactly."""
    return [((k * 7 + i * 13) % 257 - 128) / 256 for k in range(2048)]


class FakeEncoder:
    """Stands in for the model: returns ``vector(i)`` for the i-th text and five tokens a text,
    and records each call's texts and prompt name."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str]] = []

    def __call__(self, texts: list[str], prompt_name: str) -> tuple[list[list[float]], int]:
        self.calls.append((list(texts), prompt_name))
        return [vector(i) for i in range(len(texts))], 5 * len(texts)


def embed(body: dict, encode: FakeEncoder | None = None) -> tuple[int, dict]:
    return server.handle_embeddings(body, encode if encode is not None else FakeEncoder())


def test_prompt_for_model_names():
    assert server.prompt_for(QUERY_MODEL) == QUERY_PROMPT
    assert server.prompt_for(DOCUMENT_MODEL) == DOCUMENT_PROMPT
    encode = FakeEncoder()
    assert embed({"model": QUERY_MODEL, "input": "toy text"}, encode)[0] == 200
    assert embed({"model": DOCUMENT_MODEL, "input": "toy text"}, encode)[0] == 200
    assert encode.calls == [(["toy text"], "query"), (["toy text"], "document")]
    assert server.PROMPTS == {"query": QUERY_PROMPT, "document": DOCUMENT_PROMPT}


def test_unknown_model_is_400():
    encode = FakeEncoder()
    for body in ({"model": "text-embedding-3-small", "input": "x"},
                 {"model": "voyage-4-nano", "input": "x"}, {"model": ["x"], "input": "x"},
                 {"model": None, "input": "x"}, {"input": "x"}):
        status, answer = embed(body, encode)
        assert status == 400, body
        assert answer["error"]["type"] == "invalid_request_error"
        assert answer["error"]["param"] == "model"
    assert encode.calls == []


def test_string_and_list_input():
    encode = FakeEncoder()
    status, answer = embed({"model": DOCUMENT_MODEL, "input": "toy text"}, encode)
    assert status == 200
    assert [item["embedding"] for item in answer["data"]] == [vector(0)]
    status, answer = embed({"model": DOCUMENT_MODEL, "input": ["one", "two", "three"]}, encode)
    assert status == 200
    assert [item["index"] for item in answer["data"]] == [0, 1, 2]
    assert [item["embedding"] for item in answer["data"]] == [vector(0), vector(1), vector(2)]
    assert encode.calls == [(["toy text"], "document"), (["one", "two", "three"], "document")]


def test_float_encoding_round_trip():
    for options in ({}, {"encoding_format": "float"}, {"encoding_format": None}):
        status, answer = embed({"model": QUERY_MODEL, "input": ["a", "b"], **options})
        assert status == 200, options
        decoded = json.loads(json.dumps(answer))
        assert [item["embedding"] for item in decoded["data"]] == [vector(0), vector(1)]


def test_base64_encoding_is_little_endian_float32():
    body = {"model": QUERY_MODEL, "input": ["a", "b"], "encoding_format": "base64"}
    status, answer = embed(body)
    assert status == 200
    for i, item in enumerate(answer["data"]):
        assert isinstance(item["embedding"], str)
        assert list(struct.unpack("<2048f", base64.b64decode(item["embedding"]))) == vector(i)


def test_response_shape():
    status, answer = embed({"model": DOCUMENT_MODEL, "input": ["a", "b"]})
    assert status == 200
    assert set(answer) == {"object", "data", "model", "usage"}
    assert answer["object"] == "list"
    assert answer["model"] == DOCUMENT_MODEL
    assert [set(item) for item in answer["data"]] == [{"object", "index", "embedding"}] * 2
    assert [(item["object"], item["index"]) for item in answer["data"]] == [("embedding", 0),
                                                                          ("embedding", 1)]
    usage = answer["usage"]
    assert type(usage["prompt_tokens"]) is int and type(usage["total_tokens"]) is int
    assert usage == {"prompt_tokens": 10, "total_tokens": 10}


def test_empty_input_is_400():
    encode = FakeEncoder()
    for body in ({"model": DOCUMENT_MODEL, "input": ""}, {"model": DOCUMENT_MODEL, "input": []},
                 {"model": DOCUMENT_MODEL, "input": ["text", ""]}, {"model": DOCUMENT_MODEL}):
        status, answer = embed(body, encode)
        assert status == 400, body
        assert answer["error"]["param"] == "input"
    assert encode.calls == []


def test_token_arrays_and_other_options_are_400():
    encode = FakeEncoder()
    for body, param in (({"input": [[1, 2, 3]]}, "input"), ({"input": [1, 2, 3]}, "input"),
                        ({"input": {"text": "x"}}, "input"),
                        ({"input": "x", "encoding_format": "int8"}, "encoding_format"),
                        ({"input": "x", "dimensions": 1024}, "dimensions"),
                        ({"input": "x", "dimensions": True}, "dimensions")):
        status, answer = embed({"model": DOCUMENT_MODEL, **body}, encode)
        assert (status, answer["error"]["param"]) == (400, param), body
    assert encode.calls == []
    assert embed({"model": DOCUMENT_MODEL, "input": "x", "dimensions": 2048}, encode)[0] == 200


def test_routes():
    encode, health = FakeEncoder(), {"status": "ok"}
    status, answer = server.respond("GET", "/v1/models", b"", encode, health)
    assert status == 200
    assert answer["object"] == "list"
    assert sorted(model["id"] for model in answer["data"]) == [DOCUMENT_MODEL, QUERY_MODEL]
    assert server.respond("GET", "/health", b"", encode, health) == (200, health)
    request = json.dumps({"model": QUERY_MODEL, "input": "toy text"}).encode()
    assert server.respond("POST", "/v1/embeddings", request, encode, health)[0] == 200
    assert server.respond("POST", "/v1/embeddings?x=1", request, encode, health)[0] == 200
    for method, path, raw, expected in (("POST", "/v1/embeddings", b"{not json", 400),
                                        ("POST", "/v1/embeddings", b"[1, 2]", 400),
                                        ("POST", "/v1/embeddings", b"\xff\xfe\x00", 400),
                                        ("GET", "/v1/embeddings", b"", 405),
                                        ("POST", "/v1/models", b"", 405),
                                        ("GET", "/v1/chat/completions", b"", 404)):
        status, answer = server.respond(method, path, raw, encode, health)
        assert status == expected, (method, path, raw)
        assert set(answer["error"]) == {"message", "type", "param", "code"}
    assert encode.calls == [(["toy text"], "query")] * 2


def test_the_self_test_wants_unit_vectors_that_differ_by_prompt():
    def one_hot(texts: list[str], prompt_name: str) -> tuple[list[list[float]], int]:
        k = 0 if prompt_name == "query" else 1
        return [[1.0 if j == k else 0.0 for j in range(2048)] for _ in texts], 3

    def same(texts: list[str], prompt_name: str) -> tuple[list[list[float]], int]:
        return [[1.0] + [0.0] * 2047 for _ in texts], 3

    assert server.check_model(one_hot) == []
    assert server.check_model(same) == ["the query and document prompts give the same vector"]
    assert any("norm" in problem for problem in server.check_model(FakeEncoder()))


def test_the_request_handling_loads_no_model_library():
    assert not {"torch", "transformers", "sentence_transformers"} & set(sys.modules)


def test_the_server_and_the_build_name_the_same_model():
    assert server.MODEL_ID == fetch_model.MODEL_ID == "voyageai/voyage-4-nano"
    assert server.REVISION == fetch_model.REVISION == "5fb34e870f49f19e79c94d5e118f5ee730133279"
    assert server.MODEL_DIR == fetch_model.MODEL_DIR
    assert (fetch_model.WEIGHTS_SIZE, fetch_model.WEIGHTS_SHA256) == (
        692_919_112, "3dae0c63c81dcab79ac213af331940a1bf2b8a53ec8646be878552890291ad30")


def test_the_weights_check_refuses_a_wrong_size_or_digest(tmp_path):
    weights = tmp_path / "model.safetensors"
    assert "missing" in fetch_model.weights_problem(weights, 4, "0" * 64)
    weights.write_bytes(b"four")
    digest = hashlib.sha256(b"four").hexdigest()
    assert fetch_model.weights_problem(weights, 4, digest) is None
    assert "5" in fetch_model.weights_problem(weights, 5, digest)
    assert digest in fetch_model.weights_problem(weights, 4, "0" * 64)
