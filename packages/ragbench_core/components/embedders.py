from __future__ import annotations

import json
import math
import os
import urllib.request
from functools import lru_cache
from typing import Any

from ragbench_core.interfaces import BaseEmbedder
from ragbench_core.models.schema import Chunk

E5_MODEL_ALIASES = {
    "e5-base-v2": "intfloat/e5-base-v2",
    "e5-large-v2": "intfloat/e5-large-v2",
}


def _resolve_model_name(model_name: str) -> str:
    return E5_MODEL_ALIASES.get(model_name.lower(), model_name)


def _is_e5_model(model_name: str) -> bool:
    resolved_model_name = _resolve_model_name(model_name).lower()
    return resolved_model_name.startswith("intfloat/e5-") or "/e5-" in resolved_model_name


def _prefix_text(text: str, prefix: str) -> str:
    if not prefix or text.startswith(prefix):
        return text
    return f"{prefix}{text}"


def _as_list(vector):
    if hasattr(vector, "tolist"):
        return vector.tolist()
    return list(vector)


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _maybe_normalize(vectors: list[list[float]], config: dict[str, Any]) -> list[list[float]]:
    if bool(config.get("normalize", config.get("normalize_embeddings", False))):
        return [_normalize_vector(vector) for vector in vectors]
    return vectors


def _batch(items: list[str], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _batch_size(config: dict[str, Any]) -> int:
    return max(1, int(config.get("batch_size", 32)))


def _document_prefix(config: dict[str, Any], default: str = "") -> str:
    return str(config.get("document_prefix", config.get("passage_prefix", default)))


def _query_prefix(config: dict[str, Any], default: str = "") -> str:
    return str(config.get("query_prefix", default))


def _prefixed_documents(chunks: list[Chunk], config: dict[str, Any], default: str = "") -> list[str]:
    prefix = _document_prefix(config, default)
    return [_prefix_text(chunk.text, prefix) for chunk in chunks]


def _prefixed_query(query: str, config: dict[str, Any], default: str = "") -> str:
    return _prefix_text(query, _query_prefix(config, default))


def _auth_header(config: dict[str, Any], env_name: str, scheme: str = "Bearer") -> dict[str, str]:
    api_key = config.get("api_key") or os.getenv(env_name)
    if not api_key:
        raise ValueError(f"{env_name} or api_key is required for {config.get('provider', 'hosted embedding')}.")
    return {"Authorization": f"{scheme} {api_key}"}


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float = 30.0) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


@lru_cache(maxsize=4)
def _load_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class SentenceTransformerEmbedder(BaseEmbedder):
    name = "sentence_transformers"

    def embed_chunks(self, chunks: list[Chunk], config: dict) -> list[Chunk]:
        model_name = _resolve_model_name(config["model"])
        model = _load_model(model_name)
        texts = [c.text for c in chunks]
        if _is_e5_model(config["model"]):
            texts = [_prefix_text(text, _document_prefix(config, "passage: ")) for text in texts]
        else:
            texts = [_prefix_text(text, _document_prefix(config, "")) for text in texts]
        vectors = model.encode(
            texts,
            normalize_embeddings=bool(config.get("normalize", config.get("normalize_embeddings", True))),
            batch_size=_batch_size(config),
        )
        vectors = _as_list(vectors)
        for c, v in zip(chunks, vectors):
            c.embedding = _as_list(v)
        return chunks

    def embed_query(self, query: str, config: dict) -> list[float]:
        model_name = _resolve_model_name(config["model"])
        model = _load_model(model_name)
        if _is_e5_model(config["model"]):
            query = _prefixed_query(query, config, "query: ")
        else:
            query = _prefixed_query(query, config, "")
        vectors = model.encode(
            [query],
            normalize_embeddings=bool(config.get("normalize", config.get("normalize_embeddings", True))),
            batch_size=_batch_size(config),
        )
        return _as_list(vectors[0])


class BGEM3Embedder(SentenceTransformerEmbedder):
    """Local BGE-M3 embedder hosted through SentenceTransformers."""

    name = "bge_m3"

    def _config(self, config: dict[str, Any]) -> dict[str, Any]:
        return {"model": "BAAI/bge-m3", **config}

    def embed_chunks(self, chunks: list[Chunk], config: dict) -> list[Chunk]:
        return super().embed_chunks(chunks, self._config(config))

    def embed_query(self, query: str, config: dict) -> list[float]:
        return super().embed_query(query, self._config(config))


class _HostedEmbedder(BaseEmbedder):
    env_var = ""
    default_base_url = ""
    default_model = ""

    def _url(self, config: dict[str, Any]) -> str:
        base_url = str(config.get("base_url", self.default_base_url)).rstrip("/")
        return f"{base_url}/{self.path.lstrip('/')}"

    def _headers(self, config: dict[str, Any]) -> dict[str, str]:
        return _auth_header(config, self.env_var)

    def _embed_texts(self, texts: list[str], config: dict[str, Any], input_type: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch in _batch(texts, _batch_size(config)):
            response = _post_json(
                self._url(config),
                self._payload(batch, config, input_type),
                self._headers(config),
                float(config.get("timeout", 30)),
            )
            vectors.extend(self._parse_vectors(response))
        return _maybe_normalize(vectors, config)

    def embed_chunks(self, chunks: list[Chunk], config: dict) -> list[Chunk]:
        vectors = self._embed_texts(_prefixed_documents(chunks, config), config, "document")
        for chunk, vector in zip(chunks, vectors):
            chunk.embedding = vector
        return chunks

    def embed_query(self, query: str, config: dict) -> list[float]:
        return self._embed_texts([_prefixed_query(query, config)], config, "query")[0]


class OpenAIEmbedder(_HostedEmbedder):
    name = "openai"
    env_var = "OPENAI_API_KEY"
    default_base_url = "https://api.openai.com/v1"
    default_model = "text-embedding-3-small"
    path = "/embeddings"

    def _payload(self, texts, config, input_type):
        return {"model": config.get("model", self.default_model), "input": texts, "encoding_format": "float"}

    def _parse_vectors(self, response):
        return [item["embedding"] for item in sorted(response["data"], key=lambda item: item.get("index", 0))]


class VoyageEmbedder(_HostedEmbedder):
    name = "voyage"
    env_var = "VOYAGE_API_KEY"
    default_base_url = "https://api.voyageai.com/v1"
    default_model = "voyage-3"
    path = "/embeddings"

    def _payload(self, texts, config, input_type):
        return {"model": config.get("model", self.default_model), "input": texts, "input_type": input_type}

    def _parse_vectors(self, response):
        return [item["embedding"] for item in response["data"]]


class CohereEmbedder(_HostedEmbedder):
    name = "cohere"
    env_var = "COHERE_API_KEY"
    default_base_url = "https://api.cohere.com/v2"
    default_model = "embed-v4.0"
    path = "/embed"

    def _payload(self, texts, config, input_type):
        return {"model": config.get("model", self.default_model), "texts": texts, "input_type": "search_document" if input_type == "document" else "search_query", "embedding_types": ["float"]}

    def _parse_vectors(self, response):
        embeddings = response["embeddings"]
        return embeddings.get("float", embeddings) if isinstance(embeddings, dict) else embeddings


class JinaEmbedder(_HostedEmbedder):
    name = "jina"
    env_var = "JINA_API_KEY"
    default_base_url = "https://api.jina.ai/v1"
    default_model = "jina-embeddings-v3"
    path = "/embeddings"

    def _payload(self, texts, config, input_type):
        task = "retrieval.passage" if input_type == "document" else "retrieval.query"
        return {"model": config.get("model", self.default_model), "input": texts, "task": config.get(f"{input_type}_task", task)}

    def _parse_vectors(self, response):
        return [item["embedding"] for item in sorted(response["data"], key=lambda item: item.get("index", 0))]
