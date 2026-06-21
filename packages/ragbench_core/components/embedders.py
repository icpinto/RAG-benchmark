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
        return self.embed_queries([query], config)[0]

    def embed_queries(self, queries: list[str], config: dict) -> list[list[float]]:
        model_name = _resolve_model_name(config["model"])
        model = _load_model(model_name)
        if _is_e5_model(config["model"]):
            texts = [_prefixed_query(query, config, "query: ") for query in queries]
        else:
            texts = [_prefixed_query(query, config, "") for query in queries]
        vectors = model.encode(
            texts,
            normalize_embeddings=bool(config.get("normalize", config.get("normalize_embeddings", True))),
            batch_size=_batch_size(config),
        )
        return [_as_list(vector) for vector in vectors]


class BGEM3Embedder(SentenceTransformerEmbedder):
    """Local BGE-M3 embedder hosted through SentenceTransformers."""

    name = "bge_m3"

    def _config(self, config: dict[str, Any]) -> dict[str, Any]:
        return {"model": "BAAI/bge-m3", **config}

    def embed_chunks(self, chunks: list[Chunk], config: dict) -> list[Chunk]:
        return super().embed_chunks(chunks, self._config(config))

    def embed_query(self, query: str, config: dict) -> list[float]:
        return super().embed_query(query, self._config(config))

    def embed_queries(self, queries: list[str], config: dict) -> list[list[float]]:
        return super().embed_queries(queries, self._config(config))


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
        return self.embed_queries([query], config)[0]

    def embed_queries(self, queries: list[str], config: dict) -> list[list[float]]:
        return self._embed_texts([_prefixed_query(query, config) for query in queries], config, "query")


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


def _sparse_dict_from_indices_values(indices: list[int], values: list[float]) -> dict[int, float]:
    sparse: dict[int, float] = {}
    for index, value in zip(indices, values):
        if value:
            sparse[int(index)] = float(value)
    return sparse


def _top_k_sparse(sparse: dict[int, float], top_k: int | None) -> dict[int, float]:
    if not top_k or len(sparse) <= top_k:
        return sparse
    return dict(sorted(sparse.items(), key=lambda item: item[1], reverse=True)[:top_k])


class BGEM3SparseEmbedder(BaseEmbedder):
    """Generate BGE-M3 lexical sparse vectors with FlagEmbedding.

    The vectors are stored on ``Chunk.sparse_embedding`` as ``token_id -> weight``
    dictionaries so sparse indexers can persist them without changing the dense path.
    """

    name = "bge_m3_sparse"

    def _model(self, config: dict[str, Any]):
        from FlagEmbedding import BGEM3FlagModel

        return BGEM3FlagModel(config.get("model", "BAAI/bge-m3"), use_fp16=bool(config.get("use_fp16", True)))

    def _encode(self, texts: list[str], config: dict[str, Any]) -> list[dict[int, float]]:
        model = self._model(config)
        encoded = model.encode(
            texts,
            batch_size=_batch_size(config),
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        sparse_vectors = encoded.get("lexical_weights", encoded)
        top_k = config.get("sparse_top_k")
        out: list[dict[int, float]] = []
        for sparse in sparse_vectors:
            if isinstance(sparse, dict):
                out.append(_top_k_sparse({int(k): float(v) for k, v in sparse.items() if float(v)}, top_k))
            else:
                out.append(_top_k_sparse(dict(sparse), top_k))
        return out

    def embed_chunks(self, chunks: list[Chunk], config: dict) -> list[Chunk]:
        vectors = self._encode(_prefixed_documents(chunks, config), config)
        for chunk, vector in zip(chunks, vectors):
            chunk.sparse_embedding = vector
        return chunks

    def embed_query(self, query: str, config: dict) -> dict[int, float]:
        return self.embed_queries([query], config)[0]

    def embed_queries(self, queries: list[str], config: dict) -> list[dict[int, float]]:
        return self._encode([_prefixed_query(query, config) for query in queries], config)


class SPLADESparseEmbedder(BaseEmbedder):
    """Generate SPLADE-style sparse vectors with Hugging Face Transformers."""

    name = "splade_sparse"

    @lru_cache(maxsize=2)
    def _load(self, model_name: str):
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        return AutoTokenizer.from_pretrained(model_name), AutoModelForMaskedLM.from_pretrained(model_name)

    def _encode(self, texts: list[str], config: dict[str, Any]) -> list[dict[int, float]]:
        import torch

        model_name = config.get("model", "naver/splade-cocondenser-ensembledistil")
        tokenizer, model = self._load(model_name)
        max_length = int(config.get("max_length", 512))
        top_k = config.get("sparse_top_k")
        vectors: list[dict[int, float]] = []
        model.eval()
        with torch.no_grad():
            for batch in _batch(texts, _batch_size(config)):
                inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
                logits = model(**inputs).logits
                weights = torch.log1p(torch.relu(logits)) * inputs["attention_mask"].unsqueeze(-1)
                sparse_batch = torch.max(weights, dim=1).values
                for row in sparse_batch:
                    indices = torch.nonzero(row, as_tuple=False).flatten()
                    values = row[indices]
                    sparse = _sparse_dict_from_indices_values(indices.tolist(), values.tolist())
                    vectors.append(_top_k_sparse(sparse, top_k))
        return vectors

    def embed_chunks(self, chunks: list[Chunk], config: dict) -> list[Chunk]:
        vectors = self._encode(_prefixed_documents(chunks, config), config)
        for chunk, vector in zip(chunks, vectors):
            chunk.sparse_embedding = vector
        return chunks

    def embed_query(self, query: str, config: dict) -> dict[int, float]:
        return self.embed_queries([query], config)[0]

    def embed_queries(self, queries: list[str], config: dict) -> list[dict[int, float]]:
        return self._encode([_prefixed_query(query, config) for query in queries], config)


class BGEM3HybridEmbedder(BGEM3SparseEmbedder):
    """Generate both BGE-M3 dense and lexical sparse embeddings in one pass."""

    name = "bge_m3_hybrid"

    def _encode_hybrid(self, texts: list[str], config: dict[str, Any]) -> dict[str, Any]:
        model = self._model(config)
        return model.encode(
            texts,
            batch_size=_batch_size(config),
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

    def embed_chunks(self, chunks: list[Chunk], config: dict) -> list[Chunk]:
        encoded = self._encode_hybrid(_prefixed_documents(chunks, config), config)
        dense_vectors = _maybe_normalize(_as_list(encoded["dense_vecs"]), config)
        sparse_vectors = encoded.get("lexical_weights", [])
        top_k = config.get("sparse_top_k")
        for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors):
            chunk.embedding = _as_list(dense)
            chunk.sparse_embedding = _top_k_sparse({int(k): float(v) for k, v in sparse.items() if float(v)}, top_k)
        return chunks

    def embed_query(self, query: str, config: dict) -> dict[str, Any]:
        return self.embed_queries([query], config)[0]

    def embed_queries(self, queries: list[str], config: dict) -> list[dict[str, Any]]:
        encoded = self._encode_hybrid([_prefixed_query(query, config) for query in queries], config)
        dense_vectors = _maybe_normalize(_as_list(encoded["dense_vecs"]), config)
        sparse_vectors = encoded.get("lexical_weights", [{} for _ in queries])
        return [
            {
                "dense": _as_list(dense),
                "sparse": _top_k_sparse({int(k): float(v) for k, v in sparse.items() if float(v)}, config.get("sparse_top_k")),
            }
            for dense, sparse in zip(dense_vectors, sparse_vectors)
        ]
