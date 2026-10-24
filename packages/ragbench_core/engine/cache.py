from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Protocol

from ragbench_core.models.schema import Chunk, RetrievalResult

logger = logging.getLogger(__name__)
DEMO_CACHE_FAULTS = frozenset({"cache_schema", "cache_serialization"})

_SECRET_KEYS = {"api_key", "token", "password", "secret", "authorization"}
_EMBEDDER_CONFIG_KEYS = {
    "provider",
    "model",
    "document_prefix",
    "passage_prefix",
    "query_prefix",
    "normalize",
    "normalize_embeddings",
    "dimensions",
    "dimension",
    "truncate_dim",
    "embedding_dim",
}


class CacheBackend(Protocol):
    def get(self, key: str) -> Any | None: ...
    def set(self, key: str, value: Any) -> None: ...


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _safe_embedder_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return cache-key material without credentials or transport-only options."""
    safe: dict[str, Any] = {}
    for key, value in config.items():
        lower_key = key.lower()
        if lower_key in _SECRET_KEYS or any(secret in lower_key for secret in _SECRET_KEYS):
            continue
        if key in _EMBEDDER_CONFIG_KEYS:
            safe[key] = value
    return safe



def _restore_json_types(value: Any) -> Any:
    if isinstance(value, list):
        return [_restore_json_types(item) for item in value]
    if isinstance(value, dict):
        restored = {key: _restore_json_types(item) for key, item in value.items()}
        if restored and all(isinstance(key, str) and key.isdigit() for key in restored):
            return {int(key): item for key, item in restored.items()}
        return restored
    return value


def _chunk_embedding_payload(chunk: Chunk) -> dict[str, Any]:
    return {"embedding": chunk.embedding, "sparse_embedding": chunk.sparse_embedding}


def _apply_chunk_embedding_payload(chunk: Chunk, payload: Any) -> None:
    restored = _restore_json_types(payload)
    if isinstance(restored, dict) and ("embedding" in restored or "sparse_embedding" in restored):
        chunk.embedding = restored.get("embedding")
        chunk.sparse_embedding = restored.get("sparse_embedding")
    else:
        # Backward compatibility for cache entries written before sparse payloads were stored.
        chunk.embedding = restored

def embedding_namespace(embedder_config: dict[str, Any]) -> str:
    """Build the provider/model/config namespace shared by document and query keys."""
    material = {
        "provider": embedder_config.get("provider"),
        "model": embedder_config.get("model"),
        "config": _safe_embedder_config(embedder_config),
    }
    return _sha256_text(_canonical_json(material))


def document_embedding_key(embedder_config: dict[str, Any], chunk: Chunk) -> str:
    material = {
        "kind": "document",
        "namespace": embedding_namespace(embedder_config),
        "chunk_id": chunk.chunk_id,
        "text_hash": _sha256_text(chunk.text),
    }
    return f"ragbench:embedding:{_sha256_text(_canonical_json(material))}"


def query_embedding_key(embedder_config: dict[str, Any], query: str) -> str:
    material = {
        "kind": "query",
        "namespace": embedding_namespace(embedder_config),
        "text_hash": _sha256_text(query),
    }
    return f"ragbench:embedding:{_sha256_text(_canonical_json(material))}"


class FileEmbeddingCache:
    def __init__(self, root: Path, *, demo_faults: set[str] | None = None):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        # Demo faults are opt-in and fire at most once per cache instance.
        if demo_faults is None:
            demo_faults = {item.strip() for item in os.getenv("RAGBENCH_DEMO_FAULTS", "").split(",")}
        self._pending_demo_faults = DEMO_CACHE_FAULTS.intersection(demo_faults)

    def _path(self, key: str) -> Path:
        digest = key.rsplit(":", 1)[-1]
        return self.root / f"{digest}.json"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        cached = None
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                cached = json.load(handle)
        if "cache_schema" in self._pending_demo_faults:
            self._pending_demo_faults = self._pending_demo_faults - {"cache_schema"}
            try:
                # Intentional demo bug: writer and reader disagree on the envelope key.
                envelope = {"payload": cached}
                return envelope["value"]
            except KeyError:
                logger.exception(
                    "event=cache_read_failed demo=true fault_id=cache_schema "
                    "cache_key=%s cache_path=%s recovery=cache_miss", key, path,
                )
                logger.info(
                    "event=cache_read_recovered demo=true fault_id=cache_schema "
                    "cache_key=%s recovery=cache_miss", key,
                )
                return None
        return cached

    def set(self, key: str, value: Any) -> None:
        path = self._path(key)
        retry_write = False
        if "cache_serialization" in self._pending_demo_faults:
            self._pending_demo_faults = self._pending_demo_faults - {"cache_serialization"}
            try:
                # Intentional demo bug: encoding too early passes bytes to the JSON encoder.
                encoded = json.dumps(value).encode("utf-8")
                json.dumps(encoded)
            except TypeError:
                logger.exception(
                    "event=cache_write_failed demo=true fault_id=cache_serialization "
                    "cache_key=%s cache_path=%s recovery=retry_json_write", key, path,
                )
                retry_write = True
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"))
        tmp_path.replace(path)
        if retry_write:
            logger.info(
                "event=cache_write_recovered demo=true fault_id=cache_serialization "
                "cache_key=%s recovery=retry_json_write", key,
            )


class RedisEmbeddingCache:
    def __init__(self, redis_url: str):
        import redis

        self.client = redis.Redis.from_url(redis_url)

    def get(self, key: str) -> Any | None:
        value = self.client.get(key)
        if value is None:
            return None
        return json.loads(value.decode("utf-8") if isinstance(value, bytes) else value)

    def set(self, key: str, value: Any) -> None:
        self.client.set(key, json.dumps(value, sort_keys=True, separators=(",", ":")))


class NullEmbeddingCache:
    def get(self, key: str) -> Any | None:
        return None

    def set(self, key: str, value: Any) -> None:
        return None


def _safe_component_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return deterministic cache-key material with credential-like values removed."""
    safe: dict[str, Any] = {}
    for key, value in config.items():
        lower_key = key.lower()
        if lower_key in _SECRET_KEYS or any(secret in lower_key for secret in _SECRET_KEYS):
            continue
        if isinstance(value, dict):
            safe[key] = _safe_component_config(value)
        elif isinstance(value, list):
            safe[key] = [_safe_component_config(item) if isinstance(item, dict) else item for item in value]
        else:
            safe[key] = value
    return safe


def _embedding_hash(query_embedding: Any) -> str | None:
    if query_embedding is None:
        return None
    return _sha256_text(_canonical_json(_restore_json_types(query_embedding)))


def retrieval_cache_key(
    index_name: str,
    retriever_type: str,
    retriever_config: dict[str, Any],
    rewritten_query: str,
    query_embedding: Any,
    top_k: int,
) -> str:
    material = {
        "kind": "retrieval",
        "index_name": index_name,
        "retriever": {
            "type": retriever_type,
            "config": _safe_component_config(retriever_config),
        },
        "rewritten_query": rewritten_query,
        "query_embedding_hash": _embedding_hash(query_embedding),
        "top_k": top_k,
    }
    return f"ragbench:retrieval:{_sha256_text(_canonical_json(material))}"


def retrieval_result_to_payload(result: RetrievalResult) -> dict[str, Any]:
    return {
        "chunk": {
            "chunk_id": result.chunk.chunk_id,
            "doc_id": result.chunk.doc_id,
            "text": result.chunk.text,
            "metadata": result.chunk.metadata,
            "embedding": result.chunk.embedding,
            "sparse_embedding": result.chunk.sparse_embedding,
        },
        "score": result.score,
        "rank": result.rank,
        "retriever": result.retriever,
    }


def retrieval_result_from_payload(payload: dict[str, Any]) -> RetrievalResult:
    chunk_payload = payload["chunk"]
    chunk = Chunk(
        chunk_payload["chunk_id"],
        chunk_payload["doc_id"],
        chunk_payload["text"],
        _restore_json_types(chunk_payload.get("metadata", {})),
        chunk_payload.get("embedding"),
        _restore_json_types(chunk_payload.get("sparse_embedding")),
    )
    return RetrievalResult(chunk, float(payload["score"]), int(payload["rank"]), payload["retriever"])


def build_retrieval_cache(config: dict[str, Any]) -> CacheBackend:
    cache_config = config.get("runtime", {}).get("retrieval_cache", {})
    if not cache_config or not cache_config.get("enabled", False):
        return NullEmbeddingCache()
    backend = str(cache_config.get("backend", "file")).lower()
    if backend == "file":
        artifact_dir = Path(config.get("runtime", {}).get("artifact_dir") or "/app/artifacts")
        cache_dir = Path(cache_config.get("path") or artifact_dir / "retrieval_cache")
        return FileEmbeddingCache(cache_dir)
    if backend == "redis":
        if cache_config.get("redis_url"):
            redis_url = str(cache_config["redis_url"])
        else:
            from ragbench_core.settings import settings

            redis_url = settings.redis_url
        return RedisEmbeddingCache(redis_url)
    raise ValueError(f"Unsupported retrieval_cache backend: {backend}")


def retrieve_batch_cached(
    cache: CacheBackend,
    retriever: Any,
    queries: list[str],
    query_embeddings: list[Any],
    index_name: str,
    retriever_config: dict[str, Any],
    top_k: int,
) -> list[list[RetrievalResult]]:
    retriever_type = str(
        retriever_config.get("type", getattr(retriever, "name", retriever.__class__.__name__))
    )
    results: list[list[RetrievalResult] | None] = [None] * len(queries)
    missing_queries: list[str] = []
    missing_embeddings: list[Any] = []
    missing_indexes: list[int] = []
    missing_keys: list[str] = []
    for idx, query in enumerate(queries):
        query_embedding = query_embeddings[idx] if idx < len(query_embeddings) else None
        key = retrieval_cache_key(index_name, retriever_type, retriever_config, query, query_embedding, top_k)
        cached = cache.get(key)
        if cached is None:
            missing_queries.append(query)
            missing_embeddings.append(query_embedding)
            missing_indexes.append(idx)
            missing_keys.append(key)
        else:
            results[idx] = [retrieval_result_from_payload(item) for item in cached]
    if missing_queries:
        retrieved = retriever.retrieve_batch(missing_queries, missing_embeddings, index_name, retriever_config, top_k)
        for idx, key, result_set in zip(missing_indexes, missing_keys, retrieved):
            cache.set(key, [retrieval_result_to_payload(result) for result in result_set])
            results[idx] = result_set
    return [result_set or [] for result_set in results]


def build_embedding_cache(config: dict[str, Any]) -> CacheBackend:
    cache_config = config.get("runtime", {}).get("embedding_cache", {})
    if not cache_config or not cache_config.get("enabled", False):
        return NullEmbeddingCache()
    backend = str(cache_config.get("backend", "file")).lower()
    if backend == "file":
        artifact_dir = Path(config.get("runtime", {}).get("artifact_dir") or "/app/artifacts")
        cache_dir = Path(cache_config.get("path") or artifact_dir / "embedding_cache")
        return FileEmbeddingCache(cache_dir)
    if backend == "redis":
        if cache_config.get("redis_url"):
            redis_url = str(cache_config["redis_url"])
        else:
            from ragbench_core.settings import settings

            redis_url = settings.redis_url
        return RedisEmbeddingCache(redis_url)
    raise ValueError(f"Unsupported embedding_cache backend: {backend}")


def embed_chunks_cached(cache: CacheBackend, embedder: Any, chunks: list[Chunk], embedder_config: dict[str, Any]) -> list[Chunk]:
    missing_chunks: list[Chunk] = []
    missing_keys: list[str] = []
    for chunk in chunks:
        key = document_embedding_key(embedder_config, chunk)
        cached = cache.get(key)
        if cached is None:
            missing_chunks.append(chunk)
            missing_keys.append(key)
        else:
            _apply_chunk_embedding_payload(chunk, cached)
    if missing_chunks:
        embedded = embedder.embed_chunks(missing_chunks, embedder_config)
        for key, chunk in zip(missing_keys, embedded):
            cache.set(key, _chunk_embedding_payload(chunk))
    return chunks


def embed_queries_cached(cache: CacheBackend, embedder: Any, queries: list[str], embedder_config: dict[str, Any]) -> list[Any]:
    results: list[Any | None] = [None] * len(queries)
    missing_queries: list[str] = []
    missing_indexes: list[int] = []
    missing_keys: list[str] = []
    for idx, query in enumerate(queries):
        key = query_embedding_key(embedder_config, query)
        cached = cache.get(key)
        if cached is None:
            missing_queries.append(query)
            missing_indexes.append(idx)
            missing_keys.append(key)
        else:
            results[idx] = _restore_json_types(cached)
    if missing_queries:
        embedded = embedder.embed_queries(missing_queries, embedder_config)
        for idx, key, vector in zip(missing_indexes, missing_keys, embedded):
            cache.set(key, vector)
            results[idx] = _restore_json_types(vector)
    return results
