from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

from ragbench_core.models.schema import Chunk

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
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        digest = key.rsplit(":", 1)[-1]
        return self.root / f"{digest}.json"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def set(self, key: str, value: Any) -> None:
        path = self._path(key)
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"))
        tmp_path.replace(path)


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
