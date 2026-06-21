from ragbench_core.engine.cache import (
    FileEmbeddingCache,
    document_embedding_key,
    embed_chunks_cached,
    embed_queries_cached,
    query_embedding_key,
)
from ragbench_core.models.schema import Chunk


class CountingEmbedder:
    def __init__(self):
        self.chunk_calls = 0
        self.query_calls = 0

    def embed_chunks(self, chunks, config):
        self.chunk_calls += 1
        for chunk in chunks:
            chunk.embedding = [float(len(chunk.text)), float(self.chunk_calls)]
        return chunks

    def embed_queries(self, queries, config):
        self.query_calls += 1
        return [[float(len(query)), float(self.query_calls)] for query in queries]


def test_file_cache_hits_and_misses_for_chunks(tmp_path):
    config = {"provider": "fake", "model": "m", "normalize": True, "dimensions": 2, "document_prefix": "passage: "}
    cache = FileEmbeddingCache(tmp_path)
    embedder = CountingEmbedder()
    chunks = [Chunk("c1", "d1", "alpha", {}), Chunk("c2", "d1", "beta", {})]

    first = embed_chunks_cached(cache, embedder, chunks, config)
    assert [chunk.embedding for chunk in first] == [[5.0, 1.0], [4.0, 1.0]]
    assert embedder.chunk_calls == 1

    second_chunks = [Chunk("c1", "d1", "alpha", {}), Chunk("c2", "d1", "beta changed", {})]
    second = embed_chunks_cached(cache, embedder, second_chunks, config)

    assert [chunk.embedding for chunk in second] == [[5.0, 1.0], [12.0, 2.0]]
    assert embedder.chunk_calls == 2


def test_file_cache_hits_and_misses_for_queries(tmp_path):
    config = {"provider": "fake", "model": "m", "normalize_embeddings": False, "query_prefix": "query: "}
    cache = FileEmbeddingCache(tmp_path)
    embedder = CountingEmbedder()

    assert embed_queries_cached(cache, embedder, ["what", "why"], config) == [[4.0, 1.0], [3.0, 1.0]]
    assert embedder.query_calls == 1

    assert embed_queries_cached(cache, embedder, ["what", "how"], config) == [[4.0, 1.0], [3.0, 2.0]]
    assert embedder.query_calls == 2


def test_cache_keys_include_embedder_config_chunk_id_and_text_hash():
    base = {"provider": "fake", "model": "m", "normalize": True, "dimensions": 2}
    changed_config = {**base, "normalize": False}
    chunk = Chunk("c1", "d1", "alpha", {})

    assert document_embedding_key(base, chunk) != document_embedding_key(changed_config, chunk)
    assert document_embedding_key(base, chunk) != document_embedding_key(base, Chunk("c2", "d1", "alpha", {}))
    assert document_embedding_key(base, chunk) != document_embedding_key(base, Chunk("c1", "d1", "beta", {}))
    assert query_embedding_key(base, "what") != query_embedding_key(base, "why")


def test_file_cache_preserves_sparse_embedding_key_types(tmp_path):
    config = {"provider": "fake", "model": "sparse", "sparse_top_k": 2}
    cache = FileEmbeddingCache(tmp_path)
    chunk = Chunk("c1", "d1", "alpha", {})
    key = document_embedding_key(config, chunk)
    cache.set(key, {"embedding": None, "sparse_embedding": {1: 0.5, 2: 0.25}})

    class FailingEmbedder:
        def embed_chunks(self, chunks, config):  # pragma: no cover - should be a cache hit
            raise AssertionError("expected cache hit")

    cached_chunk = embed_chunks_cached(cache, FailingEmbedder(), [Chunk("c1", "d1", "alpha", {})], config)[0]
    assert cached_chunk.sparse_embedding == {1: 0.5, 2: 0.25}
    assert all(isinstance(key, int) for key in cached_chunk.sparse_embedding)

    query_key = query_embedding_key(config, "what")
    cache.set(query_key, {3: 1.0})

    class FailingQueryEmbedder:
        def embed_queries(self, queries, config):  # pragma: no cover - should be a cache hit
            raise AssertionError("expected cache hit")

    assert embed_queries_cached(cache, FailingQueryEmbedder(), ["what"], config) == [{3: 1.0}]
