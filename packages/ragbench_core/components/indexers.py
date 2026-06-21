from __future__ import annotations

from typing import Any

from ragbench_core.interfaces import BaseIndexer
from ragbench_core.models.schema import Chunk
from ragbench_core.utils.clients import get_opensearch_client, get_qdrant_client


QDRANT_DENSE_STRATEGIES: dict[str, dict[str, Any]] = {
    # Current baseline: use Qdrant defaults for in-memory HNSW.
    "dense_hnsw": {},
    # Higher graph degree / construction effort to measure recall-vs-latency trade-offs.
    "dense_hnsw_tuned": {
        "hnsw_config": {"m": 32, "ef_construct": 256, "full_scan_threshold": 5000}
    },
    # Disable HNSW graph creation so Qdrant falls back to exact nearest-neighbour scans.
    "dense_exact": {"hnsw_config": {"m": 0}},
    # Int8 scalar quantization to evaluate memory/cost trade-offs.
    "dense_quantized": {
        "quantization_config": {
            "type": "int8",
            "quantile": 0.99,
            "always_ram": True,
        }
    },
    # Keep vectors and the HNSW graph on disk for large-corpus behaviour.
    "dense_on_disk": {
        "on_disk": True,
        "hnsw_config": {"on_disk": True},
    },
    # Baseline HNSW plus payload indexes used by filtered retrieval experiments.
    "dense_hnsw_custom": {},
    "dense_payload_indexed": {
        "payload_indexes": [
            {"field_name": "doc_id", "field_schema": "keyword"},
            {"field_name": "chunk_id", "field_schema": "keyword"},
            {"field_name": "metadata.source", "field_schema": "keyword"},
            {"field_name": "metadata.char_length", "field_schema": "integer"},
            {"field_name": "metadata.word_count", "field_schema": "integer"},
        ]
    },
}

OPENSEARCH_BM25_STRATEGIES: dict[str, dict[str, Any]] = {
    # Plain text baseline using OpenSearch defaults for BM25 over the chunk text.
    "bm25_standard": {
        "fields": ["text"],
        "mappings": {
            "text": {"type": "text"},
        },
    },
    # Split each chunk into lightweight title / heading / body fields for boosted retrieval.
    "bm25_field_weighted": {
        "fields": ["title^3", "heading^2", "body"],
        "mappings": {
            "text": {"type": "text"},
            "title": {"type": "text"},
            "heading": {"type": "text"},
            "body": {"type": "text"},
        },
    },
    # Analyzer variant that normalizes case, removes stopwords, stems English words, and expands synonyms.
    "bm25_custom_analyzer": {
        "analyzer": "ragbench_bm25_custom",
        "synonyms": [
            "ai, artificial intelligence",
            "usa, united states, united states of america",
            "faq, frequently asked questions",
        ],
        "fields": ["title^3", "heading^2", "body", "text"],
        "mappings": {
            "text": {"type": "text", "analyzer": "ragbench_bm25_custom", "search_analyzer": "ragbench_bm25_custom"},
            "title": {"type": "text", "analyzer": "ragbench_bm25_custom", "search_analyzer": "ragbench_bm25_custom"},
            "heading": {"type": "text", "analyzer": "ragbench_bm25_custom", "search_analyzer": "ragbench_bm25_custom"},
            "body": {"type": "text", "analyzer": "ragbench_bm25_custom", "search_analyzer": "ragbench_bm25_custom"},
        },
    },
    # Shingle subfields keep the regular text query available while adding phrase-oriented matching.
    "bm25_phrase": {
        "analyzer": "ragbench_shingle",
        "fields": ["title^3", "heading^2", "body", "text", "title.phrase^4", "heading.phrase^3", "body.phrase^2", "text.phrase^2"],
        "phrase_fields": ["title.phrase^4", "heading.phrase^3", "body.phrase^2", "text.phrase^2"],
        "mappings": {
            "text": {"type": "text", "fields": {"phrase": {"type": "text", "analyzer": "ragbench_shingle"}}},
            "title": {"type": "text", "fields": {"phrase": {"type": "text", "analyzer": "ragbench_shingle"}}},
            "heading": {"type": "text", "fields": {"phrase": {"type": "text", "analyzer": "ragbench_shingle"}}},
            "body": {"type": "text", "fields": {"phrase": {"type": "text", "analyzer": "ragbench_shingle"}}},
        },
    },
}


class QdrantIndexer(BaseIndexer):
    name = "qdrant"

    def index(self, chunks: list[Chunk], index_name: str, config: dict) -> dict:
        if not chunks:
            return {"points": 0}
        from qdrant_client.http.models import PointStruct
        from ragbench_core.settings import settings

        client = get_qdrant_client(settings.qdrant_url)
        dim = len(chunks[0].embedding or [])
        strategy_name = config.get("strategy", "dense_hnsw")
        strategy = self._strategy_config(strategy_name, config)
        client.recreate_collection(
            collection_name=index_name,
            vectors_config=self._vector_params(dim, strategy),
            **self._collection_options(strategy),
        )
        points = []
        for idx, c in enumerate(chunks):
            points.append(
                PointStruct(
                    id=idx,
                    vector=c.embedding,
                    payload={
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "text": c.text,
                        "metadata": c.metadata,
                    },
                )
            )
        client.upsert(collection_name=index_name, points=points)
        self._create_payload_indexes(client, index_name, strategy)
        return {
            "points": len(points),
            "dim": dim,
            "backend": "qdrant",
            "strategy": strategy_name,
            "parameters": self._persisted_parameters(strategy),
        }


    def _strategy_config(self, strategy_name: str, config: dict) -> dict:
        if strategy_name not in QDRANT_DENSE_STRATEGIES:
            supported = ", ".join(sorted(QDRANT_DENSE_STRATEGIES))
            raise ValueError(f"Unsupported Qdrant dense strategy '{strategy_name}'. Supported: {supported}")
        strategy = dict(QDRANT_DENSE_STRATEGIES[strategy_name])
        if "hnsw_config" in strategy:
            strategy["hnsw_config"] = dict(strategy["hnsw_config"])
        for key in ("hnsw_config", "quantization_config", "on_disk", "on_disk_payload", "payload_indexes"):
            if key in config:
                strategy[key] = config[key]
        hnsw_config = dict(strategy.get("hnsw_config") or {})
        if "hnsw_m" in config:
            hnsw_config["m"] = config["hnsw_m"]
        elif "m" in config:
            hnsw_config["m"] = config["m"]
        if "ef_construct" in config:
            hnsw_config["ef_construct"] = config["ef_construct"]
        if "hnsw_on_disk" in config:
            hnsw_config["on_disk"] = config["hnsw_on_disk"]
        if hnsw_config:
            strategy["hnsw_config"] = hnsw_config
        for key in ("search_ef", "hnsw_ef", "payload_on_disk"):
            if key in config:
                strategy[key] = config[key]
        return strategy

    def _vector_params(self, dim: int, strategy: dict) -> Any:
        from qdrant_client.http.models import Distance, VectorParams
        return VectorParams(
            size=dim,
            distance=Distance.COSINE,
            hnsw_config=self._hnsw_config(strategy.get("hnsw_config")),
            quantization_config=self._quantization_config(strategy.get("quantization_config")),
            on_disk=strategy.get("on_disk"),
        )

    def _collection_options(self, strategy: dict) -> dict[str, Any]:
        options = {}
        on_disk_payload = strategy.get("on_disk_payload", strategy.get("payload_on_disk"))
        if on_disk_payload is not None:
            options["on_disk_payload"] = on_disk_payload
        return options

    def _persisted_parameters(self, strategy: dict) -> dict[str, Any]:
        return {
            key: strategy[key]
            for key in ("hnsw_config", "quantization_config", "on_disk", "on_disk_payload", "payload_on_disk", "search_ef", "hnsw_ef", "payload_indexes")
            if key in strategy
        }

    def _hnsw_config(self, config: dict | None) -> Any | None:
        if not config:
            return None
        from qdrant_client.http.models import HnswConfigDiff

        return HnswConfigDiff(**config)

    def _quantization_config(self, config: dict | None) -> Any | None:
        if not config:
            return None
        from qdrant_client.http.models import ScalarQuantization, ScalarQuantizationConfig, ScalarType

        quantization_type = config.get("type", ScalarType.INT8)
        if quantization_type == "int8":
            quantization_type = ScalarType.INT8
        return ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type=quantization_type,
                quantile=config.get("quantile"),
                always_ram=config.get("always_ram"),
            )
        )

    def _create_payload_indexes(self, client: Any, index_name: str, strategy: dict) -> None:
        for payload_index in strategy.get("payload_indexes", []):
            client.create_payload_index(
                collection_name=index_name,
                field_name=payload_index["field_name"],
                field_schema=payload_index["field_schema"],
                wait=True,
            )


class OpenSearchIndexer(BaseIndexer):
    name = "opensearch"

    def index(self, chunks: list[Chunk], index_name: str, config: dict) -> dict:
        from ragbench_core.settings import settings

        client = get_opensearch_client(settings.opensearch_url)
        strategy_name = config.get("strategy", "bm25_standard")
        strategy = self._strategy_config(strategy_name, config)
        if client.indices.exists(index=index_name):
            client.indices.delete(index=index_name)
        client.indices.create(index=index_name, body=self._index_body(strategy))
        for c in chunks:
            client.index(
                index=index_name,
                id=c.chunk_id,
                body=self._document_body(c, strategy),
                refresh=False,
            )
        client.indices.refresh(index=index_name)
        return {"documents": len(chunks), "backend": "opensearch", "strategy": strategy_name}


    def _strategy_config(self, strategy_name: str, config: dict) -> dict:
        if strategy_name not in OPENSEARCH_BM25_STRATEGIES:
            supported = ", ".join(sorted(OPENSEARCH_BM25_STRATEGIES))
            raise ValueError(f"Unsupported OpenSearch BM25 strategy '{strategy_name}'. Supported: {supported}")
        strategy = dict(OPENSEARCH_BM25_STRATEGIES[strategy_name])
        if "synonyms" in config:
            strategy["synonyms"] = config["synonyms"]
        return strategy

    def _index_body(self, strategy: dict) -> dict:
        properties = {
            **strategy["mappings"],
            "doc_id": {"type": "keyword"},
            "chunk_id": {"type": "keyword"},
            "metadata": {"type": "object", "enabled": True},
        }
        body: dict[str, Any] = {"mappings": {"properties": properties}}
        analyzer = strategy.get("analyzer")
        if analyzer == "ragbench_bm25_custom":
            body["settings"] = {
                "analysis": {
                    "filter": {
                        "ragbench_synonyms": {
                            "type": "synonym",
                            "synonyms": strategy["synonyms"],
                        },
                        "ragbench_english_stop": {"type": "stop", "stopwords": "_english_"},
                        "ragbench_english_stemmer": {"type": "stemmer", "language": "english"},
                    },
                    "analyzer": {
                        "ragbench_bm25_custom": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "ragbench_english_stop",
                                "ragbench_english_stemmer",
                                "ragbench_synonyms",
                            ],
                        }
                    },
                }
            }
        elif analyzer == "ragbench_shingle":
            body["settings"] = {
                "analysis": {
                    "filter": {
                        "ragbench_shingle_filter": {
                            "type": "shingle",
                            "min_shingle_size": 2,
                            "max_shingle_size": 3,
                            "output_unigrams": True,
                        }
                    },
                    "analyzer": {
                        "ragbench_shingle": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "ragbench_shingle_filter"],
                        }
                    },
                }
            }
        return body

    def _document_body(self, chunk: Chunk, strategy: dict) -> dict:
        metadata = chunk.metadata or {}
        body = {
            "text": chunk.text,
            "doc_id": chunk.doc_id,
            "chunk_id": chunk.chunk_id,
            "metadata": metadata,
        }
        if any(field in strategy["mappings"] for field in ("title", "heading", "body")):
            body.update(
                {
                    "title": metadata.get("title") or metadata.get("file_name") or chunk.doc_id,
                    "heading": metadata.get("section_heading") or metadata.get("heading") or "",
                    "body": chunk.text,
                }
            )
        return body


OPENSEARCH_NEURAL_SPARSE_STRATEGIES: dict[str, dict[str, Any]] = {
    "neural_sparse_rank_features": {
        "field": "sparse_embedding",
        "mappings": {
            "text": {"type": "text"},
            "sparse_embedding": {"type": "rank_features"},
        },
    }
}


class OpenSearchNeuralSparseIndexer(BaseIndexer):
    """Index precomputed neural sparse vectors into OpenSearch rank_features fields."""

    name = "opensearch_neural_sparse"

    def index(self, chunks: list[Chunk], index_name: str, config: dict) -> dict:
        from ragbench_core.settings import settings

        client = get_opensearch_client(settings.opensearch_url)
        strategy_name = config.get("strategy", "neural_sparse_rank_features")
        strategy = self._strategy_config(strategy_name, config)
        if client.indices.exists(index=index_name):
            client.indices.delete(index=index_name)
        client.indices.create(index=index_name, body=self._index_body(strategy))
        indexed = 0
        for c in chunks:
            if not c.sparse_embedding:
                continue
            client.index(index=index_name, id=c.chunk_id, body=self._document_body(c, strategy), refresh=False)
            indexed += 1
        client.indices.refresh(index=index_name)
        return {"documents": indexed, "backend": "opensearch", "strategy": strategy_name, "sparse": True}


    def _strategy_config(self, strategy_name: str, config: dict) -> dict:
        if strategy_name not in OPENSEARCH_NEURAL_SPARSE_STRATEGIES:
            supported = ", ".join(sorted(OPENSEARCH_NEURAL_SPARSE_STRATEGIES))
            raise ValueError(f"Unsupported OpenSearch neural sparse strategy '{strategy_name}'. Supported: {supported}")
        strategy = dict(OPENSEARCH_NEURAL_SPARSE_STRATEGIES[strategy_name])
        if "field" in config:
            strategy["field"] = config["field"]
        return strategy

    def _index_body(self, strategy: dict) -> dict:
        return {
            "mappings": {
                "properties": {
                    **strategy["mappings"],
                    "doc_id": {"type": "keyword"},
                    "chunk_id": {"type": "keyword"},
                    "metadata": {"type": "object", "enabled": True},
                }
            }
        }

    def _document_body(self, chunk: Chunk, strategy: dict) -> dict:
        return {
            "text": chunk.text,
            "doc_id": chunk.doc_id,
            "chunk_id": chunk.chunk_id,
            "metadata": chunk.metadata or {},
            strategy["field"]: {str(k): float(v) for k, v in (chunk.sparse_embedding or {}).items()},
        }


class QdrantSparseIndexer(BaseIndexer):
    """Index precomputed neural sparse vectors into a Qdrant sparse-vector collection."""

    name = "qdrant_sparse"

    def index(self, chunks: list[Chunk], index_name: str, config: dict) -> dict:
        from qdrant_client.http.models import PointStruct, SparseVector, SparseVectorParams, SparseIndexParams
        from ragbench_core.settings import settings

        collection_name = config.get("collection_name", f"{index_name}_neural_sparse")
        vector_name = config.get("vector_name", "neural_sparse")
        client = get_qdrant_client(settings.qdrant_url)
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config={},
            sparse_vectors_config={
                vector_name: SparseVectorParams(index=SparseIndexParams(on_disk=bool(config.get("on_disk", False))))
            },
        )
        points = []
        for idx, c in enumerate(chunks):
            if not c.sparse_embedding:
                continue
            sparse = dict(sorted(c.sparse_embedding.items()))
            points.append(
                PointStruct(
                    id=idx,
                    vector={vector_name: SparseVector(indices=list(sparse.keys()), values=list(sparse.values()))},
                    payload={"chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.text, "metadata": c.metadata},
                )
            )
        if points:
            client.upsert(collection_name=collection_name, points=points)
        return {"points": len(points), "backend": "qdrant", "strategy": "neural_sparse", "collection": collection_name}
