from __future__ import annotations

from typing import Any

from opensearchpy import OpenSearch
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    HnswConfigDiff,
    PayloadSchemaType,
    PointStruct,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    VectorParams,
)

from ragbench_core.interfaces import BaseIndexer
from ragbench_core.models.schema import Chunk
from ragbench_core.settings import settings
from ragbench_core.utils.clients import close_client


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
    "dense_payload_indexed": {
        "payload_indexes": [
            {"field_name": "doc_id", "field_schema": PayloadSchemaType.KEYWORD},
            {"field_name": "chunk_id", "field_schema": PayloadSchemaType.KEYWORD},
            {"field_name": "metadata.source", "field_schema": PayloadSchemaType.KEYWORD},
            {"field_name": "metadata.char_length", "field_schema": PayloadSchemaType.INTEGER},
            {"field_name": "metadata.word_count", "field_schema": PayloadSchemaType.INTEGER},
        ]
    },
}


class QdrantIndexer(BaseIndexer):
    name = "qdrant"

    def index(self, chunks: list[Chunk], index_name: str, config: dict) -> dict:
        if not chunks:
            return {"points": 0}
        client = QdrantClient(url=settings.qdrant_url)
        try:
            dim = len(chunks[0].embedding or [])
            strategy_name = config.get("strategy", "dense_hnsw")
            strategy = self._strategy_config(strategy_name, config)
            client.recreate_collection(
                collection_name=index_name,
                vectors_config=self._vector_params(dim, strategy),
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
            }
        finally:
            close_client(client)

    def _strategy_config(self, strategy_name: str, config: dict) -> dict:
        if strategy_name not in QDRANT_DENSE_STRATEGIES:
            supported = ", ".join(sorted(QDRANT_DENSE_STRATEGIES))
            raise ValueError(f"Unsupported Qdrant dense strategy '{strategy_name}'. Supported: {supported}")
        strategy = dict(QDRANT_DENSE_STRATEGIES[strategy_name])
        for key in ("hnsw_config", "quantization_config", "on_disk", "payload_indexes"):
            if key in config:
                strategy[key] = config[key]
        return strategy

    def _vector_params(self, dim: int, strategy: dict) -> VectorParams:
        return VectorParams(
            size=dim,
            distance=Distance.COSINE,
            hnsw_config=self._hnsw_config(strategy.get("hnsw_config")),
            quantization_config=self._quantization_config(strategy.get("quantization_config")),
            on_disk=strategy.get("on_disk"),
        )

    def _hnsw_config(self, config: dict | None) -> HnswConfigDiff | None:
        if not config:
            return None
        return HnswConfigDiff(**config)

    def _quantization_config(self, config: dict | None) -> ScalarQuantization | None:
        if not config:
            return None
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

    def _create_payload_indexes(self, client: QdrantClient, index_name: str, strategy: dict) -> None:
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
        client = OpenSearch(settings.opensearch_url, verify_certs=False, ssl_show_warn=False)
        try:
            if client.indices.exists(index=index_name):
                client.indices.delete(index=index_name)
            client.indices.create(
                index=index_name,
                body={
                    "mappings": {
                        "properties": {
                            "text": {"type": "text"},
                            "doc_id": {"type": "keyword"},
                            "chunk_id": {"type": "keyword"},
                            "metadata": {"type": "object", "enabled": True},
                        }
                    }
                },
            )
            for c in chunks:
                client.index(
                    index=index_name,
                    id=c.chunk_id,
                    body={
                        "text": c.text,
                        "doc_id": c.doc_id,
                        "chunk_id": c.chunk_id,
                        "metadata": c.metadata,
                    },
                    refresh=False,
                )
            client.indices.refresh(index=index_name)
            return {"documents": len(chunks), "backend": "opensearch"}
        finally:
            close_client(client)
