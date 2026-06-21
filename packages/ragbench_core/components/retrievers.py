from __future__ import annotations
from collections import defaultdict
from typing import Any

from ragbench_core.utils.clients import close_client
from ragbench_core.interfaces import BaseRetriever
from ragbench_core.models.schema import Chunk, RetrievalResult


class DenseRetriever(BaseRetriever):
    name = "dense"

    def retrieve(
        self,
        question: str,
        query_embedding: list[float] | None,
        index_name: str,
        config: dict,
        top_k: int,
    ) -> list[RetrievalResult]:
        from qdrant_client import QdrantClient
        from ragbench_core.settings import settings

        client = QdrantClient(url=settings.qdrant_url)
        try:
            hits = client.search(
                collection_name=index_name, query_vector=(query_embedding.get("dense") if isinstance(query_embedding, dict) and "dense" in query_embedding else query_embedding), limit=top_k
            )
            return self._results_from_hits(hits)
        finally:
            close_client(client)

    def _results_from_hits(self, hits) -> list[RetrievalResult]:
        out = []
        for i, h in enumerate(hits):
            p = h.payload
            out.append(
                RetrievalResult(
                    Chunk(p["chunk_id"], p["doc_id"], p["text"], p.get("metadata", {})),
                    float(h.score),
                    i + 1,
                    self.name,
                )
            )
        return out


class DenseFilteredRetriever(DenseRetriever):
    name = "dense_filtered"

    def retrieve(
        self,
        question: str,
        query_embedding: list[float] | None,
        index_name: str,
        config: dict,
        top_k: int,
    ) -> list[RetrievalResult]:
        from qdrant_client import QdrantClient
        from ragbench_core.settings import settings

        client = QdrantClient(url=settings.qdrant_url)
        try:
            hits = client.search(
                collection_name=index_name,
                query_vector=(query_embedding.get("dense") if isinstance(query_embedding, dict) and "dense" in query_embedding else query_embedding),
                query_filter=self._build_query_filter(config),
                limit=top_k,
            )
            return self._results_from_hits(hits)
        finally:
            close_client(client)

    def _build_query_filter(self, config: dict[str, Any]) -> dict[str, Any] | None:
        raw_filter = config.get("payload_filter")
        if raw_filter:
            return raw_filter

        metadata_filters = config.get("metadata_filters") or config.get("metadata_filter") or {}
        must = self._conditions_from_mapping(metadata_filters)
        must.extend(self._doc_id_conditions(config))
        if not must:
            return None
        return {"must": must}

    def _conditions_from_mapping(self, filters: dict[str, Any]) -> list[dict[str, Any]]:
        conditions = []
        for field_name, value in filters.items():
            if value is None:
                continue
            conditions.append(self._field_condition(self._payload_key(field_name), value))
        return conditions

    def _doc_id_conditions(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        conditions = []
        if config.get("doc_id") is not None:
            conditions.append(self._field_condition("doc_id", config["doc_id"]))
        if config.get("doc_ids"):
            conditions.append(self._field_condition("doc_id", config["doc_ids"]))
        return conditions

    def _payload_key(self, field_name: str) -> str:
        if field_name in {"doc_id", "chunk_id"} or field_name.startswith("metadata."):
            return field_name
        return f"metadata.{field_name}"

    def _field_condition(self, field_name: str, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            range_keys = {"gt", "gte", "lt", "lte"}
            if range_keys.intersection(value):
                return {
                    "key": field_name,
                    "range": {key: value[key] for key in range_keys if key in value},
                }
            if "any" in value:
                return {"key": field_name, "match": {"any": value["any"]}}
            if "value" in value:
                return {"key": field_name, "match": {"value": value["value"]}}
        if isinstance(value, list | tuple | set):
            return {"key": field_name, "match": {"any": list(value)}}
        return {"key": field_name, "match": {"value": value}}


class ParentDocumentRetriever(BaseRetriever):
    name = "parent_document"

    def retrieve(
        self,
        question: str,
        query_embedding: list[float] | None,
        index_name: str,
        config: dict,
        top_k: int,
    ) -> list[RetrievalResult]:
        child_top_k = self._child_top_k(config, top_k)
        child_results = self._child_retriever(config).retrieve(
            question,
            query_embedding,
            index_name,
            config,
            child_top_k,
        )
        return self._expand_parent_context(child_results, top_k)

    def _child_retriever(self, config: dict[str, Any]) -> BaseRetriever:
        base_retriever = config.get("base_retriever", "dense")
        if base_retriever == "dense":
            return DenseRetriever()
        if base_retriever == "dense_filtered":
            return DenseFilteredRetriever()
        supported = "dense, dense_filtered"
        raise ValueError(
            f"Unsupported parent_document base_retriever '{base_retriever}'. Supported: {supported}"
        )

    def _child_top_k(self, config: dict[str, Any], top_k: int) -> int:
        multiplier = int(config.get("child_top_k_multiplier", 4))
        configured_child_top_k = config.get("child_top_k")
        if configured_child_top_k is not None:
            return max(top_k, int(configured_child_top_k))
        return max(top_k, top_k * max(1, multiplier))

    def _expand_parent_context(
        self,
        child_results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        expanded: list[RetrievalResult] = []
        seen: set[str] = set()
        for child_result in child_results:
            parent_result = self._parent_result_from_child(child_result)
            dedupe_key = parent_result.chunk.chunk_id
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            expanded.append(parent_result)
            if len(expanded) >= top_k:
                break

        for rank, result in enumerate(expanded, start=1):
            result.rank = rank
        return expanded

    def _parent_result_from_child(self, child_result: RetrievalResult) -> RetrievalResult:
        child = child_result.chunk
        metadata = dict(child.metadata or {})
        parent_text = metadata.get("parent_text")
        parent_chunk_id = metadata.get("parent_chunk_id")
        if parent_text and parent_chunk_id:
            metadata.update(
                {
                    "child_chunk_id": child.chunk_id,
                    "child_text": child.text,
                    "retrieved_chunk_id": child.chunk_id,
                    "retrieved_chunk_text": child.text,
                    "chunk_role": "parent",
                    "expanded_from_retriever": child_result.retriever,
                }
            )
            chunk = Chunk(str(parent_chunk_id), child.doc_id, str(parent_text), metadata)
        else:
            metadata.update(
                {
                    "retrieved_chunk_id": child.chunk_id,
                    "retrieved_chunk_text": child.text,
                    "chunk_role": metadata.get("chunk_role", "child"),
                    "parent_expansion_missing": True,
                    "expanded_from_retriever": child_result.retriever,
                }
            )
            chunk = Chunk(child.chunk_id, child.doc_id, child.text, metadata, child.embedding)
        return RetrievalResult(chunk, child_result.score, child_result.rank, self.name)


class BM25Retriever(BaseRetriever):
    name = "bm25"

    def retrieve(
        self,
        question: str,
        query_embedding: list[float] | None,
        index_name: str,
        config: dict,
        top_k: int,
    ) -> list[RetrievalResult]:
        from opensearchpy import OpenSearch
        from ragbench_core.settings import settings

        client = OpenSearch(settings.opensearch_url, verify_certs=False, ssl_show_warn=False)
        try:
            fields, phrase_fields = self._search_fields(client, index_name)
            should = [
                {
                    "multi_match": {
                        "query": question,
                        "fields": fields,
                        "type": "best_fields",
                    }
                }
            ]
            if phrase_fields:
                should.append(
                    {
                        "multi_match": {
                            "query": question,
                            "fields": phrase_fields,
                            "type": "phrase",
                        }
                    }
                )
            res = client.search(
                index=index_name,
                body={
                    "query": {
                        "bool": {
                            "should": should,
                            "minimum_should_match": 1,
                        }
                    },
                    "size": top_k,
                },
            )
            out = []
            for i, h in enumerate(res["hits"]["hits"]):
                s = h["_source"]
                out.append(
                    RetrievalResult(
                        Chunk(s["chunk_id"], s["doc_id"], s["text"], s.get("metadata", {})),
                        float(h["_score"]),
                        i + 1,
                        self.name,
                    )
                )
            return out
        finally:
            close_client(client)

    def _search_fields(self, client: Any, index_name: str) -> tuple[list[str], list[str]]:
        mappings = client.indices.get_mapping(index=index_name)
        properties = next(iter(mappings.values()))["mappings"].get("properties", {})
        boosts = {"title": "^3", "heading": "^2", "body": "", "text": ""}
        phrase_boosts = {"title": "^4", "heading": "^3", "body": "^2", "text": "^2"}
        fields = [f"{field}{boost}" for field, boost in boosts.items() if field in properties]
        phrase_fields = [
            f"{field}.phrase{phrase_boosts[field]}"
            for field in phrase_boosts
            if "phrase" in properties.get(field, {}).get("fields", {})
        ]
        return fields or ["text"], phrase_fields


def reciprocal_rank_fusion(
    result_sets: list[list[RetrievalResult]],
    *,
    rrf_k: int = 60,
    top_k: int,
    retriever_name: str,
) -> list[RetrievalResult]:
    scores = defaultdict(float)
    chunks = {}
    source_retrievers = defaultdict(list)
    for result_set in result_sets:
        for result in result_set:
            chunk_id = result.chunk.chunk_id
            scores[chunk_id] += 1.0 / (rrf_k + result.rank)
            chunks.setdefault(chunk_id, result.chunk)
            if result.retriever not in source_retrievers[chunk_id]:
                source_retrievers[chunk_id].append(result.retriever)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    fused = []
    for rank, (chunk_id, score) in enumerate(ranked, start=1):
        chunk = chunks[chunk_id]
        metadata = dict(chunk.metadata or {})
        metadata["fusion_source_retrievers"] = source_retrievers[chunk_id]
        fused.append(
            RetrievalResult(
                Chunk(chunk.chunk_id, chunk.doc_id, chunk.text, metadata, chunk.embedding),
                score,
                rank,
                retriever_name,
            )
        )
    return fused


class HybridRetriever(BaseRetriever):
    name = "hybrid"
    supported_sources = {"dense", "bm25", "neural_sparse"}

    def retrieve(
        self,
        question: str,
        query_embedding: list[float] | dict[str, Any] | None,
        index_name: str,
        config: dict,
        top_k: int,
    ) -> list[RetrievalResult]:
        result_sets: list[list[RetrievalResult]] = []
        for source in self._sources(config):
            if source == "dense":
                result_sets.append(DenseRetriever().retrieve(question, query_embedding, index_name, config, top_k))
            elif source == "bm25":
                result_sets.append(BM25Retriever().retrieve(question, query_embedding, index_name, config, top_k))
            elif source == "neural_sparse":
                result_sets.append(
                    NeuralSparseRetriever().retrieve(
                        question,
                        query_embedding,
                        index_name,
                        config.get("neural_sparse", config),
                        top_k,
                    )
                )
        return reciprocal_rank_fusion(
            result_sets,
            rrf_k=int(config.get("rrf_k", 60)),
            top_k=top_k,
            retriever_name=self.name,
        )

    def _sources(self, config: dict[str, Any]) -> list[str]:
        sources = config.get("sources", config.get("retrievers", ["dense", "bm25"]))
        if isinstance(sources, str):
            sources = [sources]
        source_names = [str(source) for source in sources]
        unsupported = sorted(set(source_names) - self.supported_sources)
        if unsupported:
            supported = ", ".join(sorted(self.supported_sources))
            raise ValueError(
                f"Unsupported hybrid source(s) {unsupported}. Supported: {supported}"
            )
        if not source_names:
            raise ValueError("hybrid requires at least one source retriever.")
        return source_names


class MultiQueryFusionRetriever(BaseRetriever):
    name = "multi_query_fusion"
    supported_base_retrievers = {"dense", "dense_filtered", "bm25", "hybrid", "neural_sparse", "weighted_hybrid"}
    supported_fusions = {"rrf"}

    def retrieve(
        self,
        question: str,
        query_embedding: list[float] | None,
        index_name: str,
        config: dict,
        top_k: int,
    ) -> list[RetrievalResult]:
        base_retriever_name = self._base_retriever_name(config)
        results = self._base_retriever(base_retriever_name).retrieve(
            question, query_embedding, index_name, config, top_k
        )
        return [
            RetrievalResult(
                result.chunk,
                result.score,
                result.rank,
                f"{self.name}:{base_retriever_name}",
            )
            for result in results
        ]

    def fuse(
        self,
        result_sets: list[list[RetrievalResult]],
        config: dict[str, Any],
        top_k: int,
    ) -> list[RetrievalResult]:
        fusion = config.get("fusion", "rrf")
        if fusion not in self.supported_fusions:
            supported = ", ".join(sorted(self.supported_fusions))
            raise ValueError(
                f"Unsupported multi_query_fusion fusion '{fusion}'. Supported: {supported}"
            )
        base_retriever_name = self._base_retriever_name(config)
        return reciprocal_rank_fusion(
            result_sets,
            rrf_k=int(config.get("rrf_k", 60)),
            top_k=top_k,
            retriever_name=f"{self.name}:{base_retriever_name}:{fusion}",
        )

    def _base_retriever_name(self, config: dict[str, Any]) -> str:
        base_retriever_name = config.get("base_retriever", "dense")
        if base_retriever_name not in self.supported_base_retrievers:
            supported = ", ".join(sorted(self.supported_base_retrievers))
            raise ValueError(
                f"Unsupported multi_query_fusion base_retriever '{base_retriever_name}'. "
                f"Supported: {supported}"
            )
        return base_retriever_name

    def _base_retriever(self, base_retriever_name: str) -> BaseRetriever:
        if base_retriever_name == "dense":
            return DenseRetriever()
        if base_retriever_name == "dense_filtered":
            return DenseFilteredRetriever()
        if base_retriever_name == "bm25":
            return BM25Retriever()
        if base_retriever_name == "hybrid":
            return HybridRetriever()
        if base_retriever_name == "neural_sparse":
            return NeuralSparseRetriever()
        if base_retriever_name == "weighted_hybrid":
            return WeightedHybridRetriever()
        supported = ", ".join(sorted(self.supported_base_retrievers))
        raise ValueError(
            f"Unsupported multi_query_fusion base_retriever '{base_retriever_name}'. Supported: {supported}"
        )


class AdaptiveHybridRetriever(BaseRetriever):
    name = "adaptive_hybrid"
    supported_modes = {"dense", "bm25", "hybrid", "neural_sparse", "weighted_hybrid"}

    def retrieve(
        self,
        question: str,
        query_embedding: list[float] | None,
        index_name: str,
        config: dict,
        top_k: int,
    ) -> list[RetrievalResult]:
        mode = self._select_mode(question, config)
        results = self._retriever_for_mode(mode).retrieve(
            question, query_embedding, index_name, config, top_k
        )
        return [
            RetrievalResult(result.chunk, result.score, result.rank, f"{self.name}:{mode}")
            for result in results
        ]

    def _select_mode(self, question: str, config: dict[str, Any]) -> str:
        token_count = len(self._query_terms(question))
        short_query_max_terms = int(config.get("short_query_max_terms", 4))
        long_query_min_terms = int(config.get("long_query_min_terms", 12))

        if token_count <= short_query_max_terms:
            return self._validated_mode(config.get("short_query_mode", "bm25"), "short_query_mode")
        if token_count >= long_query_min_terms:
            return self._validated_mode(config.get("long_query_mode", "hybrid"), "long_query_mode")
        return self._validated_mode(config.get("default_mode", "dense"), "default_mode")

    def _query_terms(self, question: str) -> list[str]:
        import re

        return re.findall(r"[\w']+", question.lower())

    def _validated_mode(self, mode: str, config_key: str) -> str:
        if mode not in self.supported_modes:
            supported = ", ".join(sorted(self.supported_modes))
            raise ValueError(
                f"Unsupported adaptive_hybrid {config_key} '{mode}'. Supported: {supported}"
            )
        return mode

    def _retriever_for_mode(self, mode: str) -> BaseRetriever:
        if mode == "dense":
            return DenseRetriever()
        if mode == "bm25":
            return BM25Retriever()
        if mode == "hybrid":
            return HybridRetriever()
        if mode == "neural_sparse":
            return NeuralSparseRetriever()
        if mode == "weighted_hybrid":
            return WeightedHybridRetriever()
        return self._raise_unsupported_mode(mode)

    def _raise_unsupported_mode(self, mode: str) -> BaseRetriever:
        supported = ", ".join(sorted(self.supported_modes))
        raise ValueError(f"Unsupported adaptive_hybrid mode '{mode}'. Supported: {supported}")


class NeuralSparseRetriever(BaseRetriever):
    """Retrieve with precomputed neural sparse query vectors."""

    name = "neural_sparse"

    def retrieve(
        self,
        question: str,
        query_embedding: list[float] | dict[int, float] | None,
        index_name: str,
        config: dict,
        top_k: int,
    ) -> list[RetrievalResult]:
        if isinstance(query_embedding, dict) and "sparse" in query_embedding:
            query_embedding = query_embedding["sparse"]
        if not isinstance(query_embedding, dict):
            raise ValueError("neural_sparse retrieval requires a sparse query embedding dictionary.")
        backend = config.get("backend", "opensearch")
        if backend == "opensearch":
            return self._retrieve_opensearch(query_embedding, index_name, config, top_k)
        if backend == "qdrant":
            return self._retrieve_qdrant(query_embedding, index_name, config, top_k)
        raise ValueError("Unsupported neural_sparse backend '%s'. Supported: opensearch, qdrant" % backend)

    def _retrieve_opensearch(self, query_embedding: dict[int, float], index_name: str, config: dict, top_k: int) -> list[RetrievalResult]:
        from opensearchpy import OpenSearch
        from ragbench_core.settings import settings

        client = OpenSearch(settings.opensearch_url, verify_certs=False, ssl_show_warn=False)
        try:
            field = config.get("field", "sparse_embedding")
            terms = sorted(query_embedding.items(), key=lambda item: item[1], reverse=True)[: int(config.get("query_terms", 256))]
            should = [
                {"rank_feature": {"field": f"{field}.{token_id}", "boost": float(weight)}}
                for token_id, weight in terms
                if weight > 0
            ]
            res = client.search(index=index_name, body={"query": {"bool": {"should": should or [{"match_none": {}}]}}, "size": top_k})
            return self._opensearch_results(res)
        finally:
            close_client(client)

    def _retrieve_qdrant(self, query_embedding: dict[int, float], index_name: str, config: dict, top_k: int) -> list[RetrievalResult]:
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import NamedSparseVector, SparseVector
        from ragbench_core.settings import settings

        collection_name = config.get("collection_name", f"{index_name}_neural_sparse")
        vector_name = config.get("vector_name", "neural_sparse")
        sparse = dict(sorted(query_embedding.items()))
        client = QdrantClient(url=settings.qdrant_url)
        try:
            hits = client.search(
                collection_name=collection_name,
                query_vector=NamedSparseVector(name=vector_name, vector=SparseVector(indices=list(sparse.keys()), values=list(sparse.values()))),
                limit=top_k,
            )
            return DenseRetriever()._results_from_hits(hits)
        finally:
            close_client(client)

    def _opensearch_results(self, res: dict[str, Any]) -> list[RetrievalResult]:
        out = []
        for i, h in enumerate(res["hits"]["hits"]):
            s = h["_source"]
            out.append(RetrievalResult(Chunk(s["chunk_id"], s["doc_id"], s["text"], s.get("metadata", {})), float(h["_score"]), i + 1, self.name))
        return out


def weighted_rank_fusion(
    named_result_sets: list[tuple[str, list[RetrievalResult]]],
    *,
    weights: dict[str, float],
    rrf_k: int = 60,
    top_k: int,
    retriever_name: str,
) -> list[RetrievalResult]:
    scores = defaultdict(float)
    chunks = {}
    source_retrievers = defaultdict(list)
    for name, result_set in named_result_sets:
        weight = float(weights.get(name, 1.0))
        if weight == 0:
            continue
        for result in result_set:
            chunk_id = result.chunk.chunk_id
            scores[chunk_id] += weight / (rrf_k + result.rank)
            chunks.setdefault(chunk_id, result.chunk)
            if result.retriever not in source_retrievers[chunk_id]:
                source_retrievers[chunk_id].append(result.retriever)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    fused = []
    for rank, (chunk_id, score) in enumerate(ranked, start=1):
        chunk = chunks[chunk_id]
        metadata = dict(chunk.metadata or {})
        metadata["fusion_source_retrievers"] = source_retrievers[chunk_id]
        fused.append(RetrievalResult(Chunk(chunk.chunk_id, chunk.doc_id, chunk.text, metadata, chunk.embedding, chunk.sparse_embedding), score, rank, retriever_name))
    return fused


class WeightedHybridRetriever(BaseRetriever):
    """Fuse dense, BM25, and neural sparse retrieval with weighted RRF."""

    name = "weighted_hybrid"

    def retrieve(
        self,
        question: str,
        query_embedding: list[float] | dict[int, float] | None,
        index_name: str,
        config: dict,
        top_k: int,
    ) -> list[RetrievalResult]:
        weights = config.get("weights", {"dense": 1.0, "bm25": 1.0, "neural_sparse": 1.0})
        result_sets: list[tuple[str, list[RetrievalResult]]] = []
        if weights.get("dense", 1.0) and (not isinstance(query_embedding, dict) or "dense" in query_embedding):
            result_sets.append(("dense", DenseRetriever().retrieve(question, query_embedding, index_name, config, top_k)))
        if weights.get("bm25", 1.0):
            result_sets.append(("bm25", BM25Retriever().retrieve(question, query_embedding, index_name, config, top_k)))
        if weights.get("neural_sparse", 1.0) and isinstance(query_embedding, dict):
            result_sets.append(("neural_sparse", NeuralSparseRetriever().retrieve(question, query_embedding, index_name, config.get("neural_sparse", config), top_k)))
        return weighted_rank_fusion(result_sets, weights=weights, rrf_k=int(config.get("rrf_k", 60)), top_k=top_k, retriever_name=self.name)
