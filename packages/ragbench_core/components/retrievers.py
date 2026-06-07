from __future__ import annotations
from collections import defaultdict
from qdrant_client import QdrantClient
from opensearchpy import OpenSearch
from ragbench_core.utils.clients import close_client
from ragbench_core.interfaces import BaseRetriever
from ragbench_core.models.schema import Chunk, RetrievalResult
from ragbench_core.settings import settings


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
        client = QdrantClient(url=settings.qdrant_url)
        try:
            hits = client.search(
                collection_name=index_name, query_vector=query_embedding, limit=top_k
            )
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
        finally:
            close_client(client)


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

    def _search_fields(self, client: OpenSearch, index_name: str) -> tuple[list[str], list[str]]:
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


class HybridRetriever(BaseRetriever):
    name = "hybrid"

    def retrieve(
        self,
        question: str,
        query_embedding: list[float] | None,
        index_name: str,
        config: dict,
        top_k: int,
    ) -> list[RetrievalResult]:
        dense = DenseRetriever().retrieve(question, query_embedding, index_name, config, top_k)
        bm25 = BM25Retriever().retrieve(question, query_embedding, index_name, config, top_k)
        k = int(config.get("rrf_k", 60))
        scores = defaultdict(float)
        chunks = {}
        for result_set in [dense, bm25]:
            for r in result_set:
                scores[r.chunk.chunk_id] += 1.0 / (k + r.rank)
                chunks[r.chunk.chunk_id] = r.chunk
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            RetrievalResult(chunks[cid], score, i + 1, self.name)
            for i, (cid, score) in enumerate(ranked)
        ]
