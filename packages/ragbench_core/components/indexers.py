from __future__ import annotations
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from opensearchpy import OpenSearch
from ragbench_core.interfaces import BaseIndexer
from ragbench_core.models.schema import Chunk
from ragbench_core.settings import settings

class QdrantIndexer(BaseIndexer):
    name = "qdrant"
    def index(self, chunks: list[Chunk], index_name: str, config: dict) -> dict:
        if not chunks: return {"points":0}
        client=QdrantClient(url=settings.qdrant_url)
        dim=len(chunks[0].embedding or [])
        client.recreate_collection(collection_name=index_name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
        points=[]
        for idx,c in enumerate(chunks):
            points.append(PointStruct(id=idx, vector=c.embedding, payload={"chunk_id":c.chunk_id,"doc_id":c.doc_id,"text":c.text,"metadata":c.metadata}))
        client.upsert(collection_name=index_name, points=points)
        return {"points":len(points),"dim":dim,"backend":"qdrant"}

class OpenSearchIndexer(BaseIndexer):
    name = "opensearch"
    def index(self, chunks: list[Chunk], index_name: str, config: dict) -> dict:
        client=OpenSearch(settings.opensearch_url, verify_certs=False, ssl_show_warn=False)
        if client.indices.exists(index=index_name): client.indices.delete(index=index_name)
        client.indices.create(index=index_name, body={"mappings":{"properties":{"text":{"type":"text"},"doc_id":{"type":"keyword"},"chunk_id":{"type":"keyword"},"metadata":{"type":"object","enabled":True}}}})
        for c in chunks:
            client.index(index=index_name, id=c.chunk_id, body={"text":c.text,"doc_id":c.doc_id,"chunk_id":c.chunk_id,"metadata":c.metadata}, refresh=False)
        client.indices.refresh(index=index_name)
        return {"documents":len(chunks),"backend":"opensearch"}
