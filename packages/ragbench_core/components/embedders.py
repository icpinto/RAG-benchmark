from __future__ import annotations
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from ragbench_core.interfaces import BaseEmbedder
from ragbench_core.models.schema import Chunk

@lru_cache(maxsize=4)
def _load_model(model_name: str):
    return SentenceTransformer(model_name)

class SentenceTransformerEmbedder(BaseEmbedder):
    name = "sentence_transformers"
    def embed_chunks(self, chunks: list[Chunk], config: dict) -> list[Chunk]:
        model_name=config["model"]
        model=_load_model(model_name)
        texts=[c.text for c in chunks]
        vectors=model.encode(texts, normalize_embeddings=True, batch_size=int(config.get("batch_size", 32))).tolist()
        for c, v in zip(chunks, vectors): c.embedding=v
        return chunks
    def embed_query(self, query: str, config: dict) -> list[float]:
        model=_load_model(config["model"])
        return model.encode([query], normalize_embeddings=True)[0].tolist()
