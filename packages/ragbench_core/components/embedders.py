from __future__ import annotations

from functools import lru_cache

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
    if text.startswith(prefix):
        return text
    return f"{prefix}{text}"


def _as_list(vector):
    if hasattr(vector, "tolist"):
        return vector.tolist()
    return list(vector)


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
            passage_prefix = config.get("passage_prefix", "passage: ")
            texts = [_prefix_text(text, passage_prefix) for text in texts]
        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=int(config.get("batch_size", 32)),
        )
        vectors = _as_list(vectors)
        for c, v in zip(chunks, vectors):
            c.embedding = v
        return chunks

    def embed_query(self, query: str, config: dict) -> list[float]:
        model_name = _resolve_model_name(config["model"])
        model = _load_model(model_name)
        if _is_e5_model(config["model"]):
            query = _prefix_text(query, config.get("query_prefix", "query: "))
        vectors = model.encode(
            [query],
            normalize_embeddings=True,
            batch_size=int(config.get("batch_size", 32)),
        )
        return _as_list(vectors[0])
