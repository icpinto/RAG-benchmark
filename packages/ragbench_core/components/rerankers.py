from functools import lru_cache
import importlib.util
import sys
import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

from ragbench_core.interfaces import BaseReranker
from ragbench_core.models.schema import RetrievalResult


def _ensure_package(package_name: str, message: str) -> None:
    if package_name in sys.modules or importlib.util.find_spec(package_name) is not None:
        return
    raise ImportError(message)


class NoReranker(BaseReranker):
    name = "none"

    def rerank(self, question: str, results: list[RetrievalResult], config: dict) -> list[RetrievalResult]:
        return results


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


class CrossEncoderReranker(BaseReranker):
    name = "cross_encoder"

    def rerank(self, question: str, results: list[RetrievalResult], config: dict) -> list[RetrievalResult]:
        if not results:
            return results
        model = _load_cross_encoder(config.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
        scores = model.predict([(question, r.chunk.text) for r in results])
        return _apply_score_order(results, scores)


@lru_cache(maxsize=4)
def _load_flashrank_ranker(model_name: str | None, cache_dir: str | None, max_length: int | None):
    _ensure_package("flashrank", "FlashRankReranker requires `pip install flashrank`.")
    from flashrank import Ranker

    kwargs: dict[str, Any] = {}
    if model_name:
        kwargs["model_name"] = model_name
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if max_length is not None:
        kwargs["max_length"] = max_length
    return Ranker(**kwargs)


class FlashRankReranker(BaseReranker):
    """Rerank results with FlashRank's ONNX-backed cross-encoder/listwise models."""

    name = "flashrank"

    def rerank(self, question: str, results: list[RetrievalResult], config: dict) -> list[RetrievalResult]:
        if not results:
            return results
        _ensure_package("flashrank", "FlashRankReranker requires `pip install flashrank`.")
        from flashrank import RerankRequest

        ranker = _load_flashrank_ranker(
            config.get("model"),
            config.get("cache_dir"),
            _coerce_optional_int(config.get("max_length")),
        )
        passages = [
            {
                "id": str(index),
                "text": result.chunk.text,
                "meta": {"index": index, "doc_id": result.chunk.doc_id, "chunk_id": result.chunk.chunk_id},
            }
            for index, result in enumerate(results)
        ]
        ranked_passages = ranker.rerank(RerankRequest(query=question, passages=passages))
        return _apply_ranked_indices(results, ranked_passages)


@lru_cache(maxsize=4)
def _load_unified_reranker(model_name: str, model_type: str, api_key: str | None, lang: str | None):
    extra = "rankllm" if model_type == "rankllm" else "transformers"
    _ensure_package(
        "rerankers",
        f"{model_type} reranking requires `pip install 'rerankers[{extra}]'`.",
    )
    from rerankers import Reranker

    kwargs: dict[str, Any] = {"model_type": model_type}
    if api_key:
        kwargs["api_key"] = api_key
    if lang:
        kwargs["lang"] = lang
    return Reranker(model_name, **kwargs)


class RankLLMReranker(BaseReranker):
    """Rerank with Answer.AI rerankers' RankLLM integration."""

    name = "rankllm"

    def rerank(self, question: str, results: list[RetrievalResult], config: dict) -> list[RetrievalResult]:
        if not results:
            return results
        ranker = _load_unified_reranker(
            config.get("model", "rankllm"),
            "rankllm",
            config.get("api_key"),
            config.get("lang"),
        )
        return _rerank_with_unified_ranker(ranker, question, results, config)


class ColBERTReranker(BaseReranker):
    """Rerank candidates with a ColBERT late-interaction model."""

    name = "colbert"

    def rerank(self, question: str, results: list[RetrievalResult], config: dict) -> list[RetrievalResult]:
        if not results:
            return results
        ranker = _load_unified_reranker(
            config.get("model", "colbert"),
            "colbert",
            config.get("api_key"),
            config.get("lang"),
        )
        return _rerank_with_unified_ranker(ranker, question, results, config)


def _rerank_with_unified_ranker(
    ranker,
    question: str,
    results: list[RetrievalResult],
    config: dict,
) -> list[RetrievalResult]:
    ranked_results = ranker.rank(
        query=question,
        docs=[result.chunk.text for result in results],
        doc_ids=list(range(len(results))),
        metadata=[{"doc_id": result.chunk.doc_id, "chunk_id": result.chunk.chunk_id} for result in results],
    )
    ranked_items = ranked_results.top_k(len(results)) if hasattr(ranked_results, "top_k") else ranked_results.results
    return _apply_ranked_indices(results, ranked_items, score_key="score")


class DiversityMMRReranker(BaseReranker):
    """Rerank results with Maximum Marginal Relevance (MMR).

    MMR iteratively chooses results that balance relevance against similarity to
    already selected results. Relevance comes from the incoming retriever scores,
    while diversity uses chunk embeddings when available and falls back to a
    lightweight token cosine similarity over chunk text when embeddings are
    missing.
    """

    name = "diversity_mmr"

    def rerank(self, question: str, results: list[RetrievalResult], config: dict) -> list[RetrievalResult]:
        if len(results) <= 1:
            return results

        lambda_mult = _coerce_float(config.get("lambda_mult", config.get("lambda", 0.5)), default=0.5)
        lambda_mult = min(max(lambda_mult, 0.0), 1.0)
        relevance_scores = _normalize_scores([r.score for r in results])

        selected_indices: list[int] = []
        candidate_indices = list(range(len(results)))
        mmr_scores = [0.0] * len(results)

        while candidate_indices:
            best_index = max(
                candidate_indices,
                key=lambda idx: (
                    self._candidate_mmr_score(idx, selected_indices, results, relevance_scores, lambda_mult),
                    relevance_scores[idx],
                    -idx,
                ),
            )
            mmr_scores[best_index] = self._candidate_mmr_score(
                best_index,
                selected_indices,
                results,
                relevance_scores,
                lambda_mult,
            )
            selected_indices.append(best_index)
            candidate_indices.remove(best_index)

        reranked = [results[idx] for idx in selected_indices]
        for rank, result in enumerate(reranked, start=1):
            result.rank = rank
            result.score = float(mmr_scores[selected_indices[rank - 1]])
        return reranked

    def _candidate_mmr_score(
        self,
        candidate_index: int,
        selected_indices: list[int],
        results: list[RetrievalResult],
        relevance_scores: list[float],
        lambda_mult: float,
    ) -> float:
        if not selected_indices:
            return relevance_scores[candidate_index]

        candidate = results[candidate_index]
        diversity_penalty = max(
            _chunk_similarity(candidate, results[selected_index])
            for selected_index in selected_indices
        )
        return (lambda_mult * relevance_scores[candidate_index]) - ((1.0 - lambda_mult) * diversity_penalty)


def _apply_score_order(results: list[RetrievalResult], scores) -> list[RetrievalResult]:
    reranked = []
    for result, score in zip(results, scores, strict=True):
        result.score = float(score)
        reranked.append(result)
    reranked.sort(key=lambda r: r.score, reverse=True)
    return _assign_ranks(reranked)


def _apply_ranked_indices(
    results: list[RetrievalResult],
    ranked_items,
    score_key: str = "score",
) -> list[RetrievalResult]:
    reranked: list[RetrievalResult] = []
    seen: set[int] = set()
    for item in ranked_items:
        index = _extract_ranked_index(item)
        if index is None or index in seen or index < 0 or index >= len(results):
            continue
        result = results[index]
        score = _extract_score(item, score_key)
        if score is not None:
            result.score = score
        reranked.append(result)
        seen.add(index)

    # Preserve any unreturned candidates at the end so rerankers remain length-preserving.
    reranked.extend(result for index, result in enumerate(results) if index not in seen)
    return _assign_ranks(reranked)


def _extract_ranked_index(item) -> int | None:
    if isinstance(item, dict):
        if isinstance(item.get("meta"), dict) and "index" in item["meta"]:
            return _coerce_optional_int(item["meta"]["index"])
        return _coerce_optional_int(item.get("id"))

    document = getattr(item, "document", None)
    if document is not None:
        return _coerce_optional_int(getattr(document, "doc_id", None))
    return _coerce_optional_int(getattr(item, "doc_id", None))


def _extract_score(item, score_key: str) -> float | None:
    if isinstance(item, dict):
        if score_key in item:
            return _coerce_float(item[score_key], default=0.0)
        return None
    if hasattr(item, score_key):
        return _coerce_float(getattr(item, score_key), default=0.0)
    return None


def _assign_ranks(results: list[RetrievalResult]) -> list[RetrievalResult]:
    for rank, result in enumerate(results, start=1):
        result.rank = rank
    return results


def _coerce_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_optional_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_scores(scores: list[float]) -> list[float]:
    numeric_scores = [_coerce_float(score, 0.0) for score in scores]
    min_score = min(numeric_scores)
    max_score = max(numeric_scores)
    if math.isclose(max_score, min_score):
        return [1.0 for _ in numeric_scores]
    return [(score - min_score) / (max_score - min_score) for score in numeric_scores]


def _chunk_similarity(left: RetrievalResult, right: RetrievalResult) -> float:
    embedding_similarity = _embedding_similarity(left.chunk.embedding, right.chunk.embedding)
    if embedding_similarity is not None:
        return embedding_similarity
    return _text_similarity(left.chunk.text, right.chunk.text)


def _embedding_similarity(left: list[float] | None, right: list[float] | None) -> float | None:
    if not _is_numeric_sequence(left) or not _is_numeric_sequence(right):
        return None
    if len(left) != len(right) or len(left) == 0:
        return None

    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
    right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
    if math.isclose(left_norm, 0.0) or math.isclose(right_norm, 0.0):
        return None
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


def _is_numeric_sequence(value) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and all(
        isinstance(item, int | float) for item in value
    )


def _text_similarity(left: str, right: str) -> float:
    left_counts = Counter(_tokenize(left))
    right_counts = Counter(_tokenize(right))
    if not left_counts or not right_counts:
        return 0.0

    common_terms = left_counts.keys() & right_counts.keys()
    dot = sum(left_counts[term] * right_counts[term] for term in common_terms)
    left_norm = math.sqrt(sum(count * count for count in left_counts.values()))
    right_norm = math.sqrt(sum(count * count for count in right_counts.values()))
    if math.isclose(left_norm, 0.0) or math.isclose(right_norm, 0.0):
        return 0.0
    return dot / (left_norm * right_norm)


def _tokenize(text: str) -> list[str]:
    return [token for token in text.lower().split() if token]
