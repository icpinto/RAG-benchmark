from functools import lru_cache
import math
from collections import Counter
from collections.abc import Sequence

from ragbench_core.interfaces import BaseReranker
from ragbench_core.models.schema import RetrievalResult


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
        reranked = []
        for r, s in zip(results, scores):
            r.score = float(s)
            reranked.append(r)
        reranked.sort(key=lambda r: r.score, reverse=True)
        for i, r in enumerate(reranked):
            r.rank = i + 1
        return reranked


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


def _coerce_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


def _tokenize(text: str) -> list[str]:
    return [token for token in "".join(char.lower() if char.isalnum() else " " for char in text).split() if token]
