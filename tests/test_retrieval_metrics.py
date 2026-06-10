import math

import pytest

from ragbench_core.components.metrics import RetrievalMetrics
from ragbench_core.models.schema import Chunk, GeneratedAnswer, QueryExample, RetrievalResult


def _result(chunk_id: str, doc_id: str, rank: int) -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(chunk_id=chunk_id, doc_id=doc_id, text=""),
        score=1.0 / rank,
        rank=rank,
        retriever="test",
    )


def test_retrieval_metrics_deduplicate_chunk_hits():
    example = QueryExample(
        question="q",
        ground_truth_chunk_ids=["chunk-a"],
        ground_truth_doc_ids=["doc-a"],
    )
    results = [
        _result("chunk-a", "doc-a", 1),
        _result("chunk-a", "doc-a", 2),
    ]

    metrics = RetrievalMetrics().compute(example, results, None, {"retrieval_k": [2]})

    assert metrics["recall@2"] == 1.0
    assert metrics["precision@2"] == 0.5
    assert metrics["mrr@2"] == 1.0
    assert metrics["ndcg@2"] == 1.0


def test_retrieval_metrics_deduplicate_doc_hits_when_no_chunk_truth():
    example = QueryExample(question="q", ground_truth_doc_ids=["doc-a"])
    results = [
        _result("chunk-a", "doc-a", 1),
        _result("chunk-b", "doc-a", 2),
    ]

    metrics = RetrievalMetrics().compute(example, results, None, {"retrieval_k": [2]})

    assert metrics["recall@2"] == 1.0
    assert metrics["precision@2"] == 0.5
    assert metrics["mrr@2"] == 1.0
    assert metrics["ndcg@2"] == 1.0


def test_retrieval_metrics_ndcg_uses_first_unique_hit_rank():
    example = QueryExample(question="q", ground_truth_doc_ids=["doc-a", "doc-b"])
    results = [
        _result("chunk-a", "doc-a", 1),
        _result("chunk-b", "doc-a", 2),
        _result("chunk-c", "doc-b", 3),
    ]

    metrics = RetrievalMetrics().compute(example, results, None, {"retrieval_k": [3]})

    expected_dcg = 1.0 + (1.0 / math.log2(4))
    expected_idcg = 1.0 + (1.0 / math.log2(3))
    assert metrics["recall@3"] == 1.0
    assert metrics["ndcg@3"] == pytest.approx(expected_dcg / expected_idcg)
    assert metrics["ndcg@3"] <= 1.0


def test_retrieval_metrics_rejects_invalid_retrieval_k():
    example = QueryExample(question="q", ground_truth_doc_ids=["doc-a"])

    invalid_configs = [
        {"retrieval_k": []},
        {"retrieval_k": [0]},
        {"retrieval_k": [-1]},
        {"retrieval_k": [True]},
        {"retrieval_k": ["3"]},
        {"retrieval_k": 3},
    ]

    for config in invalid_configs:
        with pytest.raises(ValueError, match="retrieval_k"):
            RetrievalMetrics().compute(example, [], None, config)


def test_retrieval_metrics_marks_unavailable_when_no_ground_truth():
    example = QueryExample(question="q")
    results = [_result("chunk-a", "doc-a", 1)]

    metrics = RetrievalMetrics().compute(example, results, None, {"retrieval_k": [1]})

    assert metrics["retrieval_ground_truth_available"] is False
    assert metrics["retrieval_failed"] is False
    assert metrics["recall@1"] is None
    assert metrics["precision@1"] is None
    assert metrics["mrr@1"] is None
    assert metrics["ndcg@1"] is None


def test_retrieval_metrics_marks_failed_when_ground_truth_has_no_results():
    example = QueryExample(question="q", ground_truth_doc_ids=["doc-a"])

    metrics = RetrievalMetrics().compute(example, [], None, {"retrieval_k": [1]})

    assert metrics["retrieval_ground_truth_available"] is True
    assert metrics["retrieval_failed"] is True
    assert metrics["recall@1"] == 0.0
    assert metrics["precision@1"] == 0.0
    assert metrics["mrr@1"] == 0.0
    assert metrics["ndcg@1"] == 0.0


def test_retrieval_metrics_evaluates_raw_and_reranked_precompression_sets():
    example = QueryExample(question="q", ground_truth_doc_ids=["doc-a"])
    raw_results = [
        _result("chunk-b", "doc-b", 1),
        _result("chunk-a", "doc-a", 2),
    ]
    reranked_results = [
        _result("chunk-a", "doc-a", 1),
        _result("chunk-b", "doc-b", 2),
    ]
    compressed_results = [_result("chunk-b", "doc-b", 1)]
    answer = GeneratedAnswer(
        answer="answer",
        prompt="prompt",
        context_chunks=[result.chunk for result in compressed_results],
        metadata={
            "raw_retrieval_results": raw_results,
            "reranked_results": reranked_results,
        },
    )

    metrics = RetrievalMetrics().compute(example, compressed_results, answer, {"retrieval_k": [1, 2]})

    assert metrics["raw_retrieval_recall@1"] == 0.0
    assert metrics["raw_retrieval_recall@2"] == 1.0
    assert metrics["reranked_recall@1"] == 1.0
    assert metrics["reranked_recall@2"] == 1.0
    assert metrics["recall@1"] == 1.0
    assert metrics["recall@2"] == 1.0
    assert metrics["raw_retrieval_failed"] is False
    assert metrics["reranked_failed"] is False
    assert metrics["retrieval_failed"] is False


def test_retrieval_metrics_ignores_post_compression_results_when_stage_sets_are_available():
    example = QueryExample(question="q", ground_truth_doc_ids=["doc-a"])
    raw_results = [_result("chunk-a", "doc-a", 1)]
    reranked_results = [_result("chunk-a", "doc-a", 1)]
    compressed_results = []
    answer = GeneratedAnswer(
        answer="answer",
        prompt="prompt",
        context_chunks=[],
        metadata={
            "raw_retrieval_results": raw_results,
            "reranked_results": reranked_results,
        },
    )

    metrics = RetrievalMetrics().compute(example, compressed_results, answer, {"retrieval_k": [1]})

    assert metrics["raw_retrieval_recall@1"] == 1.0
    assert metrics["reranked_recall@1"] == 1.0
    assert metrics["recall@1"] == 1.0
    assert metrics["raw_retrieval_failed"] is False
    assert metrics["reranked_failed"] is False
    assert metrics["retrieval_failed"] is False
