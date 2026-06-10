import math

import pytest

from ragbench_core.components.metrics import RetrievalMetrics
from ragbench_core.models.schema import Chunk, QueryExample, RetrievalResult


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
