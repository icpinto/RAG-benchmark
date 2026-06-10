from ragbench_core.components.rerankers import DiversityMMRReranker
from ragbench_core.models.schema import Chunk, RetrievalResult


def _result(chunk_id: str, score: float, text: str, embedding=None) -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(chunk_id=chunk_id, doc_id="doc", text=text, embedding=embedding),
        score=score,
        rank=0,
        retriever="test",
    )


def test_diversity_mmr_uses_embeddings_to_promote_diverse_results():
    results = [
        _result("top", 10.0, "alpha policy", [1.0, 0.0]),
        _result("near_duplicate", 9.0, "alpha policy duplicate", [0.99, 0.01]),
        _result("diverse", 8.0, "beta travel rule", [0.0, 1.0]),
    ]

    reranked = DiversityMMRReranker().rerank("policy", results, {"lambda_mult": 0.5})

    assert [result.chunk.chunk_id for result in reranked] == ["top", "diverse", "near_duplicate"]
    assert [result.rank for result in reranked] == [1, 2, 3]


def test_diversity_mmr_falls_back_to_text_similarity_without_embeddings():
    results = [
        _result("top", 10.0, "alpha alpha policy details"),
        _result("near_duplicate", 9.0, "alpha alpha policy details updated"),
        _result("diverse", 8.0, "zebra benefit reimbursement"),
    ]

    reranked = DiversityMMRReranker().rerank("policy", results, {"lambda_mult": 0.5})

    assert [result.chunk.chunk_id for result in reranked] == ["top", "diverse", "near_duplicate"]

