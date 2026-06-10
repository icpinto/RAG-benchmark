import sys
from types import SimpleNamespace

from ragbench_core.components.rerankers import (
    CohereReranker,
    ColBERTReranker,
    DiversityMMRReranker,
    FlashRankReranker,
    JinaReranker,
    RankLLMReranker,
)
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



def test_flashrank_reranker_maps_ranked_passages_back_to_results(monkeypatch):
    results = [
        _result("a", 1.0, "first"),
        _result("b", 2.0, "second"),
        _result("c", 3.0, "third"),
    ]

    class FakeRequest:
        def __init__(self, query, passages):
            self.query = query
            self.passages = passages

    class FakeRanker:
        def rerank(self, request):
            assert request.query == "query"
            return [
                {"id": "1", "text": "second", "meta": {"index": 1}, "score": 0.9},
                {"id": "2", "text": "third", "meta": {"index": 2}, "score": 0.4},
                {"id": "0", "text": "first", "meta": {"index": 0}, "score": 0.1},
            ]

    monkeypatch.setitem(sys.modules, "flashrank", SimpleNamespace(RerankRequest=FakeRequest))
    monkeypatch.setattr("ragbench_core.components.rerankers._ensure_package", lambda *args: None)
    monkeypatch.setattr("ragbench_core.components.rerankers._load_flashrank_ranker", lambda *args: FakeRanker())

    reranked = FlashRankReranker().rerank("query", results, {"model": "fake"})

    assert [result.chunk.chunk_id for result in reranked] == ["b", "c", "a"]
    assert [result.score for result in reranked] == [0.9, 0.4, 0.1]
    assert [result.rank for result in reranked] == [1, 2, 3]


def test_cohere_reranker_calls_hosted_api_and_maps_results(monkeypatch):
    results = [_result("a", 1.0, "first"), _result("b", 2.0, "second"), _result("c", 3.0, "third")]
    calls = []

    def fake_post(url, payload, headers, timeout):
        calls.append((url, payload, headers, timeout))
        return {
            "results": [
                {"index": 2, "relevance_score": 0.91},
                {"index": 0, "relevance_score": 0.42},
                {"index": 1, "relevance_score": 0.11},
            ]
        }

    monkeypatch.setattr("ragbench_core.components.rerankers._post_rerank_request", fake_post)

    reranked = CohereReranker().rerank(
        "query",
        results,
        {"api_key": "co-key", "model": "rerank-v3.5", "top_n": 2, "timeout": 12},
    )

    assert calls == [
        (
            "https://api.cohere.com/v2/rerank",
            {"model": "rerank-v3.5", "query": "query", "documents": ["first", "second", "third"], "top_n": 2},
            {"Authorization": "Bearer co-key"},
            12.0,
        )
    ]
    assert [result.chunk.chunk_id for result in reranked] == ["c", "a", "b"]
    assert [result.score for result in reranked] == [0.91, 0.42, 0.11]
    assert [result.rank for result in reranked] == [1, 2, 3]


def test_jina_reranker_uses_local_open_source_model(monkeypatch):
    results = [_result("a", 1.0, "first"), _result("b", 2.0, "second"), _result("c", 3.0, "third")]
    calls = []

    class FakeCrossEncoder:
        def predict(self, pairs):
            calls.append(pairs)
            return [0.33, 0.88, 0.12]

    def fake_load(model, trust_remote_code, max_length, device):
        calls.append((model, trust_remote_code, max_length, device))
        return FakeCrossEncoder()

    monkeypatch.setattr("ragbench_core.components.rerankers._load_jina_cross_encoder", fake_load)

    reranked = JinaReranker().rerank(
        "query",
        results,
        {"model": "jinaai/local-reranker", "trust_remote_code": False, "max_length": 1024, "device": "cpu"},
    )

    assert calls == [
        ("jinaai/local-reranker", False, 1024, "cpu"),
        [("query", "first"), ("query", "second"), ("query", "third")],
    ]
    assert [result.chunk.chunk_id for result in reranked] == ["b", "a", "c"]
    assert [result.score for result in reranked] == [0.88, 0.33, 0.12]
    assert [result.rank for result in reranked] == [1, 2, 3]


class _FakeUnifiedRanker:
    def rank(self, query, docs, doc_ids, metadata):
        assert query == "query"
        assert docs == ["first", "second", "third"]
        assert doc_ids == [0, 1, 2]
        return SimpleNamespace(
            results=[
                SimpleNamespace(document=SimpleNamespace(doc_id=2), score=7.0),
                SimpleNamespace(document=SimpleNamespace(doc_id=0), score=3.0),
                SimpleNamespace(document=SimpleNamespace(doc_id=1), score=1.0),
            ]
        )


def test_rankllm_reranker_uses_unified_reranker(monkeypatch):
    results = [_result("a", 1.0, "first"), _result("b", 2.0, "second"), _result("c", 3.0, "third")]
    calls = []

    def fake_load(model, model_type, api_key, lang):
        calls.append((model, model_type, api_key, lang))
        return _FakeUnifiedRanker()

    monkeypatch.setattr("ragbench_core.components.rerankers._load_unified_reranker", fake_load)

    reranked = RankLLMReranker().rerank("query", results, {"model": "rankllm", "api_key": "key", "lang": "en"})

    assert calls == [("rankllm", "rankllm", "key", "en")]
    assert [result.chunk.chunk_id for result in reranked] == ["c", "a", "b"]
    assert [result.score for result in reranked] == [7.0, 3.0, 1.0]


def test_colbert_reranker_uses_unified_reranker(monkeypatch):
    results = [_result("a", 1.0, "first"), _result("b", 2.0, "second"), _result("c", 3.0, "third")]
    calls = []

    def fake_load(model, model_type, api_key, lang):
        calls.append((model, model_type, api_key, lang))
        return _FakeUnifiedRanker()

    monkeypatch.setattr("ragbench_core.components.rerankers._load_unified_reranker", fake_load)

    reranked = ColBERTReranker().rerank("query", results, {"model": "colbert"})

    assert calls == [("colbert", "colbert", None, None)]
    assert [result.chunk.chunk_id for result in reranked] == ["c", "a", "b"]
    assert [result.rank for result in reranked] == [1, 2, 3]
