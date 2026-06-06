from datetime import datetime

from ragbench_core.engine.evaluation_trace import build_evaluation_step
from ragbench_core.models.schema import Chunk, RetrievalResult


def test_build_evaluation_step_includes_context_query_build_and_chunks():
    chunk=Chunk(
        chunk_id="chunk-1",
        doc_id="doc-1",
        text="A saved context chunk.",
        metadata={"created_at": datetime(2026, 1, 2, 3, 4, 5)},
    )
    result=RetrievalResult(chunk=chunk, score=0.91, rank=1, retriever="dense")

    step=build_evaluation_step(
        "What happened?",
        ["What happened?", "event details"],
        [result],
        "[doc-1::chunk-1]\nA saved context chunk.",
        "Context:\n[doc-1::chunk-1]\nA saved context chunk.\n\nQuestion: What happened?",
    )

    assert step["question"] == "What happened?"
    assert step["rewritten_queries"] == ["What happened?", "event details"]
    assert step["context"].startswith("[doc-1::chunk-1]")
    assert "Question: What happened?" in step["query_build"]
    assert step["retrieved_context"] == [
        {
            "rank": 1,
            "score": 0.91,
            "retriever": "dense",
            "doc_id": "doc-1",
            "chunk_id": "chunk-1",
            "text": "A saved context chunk.",
            "metadata": {"created_at": "2026-01-02 03:04:05"},
        }
    ]
