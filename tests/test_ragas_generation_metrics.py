from ragbench_core.components.metrics import RagasGenerationMetric
from ragbench_core.models.schema import Chunk, GeneratedAnswer, QueryExample


def test_ragas_generation_metric_reports_unavailable_optional_dependencies():
    answer = GeneratedAnswer(
        answer="The policy covers annual leave.",
        prompt="prompt",
        context_chunks=[Chunk(chunk_id="chunk-1", doc_id="doc-1", text="Annual leave is covered.")],
    )

    metrics = RagasGenerationMetric().compute(
        QueryExample(question="What does the policy cover?"),
        [],
        answer,
        {"llm": {"provider": "ollama", "model": "test-model"}},
    )

    assert metrics["ragas_faithfulness"] is None
    assert metrics["ragas_answer_relevancy"] is None
    assert "ragas_error" in metrics


def test_ragas_generation_metric_exposes_new_metric_name():
    assert RagasGenerationMetric.name == "ragas_generation"
