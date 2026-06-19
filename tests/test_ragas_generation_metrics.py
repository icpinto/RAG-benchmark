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


def test_ragas_generation_metric_handles_broken_ragas_import(monkeypatch):
    answer = GeneratedAnswer(
        answer="The policy covers annual leave.",
        prompt="prompt",
        context_chunks=[Chunk(chunk_id="chunk-1", doc_id="doc-1", text="Annual leave is covered.")],
    )

    monkeypatch.setattr(
        "ragbench_core.components.metrics.importlib.util.find_spec",
        lambda dependency: object(),
    )

    class DummyLangchainHuggingFace:
        HuggingFaceEmbeddings = object

    class DummyLangchainOllama:
        ChatOllama = object

    def broken_import(name):
        if name == "langchain_huggingface":
            return DummyLangchainHuggingFace
        if name == "langchain_ollama":
            return DummyLangchainOllama
        if name == "ragas.embeddings":
            raise ModuleNotFoundError("No module named 'langchain_community.chat_models.vertexai'")
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr("ragbench_core.components.metrics.importlib.import_module", broken_import)

    metrics = RagasGenerationMetric().compute(
        QueryExample(question="What does the policy cover?"),
        [],
        answer,
        {"llm": {"provider": "ollama", "model": "test-model"}},
    )

    assert metrics["ragas_faithfulness"] is None
    assert metrics["ragas_answer_relevancy"] is None
    assert (
        metrics["ragas_error"]
        == "ragas dependency import failed: No module named 'langchain_community.chat_models.vertexai'"
    )
