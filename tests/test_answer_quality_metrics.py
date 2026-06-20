from ragbench_core.components.metrics import (
    DeepEvalFaithfulnessMetric,
    RagasFaithfulnessMetric,
)
from ragbench_core.models.schema import Chunk, GeneratedAnswer, QueryExample
from ragbench_core.registry import REGISTRY, register_builtin_components


def _answer():
    chunk = Chunk(chunk_id="c1", doc_id="d1", text="Paris is the capital of France.")
    return GeneratedAnswer(
        answer="Paris is the capital of France.",
        prompt="prompt",
        context_chunks=[chunk],
    )


def test_ragas_metric_uses_configured_ollama_judge(monkeypatch):
    captured = {}

    def fake_score(self, example, results, answer, metric_config):
        captured["model"] = metric_config.model
        captured["base_url"] = metric_config.base_url
        captured["timeout"] = metric_config.timeout
        captured["contexts"] = self._contexts(answer)
        return 0.87, {"available": True, "mocked": True}

    monkeypatch.setattr(RagasFaithfulnessMetric, "_score", fake_score)

    result = RagasFaithfulnessMetric().compute(
        QueryExample(question="What is France's capital?"),
        [],
        _answer(),
        {
            "enabled": ["ragas_faithfulness"],
            "providers": {
                "ragas": {
                    "model": "gpt-oss:20b",
                    "base_url": "http://ollama:11434",
                    "timeout": 12,
                }
            },
        },
    )

    assert result["ragas_faithfulness"] == 0.87
    assert result["ragas_faithfulness_metadata"]["available"] is True
    assert result["ragas_faithfulness_metadata"]["mocked"] is True
    assert captured == {
        "model": "gpt-oss:20b",
        "base_url": "http://ollama:11434",
        "timeout": 12.0,
        "contexts": ["Paris is the capital of France."],
    }


def test_judge_metric_gracefully_returns_null_metadata_on_error(monkeypatch):
    def fake_score(self, example, results, answer, metric_config):
        raise ModuleNotFoundError("No module named 'deepeval'")

    monkeypatch.setattr(DeepEvalFaithfulnessMetric, "_score", fake_score)

    result = DeepEvalFaithfulnessMetric().compute(
        QueryExample(question="What is France's capital?"),
        [],
        _answer(),
        {"enabled": ["deepeval_faithfulness"]},
    )

    assert result["deepeval_faithfulness"] is None
    metadata = result["deepeval_faithfulness_metadata"]
    assert metadata["available"] is False
    assert metadata["provider"] == "deepeval"
    assert metadata["reason"] == "ModuleNotFoundError"
    assert "deepeval" in metadata["error"]


def test_builtin_registry_includes_answer_quality_metrics(monkeypatch):
    # Keep the test focused on metric registration; optional pipeline dependencies
    # may be absent in the lightweight test environment.
    monkeypatch.setattr("ragbench_core.registry.REGISTRY", REGISTRY)
    try:
        register_builtin_components()
    except ModuleNotFoundError as exc:
        if exc.name != "requests":
            raise
        import types, sys

        sys.modules["requests"] = types.SimpleNamespace(post=lambda *args, **kwargs: None)
        register_builtin_components()

    assert "ragas_faithfulness" in REGISTRY.metrics
    assert "ragas_answer_relevancy" in REGISTRY.metrics
    assert "deepeval_faithfulness" in REGISTRY.metrics
    assert "deepeval_answer_correctness" in REGISTRY.metrics
