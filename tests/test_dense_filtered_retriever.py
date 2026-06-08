from ragbench_core.components.retrievers import DenseFilteredRetriever
from ragbench_core.registry import ComponentRegistry, register_builtin_components


def test_dense_filtered_builds_metadata_query_filter():
    retriever = DenseFilteredRetriever()

    query_filter = retriever._build_query_filter(
        {
            "metadata_filters": {
                "source": "sample_policy.md",
                "word_count": {"gte": 20},
                "metadata.category": ["policy", "faq"],
                "chunk_id": {"value": "chunk-1"},
            },
            "doc_ids": ["doc-a", "doc-b"],
        }
    )

    assert query_filter == {
        "must": [
            {"key": "metadata.source", "match": {"value": "sample_policy.md"}},
            {"key": "metadata.word_count", "range": {"gte": 20}},
            {"key": "metadata.category", "match": {"any": ["policy", "faq"]}},
            {"key": "chunk_id", "match": {"value": "chunk-1"}},
            {"key": "doc_id", "match": {"any": ["doc-a", "doc-b"]}},
        ]
    }


def test_dense_filtered_uses_raw_payload_filter_when_provided():
    retriever = DenseFilteredRetriever()
    payload_filter = {"must": [{"key": "doc_id", "match": {"value": "doc-a"}}]}

    assert retriever._build_query_filter({"payload_filter": payload_filter}) == payload_filter


def test_builtin_registry_includes_dense_filtered_retriever():
    registry = ComponentRegistry()
    from ragbench_core import registry as registry_module

    original_registry = registry_module.REGISTRY
    registry_module.REGISTRY = registry
    try:
        register_builtin_components()
    finally:
        registry_module.REGISTRY = original_registry

    assert "dense_filtered" in registry.retrievers
