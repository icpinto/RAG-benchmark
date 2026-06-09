from ragbench_core.components.retrievers import DenseFilteredRetriever, ParentDocumentRetriever
from ragbench_core.registry import ComponentRegistry, register_builtin_components
from ragbench_core.models.schema import Chunk, RetrievalResult


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
    assert "parent_document" in registry.retrievers


def test_parent_document_expands_hierarchical_child_hits_to_parent_context():
    retriever = ParentDocumentRetriever()
    child_results = [
        RetrievalResult(
            Chunk(
                "child-1",
                "doc-a",
                "small child one",
                {
                    "chunker": "llamaindex_hierarchical",
                    "parent_chunk_id": "parent-1",
                    "parent_text": "larger parent context",
                },
            ),
            0.9,
            1,
            "dense",
        ),
        RetrievalResult(
            Chunk(
                "child-2",
                "doc-a",
                "small child two",
                {
                    "chunker": "llamaindex_hierarchical",
                    "parent_chunk_id": "parent-1",
                    "parent_text": "larger parent context",
                },
            ),
            0.8,
            2,
            "dense",
        ),
        RetrievalResult(
            Chunk(
                "child-3",
                "doc-a",
                "second parent child",
                {
                    "chunker": "llamaindex_hierarchical",
                    "parent_chunk_id": "parent-2",
                    "parent_text": "second parent context",
                },
            ),
            0.7,
            3,
            "dense",
        ),
    ]

    results = retriever._expand_parent_context(child_results, top_k=10)

    assert [result.chunk.chunk_id for result in results] == ["parent-1", "parent-2"]
    assert [result.rank for result in results] == [1, 2]
    assert results[0].chunk.text == "larger parent context"
    assert results[0].chunk.metadata["child_chunk_id"] == "child-1"
    assert results[0].chunk.metadata["retrieved_chunk_text"] == "small child one"
    assert results[0].retriever == "parent_document"


def test_parent_document_preserves_non_hierarchical_hits_with_missing_flag():
    retriever = ParentDocumentRetriever()
    child_result = RetrievalResult(Chunk("child-1", "doc-a", "plain text", {}), 0.9, 1, "dense")

    results = retriever._expand_parent_context([child_result], top_k=10)

    assert results[0].chunk.chunk_id == "child-1"
    assert results[0].chunk.text == "plain text"
    assert results[0].chunk.metadata["parent_expansion_missing"] is True


def test_parent_document_child_top_k_uses_multiplier_or_explicit_limit():
    retriever = ParentDocumentRetriever()

    assert retriever._child_top_k({"child_top_k_multiplier": 3}, 5) == 15
    assert retriever._child_top_k({"child_top_k": 8, "child_top_k_multiplier": 3}, 5) == 8
    assert retriever._child_top_k({"child_top_k": 2}, 5) == 5
